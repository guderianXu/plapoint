#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_cluster.h>
#include <plamatrix/dense/dense_matrix.h>

namespace plapoint
{
namespace
{

struct VoxelKey
{
    int x;
    int y;
    int z;
};

struct VoxelKeyLess
{
    __host__ __device__ bool operator()(const VoxelKey& lhs, const VoxelKey& rhs) const
    {
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        return lhs.z < rhs.z;
    }
};

struct VoxelKeyEqual
{
    __host__ __device__ bool operator()(const VoxelKey& lhs, const VoxelKey& rhs) const
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

struct MeanAccum
{
    double mean;
    int count;

    __host__ __device__ MeanAccum() : mean(0.0), count(0) {}
    __host__ __device__ MeanAccum(double mean_, int count_) : mean(mean_), count(count_) {}
};

struct MeanAccumPlus
{
    __host__ __device__ MeanAccum operator()(const MeanAccum& lhs, const MeanAccum& rhs) const
    {
        if (lhs.count == 0) return rhs;
        if (rhs.count == 0) return lhs;

        const int total_count = lhs.count + rhs.count;
        const double total = static_cast<double>(total_count);
        const double lhs_weight = static_cast<double>(lhs.count) / total;
        const double rhs_weight = static_cast<double>(rhs.count) / total;
        return MeanAccum(lhs.mean * lhs_weight + rhs.mean * rhs_weight, total_count);
    }
};

template <typename Scalar>
__global__ void validateFinitePointsKernel(const Scalar* points, int point_count, int* error_code)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    const double x = static_cast<double>(points[idx]);
    const double y = static_cast<double>(points[point_count + idx]);
    const double z = static_cast<double>(points[2 * point_count + idx]);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z))
    {
        atomicCAS(error_code, 0, 1);
    }
}

template <typename Scalar>
__device__ int voxelCoordinateValidationError(Scalar coordinate, Scalar min_coordinate, Scalar cluster_size)
{
    const double relative = static_cast<double>(coordinate) - static_cast<double>(min_coordinate);
    const double scaled = floor(relative / static_cast<double>(cluster_size));
    if (!isfinite(relative) ||
        !isfinite(scaled) ||
        scaled < static_cast<double>(INT_MIN) ||
        scaled > static_cast<double>(INT_MAX))
    {
        return 1;
    }
    return 0;
}

template <typename Scalar>
__global__ void validateVoxelKeysKernel(
    const Scalar* points,
    int point_count,
    Scalar min_x,
    Scalar min_y,
    Scalar min_z,
    Scalar cluster_size,
    int* error_code)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    int error = voxelCoordinateValidationError(points[idx], min_x, cluster_size);
    if (error == 0)
    {
        error = voxelCoordinateValidationError(points[point_count + idx], min_y, cluster_size);
    }
    if (error == 0)
    {
        error = voxelCoordinateValidationError(points[2 * point_count + idx], min_z, cluster_size);
    }
    if (error != 0)
    {
        atomicCAS(error_code, 0, error);
    }
}

template <typename Scalar>
struct ComputeVoxelKey
{
    const Scalar* points;
    int point_count;
    Scalar min_x;
    Scalar min_y;
    Scalar min_z;
    Scalar cluster_size;

    __host__ __device__ VoxelKey operator()(int idx) const
    {
        const Scalar x = points[idx];
        const Scalar y = points[point_count + idx];
        const Scalar z = points[2 * point_count + idx];
        return {
            static_cast<int>(floor((static_cast<double>(x) - static_cast<double>(min_x)) /
                                   static_cast<double>(cluster_size))),
            static_cast<int>(floor((static_cast<double>(y) - static_cast<double>(min_y)) /
                                   static_cast<double>(cluster_size))),
            static_cast<int>(floor((static_cast<double>(z) - static_cast<double>(min_z)) /
                                   static_cast<double>(cluster_size)))
        };
    }
};

template <typename Scalar, int Dim>
struct GatherCoordinateMean
{
    const Scalar* points;
    int point_count;

    __host__ __device__ MeanAccum operator()(int idx) const
    {
        return MeanAccum(static_cast<double>(points[static_cast<int>(Dim) * point_count + idx]), 1);
    }
};

template <typename Scalar>
__global__ void writeCentroidsColumnMajor(
    const MeanAccum* mean_x,
    const MeanAccum* mean_y,
    const MeanAccum* mean_z,
    int cluster_count,
    Scalar* out_points)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cluster_count)
    {
        return;
    }

    out_points[idx] = static_cast<Scalar>(mean_x[idx].mean);
    out_points[cluster_count + idx] = static_cast<Scalar>(mean_y[idx].mean);
    out_points[2 * cluster_count + idx] = static_cast<Scalar>(mean_z[idx].mean);
}

template <typename Scalar>
void validateFinitePoints(const Scalar* d_points, int point_count, cudaStream_t stream)
{
    gpu::DeviceBuffer<int> d_error(1);
    int host_error = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(d_error.get(), &host_error, sizeof(host_error),
                                        cudaMemcpyHostToDevice, stream));

    constexpr int block_size = 256;
    const int grid_size = (point_count + block_size - 1) / block_size;
    validateFinitePointsKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_points, point_count, d_error.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&host_error, d_error.get(), sizeof(host_error),
                                        cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

    if (host_error != 0)
    {
        throw std::invalid_argument("voxelClusterSimplify GPU: points must be finite");
    }
}

template <typename Scalar>
void validateVoxelKeys(
    const Scalar* d_points,
    int point_count,
    Scalar min_x,
    Scalar min_y,
    Scalar min_z,
    Scalar cluster_size,
    cudaStream_t stream)
{
    gpu::DeviceBuffer<int> d_error(1);
    int host_error = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(d_error.get(), &host_error, sizeof(host_error),
                                        cudaMemcpyHostToDevice, stream));

    constexpr int block_size = 256;
    const int grid_size = (point_count + block_size - 1) / block_size;
    validateVoxelKeysKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_points, point_count, min_x, min_y, min_z, cluster_size, d_error.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&host_error, d_error.get(), sizeof(host_error),
                                        cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

    if (host_error != 0)
    {
        throw std::out_of_range("voxelClusterSimplify GPU: voxel index is outside int range");
    }
}

template <typename Scalar>
int voxelClusterSimplifyColumnMajor(
    const Scalar* d_points,
    int point_count,
    Scalar cluster_size,
    Scalar* d_out_points,
    int* d_point_remap,
    cudaStream_t stream)
{
    if (point_count <= 0)
    {
        return 0;
    }
    if (!d_points || !d_out_points || !d_point_remap)
    {
        throw std::invalid_argument("voxelClusterSimplify GPU: device pointers must not be null");
    }
    if (!std::isfinite(cluster_size) || cluster_size <= Scalar(0))
    {
        throw std::invalid_argument("voxelClusterSimplify GPU: cluster size must be finite and positive");
    }

    validateFinitePoints(d_points, point_count, stream);

    auto policy = thrust::cuda::par.on(stream);
    thrust::device_ptr<const Scalar> x_begin = thrust::device_pointer_cast(d_points);
    thrust::device_ptr<const Scalar> y_begin = x_begin + point_count;
    thrust::device_ptr<const Scalar> z_begin = y_begin + point_count;
    const Scalar min_x = thrust::reduce(policy, x_begin, y_begin,
                                        std::numeric_limits<Scalar>::max(),
                                        thrust::minimum<Scalar>());
    const Scalar min_y = thrust::reduce(policy, y_begin, z_begin,
                                        std::numeric_limits<Scalar>::max(),
                                        thrust::minimum<Scalar>());
    const Scalar min_z = thrust::reduce(policy, z_begin, z_begin + point_count,
                                        std::numeric_limits<Scalar>::max(),
                                        thrust::minimum<Scalar>());

    validateVoxelKeys(d_points, point_count, min_x, min_y, min_z, cluster_size, stream);

    thrust::device_vector<int> indices(static_cast<std::size_t>(point_count));
    thrust::device_vector<VoxelKey> keys(static_cast<std::size_t>(point_count));
    thrust::sequence(policy, indices.begin(), indices.end(), 0);
    thrust::transform(policy, indices.begin(), indices.end(), keys.begin(),
                      ComputeVoxelKey<Scalar>{d_points, point_count, min_x, min_y, min_z, cluster_size});
    thrust::sort_by_key(policy, keys.begin(), keys.end(), indices.begin(), VoxelKeyLess{});

    thrust::device_vector<VoxelKey> unique_keys(static_cast<std::size_t>(point_count));
    thrust::device_vector<MeanAccum> mean_x(static_cast<std::size_t>(point_count));
    thrust::device_vector<MeanAccum> mean_y(static_cast<std::size_t>(point_count));
    thrust::device_vector<MeanAccum> mean_z(static_cast<std::size_t>(point_count));

    auto x_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinateMean<Scalar, 0>{d_points, point_count});
    auto y_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinateMean<Scalar, 1>{d_points, point_count});
    auto z_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinateMean<Scalar, 2>{d_points, point_count});

    auto reduced_x = thrust::reduce_by_key(policy,
                                           keys.begin(), keys.end(),
                                           x_values,
                                           unique_keys.begin(),
                                           mean_x.begin(),
                                           VoxelKeyEqual{},
                                           MeanAccumPlus{});
    const int cluster_count = static_cast<int>(reduced_x.first - unique_keys.begin());

    thrust::reduce_by_key(policy,
                          keys.begin(), keys.end(),
                          y_values,
                          thrust::make_discard_iterator(),
                          mean_y.begin(),
                          VoxelKeyEqual{},
                          MeanAccumPlus{});
    thrust::reduce_by_key(policy,
                          keys.begin(), keys.end(),
                          z_values,
                          thrust::make_discard_iterator(),
                          mean_z.begin(),
                          VoxelKeyEqual{},
                          MeanAccumPlus{});

    thrust::device_vector<int> sorted_cluster_ids(static_cast<std::size_t>(point_count));
    thrust::lower_bound(policy,
                        unique_keys.begin(),
                        unique_keys.begin() + cluster_count,
                        keys.begin(),
                        keys.end(),
                        sorted_cluster_ids.begin(),
                        VoxelKeyLess{});
    thrust::scatter(policy,
                    sorted_cluster_ids.begin(),
                    sorted_cluster_ids.end(),
                    indices.begin(),
                    thrust::device_pointer_cast(d_point_remap));

    constexpr int block_size = 256;
    const int grid_size = (cluster_count + block_size - 1) / block_size;
    writeCentroidsColumnMajor<Scalar><<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(mean_x.data()),
        thrust::raw_pointer_cast(mean_y.data()),
        thrust::raw_pointer_cast(mean_z.data()),
        cluster_count,
        d_out_points);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return cluster_count;
}

plamatrix::DenseMatrix<int, plamatrix::Device::CPU> facesToMatrix(
    const std::vector<std::array<int, 3>>& faces)
{
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> matrix(
        static_cast<plamatrix::Index>(faces.size()), 3);
    for (std::size_t row = 0; row < faces.size(); ++row)
    {
        matrix.setValue(static_cast<plamatrix::Index>(row), 0, faces[row][0]);
        matrix.setValue(static_cast<plamatrix::Index>(row), 1, faces[row][1]);
        matrix.setValue(static_cast<plamatrix::Index>(row), 2, faces[row][2]);
    }
    return matrix;
}

std::vector<int> copyPointRemapToCpu(
    const plamatrix::DenseMatrix<int, plamatrix::Device::GPU>& point_remap_gpu,
    std::size_t point_count)
{
    if (point_remap_gpu.cols() != 1 || point_remap_gpu.rows() < static_cast<plamatrix::Index>(point_count))
    {
        throw std::invalid_argument("voxelClusterSimplify GPU: point remap matrix has invalid shape");
    }
    const auto point_remap_cpu = point_remap_gpu.toCpu();
    std::vector<int> point_remap(point_count);
    for (std::size_t row = 0; row < point_count; ++row)
    {
        point_remap[row] = point_remap_cpu(static_cast<plamatrix::Index>(row), 0);
    }
    return point_remap;
}

std::vector<int> computeClusterCounts(const std::vector<int>& point_remap, int cluster_count)
{
    std::vector<int> cluster_counts(static_cast<std::size_t>(cluster_count), 0);
    for (int cluster_index : point_remap)
    {
        if (cluster_index < 0 || cluster_index >= cluster_count)
        {
            throw std::runtime_error("voxelClusterSimplify GPU: point remap index out of range");
        }
        ++cluster_counts[static_cast<std::size_t>(cluster_index)];
    }
    return cluster_counts;
}

template <typename Attribute>
Attribute roundedAttribute(long double value)
{
    const long rounded = std::lround(value);
    const long lo = static_cast<long>(std::numeric_limits<Attribute>::min());
    const long hi = static_cast<long>(std::numeric_limits<Attribute>::max());
    return static_cast<Attribute>(std::clamp(rounded, lo, hi));
}

template <typename Scalar>
void setAveragedColors(
    const plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    const std::vector<int>& point_remap,
    const std::vector<int>& cluster_counts,
    plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& output)
{
    if (!mesh.hasColors())
    {
        return;
    }

    const auto colors_cpu = mesh.colors()->toCpu();
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
        static_cast<plamatrix::Index>(cluster_counts.size()), 3);
    std::vector<long double> sums(cluster_counts.size() * 3u, 0.0L);
    for (std::size_t row = 0; row < point_remap.size(); ++row)
    {
        const auto cluster = static_cast<std::size_t>(point_remap[row]);
        for (int col = 0; col < 3; ++col)
        {
            sums[cluster * 3u + static_cast<std::size_t>(col)] +=
                static_cast<long double>(colors_cpu.getValue(static_cast<plamatrix::Index>(row), col));
        }
    }

    for (std::size_t row = 0; row < cluster_counts.size(); ++row)
    {
        const long double inv = 1.0L / static_cast<long double>(cluster_counts[row]);
        for (int col = 0; col < 3; ++col)
        {
            colors.setValue(
                static_cast<plamatrix::Index>(row),
                col,
                roundedAttribute<std::uint8_t>(sums[row * 3u + static_cast<std::size_t>(col)] * inv));
        }
    }

    output.setColors(colors.toGpu());
}

template <typename Scalar>
void setAveragedIntensities(
    const plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    const std::vector<int>& point_remap,
    const std::vector<int>& cluster_counts,
    plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& output)
{
    if (!mesh.hasIntensities())
    {
        return;
    }

    const auto intensities_cpu = mesh.intensities()->toCpu();
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(
        static_cast<plamatrix::Index>(cluster_counts.size()), 1);
    std::vector<long double> sums(cluster_counts.size(), 0.0L);
    for (std::size_t row = 0; row < point_remap.size(); ++row)
    {
        const auto cluster = static_cast<std::size_t>(point_remap[row]);
        sums[cluster] += static_cast<long double>(
            intensities_cpu.getValue(static_cast<plamatrix::Index>(row), 0));
    }

    for (std::size_t row = 0; row < cluster_counts.size(); ++row)
    {
        const long double inv = 1.0L / static_cast<long double>(cluster_counts[row]);
        intensities.setValue(
            static_cast<plamatrix::Index>(row),
            0,
            roundedAttribute<std::uint16_t>(sums[row] * inv));
    }

    output.setIntensities(intensities.toGpu());
}

template <typename Scalar>
void setRemappedFaces(
    const plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    const std::vector<int>& point_remap,
    plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& output)
{
    std::vector<std::array<int, 3>> simplified_faces;
    if (mesh.hasFaces())
    {
        const auto faces_cpu = mesh.faces()->toCpu();
        simplified_faces.reserve(static_cast<std::size_t>(faces_cpu.rows()));
        for (plamatrix::Index row = 0; row < faces_cpu.rows(); ++row)
        {
            const int old_a = faces_cpu.getValue(row, 0);
            const int old_b = faces_cpu.getValue(row, 1);
            const int old_c = faces_cpu.getValue(row, 2);
            if (old_a < 0 || old_b < 0 || old_c < 0 ||
                static_cast<std::size_t>(old_a) >= point_remap.size() ||
                static_cast<std::size_t>(old_b) >= point_remap.size() ||
                static_cast<std::size_t>(old_c) >= point_remap.size())
            {
                throw std::out_of_range("voxelClusterSimplify GPU: face index out of range");
            }

            const int a = point_remap[static_cast<std::size_t>(old_a)];
            const int b = point_remap[static_cast<std::size_t>(old_b)];
            const int c = point_remap[static_cast<std::size_t>(old_c)];
            if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c)
            {
                continue;
            }
            simplified_faces.push_back({a, b, c});
        }
    }

    output.setFaces(facesToMatrix(simplified_faces).toGpu());
}

template <typename Scalar>
plapoint::PointCloud<Scalar, plamatrix::Device::GPU> makeEmptyOutput(
    const plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& mesh)
{
    plapoint::PointCloud<Scalar, plamatrix::Device::GPU> output(
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>(0, 3));
    if (mesh.hasColors())
    {
        output.setColors(plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>(0, 3));
    }
    if (mesh.hasIntensities())
    {
        output.setIntensities(plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::GPU>(0, 1));
    }
    output.setFaces(plamatrix::DenseMatrix<int, plamatrix::Device::GPU>(0, 3));
    return output;
}

template <typename Scalar>
plapoint::PointCloud<Scalar, plamatrix::Device::GPU> voxelClusterSimplifyGpuImpl(
    const plapoint::PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    Scalar cluster_size)
{
    if (!std::isfinite(cluster_size) || cluster_size <= Scalar(0))
    {
        throw std::invalid_argument("voxelClusterSimplify: cluster size must be finite and positive");
    }

    const std::size_t n = mesh.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("voxelClusterSimplify GPU: point count exceeds int range");
    }
    if (n == 0)
    {
        return makeEmptyOutput(mesh);
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> centroid_storage(
        static_cast<plamatrix::Index>(n), 3);
    plamatrix::DenseMatrix<int, plamatrix::Device::GPU> point_remap_gpu(
        static_cast<plamatrix::Index>(n), 1);
    const int cluster_count = voxelClusterSimplifyColumnMajor(
        mesh.points().data(),
        static_cast<int>(n),
        cluster_size,
        centroid_storage.data(),
        point_remap_gpu.data(),
        0);

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> points(
        static_cast<plamatrix::Index>(cluster_count), 3);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(points.data(), centroid_storage.data(),
                                   static_cast<std::size_t>(cluster_count) * 3u * sizeof(Scalar),
                                   cudaMemcpyDeviceToDevice));

    plapoint::PointCloud<Scalar, plamatrix::Device::GPU> output(std::move(points));
    const auto point_remap = copyPointRemapToCpu(point_remap_gpu, n);
    const auto cluster_counts = computeClusterCounts(point_remap, cluster_count);
    setAveragedColors(mesh, point_remap, cluster_counts, output);
    setAveragedIntensities(mesh, point_remap, cluster_counts, output);
    setRemappedFaces(mesh, point_remap, output);
    return output;
}

} // namespace

namespace mesh
{

PointCloud<float, plamatrix::Device::GPU> voxelClusterSimplify(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    float cluster_size)
{
    return voxelClusterSimplifyGpuImpl(mesh, cluster_size);
}

PointCloud<double, plamatrix::Device::GPU> voxelClusterSimplify(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    double cluster_size)
{
    return voxelClusterSimplifyGpuImpl(mesh, cluster_size);
}

} // namespace mesh
} // namespace plapoint
