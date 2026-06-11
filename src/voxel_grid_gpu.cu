#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_grid.h>

namespace plapoint
{
namespace gpu
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

template <typename Scalar>
struct ComputeVoxelKey
{
    const Scalar* points;
    int point_count;
    Scalar leaf_x;
    Scalar leaf_y;
    Scalar leaf_z;

    __host__ __device__ VoxelKey operator()(int idx) const
    {
        const Scalar x = points[idx];
        const Scalar y = points[point_count + idx];
        const Scalar z = points[2 * point_count + idx];
        return {
            static_cast<int>(floor(static_cast<double>(x) / static_cast<double>(leaf_x))),
            static_cast<int>(floor(static_cast<double>(y) / static_cast<double>(leaf_y))),
            static_cast<int>(floor(static_cast<double>(z) / static_cast<double>(leaf_z)))
        };
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
    int voxel_count,
    Scalar* out_points)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= voxel_count)
    {
        return;
    }

    out_points[idx] = static_cast<Scalar>(mean_x[idx].mean);
    out_points[voxel_count + idx] = static_cast<Scalar>(mean_y[idx].mean);
    out_points[2 * voxel_count + idx] = static_cast<Scalar>(mean_z[idx].mean);
}

template <typename Scalar>
__device__ int voxelCoordinateValidationError(Scalar coordinate, Scalar leaf)
{
    const double value = static_cast<double>(coordinate);
    if (!isfinite(value))
    {
        return 1;
    }
    const double scaled = floor(value / static_cast<double>(leaf));
    if (!isfinite(scaled) ||
        scaled < static_cast<double>(INT_MIN) ||
        scaled > static_cast<double>(INT_MAX))
    {
        return 2;
    }
    return 0;
}

template <typename Scalar>
__global__ void validateVoxelGridInputKernel(
    const Scalar* points,
    int point_count,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z,
    int* error_code)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    int error = voxelCoordinateValidationError(points[idx], leaf_x);
    if (error == 0) error = voxelCoordinateValidationError(points[point_count + idx], leaf_y);
    if (error == 0) error = voxelCoordinateValidationError(points[2 * point_count + idx], leaf_z);
    if (error != 0)
    {
        atomicCAS(error_code, 0, error);
    }
}

template <typename Scalar>
void validateVoxelGridInputColumnMajorImpl(const Scalar* d_points, int N,
                                           Scalar leaf_x, Scalar leaf_y, Scalar leaf_z,
                                           cudaStream_t stream)
{
    if (N <= 0)
    {
        return;
    }

    DeviceBuffer<int> d_error(1);
    int host_error = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(d_error.get(), &host_error, sizeof(host_error),
                                        cudaMemcpyHostToDevice, stream));

    constexpr int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    validateVoxelGridInputKernel<Scalar><<<grid_size, block_size, 0, stream>>>(
        d_points, N, leaf_x, leaf_y, leaf_z, d_error.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(&host_error, d_error.get(), sizeof(host_error),
                                        cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

    if (host_error == 1)
    {
        throw std::invalid_argument("VoxelGrid GPU: points must be finite");
    }
    if (host_error == 2)
    {
        throw std::out_of_range("VoxelGrid GPU: voxel index is outside int range");
    }
}

template <typename Scalar>
int voxelGridDownsampleColumnMajorImpl(const Scalar* d_points, int N,
                                       Scalar leaf_x, Scalar leaf_y, Scalar leaf_z,
                                       Scalar* d_out_points,
                                       cudaStream_t stream)
{
    if (N <= 0)
    {
        return 0;
    }
    if (!d_points || !d_out_points)
    {
        throw std::invalid_argument("VoxelGrid GPU: device pointers must not be null");
    }
    if (!std::isfinite(leaf_x) || !std::isfinite(leaf_y) || !std::isfinite(leaf_z) ||
        leaf_x <= Scalar(0) || leaf_y <= Scalar(0) || leaf_z <= Scalar(0))
    {
        throw std::invalid_argument("VoxelGrid GPU: leaf size must be positive");
    }
    validateVoxelGridInputColumnMajorImpl(d_points, N, leaf_x, leaf_y, leaf_z, stream);

    auto policy = thrust::cuda::par.on(stream);
    thrust::device_vector<int> indices(static_cast<std::size_t>(N));
    thrust::device_vector<VoxelKey> keys(static_cast<std::size_t>(N));
    thrust::sequence(policy, indices.begin(), indices.end(), 0);
    thrust::transform(policy, indices.begin(), indices.end(), keys.begin(),
                      ComputeVoxelKey<Scalar>{d_points, N, leaf_x, leaf_y, leaf_z});
    thrust::sort_by_key(policy, keys.begin(), keys.end(), indices.begin(), VoxelKeyLess{});

    thrust::device_vector<VoxelKey> unique_keys(static_cast<std::size_t>(N));
    thrust::device_vector<MeanAccum> mean_x(static_cast<std::size_t>(N));
    thrust::device_vector<MeanAccum> mean_y(static_cast<std::size_t>(N));
    thrust::device_vector<MeanAccum> mean_z(static_cast<std::size_t>(N));

    auto x_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinateMean<Scalar, 0>{d_points, N});
    auto y_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinateMean<Scalar, 1>{d_points, N});
    auto z_values = thrust::make_transform_iterator(
        indices.begin(), GatherCoordinateMean<Scalar, 2>{d_points, N});

    auto reduced_x = thrust::reduce_by_key(policy,
                                           keys.begin(), keys.end(),
                                           x_values,
                                           unique_keys.begin(),
                                           mean_x.begin(),
                                           VoxelKeyEqual{},
                                           MeanAccumPlus{});
    const int voxel_count = static_cast<int>(reduced_x.first - unique_keys.begin());

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

    constexpr int block_size = 256;
    const int grid_size = (voxel_count + block_size - 1) / block_size;
    writeCentroidsColumnMajor<Scalar><<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(mean_x.data()),
        thrust::raw_pointer_cast(mean_y.data()),
        thrust::raw_pointer_cast(mean_z.data()),
        voxel_count,
        d_out_points);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return voxel_count;
}

} // namespace

int voxelGridDownsampleColumnMajor(const float* d_points, int N,
                                   float leaf_x, float leaf_y, float leaf_z,
                                   float* d_out_points,
                                   cudaStream_t stream)
{
    return voxelGridDownsampleColumnMajorImpl<float>(d_points, N, leaf_x, leaf_y, leaf_z, d_out_points, stream);
}

int voxelGridDownsampleColumnMajor(const double* d_points, int N,
                                   double leaf_x, double leaf_y, double leaf_z,
                                   double* d_out_points,
                                   cudaStream_t stream)
{
    return voxelGridDownsampleColumnMajorImpl<double>(d_points, N, leaf_x, leaf_y, leaf_z, d_out_points, stream);
}

} // namespace gpu
} // namespace plapoint
