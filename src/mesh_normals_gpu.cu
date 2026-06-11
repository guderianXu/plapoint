#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/mesh_normals.h>

namespace plapoint
{
namespace mesh
{

namespace
{

constexpr int kBlockSize = 256;

int checkedIntCount(plamatrix::Index value, const char* label)
{
    if (value < 0 || value > static_cast<plamatrix::Index>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(std::string(label) + " exceeds int range");
    }
    return static_cast<int>(value);
}

std::uint64_t edgeKey(int a, int b)
{
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

template <typename T>
std::size_t checkedMatrixByteCount(const plamatrix::DenseMatrix<T, plamatrix::Device::GPU>& matrix, const char* label)
{
    return gpu::detail::checkedAllocationBytes<T>(static_cast<std::size_t>(matrix.size()), label);
}

template <typename T>
plamatrix::DenseMatrix<T, plamatrix::Device::GPU> cloneGpuMatrix(
    const plamatrix::DenseMatrix<T, plamatrix::Device::GPU>& src,
    cudaStream_t stream)
{
    plamatrix::DenseMatrix<T, plamatrix::Device::GPU> dst(src.rows(), src.cols());
    if (src.size() == 0)
    {
        return dst;
    }

    const std::size_t bytes = checkedMatrixByteCount(src, "cloneGpuMatrix");
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDeviceToDevice, stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return dst;
}

template <typename Scalar>
plamatrix::DenseMatrix<int, plamatrix::Device::GPU> cloneFacesOrEmpty(
    const PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream)
{
    if (mesh.hasFaces())
    {
        return cloneGpuMatrix(*mesh.faces(), stream);
    }
    return plamatrix::DenseMatrix<int, plamatrix::Device::GPU>(0, 3);
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::GPU> makeCloudWithCopiedAttributes(
    const PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>&& points,
    bool copy_normals,
    cudaStream_t stream)
{
    PointCloud<Scalar, plamatrix::Device::GPU> out(std::move(points));
    if (copy_normals && mesh.hasNormals())
    {
        out.setNormals(cloneGpuMatrix(*mesh.normals(), stream));
    }
    if (mesh.hasColors())
    {
        out.setColors(cloneGpuMatrix(*mesh.colors(), stream));
    }
    if (mesh.hasIntensities())
    {
        out.setIntensities(cloneGpuMatrix(*mesh.intensities(), stream));
    }
    if (mesh.hasTextureCoords())
    {
        out.setTextureCoords(cloneGpuMatrix(*mesh.textureCoords(), stream));
    }

    out.setFaces(cloneFacesOrEmpty(mesh, stream));
    if (mesh.hasFaceTextureIndices())
    {
        out.setFaceTextureIndices(cloneGpuMatrix(*mesh.faceTextureIndices(), stream));
    }
    out.setMaterialLibraryFile(mesh.materialLibraryFile());
    out.setTextureImageFile(mesh.textureImageFile());
    return out;
}

template <typename Scalar>
__global__ void accumulateFaceNormalsKernel(
    const Scalar* __restrict__ points,
    const int* __restrict__ faces,
    int point_count,
    int face_count,
    double* __restrict__ accum)
{
    const int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= face_count)
    {
        return;
    }

    const int ia = faces[face_idx];
    const int ib = faces[face_count + face_idx];
    const int ic = faces[2 * face_count + face_idx];

    const double ax = static_cast<double>(points[ia]);
    const double ay = static_cast<double>(points[point_count + ia]);
    const double az = static_cast<double>(points[2 * point_count + ia]);
    const double bx = static_cast<double>(points[ib]);
    const double by = static_cast<double>(points[point_count + ib]);
    const double bz = static_cast<double>(points[2 * point_count + ib]);
    const double cx = static_cast<double>(points[ic]);
    const double cy = static_cast<double>(points[point_count + ic]);
    const double cz = static_cast<double>(points[2 * point_count + ic]);

    const double abx = bx - ax;
    const double aby = by - ay;
    const double abz = bz - az;
    const double acx = cx - ax;
    const double acy = cy - ay;
    const double acz = cz - az;

    const double nx = aby * acz - abz * acy;
    const double ny = abz * acx - abx * acz;
    const double nz = abx * acy - aby * acx;

    atomicAdd(&accum[ia], nx);
    atomicAdd(&accum[ib], nx);
    atomicAdd(&accum[ic], nx);
    atomicAdd(&accum[point_count + ia], ny);
    atomicAdd(&accum[point_count + ib], ny);
    atomicAdd(&accum[point_count + ic], ny);
    atomicAdd(&accum[2 * point_count + ia], nz);
    atomicAdd(&accum[2 * point_count + ib], nz);
    atomicAdd(&accum[2 * point_count + ic], nz);
}

template <typename Scalar>
__global__ void normalizeVertexNormalsKernel(
    const double* __restrict__ accum,
    int point_count,
    double epsilon,
    Scalar* __restrict__ normals)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    const double nx = accum[idx];
    const double ny = accum[point_count + idx];
    const double nz = accum[2 * point_count + idx];
    const double length = sqrt(nx * nx + ny * ny + nz * nz);
    if (length <= epsilon)
    {
        normals[idx] = Scalar(0);
        normals[point_count + idx] = Scalar(0);
        normals[2 * point_count + idx] = Scalar(1);
        return;
    }

    normals[idx] = static_cast<Scalar>(nx / length);
    normals[point_count + idx] = static_cast<Scalar>(ny / length);
    normals[2 * point_count + idx] = static_cast<Scalar>(nz / length);
}

template <typename Scalar>
struct ColumnValue
{
    const Scalar* values;
    int row_count;
    int col;

    __host__ __device__ double operator()(int row) const
    {
        return static_cast<double>(values[col * row_count + row]);
    }
};

template <typename Scalar>
struct NormalCentroidDot
{
    const Scalar* points;
    const Scalar* normals;
    int point_count;
    double cx;
    double cy;
    double cz;

    __host__ __device__ double operator()(int row) const
    {
        return static_cast<double>(normals[row]) *
                   (static_cast<double>(points[row]) - cx) +
               static_cast<double>(normals[point_count + row]) *
                   (static_cast<double>(points[point_count + row]) - cy) +
               static_cast<double>(normals[2 * point_count + row]) *
                   (static_cast<double>(points[2 * point_count + row]) - cz);
    }
};

template <typename Scalar>
__global__ void flipNormalsKernel(Scalar* normals, int point_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count)
    {
        return;
    }

    normals[idx] = -normals[idx];
    normals[point_count + idx] = -normals[point_count + idx];
    normals[2 * point_count + idx] = -normals[2 * point_count + idx];
}

__global__ void flipFacesKernel(int* faces, int face_count)
{
    const int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= face_count)
    {
        return;
    }

    const int tmp = faces[face_count + face_idx];
    faces[face_count + face_idx] = faces[2 * face_count + face_idx];
    faces[2 * face_count + face_idx] = tmp;
}

template <typename Scalar>
__global__ void laplacianStepKernel(
    const Scalar* __restrict__ current,
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const unsigned char* __restrict__ boundary_flags,
    int point_count,
    Scalar factor,
    bool fix_boundary,
    Scalar* __restrict__ next)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= point_count)
    {
        return;
    }

    const int begin = offsets[row];
    const int end = offsets[row + 1];
    if ((fix_boundary && boundary_flags && boundary_flags[row] != 0) || begin == end)
    {
        next[row] = current[row];
        next[point_count + row] = current[point_count + row];
        next[2 * point_count + row] = current[2 * point_count + row];
        return;
    }

    const double inv_degree = 1.0 / static_cast<double>(end - begin);
    for (int col = 0; col < 3; ++col)
    {
        double mean = 0.0;
        const int col_offset = col * point_count;
        for (int cursor = begin; cursor < end; ++cursor)
        {
            mean += static_cast<double>(current[col_offset + indices[cursor]]);
        }
        mean *= inv_degree;
        const double current_value = static_cast<double>(current[col_offset + row]);
        next[col_offset + row] = static_cast<Scalar>(
            current_value + static_cast<double>(factor) * (mean - current_value));
    }
}

template <typename Scalar>
double reduceColumnSum(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& matrix,
    int row_count,
    int col,
    cudaStream_t stream)
{
    if (row_count == 0)
    {
        return 0.0;
    }

    auto policy = thrust::cuda::par.on(stream);
    auto begin = thrust::make_counting_iterator<int>(0);
    auto values = thrust::make_transform_iterator(begin, ColumnValue<Scalar>{matrix.data(), row_count, col});
    return thrust::reduce(policy, values, values + row_count, 0.0, thrust::plus<double>());
}

template <typename Scalar>
double reduceNormalCentroidDot(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& points,
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& normals,
    int point_count,
    double cx,
    double cy,
    double cz,
    cudaStream_t stream)
{
    if (point_count == 0)
    {
        return 0.0;
    }

    auto policy = thrust::cuda::par.on(stream);
    auto begin = thrust::make_counting_iterator<int>(0);
    auto values = thrust::make_transform_iterator(
        begin,
        NormalCentroidDot<Scalar>{points.data(), normals.data(), point_count, cx, cy, cz});
    return thrust::reduce(policy, values, values + point_count, 0.0, thrust::plus<double>());
}

template <typename Scalar>
std::vector<std::vector<int>> buildNeighbors(
    const PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    int point_count)
{
    std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(point_count));
    if (!mesh.hasFaces())
    {
        return neighbors;
    }

    const auto faces = mesh.faces()->toCpu();
    const int face_count = checkedIntCount(faces.rows(), "face count");
    for (int row = 0; row < face_count; ++row)
    {
        const int a = faces.getValue(row, 0);
        const int b = faces.getValue(row, 1);
        const int c = faces.getValue(row, 2);
        neighbors[static_cast<std::size_t>(a)].push_back(b);
        neighbors[static_cast<std::size_t>(a)].push_back(c);
        neighbors[static_cast<std::size_t>(b)].push_back(a);
        neighbors[static_cast<std::size_t>(b)].push_back(c);
        neighbors[static_cast<std::size_t>(c)].push_back(a);
        neighbors[static_cast<std::size_t>(c)].push_back(b);
    }

    for (auto& row : neighbors)
    {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return neighbors;
}

template <typename Scalar>
std::vector<unsigned char> buildBoundaryFlags(
    const PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    int point_count)
{
    std::vector<unsigned char> flags(static_cast<std::size_t>(point_count), 0);
    if (!mesh.hasFaces())
    {
        return flags;
    }

    const auto faces = mesh.faces()->toCpu();
    const int face_count = checkedIntCount(faces.rows(), "face count");
    std::map<std::uint64_t, int> edge_counts;
    for (int row = 0; row < face_count; ++row)
    {
        const int a = faces.getValue(row, 0);
        const int b = faces.getValue(row, 1);
        const int c = faces.getValue(row, 2);
        ++edge_counts[edgeKey(a, b)];
        ++edge_counts[edgeKey(b, c)];
        ++edge_counts[edgeKey(c, a)];
    }

    for (int row = 0; row < face_count; ++row)
    {
        const int a = faces.getValue(row, 0);
        const int b = faces.getValue(row, 1);
        const int c = faces.getValue(row, 2);
        const int edge_vertices[3][2]{{a, b}, {b, c}, {c, a}};
        for (const auto& edge : edge_vertices)
        {
            if (edge_counts[edgeKey(edge[0], edge[1])] == 1)
            {
                flags[static_cast<std::size_t>(edge[0])] = 1;
                flags[static_cast<std::size_t>(edge[1])] = 1;
            }
        }
    }
    return flags;
}

void buildCsr(
    const std::vector<std::vector<int>>& neighbors,
    std::vector<int>& offsets,
    std::vector<int>& indices)
{
    offsets.assign(neighbors.size() + 1, 0);
    std::size_t total = 0;
    for (std::size_t row = 0; row < neighbors.size(); ++row)
    {
        if (neighbors[row].size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("neighbor row exceeds int range");
        }
        total += neighbors[row].size();
        if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("neighbor index count exceeds int range");
        }
        offsets[row + 1] = static_cast<int>(total);
    }

    indices.reserve(total);
    for (const auto& row : neighbors)
    {
        indices.insert(indices.end(), row.begin(), row.end());
    }
}

template <typename T>
gpu::DeviceBuffer<T> uploadVector(const std::vector<T>& values, cudaStream_t stream)
{
    gpu::DeviceBuffer<T> buffer(values.size());
    if (!values.empty())
    {
        const std::size_t bytes = gpu::detail::checkedAllocationBytes<T>(values.size(), "uploadVector");
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(buffer.get(), values.data(), bytes, cudaMemcpyHostToDevice, stream));
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    return buffer;
}

template <typename Scalar>
void runLaplacianStep(
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& current,
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& scratch,
    const int* d_offsets,
    const int* d_indices,
    const unsigned char* d_boundary_flags,
    int point_count,
    Scalar factor,
    bool fix_boundary,
    cudaStream_t stream)
{
    if (factor == Scalar(0) || point_count == 0)
    {
        return;
    }

    const int grid_size = (point_count + kBlockSize - 1) / kBlockSize;
    laplacianStepKernel<Scalar><<<grid_size, kBlockSize, 0, stream>>>(
        current.data(),
        d_offsets,
        d_indices,
        d_boundary_flags,
        point_count,
        factor,
        fix_boundary,
        scratch.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    std::swap(current, scratch);
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::GPU> recomputeVertexNormalsImpl(
    const PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream)
{
    const int point_count = checkedIntCount(mesh.points().rows(), "point count");
    const int face_count = mesh.hasFaces() ? checkedIntCount(mesh.faces()->rows(), "face count") : 0;

    auto points = cloneGpuMatrix(mesh.points(), stream);
    auto out = makeCloudWithCopiedAttributes(mesh, std::move(points), false, stream);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> normals(mesh.points().rows(), 3);

    if (point_count > 0)
    {
        gpu::DeviceBuffer<double> accum(static_cast<std::size_t>(point_count) * 3u);
        PLAPOINT_CHECK_CUDA(cudaMemsetAsync(
            accum.get(),
            0,
            gpu::detail::checkedAllocationBytes<double>(accum.size(), "normal accumulation"),
            stream));

        if (face_count > 0)
        {
            const int face_grid = (face_count + kBlockSize - 1) / kBlockSize;
            accumulateFaceNormalsKernel<Scalar><<<face_grid, kBlockSize, 0, stream>>>(
                mesh.points().data(),
                mesh.faces()->data(),
                point_count,
                face_count,
                accum.get());
            PLAPOINT_CHECK_CUDA(cudaGetLastError());
        }

        const int point_grid = (point_count + kBlockSize - 1) / kBlockSize;
        normalizeVertexNormalsKernel<Scalar><<<point_grid, kBlockSize, 0, stream>>>(
            accum.get(),
            point_count,
            1.0e-18,
            normals.data());
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    out.setNormals(std::move(normals));
    return out;
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::GPU> orientNormalsOutwardFromCentroidImpl(
    const PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream)
{
    auto points = cloneGpuMatrix(mesh.points(), stream);
    auto out = makeCloudWithCopiedAttributes(mesh, std::move(points), true, stream);
    const int point_count = checkedIntCount(out.points().rows(), "point count");
    if (!out.hasNormals() || point_count == 0)
    {
        return out;
    }

    const double inv_count = 1.0 / static_cast<double>(point_count);
    const double cx = reduceColumnSum(out.points(), point_count, 0, stream) * inv_count;
    const double cy = reduceColumnSum(out.points(), point_count, 1, stream) * inv_count;
    const double cz = reduceColumnSum(out.points(), point_count, 2, stream) * inv_count;
    const double dot_sum = reduceNormalCentroidDot(out.points(), *out.normals(), point_count, cx, cy, cz, stream);

    if (dot_sum < 0.0)
    {
        const int point_grid = (point_count + kBlockSize - 1) / kBlockSize;
        flipNormalsKernel<Scalar><<<point_grid, kBlockSize, 0, stream>>>(out.normals()->data(), point_count);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());

        if (out.hasFaces())
        {
            const int face_count = checkedIntCount(out.faces()->rows(), "face count");
            if (face_count > 0)
            {
                const int face_grid = (face_count + kBlockSize - 1) / kBlockSize;
                flipFacesKernel<<<face_grid, kBlockSize, 0, stream>>>(out.faces()->data(), face_count);
                PLAPOINT_CHECK_CUDA(cudaGetLastError());
            }
        }
    }

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return out;
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::GPU> taubinSmoothImpl(
    const PointCloud<Scalar, plamatrix::Device::GPU>& mesh,
    int iterations,
    Scalar lambda,
    Scalar mu,
    bool fix_boundary,
    cudaStream_t stream)
{
    if (iterations < 0)
    {
        throw std::invalid_argument("taubinSmooth: iterations must be non-negative");
    }
    if (!std::isfinite(lambda) || !std::isfinite(mu))
    {
        throw std::invalid_argument("taubinSmooth: factors must be finite");
    }

    const int point_count = checkedIntCount(mesh.points().rows(), "point count");
    const auto neighbors = buildNeighbors(mesh, point_count);
    std::vector<int> offsets;
    std::vector<int> indices;
    buildCsr(neighbors, offsets, indices);
    const auto boundary_flags = fix_boundary
        ? buildBoundaryFlags(mesh, point_count)
        : std::vector<unsigned char>{};

    auto d_offsets = uploadVector(offsets, stream);
    auto d_indices = uploadVector(indices, stream);
    auto d_boundary_flags = uploadVector(boundary_flags, stream);

    auto current = cloneGpuMatrix(mesh.points(), stream);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> scratch(mesh.points().rows(), 3);
    for (int iter = 0; iter < iterations; ++iter)
    {
        runLaplacianStep(
            current,
            scratch,
            d_offsets.get(),
            d_indices.get(),
            d_boundary_flags.get(),
            point_count,
            lambda,
            fix_boundary,
            stream);
        runLaplacianStep(
            current,
            scratch,
            d_offsets.get(),
            d_indices.get(),
            d_boundary_flags.get(),
            point_count,
            mu,
            fix_boundary,
            stream);
    }

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    auto out = makeCloudWithCopiedAttributes(mesh, std::move(current), false, stream);
    if (mesh.hasNormals())
    {
        return recomputeVertexNormalsImpl(out, stream);
    }
    return out;
}

} // namespace

PointCloud<float, plamatrix::Device::GPU> recomputeVertexNormals(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream)
{
    return recomputeVertexNormalsImpl(mesh, stream);
}

PointCloud<double, plamatrix::Device::GPU> recomputeVertexNormals(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream)
{
    return recomputeVertexNormalsImpl(mesh, stream);
}

PointCloud<float, plamatrix::Device::GPU> orientNormalsOutwardFromCentroid(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream)
{
    return orientNormalsOutwardFromCentroidImpl(mesh, stream);
}

PointCloud<double, plamatrix::Device::GPU> orientNormalsOutwardFromCentroid(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    cudaStream_t stream)
{
    return orientNormalsOutwardFromCentroidImpl(mesh, stream);
}

PointCloud<float, plamatrix::Device::GPU> taubinSmooth(
    const PointCloud<float, plamatrix::Device::GPU>& mesh,
    int iterations,
    float lambda,
    float mu,
    bool fix_boundary,
    cudaStream_t stream)
{
    return taubinSmoothImpl(mesh, iterations, lambda, mu, fix_boundary, stream);
}

PointCloud<double, plamatrix::Device::GPU> taubinSmooth(
    const PointCloud<double, plamatrix::Device::GPU>& mesh,
    int iterations,
    double lambda,
    double mu,
    bool fix_boundary,
    cudaStream_t stream)
{
    return taubinSmoothImpl(mesh, iterations, lambda, mu, fix_boundary, stream);
}

} // namespace mesh
} // namespace plapoint
