#include <plapoint/gpu/normal_refinement.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <plapoint/gpu/cuda_check.h>

namespace plapoint
{
namespace gpu
{
namespace
{

template <typename Scalar>
__global__ void smoothNormalsKernel(
    const Scalar* source_normals,
    plamatrix::Index point_count,
    const plamatrix::Index* neighbors,
    int k,
    Scalar* output)
{
    const auto row = static_cast<plamatrix::Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= point_count)
    {
        return;
    }

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    bool has_nonfinite_neighbor = false;
    for (int neighbor = 0; neighbor < k; ++neighbor)
    {
        const auto index = neighbors[row + static_cast<plamatrix::Index>(neighbor) * point_count];
        if (index < 0 || index >= point_count)
        {
            continue;
        }
        const double x = static_cast<double>(source_normals[index]);
        const double y = static_cast<double>(source_normals[point_count + index]);
        const double z = static_cast<double>(source_normals[2 * point_count + index]);
        if (!isfinite(x) || !isfinite(y) || !isfinite(z))
        {
            has_nonfinite_neighbor = true;
            break;
        }
        sum_x += x;
        sum_y += y;
        sum_z += z;
    }

    if (has_nonfinite_neighbor)
    {
        output[row] = source_normals[row];
        output[point_count + row] = source_normals[point_count + row];
        output[2 * point_count + row] = source_normals[2 * point_count + row];
        return;
    }

    const double length = norm3d(sum_x, sum_y, sum_z);
    if (isfinite(length) && length > 1.0e-10)
    {
        output[row] = static_cast<Scalar>(sum_x / length);
        output[point_count + row] = static_cast<Scalar>(sum_y / length);
        output[2 * point_count + row] = static_cast<Scalar>(sum_z / length);
        return;
    }

    output[row] = source_normals[row];
    output[point_count + row] = source_normals[point_count + row];
    output[2 * point_count + row] = source_normals[2 * point_count + row];
}

template <typename Scalar>
__global__ void orientNormalsKernel(
    const Scalar* points,
    Scalar* normals,
    plamatrix::Index point_count,
    Scalar viewpoint_x,
    Scalar viewpoint_y,
    Scalar viewpoint_z)
{
    const auto row = static_cast<plamatrix::Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= point_count)
    {
        return;
    }

    const double px = static_cast<double>(points[row]);
    const double py = static_cast<double>(points[point_count + row]);
    const double pz = static_cast<double>(points[2 * point_count + row]);
    const double nx = static_cast<double>(normals[row]);
    const double ny = static_cast<double>(normals[point_count + row]);
    const double nz = static_cast<double>(normals[2 * point_count + row]);
    const double vx = static_cast<double>(viewpoint_x);
    const double vy = static_cast<double>(viewpoint_y);
    const double vz = static_cast<double>(viewpoint_z);
    if (!isfinite(px) || !isfinite(py) || !isfinite(pz)
        || !isfinite(nx) || !isfinite(ny) || !isfinite(nz)
        || !isfinite(vx) || !isfinite(vy) || !isfinite(vz))
    {
        return;
    }

    const double dot = (vx - px) * nx + (vy - py) * ny + (vz - pz) * nz;
    if (isfinite(dot) && dot < 0.0)
    {
        normals[row] = static_cast<Scalar>(-nx);
        normals[point_count + row] = static_cast<Scalar>(-ny);
        normals[2 * point_count + row] = static_cast<Scalar>(-nz);
    }
}

} // namespace

template <typename Scalar>
void smoothNormalsAsync(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    const GpuSpatialIndex<Scalar>& index,
    int k,
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& output,
    NormalRefinementGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    if (!cloud.hasNormals())
    {
        throw std::invalid_argument("smoothNormalsAsync requires cloud normals");
    }
    const Scalar* source_normals = cloud.normals()->data();
    if (source_normals != nullptr && output.data() == source_normals)
    {
        throw std::invalid_argument(
            "smoothNormalsAsync requires independent output storage separate from cloud normals");
    }
    if (k <= 0 || k > 32)
    {
        throw std::invalid_argument("smoothNormalsAsync requires 1 <= k <= 32");
    }
    if (!index.matches(cloud, index.cellSize()))
    {
        throw std::invalid_argument("smoothNormalsAsync requires an index matching the cloud");
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("smoothNormalsAsync point count exceeds int range");
    }

    const auto point_count = static_cast<plamatrix::Index>(cloud.size());
    workspace.bindStream(stream);
    if (output.rows() != point_count || output.cols() != 3)
    {
        output = plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>
            ::uninitialized(point_count, 3);
    }
    workspace._neighbors = index.knnSearchAsync(
        cloud.points(), k, workspace._queryWorkspace, stream);
    if (point_count == 0)
    {
        return;
    }

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((point_count + block_size - 1) / block_size);
    smoothNormalsKernel<<<grid_size, block_size, 0, stream>>>(
        cloud.normals()->data(), point_count, workspace._neighbors.indices.data(), k,
        output.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
void orientNormalsTowardViewpointAsync(
    PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    const plamatrix::Vec3<Scalar>& viewpoint,
    cudaStream_t stream)
{
    if (!cloud.hasNormals())
    {
        throw std::invalid_argument("orientNormalsTowardViewpointAsync requires cloud normals");
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            "orientNormalsTowardViewpointAsync point count exceeds int range");
    }

    const auto point_count = static_cast<plamatrix::Index>(cloud.size());
    if (point_count == 0)
    {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((point_count + block_size - 1) / block_size);
    orientNormalsKernel<<<grid_size, block_size, 0, stream>>>(
        std::as_const(cloud).points().data(), cloud.normals()->data(), point_count,
        viewpoint.x, viewpoint.y, viewpoint.z);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
}

template void smoothNormalsAsync<float>(
    const PointCloud<float, plamatrix::Device::GPU>&,
    const GpuSpatialIndex<float>&,
    int,
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU>&,
    NormalRefinementGpuWorkspace<float>&,
    cudaStream_t);
template void smoothNormalsAsync<double>(
    const PointCloud<double, plamatrix::Device::GPU>&,
    const GpuSpatialIndex<double>&,
    int,
    plamatrix::DenseMatrix<double, plamatrix::Device::GPU>&,
    NormalRefinementGpuWorkspace<double>&,
    cudaStream_t);

template void orientNormalsTowardViewpointAsync<float>(
    PointCloud<float, plamatrix::Device::GPU>&,
    const plamatrix::Vec3<float>&,
    cudaStream_t);
template void orientNormalsTowardViewpointAsync<double>(
    PointCloud<double, plamatrix::Device::GPU>&,
    const plamatrix::Vec3<double>&,
    cudaStream_t);

} // namespace gpu
} // namespace plapoint

#endif
