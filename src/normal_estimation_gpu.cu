#include <plapoint/gpu/normal_estimation.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <plapoint/gpu/cuda_check.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/ops/small_matrix.h>

namespace plapoint
{
namespace gpu
{
namespace
{

template <typename Scalar>
__global__ void accumulateCovariancesKernel(
    const Scalar* points,
    plamatrix::Index point_count,
    const plamatrix::Index* neighbors,
    int k,
    Scalar* covariances)
{
    const auto row = static_cast<plamatrix::Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= point_count)
    {
        return;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    double mean_z = 0.0;
    double xx = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yy = 0.0;
    double yz = 0.0;
    double zz = 0.0;
    int count = 0;
    for (int neighbor = 0; neighbor < k; ++neighbor)
    {
        const auto index = neighbors[row + static_cast<plamatrix::Index>(neighbor) * point_count];
        if (index < 0 || index >= point_count)
        {
            continue;
        }
        const double x = static_cast<double>(points[index]);
        const double y = static_cast<double>(points[point_count + index]);
        const double z = static_cast<double>(points[2 * point_count + index]);
        if (!isfinite(x) || !isfinite(y) || !isfinite(z))
        {
            continue;
        }
        ++count;
        const double inverse_count = 1.0 / static_cast<double>(count);
        const double dx = x - mean_x;
        const double dy = y - mean_y;
        const double dz = z - mean_z;
        mean_x += dx * inverse_count;
        mean_y += dy * inverse_count;
        mean_z += dz * inverse_count;
        xx += dx * (x - mean_x);
        xy += dx * (y - mean_y);
        xz += dx * (z - mean_z);
        yy += dy * (y - mean_y);
        yz += dy * (z - mean_z);
        zz += dz * (z - mean_z);
    }

    const bool accumulated = count >= 3
        && isfinite(xx) && isfinite(xy) && isfinite(xz)
        && isfinite(yy) && isfinite(yz) && isfinite(zz);
    const double scale = accumulated ? 1.0 / static_cast<double>(count) : 0.0;
    const double covariance_xx = xx * scale;
    const double covariance_xy = xy * scale;
    const double covariance_xz = xz * scale;
    const double covariance_yy = yy * scale;
    const double covariance_yz = yz * scale;
    const double covariance_zz = zz * scale;
    const double maximum = sizeof(Scalar) == sizeof(float) ? FLT_MAX : DBL_MAX;
    const bool representable = accumulated
        && fabs(covariance_xx) <= maximum && fabs(covariance_xy) <= maximum
        && fabs(covariance_xz) <= maximum && fabs(covariance_yy) <= maximum
        && fabs(covariance_yz) <= maximum && fabs(covariance_zz) <= maximum;
    covariances[row] = representable ? static_cast<Scalar>(covariance_xx) : Scalar(0);
    covariances[point_count + row] = representable ? static_cast<Scalar>(covariance_xy) : Scalar(0);
    covariances[2 * point_count + row] = representable ? static_cast<Scalar>(covariance_xz) : Scalar(0);
    covariances[3 * point_count + row] = representable ? static_cast<Scalar>(covariance_yy) : Scalar(0);
    covariances[4 * point_count + row] = representable ? static_cast<Scalar>(covariance_yz) : Scalar(0);
    covariances[5 * point_count + row] = representable ? static_cast<Scalar>(covariance_zz) : Scalar(0);
}

template <typename Scalar>
__global__ void extractNormalsKernel(
    const Scalar* eigenvalues,
    const Scalar* eigenvectors,
    plamatrix::Index point_count,
    Scalar* normals)
{
    const auto row = static_cast<plamatrix::Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= point_count)
    {
        return;
    }
    const double middle = static_cast<double>(eigenvalues[point_count + row]);
    const double largest = static_cast<double>(eigenvalues[2 * point_count + row]);
    double nx = static_cast<double>(eigenvectors[row]);
    double ny = static_cast<double>(eigenvectors[point_count + row]);
    double nz = static_cast<double>(eigenvectors[2 * point_count + row]);
    const double length = norm3d(nx, ny, nz);
    const double epsilon = sizeof(Scalar) == sizeof(float) ? FLT_EPSILON : DBL_EPSILON;
    const double tolerance = epsilon * 64.0 * fabs(largest);
    if (!isfinite(middle) || !isfinite(largest) || !isfinite(length)
        || middle <= tolerance || length == 0.0)
    {
        nx = 0.0;
        ny = 0.0;
        nz = 0.0;
    }
    else
    {
        nx /= length;
        ny /= length;
        nz /= length;
    }
    normals[row] = static_cast<Scalar>(nx);
    normals[point_count + row] = static_cast<Scalar>(ny);
    normals[2 * point_count + row] = static_cast<Scalar>(nz);
}

} // namespace

template <typename Scalar>
void estimateNormalsAsync(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    const GpuSpatialIndex<Scalar>& index,
    int k,
    plamatrix::internal::ResidentMatrix<Scalar>& normals,
    NormalEstimationGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    if (k < 3 || k > 32)
    {
        throw std::invalid_argument("estimateNormalsAsync requires 3 <= k <= 32");
    }
    if (!index.matches(cloud, index.cellSize()))
    {
        throw std::invalid_argument("estimateNormalsAsync requires an index matching the cloud");
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("estimateNormalsAsync point count exceeds int range");
    }

    const auto point_count = static_cast<plamatrix::Index>(cloud.size());
    normals.validateContext(*cloud.executionContext());
    if (normals.rows() != point_count || normals.cols() != 3)
    {
        throw std::invalid_argument("estimateNormalsAsync requires an N x 3 output in the cloud context");
    }
    workspace.bindStream(stream);
    workspace.resize(point_count, cloud.executionContext());
    auto neighbors = index.knnSearchAsync(
        cloud.points(), k, workspace._queryWorkspace, stream);
    if (point_count == 0)
    {
        return;
    }

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((point_count + block_size - 1) / block_size);
    accumulateCovariancesKernel<<<grid_size, block_size, 0, stream>>>(
        cloud.points().data(), point_count, neighbors.indices.data(), k,
        workspace._covariances->data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    plamatrix::internal::symmetricEigh3x3BatchedAsync(
        workspace._covariances->template view<plamatrix::internal::Device::GPU>().asConst(),
        workspace._eigenvalues->template view<plamatrix::internal::Device::GPU>(),
        workspace._eigenvectors->template view<plamatrix::internal::Device::GPU>(),
        workspace._eigenWorkspace, stream);
    extractNormalsKernel<<<grid_size, block_size, 0, stream>>>(
        workspace._eigenvalues->data(), workspace._eigenvectors->data(), point_count,
        normals.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
}

template void estimateNormalsAsync<float>(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>&,
    const GpuSpatialIndex<float>&,
    int,
    plamatrix::internal::ResidentMatrix<float>&,
    NormalEstimationGpuWorkspace<float>&,
    cudaStream_t);
template void estimateNormalsAsync<double>(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>&,
    const GpuSpatialIndex<double>&,
    int,
    plamatrix::internal::ResidentMatrix<double>&,
    NormalEstimationGpuWorkspace<double>&,
    cudaStream_t);

} // namespace gpu
} // namespace plapoint

#endif
