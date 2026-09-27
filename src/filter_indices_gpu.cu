#include <plapoint/gpu/filter_indices.h>

#include <plapoint/gpu/cuda_check.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <plamatrix/internal/cuda/native.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/ops/indexing.h>
#include <plamatrix/internal/ops/reduction.h>
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>

namespace plapoint {
namespace gpu {
namespace {

template <typename Scalar>
void validatePointBuffer(const Scalar* d_points, int point_count, const char* label)
{
    if (point_count < 0)
    {
        throw std::invalid_argument(std::string(label) + ": point count must be non-negative");
    }
    if (point_count > 0 && !d_points)
    {
        throw std::invalid_argument(std::string(label) + ": point buffer must not be null");
    }
}

template <typename Scalar>
__device__ bool finitePoint(const Scalar* points, int point_count, int idx)
{
    const Scalar x = points[idx];
    const Scalar y = points[point_count + idx];
    const Scalar z = points[2 * point_count + idx];
    return isfinite(static_cast<double>(x))
        && isfinite(static_cast<double>(y))
        && isfinite(static_cast<double>(z));
}

__global__ void saturatedCountsToKeepMaskKernel(
    const plamatrix::Index* counts,
    int point_count,
    int min_neighbors,
    std::uint8_t* keep_mask)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx < point_count)
    {
        keep_mask[idx] = counts[idx] >= min_neighbors ? 1 : 0;
    }
}

__global__ void invertKeepMaskKernel(
    const std::uint8_t* keep_mask,
    int point_count,
    std::uint8_t* remove_mask)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx < point_count)
    {
        remove_mask[idx] = keep_mask[idx] == 0 ? 1 : 0;
    }
}

template <typename Scalar>
__global__ void sorMeanDistanceKernel(
    const Scalar* points,
    int point_count,
    const plamatrix::Index* knn_indices,
    int k_use,
    double* mean_distances,
    double* finite_weights,
    double* invalid_distances)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }

    if (!finitePoint(points, point_count, idx))
    {
        finite_weights[idx] = 0.0;
        invalid_distances[idx] = 0.0;
        mean_distances[idx] = 0.0;
        return;
    }

    double mean_distance = 0.0;
    int count = 0;
    bool invalid_distance = false;
    for (int k = 0; k < k_use; ++k)
    {
        const auto neighbor =
            knn_indices[idx + static_cast<std::size_t>(k) * static_cast<std::size_t>(point_count)];
        if (neighbor < 0 || neighbor >= point_count || neighbor == idx)
        {
            continue;
        }

        const double dx = static_cast<double>(points[idx])
            - static_cast<double>(points[neighbor]);
        const double dy = static_cast<double>(points[point_count + idx])
            - static_cast<double>(points[point_count + neighbor]);
        const double dz = static_cast<double>(points[2 * point_count + idx])
            - static_cast<double>(points[2 * point_count + neighbor]);
        const double distance = norm3d(dx, dy, dz);
        if (!isfinite(distance))
        {
            invalid_distance = true;
            break;
        }
        else
        {
            ++count;
            mean_distance += (distance - mean_distance) / static_cast<double>(count);
        }
    }

    finite_weights[idx] = 1.0;
    invalid_distances[idx] = invalid_distance ? 1.0 : 0.0;
    mean_distances[idx] = !invalid_distance && count > 0 ? mean_distance : 0.0;
}

__global__ void normalizeSorMeanDistancesKernel(
    double* mean_distances,
    const double* finite_weights,
    const double* maximum_mean_distance,
    int point_count)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }
    const double scale = maximum_mean_distance[0];
    if (finite_weights[idx] != 0.0 && isfinite(scale) && scale > 0.0)
    {
        mean_distances[idx] /= scale;
    }
}

__global__ void sorSquaredDeviationKernel(
    const double* mean_distances,
    const double* finite_weights,
    const double* distance_sum,
    const double* finite_count,
    int point_count,
    double* squared_deviations)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }

    if (finite_weights[idx] == 0.0 || finite_count[0] <= 0.0)
    {
        squared_deviations[idx] = 0.0;
        return;
    }
    const double mean = distance_sum[0] / finite_count[0];
    const double difference = mean_distances[idx] - mean;
    squared_deviations[idx] = difference * difference;
}

__global__ void sorThresholdKeepMaskKernel(
    const double* mean_distances,
    const double* finite_weights,
    const double* distance_sum,
    const double* finite_count,
    const double* squared_deviation_sum,
    const double* invalid_distance_count,
    int point_count,
    double stddev_mul,
    std::uint8_t* keep_mask)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }
    if (invalid_distance_count[0] > 0.0
        || finite_weights[idx] == 0.0
        || finite_count[0] <= 0.0)
    {
        keep_mask[idx] = 0;
        return;
    }
    const double global_mean = distance_sum[0] / finite_count[0];
    const double variance = squared_deviation_sum[0] / finite_count[0];
    const double threshold = global_mean + stddev_mul * sqrt(variance > 0.0 ? variance : 0.0);
    keep_mask[idx] = mean_distances[idx] <= threshold ? 1 : 0;
}

std::vector<std::uint8_t> copyMaskToHost(
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& device_mask)
{
    auto host_matrix = device_mask.toHostMatrix();
    std::vector<std::uint8_t> host_mask(static_cast<std::size_t>(host_matrix.rows()), 0);
    for (plamatrix::Index i = 0; i < host_matrix.rows(); ++i)
    {
        host_mask[static_cast<std::size_t>(i)] = host_matrix(i, 0);
    }
    return host_mask;
}

template <typename Scalar>
void validatePointMatrix(
    const plamatrix::internal::ResidentMatrix<Scalar>& points,
    const char* label)
{
    if (points.context().backend() != plamatrix::internal::Backend::Cuda)
    {
        throw std::invalid_argument(std::string(label) + ": points require a CUDA execution context");
    }
    if (points.cols() != 3)
    {
        throw std::invalid_argument(std::string(label) + ": points must be Nx3");
    }
    if (points.rows() > std::numeric_limits<int>::max())
    {
        throw std::overflow_error(std::string(label) + ": point count exceeds int range");
    }
}

template <typename Scalar>
plamatrix::internal::ResidentMatrix<std::uint8_t> radiusOutlierRemovalKeepMaskMatrixImpl(
    const plamatrix::internal::ResidentMatrix<Scalar>& points,
    Scalar radius,
    int min_neighbors)
{
    validatePointMatrix(points, "GPU radius outlier keep mask");
    const int point_count = static_cast<int>(points.rows());
    if (!std::isfinite(radius) || radius < Scalar(0))
    {
        throw std::invalid_argument("GPU radius outlier keep mask: radius must be finite and non-negative");
    }
    if (min_neighbors <= 0)
    {
        throw std::invalid_argument("GPU radius outlier keep mask: min neighbors must be positive");
    }
    auto context = points.contextOwner();
    if (!context)
    {
        throw std::invalid_argument("GPU outlier keep mask requires an owning input context");
    }
    if (point_count == 0)
    {
        return plamatrix::internal::ResidentMatrix<std::uint8_t>(0, 1, context);
    }

    auto copied_points = points.clone();
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> cloud(std::move(copied_points), context);
    OutlierRemovalGpuWorkspace<Scalar> workspace;
    return radiusOutlierRemovalKeepMaskDevice(
        cloud, radius, min_neighbors, workspace, nullptr);
}

template <typename Scalar>
plamatrix::internal::ResidentMatrix<std::uint8_t> statisticalOutlierRemovalKeepMaskMatrixImpl(
    const plamatrix::internal::ResidentMatrix<Scalar>& points,
    int mean_k,
    Scalar stddev_mul)
{
    validatePointMatrix(points, "GPU statistical outlier keep mask");
    const int point_count = static_cast<int>(points.rows());
    if (mean_k <= 0 || mean_k > std::numeric_limits<int>::max() - 1)
    {
        throw std::invalid_argument("GPU statistical outlier keep mask: mean k must be positive");
    }
    if (!std::isfinite(stddev_mul) || stddev_mul < Scalar(0))
    {
        throw std::invalid_argument("GPU statistical outlier keep mask: stddev multiplier must be non-negative");
    }
    auto context = points.contextOwner();
    if (!context)
    {
        throw std::invalid_argument("GPU outlier keep mask requires an owning input context");
    }
    if (point_count == 0)
    {
        return plamatrix::internal::ResidentMatrix<std::uint8_t>(0, 1, context);
    }

    const int k_use = std::min(mean_k + 1, point_count);
    if (k_use > 32)
    {
        throw std::invalid_argument("GPU statistical outlier keep mask supports mean_k + 1 <= 32");
    }
    auto copied_points = points.clone();
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> cloud(std::move(copied_points), context);
    OutlierRemovalGpuWorkspace<Scalar> workspace;
    return statisticalOutlierRemovalKeepMaskDevice(
        cloud, mean_k, stddev_mul, workspace, nullptr);
}

} // namespace

std::vector<int> keptIndicesFromKeepMask(const std::vector<std::uint8_t>& keep_mask)
{
    std::vector<int> indices;
    indices.reserve(keep_mask.size());
    for (std::size_t i = 0; i < keep_mask.size(); ++i)
    {
        if (keep_mask[i] != 0)
        {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

std::vector<int> removedIndicesFromKeepMask(const std::vector<std::uint8_t>& keep_mask)
{
    std::vector<int> indices;
    indices.reserve(keep_mask.size());
    for (std::size_t i = 0; i < keep_mask.size(); ++i)
    {
        if (keep_mask[i] == 0)
        {
            indices.push_back(static_cast<int>(i));
        }
    }
    return indices;
}

std::vector<std::uint8_t> keepMaskToHost(
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask)
{
    return copyMaskToHost(keep_mask);
}

std::vector<int> removedIndicesFromKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask,
    cudaStream_t stream)
{
    if (keep_mask.cols() != 1)
    {
        throw std::invalid_argument(
            "removedIndicesFromKeepMaskDevice: keep mask must have one column");
    }
    if (keep_mask.rows() > std::numeric_limits<int>::max())
    {
        throw std::overflow_error(
            "removedIndicesFromKeepMaskDevice: point count exceeds int range");
    }

    const int point_count = static_cast<int>(keep_mask.rows());
    if (point_count == 0)
    {
        return {};
    }

    plamatrix::internal::ResidentMatrix<std::uint8_t> remove_mask(
        keep_mask.rows(), 1, keep_mask.context());
    constexpr int kBlockSize = 256;
    const int grid_size = (point_count + kBlockSize - 1) / kBlockSize;
    invertKeepMaskKernel<<<grid_size, kBlockSize, 0, stream>>>(
        keep_mask.data(), point_count, remove_mask.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    plamatrix::internal::ResidentMatrix<float> empty_values(keep_mask.rows(), 0, keep_mask.context());
    plamatrix::internal::ResidentMatrix<float> compacted_values(keep_mask.rows(), 0, keep_mask.context());
    plamatrix::internal::ResidentMatrix<plamatrix::Index> compacted_indices(
        keep_mask.rows(), 1, keep_mask.context());
    plamatrix::internal::ResidentMatrix<plamatrix::Index> selected_count(1, 1, keep_mask.context());
    plamatrix::internal::IndexingWorkspace indexing_workspace;
    plamatrix::internal::compactRowsAsync(
        empty_values.template view<plamatrix::internal::Device::GPU>().asConst(),
        remove_mask.template view<plamatrix::internal::Device::GPU>().asConst(),
        compacted_values.template view<plamatrix::internal::Device::GPU>(),
        compacted_indices.template view<plamatrix::internal::Device::GPU>(),
        selected_count.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    indexing_workspace.checkStatus("removedIndicesFromKeepMaskDevice");
    indexing_workspace.closeAsyncAllocation();
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto count = selected_count.toHostMatrix()(0, 0);
    const auto host_indices = compacted_indices.toHostMatrix();
    std::vector<int> removed(static_cast<std::size_t>(count));
    for (plamatrix::Index row = 0; row < count; ++row)
    {
        removed[static_cast<std::size_t>(row)] = static_cast<int>(host_indices(row, 0));
    }
    return removed;
}

template <typename Scalar>
plamatrix::internal::ResidentMatrix<std::uint8_t>
radiusOutlierRemovalKeepMaskDevice(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    Scalar radius,
    int min_neighbors,
    OutlierRemovalGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    validatePointMatrix(cloud.points(), "GPU indexed radius outlier keep mask");
    if (!std::isfinite(radius) || radius < Scalar(0))
    {
        throw std::invalid_argument(
            "GPU indexed radius outlier keep mask: radius must be finite and non-negative");
    }
    if (min_neighbors <= 0)
    {
        throw std::invalid_argument(
            "GPU indexed radius outlier keep mask: min neighbors must be positive");
    }

    const Scalar indexed_cell_size = workspace._index.cellSize();
    if (indexed_cell_size <= Scalar(0)
        || !workspace._index.matches(cloud, indexed_cell_size))
    {
        workspace._index.buildAdaptive(cloud, stream);
        ++workspace._indexBuildCount;
    }

    const int point_count = static_cast<int>(cloud.points().rows());
    plamatrix::internal::ResidentMatrix<std::uint8_t> keep_mask(
        cloud.points().rows(), 1, cloud.executionContext());
    if (point_count != 0)
    {
        auto counts = workspace._index.radiusCountAsync(
            cloud.points(), radius, min_neighbors, workspace._queryWorkspace, stream);
        constexpr int kBlockSize = 256;
        const int grid_size = (point_count + kBlockSize - 1) / kBlockSize;
        saturatedCountsToKeepMaskKernel<<<grid_size, kBlockSize, 0, stream>>>(
            counts.data(), point_count, min_neighbors, keep_mask.data());
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
    }
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace._lastBackend = GpuOutlierRemovalBackend::UniformGrid;
    return keep_mask;
}

template <typename Scalar>
plamatrix::internal::ResidentMatrix<std::uint8_t>
statisticalOutlierRemovalKeepMaskDevice(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    int mean_k,
    Scalar stddev_mul,
    OutlierRemovalGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    validatePointMatrix(cloud.points(), "GPU indexed statistical outlier keep mask");
    if (mean_k <= 0 || mean_k > std::numeric_limits<int>::max() - 1)
    {
        throw std::invalid_argument(
            "GPU indexed statistical outlier keep mask: mean k must be positive");
    }
    if (!std::isfinite(stddev_mul) || stddev_mul < Scalar(0))
    {
        throw std::invalid_argument(
            "GPU indexed statistical outlier keep mask: stddev multiplier must be non-negative");
    }

    const int point_count = static_cast<int>(cloud.points().rows());
    const int k_use = point_count == 0 ? 0 : std::min(mean_k + 1, point_count);
    if (k_use > 32)
    {
        throw std::invalid_argument(
            "GPU indexed statistical outlier keep mask supports mean_k + 1 <= 32");
    }
    const Scalar indexed_cell_size = workspace._index.cellSize();
    if (indexed_cell_size <= Scalar(0)
        || !workspace._index.matches(cloud, indexed_cell_size))
    {
        workspace._index.buildAdaptive(cloud, stream);
        ++workspace._indexBuildCount;
    }

    plamatrix::internal::ResidentMatrix<std::uint8_t> keep_mask(
        cloud.points().rows(), 1, cloud.executionContext());
    if (point_count == 0)
    {
        workspace._lastBackend = GpuOutlierRemovalBackend::UniformGrid;
        return keep_mask;
    }

    workspace.ensureBuffers(cloud.points().rows(), cloud.executionContext());
    auto neighbors = workspace._index.knnSearchAsync(
        cloud.points(), k_use, workspace._queryWorkspace, stream);
    constexpr int kBlockSize = 256;
    const int grid_size = (point_count + kBlockSize - 1) / kBlockSize;
    sorMeanDistanceKernel<Scalar><<<grid_size, kBlockSize, 0, stream>>>(
        cloud.points().data(),
        point_count,
        neighbors.indices.data(),
        k_use,
        workspace._meanDistances->data(),
        workspace._finiteWeights->data(),
        workspace._invalidDistances->data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    plamatrix::internal::ReductionWorkspace reduction_workspace;
    plamatrix::internal::ResidentMatrix<double> maximum_mean_distance(1, 1, cloud.executionContext());
    plamatrix::internal::maxAsync(
        workspace._meanDistances->template view<plamatrix::internal::Device::GPU>().asConst(),
        plamatrix::internal::ReductionAxis::All,
        maximum_mean_distance.template view<plamatrix::internal::Device::GPU>(),
        reduction_workspace, stream);
    normalizeSorMeanDistancesKernel<<<grid_size, kBlockSize, 0, stream>>>(
        workspace._meanDistances->data(),
        workspace._finiteWeights->data(),
        maximum_mean_distance.data(),
        point_count);
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    plamatrix::internal::ResidentMatrix<double> distance_sum(1, 1, cloud.executionContext());
    plamatrix::internal::ResidentMatrix<double> finite_count(1, 1, cloud.executionContext());
    plamatrix::internal::ResidentMatrix<double> invalid_distance_count(1, 1, cloud.executionContext());
    plamatrix::internal::ResidentMatrix<double> squared_deviation_sum(1, 1, cloud.executionContext());
    plamatrix::internal::sumAsync(
        workspace._meanDistances->template view<plamatrix::internal::Device::GPU>().asConst(),
        plamatrix::internal::ReductionAxis::All,
        distance_sum.template view<plamatrix::internal::Device::GPU>(), reduction_workspace, stream);
    plamatrix::internal::sumAsync(
        workspace._finiteWeights->template view<plamatrix::internal::Device::GPU>().asConst(),
        plamatrix::internal::ReductionAxis::All,
        finite_count.template view<plamatrix::internal::Device::GPU>(), reduction_workspace, stream);
    plamatrix::internal::sumAsync(
        workspace._invalidDistances->template view<plamatrix::internal::Device::GPU>().asConst(),
        plamatrix::internal::ReductionAxis::All,
        invalid_distance_count.template view<plamatrix::internal::Device::GPU>(), reduction_workspace, stream);
    sorSquaredDeviationKernel<<<grid_size, kBlockSize, 0, stream>>>(
        workspace._meanDistances->data(),
        workspace._finiteWeights->data(),
        distance_sum.data(),
        finite_count.data(),
        point_count,
        workspace._squaredDeviations->data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    plamatrix::internal::sumAsync(
        workspace._squaredDeviations->template view<plamatrix::internal::Device::GPU>().asConst(),
        plamatrix::internal::ReductionAxis::All,
        squared_deviation_sum.template view<plamatrix::internal::Device::GPU>(), reduction_workspace, stream);
    sorThresholdKeepMaskKernel<<<grid_size, kBlockSize, 0, stream>>>(
        workspace._meanDistances->data(),
        workspace._finiteWeights->data(),
        distance_sum.data(),
        finite_count.data(),
        squared_deviation_sum.data(),
        invalid_distance_count.data(),
        point_count,
        static_cast<double>(stddev_mul),
        keep_mask.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    reduction_workspace.closeAsyncAllocation();
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace._lastBackend = GpuOutlierRemovalBackend::UniformGrid;
    return keep_mask;
}

std::vector<std::uint8_t> radiusOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, float radius, int min_neighbors)
{
    validatePointBuffer(d_points, point_count, "GPU radius outlier keep mask");
    auto context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
    plamatrix::internal::ResidentMatrix<float> points(point_count, 3, context);
    if (point_count > 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            points.data(),
            d_points,
            static_cast<std::size_t>(point_count) * 3u * sizeof(float),
            cudaMemcpyDeviceToDevice,
            plamatrix::internal::cuda::NativeAccess::stream(*context)));
        context->synchronize();
    }
    return copyMaskToHost(radiusOutlierRemovalKeepMaskMatrixImpl(points, radius, min_neighbors));
}

std::vector<std::uint8_t> radiusOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, double radius, int min_neighbors)
{
    validatePointBuffer(d_points, point_count, "GPU radius outlier keep mask");
    auto context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
    plamatrix::internal::ResidentMatrix<double> points(point_count, 3, context);
    if (point_count > 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            points.data(),
            d_points,
            static_cast<std::size_t>(point_count) * 3u * sizeof(double),
            cudaMemcpyDeviceToDevice,
            plamatrix::internal::cuda::NativeAccess::stream(*context)));
        context->synchronize();
    }
    return copyMaskToHost(radiusOutlierRemovalKeepMaskMatrixImpl(points, radius, min_neighbors));
}

plamatrix::internal::ResidentMatrix<std::uint8_t> radiusOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<float>& points,
    float radius,
    int min_neighbors)
{
    return radiusOutlierRemovalKeepMaskMatrixImpl(points, radius, min_neighbors);
}

plamatrix::internal::ResidentMatrix<std::uint8_t> radiusOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<double>& points,
    double radius,
    int min_neighbors)
{
    return radiusOutlierRemovalKeepMaskMatrixImpl(points, radius, min_neighbors);
}

std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, int mean_k, float stddev_mul)
{
    validatePointBuffer(d_points, point_count, "GPU statistical outlier keep mask");
    auto context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
    plamatrix::internal::ResidentMatrix<float> points(point_count, 3, context);
    if (point_count > 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            points.data(),
            d_points,
            static_cast<std::size_t>(point_count) * 3u * sizeof(float),
            cudaMemcpyDeviceToDevice,
            plamatrix::internal::cuda::NativeAccess::stream(*context)));
        context->synchronize();
    }
    return copyMaskToHost(statisticalOutlierRemovalKeepMaskMatrixImpl(points, mean_k, stddev_mul));
}

std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, int mean_k, double stddev_mul)
{
    validatePointBuffer(d_points, point_count, "GPU statistical outlier keep mask");
    auto context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
    plamatrix::internal::ResidentMatrix<double> points(point_count, 3, context);
    if (point_count > 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            points.data(),
            d_points,
            static_cast<std::size_t>(point_count) * 3u * sizeof(double),
            cudaMemcpyDeviceToDevice,
            plamatrix::internal::cuda::NativeAccess::stream(*context)));
        context->synchronize();
    }
    return copyMaskToHost(statisticalOutlierRemovalKeepMaskMatrixImpl(points, mean_k, stddev_mul));
}

plamatrix::internal::ResidentMatrix<std::uint8_t> statisticalOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<float>& points,
    int mean_k,
    float stddev_mul)
{
    return statisticalOutlierRemovalKeepMaskMatrixImpl(points, mean_k, stddev_mul);
}

template plamatrix::internal::ResidentMatrix<std::uint8_t>
radiusOutlierRemovalKeepMaskDevice<float>(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>&,
    float,
    int,
    OutlierRemovalGpuWorkspace<float>&,
    cudaStream_t);

template plamatrix::internal::ResidentMatrix<std::uint8_t>
radiusOutlierRemovalKeepMaskDevice<double>(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>&,
    double,
    int,
    OutlierRemovalGpuWorkspace<double>&,
    cudaStream_t);

template plamatrix::internal::ResidentMatrix<std::uint8_t>
statisticalOutlierRemovalKeepMaskDevice<float>(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>&,
    int,
    float,
    OutlierRemovalGpuWorkspace<float>&,
    cudaStream_t);

template plamatrix::internal::ResidentMatrix<std::uint8_t>
statisticalOutlierRemovalKeepMaskDevice<double>(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>&,
    int,
    double,
    OutlierRemovalGpuWorkspace<double>&,
    cudaStream_t);

plamatrix::internal::ResidentMatrix<std::uint8_t> statisticalOutlierRemovalKeepMaskDevice(
    const plamatrix::internal::ResidentMatrix<double>& points,
    int mean_k,
    double stddev_mul)
{
    return statisticalOutlierRemovalKeepMaskMatrixImpl(points, mean_k, stddev_mul);
}

} // namespace gpu
} // namespace plapoint
