#include <plapoint/gpu/filter_indices.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/knn.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

template <typename Scalar>
__device__ double finiteDistance(const Scalar* points, int point_count, int lhs, int rhs)
{
    const double dx = static_cast<double>(points[lhs]) - static_cast<double>(points[rhs]);
    const double dy = static_cast<double>(points[point_count + lhs])
        - static_cast<double>(points[point_count + rhs]);
    const double dz = static_cast<double>(points[2 * point_count + lhs])
        - static_cast<double>(points[2 * point_count + rhs]);
    const double distance = norm3d(dx, dy, dz);
    return isfinite(distance) ? distance : DBL_MAX;
}

template <typename Scalar>
__global__ void radiusOutlierKeepMaskKernel(
    const Scalar* points,
    int point_count,
    Scalar radius,
    int min_neighbors,
    std::uint8_t* keep_mask)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }

    if (!finitePoint(points, point_count, idx))
    {
        keep_mask[idx] = 0;
        return;
    }

    const double radius_value = static_cast<double>(radius);
    int neighbor_count = 0;
    for (int other = 0; other < point_count; ++other)
    {
        const double distance = finiteDistance(points, point_count, idx, other);
        if (distance <= radius_value)
        {
            ++neighbor_count;
            if (neighbor_count >= min_neighbors)
            {
                break;
            }
        }
    }

    keep_mask[idx] = neighbor_count >= min_neighbors ? 1 : 0;
}

template <typename Scalar>
__global__ void columnMajorPointsToRowMajorKernel(
    const Scalar* column_major_points,
    int point_count,
    Scalar* row_major_points)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }

    const std::size_t row_offset = static_cast<std::size_t>(idx) * 3u;
    row_major_points[row_offset] = column_major_points[idx];
    row_major_points[row_offset + 1u] = column_major_points[point_count + idx];
    row_major_points[row_offset + 2u] = column_major_points[2 * point_count + idx];
}

template <typename Scalar>
__global__ void sorMeanDistanceKernel(
    const Scalar* points,
    int point_count,
    const int* knn_indices,
    int k_use,
    double* mean_distances,
    std::uint8_t* finite_mask)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }

    if (!finitePoint(points, point_count, idx))
    {
        finite_mask[idx] = 0;
        mean_distances[idx] = DBL_MAX;
        return;
    }

    double sum = 0.0;
    int count = 0;
    const std::size_t row_offset = static_cast<std::size_t>(idx) * static_cast<std::size_t>(k_use);
    for (int k = 0; k < k_use; ++k)
    {
        const int neighbor = knn_indices[row_offset + static_cast<std::size_t>(k)];
        if (neighbor < 0 || neighbor >= point_count || neighbor == idx)
        {
            continue;
        }

        const double distance = finiteDistance(points, point_count, idx, neighbor);
        if (isfinite(distance))
        {
            sum += distance;
            ++count;
        }
    }

    finite_mask[idx] = 1;
    mean_distances[idx] = count > 0 ? sum / static_cast<double>(count) : 0.0;
}

__global__ void sorThresholdKeepMaskKernel(
    const double* mean_distances,
    const std::uint8_t* finite_mask,
    int point_count,
    double threshold,
    std::uint8_t* keep_mask)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    if (idx >= point_count)
    {
        return;
    }

    keep_mask[idx] = finite_mask[idx] && mean_distances[idx] <= threshold ? 1 : 0;
}

template <typename Scalar>
std::vector<std::uint8_t> copyMaskToHost(const DeviceBuffer<std::uint8_t>& device_mask, int point_count)
{
    std::vector<std::uint8_t> host_mask(static_cast<std::size_t>(point_count), 0);
    if (point_count == 0)
    {
        return host_mask;
    }

    const std::size_t bytes = static_cast<std::size_t>(point_count) * sizeof(std::uint8_t);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(host_mask.data(), device_mask.get(), bytes, cudaMemcpyDeviceToHost));
    return host_mask;
}

template <typename Scalar>
std::vector<std::uint8_t> radiusOutlierRemovalKeepMaskImpl(
    const Scalar* d_points,
    int point_count,
    Scalar radius,
    int min_neighbors)
{
    validatePointBuffer(d_points, point_count, "GPU radius outlier keep mask");
    if (!std::isfinite(radius) || radius < Scalar(0))
    {
        throw std::invalid_argument("GPU radius outlier keep mask: radius must be finite and non-negative");
    }
    if (min_neighbors <= 0)
    {
        throw std::invalid_argument("GPU radius outlier keep mask: min neighbors must be positive");
    }
    if (point_count == 0)
    {
        return {};
    }

    constexpr int kBlockSize = 256;
    const int grid_size = (point_count + kBlockSize - 1) / kBlockSize;
    DeviceBuffer<std::uint8_t> d_keep_mask(static_cast<std::size_t>(point_count));
    radiusOutlierKeepMaskKernel<Scalar>
        <<<grid_size, kBlockSize>>>(d_points, point_count, radius, min_neighbors, d_keep_mask.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaDeviceSynchronize());

    return copyMaskToHost<Scalar>(d_keep_mask, point_count);
}

template <typename Scalar>
std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskImpl(
    const Scalar* d_points,
    int point_count,
    int mean_k,
    Scalar stddev_mul)
{
    validatePointBuffer(d_points, point_count, "GPU statistical outlier keep mask");
    if (mean_k <= 0 || mean_k > std::numeric_limits<int>::max() - 1)
    {
        throw std::invalid_argument("GPU statistical outlier keep mask: mean k must be positive");
    }
    if (!std::isfinite(stddev_mul) || stddev_mul < Scalar(0))
    {
        throw std::invalid_argument("GPU statistical outlier keep mask: stddev multiplier must be non-negative");
    }
    if (point_count == 0)
    {
        return {};
    }

    const int k_use = std::min(mean_k + 1, point_count);
    if (k_use > 32)
    {
        throw std::invalid_argument("GPU statistical outlier keep mask supports mean_k + 1 <= 32");
    }

    constexpr int kBlockSize = 256;
    const int grid_size = (point_count + kBlockSize - 1) / kBlockSize;
    const std::size_t query_scalars = static_cast<std::size_t>(point_count) * 3u;
    const std::size_t result_count =
        static_cast<std::size_t>(point_count) * static_cast<std::size_t>(k_use);

    DeviceBuffer<Scalar> d_queries(query_scalars);
    DeviceBuffer<int> d_indices(result_count);
    DeviceBuffer<Scalar> d_distances(result_count);
    DeviceBuffer<double> d_mean_distances(static_cast<std::size_t>(point_count));
    DeviceBuffer<std::uint8_t> d_finite_mask(static_cast<std::size_t>(point_count));
    DeviceBuffer<std::uint8_t> d_keep_mask(static_cast<std::size_t>(point_count));

    columnMajorPointsToRowMajorKernel<Scalar>
        <<<grid_size, kBlockSize>>>(d_points, point_count, d_queries.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());

    batchKnnDeviceColumnMajor(
        d_queries.get(), point_count, d_points, point_count, k_use, d_indices.get(), d_distances.get());

    sorMeanDistanceKernel<Scalar>
        <<<grid_size, kBlockSize>>>(
            d_points, point_count, d_indices.get(), k_use, d_mean_distances.get(), d_finite_mask.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> mean_distances(static_cast<std::size_t>(point_count), 0.0);
    std::vector<std::uint8_t> finite_mask(static_cast<std::size_t>(point_count), 0);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        mean_distances.data(),
        d_mean_distances.get(),
        mean_distances.size() * sizeof(double),
        cudaMemcpyDeviceToHost));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        finite_mask.data(),
        d_finite_mask.get(),
        finite_mask.size() * sizeof(std::uint8_t),
        cudaMemcpyDeviceToHost));

    long double global_mean = 0;
    std::size_t finite_count = 0;
    for (int i = 0; i < point_count; ++i)
    {
        if (finite_mask[static_cast<std::size_t>(i)])
        {
            global_mean += mean_distances[static_cast<std::size_t>(i)];
            ++finite_count;
        }
    }
    if (finite_count == 0)
    {
        return std::vector<std::uint8_t>(static_cast<std::size_t>(point_count), 0);
    }
    global_mean /= static_cast<long double>(finite_count);

    long double global_var = 0;
    for (int i = 0; i < point_count; ++i)
    {
        if (finite_mask[static_cast<std::size_t>(i)])
        {
            const long double diff =
                static_cast<long double>(mean_distances[static_cast<std::size_t>(i)]) - global_mean;
            global_var += diff * diff;
        }
    }
    global_var /= static_cast<long double>(finite_count);
    const long double global_stddev = std::sqrt(global_var);
    const double threshold = static_cast<double>(
        global_mean + static_cast<long double>(stddev_mul) * global_stddev);

    sorThresholdKeepMaskKernel
        <<<grid_size, kBlockSize>>>(
            d_mean_distances.get(), d_finite_mask.get(), point_count, threshold, d_keep_mask.get());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    PLAPOINT_CHECK_CUDA(cudaDeviceSynchronize());

    return copyMaskToHost<Scalar>(d_keep_mask, point_count);
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

std::vector<std::uint8_t> radiusOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, float radius, int min_neighbors)
{
    return radiusOutlierRemovalKeepMaskImpl(d_points, point_count, radius, min_neighbors);
}

std::vector<std::uint8_t> radiusOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, double radius, int min_neighbors)
{
    return radiusOutlierRemovalKeepMaskImpl(d_points, point_count, radius, min_neighbors);
}

std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const float* d_points, int point_count, int mean_k, float stddev_mul)
{
    return statisticalOutlierRemovalKeepMaskImpl(d_points, point_count, mean_k, stddev_mul);
}

std::vector<std::uint8_t> statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
    const double* d_points, int point_count, int mean_k, double stddev_mul)
{
    return statisticalOutlierRemovalKeepMaskImpl(d_points, point_count, mean_k, stddev_mul);
}

} // namespace gpu
} // namespace plapoint
