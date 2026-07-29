#include <plapoint/gpu/spatial_index.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cub/cub.cuh>

#include <plamatrix/ops/indexing.h>
#include <plamatrix/ops/reduction.h>

namespace plapoint
{
namespace gpu
{
namespace
{

constexpr std::uint64_t kCellAxisLimit = std::uint64_t{1} << 21;
constexpr double kInt64CellLimit = 9223372036854775808.0;

template <typename Scalar>
__global__ void markFinitePointsKernel(
    const Scalar* points,
    plamatrix::Index point_count,
    std::uint8_t* finite_mask)
{
    const auto row = static_cast<plamatrix::Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= point_count)
    {
        return;
    }

    const Scalar x = points[row];
    const Scalar y = points[point_count + row];
    const Scalar z = points[2 * point_count + row];
    finite_mask[row] = static_cast<std::uint8_t>(isfinite(x) && isfinite(y) && isfinite(z));
}

template <typename Scalar>
__global__ void makeCellKeysKernel(
    const Scalar* points,
    plamatrix::Index point_count,
    Scalar cell_size,
    std::int64_t origin_x,
    std::int64_t origin_y,
    std::int64_t origin_z,
    std::uint64_t* keys,
    int* error_flag)
{
    const auto row = static_cast<plamatrix::Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= point_count)
    {
        return;
    }

    const double cell = static_cast<double>(cell_size);
    const double raw_x = floor(static_cast<double>(points[row]) / cell);
    const double raw_y = floor(static_cast<double>(points[point_count + row]) / cell);
    const double raw_z = floor(static_cast<double>(points[2 * point_count + row]) / cell);
    if (!isfinite(raw_x) || !isfinite(raw_y) || !isfinite(raw_z)
        || raw_x < -kInt64CellLimit || raw_x >= kInt64CellLimit
        || raw_y < -kInt64CellLimit || raw_y >= kInt64CellLimit
        || raw_z < -kInt64CellLimit || raw_z >= kInt64CellLimit)
    {
        atomicExch(error_flag, 1);
        keys[row] = 0;
        return;
    }
    const auto x = static_cast<std::int64_t>(raw_x);
    const auto y = static_cast<std::int64_t>(raw_y);
    const auto z = static_cast<std::int64_t>(raw_z);
    const auto normalized_x = static_cast<std::uint64_t>(x)
        - static_cast<std::uint64_t>(origin_x);
    const auto normalized_y = static_cast<std::uint64_t>(y)
        - static_cast<std::uint64_t>(origin_y);
    const auto normalized_z = static_cast<std::uint64_t>(z)
        - static_cast<std::uint64_t>(origin_z);
    if (x < origin_x || y < origin_y || z < origin_z
        || normalized_x >= kCellAxisLimit
        || normalized_y >= kCellAxisLimit
        || normalized_z >= kCellAxisLimit)
    {
        atomicExch(error_flag, 1);
        keys[row] = 0;
        return;
    }

    keys[row] = (normalized_x << 42) | (normalized_y << 21) | normalized_z;
}

std::int64_t checkedCellCoordinate(double coordinate, double cell_size)
{
    const double cell = std::floor(coordinate / cell_size);
    const double int64_limit = std::ldexp(1.0, 63);
    if (!std::isfinite(cell)
        || cell < -int64_limit
        || cell >= int64_limit)
    {
        throw std::overflow_error("GpuSpatialIndex coordinate exceeds the 64-bit cell range");
    }
    return static_cast<std::int64_t>(cell);
}

void checkAxisSpan(std::int64_t minimum, std::int64_t maximum)
{
    const auto span = static_cast<std::uint64_t>(maximum) - static_cast<std::uint64_t>(minimum);
    if (span >= kCellAxisLimit)
    {
        throw std::overflow_error("GpuSpatialIndex axis requires more than 21 cell bits");
    }
}

template <typename Scalar>
Scalar adaptiveCellSize(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& minima,
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& maxima,
    int finite_count)
{
    if (finite_count == 0)
    {
        return Scalar(1);
    }
    double scale = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        scale = std::max(scale, std::abs(static_cast<double>(minima(0, axis))));
        scale = std::max(scale, std::abs(static_cast<double>(maxima(0, axis))));
    }
    if (scale == 0.0)
    {
        return std::numeric_limits<Scalar>::denorm_min();
    }

    double normalized_extent = 0.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double minimum = static_cast<double>(minima(0, axis)) / scale;
        const double maximum = static_cast<double>(maxima(0, axis)) / scale;
        normalized_extent = std::max(normalized_extent, maximum - minimum);
    }
    const double normalized_cell = normalized_extent > 0.0
        ? normalized_extent / std::cbrt(static_cast<double>(finite_count))
        : static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    double estimate = scale * std::min(normalized_cell, 1.0);
    if (!std::isfinite(estimate) || estimate <= 0.0)
    {
        estimate = scale;
    }
    const double spacing = std::max(
        static_cast<double>(std::numeric_limits<Scalar>::denorm_min()),
        static_cast<double>(std::numeric_limits<Scalar>::epsilon()) * scale);
    return static_cast<Scalar>(std::max(estimate, spacing));
}

} // namespace

template <typename Scalar>
void GpuSpatialIndex<Scalar>::build(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    Scalar cell_size,
    cudaStream_t stream)
{
    if (!std::isfinite(cell_size) || cell_size <= Scalar(0))
    {
        throw std::invalid_argument("GpuSpatialIndex cell_size must be finite and positive");
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("GpuSpatialIndex point count exceeds int range");
    }

    GpuSpatialIndex replacement;
    replacement._sourceData = cloud.points().data();
    replacement._sourceIdentity = cloud.pointsIdentity();
    replacement._pointCount = cloud.points().rows();
    replacement._cloudRevision = cloud.pointsRevision();
    replacement._cellSize = cell_size;
    replacement._points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>
        ::uninitialized(replacement._pointCount, 3);
    if (replacement._pointCount != 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            replacement._points.data(), cloud.points().data(),
            static_cast<std::size_t>(replacement._pointCount) * 3 * sizeof(Scalar),
            cudaMemcpyDeviceToDevice, stream));
    }

    if (replacement._pointCount != 0)
    {
        const int block_size = 256;
        const int grid_size = static_cast<int>((replacement._pointCount + block_size - 1) / block_size);
        auto finite_mask = plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>
            ::uninitialized(replacement._pointCount, 1);
        markFinitePointsKernel<<<grid_size, block_size, 0, stream>>>(
            replacement._points.data(), replacement._pointCount, finite_mask.data());
        PLAPOINT_CHECK_CUDA(cudaGetLastError());

        plamatrix::IndexingWorkspace compact_workspace;
        auto compacted = plamatrix::compactRows(
            replacement._points, finite_mask, compact_workspace, stream);
        replacement._finitePointCount = static_cast<int>(compacted.values.rows());

        if (replacement._finitePointCount != 0)
        {
            plamatrix::ReductionWorkspace reduction_workspace;
            auto minima_gpu = plamatrix::min(
                compacted.values, plamatrix::ReductionAxis::Columns, reduction_workspace, stream);
            auto maxima_gpu = plamatrix::max(
                compacted.values, plamatrix::ReductionAxis::Columns, reduction_workspace, stream);
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            auto minima = minima_gpu.toCpu();
            auto maxima = maxima_gpu.toCpu();

            const double cell = static_cast<double>(cell_size);
            const auto min_x = checkedCellCoordinate(static_cast<double>(minima(0, 0)), cell);
            const auto min_y = checkedCellCoordinate(static_cast<double>(minima(0, 1)), cell);
            const auto min_z = checkedCellCoordinate(static_cast<double>(minima(0, 2)), cell);
            const auto max_x = checkedCellCoordinate(static_cast<double>(maxima(0, 0)), cell);
            const auto max_y = checkedCellCoordinate(static_cast<double>(maxima(0, 1)), cell);
            const auto max_z = checkedCellCoordinate(static_cast<double>(maxima(0, 2)), cell);
            checkAxisSpan(min_x, max_x);
            checkAxisSpan(min_y, max_y);
            checkAxisSpan(min_z, max_z);
            replacement._originX = min_x;
            replacement._originY = min_y;
            replacement._originZ = min_z;
            replacement._axisSpanX = static_cast<std::uint64_t>(max_x)
                - static_cast<std::uint64_t>(min_x);
            replacement._axisSpanY = static_cast<std::uint64_t>(max_y)
                - static_cast<std::uint64_t>(min_y);
            replacement._axisSpanZ = static_cast<std::uint64_t>(max_z)
                - static_cast<std::uint64_t>(min_z);

            const auto count = static_cast<std::size_t>(replacement._finitePointCount);
            DeviceBuffer<std::uint64_t> unsorted_keys(count);
            DeviceBuffer<int> key_error(1);
            PLAPOINT_CHECK_CUDA(cudaMemsetAsync(key_error.get(), 0, sizeof(int), stream));
            makeCellKeysKernel<<<
                static_cast<int>((static_cast<std::int64_t>(replacement._finitePointCount)
                    + block_size - 1) / block_size),
                block_size,
                0,
                stream>>>(
                compacted.values.data(), compacted.values.rows(), cell_size,
                min_x, min_y, min_z, unsorted_keys.get(), key_error.get());
            PLAPOINT_CHECK_CUDA(cudaGetLastError());
            int host_key_error = 0;
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                &host_key_error, key_error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            if (host_key_error != 0)
            {
                throw std::overflow_error("GpuSpatialIndex device cell quantization overflowed");
            }

            replacement._sortedCellKeys.allocate(count);
            replacement._sortedPointIndices =
                plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
                    ::uninitialized(static_cast<plamatrix::Index>(count), 1);
            std::size_t sort_bytes = 0;
            PLAPOINT_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                nullptr, sort_bytes, unsorted_keys.get(), replacement._sortedCellKeys.get(),
                compacted.sourceIndices.data(), replacement._sortedPointIndices.data(),
                replacement._finitePointCount, 0, 64, stream));
            DeviceBuffer<std::uint8_t> sort_storage(sort_bytes);
            PLAPOINT_CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                sort_storage.get(), sort_bytes, unsorted_keys.get(), replacement._sortedCellKeys.get(),
                compacted.sourceIndices.data(), replacement._sortedPointIndices.data(),
                replacement._finitePointCount, 0, 64, stream));

            replacement._uniqueCellKeys.allocate(count);
            replacement._cellCounts =
                plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
                    ::uninitialized(static_cast<plamatrix::Index>(count), 1);
            replacement._cellOffsets =
                plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
                    ::uninitialized(static_cast<plamatrix::Index>(count), 1);
            PLAPOINT_CHECK_CUDA(cudaMemsetAsync(
                replacement._cellCounts.data(), 0, count * sizeof(plamatrix::Index), stream));
            DeviceBuffer<int> run_count(1);
            std::size_t encode_bytes = 0;
            PLAPOINT_CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(
                nullptr, encode_bytes, replacement._sortedCellKeys.get(),
                replacement._uniqueCellKeys.get(), replacement._cellCounts.data(),
                run_count.get(), replacement._finitePointCount, stream));
            DeviceBuffer<std::uint8_t> encode_storage(encode_bytes);
            PLAPOINT_CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(
                encode_storage.get(), encode_bytes, replacement._sortedCellKeys.get(),
                replacement._uniqueCellKeys.get(), replacement._cellCounts.data(),
                run_count.get(), replacement._finitePointCount, stream));
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                &replacement._cellCount, run_count.get(), sizeof(int),
                cudaMemcpyDeviceToHost, stream));
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));

            DeviceBuffer<plamatrix::Index> maximum_count(1);
            std::size_t reduce_bytes = 0;
            PLAPOINT_CHECK_CUDA(cub::DeviceReduce::Max(
                nullptr, reduce_bytes, replacement._cellCounts.data(), maximum_count.get(),
                replacement._cellCount, stream));
            DeviceBuffer<std::uint8_t> reduce_storage(reduce_bytes);
            PLAPOINT_CHECK_CUDA(cub::DeviceReduce::Max(
                reduce_storage.get(), reduce_bytes, replacement._cellCounts.data(), maximum_count.get(),
                replacement._cellCount, stream));
            plamatrix::Index host_maximum_count = 0;
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                &host_maximum_count, maximum_count.get(), sizeof(plamatrix::Index),
                cudaMemcpyDeviceToHost, stream));
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            replacement._maxCellOccupancy = static_cast<int>(host_maximum_count);

            plamatrix::IndexingWorkspace scan_workspace;
            plamatrix::exclusiveScan(
                replacement._cellCounts, replacement._cellOffsets, scan_workspace, stream);
        }
    }

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    *this = std::move(replacement);
}

template <typename Scalar>
void GpuSpatialIndex<Scalar>::buildAdaptive(
    const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
    cudaStream_t stream)
{
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("GpuSpatialIndex point count exceeds int range");
    }
    const auto point_count = cloud.points().rows();
    if (point_count == 0)
    {
        build(cloud, Scalar(1), stream);
        return;
    }

    auto finite_mask = plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU>
        ::uninitialized(point_count, 1);
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>((point_count + block_size - 1) / block_size);
    markFinitePointsKernel<<<grid_size, block_size, 0, stream>>>(
        cloud.points().data(), point_count, finite_mask.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    plamatrix::IndexingWorkspace compact_workspace;
    auto compacted = plamatrix::compactRows(
        cloud.points(), finite_mask, compact_workspace, stream);
    const int finite_count = static_cast<int>(compacted.values.rows());
    if (finite_count == 0)
    {
        build(cloud, Scalar(1), stream);
        return;
    }

    plamatrix::ReductionWorkspace reduction_workspace;
    auto minima_gpu = plamatrix::min(
        compacted.values, plamatrix::ReductionAxis::Columns, reduction_workspace, stream);
    auto maxima_gpu = plamatrix::max(
        compacted.values, plamatrix::ReductionAxis::Columns, reduction_workspace, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto minima = minima_gpu.toCpu();
    const auto maxima = maxima_gpu.toCpu();
    build(cloud, adaptiveCellSize(minima, maxima, finite_count), stream);
}

template void GpuSpatialIndex<float>::build(
    const PointCloud<float, plamatrix::Device::GPU>&, float, cudaStream_t);
template void GpuSpatialIndex<double>::build(
    const PointCloud<double, plamatrix::Device::GPU>&, double, cudaStream_t);
template void GpuSpatialIndex<float>::buildAdaptive(
    const PointCloud<float, plamatrix::Device::GPU>&, cudaStream_t);
template void GpuSpatialIndex<double>::buildAdaptive(
    const PointCloud<double, plamatrix::Device::GPU>&, cudaStream_t);

} // namespace gpu
} // namespace plapoint

#endif
