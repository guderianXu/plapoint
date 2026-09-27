#include <plapoint/gpu/spatial_index.h>

#ifdef PLAPOINT_WITH_CUDA

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <plamatrix/internal/ops/grouping.h>
#include <plamatrix/internal/ops/indexing.h>
#include <plamatrix/internal/ops/statistics.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/dense/matrix_view.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/ops/reduction.h>

namespace plapoint
{
namespace gpu
{
namespace
{

constexpr std::uint64_t kCellAxisLimit = std::uint64_t{1} << 21;
constexpr double kInt64CellLimit = 9223372036854775808.0;

template <typename Scalar>
struct FiniteStatistics
{
    plamatrix::internal::ResidentMatrix<std::uint8_t> rowMask;
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> minimum;
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> maximum;
    int validCount;
};

struct GroupingHostMetadata
{
    plamatrix::Index segmentOffsets[2];
    plamatrix::Index cellCount;
    plamatrix::Index maximumCellCount;
    int keyError;
};

template <typename Scalar>
FiniteStatistics<Scalar> computeFiniteStatistics(
    const plamatrix::internal::ResidentMatrix<Scalar>& points,
    const std::shared_ptr<plamatrix::internal::ExecutionContext>& context,
    cudaStream_t stream)
{
    plamatrix::internal::ResidentMatrix<std::uint8_t> row_mask(points.rows(), 1, context);
    plamatrix::internal::ResidentMatrix<Scalar> minimum(1, points.cols(), context);
    plamatrix::internal::ResidentMatrix<Scalar> maximum(1, points.cols(), context);
    plamatrix::internal::ResidentMatrix<plamatrix::Index> valid_count(1, 1, context);
    plamatrix::internal::ReductionWorkspace workspace;
    plamatrix::internal::finiteColumnBoundsWithMaskAsync(
        points.template view<plamatrix::internal::Device::GPU>(),
        row_mask.template view<plamatrix::internal::Device::GPU>(),
        minimum.template view<plamatrix::internal::Device::GPU>(),
        maximum.template view<plamatrix::internal::Device::GPU>(),
        valid_count.template view<plamatrix::internal::Device::GPU>(), workspace, stream);

    workspace.closeAsyncAllocation();
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    auto host_minimum = minimum.toHostMatrix();
    auto host_maximum = maximum.toHostMatrix();
    auto host_valid_count = valid_count.toHostMatrix();

    const auto finite_count = host_valid_count(0, 0);
    if (finite_count < 0 || finite_count > points.rows()
        || finite_count > static_cast<plamatrix::Index>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error(
            "GpuSpatialIndex: PlaMatrix returned an invalid finite point count");
    }
    return {
        std::move(row_mask),
        std::move(host_minimum),
        std::move(host_maximum),
        static_cast<int>(finite_count)};
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
    const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& minima,
    const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& maxima,
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
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    Scalar cell_size,
    cudaStream_t stream)
{
    buildImpl(cloud, cell_size, false, stream);
}

template <typename Scalar>
void GpuSpatialIndex<Scalar>::buildImpl(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    Scalar cell_size,
    bool adaptive_cell_size,
    cudaStream_t stream)
{
    cloud.validate();
    if (!adaptive_cell_size && (!std::isfinite(cell_size) || cell_size <= Scalar(0)))
    {
        throw std::invalid_argument("GpuSpatialIndex cell_size must be finite and positive");
    }
    if (cloud.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("GpuSpatialIndex point count exceeds int range");
    }

    GpuSpatialIndex replacement;
    replacement._context = cloud.executionContext();
    replacement._sourceData = cloud.points().data();
    replacement._sourceIdentity = cloud.pointsIdentity();
    replacement._pointCount = cloud.points().rows();
    replacement._cloudRevision = cloud.pointsRevision();
    replacement._cellSize = adaptive_cell_size ? Scalar(1) : cell_size;
    replacement._points.emplace(replacement._pointCount, 3, replacement._context);
    if (replacement._pointCount != 0)
    {
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            replacement._points->data(), cloud.points().data(),
            static_cast<std::size_t>(replacement._pointCount) * 3 * sizeof(Scalar),
            cudaMemcpyDeviceToDevice, stream));
    }

    if (replacement._pointCount != 0)
    {
        const int block_size = 256;
        auto statistics = computeFiniteStatistics(*replacement._points, replacement._context, stream);
        replacement._finitePointCount = statistics.validCount;
        if (adaptive_cell_size && replacement._finitePointCount != 0)
        {
            cell_size = adaptiveCellSize(
                statistics.minimum, statistics.maximum, statistics.validCount);
            replacement._cellSize = cell_size;
        }

        if (replacement._finitePointCount != 0)
        {
            plamatrix::internal::IndexingWorkspace compact_workspace;
            plamatrix::internal::ResidentMatrix<Scalar> compacted_points(
                replacement._pointCount, 3, replacement._context);
            plamatrix::internal::ResidentMatrix<plamatrix::Index> compacted_indices(
                replacement._pointCount, 1, replacement._context);
            plamatrix::internal::ResidentMatrix<plamatrix::Index> compacted_count(1, 1, replacement._context);
            plamatrix::internal::compactRowsAsync(
                replacement._points->template view<plamatrix::internal::Device::GPU>().asConst(),
                statistics.rowMask.template view<plamatrix::internal::Device::GPU>().asConst(),
                compacted_points.template view<plamatrix::internal::Device::GPU>(),
                compacted_indices.template view<plamatrix::internal::Device::GPU>(),
                compacted_count.template view<plamatrix::internal::Device::GPU>(),
                compact_workspace, stream);
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            compact_workspace.checkStatus("GpuSpatialIndex finite point compaction");
            compact_workspace.closeAsyncAllocation();
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            if (compacted_count.toHostMatrix()(0, 0) != replacement._finitePointCount)
            {
                throw std::runtime_error(
                    "GpuSpatialIndex: inconsistent PlaMatrix finite point count");
            }

            const double cell = static_cast<double>(cell_size);
            const auto min_x = checkedCellCoordinate(static_cast<double>(statistics.minimum(0, 0)), cell);
            const auto min_y = checkedCellCoordinate(static_cast<double>(statistics.minimum(0, 1)), cell);
            const auto min_z = checkedCellCoordinate(static_cast<double>(statistics.minimum(0, 2)), cell);
            const auto max_x = checkedCellCoordinate(static_cast<double>(statistics.maximum(0, 0)), cell);
            const auto max_y = checkedCellCoordinate(static_cast<double>(statistics.maximum(0, 1)), cell);
            const auto max_z = checkedCellCoordinate(static_cast<double>(statistics.maximum(0, 2)), cell);
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

            const auto matrix_count = static_cast<plamatrix::Index>(replacement._finitePointCount);
            const auto count = static_cast<std::size_t>(matrix_count);
            plamatrix::internal::ResidentMatrix<Scalar> finite_points(matrix_count, 3, replacement._context);
            PLAPOINT_CHECK_CUDA(cudaMemcpy2DAsync(
                finite_points.data(), count * sizeof(Scalar),
                compacted_points.data(), static_cast<std::size_t>(replacement._pointCount) * sizeof(Scalar),
                count * sizeof(Scalar), 3, cudaMemcpyDeviceToDevice, stream));
            plamatrix::internal::ResidentMatrix<std::uint64_t> unsorted_keys(
                matrix_count, 1, replacement._context);
            DeviceBuffer<int> key_error(1);
            HostPinnedBuffer<GroupingHostMetadata> host_metadata(1);
            auto* metadata = host_metadata.get();
            metadata->segmentOffsets[0] = 0;
            metadata->segmentOffsets[1] = matrix_count;
            PLAPOINT_CHECK_CUDA(cudaMemsetAsync(key_error.get(), 0, sizeof(int), stream));
            makeCellKeysKernel<<<
                static_cast<int>((static_cast<std::int64_t>(replacement._finitePointCount)
                    + block_size - 1) / block_size),
                block_size,
                0,
                stream>>>(
                finite_points.data(), matrix_count, cell_size,
                min_x, min_y, min_z, unsorted_keys.data(), key_error.get());
            PLAPOINT_CHECK_CUDA(cudaGetLastError());
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                &metadata->keyError, key_error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));

            replacement._sortedCellKeys.emplace(matrix_count, 1, replacement._context);
            replacement._sortedPointIndices.emplace(matrix_count, 1, replacement._context);
            plamatrix::internal::GroupingWorkspace grouping_workspace;
            plamatrix::internal::sortByKeyAsync(
                unsorted_keys.template view<plamatrix::internal::Device::GPU>().asConst(),
                plamatrix::internal::ConstMatrixView<plamatrix::Index, plamatrix::internal::Device::GPU>(
                    compacted_indices.data(), matrix_count, 1, 1, matrix_count),
                replacement._sortedCellKeys->template view<plamatrix::internal::Device::GPU>(),
                replacement._sortedPointIndices->template view<plamatrix::internal::Device::GPU>(),
                grouping_workspace,
                stream);

            replacement._uniqueCellKeys.emplace(matrix_count, 1, replacement._context);
            replacement._cellCounts.emplace(matrix_count, 1, replacement._context);
            replacement._cellOffsets.emplace(matrix_count, 1, replacement._context);
            PLAPOINT_CHECK_CUDA(cudaMemsetAsync(
                replacement._cellCounts->data(), 0, count * sizeof(plamatrix::Index), stream));
            plamatrix::internal::ResidentMatrix<plamatrix::Index> run_count(1, 1, replacement._context);
            plamatrix::internal::runLengthEncodeAsync(
                replacement._sortedCellKeys->template view<plamatrix::internal::Device::GPU>().asConst(),
                replacement._uniqueCellKeys->template view<plamatrix::internal::Device::GPU>(),
                replacement._cellCounts->template view<plamatrix::internal::Device::GPU>(),
                run_count.template view<plamatrix::internal::Device::GPU>(),
                grouping_workspace,
                stream);

            plamatrix::internal::ResidentMatrix<plamatrix::Index> maximum_count(1, 1, replacement._context);
            plamatrix::internal::ResidentMatrix<plamatrix::Index> device_segment_offsets(2, 1, replacement._context);
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                device_segment_offsets.data(), metadata->segmentOffsets,
                sizeof(metadata->segmentOffsets), cudaMemcpyHostToDevice, stream));
            plamatrix::internal::segmentedReduceAsync(
                replacement._cellCounts->template view<plamatrix::internal::Device::GPU>().asConst(),
                device_segment_offsets.template view<plamatrix::internal::Device::GPU>().asConst(),
                plamatrix::internal::GroupReduction::Maximum,
                maximum_count.template view<plamatrix::internal::Device::GPU>(),
                grouping_workspace,
                stream);

            plamatrix::internal::IndexingWorkspace scan_workspace;
            plamatrix::internal::exclusiveScanAsync(
                replacement._cellCounts->template view<plamatrix::internal::Device::GPU>().asConst(),
                replacement._cellOffsets->template view<plamatrix::internal::Device::GPU>(),
                scan_workspace, stream);

            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                &metadata->cellCount, run_count.data(), sizeof(plamatrix::Index),
                cudaMemcpyDeviceToHost, stream));
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
                &metadata->maximumCellCount, maximum_count.data(), sizeof(plamatrix::Index),
                cudaMemcpyDeviceToHost, stream));
            grouping_workspace.closeAsyncAllocation();
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            scan_workspace.checkStatus("GpuSpatialIndex cell offsets");
            scan_workspace.closeAsyncAllocation();

            if (metadata->keyError != 0)
            {
                throw std::overflow_error("GpuSpatialIndex device cell quantization overflowed");
            }
            const auto encoded_cell_count = metadata->cellCount;
            const auto maximum_cell_count = metadata->maximumCellCount;
            if (encoded_cell_count <= 0 || encoded_cell_count > matrix_count
                || maximum_cell_count <= 0
                || maximum_cell_count > matrix_count)
            {
                throw std::runtime_error("GpuSpatialIndex: PlaMatrix returned invalid grouping metadata");
            }
            replacement._cellCount = static_cast<int>(encoded_cell_count);
            replacement._maxCellOccupancy = static_cast<int>(maximum_cell_count);
        }
    }

    *this = std::move(replacement);
}

template <typename Scalar>
void GpuSpatialIndex<Scalar>::buildAdaptive(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
    cudaStream_t stream)
{
    buildImpl(cloud, Scalar(1), true, stream);
}

template void GpuSpatialIndex<float>::build(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>&, float, cudaStream_t);
template void GpuSpatialIndex<double>::build(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>&, double, cudaStream_t);
template void GpuSpatialIndex<float>::buildAdaptive(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>&, cudaStream_t);
template void GpuSpatialIndex<double>::buildAdaptive(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>&, cudaStream_t);

} // namespace gpu
} // namespace plapoint

#endif
