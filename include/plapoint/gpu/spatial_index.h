#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cstdint>
#include <memory>
#include <optional>

#include <cuda_runtime.h>

#include <plamatrix/internal/device/device_matrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>

namespace plapoint
{
namespace gpu
{

template <typename Scalar>
class GpuSpatialIndex;

template <typename Scalar>
class GpuSpatialQueryWorkspace
{
public:
    GpuSpatialQueryWorkspace() = default;
    GpuSpatialQueryWorkspace(const GpuSpatialQueryWorkspace&) = delete;
    GpuSpatialQueryWorkspace& operator=(const GpuSpatialQueryWorkspace&) = delete;
    GpuSpatialQueryWorkspace(GpuSpatialQueryWorkspace&&) noexcept = default;
    GpuSpatialQueryWorkspace& operator=(GpuSpatialQueryWorkspace&&) noexcept = default;

private:
    friend class GpuSpatialIndex<Scalar>;

    plamatrix::internal::ResidentMatrix<double>& distanceKeys(
        plamatrix::Index rows,
        plamatrix::Index columns,
        const std::shared_ptr<plamatrix::internal::ExecutionContext>& context)
    {
        if (_context != context)
        {
            _distanceKeys.reset();
            _distanceExponents.reset();
            _context = context;
        }
        if (!_distanceKeys || _distanceKeys->rows() != rows || _distanceKeys->cols() != columns)
        {
            _distanceKeys.emplace(rows, columns, context);
        }
        return *_distanceKeys;
    }

    plamatrix::internal::ResidentMatrix<int>& distanceExponents(
        plamatrix::Index rows,
        plamatrix::Index columns,
        const std::shared_ptr<plamatrix::internal::ExecutionContext>& context)
    {
        if (_context != context)
        {
            _distanceKeys.reset();
            _distanceExponents.reset();
            _context = context;
        }
        if (!_distanceExponents || _distanceExponents->rows() != rows
            || _distanceExponents->cols() != columns)
        {
            _distanceExponents.emplace(rows, columns, context);
        }
        return *_distanceExponents;
    }

    std::shared_ptr<plamatrix::internal::ExecutionContext> _context;
    std::optional<plamatrix::internal::ResidentMatrix<double>> _distanceKeys;
    std::optional<plamatrix::internal::ResidentMatrix<int>> _distanceExponents;
};

template <typename Scalar>
struct GpuRadiusSearchResult
{
    plamatrix::internal::ResidentMatrix<plamatrix::Index> indices;
    plamatrix::internal::ResidentMatrix<Scalar> squaredDistances;
    plamatrix::internal::ResidentMatrix<plamatrix::Index> counts;
};

template <typename Scalar>
struct GpuKnnSearchResult
{
    plamatrix::internal::ResidentMatrix<plamatrix::Index> indices;
    plamatrix::internal::ResidentMatrix<Scalar> squaredDistances;
};

/// Deterministic uniform-grid index over finite points in one GPU point-cloud revision.
template <typename Scalar>
class GpuSpatialIndex
{
public:
    GpuSpatialIndex() = default;
    ~GpuSpatialIndex() = default;

    GpuSpatialIndex(const GpuSpatialIndex&) = delete;
    GpuSpatialIndex& operator=(const GpuSpatialIndex&) = delete;
    GpuSpatialIndex(GpuSpatialIndex&&) noexcept = default;
    GpuSpatialIndex& operator=(GpuSpatialIndex&&) noexcept = default;

    /// Build synchronously on stream. The previous index remains intact if construction fails.
    void build(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
               Scalar cell_size,
               cudaStream_t stream = nullptr);

    /// Build with a cell size derived from finite column bounds and the finite-point count.
    void buildAdaptive(
        const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
        cudaStream_t stream = nullptr);

    /// Return true when this index belongs to the current cloud positions and exact cell size.
    bool matches(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
                 Scalar cell_size) const noexcept
    {
        const auto& points = cloud.points();
        return _cloudRevision != 0
            && cloud.pointCachesReusable()
            && _sourceIdentity == cloud.pointsIdentity()
            && _sourceData == points.data()
            && _context == cloud.executionContext()
            && _pointCount == points.rows()
            && _cloudRevision == cloud.pointsRevision()
            && _cellSize == cell_size;
    }

    int finitePointCount() const noexcept { return _finitePointCount; }
    int cellCount() const noexcept { return _cellCount; }
    int maxCellOccupancy() const noexcept { return _maxCellOccupancy; }
    Scalar cellSize() const noexcept { return _cellSize; }
    const std::shared_ptr<plamatrix::internal::ExecutionContext>& executionContext() const noexcept
    {
        return _context;
    }
    std::uint64_t axisSpanX() const noexcept { return _axisSpanX; }
    std::uint64_t axisSpanY() const noexcept { return _axisSpanY; }
    std::uint64_t axisSpanZ() const noexcept { return _axisSpanZ; }

    const std::uint64_t* sortedCellKeysData() const noexcept
    {
        return _sortedCellKeys ? _sortedCellKeys->data() : nullptr;
    }
    const std::uint64_t* uniqueCellKeysData() const noexcept
    {
        return _uniqueCellKeys ? _uniqueCellKeys->data() : nullptr;
    }
    const plamatrix::Index* sortedPointIndicesData() const noexcept
    {
        return _sortedPointIndices ? _sortedPointIndices->data() : nullptr;
    }
    const plamatrix::Index* cellOffsetsData() const noexcept
    {
        return _cellOffsets ? _cellOffsets->data() : nullptr;
    }
    const plamatrix::Index* cellCountsData() const noexcept
    {
        return _cellCounts ? _cellCounts->data() : nullptr;
    }

    std::int64_t originX() const noexcept { return _originX; }
    std::int64_t originY() const noexcept { return _originY; }
    std::int64_t originZ() const noexcept { return _originZ; }

    /// Enqueue saturated radius counts for Qx3 GPU queries.
    /// Index, queries, result, workspace, and stream must outlive queued work.
    plamatrix::internal::ResidentMatrix<plamatrix::Index> radiusCountAsync(
        const plamatrix::internal::ResidentMatrix<Scalar>& queries,
        Scalar radius,
        int max_count,
        GpuSpatialQueryWorkspace<Scalar>& workspace,
        cudaStream_t stream) const;

    /// Enqueue bounded radius neighbors ordered by squared distance then source index.
    /// Do not overlap operations sharing a workspace; destroy results before their stream.
    GpuRadiusSearchResult<Scalar> radiusSearchAsync(
        const plamatrix::internal::ResidentMatrix<Scalar>& queries,
        Scalar radius,
        int max_neighbors,
        GpuSpatialQueryWorkspace<Scalar>& workspace,
        cudaStream_t stream) const;

    /// Enqueue KNN ordered by squared distance then source index, for 1 <= k <= 32.
    GpuKnnSearchResult<Scalar> knnSearchAsync(
        const plamatrix::internal::ResidentMatrix<Scalar>& queries,
        int k,
        GpuSpatialQueryWorkspace<Scalar>& workspace,
        cudaStream_t stream) const;

private:
    void buildImpl(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
                   Scalar cell_size,
                   bool adaptive_cell_size,
                   cudaStream_t stream);

    const Scalar* _sourceData = nullptr;
    std::shared_ptr<const void> _sourceIdentity;
    std::shared_ptr<plamatrix::internal::ExecutionContext> _context;
    std::optional<plamatrix::internal::ResidentMatrix<Scalar>> _points;
    plamatrix::Index _pointCount = 0;
    std::uint64_t _cloudRevision = 0;
    Scalar _cellSize = Scalar(0);
    int _finitePointCount = 0;
    int _cellCount = 0;
    int _maxCellOccupancy = 0;
    std::uint64_t _axisSpanX = 0;
    std::uint64_t _axisSpanY = 0;
    std::uint64_t _axisSpanZ = 0;
    std::int64_t _originX = 0;
    std::int64_t _originY = 0;
    std::int64_t _originZ = 0;
    std::optional<plamatrix::internal::ResidentMatrix<std::uint64_t>> _sortedCellKeys;
    std::optional<plamatrix::internal::ResidentMatrix<std::uint64_t>> _uniqueCellKeys;
    std::optional<plamatrix::internal::ResidentMatrix<plamatrix::Index>> _sortedPointIndices;
    std::optional<plamatrix::internal::ResidentMatrix<plamatrix::Index>> _cellOffsets;
    std::optional<plamatrix::internal::ResidentMatrix<plamatrix::Index>> _cellCounts;
};

} // namespace gpu
} // namespace plapoint

#endif
