#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include <plamatrix/dense/dense_matrix.h>

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

    plamatrix::DenseMatrix<double, plamatrix::Device::GPU>& distanceKeys(
        plamatrix::Index rows,
        plamatrix::Index columns,
        cudaStream_t)
    {
        if (_distanceKeys.rows() != rows || _distanceKeys.cols() != columns)
        {
            _distanceKeys = plamatrix::DenseMatrix<double, plamatrix::Device::GPU>
                ::uninitialized(rows, columns);
        }
        return _distanceKeys;
    }

    plamatrix::DenseMatrix<int, plamatrix::Device::GPU>& distanceExponents(
        plamatrix::Index rows,
        plamatrix::Index columns,
        cudaStream_t)
    {
        if (_distanceExponents.rows() != rows || _distanceExponents.cols() != columns)
        {
            _distanceExponents = plamatrix::DenseMatrix<int, plamatrix::Device::GPU>
                ::uninitialized(rows, columns);
        }
        return _distanceExponents;
    }

    plamatrix::DenseMatrix<double, plamatrix::Device::GPU> _distanceKeys;
    plamatrix::DenseMatrix<int, plamatrix::Device::GPU> _distanceExponents;
};

template <typename Scalar>
struct GpuRadiusSearchResult
{
    plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU> indices;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> squaredDistances;
    plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU> counts;
};

template <typename Scalar>
struct GpuKnnSearchResult
{
    plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU> indices;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> squaredDistances;
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
    void build(const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
               Scalar cell_size,
               cudaStream_t stream = nullptr);

    /// Build with a cell size derived from GPU min/max reductions and finite-point count.
    void buildAdaptive(
        const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
        cudaStream_t stream = nullptr);

    /// Return true when this index belongs to the current cloud positions and exact cell size.
    bool matches(const PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
                 Scalar cell_size) const noexcept
    {
        const auto& points = cloud.points();
        return _cloudRevision != 0
            && !cloud.hasUntrackedMutablePointAlias()
            && _sourceIdentity == cloud.pointsIdentity()
            && _sourceData == points.data()
            && _pointCount == points.rows()
            && _cloudRevision == cloud.pointsRevision()
            && _cellSize == cell_size;
    }

    int finitePointCount() const noexcept { return _finitePointCount; }
    int cellCount() const noexcept { return _cellCount; }
    int maxCellOccupancy() const noexcept { return _maxCellOccupancy; }
    Scalar cellSize() const noexcept { return _cellSize; }
    std::uint64_t axisSpanX() const noexcept { return _axisSpanX; }
    std::uint64_t axisSpanY() const noexcept { return _axisSpanY; }
    std::uint64_t axisSpanZ() const noexcept { return _axisSpanZ; }

    const std::uint64_t* sortedCellKeysData() const noexcept { return _sortedCellKeys.get(); }
    const std::uint64_t* uniqueCellKeysData() const noexcept { return _uniqueCellKeys.get(); }
    const plamatrix::Index* sortedPointIndicesData() const noexcept
    {
        return _sortedPointIndices.data();
    }
    const plamatrix::Index* cellOffsetsData() const noexcept { return _cellOffsets.data(); }
    const plamatrix::Index* cellCountsData() const noexcept { return _cellCounts.data(); }

    std::int64_t originX() const noexcept { return _originX; }
    std::int64_t originY() const noexcept { return _originY; }
    std::int64_t originZ() const noexcept { return _originZ; }

    /// Enqueue saturated radius counts for Qx3 GPU queries.
    /// Index, queries, result, workspace, and stream must outlive queued work.
    plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU> radiusCountAsync(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& queries,
        Scalar radius,
        int max_count,
        GpuSpatialQueryWorkspace<Scalar>& workspace,
        cudaStream_t stream) const;

    /// Enqueue bounded radius neighbors ordered by squared distance then source index.
    /// Do not overlap operations sharing a workspace; destroy results before their stream.
    GpuRadiusSearchResult<Scalar> radiusSearchAsync(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& queries,
        Scalar radius,
        int max_neighbors,
        GpuSpatialQueryWorkspace<Scalar>& workspace,
        cudaStream_t stream) const;

    /// Enqueue KNN ordered by squared distance then source index, for 1 <= k <= 32.
    GpuKnnSearchResult<Scalar> knnSearchAsync(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& queries,
        int k,
        GpuSpatialQueryWorkspace<Scalar>& workspace,
        cudaStream_t stream) const;

private:
    const Scalar* _sourceData = nullptr;
    std::shared_ptr<const void> _sourceIdentity;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> _points;
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
    DeviceBuffer<std::uint64_t> _sortedCellKeys;
    DeviceBuffer<std::uint64_t> _uniqueCellKeys;
    plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU> _sortedPointIndices;
    plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU> _cellOffsets;
    plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU> _cellCounts;
};

} // namespace gpu
} // namespace plapoint

#endif
