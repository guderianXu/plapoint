#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include <plapoint/gpu/cuda_check.h>

namespace plapoint
{
namespace gpu
{

/// Small host-side summary of GPU ICP correspondences and centered moment sums.
template <typename Scalar>
struct IcpCorrespondenceStats
{
    int active_count = 0;
    int invalid_source_count = 0;
    double src_centroid[3]{};
    double tgt_centroid[3]{};
    double cross_covariance[9]{};
    double src_covariance[9]{};
    double tgt_covariance[9]{};
    double residual_sq_sum = 0.0;
    bool src_has_non_collinear_geometry = false;
    bool tgt_has_non_collinear_geometry = false;
};

/// Small host-side summary of GPU ICP residual metrics.
template <typename Scalar>
struct IcpResidualStats
{
    int active_count = 0;
    int invalid_source_count = 0;
    double residual_sq_sum = 0.0;
};

/// Small host-side result from the GPU ICP step-transform solver.
template <typename Scalar>
struct IcpStepTransformResult
{
    Scalar delta = Scalar(0);
};

/// Host-side result from a fused GPU ICP stats reduction and step-transform solve.
template <typename Scalar>
struct IcpStatsAndStepTransformResult
{
    IcpCorrespondenceStats<Scalar> stats;
    IcpStepTransformResult<Scalar> step;
    bool step_valid = false;
};

/// Host-side compact result for one GPU ICP alignment iteration.
/// It omits full covariance data that the alignment loop does not need.
template <typename Scalar>
struct IcpAlignmentStepResult
{
    int active_count = 0;
    int invalid_source_count = 0;
    double residual_sq_sum = 0.0;
    double step_residual_sq_sum = 0.0;
    bool src_has_non_collinear_geometry = false;
    bool tgt_has_non_collinear_geometry = false;
    bool all_correspondences_same_index = false;
    bool step_maps_correspondences_exactly = false;
    IcpStepTransformResult<Scalar> step;
    bool step_valid = false;
};

/// Host-side compact result for a terminal GPU ICP step fused with final residual metrics.
template <typename Scalar>
struct IcpTerminalAlignmentAndResidualResult
{
    IcpAlignmentStepResult<Scalar> alignment_step;
    IcpResidualStats<Scalar> residual_stats;
    bool launched = false;
};

/// Host-side compact result for two queued small-target GPU ICP steps and terminal residual metrics.
template <typename Scalar>
struct IcpSmallTargetTwoStepTerminalAlignmentAndResidualResult
{
    IcpAlignmentStepResult<Scalar> first_alignment_step;
    IcpTerminalAlignmentAndResidualResult<Scalar> terminal_result;
    bool launched = false;
};

/// Host-side compact result for two queued small-target GPU ICP alignment iterations.
template <typename Scalar>
struct IcpSmallTargetTwoStepAlignmentResult
{
    IcpAlignmentStepResult<Scalar> first_alignment_step;
    IcpAlignmentStepResult<Scalar> second_alignment_step;
    bool launched = false;
};

/// Reusable device storage for ICP correspondence stats reductions.
/// Reserve it once for a source size and pass it to repeated stats calls to avoid repeated allocations.
class IcpCorrespondenceStatsWorkspace
{
public:
    IcpCorrespondenceStatsWorkspace() = default;

    /// Reserve enough storage for the supplied source point count. Throws for negative counts.
    void reserve(int source_count);

    /// Reserve storage for correspondence partials and the double-sized compact alignment-step result.
    void reserveAlignmentStep(int source_count);

    /// Reserve storage for correspondence partials and the float compact alignment-step result.
    void reserveFloatAlignmentStep(int source_count);

    /// Reserve storage for correspondence partials and the double compact alignment-step result.
    void reserveDoubleAlignmentStep(int source_count);

    /// Reserve storage for correspondence partials and a fused full-stats plus step-transform result.
    void reserveStatsAndStep(int source_count);

    /// Reserve storage for compact residual-stats partials and result.
    void reserveResidualStats(int source_count);

    /// Reserve reusable target tile bounds storage for finite-radius pruning.
    void reserveTargetTileBounds(int target_count);

    /// Clear cached target tile-bound metadata. Call this if target device contents mutate in place.
    void invalidateTargetTileBoundsCache();

    /// Reserve reusable target spatial grid storage for finite-radius candidate search.
    void reserveTargetSpatialGrid(int target_count);

    /// Reserve reusable target spatial grid storage using Scalar-sized sorted target coordinates.
    template <typename Scalar>
    void reserveTargetSpatialGridForScalar(int target_count)
    {
        reserveTargetSpatialGrid(target_count, sizeof(Scalar));
    }

    /// Clear cached target spatial-grid metadata. Call this if target device contents mutate in place.
    void invalidateTargetSpatialGridCache();

    /// Return true when the cached double-width target spatial grid matches the target identity and cell size.
    bool targetSpatialGridCacheMatches(const void* target_points, int target_count, double cell_size) const;

    /// Return true when the cached target spatial grid matches the target identity, cell size, and Scalar width.
    template <typename Scalar>
    bool targetSpatialGridCacheMatchesForScalar(const void* target_points, int target_count, double cell_size) const
    {
        return targetSpatialGridCacheMatches(target_points, target_count, cell_size, sizeof(Scalar));
    }

    /// Return true when the cached target tile bounds match the supplied target identity.
    bool targetTileBoundsCacheMatches(const void* target_points, int target_count) const;

    /// Mark the reusable target spatial grid storage as containing the supplied target grid.
    void markTargetSpatialGridCache(const void* target_points, int target_count, double cell_size, int cell_count);

    /// Reserve the optional dense target-grid cell lookup table.
    void reserveTargetSpatialGridDirectLookup(int entry_count);

    /// Mark the optional dense target-grid cell lookup metadata for the current cached spatial grid.
    void markTargetSpatialGridDirectLookupCache(
        int min_x,
        int min_y,
        int min_z,
        int range_x,
        int range_y,
        int range_z,
        int entry_count);

    /// Mark the reusable target tile bounds storage as containing bounds for the supplied target.
    void markTargetTileBoundsCache(const void* target_points, int target_count);

    /// Return the currently reserved partial reduction capacity, in blocks.
    int partialCapacity() const { return _partial_capacity; }

    /// Return the currently reserved target tile bound capacity, in tiles.
    int targetTileBoundCapacity() const { return _target_tile_bound_capacity; }

    /// Return the currently reserved target spatial grid capacity, in points.
    int targetSpatialGridCapacity() const { return _target_spatial_grid_capacity; }

    /// Return the number of unique cells in the currently cached target spatial grid.
    int targetSpatialGridCellCount() const { return _target_spatial_grid_cell_count; }

    /// Return the target point buffer identity used by the cached spatial grid.
    const void* targetSpatialGridPoints() const { return _target_spatial_grid_points; }

    /// Return the target point count used by the cached spatial grid.
    int targetSpatialGridPointCount() const { return _target_spatial_grid_point_count; }

    /// Return the reusable partial reduction storage pointer, or null before reserve().
    unsigned char* partialStorage() { return _partial_storage.get(); }

    /// Return the reusable final stats storage pointer, or null before reserve().
    unsigned char* statsStorage() { return _stats_storage.get(); }

    /// Return the reusable pinned host result storage pointer, or null before a result-producing reserve call.
    unsigned char* hostResultStorage() { return _host_result_storage.get(); }

    /// Return the currently reserved pinned host result storage capacity, in bytes.
    std::size_t hostResultStorageCapacity() const { return _host_result_storage.size(); }

    /// Return the reusable target tile bounds storage pointer, or null before reserveTargetTileBounds().
    unsigned char* targetTileBoundsStorage() { return _target_tile_bounds_storage.get(); }

    /// Return the reusable sorted target-grid key storage pointer.
    unsigned char* targetSpatialGridKeysStorage() { return _target_spatial_grid_keys_storage.get(); }

    /// Return the reusable unique target-grid cell key storage pointer.
    unsigned char* targetSpatialGridUniqueKeysStorage() { return _target_spatial_grid_unique_keys_storage.get(); }

    /// Return the reusable sorted target index storage pointer.
    unsigned char* targetSpatialGridIndicesStorage() { return _target_spatial_grid_indices_storage.get(); }

    /// Return the reusable target-index to sorted-offset storage pointer.
    unsigned char* targetSpatialGridSortedOffsetsStorage() { return _target_spatial_grid_sorted_offsets_storage.get(); }

    /// Return the reusable sorted target x-coordinate storage pointer.
    unsigned char* targetSpatialGridSortedXStorage() { return _target_spatial_grid_sorted_x_storage.get(); }

    /// Return the reusable sorted target y-coordinate storage pointer.
    unsigned char* targetSpatialGridSortedYStorage() { return _target_spatial_grid_sorted_y_storage.get(); }

    /// Return the reusable sorted target z-coordinate storage pointer.
    unsigned char* targetSpatialGridSortedZStorage() { return _target_spatial_grid_sorted_z_storage.get(); }

    /// Return the reusable target-grid cell start storage pointer.
    unsigned char* targetSpatialGridCellStartsStorage() { return _target_spatial_grid_cell_starts_storage.get(); }

    /// Return the reusable target-grid cell count storage pointer.
    unsigned char* targetSpatialGridCellCountsStorage() { return _target_spatial_grid_cell_counts_storage.get(); }

    /// Return the reusable dense target-grid direct cell lookup storage pointer.
    unsigned char* targetSpatialGridDirectLookupStorage() { return _target_spatial_grid_direct_lookup_storage.get(); }

    /// Return the currently reserved dense target-grid direct lookup capacity, in entries.
    int targetSpatialGridDirectLookupCapacity() const { return _target_spatial_grid_direct_lookup_capacity; }

    /// Return the dense target-grid direct lookup entry count for the currently cached grid, or zero if inactive.
    int targetSpatialGridDirectLookupEntryCount() const { return _target_spatial_grid_direct_lookup_entry_count; }

    /// Return true once the current cached spatial grid has evaluated direct lookup eligibility.
    bool targetSpatialGridDirectLookupEvaluated() const { return _target_spatial_grid_direct_lookup_evaluated; }

    /// Return the minimum x-cell coordinate covered by the direct target-grid lookup table.
    int targetSpatialGridDirectLookupMinX() const { return _target_spatial_grid_direct_lookup_min_x; }

    /// Return the minimum y-cell coordinate covered by the direct target-grid lookup table.
    int targetSpatialGridDirectLookupMinY() const { return _target_spatial_grid_direct_lookup_min_y; }

    /// Return the minimum z-cell coordinate covered by the direct target-grid lookup table.
    int targetSpatialGridDirectLookupMinZ() const { return _target_spatial_grid_direct_lookup_min_z; }

    /// Return the x-cell count covered by the direct target-grid lookup table.
    int targetSpatialGridDirectLookupRangeX() const { return _target_spatial_grid_direct_lookup_range_x; }

    /// Return the y-cell count covered by the direct target-grid lookup table.
    int targetSpatialGridDirectLookupRangeY() const { return _target_spatial_grid_direct_lookup_range_y; }

    /// Return the z-cell count covered by the direct target-grid lookup table.
    int targetSpatialGridDirectLookupRangeZ() const { return _target_spatial_grid_direct_lookup_range_z; }

private:
    void reservePartialStorage(int source_count, std::size_t bytes_per_partial);
    void reserveStatsStorage(std::size_t byte_count);
    void reserveHostResultStorage(std::size_t byte_count);
    void reserveAlignmentStepStorage(int source_count, std::size_t result_byte_count);
    void reserveTargetSpatialGrid(int target_count, std::size_t coordinate_value_bytes);
    bool targetSpatialGridCacheMatches(
        const void* target_points,
        int target_count,
        double cell_size,
        std::size_t coordinate_value_bytes) const;

    DeviceBuffer<unsigned char> _partial_storage;
    DeviceBuffer<unsigned char> _stats_storage;
    HostPinnedBuffer<unsigned char> _host_result_storage;
    DeviceBuffer<unsigned char> _target_tile_bounds_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_keys_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_unique_keys_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_indices_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_sorted_offsets_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_sorted_x_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_sorted_y_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_sorted_z_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_cell_starts_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_cell_counts_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_direct_lookup_storage;
    int _partial_capacity = 0;
    int _target_tile_bound_capacity = 0;
    const void* _target_tile_bounds_points = nullptr;
    int _target_tile_bounds_point_count = 0;
    bool _target_tile_bounds_cache_valid = false;
    int _target_spatial_grid_capacity = 0;
    int _target_spatial_grid_cell_count = 0;
    const void* _target_spatial_grid_points = nullptr;
    int _target_spatial_grid_point_count = 0;
    double _target_spatial_grid_cell_size = 0.0;
    std::size_t _target_spatial_grid_coordinate_value_bytes = 0;
    bool _target_spatial_grid_cache_valid = false;
    int _target_spatial_grid_direct_lookup_capacity = 0;
    int _target_spatial_grid_direct_lookup_entry_count = 0;
    int _target_spatial_grid_direct_lookup_min_x = 0;
    int _target_spatial_grid_direct_lookup_min_y = 0;
    int _target_spatial_grid_direct_lookup_min_z = 0;
    int _target_spatial_grid_direct_lookup_range_x = 0;
    int _target_spatial_grid_direct_lookup_range_y = 0;
    int _target_spatial_grid_direct_lookup_range_z = 0;
    bool _target_spatial_grid_direct_lookup_evaluated = false;
};

/// Reusable device storage for converting ICP correspondence stats into a 4x4 step transform.
class IcpStepTransformWorkspace
{
public:
    IcpStepTransformWorkspace() = default;

    /// Reserve the fixed-size input and result buffers used by the GPU step solver.
    void reserve();

    /// Reserve only the fixed-size result buffer used by device-stats step solvers.
    void reserveResult();

    /// Return the reusable step input storage pointer, or null before reserve().
    unsigned char* inputStorage() { return _input_storage.get(); }

    /// Return the reusable step result storage pointer, or null before reserve().
    unsigned char* resultStorage() { return _result_storage.get(); }

    /// Return the reusable pinned host step result storage pointer, or null before reserveResult().
    unsigned char* hostResultStorage() { return _host_result_storage.get(); }

    /// Return the currently reserved pinned host step result storage capacity, in bytes.
    std::size_t hostResultStorageCapacity() const { return _host_result_storage.size(); }

private:
    DeviceBuffer<unsigned char> _input_storage;
    DeviceBuffer<unsigned char> _result_storage;
    HostPinnedBuffer<unsigned char> _host_result_storage;
};

/// Compute nearest-neighbor ICP correspondences for PlaMatrix column-major Nx3 device point arrays.
/// The returned stats are copied to host after synchronizing the supplied stream. If
/// d_correspondence_indices is null, per-source correspondence indices are not written.
IcpCorrespondenceStats<float> computeIcpCorrespondenceStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream = 0);

/// Compute nearest-neighbor ICP stats using caller-owned reusable device workspace.
IcpCorrespondenceStats<float> computeIcpCorrespondenceStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    int* d_correspondence_indices,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

IcpCorrespondenceStats<double> computeIcpCorrespondenceStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    int* d_correspondence_indices,
    cudaStream_t stream = 0);

/// Compute nearest-neighbor ICP stats using caller-owned reusable device workspace.
IcpCorrespondenceStats<double> computeIcpCorrespondenceStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    int* d_correspondence_indices,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Compute only final residual metrics using caller-owned reusable device workspace.
IcpResidualStats<float> computeIcpResidualStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Compute only final residual metrics using caller-owned reusable device workspace.
/// Set assume_ordered_correspondences when source[i] should be compared only with target[i] instead of nearest search.
IcpResidualStats<float> computeIcpResidualStatsColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream,
    bool assume_ordered_correspondences);

/// Compute only final residual metrics using caller-owned reusable device workspace.
IcpResidualStats<double> computeIcpResidualStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Compute only final residual metrics using caller-owned reusable device workspace.
/// Set assume_ordered_correspondences when source[i] should be compared only with target[i] instead of nearest search.
IcpResidualStats<double> computeIcpResidualStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream,
    bool assume_ordered_correspondences);

/// Transform points into caller-owned output storage and compute final residual metrics in one GPU pass.
/// Throws if d_output_points aliases d_target_points because residual search must read the original target points.
IcpResidualStats<float> transformPointsAndComputeIcpResidualStatsColumnMajor(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Transform points into caller-owned output storage and compute final residual metrics in one GPU pass.
/// Throws if d_output_points aliases d_target_points because residual search must read the original target points.
IcpResidualStats<double> transformPointsAndComputeIcpResidualStatsColumnMajor(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Transform points and compute final residual metrics against a cached target spatial-grid snapshot.
IcpResidualStats<float> transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajor(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream = 0);

/// Transform points and compute final residual metrics against a cached target spatial-grid snapshot.
IcpResidualStats<double> transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajor(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream = 0);

/// Compute the 4x4 ICP step transform from correspondence stats and write it to device memory.
/// The returned delta is copied to host after synchronizing the supplied stream.
IcpStepTransformResult<float> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<float>& stats,
    float* d_step_transform,
    cudaStream_t stream = 0);

/// Compute the 4x4 ICP step transform from correspondence stats using caller-owned reusable workspace.
IcpStepTransformResult<float> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<float>& stats,
    float* d_step_transform,
    IcpStepTransformWorkspace& workspace,
    cudaStream_t stream = 0);

/// Compute the 4x4 ICP step transform from correspondence stats and write it to device memory.
/// The returned delta is copied to host after synchronizing the supplied stream.
IcpStepTransformResult<double> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<double>& stats,
    double* d_step_transform,
    cudaStream_t stream = 0);

/// Compute the 4x4 ICP step transform from correspondence stats using caller-owned reusable workspace.
IcpStepTransformResult<double> computeIcpStepTransformFromStats(
    const IcpCorrespondenceStats<double>& stats,
    double* d_step_transform,
    IcpStepTransformWorkspace& workspace,
    cudaStream_t stream = 0);

/// Compute the 4x4 ICP step transform from the device-side stats reduction in stats_workspace.
/// stats must be the host summary returned by the most recent stats call that used stats_workspace.
IcpStepTransformResult<float> computeIcpStepTransformFromDeviceStats(
    const IcpCorrespondenceStats<float>& stats,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream = 0);

/// Compute the 4x4 ICP step transform from the device-side stats reduction in stats_workspace.
/// stats must be the host summary returned by the most recent stats call that used stats_workspace.
IcpStepTransformResult<double> computeIcpStepTransformFromDeviceStats(
    const IcpCorrespondenceStats<double>& stats,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream = 0);

/// Compute ICP correspondence stats and the step transform from one device-side reduction sequence.
/// Copies the reduced stats and step delta back to host with a single stream synchronization.
IcpStatsAndStepTransformResult<float> computeIcpStatsAndStepTransformColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false);

/// Compute ICP correspondence stats and the step transform from one device-side reduction sequence.
/// Copies the reduced stats and step delta back to host with a single stream synchronization.
IcpStatsAndStepTransformResult<double> computeIcpStatsAndStepTransformColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    IcpStepTransformWorkspace& step_workspace,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false);

/// Compute compact stats and the step transform needed by the GPU ICP alignment loop.
/// Copies only the compact convergence/error summary and step delta back to host.
IcpAlignmentStepResult<float> computeIcpAlignmentStepColumnMajor(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false);

/// Compute compact stats and the step transform needed by the GPU ICP alignment loop.
/// Copies only the compact convergence/error summary and step delta back to host.
IcpAlignmentStepResult<double> computeIcpAlignmentStepColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false);

namespace detail
{

/// Return the byte count for one sorted target-coordinate column stored with Scalar precision.
template <typename Scalar>
inline std::size_t targetSpatialGridSortedCoordinateByteCount(int target_count)
{
    if (target_count <= 0)
    {
        return 0;
    }
    return static_cast<std::size_t>(target_count) * sizeof(Scalar);
}

/// Return true when the reusable sorted-coordinate storage is too small or has the wrong scalar width.
inline bool targetSpatialGridCoordinateStorageNeedsReserve(
    int current_point_capacity,
    std::size_t current_coordinate_value_bytes,
    int target_count,
    std::size_t requested_coordinate_value_bytes)
{
    return target_count > current_point_capacity ||
           current_coordinate_value_bytes != requested_coordinate_value_bytes;
}

/// Return true when exact pointwise stats can skip target-coordinate loads because source and target alias.
template <typename Scalar>
inline bool canUseSameBufferExactPointwiseStats(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    const int* d_correspondence_indices)
{
    return d_correspondence_indices == nullptr &&
           source_count == target_count &&
           d_source_points == d_target_points;
}

/// Return true when exact pointwise stats may be attempted before falling back to nearest-neighbor search.
template <typename Scalar>
inline bool canProbeExactPointwiseStats(
    const Scalar* d_source_points,
    int source_count,
    const Scalar* d_target_points,
    int target_count,
    Scalar max_correspondence_distance,
    const int* d_correspondence_indices,
    bool probe_exact_pointwise_on_finite_radius = false)
{
    if (d_correspondence_indices || source_count != target_count)
    {
        return false;
    }
    if (canUseSameBufferExactPointwiseStats(
            d_source_points,
            source_count,
            d_target_points,
            target_count,
            d_correspondence_indices))
    {
        return true;
    }

    const double max_dist = static_cast<double>(max_correspondence_distance);
    return probe_exact_pointwise_on_finite_radius || !std::isfinite(max_dist);
}

/// Return true when transformed stats may accept same-index target matches before nearest-neighbor search.
inline bool canProbeTransformedExactPointwiseStats(
    int source_count,
    const void* target_points,
    int target_count,
    const int* d_correspondence_indices)
{
    return d_correspondence_indices == nullptr &&
           target_points != nullptr &&
           source_count == target_count;
}

/// Compute the compact ICP alignment step using workspace already reserved for source_count.
/// The caller must call IcpCorrespondenceStatsWorkspace::reserveFloatAlignmentStep(source_count) first.
IcpAlignmentStepResult<float> computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false,
    bool probe_exact_pointwise_on_finite_radius = false);

/// Compute the compact ICP alignment step using workspace already reserved for source_count.
/// The caller must call IcpCorrespondenceStatsWorkspace::reserveDoubleAlignmentStep(source_count) first.
IcpAlignmentStepResult<double> computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false,
    bool probe_exact_pointwise_on_finite_radius = false);

/// Enqueue a small-target compact ICP alignment step without copying the result to host.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetAlignmentStepColumnMajorWithReservedWorkspace(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream = 0);

/// Enqueue a small-target compact ICP alignment step without copying the result to host.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetAlignmentStepColumnMajorWithReservedWorkspace(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream = 0);

/// Compute the compact ICP alignment step from source points transformed by d_source_transform.
/// The caller must call IcpCorrespondenceStatsWorkspace::reserveFloatAlignmentStep(source_count) first.
/// Set probe_transformed_exact_pointwise_on_cache_hit only when same-index transformed matches are expected often
/// enough to justify an extra O(source_count) exact preflight before reusing a cached target spatial grid.
IcpAlignmentStepResult<float> computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const float* d_source_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false,
    bool probe_transformed_exact_pointwise_on_cache_hit = false);

/// Compute the compact ICP alignment step from source points transformed by d_source_transform.
/// The caller must call IcpCorrespondenceStatsWorkspace::reserveDoubleAlignmentStep(source_count) first.
/// Set probe_transformed_exact_pointwise_on_cache_hit only when same-index transformed matches are expected often
/// enough to justify an extra O(source_count) exact preflight before reusing a cached target spatial grid.
IcpAlignmentStepResult<double> computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
    const double* d_source_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false,
    bool probe_transformed_exact_pointwise_on_cache_hit = false);

/// Compute a transformed compact ICP alignment step and write accumulated_transform = step * previous_accumulated.
/// The caller must call IcpCorrespondenceStatsWorkspace::reserveFloatAlignmentStep(source_count) first.
/// Set probe_transformed_exact_pointwise_on_cache_hit only when same-index transformed matches are expected often
/// enough to justify an extra O(source_count) exact preflight before reusing a cached target spatial grid.
IcpAlignmentStepResult<float> computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
    const float* d_source_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    const float* d_previous_accumulated_transform,
    float* d_accumulated_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false,
    bool probe_transformed_exact_pointwise_on_cache_hit = false);

/// Compute a transformed compact ICP alignment step and write accumulated_transform = step * previous_accumulated.
/// The caller must call IcpCorrespondenceStatsWorkspace::reserveDoubleAlignmentStep(source_count) first.
/// Set probe_transformed_exact_pointwise_on_cache_hit only when same-index transformed matches are expected often
/// enough to justify an extra O(source_count) exact preflight before reusing a cached target spatial grid.
IcpAlignmentStepResult<double> computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
    const double* d_source_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    const double* d_previous_accumulated_transform,
    double* d_accumulated_transform,
    cudaStream_t stream = 0,
    bool assume_ordered_correspondences = false,
    bool probe_transformed_exact_pointwise_on_cache_hit = false);

/// Enqueue a small-target transformed alignment step and write accumulated_transform = step * previous_accumulated.
/// The helper does not copy the compact alignment-step result to host or synchronize with the host.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchTransformedSmallTargetAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
    const float* d_source_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    const float* d_previous_accumulated_transform,
    float* d_accumulated_transform,
    cudaStream_t stream = 0);

/// Enqueue a small-target transformed alignment step and write accumulated_transform = step * previous_accumulated.
/// The helper does not copy the compact alignment-step result to host or synchronize with the host.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchTransformedSmallTargetAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
    const double* d_source_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    const double* d_previous_accumulated_transform,
    double* d_accumulated_transform,
    cudaStream_t stream = 0);

/// Copy the compact alignment-step result produced by an async alignment-step helper and synchronize the stream.
template <typename Scalar>
IcpAlignmentStepResult<Scalar> copyAlignmentStepResultFromReservedWorkspace(
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    cudaStream_t stream = 0);

/// Enqueue two small-target alignment steps on one stream without copying either result to host.
/// The second step consumes the first step transform only when the first compact result is acceptable to ICP.
/// The two workspaces must be distinct because both compact device results are copied together later.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetTwoStepAlignmentColumnMajorWithReservedWorkspaces(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& first_step_workspace,
    IcpCorrespondenceStatsWorkspace& second_step_workspace,
    float* d_first_step_transform,
    float* d_second_step_transform,
    float* d_accumulated_transform,
    cudaStream_t stream = 0);

/// Enqueue two small-target alignment steps on one stream without copying either result to host.
/// The second step consumes the first step transform only when the first compact result is acceptable to ICP.
/// The two workspaces must be distinct because both compact device results are copied together later.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetTwoStepAlignmentColumnMajorWithReservedWorkspaces(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& first_step_workspace,
    IcpCorrespondenceStatsWorkspace& second_step_workspace,
    double* d_first_step_transform,
    double* d_second_step_transform,
    double* d_accumulated_transform,
    cudaStream_t stream = 0);

/// Copy both compact alignment-step results produced by the two-step async helper and synchronize the stream once.
template <typename Scalar>
IcpSmallTargetTwoStepAlignmentResult<Scalar>
copySmallTargetTwoStepAlignmentResultFromReservedWorkspaces(
    IcpCorrespondenceStatsWorkspace& first_step_workspace,
    IcpCorrespondenceStatsWorkspace& second_step_workspace,
    cudaStream_t stream = 0);

/// Enqueue an initial small-target alignment step and exact post-step residual metrics.
/// The helper writes d_step_transform as the final one-step transform and does not synchronize with the host.
/// If d_output_points is not null, the helper also writes the final transformed source points.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetSingleStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    cudaStream_t stream = 0,
    float* d_output_points = nullptr);

/// Enqueue an initial small-target alignment step and exact post-step residual metrics.
/// The helper writes d_step_transform as the final one-step transform and does not synchronize with the host.
/// If d_output_points is not null, the helper also writes the final transformed source points.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetSingleStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    cudaStream_t stream = 0,
    double* d_output_points = nullptr);

/// Enqueue a small-target terminal transformed alignment step and exact post-step residual metrics.
/// The helper writes accumulated_transform = step * previous_accumulated and does not synchronize with the host.
/// If d_output_points is not null, the helper also writes the final transformed source points.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
    const float* d_source_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    const float* d_previous_accumulated_transform,
    float* d_accumulated_transform,
    cudaStream_t stream = 0,
    float* d_output_points = nullptr);

/// Enqueue a small-target terminal transformed alignment step and exact post-step residual metrics.
/// The helper writes accumulated_transform = step * previous_accumulated and does not synchronize with the host.
/// If d_output_points is not null, the helper also writes the final transformed source points.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
    const double* d_source_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    const double* d_previous_accumulated_transform,
    double* d_accumulated_transform,
    cudaStream_t stream = 0,
    double* d_output_points = nullptr);

/// Copy the result produced by launchTransformedSmallTargetTerminalAlignmentAndResidual... and synchronize the stream.
template <typename Scalar>
IcpTerminalAlignmentAndResidualResult<Scalar>
copySmallTargetTerminalAlignmentAndResidualResultFromReservedWorkspace(
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    cudaStream_t stream = 0);

/// Enqueue an initial small-target alignment step followed by terminal transformed metrics on the same stream.
/// The two workspaces must be distinct because both compact device results are copied together later.
/// The terminal kernel writes an empty terminal result when the queued first step is not acceptable to ICP.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetTwoStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspaces(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& first_step_workspace,
    IcpCorrespondenceStatsWorkspace& terminal_workspace,
    float* d_first_step_transform,
    float* d_terminal_step_transform,
    float* d_accumulated_transform,
    cudaStream_t stream = 0,
    float* d_output_points = nullptr);

/// Enqueue an initial small-target alignment step followed by terminal transformed metrics on the same stream.
/// The two workspaces must be distinct because both compact device results are copied together later.
/// The terminal kernel writes an empty terminal result when the queued first step is not acceptable to ICP.
/// It returns false when the source/target sizes or correspondence radius are outside the small-target path.
bool launchSmallTargetTwoStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspaces(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& first_step_workspace,
    IcpCorrespondenceStatsWorkspace& terminal_workspace,
    double* d_first_step_transform,
    double* d_terminal_step_transform,
    double* d_accumulated_transform,
    cudaStream_t stream = 0,
    double* d_output_points = nullptr);

/// Copy both compact results produced by the two-step async helper and synchronize the stream once.
template <typename Scalar>
IcpSmallTargetTwoStepTerminalAlignmentAndResidualResult<Scalar>
copySmallTargetTwoStepTerminalAlignmentAndResidualResultFromReservedWorkspaces(
    IcpCorrespondenceStatsWorkspace& first_step_workspace,
    IcpCorrespondenceStatsWorkspace& terminal_workspace,
    cudaStream_t stream = 0);

/// Compute a small-target terminal transformed alignment step and exact post-step residual metrics.
/// The helper writes accumulated_transform = step * previous_accumulated and copies one compact result to host.
/// If d_output_points is not null, the helper also writes the final transformed source points.
/// It returns launched=false when the source/target sizes or correspondence radius are outside the small-target path.
IcpTerminalAlignmentAndResidualResult<float>
computeTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
    const float* d_source_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    float* d_step_transform,
    const float* d_previous_accumulated_transform,
    float* d_accumulated_transform,
    cudaStream_t stream = 0,
    float* d_output_points = nullptr);

/// Compute a small-target terminal transformed alignment step and exact post-step residual metrics.
/// The helper writes accumulated_transform = step * previous_accumulated and copies one compact result to host.
/// If d_output_points is not null, the helper also writes the final transformed source points.
/// It returns launched=false when the source/target sizes or correspondence radius are outside the small-target path.
IcpTerminalAlignmentAndResidualResult<double>
computeTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
    const double* d_source_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& stats_workspace,
    double* d_step_transform,
    const double* d_previous_accumulated_transform,
    double* d_accumulated_transform,
    cudaStream_t stream = 0,
    double* d_output_points = nullptr);

/// Compute final residual metrics using workspace already reserved for source_count.
/// The caller must reserve residual-compatible partial and result storage first.
IcpResidualStats<float> computeIcpResidualStatsColumnMajorWithReservedWorkspace(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Compute final residual metrics using workspace already reserved for source_count.
/// The caller must reserve residual-compatible partial and result storage first.
/// Set assume_ordered_correspondences when source[i] should be compared only with target[i] instead of nearest search.
IcpResidualStats<float> computeIcpResidualStatsColumnMajorWithReservedWorkspace(
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream,
    bool assume_ordered_correspondences);

/// Compute final residual metrics using workspace already reserved for source_count.
/// The caller must reserve residual-compatible partial and result storage first.
IcpResidualStats<double> computeIcpResidualStatsColumnMajorWithReservedWorkspace(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Compute final residual metrics using workspace already reserved for source_count.
/// The caller must reserve residual-compatible partial and result storage first.
/// Set assume_ordered_correspondences when source[i] should be compared only with target[i] instead of nearest search.
IcpResidualStats<double> computeIcpResidualStatsColumnMajorWithReservedWorkspace(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream,
    bool assume_ordered_correspondences);

/// Transform points and compute final residual metrics using workspace already reserved for source_count.
/// The caller must reserve residual-compatible partial and result storage first.
/// Throws if d_output_points aliases d_target_points because residual search must read the original target points.
/// d_output_points may be null to compute metrics without materializing transformed points.
IcpResidualStats<float> transformPointsAndComputeIcpResidualStatsColumnMajorWithReservedWorkspace(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Transform points and compute final residual metrics using workspace already reserved for source_count.
/// The caller must reserve residual-compatible partial and result storage first.
/// Throws if d_output_points aliases d_target_points because residual search must read the original target points.
/// d_output_points may be null to compute metrics without materializing transformed points.
IcpResidualStats<double> transformPointsAndComputeIcpResidualStatsColumnMajorWithReservedWorkspace(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Transform points and compute final residual metrics against same-index target points using reserved workspace.
/// The caller must reserve residual-compatible partial and result storage first, and source_count must equal
/// target_count.
/// d_output_points may alias d_target_points; each source row reads its same-index target before writing output.
/// d_output_points may be null to compute metrics without materializing transformed points.
IcpResidualStats<float> transformPointsAndComputeOrderedIcpResidualStatsColumnMajorWithReservedWorkspace(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    const float* d_target_points,
    int target_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Transform points and compute final residual metrics against same-index target points using reserved workspace.
/// The caller must reserve residual-compatible partial and result storage first, and source_count must equal
/// target_count.
/// d_output_points may alias d_target_points; each source row reads its same-index target before writing output.
/// d_output_points may be null to compute metrics without materializing transformed points.
IcpResidualStats<double> transformPointsAndComputeOrderedIcpResidualStatsColumnMajorWithReservedWorkspace(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Transform points and compute final residual metrics against a cached target spatial-grid snapshot using reserved
/// residual-compatible partial and result storage.
/// d_output_points may be null to compute metrics without materializing transformed points.
IcpResidualStats<float>
transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorWithReservedWorkspace(
    const float* d_transform,
    const float* d_source_points,
    int source_count,
    float max_correspondence_distance,
    float* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream = 0);

/// Transform points and compute final residual metrics against a cached target spatial-grid snapshot using reserved
/// residual-compatible partial and result storage.
/// d_output_points may be null to compute metrics without materializing transformed points.
IcpResidualStats<double>
transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorWithReservedWorkspace(
    const double* d_transform,
    const double* d_source_points,
    int source_count,
    double max_correspondence_distance,
    double* d_output_points,
    IcpCorrespondenceStatsWorkspace& workspace,
    int target_spatial_grid_cell_count,
    cudaStream_t stream = 0);

} // namespace detail

/// Multiply two 4x4 column-major device transforms and write C = A * B.
/// Throws if any device pointer is null or the CUDA launch fails.
void multiplyTransform4x4(
    const float* d_A,
    const float* d_B,
    float* d_C,
    cudaStream_t stream = 0);

/// Multiply two 4x4 column-major device transforms and write C = A * B.
/// Throws if any device pointer is null or the CUDA launch fails.
void multiplyTransform4x4(
    const double* d_A,
    const double* d_B,
    double* d_C,
    cudaStream_t stream = 0);

/// Async 4x4 column-major device transform multiplication on the caller-provided CUDA stream.
void multiplyTransform4x4Async(
    const float* d_A,
    const float* d_B,
    float* d_C,
    cudaStream_t stream);

/// Async 4x4 column-major device transform multiplication on the caller-provided CUDA stream.
void multiplyTransform4x4Async(
    const double* d_A,
    const double* d_B,
    double* d_C,
    cudaStream_t stream);

/// Write a 4x4 column-major identity transform to device memory.
/// Synchronizes the supplied stream before returning.
void setIdentityTransform4x4(float* d_transform, cudaStream_t stream = 0);

/// Write a 4x4 column-major identity transform to device memory.
/// Synchronizes the supplied stream before returning.
void setIdentityTransform4x4(double* d_transform, cudaStream_t stream = 0);

/// Async 4x4 column-major identity transform write on the caller-provided CUDA stream.
void setIdentityTransform4x4Async(float* d_transform, cudaStream_t stream);

/// Async 4x4 column-major identity transform write on the caller-provided CUDA stream.
void setIdentityTransform4x4Async(double* d_transform, cudaStream_t stream);

/// Apply a 4x4 column-major device transform to an Nx3 column-major device point array.
/// d_output_points must have capacity for point_count x 3 Scalars. In-place output is allowed.
/// Synchronizes the supplied stream before returning.
void transformPointsColumnMajor(
    const float* d_transform,
    const float* d_points,
    int point_count,
    float* d_output_points,
    cudaStream_t stream = 0);

/// Apply a 4x4 column-major device transform to an Nx3 column-major device point array.
/// d_output_points must have capacity for point_count x 3 Scalars. In-place output is allowed.
/// Synchronizes the supplied stream before returning.
void transformPointsColumnMajor(
    const double* d_transform,
    const double* d_points,
    int point_count,
    double* d_output_points,
    cudaStream_t stream = 0);

/// Async column-major point transform on the caller-provided CUDA stream.
/// The caller owns stream synchronization before reading output buffers.
void transformPointsColumnMajorAsync(
    const float* d_transform,
    const float* d_points,
    int point_count,
    float* d_output_points,
    cudaStream_t stream);

/// Async column-major point transform on the caller-provided CUDA stream.
/// The caller owns stream synchronization before reading output buffers.
void transformPointsColumnMajorAsync(
    const double* d_transform,
    const double* d_points,
    int point_count,
    double* d_output_points,
    cudaStream_t stream);

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
