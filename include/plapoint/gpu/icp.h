#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <plapoint/gpu/cuda_check.h>

#include <cuda_runtime.h>

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

/// Reusable device storage for ICP correspondence stats reductions.
/// Reserve it once for a source size and pass it to repeated stats calls to avoid repeated allocations.
class IcpCorrespondenceStatsWorkspace
{
public:
    IcpCorrespondenceStatsWorkspace() = default;

    /// Reserve enough storage for the supplied source point count. Throws for negative counts.
    void reserve(int source_count);

    /// Reserve reusable target tile bounds storage for finite-radius pruning.
    void reserveTargetTileBounds(int target_count);

    /// Reserve reusable target spatial grid storage for finite-radius candidate search.
    void reserveTargetSpatialGrid(int target_count);

    /// Clear cached target spatial-grid metadata. Call this if target device contents mutate in place.
    void invalidateTargetSpatialGridCache();

    /// Return true when the cached target spatial grid matches the supplied target identity and cell size.
    bool targetSpatialGridCacheMatches(const void* target_points, int target_count, double cell_size) const;

    /// Mark the reusable target spatial grid storage as containing the supplied target grid.
    void markTargetSpatialGridCache(const void* target_points, int target_count, double cell_size, int cell_count);

    /// Return the currently reserved partial reduction capacity, in blocks.
    int partialCapacity() const { return _partial_capacity; }

    /// Return the currently reserved target tile bound capacity, in tiles.
    int targetTileBoundCapacity() const { return _target_tile_bound_capacity; }

    /// Return the currently reserved target spatial grid capacity, in points.
    int targetSpatialGridCapacity() const { return _target_spatial_grid_capacity; }

    /// Return the number of unique cells in the currently cached target spatial grid.
    int targetSpatialGridCellCount() const { return _target_spatial_grid_cell_count; }

    /// Return the reusable partial reduction storage pointer, or null before reserve().
    unsigned char* partialStorage() { return _partial_storage.get(); }

    /// Return the reusable final stats storage pointer, or null before reserve().
    unsigned char* statsStorage() { return _stats_storage.get(); }

    /// Return the reusable target tile bounds storage pointer, or null before reserveTargetTileBounds().
    unsigned char* targetTileBoundsStorage() { return _target_tile_bounds_storage.get(); }

    /// Return the reusable sorted target-grid key storage pointer.
    unsigned char* targetSpatialGridKeysStorage() { return _target_spatial_grid_keys_storage.get(); }

    /// Return the reusable unique target-grid cell key storage pointer.
    unsigned char* targetSpatialGridUniqueKeysStorage() { return _target_spatial_grid_unique_keys_storage.get(); }

    /// Return the reusable sorted target index storage pointer.
    unsigned char* targetSpatialGridIndicesStorage() { return _target_spatial_grid_indices_storage.get(); }

    /// Return the reusable target-grid cell start storage pointer.
    unsigned char* targetSpatialGridCellStartsStorage() { return _target_spatial_grid_cell_starts_storage.get(); }

    /// Return the reusable target-grid cell count storage pointer.
    unsigned char* targetSpatialGridCellCountsStorage() { return _target_spatial_grid_cell_counts_storage.get(); }

private:
    DeviceBuffer<unsigned char> _partial_storage;
    DeviceBuffer<unsigned char> _stats_storage;
    DeviceBuffer<unsigned char> _target_tile_bounds_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_keys_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_unique_keys_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_indices_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_cell_starts_storage;
    DeviceBuffer<unsigned char> _target_spatial_grid_cell_counts_storage;
    int _partial_capacity = 0;
    int _target_tile_bound_capacity = 0;
    int _target_spatial_grid_capacity = 0;
    int _target_spatial_grid_cell_count = 0;
    const void* _target_spatial_grid_points = nullptr;
    int _target_spatial_grid_point_count = 0;
    double _target_spatial_grid_cell_size = 0.0;
    bool _target_spatial_grid_cache_valid = false;
};

/// Reusable device storage for converting ICP correspondence stats into a 4x4 step transform.
class IcpStepTransformWorkspace
{
public:
    IcpStepTransformWorkspace() = default;

    /// Reserve the fixed-size input and result buffers used by the GPU step solver.
    void reserve();

    /// Return the reusable step input storage pointer, or null before reserve().
    unsigned char* inputStorage() { return _input_storage.get(); }

    /// Return the reusable step result storage pointer, or null before reserve().
    unsigned char* resultStorage() { return _result_storage.get(); }

private:
    DeviceBuffer<unsigned char> _input_storage;
    DeviceBuffer<unsigned char> _result_storage;
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
IcpResidualStats<double> computeIcpResidualStatsColumnMajor(
    const double* d_source_points,
    int source_count,
    const double* d_target_points,
    int target_count,
    double max_correspondence_distance,
    IcpCorrespondenceStatsWorkspace& workspace,
    cudaStream_t stream = 0);

/// Transform points into caller-owned output storage and compute final residual metrics in one GPU pass.
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
    cudaStream_t stream = 0);

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
    cudaStream_t stream = 0);

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
