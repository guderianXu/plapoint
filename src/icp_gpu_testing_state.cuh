#pragma once

// Internal implementation detail for icp_gpu.cu. This header is included only
// when PLAPOINT_ENABLE_TESTING is enabled and from the translation unit's
// anonymous namespace, keeping all instrumentation state local to that unit.

#define PLAPOINT_DEFINE_ICP_HOST_COUNTER(name, accessor_name) \
    std::atomic<int> g_icp_##name##_count{0};

PLAPOINT_DEFINE_ICP_HOST_COUNTER(correspondence_stats_call, CorrespondenceStatsCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(residual_stats_call, ResidualStatsCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(step_transform_input_copy, StepTransformInputCopy)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(exact_pointwise_step_call, ExactPointwiseStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    exact_pointwise_identity_step_kernel_launch,
    ExactPointwiseIdentityStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(same_buffer_identity_alignment_step, SameBufferIdentityAlignmentStep)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(transformed_identity_alignment_step, TransformedIdentityAlignmentStep)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    transformed_exact_pointwise_alignment_step_call,
    TransformedExactPointwiseAlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(raw_stats_step_kernel_launch, RawStatsStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(small_alignment_step_kernel_launch, SmallAlignmentStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(small_stats_step_kernel_launch, SmallStatsStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(small_residual_stats_kernel_launch, SmallResidualStatsKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    small_terminal_alignment_residual_kernel_launch,
    SmallTerminalAlignmentResidualKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(stats_step_host_result_copy, StatsStepHostResultCopy)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(alignment_step_host_result_copy, AlignmentStepHostResultCopy)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(alignment_step_call, AlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(transformed_alignment_step_call, TransformedAlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(accumulated_alignment_step_call, AccumulatedAlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    transformed_exact_pointwise_residual_call,
    TransformedExactPointwiseResidualCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(alignment_step_reserve, AlignmentStepReserve)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(alignment_step_reserve_check, AlignmentStepReserveCheck)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(residual_stats_reserve_check, ResidualStatsReserveCheck)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(host_synchronization, HostSynchronization)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(target_spatial_grid_build, TargetSpatialGridBuild)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    target_spatial_grid_key_init_kernel_launch,
    TargetSpatialGridKeyInitKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    target_spatial_grid_bounds_partial_kernel_launch,
    TargetSpatialGridBoundsPartialKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    target_spatial_grid_run_length_encode,
    TargetSpatialGridRunLengthEncode)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(fallback_tile_bound_kernel_launch, FallbackTileBoundKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(fallback_unbounded_kernel_launch, FallbackUnboundedKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(transform_points_call, TransformPointsCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(transform_multiply_call, TransformMultiplyCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(identity_transform_write, IdentityTransformWrite)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(target_spatial_grid_prepare, TargetSpatialGridPrepare)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(target_spatial_grid_reserve, TargetSpatialGridReserve)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(target_tile_bounds_reserve, TargetTileBoundsReserve)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(host_result_storage_allocation, HostResultStorageAllocation)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    direct_spatial_grid_kernel_launch,
    DirectSpatialGridKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER(
    direct_spatial_grid_target_point_bounds_fallback,
    DirectSpatialGridTargetPointBoundsFallback)

#undef PLAPOINT_DEFINE_ICP_HOST_COUNTER

std::atomic<std::uintptr_t> g_icp_first_stats_source_pointer{0};
std::atomic<std::uintptr_t> g_icp_last_transform_output_pointer{0};

#define PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(name, accessor_name) \
    __device__ unsigned long long g_icp_##name##_count;

PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(full_distance_evaluation, FullDistanceEvaluation)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(target_candidate_visit, TargetCandidateVisit)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(target_index_load, TargetIndexLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(sorted_target_coordinate_load, SortedTargetCoordinateLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(target_tile_bound_computation, TargetTileBoundComputation)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(target_tile_load, TargetTileLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(exact_pointwise_target_load, ExactPointwiseTargetLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(
    transformed_exact_pointwise_residual_probe,
    TransformedExactPointwiseResidualProbe)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(
    transform_residual_output_point_write,
    TransformResidualOutputPointWrite)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(
    transform_residual_point_transform,
    TransformResidualPointTransform)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(grid_cell_lookup, GridCellLookup)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(grid_cell_offset, GridCellOffset)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(grid_cell_center_min_distance, GridCellCenterMinDistance)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(grid_cell_neighbor_min_distance, GridCellNeighborMinDistance)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(grid_cell_neighbor_xy_distance, GridCellNeighborXyDistance)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(direct_grid_lookup_xy_check, DirectGridLookupXyCheck)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(direct_grid_lookup_active_guard, DirectGridLookupActiveGuard)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(direct_grid_lookup_xy_base_guard, DirectGridLookupXyBaseGuard)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(direct_grid_lookup_linear_guard, DirectGridLookupLinearGuard)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER(
    outer_lower_triangle_accumulation,
    OuterLowerTriangleAccumulation)

#undef PLAPOINT_DEFINE_ICP_DEVICE_COUNTER

void recordFallbackKernelLaunchForTesting(bool uses_target_tile_bounds)
{
    if (uses_target_tile_bounds)
    {
        g_icp_fallback_tile_bound_kernel_launch_count.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        g_icp_fallback_unbounded_kernel_launch_count.fetch_add(1, std::memory_order_relaxed);
    }
}

void recordTransformedExactPointwiseResidualCallForTesting(bool enabled)
{
    if (enabled)
    {
        g_icp_transformed_exact_pointwise_residual_call_count.fetch_add(1, std::memory_order_relaxed);
    }
}

void recordDirectSpatialGridKernelLaunchForTesting(bool enabled)
{
    if (enabled)
    {
        g_icp_direct_spatial_grid_kernel_launch_count.fetch_add(1, std::memory_order_relaxed);
    }
}
