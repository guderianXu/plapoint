#pragma once

// Internal implementation detail for icp_gpu.cu. The including translation
// unit supplies the counters from icp_gpu_testing_state.cuh and the raw ICP
// result types used by the byte-count accessors below.

#define PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(name, accessor_name)               \
    void resetIcp##accessor_name##CountForTesting()                                  \
    {                                                                                 \
        g_icp_##name##_count.store(0, std::memory_order_relaxed);                     \
    }                                                                                 \
                                                                                      \
    int icp##accessor_name##CountForTesting()                                         \
    {                                                                                 \
        return g_icp_##name##_count.load(std::memory_order_relaxed);                  \
    }

PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(correspondence_stats_call, CorrespondenceStatsCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(residual_stats_call, ResidualStatsCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(step_transform_input_copy, StepTransformInputCopy)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(exact_pointwise_step_call, ExactPointwiseStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    exact_pointwise_identity_step_kernel_launch,
    ExactPointwiseIdentityStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    same_buffer_identity_alignment_step,
    SameBufferIdentityAlignmentStep)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    transformed_identity_alignment_step,
    TransformedIdentityAlignmentStep)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    transformed_exact_pointwise_alignment_step_call,
    TransformedExactPointwiseAlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(raw_stats_step_kernel_launch, RawStatsStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    small_alignment_step_kernel_launch,
    SmallAlignmentStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(small_stats_step_kernel_launch, SmallStatsStepKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    small_residual_stats_kernel_launch,
    SmallResidualStatsKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    small_terminal_alignment_residual_kernel_launch,
    SmallTerminalAlignmentResidualKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(stats_step_host_result_copy, StatsStepHostResultCopy)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    alignment_step_host_result_copy,
    AlignmentStepHostResultCopy)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(alignment_step_call, AlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    transformed_alignment_step_call,
    TransformedAlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    accumulated_alignment_step_call,
    AccumulatedAlignmentStepCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    transformed_exact_pointwise_residual_call,
    TransformedExactPointwiseResidualCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(alignment_step_reserve, AlignmentStepReserve)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    alignment_step_reserve_check,
    AlignmentStepReserveCheck)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    residual_stats_reserve_check,
    ResidualStatsReserveCheck)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(host_synchronization, HostSynchronization)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(target_spatial_grid_build, TargetSpatialGridBuild)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    target_spatial_grid_key_init_kernel_launch,
    TargetSpatialGridKeyInitKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    target_spatial_grid_bounds_partial_kernel_launch,
    TargetSpatialGridBoundsPartialKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    target_spatial_grid_run_length_encode,
    TargetSpatialGridRunLengthEncode)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    fallback_tile_bound_kernel_launch,
    FallbackTileBoundKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    fallback_unbounded_kernel_launch,
    FallbackUnboundedKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(transform_points_call, TransformPointsCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(transform_multiply_call, TransformMultiplyCall)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(identity_transform_write, IdentityTransformWrite)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(target_spatial_grid_prepare, TargetSpatialGridPrepare)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(target_spatial_grid_reserve, TargetSpatialGridReserve)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(target_tile_bounds_reserve, TargetTileBoundsReserve)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    host_result_storage_allocation,
    HostResultStorageAllocation)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    direct_spatial_grid_kernel_launch,
    DirectSpatialGridKernelLaunch)
PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS(
    direct_spatial_grid_target_point_bounds_fallback,
    DirectSpatialGridTargetPointBoundsFallback)

#undef PLAPOINT_DEFINE_ICP_HOST_COUNTER_ACCESSORS

#define PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(name, accessor_name)             \
    void resetIcp##accessor_name##CountForTesting()                                  \
    {                                                                                 \
        const unsigned long long zero = 0;                                            \
        PLAPOINT_CHECK_CUDA(cudaMemcpyToSymbol(g_icp_##name##_count, &zero, sizeof(zero))); \
    }                                                                                 \
                                                                                      \
    unsigned long long icp##accessor_name##CountForTesting()                          \
    {                                                                                 \
        unsigned long long count = 0;                                                 \
        PLAPOINT_CHECK_CUDA(cudaMemcpyFromSymbol(                                     \
            &count, g_icp_##name##_count, sizeof(count)));                            \
        return count;                                                                 \
    }

PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(full_distance_evaluation, FullDistanceEvaluation)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(target_candidate_visit, TargetCandidateVisit)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(target_index_load, TargetIndexLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    sorted_target_coordinate_load,
    SortedTargetCoordinateLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    target_tile_bound_computation,
    TargetTileBoundComputation)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(target_tile_load, TargetTileLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(exact_pointwise_target_load, ExactPointwiseTargetLoad)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    transformed_exact_pointwise_residual_probe,
    TransformedExactPointwiseResidualProbe)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    transform_residual_output_point_write,
    TransformResidualOutputPointWrite)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    transform_residual_point_transform,
    TransformResidualPointTransform)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(grid_cell_lookup, GridCellLookup)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(grid_cell_offset, GridCellOffset)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    grid_cell_center_min_distance,
    GridCellCenterMinDistance)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    grid_cell_neighbor_min_distance,
    GridCellNeighborMinDistance)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    grid_cell_neighbor_xy_distance,
    GridCellNeighborXyDistance)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(direct_grid_lookup_xy_check, DirectGridLookupXyCheck)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    direct_grid_lookup_active_guard,
    DirectGridLookupActiveGuard)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    direct_grid_lookup_xy_base_guard,
    DirectGridLookupXyBaseGuard)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    direct_grid_lookup_linear_guard,
    DirectGridLookupLinearGuard)
PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS(
    outer_lower_triangle_accumulation,
    OuterLowerTriangleAccumulation)

#undef PLAPOINT_DEFINE_ICP_DEVICE_COUNTER_ACCESSORS

void resetIcpFirstStatsSourcePointerForTesting()
{
    g_icp_first_stats_source_pointer.store(0, std::memory_order_relaxed);
}

const void* icpFirstStatsSourcePointerForTesting()
{
    return reinterpret_cast<const void*>(
        g_icp_first_stats_source_pointer.load(std::memory_order_relaxed));
}

void resetIcpLastTransformOutputPointerForTesting()
{
    g_icp_last_transform_output_pointer.store(0, std::memory_order_relaxed);
}

const void* icpLastTransformOutputPointerForTesting()
{
    return reinterpret_cast<const void*>(
        g_icp_last_transform_output_pointer.load(std::memory_order_relaxed));
}

std::size_t icpAlignmentStepRawResultByteCountForTesting()
{
    return sizeof(IcpAlignmentStepRawResult<double>);
}

std::size_t icpRawStatsByteCountForTesting()
{
    return sizeof(RawIcpStats);
}

std::size_t icpFloatAlignmentStepRawResultByteCountForTesting()
{
    return sizeof(IcpAlignmentStepRawResult<float>);
}

std::size_t icpDoubleAlignmentStepRawResultByteCountForTesting()
{
    return sizeof(IcpAlignmentStepRawResult<double>);
}
