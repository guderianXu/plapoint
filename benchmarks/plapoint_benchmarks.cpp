#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/registration/icp.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/plamatrix.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/icp.h>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

#include "plapoint_benchmark_common.h"
#include "plapoint_benchmark_cpu.h"
#include "plapoint_benchmark_gpu.h"

} // namespace

int main(int argc, char** argv)
{
    Options options;
    try
    {
        options = parseOptions(argc, argv);
    }
    catch (const std::invalid_argument& error)
    {
        std::cerr << error.what() << '\n';
        return 2;
    }

    if (options.self_test_benchmark_gpu_sync)
    {
#ifdef PLAPOINT_WITH_CUDA
        return runBenchmarkGpuSyncSelfTest();
#else
        std::cout << "benchmark_gpu_sync_self_test,skipped,cuda_disabled\n";
        return 0;
#endif
    }

    std::cout << "benchmark,points,iterations,best_ms\n";
    benchmarkCpuKnn(options.points, options.iterations);
    benchmarkCpuVoxelGrid(options.points, options.iterations);
    benchmarkCpuNormalEstimation(options.points, options.iterations);
    if (options.skip_cpu_icp)
    {
        printSkipped("cpu_icp_identity", "disabled");
        printSkipped("cpu_icp_finite_radius", "disabled");
        printSkipped("cpu_icp_finite_radius_translation", "disabled");
        printSkipped("cpu_icp_finite_radius_translation_reuse", "disabled");
    }
    else
    {
        if (options.skip_icp_identity)
        {
            printSkipped("cpu_icp_identity", "disabled");
        }
        else
        {
            benchmarkCpuIcp(options.icp_points, options.icp_max_iterations, options.iterations);
        }
        benchmarkCpuIcpFiniteRadius(options.icp_points, options.icp_max_iterations, options.iterations);
        benchmarkCpuIcpFiniteRadiusTranslation(options.icp_points, options.icp_max_iterations, options.iterations);
        benchmarkCpuIcpFiniteRadiusTranslationReuse(
            options.icp_points,
            options.icp_max_iterations,
            options.iterations);
    }
#ifdef PLAPOINT_WITH_CUDA
    const ScopedCudaBenchmarkSynchronization scoped_gpu_sync(true);
    benchmarkGpuKnn(options.points, options.iterations);
    benchmarkGpuVoxelGrid(options.points, options.iterations);
    if (options.skip_icp_identity)
    {
        printSkipped("gpu_icp_identity", "disabled");
        printSkipped("gpu_icp_identity_same_buffer_reuse_output", "disabled");
    }
    else
    {
        benchmarkGpuIcp(options.icp_points, options.icp_max_iterations, options.iterations);
        benchmarkGpuIcpIdentitySameBufferReuseOutput(
            options.icp_points,
            options.icp_max_iterations,
            options.iterations);
    }
    benchmarkGpuIcpFiniteRadius(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusIdentityReuseOutput(
        "gpu_icp_finite_radius_identity_reuse_output",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusIdentityReuseOutput(
        "gpu_icp_finite_radius_identity_exact_probe_reuse_output",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslation(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuse(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseShrinking(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutput(
        "gpu_icp_finite_radius_translation_reuse_output",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutput(
        "gpu_icp_finite_radius_translation_reuse_output_one_iteration",
        options.icp_points,
        1,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationNoOutputOneIteration(options.icp_points, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationTargetOutputOneIteration(options.icp_points, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationSourceOutputOneIteration(options.icp_points, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutputSkipFinalMetrics(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutputSkipFinalMetricsOneIteration(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output_one_iteration",
        options.icp_points,
        1,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedTransformOnly(
        "gpu_icp_finite_radius_translation_ordered_transform_only",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedTransformOnly(
        "gpu_icp_finite_radius_translation_ordered_transform_only_one_iteration",
        options.icp_points,
        1,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output_skip_final_metrics",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output_skip_final_metrics_one_iteration",
        options.icp_points,
        1,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusOrderedLowResidualOutputOneIteration(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpOrderedInfiniteRadiusOutputOneIteration(options.icp_points, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedReuseTargetOutput(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_transform_only_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        false,
        false,
        false);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_transform_only_prewarmed_workspace_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        false,
        false,
        false,
        false,
        true);
    benchmarkGpuIcpFiniteRadiusNonRigidOutputTransformOnlyTwoIterations(
        "gpu_icp_finite_radius_nonrigid_output_transform_only_two_iterations",
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusNonRigidOutputTransformOnlyTwoIterations(
        "gpu_icp_finite_radius_nonrigid_same_size_output_transform_only_two_iterations",
        options.icp_points,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusNonRigidOutputTransformOnlyFreshTargetTwoIterations(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusNonRigidTargetAliasTransformOnlyTwoIterations(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_final_metrics_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        false,
        false,
        false,
        true);
    benchmarkGpuIcpFiniteRadiusNonRigidFinalMetricsOutput(
        "gpu_icp_finite_radius_nonrigid_output_final_metrics_two_iterations",
        options.icp_points,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusNonRigidFinalMetricsOutput(
        "gpu_icp_finite_radius_nonrigid_same_size_output_final_metrics_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        true);
    benchmarkGpuIcpFiniteRadiusNonRigidFinalMetricsOutput(
        "gpu_icp_finite_radius_nonrigid_target_alias_final_metrics_two_iterations",
        options.icp_points,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslationSourceAliasTransformOnlyTwoIterations(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationSourceAliasFinalMetricsTwoIterations(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_transform_only_preflight_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        true,
        false,
        false);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_transform_only_cache_reuse_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        false,
        true,
        false);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_verified_ordered_transform_only_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        false,
        false,
        true);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_verified_ordered_final_metrics_two_iterations",
        options.icp_points,
        options.iterations,
        false,
        false,
        false,
        true,
        true);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_ordered_transform_only_two_iterations",
        options.icp_points,
        options.iterations,
        true,
        false,
        false,
        false);
    benchmarkGpuIcpFiniteRadiusTranslationTransformOnlySkipFinalMetrics(
        "gpu_icp_finite_radius_translation_transform_only_skip_final_metrics",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationTransformOnlySkipFinalMetrics(
        "gpu_icp_finite_radius_translation_transform_only_skip_final_metrics_one_iteration",
        options.icp_points,
        1,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusBinaryTranslationTransformOnly(
        "gpu_icp_finite_radius_binary_translation_transform_only_two_iterations",
        options.icp_points,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusBinaryTranslationTransformOnly(
        "gpu_icp_finite_radius_binary_translation_transform_only_preflight_two_iterations",
        options.icp_points,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusBinaryTranslationReuseOutput(
        "gpu_icp_finite_radius_binary_translation_reuse_output_one_iteration",
        options.icp_points,
        1,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusBinaryTranslationReuseOutput(
        "gpu_icp_finite_radius_binary_translation_reuse_output_two_iterations",
        options.icp_points,
        2,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusBinaryTranslationReuseOutput(
        "gpu_icp_finite_radius_binary_translation_reuse_output_preflight_two_iterations",
        options.icp_points,
        2,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutput(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutputSkipFinalMetrics(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationOrdered(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationNewWorkspaceOneSource(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationRebuildReservedGrid(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationCachedGridReservedWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationAsyncLaunchCachedGrid(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepTransformedAccumulatedAsyncLaunchCachedGrid(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepTwoStepAsyncLaunchSeparateWorkspaces(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepExactPointwiseSameBuffer(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepExactPointwiseSameBufferReservedWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepOrderedSameBufferFiniteRadius(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCachedGrid(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCacheHit(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCacheHitPreflight(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationCachedGridReservedWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationOrdered(options.icp_points, options.iterations);
    benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpTransformResidualStatsTransformedExactPointwiseNewWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpStatsFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTarget(
        "gpu_icp_alignment_step_small_finite_radius_target_4",
        4,
        options.iterations);
    benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTarget(
        "gpu_icp_alignment_step_small_finite_radius_target_16",
        16,
        options.iterations);
    benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTarget(
        "gpu_icp_alignment_step_small_finite_radius_target_64",
        64,
        options.iterations);
    benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTarget(
        "gpu_icp_alignment_step_small_finite_radius_target_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTargetAsyncLaunch(
        "gpu_icp_alignment_step_small_finite_radius_target_async_launch_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTarget(
        "gpu_icp_alignment_step_small_finite_radius_target_at_grid_threshold",
        128,
        options.iterations);
    benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTarget(
        "gpu_icp_alignment_step_small_finite_radius_target_256",
        256,
        options.iterations);
    benchmarkGpuIcpStatsStepSmallFiniteRadiusTarget(
        "gpu_icp_stats_step_small_finite_radius_target_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpResidualStatsSmallFiniteRadiusTarget(
        "gpu_icp_residual_stats_small_finite_radius_target_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpTransformResidualStatsSmallFiniteRadiusTarget(
        "gpu_icp_transform_residual_stats_small_finite_radius_target_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusTransformOnlyTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_transform_only_two_iterations_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusOutputTransformOnlyTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_output_transform_only_two_iterations_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusTargetAliasTransformOnlyTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_target_alias_transform_only_two_iterations_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusFinalMetricsTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_final_metrics_one_iteration_below_grid_threshold",
        127,
        options.iterations,
        1);
    benchmarkGpuIcpSmallFiniteRadiusFinalMetricsTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_final_metrics_two_iterations_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusReuseFinalMetricsTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_reuse_final_metrics_two_iterations_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusOutputFinalMetricsTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_output_final_metrics_two_iterations_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusTargetAliasFinalMetricsTwoIterations(
        "gpu_icp_small_finite_radius_nonrigid_target_alias_final_metrics_two_iterations_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusTransformedAccumulatedAsyncLaunch(
        "gpu_icp_small_finite_radius_transformed_accumulated_async_launch_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusTwoStepTransformOnlyAsyncLaunch(
        "gpu_icp_small_finite_radius_two_step_transform_only_async_launch_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusTerminalAsyncLaunch(
        "gpu_icp_small_finite_radius_terminal_async_launch_below_grid_threshold",
        127,
        options.iterations);
    benchmarkGpuIcpSmallFiniteRadiusTwoStepTerminalAsyncLaunch(
        "gpu_icp_small_finite_radius_two_step_terminal_async_launch_below_grid_threshold",
        127,
        options.iterations);
#endif
    return 0;
}
