#include "icp_gpu_path_test_support.h"

#ifdef PLAPOINT_WITH_CUDA

TEST(ICPGpuPathTest, AlignUsesReservedWorkspaceForTerminalResidualStats)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsReserveCheckCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsReserveCheckCountForTesting(), 0);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignReusesSpatialGridSnapshotForTerminalResidualStatsWithRegularOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 2);
    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignSkipsInitialIdentityTransformWriteForNonIdentityFirstStep)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpIdentityTransformWriteCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpIdentityTransformWriteCountForTesting(), 0);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignChecksTwoStepAlignmentWorkspacesOnceBeforeReturn)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepReserveCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepReserveCheckCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepReserveCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepReserveCheckCountForTesting(), 2);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignmentStepWorkspaceReservationCacheMatchesReservedCapacity)
{
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;

    EXPECT_FALSE(icp.gpuAlignmentStepWorkspaceReservationMatches(0));
    EXPECT_FALSE(icp.gpuAlignmentStepWorkspaceReservationMatches(4));

    icp._gpu_alignment_step_workspace_source_capacity = 4;

    EXPECT_TRUE(icp.gpuAlignmentStepWorkspaceReservationMatches(3));
    EXPECT_TRUE(icp.gpuAlignmentStepWorkspaceReservationMatches(4));
    EXPECT_FALSE(icp.gpuAlignmentStepWorkspaceReservationMatches(5));
}

TEST(ICPGpuPathTest, ReserveGpuWorkspacePreallocatesLargeTargetAlignmentState)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeTranslatedGridPoints(
        kMinTargetSpatialGridRowsForTesting,
        0.003f,
        -0.002f,
        0.001f);
    auto target_points = makeGridPoints(kMinTargetSpatialGridRowsForTesting);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);

    plapoint::gpu::resetIcpTargetSpatialGridReserveCountForTesting();
    icp.reserveGpuWorkspace();
    auto* first_step = icp._gpu_T_step->data();
    auto* first_acc = icp._gpu_T_acc->data();
    auto* first_next = icp._gpu_next_T_acc->data();

    EXPECT_TRUE(icp.gpuAlignmentStepWorkspaceReservationMatches(static_cast<int>(source->size())));
    EXPECT_GT(icp._gpu_terminal_stats_workspace.partialCapacity(), 0);
    EXPECT_GT(icp._gpu_final_stats_workspace.partialCapacity(), 0);
    EXPECT_GE(icp._gpu_stats_workspace.targetSpatialGridCapacity(), static_cast<int>(target->size()));
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridReserveCountForTesting(), 1);

    icp.reserveGpuWorkspace();

    EXPECT_EQ(icp._gpu_T_step->data(), first_step);
    EXPECT_EQ(icp._gpu_T_acc->data(), first_acc);
    EXPECT_EQ(icp._gpu_next_T_acc->data(), first_next);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridReserveCountForTesting(), 1);
}

TEST(ICPGpuPathTest, ReserveGpuWorkspaceSkipsFinalStatsWhenFinalMetricsAreDisabled)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeTranslatedGridPoints(
        kMinTargetSpatialGridRowsForTesting,
        0.003f,
        -0.002f,
        0.001f);
    auto target_points = makeGridPoints(kMinTargetSpatialGridRowsForTesting);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setComputeFinalMetrics(false);

    icp.reserveGpuWorkspace();

    EXPECT_TRUE(icp.gpuAlignmentStepWorkspaceReservationMatches(static_cast<int>(source->size())));
    EXPECT_GT(icp._gpu_terminal_stats_workspace.partialCapacity(), 0);
    EXPECT_EQ(icp._gpu_final_stats_workspace.partialCapacity(), 0);
    EXPECT_GE(icp._gpu_stats_workspace.targetSpatialGridCapacity(), static_cast<int>(target->size()));
}

TEST(ICPGpuPathTest, ReserveGpuWorkspaceSkipsTerminalStatsForSingleIteration)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeTranslatedGridPoints(
        kMinTargetSpatialGridRowsForTesting,
        0.003f,
        -0.002f,
        0.001f);
    auto target_points = makeGridPoints(kMinTargetSpatialGridRowsForTesting);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    icp.reserveGpuWorkspace();

    EXPECT_TRUE(icp.gpuAlignmentStepWorkspaceReservationMatches(static_cast<int>(source->size())));
    EXPECT_EQ(icp._gpu_terminal_stats_workspace.partialCapacity(), 0);
    EXPECT_GT(icp._gpu_final_stats_workspace.partialCapacity(), 0);
    EXPECT_NE(icp._gpu_T_step, nullptr);
    EXPECT_EQ(icp._gpu_T_acc, nullptr);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
    EXPECT_GE(icp._gpu_stats_workspace.targetSpatialGridCapacity(), static_cast<int>(target->size()));
}

TEST(ICPGpuPathTest, AlignReusesAlignmentStepWorkspaceAcrossRepeatedCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpAlignmentStepReserveCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepReserveCheckCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);
    GpuCloud second_output;
    icp.align(second_output);

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepReserveCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepReserveCheckCountForTesting(), 1);
    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignChecksStepTransformBufferOnceBeforeLoop)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(icp._gpu_step_transform_reserve_check_count, 1);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, GpuTransformBuffersSkipRepeatedReserveChecks)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;

    icp.reserveGpuStepTransformBuffer();
    auto* first_step = icp._gpu_T_step->data();
    icp.reserveGpuStepTransformBuffer();
    EXPECT_EQ(icp._gpu_T_step->data(), first_step);
    EXPECT_EQ(icp._gpu_step_transform_reserve_check_count, 1);

    icp.reserveGpuAccumulatedTransformBuffer();
    auto* first_acc = icp._gpu_T_acc->data();
    icp.reserveGpuAccumulatedTransformBuffer();
    EXPECT_EQ(icp._gpu_T_acc->data(), first_acc);
    EXPECT_EQ(icp._gpu_accumulated_transform_reserve_check_count, 1);

    icp.reserveGpuNextTransformBuffer();
    auto* first_next = icp._gpu_next_T_acc->data();
    icp.reserveGpuNextTransformBuffer();
    EXPECT_EQ(icp._gpu_next_T_acc->data(), first_next);
    EXPECT_EQ(icp._gpu_next_transform_reserve_check_count, 1);
}

TEST(ICPGpuPathTest, GpuPointScratchBufferSkipsRepeatedReserveCheckForSameShape)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;

    auto* first_a = icp.gpuPointScratchBuffer(4, true);
    auto* second_a = icp.gpuPointScratchBuffer(4, true);
    auto* first_b = icp.gpuPointScratchBuffer(4, false);
    auto* second_b = icp.gpuPointScratchBuffer(4, false);

    EXPECT_EQ(first_a, second_a);
    EXPECT_EQ(first_b, second_b);
    EXPECT_NE(first_a, first_b);
    EXPECT_EQ(icp._gpu_point_scratch_reserve_check_count, 2);

    static_cast<void>(icp.gpuPointScratchBuffer(5, true));
    EXPECT_EQ(icp._gpu_point_scratch_reserve_check_count, 3);

    auto* grown_a = icp._gpu_points_a->data();
    auto* smaller_a = icp.gpuPointScratchBuffer(3, true);
    EXPECT_EQ(smaller_a, grown_a);
    EXPECT_EQ(icp._gpu_point_scratch_reserve_check_count, 3);
}

TEST(ICPGpuPathTest, GpuPointScratchReservationCacheMatchesReservedCapacity)
{
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;

    EXPECT_FALSE(icp.gpuPointScratchBufferReservationMatches(0, 0));
    EXPECT_FALSE(icp.gpuPointScratchBufferReservationMatches(4, 0));
    EXPECT_TRUE(icp.gpuPointScratchBufferReservationMatches(3, 4));
    EXPECT_TRUE(icp.gpuPointScratchBufferReservationMatches(4, 4));
    EXPECT_FALSE(icp.gpuPointScratchBufferReservationMatches(5, 4));
}

TEST(ICPGpuPathTest, AlignSkipsNextTransformBufferAllocationForSingleIteration)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    GpuCloud output;
    icp.align(output);

    EXPECT_NE(icp._gpu_T_acc, nullptr);
    EXPECT_EQ(icp._gpu_T_step, nullptr);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(output.size(), source->size());

    const auto output_cpu = output.toCpu();
    ASSERT_EQ(output_cpu.points().rows(), target_cpu->points().rows());
    for (plamatrix::Index row = 0; row < target_cpu->points().rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < target_cpu->points().cols(); ++col)
        {
            EXPECT_NEAR(output_cpu.points().getValue(row, col), target_cpu->points().getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignWithoutOutputSkipsTerminalPointTransformWhenFinalMetricsAreDisabled)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    EXPECT_NEAR(final_transform.getValue(0, 3), 0.1f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(1, 3), -0.05f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(2, 3), 0.025f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignWithoutOutputKeepsAccumulatedTransformAcrossIterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    EXPECT_NEAR(final_transform.getValue(0, 3), 0.1f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(1, 3), -0.05f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(2, 3), 0.025f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationTransformOnlyAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationTransformOnlyWithOutputAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(5000));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    ASSERT_EQ(output.size(), source->size());
    const auto output_cpu = output.toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationTransformOnlyWithSameSizeOutputAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    ASSERT_EQ(output.size(), source->size());
    const auto output_cpu = output.toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationTransformOnlyWithTargetAliasAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align(*target);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    ASSERT_EQ(target->size(), source->size());
    const auto output_cpu = target->toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationTransformOnlyWithSourceAliasAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(5000));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* source_points_ptr = static_cast<const GpuCloud&>(*source).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align(*source);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(static_cast<const GpuCloud&>(*source).points().data(), source_points_ptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    ASSERT_EQ(source->size(), 4096);
    const auto output_cpu = source->toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    icp.align();
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    const auto& second_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(second_transform.getValue(row, col), expected, 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetThreeIterationTransformOnlyQueuesInitialTwoSteps)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(3);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(3);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetFourIterationTransformOnlyQueuesTwoStepBatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(4);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(4);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 4);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 4);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetSixIterationTransformOnlyQueuesTwoStepBatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(6);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(6);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 6);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 6);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetFiveIterationTransformOnlyQueuesTwoStepBatchesAndTerminalStep)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(5);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(5);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 5);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 5);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationFinalMetricsAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
}

TEST(ICPGpuPathTest, AlignLargeTargetThreeIterationFinalMetricsQueuesInitialTwoSteps)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(3);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(3);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
}

TEST(ICPGpuPathTest, AlignLargeTargetFourIterationFinalMetricsQueuesTwoStepBatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(4);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(4);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 4);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 4);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
}

TEST(ICPGpuPathTest, AlignLargeTargetFiveIterationFinalMetricsQueuesTwoStepBatchesAndTerminalStep)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(5);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(5);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 5);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 5);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
}

TEST(ICPGpuPathTest, AlignLargeTargetSixIterationFinalMetricsQueuesThreeTwoStepBatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(6);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(6);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 3);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 6);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 6);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
}

TEST(ICPGpuPathTest, AlignLargeTargetSevenIterationFinalMetricsQueuesThreeBatchesAndTerminalStep)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(7);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(7);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 4);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 7);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 7);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationFinalMetricsWithOutputAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(5000));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
    ASSERT_EQ(output.size(), source->size());
    const auto output_cpu = output.toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationFinalMetricsWithSameSizeOutputAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
    ASSERT_EQ(output.size(), source->size());
    const auto output_cpu = output.toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationFinalMetricsWithTargetAliasAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.02f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-12f);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());
    const float baseline_fitness = baseline_icp.getFitnessScore();
    const float baseline_rmse = baseline_icp.getFinalRmse();
    const bool baseline_converged = baseline_icp.hasConverged();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align(*target);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFitnessScore(), baseline_fitness, 1.0e-5f);
    EXPECT_NEAR(icp.getFinalRmse(), baseline_rmse, 1.0e-5f);
    EXPECT_EQ(icp.hasConverged(), baseline_converged);
    ASSERT_EQ(target->size(), source->size());
    const auto output_cpu = target->toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignLargeTargetTwoIterationFinalMetricsWithSourceAliasAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeTranslatedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(5000));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* source_points_ptr = static_cast<const GpuCloud&>(*source).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align(*source);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(static_cast<const GpuCloud&>(*source).points().data(), source_points_ptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
    ASSERT_EQ(source->size(), 4096);
    const auto expected_output = makeGridPoints(4096);
    const auto output_cpu = source->toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    icp.align();
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    const auto& second_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(second_transform.getValue(row, col), expected, 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignSmallTargetTwoIterationTransformOnlyAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.01f, -0.005f, 0.0025f);
    target_points.setValue(1, 0, target_points.getValue(1, 0) + 0.03f);
    target_points.setValue(2, 1, target_points.getValue(2, 1) - 0.02f);
    target_points.setValue(3, 2, target_points.getValue(3, 2) + 0.015f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.08f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-8f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto& baseline_transform = baseline_icp.getFinalTransformation();
    float baseline_values[16]{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            baseline_values[row * 4 + col] = baseline_transform.getValue(row, col);
        }
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    const auto& final_transform = icp.getFinalTransformation();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(final_transform.getValue(row, col), baseline_values[row * 4 + col], 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignSmallTargetTwoIterationTransformOnlyWithOutputAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.01f, -0.005f, 0.0025f);
    target_points.setValue(1, 0, target_points.getValue(1, 0) + 0.03f);
    target_points.setValue(2, 1, target_points.getValue(2, 1) - 0.02f);
    target_points.setValue(3, 2, target_points.getValue(3, 2) + 0.015f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.08f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-8f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    ASSERT_EQ(output.size(), source->size());
    const auto output_cpu = output.toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignSmallTargetTwoIterationTransformOnlyWithTargetAliasAvoidsPerIterationHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.01f, -0.005f, 0.0025f);
    target_points.setValue(1, 0, target_points.getValue(1, 0) + 0.03f);
    target_points.setValue(2, 1, target_points.getValue(2, 1) - 0.02f);
    target_points.setValue(3, 2, target_points.getValue(3, 2) + 0.015f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> baseline_icp;
    baseline_icp.setInputSource(source);
    baseline_icp.setInputTarget(target);
    baseline_icp.setMaxCorrespondenceDistance(0.08f);
    baseline_icp.setMaxIterations(2);
    baseline_icp.setTransformationEpsilon(1.0e-8f);
    baseline_icp.setComputeFinalMetrics(false);
    baseline_icp.align();
    const auto expected_output = plamatrix::transformPoints(
        baseline_icp.getFinalTransformation(),
        source_cpu->points());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align(*target);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    ASSERT_EQ(target->size(), source->size());
    const auto output_cpu = target->toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), expected_output.rows());
    ASSERT_EQ(output_points.cols(), expected_output.cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.getValue(row, col), expected_output.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignWithoutOutputOrderedFinalMetricsSkipsScratchPointBuffer)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), nullptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignWithoutOutputFinalMetricsSkipsScratchPointBuffer)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), nullptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignComputesStepFromDeviceStatsWithoutHostInputCopy)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto target_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpStepTransformInputCopyCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpStepTransformInputCopyCountForTesting(), 0);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignFusesStatsAndStepToAvoidExtraHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> transform(4, 4);
    transform.fill(0.0f);
    transform.setValue(0, 0, 1.0f);
    transform.setValue(1, 1, 1.0f);
    transform.setValue(2, 2, 1.0f);
    transform.setValue(3, 3, 1.0f);
    transform.setValue(0, 3, 0.2f);
    transform.setValue(1, 3, -0.1f);
    transform.setValue(2, 3, 0.05f);
    auto target_points = plamatrix::transformPoints(transform, source_points);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setTransformationEpsilon(1.0e-8f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpRawStatsStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpRawStatsStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(output.size(), source->size());
}

#endif // PLAPOINT_WITH_CUDA
