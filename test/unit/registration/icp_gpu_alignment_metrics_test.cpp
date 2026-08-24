#include "icp_gpu_path_test_support.h"

#ifdef PLAPOINT_WITH_CUDA

TEST(ICPGpuPathTest, AlignProbeTransformedExactCacheHitUsesTwoStepFastPath)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseResidualCallCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseResidualCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_NE(icp._gpu_next_T_acc, nullptr);
    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
}

TEST(ICPGpuPathTest, AlignWritesPostLoopOutputTransformWithoutExtraHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.5f, -0.25f, 0.125f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(output.size(), source->size());
    EXPECT_TRUE(icp.hasConverged());
}

TEST(ICPGpuPathTest, AlignDeferredLastTransformedStepAccumulatesNonIdentityBeforeFinalMetrics)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting + 1);
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.2f, -0.1f, 0.05f);
    target_points.setValue(1, 0, target_points.getValue(1, 0) + 0.025f);
    target_points.setValue(2, 1, target_points.getValue(2, 1) - 0.015f);
    target_points.setValue(3, 2, target_points.getValue(3, 2) + 0.01f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);

    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpAccumulatedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAccumulatedAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 1);
    EXPECT_NE(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(output.size(), source->size());
    EXPECT_TRUE(std::isfinite(icp.getFinalRmse()));
}

TEST(ICPGpuPathTest, AlignOrderedFinalMetricsSkipTargetSpatialGridSearch)
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
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignReusesOrderedFiniteRadiusStepWhenResidualSumFitsRadius)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    target_points.setValue(1, 0, target_points.getValue(1, 0) + 0.025f);
    target_points.setValue(2, 1, target_points.getValue(2, 1) - 0.015f);
    target_points.setValue(3, 2, target_points.getValue(3, 2) + 0.01f);

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

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_LT(icp.getFinalRmse(), 2.0f);
    EXPECT_TRUE(std::isfinite(icp.getFinalRmse()));
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignKeepsOrderedFiniteRadiusResidualStatsWhenResidualSumExceedsRadius)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.01f, -0.005f, 0.0025f);
    target_points.setValue(0, 0, target_points.getValue(0, 0) + 0.03f);
    target_points.setValue(1, 0, target_points.getValue(1, 0) - 0.03f);
    target_points.setValue(2, 1, target_points.getValue(2, 1) + 0.03f);
    target_points.setValue(3, 2, target_points.getValue(3, 2) - 0.03f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.05f);
    icp.setMaxIterations(1);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_TRUE(std::isfinite(icp.getFinalRmse()));
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignReusesOrderedInfiniteRadiusStepForTerminalMetrics)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    target_points.setValue(1, 0, target_points.getValue(1, 0) + 0.025f);
    target_points.setValue(2, 1, target_points.getValue(2, 1) - 0.015f);
    target_points.setValue(3, 2, target_points.getValue(3, 2) + 0.01f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_TRUE(std::isfinite(icp.getFinalRmse()));
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignReusesIterationStatsForExactIdentityTerminalMetrics)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_output_device_to_device_copy_sync_count, 0);
    EXPECT_EQ(icp._gpu_output_device_to_device_copy_async_count, 1);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignSkipsRepeatedIdentityOutputCopyWhenOutputIsUnchanged)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_output_device_to_device_copy_sync_count, 0);
    EXPECT_EQ(icp._gpu_output_device_to_device_copy_async_count, 1);
}

TEST(ICPGpuPathTest, AlignReusesSameBufferIdentityResultAcrossRepeatedCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto cloud_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto cloud = std::make_shared<GpuCloud>(cloud_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), cloud->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(icp._gpu_output_device_to_device_copy_async_count, 1);
}

TEST(ICPGpuPathTest, AlignRecomputesSameBufferIdentityAfterMutableSourceAccess)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto cloud_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto cloud = std::make_shared<GpuCloud>(cloud_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    (void)cloud->points();
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), cloud->size());
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignReusesExactIdentityResultAcrossSeparateBufferCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(icp._gpu_output_device_to_device_copy_async_count, 1);
}

TEST(ICPGpuPathTest, AlignRecomputesExactIdentityAfterMutableTargetAccess)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    GpuCloud output;
    icp.align(output);
    (void)target->points();
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignReusesExactNonIdentityStepForTerminalMetrics)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignReusesExactNonIdentityResultAcrossSeparateBufferCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignRecomputesExactNonIdentityAfterMutableSourceAccess)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    (void)source->points();
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignRecomputesExactNonIdentityAfterMutableTargetAccess)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    (void)target->points();
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignReusesExactNonIdentityResultAfterMutableOutputAccess)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    (void)output.points();
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignLargeTargetSingleIterationFinalMetricsAvoidsExtraHostSynchronization)
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

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

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
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignLargeTargetSingleIterationFinalMetricsWithTargetAliasAvoidsExtraHostSynchronization)
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

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

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
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
    ASSERT_EQ(target->size(), source->size());
    const auto expected_output = makeGridPoints(4096);
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

TEST(ICPGpuPathTest, AlignLargeTargetSingleIterationFinalMetricsWithSourceAliasAvoidsExtraHostSynchronization)
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

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align(*source);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
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
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
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

TEST(ICPGpuPathTest, AlignReusesFullCoverageGridTranslationResultAcrossSeparateBufferCalls)
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

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    const auto expected_output = makeGridPoints(4096);
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
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignReusesFullCoverageSkipFinalMetricsOneIterationResultAcrossSeparateBufferCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeTranslatedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignReusesFullCoverageSkipFinalMetricsTransformedIdentityResultAcrossSeparateBufferCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeTranslatedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignDoesNotReuseSkipFinalMetricsForNonRigidSameIndexResiduals)
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
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();
    icp.align();

    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 4);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignCanReuseSkipFinalMetricsForNonRigidSameIndexResidualsWhenEnabled)
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
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuCacheFullCoverageResidualResults(true);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();
    icp.align();

    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignCanUseOrderedCorrespondencesAfterSameIndexStepWhenEnabled)
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
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuAssumeOrderedCorrespondencesAfterSameIndexStep(true);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();

    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignUsesVerifiedOrderedResidualStatsForTerminalMetrics)
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
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setGpuAssumeOrderedCorrespondencesAfterSameIndexStep(true);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    icp.align();

    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignRecomputesCachedResidualResultAfterMutableTargetAccessWhenEnabled)
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
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuCacheFullCoverageResidualResults(true);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align();
    (void)target->points();
    icp.align();

    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 4);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics)
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

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
    const auto& final_transform = icp.getFinalTransformation();
    EXPECT_NEAR(final_transform.getValue(0, 0), 1.0f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(1, 1), 1.0f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(2, 2), 1.0f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(0, 3), 0.1f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(1, 3), -0.05f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(2, 3), 0.025f, 1.0e-5f);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, TransformResidualStatsSkipsSearchForExactPointwiseMatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_points = makeNonCollinearPoints();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> transform(4, 4);
    transform.fill(0.0f);
    transform.setValue(0, 0, 1.0f);
    transform.setValue(1, 1, 1.0f);
    transform.setValue(2, 2, 1.0f);
    transform.setValue(3, 3, 1.0f);
    transform.setValue(0, 3, 0.5f);
    transform.setValue(1, 3, -0.25f);
    transform.setValue(2, 3, 0.125f);
    auto target_points = plamatrix::transformPoints(transform, source_points);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto transform_gpu = transform.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source_points.rows(), 3);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseResidualCallCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        transform_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        output_gpu.data(),
        workspace);

    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseResidualCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    EXPECT_EQ(stats.active_count, static_cast<int>(source_points.rows()));
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-8);
}

TEST(ICPGpuPathTest, TransformResidualStatsFallbackSkipsDuplicateExactPointwiseProbeAfterPreflightMiss)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(-0.25f, 0.125f, -0.0625f);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto transform_gpu = transform.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source_points.rows(), 3);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTransformedExactPointwiseResidualCallCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseResidualProbeCountForTesting();
    plapoint::gpu::resetIcpTransformResidualOutputPointWriteCountForTesting();
    plapoint::gpu::resetIcpTransformResidualPointTransformCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        transform_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        output_gpu.data(),
        workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(source_points.rows()));
    EXPECT_TRUE(std::isfinite(stats.residual_sq_sum));
    EXPECT_GT(stats.residual_sq_sum, 0.0);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseResidualCallCountForTesting(), 1);
    EXPECT_EQ(
        plapoint::gpu::icpTransformedExactPointwiseResidualProbeCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()));
    EXPECT_EQ(
        plapoint::gpu::icpExactPointwiseTargetLoadCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()) * 3ull);
    EXPECT_EQ(
        plapoint::gpu::icpTransformResidualOutputPointWriteCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()));
    EXPECT_EQ(
        plapoint::gpu::icpTransformResidualPointTransformCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()));
}

TEST(ICPGpuPathTest, TransformResidualStatsSkipsExactPointwiseProbeWhenCountsDiffer)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source_points(2, 3);
    source_points.setValue(0, 0, 0.0f);
    source_points.setValue(0, 1, 0.0f);
    source_points.setValue(0, 2, 0.0f);
    source_points.setValue(1, 0, 0.5f);
    source_points.setValue(1, 1, 0.0f);
    source_points.setValue(1, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target_points(3, 3);
    target_points.setValue(0, 0, 0.0f);
    target_points.setValue(0, 1, 0.0f);
    target_points.setValue(0, 2, 0.0f);
    target_points.setValue(1, 0, 0.5f);
    target_points.setValue(1, 1, 0.0f);
    target_points.setValue(1, 2, 0.0f);
    target_points.setValue(2, 0, 2.0f);
    target_points.setValue(2, 1, 0.0f);
    target_points.setValue(2, 2, 0.0f);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source_points.rows(), 3);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTransformedExactPointwiseResidualProbeCountForTesting();
    const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.75f,
        output_gpu.data(),
        workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(source_points.rows()));
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseResidualProbeCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, ResidualStatsOrderedHintSkipsSpatialGridSearchForFiniteRadius)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();

    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        workspace,
        0,
        true);

    EXPECT_EQ(stats.active_count, static_cast<int>(source_points.rows()));
    EXPECT_NEAR(stats.residual_sq_sum, 0.0525, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
}

TEST(ICPGpuPathTest, ResidualStatsOrderedHintRejectsUnequalCountsBeforeEmptyReturn)
{
    auto* source = reinterpret_cast<float*>(std::uintptr_t{0x1000});
    auto* target = reinterpret_cast<float*>(std::uintptr_t{0x2000});
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    EXPECT_THROW(
        (void)plapoint::gpu::computeIcpResidualStatsColumnMajor(
            source,
            0,
            target,
            1,
            2.0f,
            workspace,
            0,
            true),
        std::invalid_argument);
}

TEST(ICPGpuPathTest, TransformOrderedResidualStatsAllowsTargetOutputAliasAfterTargetLoad)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeNonCollinearPoints();
    auto transform = makeTranslationTransform(0.1f, -0.05f, 0.025f).toGpu();
    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveResidualStats(static_cast<int>(source_gpu.rows()));

    const auto stats =
        plapoint::gpu::detail::transformPointsAndComputeOrderedIcpResidualStatsColumnMajorWithReservedWorkspace(
            transform.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            2.0f,
            target_gpu.data(),
            workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(source_points.rows()));
    EXPECT_NEAR(stats.residual_sq_sum, 0.0525, 1.0e-6);

    const auto output_cpu = target_gpu.toCpu();
    for (plamatrix::Index row = 0; row < source_points.rows(); ++row)
    {
        EXPECT_NEAR(output_cpu.getValue(row, 0), source_points.getValue(row, 0) + 0.1f, 1.0e-6f);
        EXPECT_NEAR(output_cpu.getValue(row, 1), source_points.getValue(row, 1) - 0.05f, 1.0e-6f);
        EXPECT_NEAR(output_cpu.getValue(row, 2), source_points.getValue(row, 2) + 0.025f, 1.0e-6f);
    }
}

TEST(ICPGpuPathTest, TransformResidualStatsRejectsTargetOutputAliasBeforeCudaAllocation)
{
    auto* transform = reinterpret_cast<float*>(std::uintptr_t{0x1000});
    auto* source = reinterpret_cast<float*>(std::uintptr_t{0x2000});
    auto* target_and_output = reinterpret_cast<float*>(std::uintptr_t{0x3000});
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    EXPECT_THROW(
        (void)plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
            transform,
            source,
            4,
            target_and_output,
            4,
            2.0f,
            target_and_output,
            workspace),
        std::invalid_argument);
}

#endif // PLAPOINT_WITH_CUDA
