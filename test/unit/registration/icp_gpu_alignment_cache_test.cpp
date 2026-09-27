#include "icp_gpu_path_test_support.h"
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_matrix.h>

#ifdef PLAPOINT_WITH_CUDA

TEST(ICPGpuPathTest, AlignDoesNotPopulateGpuPointCpuCaches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    ASSERT_EQ(source->_points_cpu_cache.get(), nullptr);
    ASSERT_EQ(target->_points_cpu_cache.get(), nullptr);

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(3);

    GpuCloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), source->size());
    EXPECT_EQ(source->_points_cpu_cache.get(), nullptr);
    EXPECT_EQ(target->_points_cpu_cache.get(), nullptr);
}

TEST(ICPGpuPathTest, AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, PrepareGpuTargetSpatialIndexMovesBuildBeforeAlign)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    EXPECT_TRUE(icp.prepareGpuTargetSpatialIndex());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);

    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, CorrespondenceStatsSkipsSpatialGridReserveOnCacheHit)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f);
    auto target_points = makeBinaryGridPoints(4096);

    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_points, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_points, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridReserveCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBoundsPartialKernelLaunchCountForTesting();
    const auto first_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0625f,
        nullptr,
        workspace);
    const auto second_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0625f,
        nullptr,
        workspace);

    EXPECT_EQ(first_stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_EQ(second_stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridReserveCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBoundsPartialKernelLaunchCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignReusesGpuWorkspacesAcrossRepeatedCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    icp.align(*source);

    auto* first_partial_storage = icp._gpu_stats_workspace.partialStorage();
    auto* first_stats_storage = icp._gpu_stats_workspace.statsStorage();
    auto* first_grid_keys = icp._gpu_stats_workspace.targetSpatialGridKeysStorage();
    auto* first_grid_indices = icp._gpu_stats_workspace.targetSpatialGridIndicesStorage();
    auto* first_grid_sorted_offsets = icp._gpu_stats_workspace.targetSpatialGridSortedOffsetsStorage();
    auto* first_grid_sorted_x = icp._gpu_stats_workspace.targetSpatialGridSortedXStorage();
    auto* first_grid_sorted_y = icp._gpu_stats_workspace.targetSpatialGridSortedYStorage();
    auto* first_grid_sorted_z = icp._gpu_stats_workspace.targetSpatialGridSortedZStorage();
    auto* first_acc_transform = icp._gpu_T_acc->data();
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    const int first_partial_capacity = icp._gpu_stats_workspace.partialCapacity();
    EXPECT_EQ(icp._gpu_T_step, nullptr);

    auto second_source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto second_source = std::make_shared<GpuCloud>(second_source_cpu->toGpu());
    icp.setInputSource(second_source);
    icp.align(*second_source);

    ASSERT_NE(icp._gpu_T_step, nullptr);
    ASSERT_NE(icp._gpu_T_acc, nullptr);
    std::vector<const float*> first_transform_buffers = {
        icp._gpu_T_step->data(),
        icp._gpu_T_acc->data()};
    std::sort(first_transform_buffers.begin(), first_transform_buffers.end());

    auto third_source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto third_source = std::make_shared<GpuCloud>(third_source_cpu->toGpu());
    icp.setInputSource(third_source);
    icp.align(*third_source);

    EXPECT_EQ(source->size(), target->size());
    EXPECT_EQ(second_source->size(), target->size());
    EXPECT_EQ(third_source->size(), target->size());
    EXPECT_NE(first_partial_storage, nullptr);
    EXPECT_NE(first_stats_storage, nullptr);
    EXPECT_NE(first_grid_keys, nullptr);
    EXPECT_NE(first_grid_indices, nullptr);
    EXPECT_NE(first_grid_sorted_offsets, nullptr);
    EXPECT_NE(first_grid_sorted_x, nullptr);
    EXPECT_NE(first_grid_sorted_y, nullptr);
    EXPECT_NE(first_grid_sorted_z, nullptr);
    EXPECT_NE(first_acc_transform, nullptr);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(icp._gpu_stats_workspace.partialStorage(), first_partial_storage);
    EXPECT_EQ(icp._gpu_stats_workspace.statsStorage(), first_stats_storage);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridKeysStorage(), first_grid_keys);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridIndicesStorage(), first_grid_indices);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridSortedOffsetsStorage(), first_grid_sorted_offsets);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridSortedXStorage(), first_grid_sorted_x);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridSortedYStorage(), first_grid_sorted_y);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridSortedZStorage(), first_grid_sorted_z);
    std::vector<const float*> current_transform_buffers = {
        icp._gpu_T_step->data(),
        icp._gpu_T_acc->data()};
    std::sort(current_transform_buffers.begin(), current_transform_buffers.end());
    EXPECT_EQ(current_transform_buffers, first_transform_buffers);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(icp._gpu_stats_workspace.partialCapacity(), first_partial_capacity);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignReusesCallerGpuOutputStorageWhenShapeMatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    GpuCloud output;
    icp.align(output);

    const auto* first_output_points = static_cast<const GpuCloud&>(output).points().data();
    ASSERT_NE(first_output_points, nullptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);

    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(output).points().data(), first_output_points);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
}

TEST(ICPGpuPathTest, AlignWritesTerminalGpuTransformDirectlyToReusableOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    GpuCloud output(source->size());
    const auto* output_points = static_cast<const GpuCloud&>(output).points().data();

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    plapoint::gpu::resetIcpTransformResidualOutputPointWriteCountForTesting();
    icp.align(output);

    EXPECT_EQ(static_cast<const GpuCloud&>(output).points().data(), output_points);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), nullptr);
    EXPECT_EQ(
        plapoint::gpu::icpTransformResidualOutputPointWriteCountForTesting(),
        static_cast<unsigned long long>(source->size()));
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
}

TEST(ICPGpuPathTest, AlignWritesTerminalTransformDirectlyWhenOutputAliasesSource)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* source_points = static_cast<const GpuCloud&>(*source).points().data();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(*source);

    EXPECT_EQ(source->size(), target->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(*source).points().data(), source_points);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), source_points);
}

TEST(ICPGpuPathTest, AlignSmallTargetSingleIterationSourceAliasAvoidsStaleFullCoverageCache)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    icp.align(*source);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    ASSERT_EQ(source->size(), target->size());
    const auto output_cpu = source->toCpu();
    const auto& output_points = output_cpu.points();
    ASSERT_EQ(output_points.rows(), target_cpu->points().rows());
    ASSERT_EQ(output_points.cols(), target_cpu->points().cols());
    for (plamatrix::Index row = 0; row < output_points.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_points.cols(); ++col)
        {
            EXPECT_NEAR(output_points.operator()(row, col), target_cpu->points().operator()(row, col), 1.0e-5f);
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
            EXPECT_NEAR(second_transform.operator()(row, col), expected, 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignUsesScratchForTerminalTransformWhenOutputAliasesTargetWithoutSpatialGridSnapshot)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(*target);

    EXPECT_EQ(target->size(), source->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    ASSERT_NE(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), icp._gpu_points_a->data());
}

TEST(ICPGpuPathTest, AlignWritesTerminalTransformDirectlyWhenOutputAliasesTargetAndFinalMetricsUseSpatialGrid)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    icp.align(*target);

    EXPECT_EQ(target->size(), source->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), target_points_ptr);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignWritesTerminalOrderedTransformDirectlyWhenOutputAliasesTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(*target);

    EXPECT_EQ(target->size(), source->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), target_points_ptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignUsesScratchForTerminalOrderedTransformWhenAttributedOutputAliasesTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* stale_target_points = static_cast<const GpuCloud&>(*target).points().data();
    plamatrix::internal::ResidentMatrix<float> stale_normals(target->size(), 3, target->executionContext());
    target->setNormals(std::move(stale_normals));
    target->setMaterialLibraryFile("stale.mtl");

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(*target);

    EXPECT_EQ(target->size(), source->size());
    EXPECT_NE(static_cast<const GpuCloud&>(*target).points().data(), stale_target_points);
    ASSERT_NE(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), icp._gpu_points_a->data());
    EXPECT_FALSE(target->hasNormals());
    EXPECT_TRUE(target->materialLibraryFile().empty());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignWritesTerminalTransformDirectlyWhenOutputAliasesTargetAndFinalMetricsDisabled)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points_cpu = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points_cpu, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points_cpu));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(*target);

    EXPECT_EQ(target->size(), source->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), target_points_ptr);
}

TEST(ICPGpuPathTest, AlignReplacesAttributedGpuOutputInsteadOfKeepingStaleMetadata)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    GpuCloud output(source->size());
    plamatrix::internal::ResidentMatrix<float> stale_normals(source->size(), 3, output.executionContext());
    output.setNormals(std::move(stale_normals));
    output.setMaterialLibraryFile("stale.mtl");
    const auto* stale_output_points = static_cast<const GpuCloud&>(output).points().data();

    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NE(static_cast<const GpuCloud&>(output).points().data(), stale_output_points);
    EXPECT_FALSE(output.hasNormals());
    EXPECT_TRUE(output.materialLibraryFile().empty());
}

TEST(ICPGpuPathTest, AlignReplacesIntensityAndScalarFieldOutputInsteadOfKeepingStaleMetadata)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    GpuCloud output(source->size());
    plamatrix::internal::ResidentMatrix<std::uint16_t> stale_intensities(source->size(), 1, output.executionContext());
    plamatrix::internal::ResidentMatrix<float> stale_scalar_fields(source->size(), 1, output.executionContext());
    output.setIntensities(std::move(stale_intensities));
    output.setScalarFields({"error"}, std::move(stale_scalar_fields));
    const auto* stale_output_points = static_cast<const GpuCloud&>(output).points().data();

    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NE(static_cast<const GpuCloud&>(output).points().data(), stale_output_points);
    EXPECT_FALSE(output.hasIntensities());
    EXPECT_FALSE(output.hasScalarFields());
}

TEST(ICPGpuPathTest, RobustTrimmedOverlapMatchesCpuMetrics)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = plamatrix::MatrixXf(5, 3);
    source_points.operator()(0, 0) = 0.0f;
    source_points.operator()(0, 1) = 0.0f;
    source_points.operator()(0, 2) = 0.0f;
    source_points.operator()(1, 0) = 1.0f;
    source_points.operator()(1, 1) = 0.0f;
    source_points.operator()(1, 2) = 0.0f;
    source_points.operator()(2, 0) = 0.0f;
    source_points.operator()(2, 1) = 1.0f;
    source_points.operator()(2, 2) = 0.0f;
    source_points.operator()(3, 0) = 0.0f;   source_points.operator()(3, 1) = 0.0f;   source_points.operator()(3, 2) = 1.0f;
    source_points.operator()(4, 0) = 100.0f; source_points.operator()(4, 1) = 100.0f; source_points.operator()(4, 2) = 100.0f;

    auto target_points = makeNonCollinearPoints();
    CpuCloud source_cpu(std::move(source_points));
    CpuCloud target_cpu(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu.toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu.toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(1);
    icp.setComputeFinalMetrics(false);
    icp.setTrimmedOverlapRatio(0.6f);

    GpuCloud output;
    icp.align(output);

    ASSERT_EQ(output.size(), source->size());
    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getFitnessScore(), 0.6f, 1.0e-6f);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
}

TEST(ICPGpuPathTest, SetInputTargetKeepsPersistentGpuTargetSpatialGridCacheForSameTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);

    icp.setInputTarget(target);
    GpuCloud second_output;
    icp.align(second_output);

    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignInvalidatesTargetSpatialGridCacheWhenCorrespondenceRadiusChanges)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);
    GpuCloud second_output;
    icp.align(second_output);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);

    icp.setMaxCorrespondenceDistance(0.05f);
    GpuCloud third_output;
    icp.align(third_output);
    GpuCloud fourth_output;
    icp.align(fourth_output);

    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(third_output.size(), source->size());
    EXPECT_EQ(fourth_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignInvalidatesPersistentGpuTargetSpatialGridCacheAfterTargetAliasedOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    icp.align(*target);

    GpuCloud second_output;
    icp.align(second_output);

    EXPECT_EQ(target->size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignInvalidatesPersistentGpuTargetSpatialGridCacheAfterMutableTargetPointsAccess)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);

    auto& mutable_target_points = target->points();
    auto edited_points = mutable_target_points.toHostMatrix();
    edited_points(0, 0) += 0.01f;
    mutable_target_points.copyFromHost(edited_points.data(), static_cast<std::size_t>(edited_points.size()));
    GpuCloud second_output;
    icp.align(second_output);

    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 2);
}

TEST(ICPGpuPathTest, RetainedMutableTargetAliasDisablesSpatialGridReuse)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    auto& retained_target_alias = target->points();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);

    auto edited_points = retained_target_alias.toHostMatrix();
    edited_points(0, 0) += 0.01f;
    retained_target_alias.copyFromHost(edited_points.data(), static_cast<std::size_t>(edited_points.size()));
    GpuCloud second_output;
    icp.align(second_output);

    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 2);
}

TEST(ICPGpuPathTest, ScopedTargetEditRebuildsSpatialGridOnceThenRestoresReuse)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);

    {
        auto edit = target->editPoints();
        auto edited_points = edit->toHostMatrix();
        edited_points(0, 0) += 0.01f;
        edit->copyFromHost(edited_points.data(), static_cast<std::size_t>(edited_points.size()));
    }
    GpuCloud second_output;
    icp.align(second_output);
    GpuCloud third_output;
    icp.align(third_output);

    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(third_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignDoesNotIncrementTargetPointsVersionForSameBufferNoWriteOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto cpu_cloud = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    const auto initial_version = cloud->pointsVersion();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    icp.align(*cloud);

    EXPECT_EQ(cloud->pointsVersion(), initial_version);
}

TEST(ICPGpuPathTest, SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCacheForNewTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto first_target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto second_target_cpu =
        std::make_shared<CpuCloud>(makeTranslatedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto first_target = std::make_shared<GpuCloud>(first_target_cpu->toGpu());
    auto second_target = std::make_shared<GpuCloud>(second_target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(first_target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);

    icp.setInputTarget(second_target);
    GpuCloud second_output;
    icp.align(second_output);

    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignReadsInitialGpuSourceBufferDirectly)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    const void* source_points = source->points().data();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(3);

    plapoint::gpu::resetIcpFirstStatsSourcePointerForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpFirstStatsSourcePointerForTesting(), source_points);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, FinalTransformationDeviceIsAvailableAfterGpuAlign)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(3);

    GpuCloud output;
    icp.align(output);

    const auto& transform_gpu = icp.getFinalTransformationDevice();
    auto transform_cpu = transform_gpu.toHostMatrix();

    ASSERT_EQ(transform_cpu.rows(), 4);
    ASSERT_EQ(transform_cpu.cols(), 4);
    EXPECT_NEAR(transform_cpu.operator()(0, 0), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(1, 1), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(2, 2), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(3, 3), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(0, 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(1, 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(2, 3), 0.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, GpuAlignMaterializesCpuFinalTransformLazily)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(3);

    GpuCloud output;
    icp.align(output);

    EXPECT_TRUE(icp._final_T_gpu_valid);
    EXPECT_NE(icp._gpu_T_acc.get(), nullptr);
    EXPECT_FALSE(icp._final_T_cpu_valid);

    const auto& transform_cpu = icp.getFinalTransformation();
    EXPECT_TRUE(icp._final_T_cpu_valid);
    ASSERT_EQ(transform_cpu.rows(), 4);
    ASSERT_EQ(transform_cpu.cols(), 4);
    EXPECT_NEAR(transform_cpu.operator()(0, 0), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(1, 1), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(2, 2), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.operator()(3, 3), 1.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignSkipsFinalStatsForNonTerminalGpuIterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    plamatrix::MatrixXf transform(4, 4);
    transform.setConstant(0.0f);
    transform.operator()(0, 0) = 1.0f;
    transform.operator()(1, 1) = 1.0f;
    transform.operator()(2, 2) = 1.0f;
    transform.operator()(3, 3) = 1.0f;
    transform.operator()(0, 3) = 0.2f;
    transform.operator()(1, 3) = -0.1f;
    transform.operator()(2, 3) = 0.05f;
    auto target_points = transformedPoints(transform, source_points);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 3);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignSkipsMetricUpdateForNonTerminalGpuIterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.2f, -0.1f, 0.05f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);

    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(icp._gpu_metric_update_count, 1);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignSkipsNonTerminalPointTransformMaterialization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.2f, -0.1f, 0.05f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignFusesTransformedAlignmentStepWithAccumulatedTransformUpdate)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.2f, -0.1f, 0.05f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);

    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpAccumulatedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAccumulatedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformMultiplyCallCountForTesting(), 0);
    EXPECT_NE(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignAutoProbesTransformedExactPointwiseAfterAllSameIndexStep)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuProbeExactPointwiseOnFiniteRadius(true);

    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpAccumulatedAlignmentStepCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAccumulatedAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignDoesNotAutoProbeTransformedExactPointwiseWhenSameIndexStepIsNotExact)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuProbeExactPointwiseOnFiniteRadius(true);

    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpAccumulatedAlignmentStepCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAccumulatedAlignmentStepCallCountForTesting(), 1);
    EXPECT_NE(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, TransformedAlignmentStepSkipsSpatialGridSearchForExactPointwiseMatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_points, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_points, matrix_context);
    auto transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(transform, matrix_context);
    plamatrix::internal::ResidentMatrix<float> step_transform(4, 4, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedIdentityAlignmentStepCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    const auto result =
        plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
            transform_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            2.0f,
            workspace,
            step_transform.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_NEAR(result.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedIdentityAlignmentStepCountForTesting(), 1);
    EXPECT_EQ(
        plapoint::gpu::icpExactPointwiseTargetLoadCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()) * 3ull);

    const auto step_cpu = step_transform.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(step_cpu.operator()(row, col), expected, 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, TransformedAccumulatedAlignmentStepSkipsSpatialGridPrepareForExactPointwiseMatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_points, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_points, matrix_context);
    auto transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(transform, matrix_context);
    plamatrix::internal::ResidentMatrix<float> step_transform(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> accumulated_transform(4, 4, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedIdentityAlignmentStepCountForTesting();
    const auto result =
        plapoint::gpu::detail::
            computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                transform_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                2.0f,
                workspace,
                step_transform.data(),
                transform_gpu.data(),
                accumulated_transform.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_NEAR(result.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedIdentityAlignmentStepCountForTesting(), 1);

    const auto accumulated_cpu = accumulated_transform.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(accumulated_cpu.operator()(row, col), transform.operator()(row, col), 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, TransformedExactPointwiseAlignmentStepFallsBackToSpatialGridOnMismatch)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(-0.25f, 0.125f, -0.0625f);

    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_points, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_points, matrix_context);
    auto transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(transform, matrix_context);
    plamatrix::internal::ResidentMatrix<float> step_transform(4, 4, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    const auto result =
        plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
            transform_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            2.0f,
            workspace,
            step_transform.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_TRUE(std::isfinite(result.residual_sq_sum));
    EXPECT_GT(result.residual_sq_sum, 0.0);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(
        plapoint::gpu::icpExactPointwiseTargetLoadCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()) * 3ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, TransformedAlignmentStepUsesCachedSpatialGridWithoutExactPointwiseProbe)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto cache_build_transform = makeTranslationTransform(-0.25f, 0.125f, -0.0625f);
    auto exact_transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_points, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_points, matrix_context);
    auto cache_build_transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(cache_build_transform, matrix_context);
    auto exact_transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(exact_transform, matrix_context);
    plamatrix::internal::ResidentMatrix<float> step_transform(4, 4, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    const auto cache_build_result =
        plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
            cache_build_transform_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            2.0f,
            workspace,
            step_transform.data());

    ASSERT_TRUE(cache_build_result.step_valid);
    ASSERT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    ASSERT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    ASSERT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    const auto result =
        plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
            exact_transform_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            2.0f,
            workspace,
            step_transform.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_NEAR(result.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
}

TEST(ICPGpuPathTest, TransformedAlignmentStepCanProbeExactPointwiseOnCacheHitWhenRequested)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto cache_build_transform = makeTranslationTransform(-0.25f, 0.125f, -0.0625f);
    auto exact_transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_points, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_points, matrix_context);
    auto cache_build_transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(cache_build_transform, matrix_context);
    auto exact_transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(exact_transform, matrix_context);
    plamatrix::internal::ResidentMatrix<float> step_transform(4, 4, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));

    const auto cache_build_result =
        plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
            cache_build_transform_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            2.0f,
            workspace,
            step_transform.data());
    ASSERT_TRUE(cache_build_result.step_valid);

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    const auto result =
        plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
            exact_transform_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            2.0f,
            workspace,
            step_transform.data(),
            0,
            false,
            true);

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_NEAR(result.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
}

TEST(ICPGpuPathTest, TransformedExactPointwiseAccumulatedFallbackDoesNotWriteInvalidTransform)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(10.0f, -10.0f, 5.0f);

    plamatrix::MatrixXf accumulated_sentinel(4, 4);
    accumulated_sentinel.setConstant(7.0f);

    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_points, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_points, matrix_context);
    auto transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(transform, matrix_context);
    auto accumulated_transform = plamatrix::internal::ResidentMatrix<float>::copyFrom(accumulated_sentinel, matrix_context);
    plamatrix::internal::ResidentMatrix<float> step_transform(4, 4, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    const auto result =
        plapoint::gpu::detail::
            computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                transform_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.01f,
                workspace,
                step_transform.data(),
                transform_gpu.data(),
                accumulated_transform.data());

    EXPECT_EQ(result.active_count, 0);
    EXPECT_FALSE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);

    const auto accumulated_cpu = accumulated_transform.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                accumulated_cpu.operator()(row, col),
                accumulated_sentinel.operator()(row, col),
                1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignDoesNotProbeExactPointwiseForEqualFiniteRadiusInputsByDefault)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpExactPointwiseStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignCanProbeExactPointwiseForEqualFiniteRadiusInputsWhenEnabled)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);
    icp.setGpuProbeExactPointwiseOnFiniteRadius(true);

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_TRUE(icp.hasConverged());
    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignCanUseOrderedPointwiseCorrespondencesForFiniteRadiusTranslation)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpExactPointwiseStepCallCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpExactPointwiseStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    const auto& final_transform = icp.getFinalTransformation();
    EXPECT_NEAR(final_transform.operator()(0, 3), -0.1f, 1.0e-5f);
    EXPECT_NEAR(final_transform.operator()(1, 3), 0.05f, 1.0e-5f);
    EXPECT_NEAR(final_transform.operator()(2, 3), -0.025f, 1.0e-5f);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignmentStepPrefersSameBufferExactIdentityWhenOrdered)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto points = plamatrix::internal::ResidentMatrix<float>::copyFrom(makeNonCollinearPoints(), matrix_context);
    plamatrix::internal::ResidentMatrix<float> step_transform(4, 4, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(static_cast<int>(points.rows()));

    plapoint::gpu::resetIcpExactPointwiseIdentityStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSameBufferIdentityAlignmentStepCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    const auto result = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
        points.data(),
        static_cast<int>(points.rows()),
        points.data(),
        static_cast<int>(points.rows()),
        2.0f,
        workspace,
        step_transform.data(),
        0,
        true);

    EXPECT_EQ(result.active_count, static_cast<int>(points.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_NEAR(result.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseTargetLoadCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseIdentityStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSameBufferIdentityAlignmentStepCountForTesting(), 1);

    const auto step_cpu = step_transform.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(step_cpu.operator()(row, col), expected, 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, AlignPropagatesOrderedCorrespondencesToTransformedIterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.1f, -0.05f, 0.025f);
    target_points.operator()(1, 0) = target_points.operator()(1, 0) + 0.025f;
    target_points.operator()(2, 1) = target_points.operator()(2, 1) - 0.015f;
    target_points.operator()(3, 2) = target_points.operator()(3, 2) + 0.01f;
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignCanProbeTransformedExactPointwiseOnCacheHitWhenRequested)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
}

#endif // PLAPOINT_WITH_CUDA
