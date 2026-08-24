#include "icp_gpu_path_test_support.h"

#ifdef PLAPOINT_WITH_CUDA

TEST(ICPGpuPathTest, CorrespondenceStatsAllowOmittedIndexOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeNonCollinearPoints().toGpu();
    auto target = makeNonCollinearPoints().toGpu();

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        std::numeric_limits<float>::infinity(),
        nullptr);

    EXPECT_EQ(stats.active_count, 4);
    EXPECT_EQ(stats.invalid_source_count, 0);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_NEAR(stats.src_centroid[0], 0.25, 1.0e-12);
    EXPECT_NEAR(stats.src_centroid[1], 0.25, 1.0e-12);
    EXPECT_NEAR(stats.src_centroid[2], 0.25, 1.0e-12);
    EXPECT_NEAR(stats.tgt_centroid[0], 0.25, 1.0e-12);
    EXPECT_NEAR(stats.tgt_centroid[1], 0.25, 1.0e-12);
    EXPECT_NEAR(stats.tgt_centroid[2], 0.25, 1.0e-12);
    EXPECT_TRUE(stats.src_has_non_collinear_geometry);
    EXPECT_TRUE(stats.tgt_has_non_collinear_geometry);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsStillWriteRequestedIndexOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeNonCollinearPoints().toGpu();
    auto target = makeNonCollinearPoints().toGpu();
    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        std::numeric_limits<float>::infinity(),
        indices.get());

    std::vector<int> host_indices(static_cast<std::size_t>(source.rows()), -1);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        host_indices.data(),
        indices.get(),
        host_indices.size() * sizeof(int),
        cudaMemcpyDeviceToHost));

    EXPECT_EQ(stats.active_count, 4);
    ASSERT_EQ(host_indices.size(), 4u);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(host_indices[static_cast<std::size_t>(i)], i);
    }
    EXPECT_GT(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsReportsDegenerateGeometry)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeCollinearPoints().toGpu();
    auto target = makeCollinearPoints().toGpu();

    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        std::numeric_limits<float>::infinity(),
        nullptr);

    EXPECT_EQ(stats.active_count, 4);
    EXPECT_FALSE(stats.src_has_non_collinear_geometry);
    EXPECT_FALSE(stats.tgt_has_non_collinear_geometry);
}

TEST(ICPGpuPathTest, CorrespondenceStatsSkipsRedundantOuterLowerTriangleAccumulation)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeNonCollinearPoints();
    auto target_cpu = makeTranslatedNonCollinearPoints(source_cpu, 0.1f, 0.2f, 0.3f);
    const auto expected = makeMatchedStats(source_cpu, target_cpu);
    auto source = source_cpu.toGpu();
    auto target = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    plapoint::gpu::resetIcpOuterLowerTriangleAccumulationCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, expected.active_count);
    EXPECT_EQ(stats.invalid_source_count, expected.invalid_source_count);
    EXPECT_NEAR(stats.residual_sq_sum, expected.residual_sq_sum, 1.0e-6);
    for (int idx = 0; idx < 9; ++idx)
    {
        EXPECT_NEAR(stats.cross_covariance[idx], expected.cross_covariance[idx], 1.0e-6);
        EXPECT_NEAR(stats.src_covariance[idx], expected.src_covariance[idx], 1.0e-6);
        EXPECT_NEAR(stats.tgt_covariance[idx], expected.tgt_covariance[idx], 1.0e-6);
    }
    EXPECT_EQ(plapoint::gpu::icpOuterLowerTriangleAccumulationCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsFindsNearestTargetsPastFirstTile)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(3, 3);
    source.setValue(0, 0, -3.0f); source.setValue(0, 1, 1.0f);  source.setValue(0, 2, 0.5f);
    source.setValue(1, 0, 4.0f);  source.setValue(1, 1, -2.0f); source.setValue(1, 2, 1.5f);
    source.setValue(2, 0, 8.0f);  source.setValue(2, 1, 2.0f);  source.setValue(2, 2, -1.0f);

    constexpr int target_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    for (int i = 0; i < target_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i));
        target.setValue(i, 1, -1000.0f - static_cast<float>(i));
        target.setValue(i, 2, 500.0f + static_cast<float>(i));
    }

    const int expected_indices[3]{130, 200, 256};
    for (int row = 0; row < 3; ++row)
    {
        const int target_row = expected_indices[row];
        target.setValue(target_row, 0, source.getValue(row, 0));
        target.setValue(target_row, 1, source.getValue(row, 1));
        target.setValue(target_row, 2, source.getValue(row, 2));
    }

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));

    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        indices.get());

    std::vector<int> host_indices(static_cast<std::size_t>(source.rows()), -1);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        host_indices.data(),
        indices.get(),
        host_indices.size() * sizeof(int),
        cudaMemcpyDeviceToHost));

    EXPECT_EQ(stats.active_count, 3);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    for (int row = 0; row < 3; ++row)
    {
        EXPECT_EQ(host_indices[static_cast<std::size_t>(row)], expected_indices[row]);
    }
}

TEST(ICPGpuPathTest, CorrespondenceStatsPrunesFarTargetsBeforeFullDistanceEvaluation)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(129, 3);
    target.setValue(0, 0, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 1, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 2, std::numeric_limits<float>::quiet_NaN());
    target.setValue(1, 0, 0.0f);   target.setValue(1, 1, 0.0f);    target.setValue(1, 2, 0.0f);
    target.setValue(2, 0, 100.0f); target.setValue(2, 1, 0.0f);    target.setValue(2, 2, 0.0f);
    target.setValue(3, 0, 0.0f);   target.setValue(3, 1, -200.0f); target.setValue(3, 2, 0.0f);
    for (int row = 4; row < static_cast<int>(target.rows()); ++row)
    {
        target.setValue(row, 0, 1000.0f + static_cast<float>(row));
        target.setValue(row, 1, 1000.0f);
        target.setValue(row, 2, 1000.0f);
    }

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        nullptr);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 1ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    for (int i = 0; i < target_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i));
        target.setValue(i, 1, 1000.0f);
        target.setValue(i, 2, 1000.0f);
    }
    target.setValue(0, 0, 0.0f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        indices.get());

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 128ull);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 1ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(3, 3);
    target.setValue(0, 0, 0.0f); target.setValue(0, 1, 0.0f); target.setValue(0, 2, 0.0f);
    target.setValue(1, 0, 1.0f); target.setValue(1, 1, 0.0f); target.setValue(1, 2, 0.0f);
    target.setValue(2, 0, 2.0f); target.setValue(2, 1, 0.0f); target.setValue(2, 2, 0.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        nullptr);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 1ull);

    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    const auto indexed_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        indices.get());

    int host_index = -1;
    PLAPOINT_CHECK_CUDA(cudaMemcpy(&host_index, indices.get(), sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(indexed_stats.active_count, 1);
    EXPECT_NEAR(indexed_stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(host_index, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 3ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsPrecomputesFiniteRadiusTargetTileBoundsOnce)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    constexpr int point_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(point_count, 3);
    for (int i = 0; i < point_count; ++i)
    {
        source.setValue(i, 0, 0.0f);
        source.setValue(i, 1, 0.0f);
        source.setValue(i, 2, 0.0f);
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(point_count, 3);
    for (int i = 0; i < point_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i));
        target.setValue(i, 1, 1000.0f);
        target.setValue(i, 2, 1000.0f);
    }
    target.setValue(0, 0, 0.0f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetTileBoundComputationCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, point_count);
    EXPECT_EQ(plapoint::gpu::icpTargetTileBoundComputationCountForTesting(), 3ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsReusesFiniteRadiusTargetTileBoundsForSameTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    constexpr int point_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(point_count, 3);
    for (int i = 0; i < point_count; ++i)
    {
        source.setValue(i, 0, 0.0f);
        source.setValue(i, 1, 0.0f);
        source.setValue(i, 2, 0.0f);
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(point_count, 3);
    for (int i = 0; i < point_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i));
        target.setValue(i, 1, 1000.0f);
        target.setValue(i, 2, 1000.0f);
    }
    target.setValue(0, 0, 0.0f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetTileBoundComputationCountForTesting();
    plapoint::gpu::resetIcpTargetTileBoundsReserveCountForTesting();
    const auto first_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        nullptr,
        workspace);
    const auto second_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        nullptr,
        workspace);

    EXPECT_EQ(first_stats.active_count, point_count);
    EXPECT_EQ(second_stats.active_count, point_count);
    EXPECT_EQ(plapoint::gpu::icpTargetTileBoundComputationCountForTesting(), 3ull);
    EXPECT_EQ(plapoint::gpu::icpTargetTileBoundsReserveCountForTesting(), 1);
}

TEST(ICPGpuPathTest, CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 256;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    target.setValue(0, 0, 0.5f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);
    for (int i = 1; i < target_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i * 3));
        target.setValue(i, 1, 1000.0f);
        target.setValue(i, 2, 1000.0f);
    }

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_LE(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 4ull);
    EXPECT_GE(workspace.targetSpatialGridCapacity(), target_count);
    auto* first_grid_keys = workspace.targetSpatialGridKeysStorage();
    auto* first_grid_unique_keys = workspace.targetSpatialGridUniqueKeysStorage();
    auto* first_grid_indices = workspace.targetSpatialGridIndicesStorage();
    auto* first_grid_sorted_offsets = workspace.targetSpatialGridSortedOffsetsStorage();
    auto* first_grid_sorted_x = workspace.targetSpatialGridSortedXStorage();
    auto* first_grid_sorted_y = workspace.targetSpatialGridSortedYStorage();
    auto* first_grid_sorted_z = workspace.targetSpatialGridSortedZStorage();
    auto* first_grid_cell_starts = workspace.targetSpatialGridCellStartsStorage();
    auto* first_grid_cell_counts = workspace.targetSpatialGridCellCountsStorage();
    EXPECT_NE(first_grid_keys, nullptr);
    EXPECT_NE(first_grid_unique_keys, nullptr);
    EXPECT_NE(first_grid_indices, nullptr);
    EXPECT_NE(first_grid_sorted_offsets, nullptr);
    EXPECT_NE(first_grid_sorted_x, nullptr);
    EXPECT_NE(first_grid_sorted_y, nullptr);
    EXPECT_NE(first_grid_sorted_z, nullptr);
    EXPECT_NE(first_grid_cell_starts, nullptr);
    EXPECT_NE(first_grid_cell_counts, nullptr);

    const auto second_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(second_stats.active_count, 1);
    EXPECT_EQ(workspace.targetSpatialGridKeysStorage(), first_grid_keys);
    EXPECT_EQ(workspace.targetSpatialGridUniqueKeysStorage(), first_grid_unique_keys);
    EXPECT_EQ(workspace.targetSpatialGridIndicesStorage(), first_grid_indices);
    EXPECT_EQ(workspace.targetSpatialGridSortedOffsetsStorage(), first_grid_sorted_offsets);
    EXPECT_EQ(workspace.targetSpatialGridSortedXStorage(), first_grid_sorted_x);
    EXPECT_EQ(workspace.targetSpatialGridSortedYStorage(), first_grid_sorted_y);
    EXPECT_EQ(workspace.targetSpatialGridSortedZStorage(), first_grid_sorted_z);
    EXPECT_EQ(workspace.targetSpatialGridCellStartsStorage(), first_grid_cell_starts);
    EXPECT_EQ(workspace.targetSpatialGridCellCountsStorage(), first_grid_cell_counts);
}

TEST(ICPGpuPathTest, CorrespondenceStatsSpatialGridSkipsNonFiniteTargetInSaturatedCell)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    constexpr float saturated_cell_value = 2147483648.0f;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, saturated_cell_value);
    source.setValue(0, 1, saturated_cell_value);
    source.setValue(0, 2, saturated_cell_value);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(2, 3);
    target.setValue(0, 0, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 1, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 2, std::numeric_limits<float>::quiet_NaN());
    target.setValue(1, 0, saturated_cell_value);
    target.setValue(1, 1, saturated_cell_value);
    target.setValue(1, 2, saturated_cell_value);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::DeviceBuffer<int> indices(1);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        indices.get(),
        workspace);

    int host_index = -1;
    PLAPOINT_CHECK_CUDA(cudaMemcpy(&host_index, indices.get(), sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(host_index, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 2ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsBatchesSpatialGridNeighborLookupsByXY)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, x == 0 ? 0.0f : static_cast<float>(x));
                target.setValue(idx, 1, y == 0 ? 0.0f : static_cast<float>(y));
                target.setValue(idx, 2, z == 0 ? 0.0f : static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_LE(plapoint::gpu::icpGridCellLookupCountForTesting(), 9ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsUsesDirectSpatialGridCellLookupForCompactTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x));
                target.setValue(idx, 1, static_cast<float>(y));
                target.setValue(idx, 2, static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, SpatialGridDirectLookupIgnoresNonFiniteTargetSentinelForCompactValidCells)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 28;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    target.setValue(0, 0, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 1, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 2, std::numeric_limits<float>::quiet_NaN());
    int idx = 1;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x));
                target.setValue(idx, 1, static_cast<float>(y));
                target.setValue(idx, 2, static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpDirectSpatialGridTargetPointBoundsFallbackCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridTargetPointBoundsFallbackCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, ResidualStatsUsesDirectSpatialGridCellLookupForCompactTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x));
                target.setValue(idx, 1, static_cast<float>(y));
                target.setValue(idx, 2, static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    const auto correspondence_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(correspondence_stats.active_count, 1);
    EXPECT_NEAR(correspondence_stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, TransformResidualStatsUsesDirectSpatialGridCellLookupForCompactTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x));
                target.setValue(idx, 1, static_cast<float>(y));
                target.setValue(idx, 2, static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, TransformResidualStatsSnapshotSeedsSameIndexWhenOutputAliasesTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeNonCollinearPoints();
    auto target = makeTranslatedNonCollinearPoints(source, 0.1f, -0.05f, 0.025f);
    auto transform = makeTranslationTransform(0.1f, -0.05f, 0.025f);
    target = padTargetWithNonFiniteRows(target);
    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    auto transform_gpu = transform.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto seed_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        nullptr,
        workspace);

    ASSERT_EQ(seed_stats.active_count, static_cast<int>(source_gpu.rows()));
    ASSERT_GT(workspace.targetSpatialGridCellCount(), 0);
    ASSERT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    workspace.reserveResidualStats(static_cast<int>(source_gpu.rows()));

    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    const auto final_stats =
        plapoint::gpu::detail::
            transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajorWithReservedWorkspace(
                transform_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                2.0f,
                target_gpu.data(),
                workspace,
                workspace.targetSpatialGridCellCount());

    EXPECT_EQ(final_stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_NEAR(final_stats.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, SpatialGridDirectLookupUsesSpecializedKernelLaunches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x));
                target.setValue(idx, 1, static_cast<float>(y));
                target.setValue(idx, 2, static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 1);

    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 1);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpDirectSpatialGridKernelLaunchCountForTesting();
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpDirectSpatialGridKernelLaunchCountForTesting(), 1);
}

TEST(ICPGpuPathTest, SpatialGridDirectLookupSpecializationSkipsActiveGuard)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x));
                target.setValue(idx, 1, static_cast<float>(y));
                target.setValue(idx, 2, static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpDirectGridLookupActiveGuardCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupActiveGuardCountForTesting(), 0ull);

    plapoint::gpu::resetIcpDirectGridLookupActiveGuardCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupActiveGuardCountForTesting(), 0ull);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpDirectGridLookupActiveGuardCountForTesting();
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupActiveGuardCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, SpatialGridDirectLookupSpecializationSkipsXyBaseGuard)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x));
                target.setValue(idx, 1, static_cast<float>(y));
                target.setValue(idx, 2, static_cast<float>(z));
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpDirectGridLookupXyBaseGuardCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyBaseGuardCountForTesting(), 0ull);

    plapoint::gpu::resetIcpDirectGridLookupXyBaseGuardCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyBaseGuardCountForTesting(), 0ull);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpDirectGridLookupXyBaseGuardCountForTesting();
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyBaseGuardCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, SpatialGridExactMatchChecksCenterZCellBeforeAdjacentZCells)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(2, 3);
    target.setValue(0, 0, 0.0f); target.setValue(0, 1, 0.0f); target.setValue(0, 2, -0.5f);
    target.setValue(1, 0, 0.0f); target.setValue(1, 1, 0.0f); target.setValue(1, 2, 0.0f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellOffsetCountForTesting();
    plapoint::gpu::resetIcpGridCellCenterMinDistanceCountForTesting();
    plapoint::gpu::resetIcpDirectGridLookupLinearGuardCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellOffsetCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellCenterMinDistanceCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupLinearGuardCountForTesting(), 0ull);

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellOffsetCountForTesting();
    plapoint::gpu::resetIcpGridCellCenterMinDistanceCountForTesting();
    plapoint::gpu::resetIcpDirectGridLookupLinearGuardCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_NEAR(residual_stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellOffsetCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellCenterMinDistanceCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupLinearGuardCountForTesting(), 0ull);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellOffsetCountForTesting();
    plapoint::gpu::resetIcpGridCellCenterMinDistanceCountForTesting();
    plapoint::gpu::resetIcpDirectGridLookupLinearGuardCountForTesting();
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_NEAR(transform_residual_stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellOffsetCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellCenterMinDistanceCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupLinearGuardCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, SpatialGridDirectLookupChecksXYRangeOnceForNeighborZColumn)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.5f);
    source.setValue(0, 1, 0.5f);
    source.setValue(0, 2, 0.99f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(1, 3);
    target.setValue(0, 0, 0.5f);
    target.setValue(0, 1, 0.5f);
    target.setValue(0, 2, 1.01f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpDirectGridLookupXyCheckCountForTesting();
    plapoint::gpu::resetIcpGridCellNeighborMinDistanceCountForTesting();
    plapoint::gpu::resetIcpGridCellNeighborXyDistanceCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0004, 1.0e-6);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyCheckCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellNeighborMinDistanceCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellNeighborXyDistanceCountForTesting(), 0ull);

    plapoint::gpu::resetIcpDirectGridLookupXyCheckCountForTesting();
    plapoint::gpu::resetIcpGridCellNeighborMinDistanceCountForTesting();
    plapoint::gpu::resetIcpGridCellNeighborXyDistanceCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_NEAR(residual_stats.residual_sq_sum, 0.0004, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyCheckCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellNeighborMinDistanceCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellNeighborXyDistanceCountForTesting(), 0ull);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpDirectGridLookupXyCheckCountForTesting();
    plapoint::gpu::resetIcpGridCellNeighborMinDistanceCountForTesting();
    plapoint::gpu::resetIcpGridCellNeighborXyDistanceCountForTesting();
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_NEAR(transform_residual_stats.residual_sq_sum, 0.0004, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyCheckCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellNeighborMinDistanceCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellNeighborXyDistanceCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, SpatialGridDirectLookupPrunesXYBeforeBaseLookup)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.2f);
    source.setValue(0, 1, 0.2f);
    source.setValue(0, 2, 0.2f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x) + 0.25f);
                target.setValue(idx, 1, static_cast<float>(y) + 0.25f);
                target.setValue(idx, 2, static_cast<float>(z) + 0.25f);
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpDirectGridLookupXyCheckCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0075, 1.0e-6);
    EXPECT_GT(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyCheckCountForTesting(), 1ull);

    plapoint::gpu::resetIcpDirectGridLookupXyCheckCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_NEAR(residual_stats.residual_sq_sum, 0.0075, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyCheckCountForTesting(), 1ull);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpDirectGridLookupXyCheckCountForTesting();
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_NEAR(transform_residual_stats.residual_sq_sum, 0.0075, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpDirectGridLookupXyCheckCountForTesting(), 1ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsKeepsSparseUniqueCellRangeOnLowerBoundPath)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 2000;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    for (int idx = 0; idx < target_count; ++idx)
    {
        target.setValue(idx, 0, 0.0f);
        target.setValue(idx, 1, 0.0f);
        target.setValue(idx, 2, 0.0f);
    }
    target.setValue(0, 0, 2000.0f);
    target.setValue(target_count - 1, 0, 2000.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpGridCellCenterMinDistanceCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_GT(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellCenterMinDistanceCountForTesting(), 0ull);

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpGridCellCenterMinDistanceCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace residual_workspace;
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        residual_workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_NEAR(residual_stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(residual_workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_GT(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellCenterMinDistanceCountForTesting(), 0ull);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpGridCellCenterMinDistanceCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace transform_residual_workspace;
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        transform_residual_workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_NEAR(transform_residual_stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(transform_residual_workspace.targetSpatialGridDirectLookupEntryCount(), 0);
    EXPECT_GT(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellCenterMinDistanceCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsPrunesSpatialGridXYLookupsBeforeSearch)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.01f);
    source.setValue(0, 1, 0.01f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 9;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            target.setValue(idx, 0, x < 0 ? -0.5f : (x == 0 ? 0.99f : 1.0f));
            target.setValue(idx, 1, y < 0 ? -0.5f : (y == 0 ? 0.99f : 1.0f));
            target.setValue(idx, 2, 0.0f);
            ++idx;
        }
    }

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_LE(plapoint::gpu::icpGridCellLookupCountForTesting(), 5ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.25f);
    source.setValue(0, 1, 0.25f);
    source.setValue(0, 2, 0.25f);

    constexpr int target_count = 27;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    int idx = 0;
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                target.setValue(idx, 0, static_cast<float>(x) + 0.25f);
                target.setValue(idx, 1, static_cast<float>(y) + 0.25f);
                target.setValue(idx, 2, static_cast<float>(z) + 0.25f);
                ++idx;
            }
        }
    }
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_LE(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 2ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsSeedsSameIndexCandidateBeforeSpatialGridSearch)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.99f);
    source.setValue(0, 1, 0.99f);
    source.setValue(0, 2, 0.99f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(4, 3);
    target.setValue(0, 0, 1.01f); target.setValue(0, 1, 1.01f); target.setValue(0, 2, 1.01f);
    target.setValue(1, 0, 0.10f); target.setValue(1, 1, 0.10f); target.setValue(1, 2, -0.10f);
    target.setValue(2, 0, 0.10f); target.setValue(2, 1, -0.10f); target.setValue(2, 2, 0.10f);
    target.setValue(3, 0, -0.10f); target.setValue(3, 1, 0.10f); target.setValue(3, 2, 0.10f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        indices.get(),
        workspace);

    int host_index = -1;
    PLAPOINT_CHECK_CUDA(cudaMemcpy(&host_index, indices.get(), sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0012, 1.0e-6);
    EXPECT_EQ(host_index, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_NEAR(residual_stats.residual_sq_sum, 0.0012, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);

    auto identity = makeTranslationTransform(0.0f, 0.0f, 0.0f);
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    const auto transform_residual_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transform_residual_stats.active_count, 1);
    EXPECT_NEAR(transform_residual_stats.residual_sq_sum, 0.0012, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.8f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(3, 3);
    target.setValue(0, 0, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 1, std::numeric_limits<float>::quiet_NaN());
    target.setValue(0, 2, std::numeric_limits<float>::quiet_NaN());
    target.setValue(1, 0, 1.6f);
    target.setValue(1, 1, 0.0f);
    target.setValue(1, 2, 0.0f);
    target.setValue(2, 0, 0.2f);
    target.setValue(2, 1, 0.0f);
    target.setValue(2, 2, 0.0f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpTargetIndexLoadCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.36, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 2ull);
    EXPECT_EQ(plapoint::gpu::icpTargetIndexLoadCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.95f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(1, 3);
    target.setValue(0, 0, -0.2f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpSortedTargetCoordinateLoadCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpSortedTargetCoordinateLoadCountForTesting(), 1ull);

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpSortedTargetCoordinateLoadCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);

    EXPECT_EQ(residual_stats.active_count, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpSortedTargetCoordinateLoadCountForTesting(), 1ull);
}

TEST(ICPGpuPathTest, SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.2f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(3, 3);
    target.setValue(0, 0, 2.1f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);
    target.setValue(1, 0, 0.1f);
    target.setValue(1, 1, 0.0f);
    target.setValue(1, 2, 0.0f);
    target.setValue(2, 0, 0.25f);
    target.setValue(2, 1, 0.9f);
    target.setValue(2, 2, 0.9f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpSortedTargetCoordinateLoadCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.01, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 2ull);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpSortedTargetCoordinateLoadCountForTesting(), 5ull);

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpSortedTargetCoordinateLoadCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);

    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_NEAR(residual_stats.residual_sq_sum, 0.01, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 2ull);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpSortedTargetCoordinateLoadCountForTesting(), 5ull);
}

TEST(ICPGpuPathTest, SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(1, 3);
    target.setValue(0, 0, 0.8f);
    target.setValue(0, 1, 0.8f);
    target.setValue(0, 2, 0.0f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpSortedTargetCoordinateLoadCountForTesting();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpSortedTargetCoordinateLoadCountForTesting(), 2ull);

    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpSortedTargetCoordinateLoadCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        workspace);

    EXPECT_EQ(residual_stats.active_count, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpSortedTargetCoordinateLoadCountForTesting(), 2ull);
}

TEST(ICPGpuPathTest, ResidualStatsStopsSpatialGridLookupsAfterExactMatch)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeNonCollinearPoints().toGpu();
    auto target = makeNonCollinearPoints().toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        2.0f,
        workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(source.rows()));
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_LE(plapoint::gpu::icpGridCellLookupCountForTesting(),
              static_cast<unsigned long long>(source.rows()));
}

TEST(ICPGpuPathTest, ResidualStatsUsesExactPointwiseFastPathForSameBuffer)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto points = makeNonCollinearPoints().toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        points.data(),
        static_cast<int>(points.rows()),
        points.data(),
        static_cast<int>(points.rows()),
        2.0f,
        workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(points.rows()));
    EXPECT_EQ(stats.invalid_source_count, 0);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseTargetLoadCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, ResidualStatsReservedWorkspaceSkipsReserveCheck)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeTranslatedNonCollinearPoints(makeNonCollinearPoints(), 0.1f, -0.05f, 0.025f).toGpu();
    auto target = makeNonCollinearPoints().toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveResidualStats(static_cast<int>(source.rows()));

    plapoint::gpu::resetIcpResidualStatsReserveCheckCountForTesting();
    const auto stats = plapoint::gpu::detail::computeIcpResidualStatsColumnMajorWithReservedWorkspace(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        2.0f,
        workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(source.rows()));
    EXPECT_EQ(plapoint::gpu::icpResidualStatsReserveCheckCountForTesting(), 0);
}

TEST(ICPGpuPathTest, ExactPointwiseStatsPredicatesSeparateSameBufferFromEqualityProbe)
{
    float source_points[3]{};
    float target_points[3]{};
    int correspondence_indices[1]{};

    const float* source = source_points;
    const float* same_source = source_points;
    const float* target = target_points;
    const int* indices = correspondence_indices;

    EXPECT_TRUE(plapoint::gpu::detail::canUseSameBufferExactPointwiseStats(
        source, 4, same_source, 4, nullptr));
    EXPECT_TRUE(plapoint::gpu::detail::canProbeExactPointwiseStats(
        source, 4, same_source, 4, 2.0f, nullptr));

    EXPECT_FALSE(plapoint::gpu::detail::canUseSameBufferExactPointwiseStats(
        source, 4, same_source, 4, indices));
    EXPECT_FALSE(plapoint::gpu::detail::canProbeExactPointwiseStats(
        source, 4, same_source, 4, 2.0f, indices));

    EXPECT_FALSE(plapoint::gpu::detail::canUseSameBufferExactPointwiseStats(
        source, 4, same_source, 3, nullptr));
    EXPECT_FALSE(plapoint::gpu::detail::canProbeExactPointwiseStats(
        source, 4, same_source, 3, std::numeric_limits<float>::infinity(), nullptr));

    EXPECT_FALSE(plapoint::gpu::detail::canUseSameBufferExactPointwiseStats(
        source, 4, target, 4, nullptr));
    EXPECT_FALSE(plapoint::gpu::detail::canProbeExactPointwiseStats(
        source, 4, target, 4, 2.0f, nullptr));
    EXPECT_TRUE(plapoint::gpu::detail::canProbeExactPointwiseStats(
        source, 4, target, 4, std::numeric_limits<float>::infinity(), nullptr));
}

TEST(ICPGpuPathTest, TransformedExactPointwiseStatsPredicateRequiresSameCountAndNoIndexOutput)
{
    int indices[4]{};
    const auto* target_points = reinterpret_cast<const float*>(0x1000);

    EXPECT_TRUE(plapoint::gpu::detail::canProbeTransformedExactPointwiseStats(
        4,
        target_points,
        4,
        nullptr));
    EXPECT_FALSE(plapoint::gpu::detail::canProbeTransformedExactPointwiseStats(
        4,
        nullptr,
        4,
        nullptr));
    EXPECT_FALSE(plapoint::gpu::detail::canProbeTransformedExactPointwiseStats(
        4,
        target_points,
        5,
        nullptr));
    EXPECT_FALSE(plapoint::gpu::detail::canProbeTransformedExactPointwiseStats(
        4,
        target_points,
        4,
        indices));
}

TEST(ICPGpuPathTest, CorrespondenceStatsSameBufferExactPointwiseAvoidsTargetCoordinateLoads)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto points = makeNonCollinearPoints().toGpu();
    auto copied_points = makeNonCollinearPoints().toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    const auto copied_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        points.data(),
        static_cast<int>(points.rows()),
        copied_points.data(),
        static_cast<int>(copied_points.rows()),
        std::numeric_limits<float>::infinity(),
        nullptr,
        workspace);

    EXPECT_EQ(copied_stats.active_count, static_cast<int>(points.rows()));
    EXPECT_EQ(
        plapoint::gpu::icpExactPointwiseTargetLoadCountForTesting(),
        static_cast<unsigned long long>(3 * points.rows()));

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseTargetLoadCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        points.data(),
        static_cast<int>(points.rows()),
        points.data(),
        static_cast<int>(points.rows()),
        std::numeric_limits<float>::infinity(),
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(points.rows()));
    EXPECT_EQ(stats.invalid_source_count, 0);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpExactPointwiseTargetLoadCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsRequestedIndicesKeepLowerDuplicateIndexForSameBuffer)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points_cpu(4, 3);
    points_cpu.setValue(0, 0, 0.0f);
    points_cpu.setValue(0, 1, 0.0f);
    points_cpu.setValue(0, 2, 0.0f);
    points_cpu.setValue(1, 0, 0.0f);
    points_cpu.setValue(1, 1, 0.0f);
    points_cpu.setValue(1, 2, 0.0f);
    points_cpu.setValue(2, 0, 1.0f);
    points_cpu.setValue(2, 1, 0.0f);
    points_cpu.setValue(2, 2, 0.0f);
    points_cpu.setValue(3, 0, 0.0f);
    points_cpu.setValue(3, 1, 1.0f);
    points_cpu.setValue(3, 2, 0.0f);

    auto points = points_cpu.toGpu();
    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(points.rows()));
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        points.data(),
        static_cast<int>(points.rows()),
        points.data(),
        static_cast<int>(points.rows()),
        2.0f,
        indices.get(),
        workspace);

    std::vector<int> host_indices(static_cast<std::size_t>(points.rows()), -1);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        host_indices.data(),
        indices.get(),
        host_indices.size() * sizeof(int),
        cudaMemcpyDeviceToHost));

    EXPECT_EQ(stats.active_count, static_cast<int>(points.rows()));
    EXPECT_EQ(stats.invalid_source_count, 0);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-8);
    ASSERT_EQ(host_indices.size(), 4u);
    EXPECT_EQ(host_indices[0], 0);
    EXPECT_EQ(host_indices[1], 0);
    EXPECT_EQ(host_indices[2], 2);
    EXPECT_EQ(host_indices[3], 3);
}

TEST(ICPGpuPathTest, ResidualStatsStopsNonSpatialScanAfterExactMatch)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(3, 3);
    target.setValue(0, 0, 0.0f); target.setValue(0, 1, 0.0f); target.setValue(0, 2, 0.0f);
    target.setValue(1, 0, 1.0f); target.setValue(1, 1, 0.0f); target.setValue(1, 2, 0.0f);
    target.setValue(2, 0, 2.0f); target.setValue(2, 1, 0.0f); target.setValue(2, 2, 0.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_EQ(stats.invalid_source_count, 0);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-8);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 1ull);
}

TEST(ICPGpuPathTest, FallbackStatsLaunchesTileBoundSpecializationsByRadius)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int bounded_target_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> bounded_target(bounded_target_count, 3);
    for (int i = 0; i < bounded_target_count; ++i)
    {
        bounded_target.setValue(i, 0, 1000.0f + static_cast<float>(i));
        bounded_target.setValue(i, 1, 1000.0f);
        bounded_target.setValue(i, 2, 1000.0f);
    }
    bounded_target.setValue(0, 0, 0.0f);
    bounded_target.setValue(0, 1, 0.0f);
    bounded_target.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> unbounded_target(2, 3);
    unbounded_target.setValue(0, 0, 0.0f);
    unbounded_target.setValue(0, 1, 0.0f);
    unbounded_target.setValue(0, 2, 0.0f);
    unbounded_target.setValue(1, 0, 1.0f);
    unbounded_target.setValue(1, 1, 0.0f);
    unbounded_target.setValue(1, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> identity(4, 4);
    identity.fill(0.0f);
    identity.setValue(0, 0, 1.0f);
    identity.setValue(1, 1, 1.0f);
    identity.setValue(2, 2, 1.0f);
    identity.setValue(3, 3, 1.0f);

    auto source_gpu = source.toGpu();
    auto bounded_target_gpu = bounded_target.toGpu();
    auto unbounded_target_gpu = unbounded_target.toGpu();
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        bounded_target_gpu.data(),
        static_cast<int>(bounded_target_gpu.rows()),
        0.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);

    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        unbounded_target_gpu.data(),
        static_cast<int>(unbounded_target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 1);

    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    const auto transformed_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        bounded_target_gpu.data(),
        static_cast<int>(bounded_target_gpu.rows()),
        0.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transformed_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
}

TEST(ICPGpuPathTest, FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    target.setValue(0, 0, 0.0f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);
    for (int i = 1; i < target_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i));
        target.setValue(i, 1, 1000.0f);
        target.setValue(i, 2, 1000.0f);
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> identity(4, 4);
    identity.fill(0.0f);
    identity.setValue(0, 0, 1.0f);
    identity.setValue(1, 1, 1.0f);
    identity.setValue(2, 2, 1.0f);
    identity.setValue(3, 3, 1.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        nullptr,
        workspace);
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 1ull);

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        workspace);
    EXPECT_EQ(residual_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 1ull);

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto transformed_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transformed_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 1ull);
}

TEST(ICPGpuPathTest, FallbackStatsSkipTargetTileLoadsWhenBoundsRejectWholeBlock)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    for (int i = 0; i < target_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i));
        target.setValue(i, 1, 1000.0f);
        target.setValue(i, 2, 1000.0f);
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> identity(4, 4);
    identity.fill(0.0f);
    identity.setValue(0, 0, 1.0f);
    identity.setValue(1, 1, 1.0f);
    identity.setValue(2, 2, 1.0f);
    identity.setValue(3, 3, 1.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        indices.get(),
        workspace);
    std::vector<int> host_indices(static_cast<std::size_t>(source.rows()), 0);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        host_indices.data(),
        indices.get(),
        host_indices.size() * sizeof(int),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(stats.active_count, 0);
    ASSERT_EQ(host_indices.size(), 1u);
    EXPECT_EQ(host_indices[0], -1);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 0ull);

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        workspace);
    EXPECT_EQ(residual_stats.active_count, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 0ull);

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto transformed_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.0f,
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transformed_stats.active_count, 0);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, FallbackStatsSkipUnboundedTargetTileLoadsWhenBlockHasNoValidSources)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, nan);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 257;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    for (int i = 0; i < target_count; ++i)
    {
        target.setValue(i, 0, static_cast<float>(i));
        target.setValue(i, 1, 0.0f);
        target.setValue(i, 2, 0.0f);
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> identity(4, 4);
    identity.fill(0.0f);
    identity.setValue(0, 0, 1.0f);
    identity.setValue(1, 1, 1.0f);
    identity.setValue(2, 2, 1.0f);
    identity.setValue(3, 3, 1.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    auto identity_gpu = identity.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source.rows(), 3);
    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        indices.get(),
        workspace);
    std::vector<int> host_indices(static_cast<std::size_t>(source.rows()), 0);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        host_indices.data(),
        indices.get(),
        host_indices.size() * sizeof(int),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(stats.active_count, 0);
    EXPECT_EQ(stats.invalid_source_count, 1);
    ASSERT_EQ(host_indices.size(), 1u);
    EXPECT_EQ(host_indices[0], -1);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 0ull);

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto residual_stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        workspace);
    EXPECT_EQ(residual_stats.active_count, 0);
    EXPECT_EQ(residual_stats.invalid_source_count, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 0ull);

    plapoint::gpu::resetIcpTargetTileLoadCountForTesting();
    const auto transformed_stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        identity_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        std::numeric_limits<float>::infinity(),
        output_gpu.data(),
        workspace);
    EXPECT_EQ(transformed_stats.active_count, 0);
    EXPECT_EQ(transformed_stats.invalid_source_count, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetTileLoadCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsStopsSpatialGridAfterExactMatchWhenIndicesOmitted)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(5, 3);
    target.setValue(0, 0, 0.0f);  target.setValue(0, 1, 0.0f);  target.setValue(0, 2, 0.0f);
    target.setValue(1, 0, 0.0f);  target.setValue(1, 1, -0.5f); target.setValue(1, 2, 0.0f);
    target.setValue(2, 0, -0.5f); target.setValue(2, 1, 0.0f);  target.setValue(2, 2, 0.0f);
    target.setValue(3, 0, 0.0f);  target.setValue(3, 1, -0.5f); target.setValue(3, 2, -0.5f);
    target.setValue(4, 0, -0.5f); target.setValue(4, 1, -0.5f); target.setValue(4, 2, 0.0f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(stats.active_count, 1);
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-12);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);

    plapoint::gpu::DeviceBuffer<int> indices(static_cast<std::size_t>(source.rows()));
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    const auto indexed_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        indices.get(),
        workspace);

    std::vector<int> host_indices(static_cast<std::size_t>(source.rows()), -1);
    PLAPOINT_CHECK_CUDA(cudaMemcpy(
        host_indices.data(),
        indices.get(),
        host_indices.size() * sizeof(int),
        cudaMemcpyDeviceToHost));

    EXPECT_EQ(indexed_stats.active_count, 1);
    EXPECT_NEAR(indexed_stats.residual_sq_sum, 0.0, 1.0e-12);
    ASSERT_EQ(host_indices.size(), 1u);
    EXPECT_EQ(host_indices[0], 0);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.5f);
    source.setValue(0, 1, 0.5f);
    source.setValue(0, 2, 0.5f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(2, 3);
    target.setValue(0, 0, 1.5f);
    target.setValue(0, 1, 0.5f);
    target.setValue(0, 2, 0.5f);
    target.setValue(1, 0, -0.5f);
    target.setValue(1, 1, 0.5f);
    target.setValue(1, 2, 0.5f);
    target = padTargetWithNonFiniteRows(target);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    plapoint::gpu::DeviceBuffer<int> indices(1);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        indices.get(),
        workspace);

    int host_index = -1;
    PLAPOINT_CHECK_CUDA(cudaMemcpy(&host_index, indices.get(), sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(stats.active_count, 1);
    EXPECT_EQ(host_index, 0);

    plapoint::gpu::resetIcpTargetIndexLoadCountForTesting();
    const auto unindexed_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);

    EXPECT_EQ(unindexed_stats.active_count, 1);
    EXPECT_NEAR(unindexed_stats.tgt_centroid[0], 1.5, 1.0e-6);
    EXPECT_EQ(plapoint::gpu::icpTargetIndexLoadCountForTesting(), 1ull);
}

TEST(ICPGpuPathTest, CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source(1, 3);
    source.setValue(0, 0, 0.0f);
    source.setValue(0, 1, 0.0f);
    source.setValue(0, 2, 0.0f);

    constexpr int target_count = 256;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(target_count, 3);
    target.setValue(0, 0, 0.5f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);
    for (int i = 1; i < target_count; ++i)
    {
        target.setValue(i, 0, 1000.0f + static_cast<float>(i * 3));
        target.setValue(i, 1, 1000.0f);
        target.setValue(i, 2, 1000.0f);
    }
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> second_target(target_count, 3);
    second_target.setValue(0, 0, 0.5f);
    second_target.setValue(0, 1, 0.0f);
    second_target.setValue(0, 2, 0.0f);
    for (int i = 1; i < target_count; ++i)
    {
        second_target.setValue(i, 0, 2000.0f + static_cast<float>(i * 3));
        second_target.setValue(i, 1, 1000.0f);
        second_target.setValue(i, 2, 1000.0f);
    }

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();
    auto second_target_gpu = second_target.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const auto first_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    const auto second_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
        nullptr,
        workspace);
    const auto changed_radius_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        nullptr,
        workspace);
    const auto changed_target_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        second_target_gpu.data(),
        static_cast<int>(second_target_gpu.rows()),
        2.0f,
        nullptr,
        workspace);

    EXPECT_EQ(first_stats.active_count, 1);
    EXPECT_EQ(second_stats.active_count, 1);
    EXPECT_EQ(changed_radius_stats.active_count, 1);
    EXPECT_EQ(changed_target_stats.active_count, 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 3);
}

TEST(ICPGpuPathTest, CorrespondenceStatsWorkspaceReusesDeviceStorage)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeNonCollinearPoints().toGpu();
    auto target = makeNonCollinearPoints().toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    const auto first_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        std::numeric_limits<float>::infinity(),
        nullptr,
        workspace);
    auto* first_partial_storage = workspace.partialStorage();
    auto* first_stats_storage = workspace.statsStorage();
    const int first_partial_capacity = workspace.partialCapacity();

    const auto second_stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source.data(),
        static_cast<int>(source.rows()),
        target.data(),
        static_cast<int>(target.rows()),
        std::numeric_limits<float>::infinity(),
        nullptr,
        workspace);

    EXPECT_EQ(first_stats.active_count, 4);
    EXPECT_EQ(second_stats.active_count, 4);
    EXPECT_NE(first_partial_storage, nullptr);
    EXPECT_NE(first_stats_storage, nullptr);
    EXPECT_EQ(workspace.partialStorage(), first_partial_storage);
    EXPECT_EQ(workspace.statsStorage(), first_stats_storage);
    EXPECT_EQ(workspace.partialCapacity(), first_partial_capacity);
}

TEST(ICPGpuPathTest, CorrespondenceStatsWorkspaceCanReserveCompactAlignmentStepResult)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpCorrespondenceStatsWorkspace full_workspace;
    full_workspace.reserve(4);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace compact_workspace;
    compact_workspace.reserveAlignmentStep(4);

    EXPECT_NE(compact_workspace.partialStorage(), nullptr);
    EXPECT_NE(compact_workspace.statsStorage(), nullptr);
    EXPECT_EQ(compact_workspace.partialCapacity(), full_workspace.partialCapacity());
    EXPECT_LT(compact_workspace._stats_storage.size(), full_workspace._stats_storage.size());
}

TEST(ICPGpuPathTest, TargetSpatialGridSortedCoordinateStorageUsesScalarWidth)
{
    constexpr int target_count = 11;

    EXPECT_EQ(
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<float>(target_count),
        static_cast<std::size_t>(target_count) * sizeof(float));
    EXPECT_EQ(
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<double>(target_count),
        static_cast<std::size_t>(target_count) * sizeof(double));
    EXPECT_LT(
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<float>(target_count),
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<double>(target_count));

    EXPECT_FALSE(plapoint::gpu::detail::targetSpatialGridCoordinateStorageNeedsReserve(
        target_count,
        sizeof(float),
        target_count,
        sizeof(float)));
    EXPECT_TRUE(plapoint::gpu::detail::targetSpatialGridCoordinateStorageNeedsReserve(
        target_count,
        sizeof(float),
        target_count,
        sizeof(double)));
    EXPECT_TRUE(plapoint::gpu::detail::targetSpatialGridCoordinateStorageNeedsReserve(
        target_count,
        sizeof(double),
        target_count + 1,
        sizeof(double)));
    EXPECT_TRUE(plapoint::gpu::detail::targetSpatialGridCoordinateStorageNeedsReserve(
        target_count,
        std::size_t{0},
        target_count,
        sizeof(float)));
}

TEST(ICPGpuPathTest, TargetSpatialGridCacheMatchRequiresCoordinateStorageWidth)
{
    constexpr int target_count = 11;
    const void* target_points = reinterpret_cast<const void*>(0x1000);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    workspace.markTargetSpatialGridCache(target_points, target_count, 1.0, 1);

    EXPECT_FALSE(workspace.targetSpatialGridCacheMatches(target_points, target_count, 1.0));
    EXPECT_FALSE(workspace.targetSpatialGridCacheMatchesForScalar<float>(target_points, target_count, 1.0));
    EXPECT_FALSE(workspace.targetSpatialGridCacheMatchesForScalar<double>(target_points, target_count, 1.0));
}

TEST(ICPGpuPathTest, FinalMetricsSnapshotPredicateUsesScalarSpatialGridCacheWidth)
{
    constexpr int target_count = 11;
    const auto* target_points = reinterpret_cast<const float*>(0x1000);

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setMaxCorrespondenceDistance(1.0f);
    icp._gpu_stats_workspace.markTargetSpatialGridCache(target_points, target_count, 1.0, 2);
    icp._gpu_stats_workspace._target_spatial_grid_coordinate_value_bytes = sizeof(float);

    EXPECT_TRUE(icp.gpuFinalMetricsCanUseCachedTargetSpatialGridSnapshot(target_points, target_count));

    icp._gpu_stats_workspace._target_spatial_grid_coordinate_value_bytes = sizeof(double);
    EXPECT_FALSE(icp.gpuFinalMetricsCanUseCachedTargetSpatialGridSnapshot(target_points, target_count));
}

TEST(ICPGpuPathTest, TargetSpatialGridWorkspaceReservesFloatSizedSortedCoordinates)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    constexpr int target_count = 11;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    workspace.reserveTargetSpatialGridForScalar<float>(target_count);
    EXPECT_EQ(
        workspace._target_spatial_grid_sorted_x_storage.size(),
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<float>(target_count));
    EXPECT_EQ(
        workspace._target_spatial_grid_sorted_y_storage.size(),
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<float>(target_count));
    EXPECT_EQ(
        workspace._target_spatial_grid_sorted_z_storage.size(),
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<float>(target_count));

    workspace.markTargetSpatialGridCache(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0,
        1);
    ASSERT_TRUE(workspace.targetSpatialGridCacheMatchesForScalar<float>(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0));
    EXPECT_FALSE(workspace.targetSpatialGridCacheMatches(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0));
    EXPECT_FALSE(workspace.targetSpatialGridCacheMatchesForScalar<double>(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0));

    workspace.reserveTargetSpatialGridForScalar<double>(target_count);
    EXPECT_EQ(
        workspace._target_spatial_grid_sorted_x_storage.size(),
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<double>(target_count));
    EXPECT_EQ(
        workspace._target_spatial_grid_sorted_y_storage.size(),
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<double>(target_count));
    EXPECT_EQ(
        workspace._target_spatial_grid_sorted_z_storage.size(),
        plapoint::gpu::detail::targetSpatialGridSortedCoordinateByteCount<double>(target_count));
    EXPECT_FALSE(workspace.targetSpatialGridCacheMatches(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0));

    workspace.markTargetSpatialGridCache(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0,
        1);
    EXPECT_TRUE(workspace.targetSpatialGridCacheMatches(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0));
    EXPECT_TRUE(workspace.targetSpatialGridCacheMatchesForScalar<double>(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0));
    EXPECT_FALSE(workspace.targetSpatialGridCacheMatchesForScalar<float>(
        reinterpret_cast<const void*>(0x1000),
        target_count,
        1.0));
}

TEST(ICPGpuPathTest, FloatAlignmentStepWorkspaceReservesFloatSizedResultStorage)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    workspace.reserveFloatAlignmentStep(4);

    EXPECT_NE(workspace.partialStorage(), nullptr);
    EXPECT_NE(workspace.statsStorage(), nullptr);
    EXPECT_EQ(workspace._stats_storage.size(), plapoint::gpu::icpFloatAlignmentStepRawResultByteCountForTesting());
    EXPECT_EQ(workspace.hostResultStorageCapacity(),
              plapoint::gpu::icpFloatAlignmentStepRawResultByteCountForTesting());
    EXPECT_LT(workspace.hostResultStorageCapacity(),
              plapoint::gpu::icpDoubleAlignmentStepRawResultByteCountForTesting());
}

TEST(ICPGpuPathTest, AlignmentStepWorkspaceReusesPinnedHostResultStorage)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpHostResultStorageAllocationCountForTesting();
    workspace.reserveAlignmentStep(4);
    auto* first_host_result = workspace.hostResultStorage();

    ASSERT_NE(first_host_result, nullptr);
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);

    workspace.reserveAlignmentStep(4);
    EXPECT_EQ(workspace.hostResultStorage(), first_host_result);
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, StatsWorkspaceReusesPinnedHostResultStorage)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpHostResultStorageAllocationCountForTesting();
    workspace.reserve(4);
    auto* first_host_result = workspace.hostResultStorage();
    const auto first_capacity = workspace.hostResultStorageCapacity();

    ASSERT_NE(first_host_result, nullptr);
    EXPECT_GT(first_capacity, std::size_t{0});
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);

    workspace.reserve(4);
    EXPECT_EQ(workspace.hostResultStorage(), first_host_result);
    EXPECT_EQ(workspace.hostResultStorageCapacity(), first_capacity);
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, CorrespondenceStatsWorkspaceCanReserveCompactResidualStats)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpCorrespondenceStatsWorkspace full_workspace;
    full_workspace.reserve(4);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace residual_workspace;
    residual_workspace.reserveResidualStats(4);

    EXPECT_NE(residual_workspace.partialStorage(), nullptr);
    EXPECT_NE(residual_workspace.statsStorage(), nullptr);
    EXPECT_EQ(residual_workspace.partialCapacity(), full_workspace.partialCapacity());
    EXPECT_LT(residual_workspace._partial_storage.size(), full_workspace._partial_storage.size());
    EXPECT_LT(residual_workspace._stats_storage.size(), full_workspace._stats_storage.size());
}

TEST(ICPGpuPathTest, ResidualStatsWorkspaceReusesPinnedHostResultStorage)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpHostResultStorageAllocationCountForTesting();
    workspace.reserveResidualStats(4);
    auto* first_host_result = workspace.hostResultStorage();
    const auto first_capacity = workspace.hostResultStorageCapacity();

    ASSERT_NE(first_host_result, nullptr);
    EXPECT_GT(first_capacity, std::size_t{0});
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);

    workspace.reserveResidualStats(4);
    EXPECT_EQ(workspace.hostResultStorage(), first_host_result);
    EXPECT_EQ(workspace.hostResultStorageCapacity(), first_capacity);
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);
}

#endif // PLAPOINT_WITH_CUDA
