#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <plamatrix/plamatrix.h>
#include <plapoint/gpu/cuda_check.h>

#define private public
#include <plapoint/core/point_cloud.h>
#include <plapoint/registration/icp.h>
#undef private

namespace
{

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeNonCollinearPoints()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 0.0f); points.setValue(2, 1, 1.0f); points.setValue(2, 2, 0.0f);
    points.setValue(3, 0, 0.0f); points.setValue(3, 1, 0.0f); points.setValue(3, 2, 1.0f);
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeCollinearPoints()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 2.0f); points.setValue(2, 1, 0.0f); points.setValue(2, 2, 0.0f);
    points.setValue(3, 0, 3.0f); points.setValue(3, 1, 0.0f); points.setValue(3, 2, 0.0f);
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> multiplyCpu4x4(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& A,
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& B)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> C(4, 4);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                sum += static_cast<double>(A.getValue(row, k)) * static_cast<double>(B.getValue(k, col));
            }
            C.setValue(row, col, static_cast<float>(sum));
        }
    }
    return C;
}

plapoint::gpu::IcpCorrespondenceStats<float> makeMatchedStats(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& source,
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& target)
{
    plapoint::gpu::IcpCorrespondenceStats<float> stats;
    stats.active_count = static_cast<int>(source.rows());

    double src_sum[3]{};
    double tgt_sum[3]{};
    double cross_sum[9]{};
    double src_outer_sum[9]{};
    double tgt_outer_sum[9]{};

    for (plamatrix::Index row = 0; row < source.rows(); ++row)
    {
        const double src_values[3]{
            static_cast<double>(source.getValue(row, 0)),
            static_cast<double>(source.getValue(row, 1)),
            static_cast<double>(source.getValue(row, 2))};
        const double tgt_values[3]{
            static_cast<double>(target.getValue(row, 0)),
            static_cast<double>(target.getValue(row, 1)),
            static_cast<double>(target.getValue(row, 2))};

        for (int r = 0; r < 3; ++r)
        {
            src_sum[r] += src_values[r];
            tgt_sum[r] += tgt_values[r];
            const double residual = src_values[r] - tgt_values[r];
            stats.residual_sq_sum += residual * residual;
            for (int c = 0; c < 3; ++c)
            {
                cross_sum[r * 3 + c] += src_values[r] * tgt_values[c];
                src_outer_sum[r * 3 + c] += src_values[r] * src_values[c];
                tgt_outer_sum[r * 3 + c] += tgt_values[r] * tgt_values[c];
            }
        }
    }

    const double inv_count = 1.0 / static_cast<double>(stats.active_count);
    for (int c = 0; c < 3; ++c)
    {
        stats.src_centroid[c] = src_sum[c] * inv_count;
        stats.tgt_centroid[c] = tgt_sum[c] * inv_count;
    }
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            const int idx = r * 3 + c;
            stats.cross_covariance[idx] = cross_sum[idx] - src_sum[r] * tgt_sum[c] * inv_count;
            stats.src_covariance[idx] = src_outer_sum[idx] - src_sum[r] * src_sum[c] * inv_count;
            stats.tgt_covariance[idx] = tgt_outer_sum[idx] - tgt_sum[r] * tgt_sum[c] * inv_count;
        }
    }
    return stats;
}

} // namespace

namespace plapoint
{
namespace gpu
{

void resetIcpCorrespondenceStatsCallCountForTesting();
int icpCorrespondenceStatsCallCountForTesting();

} // namespace gpu
} // namespace plapoint

TEST(ICPGpuPathTest, MultiplyTransform4x4UsesColumnMajorTransformComposition)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> A(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> B(4, 4);
    A.fill(0.0f);
    B.fill(0.0f);

    A.setValue(0, 0, 0.0f);  A.setValue(0, 1, -1.0f); A.setValue(0, 2, 0.0f); A.setValue(0, 3, 2.0f);
    A.setValue(1, 0, 1.0f);  A.setValue(1, 1, 0.0f);  A.setValue(1, 2, 0.0f); A.setValue(1, 3, -1.0f);
    A.setValue(2, 0, 0.0f);  A.setValue(2, 1, 0.0f);  A.setValue(2, 2, 1.0f); A.setValue(2, 3, 0.5f);
    A.setValue(3, 0, 0.0f);  A.setValue(3, 1, 0.0f);  A.setValue(3, 2, 0.0f); A.setValue(3, 3, 1.0f);

    B.setValue(0, 0, 1.0f);  B.setValue(0, 1, 0.0f);  B.setValue(0, 2, 0.0f); B.setValue(0, 3, -3.0f);
    B.setValue(1, 0, 0.0f);  B.setValue(1, 1, 1.0f);  B.setValue(1, 2, 0.0f); B.setValue(1, 3, 4.0f);
    B.setValue(2, 0, 0.0f);  B.setValue(2, 1, 0.0f);  B.setValue(2, 2, 1.0f); B.setValue(2, 3, 1.5f);
    B.setValue(3, 0, 0.0f);  B.setValue(3, 1, 0.0f);  B.setValue(3, 2, 0.0f); B.setValue(3, 3, 1.0f);

    auto A_gpu = A.toGpu();
    auto B_gpu = B.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> C_gpu(4, 4);

    plapoint::gpu::multiplyTransform4x4(A_gpu.data(), B_gpu.data(), C_gpu.data());
    auto C = C_gpu.toCpu();
    auto expected = multiplyCpu4x4(A, B);

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(C.getValue(row, col), expected.getValue(row, col), 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, StepTransformFromStatsWritesDeviceTransform)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeNonCollinearPoints();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> expected(4, 4);
    expected.fill(0.0f);
    expected.setValue(0, 0, 0.0f);  expected.setValue(0, 1, -1.0f); expected.setValue(0, 2, 0.0f); expected.setValue(0, 3, 2.0f);
    expected.setValue(1, 0, 1.0f);  expected.setValue(1, 1, 0.0f);  expected.setValue(1, 2, 0.0f); expected.setValue(1, 3, -1.0f);
    expected.setValue(2, 0, 0.0f);  expected.setValue(2, 1, 0.0f);  expected.setValue(2, 2, 1.0f); expected.setValue(2, 3, 0.5f);
    expected.setValue(3, 0, 0.0f);  expected.setValue(3, 1, 0.0f);  expected.setValue(3, 2, 0.0f); expected.setValue(3, 3, 1.0f);

    auto target = plamatrix::transformPoints(expected, source);
    const auto stats = makeMatchedStats(source, target);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    const auto result = plapoint::gpu::computeIcpStepTransformFromStats(stats, step_gpu.data());
    auto step = step_gpu.toCpu();

    float expected_delta = 0.0f;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            expected_delta += std::abs(expected.getValue(row, col) - (row == col ? 1.0f : 0.0f));
        }
        expected_delta += std::abs(expected.getValue(row, 3));
    }

    EXPECT_NEAR(result.delta, expected_delta, 1.0e-5f);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(step.getValue(row, col), expected.getValue(row, col), 1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, TransformPointsColumnMajorWritesCallerOwnedOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto points_cpu = makeNonCollinearPoints();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> T(4, 4);
    T.fill(0.0f);
    T.setValue(0, 0, 0.0f);  T.setValue(0, 1, -1.0f); T.setValue(0, 2, 0.0f); T.setValue(0, 3, 2.0f);
    T.setValue(1, 0, 1.0f);  T.setValue(1, 1, 0.0f);  T.setValue(1, 2, 0.0f); T.setValue(1, 3, -1.0f);
    T.setValue(2, 0, 0.0f);  T.setValue(2, 1, 0.0f);  T.setValue(2, 2, 1.0f); T.setValue(2, 3, 0.5f);
    T.setValue(3, 0, 0.0f);  T.setValue(3, 1, 0.0f);  T.setValue(3, 2, 0.0f); T.setValue(3, 3, 1.0f);

    auto points_gpu = points_cpu.toGpu();
    auto T_gpu = T.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> transformed_gpu(points_cpu.rows(), 3);

    plapoint::gpu::transformPointsColumnMajor(
        T_gpu.data(),
        points_gpu.data(),
        static_cast<int>(points_cpu.rows()),
        transformed_gpu.data());

    auto transformed = transformed_gpu.toCpu();
    auto expected = plamatrix::transformPoints(T, points_cpu);
    for (plamatrix::Index row = 0; row < points_cpu.rows(); ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_NEAR(transformed.getValue(row, col), expected.getValue(row, col), 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, CorrespondenceStatsAllowOmittedIndexOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source = makeNonCollinearPoints().toGpu();
    auto target = makeNonCollinearPoints().toGpu();

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

TEST(ICPGpuPathTest, AlignDoesNotPopulateGpuPointCpuCaches)
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

    ASSERT_EQ(source->_points_cpu_cache.get(), nullptr);
    ASSERT_EQ(target->_points_cpu_cache.get(), nullptr);

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

TEST(ICPGpuPathTest, FinalTransformationDeviceIsAvailableAfterGpuAlign)
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
    icp.setMaxIterations(3);

    GpuCloud output;
    icp.align(output);

    const auto& transform_gpu = icp.getFinalTransformationDevice();
    auto transform_cpu = transform_gpu.toCpu();

    ASSERT_EQ(transform_cpu.rows(), 4);
    ASSERT_EQ(transform_cpu.cols(), 4);
    EXPECT_NEAR(transform_cpu.getValue(0, 0), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(1, 1), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(2, 2), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(3, 3), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(0, 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(1, 3), 0.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(2, 3), 0.0f, 1.0e-5f);
}

TEST(ICPGpuPathTest, AlignSkipsFinalStatsForNonTerminalGpuIterations)
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
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);

    plapoint::gpu::resetIcpCorrespondenceStatsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 3);
    EXPECT_EQ(output.size(), source->size());
}

#endif // PLAPOINT_WITH_CUDA
