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

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedNonCollinearPoints(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& source,
    float tx,
    float ty,
    float tz)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> transform(4, 4);
    transform.fill(0.0f);
    transform.setValue(0, 0, 1.0f);
    transform.setValue(1, 1, 1.0f);
    transform.setValue(2, 2, 1.0f);
    transform.setValue(3, 3, 1.0f);
    transform.setValue(0, 3, tx);
    transform.setValue(1, 3, ty);
    transform.setValue(2, 3, tz);
    return plamatrix::transformPoints(transform, source);
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
void resetIcpResidualStatsCallCountForTesting();
int icpResidualStatsCallCountForTesting();
void resetIcpFirstStatsSourcePointerForTesting();
const void* icpFirstStatsSourcePointerForTesting();
void resetIcpFullDistanceEvaluationCountForTesting();
unsigned long long icpFullDistanceEvaluationCountForTesting();
void resetIcpTargetCandidateVisitCountForTesting();
unsigned long long icpTargetCandidateVisitCountForTesting();
void resetIcpTargetIndexLoadCountForTesting();
unsigned long long icpTargetIndexLoadCountForTesting();
void resetIcpSortedTargetCoordinateLoadCountForTesting();
unsigned long long icpSortedTargetCoordinateLoadCountForTesting();
void resetIcpStepTransformInputCopyCountForTesting();
int icpStepTransformInputCopyCountForTesting();
void resetIcpExactPointwiseStepCallCountForTesting();
int icpExactPointwiseStepCallCountForTesting();
void resetIcpExactPointwiseTargetLoadCountForTesting();
unsigned long long icpExactPointwiseTargetLoadCountForTesting();
void resetIcpRawStatsStepKernelLaunchCountForTesting();
int icpRawStatsStepKernelLaunchCountForTesting();
void resetIcpAlignmentStepCallCountForTesting();
int icpAlignmentStepCallCountForTesting();
void resetIcpAlignmentStepReserveCountForTesting();
int icpAlignmentStepReserveCountForTesting();
void resetIcpHostSynchronizationCountForTesting();
int icpHostSynchronizationCountForTesting();
void resetIcpTargetTileBoundComputationCountForTesting();
unsigned long long icpTargetTileBoundComputationCountForTesting();
void resetIcpTargetTileLoadCountForTesting();
unsigned long long icpTargetTileLoadCountForTesting();
void resetIcpFallbackTileBoundKernelLaunchCountForTesting();
int icpFallbackTileBoundKernelLaunchCountForTesting();
void resetIcpFallbackUnboundedKernelLaunchCountForTesting();
int icpFallbackUnboundedKernelLaunchCountForTesting();
void resetIcpTargetSpatialGridBuildCountForTesting();
int icpTargetSpatialGridBuildCountForTesting();
void resetIcpGridCellLookupCountForTesting();
unsigned long long icpGridCellLookupCountForTesting();
void resetIcpLastTransformOutputPointerForTesting();
const void* icpLastTransformOutputPointerForTesting();
void resetIcpTransformPointsCallCountForTesting();
int icpTransformPointsCallCountForTesting();
void resetIcpTransformMultiplyCallCountForTesting();
int icpTransformMultiplyCallCountForTesting();
void resetIcpIdentityTransformWriteCountForTesting();
int icpIdentityTransformWriteCountForTesting();

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

TEST(ICPGpuPathTest, MultiplyTransform4x4AsyncUsesCallerStream)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> A(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> B(4, 4);
    A.fill(0.0f);
    B.fill(0.0f);

    A.setValue(0, 0, 1.0f); A.setValue(0, 3, 1.5f);
    A.setValue(1, 1, 1.0f); A.setValue(1, 3, -2.0f);
    A.setValue(2, 2, 1.0f); A.setValue(2, 3, 0.25f);
    A.setValue(3, 3, 1.0f);

    B.setValue(0, 0, 0.0f);  B.setValue(0, 1, -1.0f); B.setValue(0, 3, 3.0f);
    B.setValue(1, 0, 1.0f);  B.setValue(1, 1, 0.0f);  B.setValue(1, 3, 4.0f);
    B.setValue(2, 2, 1.0f);  B.setValue(2, 3, -1.0f);
    B.setValue(3, 3, 1.0f);

    auto A_gpu = A.toGpu();
    auto B_gpu = B.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> C_gpu(4, 4);

    cudaStream_t stream{};
    PLAPOINT_CHECK_CUDA(cudaStreamCreate(&stream));
    plapoint::gpu::multiplyTransform4x4Async(A_gpu.data(), B_gpu.data(), C_gpu.data(), stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));

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

TEST(ICPGpuPathTest, SetIdentityTransform4x4WritesColumnMajorIdentity)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> identity_gpu(4, 4);
    plapoint::gpu::setIdentityTransform4x4(identity_gpu.data());
    auto identity = identity_gpu.toCpu();

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(identity.getValue(row, col), expected, 1.0e-6f);
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

TEST(ICPGpuPathTest, StepTransformWorkspaceCanReserveOnlyResultStorage)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpStepTransformWorkspace workspace;
    workspace.reserveResult();

    EXPECT_EQ(workspace.inputStorage(), nullptr);
    EXPECT_NE(workspace.resultStorage(), nullptr);
}

TEST(ICPGpuPathTest, AlignmentStepCompactResultMatchesFullStatsStepResult)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeNonCollinearPoints();
    auto target_cpu = makeTranslatedNonCollinearPoints(source_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace full_stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace full_step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> full_step_gpu(4, 4);
    const auto full_result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        full_stats_workspace,
        full_step_gpu.data(),
        full_step_workspace);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace compact_stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> compact_step_gpu(4, 4);
    const auto compact_result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        compact_stats_workspace,
        compact_step_gpu.data());

    EXPECT_EQ(compact_result.active_count, full_result.stats.active_count);
    EXPECT_EQ(compact_result.invalid_source_count, full_result.stats.invalid_source_count);
    EXPECT_NEAR(compact_result.residual_sq_sum, full_result.stats.residual_sq_sum, 1.0e-12);
    EXPECT_EQ(compact_result.src_has_non_collinear_geometry, full_result.stats.src_has_non_collinear_geometry);
    EXPECT_EQ(compact_result.tgt_has_non_collinear_geometry, full_result.stats.tgt_has_non_collinear_geometry);
    EXPECT_EQ(compact_result.step_valid, full_result.step_valid);
    EXPECT_NEAR(compact_result.step.delta, full_result.step.delta, 1.0e-6f);

    auto full_step = full_step_gpu.toCpu();
    auto compact_step = compact_step_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(compact_step.getValue(row, col), full_step.getValue(row, col), 1.0e-6f);
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

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(3, 3);
    target.setValue(0, 0, 0.5f);   target.setValue(0, 1, 0.0f);    target.setValue(0, 2, 0.0f);
    target.setValue(1, 0, 100.0f); target.setValue(1, 1, 0.0f);    target.setValue(1, 2, 0.0f);
    target.setValue(2, 0, 0.0f);   target.setValue(2, 1, -200.0f); target.setValue(2, 2, 0.0f);

    auto source_gpu = source.toGpu();
    auto target_gpu = target.toGpu();

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        1.0f,
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
    auto* first_grid_sorted_x = workspace.targetSpatialGridSortedXStorage();
    auto* first_grid_sorted_y = workspace.targetSpatialGridSortedYStorage();
    auto* first_grid_sorted_z = workspace.targetSpatialGridSortedZStorage();
    auto* first_grid_cell_starts = workspace.targetSpatialGridCellStartsStorage();
    auto* first_grid_cell_counts = workspace.targetSpatialGridCellCountsStorage();
    EXPECT_NE(first_grid_keys, nullptr);
    EXPECT_NE(first_grid_unique_keys, nullptr);
    EXPECT_NE(first_grid_indices, nullptr);
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

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(2, 3);
    target.setValue(0, 0, 0.2f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);
    target.setValue(1, 0, 1.6f);
    target.setValue(1, 1, 0.0f);
    target.setValue(1, 2, 0.0f);

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

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target(2, 3);
    target.setValue(0, 0, 0.1f);
    target.setValue(0, 1, 0.0f);
    target.setValue(0, 2, 0.0f);
    target.setValue(1, 0, 0.25f);
    target.setValue(1, 1, 0.9f);
    target.setValue(1, 2, 0.9f);

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
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 1ull);

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
    EXPECT_GT(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 1ull);
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
    EXPECT_EQ(plapoint::gpu::icpTargetIndexLoadCountForTesting(), 2ull);
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

TEST(ICPGpuPathTest, AlignDoesNotPopulateGpuPointCpuCaches)
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

TEST(ICPGpuPathTest, AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls)
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

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignReusesGpuWorkspacesAcrossRepeatedCalls)
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

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    icp.align(*source);

    auto* first_partial_storage = icp._gpu_stats_workspace.partialStorage();
    auto* first_stats_storage = icp._gpu_stats_workspace.statsStorage();
    auto* first_grid_keys = icp._gpu_stats_workspace.targetSpatialGridKeysStorage();
    auto* first_grid_indices = icp._gpu_stats_workspace.targetSpatialGridIndicesStorage();
    auto* first_grid_sorted_x = icp._gpu_stats_workspace.targetSpatialGridSortedXStorage();
    auto* first_grid_sorted_y = icp._gpu_stats_workspace.targetSpatialGridSortedYStorage();
    auto* first_grid_sorted_z = icp._gpu_stats_workspace.targetSpatialGridSortedZStorage();
    auto* first_acc_transform = icp._gpu_T_acc->data();
    ASSERT_NE(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    auto* first_points_a = icp._gpu_points_a->data();
    const int first_partial_capacity = icp._gpu_stats_workspace.partialCapacity();
    EXPECT_EQ(icp._gpu_T_step, nullptr);

    auto second_source_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto second_source = std::make_shared<GpuCloud>(second_source_cpu->toGpu());
    icp.setInputSource(second_source);
    icp.align(*second_source);

    ASSERT_NE(icp._gpu_T_step, nullptr);
    ASSERT_NE(icp._gpu_T_acc, nullptr);
    std::vector<const float*> first_transform_buffers = {
        icp._gpu_T_step->data(),
        icp._gpu_T_acc->data()};
    std::sort(first_transform_buffers.begin(), first_transform_buffers.end());

    auto third_source_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
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
    EXPECT_NE(first_grid_sorted_x, nullptr);
    EXPECT_NE(first_grid_sorted_y, nullptr);
    EXPECT_NE(first_grid_sorted_z, nullptr);
    EXPECT_NE(first_acc_transform, nullptr);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
    EXPECT_NE(first_points_a, nullptr);
    EXPECT_EQ(icp._gpu_stats_workspace.partialStorage(), first_partial_storage);
    EXPECT_EQ(icp._gpu_stats_workspace.statsStorage(), first_stats_storage);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridKeysStorage(), first_grid_keys);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridIndicesStorage(), first_grid_indices);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridSortedXStorage(), first_grid_sorted_x);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridSortedYStorage(), first_grid_sorted_y);
    EXPECT_EQ(icp._gpu_stats_workspace.targetSpatialGridSortedZStorage(), first_grid_sorted_z);
    std::vector<const float*> current_transform_buffers = {
        icp._gpu_T_step->data(),
        icp._gpu_T_acc->data()};
    std::sort(current_transform_buffers.begin(), current_transform_buffers.end());
    EXPECT_EQ(current_transform_buffers, first_transform_buffers);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
    EXPECT_EQ(icp._gpu_points_a->data(), first_points_a);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NE(source->points().data(), first_points_a);
    EXPECT_NE(second_source->points().data(), first_points_a);
    EXPECT_NE(third_source->points().data(), first_points_a);
    EXPECT_EQ(icp._gpu_stats_workspace.partialCapacity(), first_partial_capacity);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignReusesCallerGpuOutputStorageWhenShapeMatches)
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

    GpuCloud output(source->size());
    const auto* output_points = static_cast<const GpuCloud&>(output).points().data();

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(output);

    EXPECT_EQ(static_cast<const GpuCloud&>(output).points().data(), output_points);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), output_points);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
}

TEST(ICPGpuPathTest, AlignWritesTerminalTransformDirectlyWhenOutputAliasesSource)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points_cpu = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points_cpu, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points_cpu));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* source_points = static_cast<const GpuCloud&>(*source).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(*source);

    EXPECT_EQ(source->size(), target->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(*source).points().data(), source_points);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), source_points);
}

TEST(ICPGpuPathTest, AlignUsesScratchForTerminalTransformWhenOutputAliasesTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points_cpu = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points_cpu, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points_cpu));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    plapoint::gpu::resetIcpLastTransformOutputPointerForTesting();
    icp.align(*target);

    EXPECT_EQ(target->size(), source->size());
    EXPECT_EQ(static_cast<const GpuCloud&>(*target).points().data(), target_points_ptr);
    ASSERT_NE(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(plapoint::gpu::icpLastTransformOutputPointerForTesting(), icp._gpu_points_a->data());
}

TEST(ICPGpuPathTest, AlignReplacesAttributedGpuOutputInsteadOfKeepingStaleMetadata)
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
    icp.setMaxCorrespondenceDistance(2.0f);
    icp.setMaxIterations(1);

    GpuCloud output(source->size());
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> stale_normals(source->size(), 3);
    output.setNormals(std::move(stale_normals));
    output.setMaterialLibraryFile("stale.mtl");
    const auto* stale_output_points = static_cast<const GpuCloud&>(output).points().data();

    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NE(static_cast<const GpuCloud&>(output).points().data(), stale_output_points);
    EXPECT_FALSE(output.hasNormals());
    EXPECT_TRUE(output.materialLibraryFile().empty());
}

TEST(ICPGpuPathTest, SetInputTargetKeepsPersistentGpuTargetSpatialGridCacheForSameTarget)
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
    icp.setMaxCorrespondenceDistance(2.0f);
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

TEST(ICPGpuPathTest, SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCacheForNewTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto first_target_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto second_target_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto first_target = std::make_shared<GpuCloud>(first_target_cpu->toGpu());
    auto second_target = std::make_shared<GpuCloud>(second_target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(first_target);
    icp.setMaxCorrespondenceDistance(2.0f);
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto target_cpu = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    const void* source_points = source->points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

TEST(ICPGpuPathTest, GpuAlignMaterializesCpuFinalTransformLazily)
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

    EXPECT_TRUE(icp._final_T_gpu_valid);
    EXPECT_NE(icp._gpu_T_acc.get(), nullptr);
    EXPECT_FALSE(icp._final_T_cpu_valid);

    const auto& transform_cpu = icp.getFinalTransformation();
    EXPECT_TRUE(icp._final_T_cpu_valid);
    ASSERT_EQ(transform_cpu.rows(), 4);
    ASSERT_EQ(transform_cpu.cols(), 4);
    EXPECT_NEAR(transform_cpu.getValue(0, 0), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(1, 1), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(2, 2), 1.0f, 1.0e-5f);
    EXPECT_NEAR(transform_cpu.getValue(3, 3), 1.0f, 1.0e-5f);
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

TEST(ICPGpuPathTest, AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs)
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

TEST(ICPGpuPathTest, AlignReusesIterationStatsForExactIdentityTerminalMetrics)
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
    icp.setMaxCorrespondenceDistance(2.0f);
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
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
    EXPECT_EQ(output.size(), source->size());
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
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    plapoint::gpu::resetIcpTransformMultiplyCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpCorrespondenceStatsCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
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

TEST(ICPGpuPathTest, AlignReservesAlignmentStepWorkspaceOncePerCall)
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

    plapoint::gpu::resetIcpAlignmentStepReserveCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepReserveCountForTesting(), 1);
    EXPECT_EQ(output.size(), source->size());
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
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(output.size(), source->size());
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
    plapoint::gpu::resetIcpRawStatsStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpRawStatsStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(output.size(), source->size());
}

#endif // PLAPOINT_WITH_CUDA
