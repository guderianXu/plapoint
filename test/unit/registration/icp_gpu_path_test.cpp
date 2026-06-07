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
#include <plapoint/gpu/icp.h>
#include <plapoint/registration/icp.h>
#undef private

namespace
{

// Mirrors the production GPU ICP spatial-grid target-count threshold.
constexpr int kMinTargetSpatialGridRowsForTesting = 128;

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeNonCollinearPoints()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(4, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 0.0f); points.setValue(2, 1, 1.0f); points.setValue(2, 2, 0.0f);
    points.setValue(3, 0, 0.0f); points.setValue(3, 1, 0.0f); points.setValue(3, 2, 1.0f);
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslationTransform(
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
    return transform;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedNonCollinearPoints(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& source,
    float tx,
    float ty,
    float tz)
{
    auto transform = makeTranslationTransform(tx, ty, tz);
    return plamatrix::transformPoints(transform, source);
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeCompactNonCollinearGridPoints(int count)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 8;
        const int y = (i / 8) % 8;
        const int z = i / 64;
        points.setValue(i, 0, static_cast<float>(x) * 0.25f);
        points.setValue(i, 1, static_cast<float>(y) * 0.25f);
        points.setValue(i, 2, static_cast<float>(z) * 0.25f);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> padTargetWithNonFiniteRows(
    const plamatrix::DenseMatrix<float, plamatrix::Device::CPU>& points,
    int min_rows = kMinTargetSpatialGridRowsForTesting)
{
    const int rows = static_cast<int>(points.rows());
    const int cols = static_cast<int>(points.cols());
    const int padded_rows = std::max(rows, min_rows);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> padded(padded_rows, cols);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int row = 0; row < padded_rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            padded.setValue(row, col, row < rows ? points.getValue(row, col) : nan);
        }
    }
    return padded;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeGridPoints(int count)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 257;
        const int y = (i / 257) % 251;
        const int z = (i / (257 * 251)) % 241;
        points.setValue(i, 0, static_cast<float>(x) * 0.01f);
        points.setValue(i, 1, static_cast<float>(y) * 0.01f);
        points.setValue(i, 2, static_cast<float>(z) * 0.01f);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedGridPoints(
    int count,
    float tx,
    float ty,
    float tz)
{
    auto points = makeGridPoints(count);
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, points.getValue(i, 0) + tx);
        points.setValue(i, 1, points.getValue(i, 1) + ty);
        points.setValue(i, 2, points.getValue(i, 2) + tz);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeBinaryGridPoints(int count)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 257;
        const int y = (i / 257) % 251;
        const int z = (i / (257 * 251)) % 241;
        points.setValue(i, 0, static_cast<float>(x) * 0.125f);
        points.setValue(i, 1, static_cast<float>(y) * 0.125f);
        points.setValue(i, 2, static_cast<float>(z) * 0.125f);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedBinaryGridPoints(
    int count,
    float tx,
    float ty,
    float tz)
{
    auto points = makeBinaryGridPoints(count);
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, points.getValue(i, 0) + tx);
        points.setValue(i, 1, points.getValue(i, 1) + ty);
        points.setValue(i, 2, points.getValue(i, 2) + tz);
    }
    return points;
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeTranslatedPerturbedGridPoints(
    int count,
    float tx,
    float ty,
    float tz)
{
    auto points = makeTranslatedGridPoints(count, tx, ty, tz);
    for (int i = 0; i < count; ++i)
    {
        points.setValue(i, 0, points.getValue(i, 0) + static_cast<float>((i % 7) - 3) * 0.0002f);
        points.setValue(i, 1, points.getValue(i, 1) + static_cast<float>(((i / 7) % 5) - 2) * 0.00015f);
        points.setValue(i, 2, points.getValue(i, 2) + static_cast<float>(((i / 35) % 3) - 1) * 0.0001f);
    }
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
void resetIcpResidualStatsCallCountForTesting();
int icpResidualStatsCallCountForTesting();
void resetIcpResidualStatsReserveCheckCountForTesting();
int icpResidualStatsReserveCheckCountForTesting();
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
void resetIcpExactPointwiseIdentityStepKernelLaunchCountForTesting();
int icpExactPointwiseIdentityStepKernelLaunchCountForTesting();
void resetIcpSameBufferIdentityAlignmentStepCountForTesting();
int icpSameBufferIdentityAlignmentStepCountForTesting();
void resetIcpTransformedIdentityAlignmentStepCountForTesting();
int icpTransformedIdentityAlignmentStepCountForTesting();
void resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
int icpTransformedExactPointwiseAlignmentStepCallCountForTesting();
void resetIcpTransformedExactPointwiseResidualCallCountForTesting();
int icpTransformedExactPointwiseResidualCallCountForTesting();
void resetIcpTransformedExactPointwiseResidualProbeCountForTesting();
unsigned long long icpTransformedExactPointwiseResidualProbeCountForTesting();
void resetIcpTransformResidualOutputPointWriteCountForTesting();
unsigned long long icpTransformResidualOutputPointWriteCountForTesting();
void resetIcpTransformResidualPointTransformCountForTesting();
unsigned long long icpTransformResidualPointTransformCountForTesting();
void resetIcpExactPointwiseTargetLoadCountForTesting();
unsigned long long icpExactPointwiseTargetLoadCountForTesting();
void resetIcpRawStatsStepKernelLaunchCountForTesting();
int icpRawStatsStepKernelLaunchCountForTesting();
void resetIcpStatsStepHostResultCopyCountForTesting();
int icpStatsStepHostResultCopyCountForTesting();
void resetIcpAlignmentStepHostResultCopyCountForTesting();
int icpAlignmentStepHostResultCopyCountForTesting();
void resetIcpAlignmentStepCallCountForTesting();
int icpAlignmentStepCallCountForTesting();
void resetIcpTransformedAlignmentStepCallCountForTesting();
int icpTransformedAlignmentStepCallCountForTesting();
void resetIcpAccumulatedAlignmentStepCallCountForTesting();
int icpAccumulatedAlignmentStepCallCountForTesting();
void resetIcpAlignmentStepReserveCountForTesting();
int icpAlignmentStepReserveCountForTesting();
void resetIcpAlignmentStepReserveCheckCountForTesting();
int icpAlignmentStepReserveCheckCountForTesting();
std::size_t icpAlignmentStepRawResultByteCountForTesting();
std::size_t icpRawStatsByteCountForTesting();
std::size_t icpFloatAlignmentStepRawResultByteCountForTesting();
std::size_t icpDoubleAlignmentStepRawResultByteCountForTesting();
void resetIcpHostSynchronizationCountForTesting();
int icpHostSynchronizationCountForTesting();
void resetIcpHostResultStorageAllocationCountForTesting();
int icpHostResultStorageAllocationCountForTesting();
void resetIcpTargetTileBoundComputationCountForTesting();
unsigned long long icpTargetTileBoundComputationCountForTesting();
void resetIcpTargetTileLoadCountForTesting();
unsigned long long icpTargetTileLoadCountForTesting();
void resetIcpFallbackTileBoundKernelLaunchCountForTesting();
int icpFallbackTileBoundKernelLaunchCountForTesting();
void resetIcpFallbackUnboundedKernelLaunchCountForTesting();
int icpFallbackUnboundedKernelLaunchCountForTesting();
void resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
int icpSmallAlignmentStepKernelLaunchCountForTesting();
void resetIcpSmallStatsStepKernelLaunchCountForTesting();
int icpSmallStatsStepKernelLaunchCountForTesting();
void resetIcpSmallResidualStatsKernelLaunchCountForTesting();
int icpSmallResidualStatsKernelLaunchCountForTesting();
void resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
int icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
void resetIcpTargetSpatialGridBuildCountForTesting();
int icpTargetSpatialGridBuildCountForTesting();
void resetIcpTargetSpatialGridPrepareCountForTesting();
int icpTargetSpatialGridPrepareCountForTesting();
void resetIcpTargetSpatialGridReserveCountForTesting();
int icpTargetSpatialGridReserveCountForTesting();
void resetIcpTargetTileBoundsReserveCountForTesting();
int icpTargetTileBoundsReserveCountForTesting();
void resetIcpDirectSpatialGridKernelLaunchCountForTesting();
int icpDirectSpatialGridKernelLaunchCountForTesting();
void resetIcpGridCellLookupCountForTesting();
unsigned long long icpGridCellLookupCountForTesting();
void resetIcpGridCellOffsetCountForTesting();
unsigned long long icpGridCellOffsetCountForTesting();
void resetIcpGridCellCenterMinDistanceCountForTesting();
unsigned long long icpGridCellCenterMinDistanceCountForTesting();
void resetIcpGridCellNeighborMinDistanceCountForTesting();
unsigned long long icpGridCellNeighborMinDistanceCountForTesting();
void resetIcpGridCellNeighborXyDistanceCountForTesting();
unsigned long long icpGridCellNeighborXyDistanceCountForTesting();
void resetIcpDirectGridLookupXyCheckCountForTesting();
unsigned long long icpDirectGridLookupXyCheckCountForTesting();
void resetIcpDirectGridLookupActiveGuardCountForTesting();
unsigned long long icpDirectGridLookupActiveGuardCountForTesting();
void resetIcpDirectGridLookupXyBaseGuardCountForTesting();
unsigned long long icpDirectGridLookupXyBaseGuardCountForTesting();
void resetIcpDirectGridLookupLinearGuardCountForTesting();
unsigned long long icpDirectGridLookupLinearGuardCountForTesting();
void resetIcpOuterLowerTriangleAccumulationCountForTesting();
unsigned long long icpOuterLowerTriangleAccumulationCountForTesting();
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

TEST(ICPGpuPathTest, StepTransformWorkspaceReusesPinnedHostResultStorage)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plapoint::gpu::IcpStepTransformWorkspace workspace;

    plapoint::gpu::resetIcpHostResultStorageAllocationCountForTesting();
    workspace.reserveResult();
    auto* first_host_result = workspace.hostResultStorage();
    const auto first_capacity = workspace.hostResultStorageCapacity();

    ASSERT_NE(first_host_result, nullptr);
    EXPECT_GT(first_capacity, std::size_t{0});
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);

    workspace.reserveResult();
    EXPECT_EQ(workspace.hostResultStorage(), first_host_result);
    EXPECT_EQ(workspace.hostResultStorageCapacity(), first_capacity);
    EXPECT_EQ(plapoint::gpu::icpHostResultStorageAllocationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignmentStepRawResultFitsCompactHostCopy)
{
    EXPECT_GT(plapoint::gpu::icpAlignmentStepRawResultByteCountForTesting(), std::size_t{0});
    EXPECT_LE(plapoint::gpu::icpAlignmentStepRawResultByteCountForTesting(), std::size_t{40});
}

TEST(ICPGpuPathTest, RawIcpStatsUsesCompactOuterCovarianceStorage)
{
    EXPECT_GT(plapoint::gpu::icpRawStatsByteCountForTesting(), std::size_t{0});
    EXPECT_LE(plapoint::gpu::icpRawStatsByteCountForTesting(), std::size_t{240});
}

TEST(ICPGpuPathTest, FloatAlignmentStepRawResultUsesFloatSizedDelta)
{
    EXPECT_GT(plapoint::gpu::icpFloatAlignmentStepRawResultByteCountForTesting(), std::size_t{0});
    EXPECT_LE(plapoint::gpu::icpFloatAlignmentStepRawResultByteCountForTesting(), std::size_t{32});
    EXPECT_LE(plapoint::gpu::icpDoubleAlignmentStepRawResultByteCountForTesting(), std::size_t{40});
    EXPECT_LT(
        plapoint::gpu::icpFloatAlignmentStepRawResultByteCountForTesting(),
        plapoint::gpu::icpDoubleAlignmentStepRawResultByteCountForTesting());
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
    EXPECT_TRUE(std::isfinite(compact_result.step_residual_sq_sum));
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

TEST(ICPGpuPathTest, StatsAndStepTransformCopiesOneHostResult)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeNonCollinearPoints();
    auto target_cpu = makeTranslatedNonCollinearPoints(source_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpStatsStepHostResultCopyCountForTesting();
    const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        stats_workspace,
        step_gpu.data(),
        step_workspace);

    EXPECT_EQ(result.stats.active_count, 4);
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpStatsStepHostResultCopyCountForTesting(), 1);
}

TEST(ICPGpuPathTest, StatsAndStepTransformReusesCachedSpatialGridAcrossCalls)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeNonCollinearPoints();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source_cpu(4, 3);
    for (int col = 0; col < 3; ++col)
    {
        source_cpu.setValue(0, col, target_cpu.getValue(1, col));
        source_cpu.setValue(1, col, target_cpu.getValue(2, col));
        source_cpu.setValue(2, col, target_cpu.getValue(3, col));
        source_cpu.setValue(3, col, target_cpu.getValue(0, col));
    }
    target_cpu = padTargetWithNonFiniteRows(target_cpu);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridReserveCountForTesting();
    plapoint::gpu::resetIcpStatsStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    const auto first_result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.2f,
        stats_workspace,
        step_gpu.data(),
        step_workspace);
    const auto second_result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.2f,
        stats_workspace,
        step_gpu.data(),
        step_workspace);

    EXPECT_EQ(first_result.stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_EQ(second_result.stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(first_result.step_valid);
    EXPECT_TRUE(second_result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridReserveCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpStatsStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignmentStepCopiesOneCompactHostResultPerCall)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeNonCollinearPoints();
    auto target_cpu = makeTranslatedNonCollinearPoints(source_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpStepTransformInputCopyCountForTesting();
    plapoint::gpu::resetIcpRawStatsStepKernelLaunchCountForTesting();
    const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        stats_workspace,
        step_gpu.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpStepTransformInputCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpRawStatsStepKernelLaunchCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignmentStepSkipsTargetSpatialGridForSmallFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeNonCollinearPoints();
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        stats_workspace,
        step_gpu.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignmentStepSkipsTargetSpatialGridBelowTargetCountThreshold)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        step_gpu.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignmentStepSkipsTargetTileBoundsForSmallFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpTargetTileBoundsReserveCountForTesting();
    plapoint::gpu::resetIcpTargetTileBoundComputationCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        step_gpu.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpTargetTileBoundsReserveCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetTileBoundComputationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignmentStepUsesFusedSmallTargetKernelForFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        step_gpu.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, StatsAndStepUsesFusedSmallTargetKernelForFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpSmallStatsStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpStatsStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        step_gpu.data(),
        step_workspace);

    EXPECT_EQ(result.stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.stats.src_has_non_collinear_geometry);
    EXPECT_TRUE(result.stats.tgt_has_non_collinear_geometry);
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpSmallStatsStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpStatsStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, ResidualStatsUsesFusedSmallTargetKernelForFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;

    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_GT(stats.residual_sq_sum, 0.0);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, TransformResidualStatsUsesFusedSmallTargetKernelForFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto transform_cpu = makeTranslationTransform(-0.01f, 0.005f, -0.0025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();
    auto transform_gpu = transform_cpu.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source_gpu.rows(), 3);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;

    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTransformResidualOutputPointWriteCountForTesting();
    plapoint::gpu::resetIcpTransformResidualPointTransformCountForTesting();
    const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
        transform_gpu.data(),
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        output_gpu.data(),
        stats_workspace);

    EXPECT_EQ(stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_NEAR(stats.residual_sq_sum, 0.0, 1.0e-9);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(
        plapoint::gpu::icpTransformResidualPointTransformCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()));
    EXPECT_EQ(
        plapoint::gpu::icpTransformResidualOutputPointWriteCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()));
}

TEST(ICPGpuPathTest, TerminalAlignmentResidualMatchesSeparateSmallTargetBaseline)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.setValue(1, 0, target_cpu.getValue(1, 0) + 0.03f);
    target_cpu.setValue(2, 1, target_cpu.getValue(2, 1) - 0.02f);
    target_cpu.setValue(3, 2, target_cpu.getValue(3, 2) + 0.015f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        first_step_gpu.data());
    ASSERT_TRUE(first_step.step_valid);

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> fused_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> fused_accumulated_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> fused_output_gpu(source_gpu.rows(), 3);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTransformResidualOutputPointWriteCountForTesting();
    const auto fused_result =
        plapoint::gpu::detail::
            computeTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
            first_step_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.08f,
            stats_workspace,
            fused_step_gpu.data(),
            first_step_gpu.data(),
            fused_accumulated_gpu.data(),
            0,
            fused_output_gpu.data());

    ASSERT_TRUE(fused_result.launched);
    EXPECT_TRUE(fused_result.alignment_step.step_valid);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(
        plapoint::gpu::icpTransformResidualOutputPointWriteCountForTesting(),
        static_cast<unsigned long long>(source_gpu.rows()));

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_accumulated_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_output_gpu(source_gpu.rows(), 3);
    const auto baseline_step =
        plapoint::gpu::detail::
            computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
            first_step_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.08f,
            stats_workspace,
            baseline_step_gpu.data(),
            first_step_gpu.data(),
            baseline_accumulated_gpu.data());
    const auto baseline_residual =
        plapoint::gpu::detail::transformPointsAndComputeIcpResidualStatsColumnMajorWithReservedWorkspace(
            baseline_accumulated_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.08f,
            baseline_output_gpu.data(),
            stats_workspace);

    EXPECT_EQ(fused_result.alignment_step.active_count, baseline_step.active_count);
    EXPECT_EQ(fused_result.alignment_step.invalid_source_count, baseline_step.invalid_source_count);
    EXPECT_NEAR(fused_result.alignment_step.step.delta, baseline_step.step.delta, 1.0e-6f);
    EXPECT_EQ(fused_result.residual_stats.active_count, baseline_residual.active_count);
    EXPECT_EQ(fused_result.residual_stats.invalid_source_count, baseline_residual.invalid_source_count);
    EXPECT_NEAR(fused_result.residual_stats.residual_sq_sum, baseline_residual.residual_sq_sum, 1.0e-9);
    EXPECT_GT(fused_result.residual_stats.residual_sq_sum, 0.0);

    const auto fused_output_cpu = fused_output_gpu.toCpu();
    const auto baseline_output_cpu = baseline_output_gpu.toCpu();
    ASSERT_EQ(fused_output_cpu.rows(), baseline_output_cpu.rows());
    ASSERT_EQ(fused_output_cpu.cols(), baseline_output_cpu.cols());
    for (plamatrix::Index row = 0; row < fused_output_cpu.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < fused_output_cpu.cols(); ++col)
        {
            EXPECT_NEAR(
                fused_output_cpu.getValue(row, col),
                baseline_output_cpu.getValue(row, col),
                1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, SmallTargetAlignmentStepAsyncLaunchDefersHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.setValue(1, 0, target_cpu.getValue(1, 0) + 0.03f);
    target_cpu.setValue(2, 1, target_cpu.getValue(2, 1) - 0.02f);
    target_cpu.setValue(3, 2, target_cpu.getValue(3, 2) + 0.015f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_step_gpu(4, 4);
    const auto baseline = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        baseline_workspace,
        baseline_step_gpu.data());
    ASSERT_TRUE(baseline.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_step_gpu(4, 4);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    const bool launched =
        plapoint::gpu::detail::launchSmallTargetAlignmentStepColumnMajorWithReservedWorkspace(
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.08f,
            async_workspace,
            async_step_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto async_result =
        plapoint::gpu::detail::copyAlignmentStepResultFromReservedWorkspace<float>(
            async_workspace,
            0);

    ASSERT_TRUE(async_result.step_valid);
    EXPECT_EQ(async_result.active_count, baseline.active_count);
    EXPECT_EQ(async_result.invalid_source_count, baseline.invalid_source_count);
    EXPECT_NEAR(async_result.step.delta, baseline.step.delta, 1.0e-6f);
    EXPECT_NEAR(async_result.residual_sq_sum, baseline.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);

    const auto baseline_step_cpu = baseline_step_gpu.toCpu();
    const auto async_step_cpu = async_step_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_step_cpu.getValue(row, col),
                baseline_step_cpu.getValue(row, col),
                1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignmentStepAsyncLaunchUsesSpatialGridAndDefersHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_step_gpu(4, 4);
    const auto baseline = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.02f,
        baseline_workspace,
        baseline_step_gpu.data());
    ASSERT_TRUE(baseline.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_step_gpu(4, 4);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const bool launched =
        plapoint::gpu::detail::launchIcpAlignmentStepColumnMajorWithReservedWorkspace(
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.02f,
            async_workspace,
            async_step_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto async_result =
        plapoint::gpu::detail::copyAlignmentStepResultFromReservedWorkspace<float>(
            async_workspace,
            0);

    ASSERT_TRUE(async_result.step_valid);
    EXPECT_EQ(async_result.active_count, baseline.active_count);
    EXPECT_EQ(async_result.invalid_source_count, baseline.invalid_source_count);
    EXPECT_NEAR(async_result.step.delta, baseline.step.delta, 1.0e-6f);
    EXPECT_NEAR(async_result.residual_sq_sum, baseline.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);

    const auto baseline_step_cpu = baseline_step_gpu.toCpu();
    const auto async_step_cpu = async_step_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_step_cpu.getValue(row, col),
                baseline_step_cpu.getValue(row, col),
                1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, AlignmentStepAsyncLaunchRejectsHostGuidedFallbackRequests)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    const bool ordered_launched =
        plapoint::gpu::detail::launchIcpAlignmentStepColumnMajorWithReservedWorkspace(
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.02f,
            workspace,
            step_gpu.data(),
            0,
            true);

    EXPECT_FALSE(ordered_launched);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    auto exact_source_cpu = makeGridPoints(4096);
    auto exact_target_cpu = makeGridPoints(4096);
    auto exact_source_gpu = exact_source_cpu.toGpu();
    auto exact_target_gpu = exact_target_cpu.toGpu();

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    const bool probe_launched =
        plapoint::gpu::detail::launchIcpAlignmentStepColumnMajorWithReservedWorkspace(
            exact_source_gpu.data(),
            static_cast<int>(exact_source_gpu.rows()),
            exact_target_gpu.data(),
            static_cast<int>(exact_target_gpu.rows()),
            0.02f,
            workspace,
            step_gpu.data(),
            0,
            false,
            true);

    EXPECT_FALSE(probe_launched);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);
}

TEST(ICPGpuPathTest, TerminalAlignmentResidualAsyncLaunchDefersHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.setValue(1, 0, target_cpu.getValue(1, 0) + 0.03f);
    target_cpu.setValue(2, 1, target_cpu.getValue(2, 1) - 0.02f);
    target_cpu.setValue(3, 2, target_cpu.getValue(3, 2) + 0.015f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        first_step_gpu.data());
    ASSERT_TRUE(first_step.step_valid);

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> fused_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> fused_accumulated_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> fused_output_gpu(source_gpu.rows(), 3);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    const bool launched =
        plapoint::gpu::detail::
            launchTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
            first_step_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.08f,
            stats_workspace,
            fused_step_gpu.data(),
            first_step_gpu.data(),
            fused_accumulated_gpu.data(),
            0,
            fused_output_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto result =
        plapoint::gpu::detail::
            copySmallTargetTerminalAlignmentAndResidualResultFromReservedWorkspace<float>(
                stats_workspace,
                0);

    ASSERT_TRUE(result.launched);
    EXPECT_TRUE(result.alignment_step.step_valid);
    EXPECT_GT(result.residual_stats.residual_sq_sum, 0.0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, SmallTargetSingleStepTerminalAsyncLaunchCopiesResultWithOneHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.setValue(1, 0, target_cpu.getValue(1, 0) + 0.03f);
    target_cpu.setValue(2, 1, target_cpu.getValue(2, 1) - 0.02f);
    target_cpu.setValue(3, 2, target_cpu.getValue(3, 2) + 0.015f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_step_gpu(4, 4);
    const auto baseline_step = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        baseline_workspace,
        baseline_step_gpu.data());
    ASSERT_TRUE(baseline_step.step_valid);

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_output_gpu(source_gpu.rows(), 3);
    const auto baseline_residual =
        plapoint::gpu::detail::transformPointsAndComputeIcpResidualStatsColumnMajorWithReservedWorkspace(
            baseline_step_gpu.data(),
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.08f,
            baseline_output_gpu.data(),
            baseline_workspace);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_output_gpu(source_gpu.rows(), 3);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    const bool launched =
        plapoint::gpu::detail::
            launchSmallTargetSingleStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                async_workspace,
                async_step_gpu.data(),
                0,
                async_output_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto async_result =
        plapoint::gpu::detail::copySmallTargetTerminalAlignmentAndResidualResultFromReservedWorkspace<float>(
            async_workspace,
            0);

    ASSERT_TRUE(async_result.launched);
    ASSERT_TRUE(async_result.alignment_step.step_valid);
    EXPECT_EQ(async_result.alignment_step.active_count, baseline_step.active_count);
    EXPECT_EQ(async_result.alignment_step.invalid_source_count, baseline_step.invalid_source_count);
    EXPECT_NEAR(async_result.alignment_step.step.delta, baseline_step.step.delta, 1.0e-6f);
    EXPECT_NEAR(async_result.alignment_step.residual_sq_sum, baseline_step.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(async_result.residual_stats.active_count, baseline_residual.active_count);
    EXPECT_EQ(async_result.residual_stats.invalid_source_count, baseline_residual.invalid_source_count);
    EXPECT_NEAR(async_result.residual_stats.residual_sq_sum, baseline_residual.residual_sq_sum, 1.0e-9);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);

    const auto baseline_step_cpu = baseline_step_gpu.toCpu();
    const auto async_step_cpu = async_step_gpu.toCpu();
    const auto baseline_output_cpu = baseline_output_gpu.toCpu();
    const auto async_output_cpu = async_output_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_step_cpu.getValue(row, col),
                baseline_step_cpu.getValue(row, col),
                1.0e-5f);
        }
    }
    for (plamatrix::Index row = 0; row < async_output_cpu.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < async_output_cpu.cols(); ++col)
        {
            EXPECT_NEAR(
                async_output_cpu.getValue(row, col),
                baseline_output_cpu.getValue(row, col),
                1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, SmallTargetTwoStepTerminalAsyncLaunchCopiesResultsWithOneHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.setValue(1, 0, target_cpu.getValue(1, 0) + 0.03f);
    target_cpu.setValue(2, 1, target_cpu.getValue(2, 1) - 0.02f);
    target_cpu.setValue(3, 2, target_cpu.getValue(3, 2) + 0.015f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_first_workspace;
    baseline_first_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_first_step_gpu(4, 4);
    const auto baseline_first =
        plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.08f,
            baseline_first_workspace,
            baseline_first_step_gpu.data());
    ASSERT_TRUE(baseline_first.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_terminal_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_terminal_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_accumulated_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_output_gpu(source_gpu.rows(), 3);
    const auto baseline_terminal =
        plapoint::gpu::detail::
            computeTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
                baseline_first_step_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                baseline_terminal_workspace,
                baseline_terminal_step_gpu.data(),
                baseline_first_step_gpu.data(),
                baseline_accumulated_gpu.data(),
                0,
                baseline_output_gpu.data());
    ASSERT_TRUE(baseline_terminal.launched);
    ASSERT_TRUE(baseline_terminal.alignment_step.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_first_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_terminal_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_first_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_terminal_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_accumulated_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_output_gpu(source_gpu.rows(), 3);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    const bool launched =
        plapoint::gpu::detail::
            launchSmallTargetTwoStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspaces(
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                async_first_workspace,
                async_terminal_workspace,
                async_first_step_gpu.data(),
                async_terminal_step_gpu.data(),
                async_accumulated_gpu.data(),
                0,
                async_output_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto async_result =
        plapoint::gpu::detail::
            copySmallTargetTwoStepTerminalAlignmentAndResidualResultFromReservedWorkspaces<float>(
                async_first_workspace,
                async_terminal_workspace,
                0);

    ASSERT_TRUE(async_result.launched);
    ASSERT_TRUE(async_result.first_alignment_step.step_valid);
    ASSERT_TRUE(async_result.terminal_result.launched);
    ASSERT_TRUE(async_result.terminal_result.alignment_step.step_valid);
    EXPECT_EQ(async_result.first_alignment_step.active_count, baseline_first.active_count);
    EXPECT_EQ(async_result.first_alignment_step.invalid_source_count, baseline_first.invalid_source_count);
    EXPECT_NEAR(async_result.first_alignment_step.step.delta, baseline_first.step.delta, 1.0e-6f);
    EXPECT_NEAR(async_result.first_alignment_step.residual_sq_sum, baseline_first.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(async_result.terminal_result.alignment_step.active_count, baseline_terminal.alignment_step.active_count);
    EXPECT_EQ(
        async_result.terminal_result.alignment_step.invalid_source_count,
        baseline_terminal.alignment_step.invalid_source_count);
    EXPECT_NEAR(
        async_result.terminal_result.alignment_step.step.delta,
        baseline_terminal.alignment_step.step.delta,
        1.0e-6f);
    EXPECT_EQ(async_result.terminal_result.residual_stats.active_count, baseline_terminal.residual_stats.active_count);
    EXPECT_EQ(
        async_result.terminal_result.residual_stats.invalid_source_count,
        baseline_terminal.residual_stats.invalid_source_count);
    EXPECT_NEAR(
        async_result.terminal_result.residual_stats.residual_sq_sum,
        baseline_terminal.residual_stats.residual_sq_sum,
        1.0e-9);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toCpu();
    const auto async_accumulated_cpu = async_accumulated_gpu.toCpu();
    const auto baseline_output_cpu = baseline_output_gpu.toCpu();
    const auto async_output_cpu = async_output_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.getValue(row, col),
                baseline_accumulated_cpu.getValue(row, col),
                1.0e-5f);
        }
    }
    for (plamatrix::Index row = 0; row < async_output_cpu.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < async_output_cpu.cols(); ++col)
        {
            EXPECT_NEAR(
                async_output_cpu.getValue(row, col),
                baseline_output_cpu.getValue(row, col),
                1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, SmallTargetTwoStepTerminalAsyncLaunchSkipsTerminalWhenFirstStepInvalid)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> target_cpu(4, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source_cpu(4, 3);
    target_cpu.setValue(0, 0, 0.0f); target_cpu.setValue(0, 1, 0.0f); target_cpu.setValue(0, 2, 0.0f);
    target_cpu.setValue(1, 0, 0.1f); target_cpu.setValue(1, 1, 0.0f); target_cpu.setValue(1, 2, 0.0f);
    target_cpu.setValue(2, 0, 0.0f); target_cpu.setValue(2, 1, 0.1f); target_cpu.setValue(2, 2, 0.0f);
    target_cpu.setValue(3, 0, 0.0f); target_cpu.setValue(3, 1, 0.0f); target_cpu.setValue(3, 2, 0.1f);
    source_cpu.setValue(0, 0, 0.005f); source_cpu.setValue(0, 1, 0.0f); source_cpu.setValue(0, 2, 0.0f);
    source_cpu.setValue(1, 0, 0.105f); source_cpu.setValue(1, 1, 0.0f); source_cpu.setValue(1, 2, 0.0f);
    source_cpu.setValue(2, 0, 10.0f); source_cpu.setValue(2, 1, 10.0f); source_cpu.setValue(2, 2, 10.0f);
    source_cpu.setValue(3, 0, 20.0f); source_cpu.setValue(3, 1, 20.0f); source_cpu.setValue(3, 2, 20.0f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace terminal_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> terminal_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_gpu(4, 4);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    const bool launched =
        plapoint::gpu::detail::
            launchSmallTargetTwoStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspaces(
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                first_workspace,
                terminal_workspace,
                first_step_gpu.data(),
                terminal_step_gpu.data(),
                accumulated_gpu.data());

    ASSERT_TRUE(launched);

    const auto result =
        plapoint::gpu::detail::
            copySmallTargetTwoStepTerminalAlignmentAndResidualResultFromReservedWorkspaces<float>(
                first_workspace,
                terminal_workspace);

    ASSERT_TRUE(result.launched);
    EXPECT_EQ(result.first_alignment_step.active_count, 2);
    EXPECT_LT(result.first_alignment_step.active_count, 3);
    EXPECT_EQ(result.terminal_result.alignment_step.active_count, 0);
    EXPECT_FALSE(result.terminal_result.alignment_step.step_valid);
    EXPECT_EQ(result.terminal_result.residual_stats.active_count, 0);
    EXPECT_EQ(result.terminal_result.residual_stats.invalid_source_count, 0);
    EXPECT_EQ(result.terminal_result.residual_stats.residual_sq_sum, 0.0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, TransformedAccumulatedAlignmentStepAsyncLaunchDefersHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.setValue(1, 0, target_cpu.getValue(1, 0) + 0.03f);
    target_cpu.setValue(2, 1, target_cpu.getValue(2, 1) - 0.02f);
    target_cpu.setValue(3, 2, target_cpu.getValue(3, 2) + 0.015f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        first_step_workspace,
        first_step_gpu.data());
    ASSERT_TRUE(first_step.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_accumulated_gpu(4, 4);
    const auto baseline =
        plapoint::gpu::detail::
            computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                first_step_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                baseline_workspace,
                baseline_step_gpu.data(),
                first_step_gpu.data(),
                baseline_accumulated_gpu.data());
    ASSERT_TRUE(baseline.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_accumulated_gpu(4, 4);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    const bool launched =
        plapoint::gpu::detail::
            launchTransformedSmallTargetAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                first_step_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                async_workspace,
                async_step_gpu.data(),
                first_step_gpu.data(),
                async_accumulated_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto async_result =
        plapoint::gpu::detail::copyAlignmentStepResultFromReservedWorkspace<float>(
            async_workspace,
            0);

    ASSERT_TRUE(async_result.step_valid);
    EXPECT_EQ(async_result.active_count, baseline.active_count);
    EXPECT_EQ(async_result.invalid_source_count, baseline.invalid_source_count);
    EXPECT_NEAR(async_result.step.delta, baseline.step.delta, 1.0e-6f);
    EXPECT_NEAR(async_result.residual_sq_sum, baseline.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toCpu();
    const auto async_accumulated_cpu = async_accumulated_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.getValue(row, col),
                baseline_accumulated_cpu.getValue(row, col),
                1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, TransformedAccumulatedAlignmentStepAsyncLaunchUsesSpatialGridAndDefersHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.02f,
        first_step_workspace,
        first_step_gpu.data());
    ASSERT_TRUE(first_step.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_accumulated_gpu(4, 4);
    const auto baseline =
        plapoint::gpu::detail::
            computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                first_step_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.02f,
                baseline_workspace,
                baseline_step_gpu.data(),
                first_step_gpu.data(),
                baseline_accumulated_gpu.data());
    ASSERT_TRUE(baseline.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_accumulated_gpu(4, 4);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    const bool launched =
        plapoint::gpu::detail::
            launchTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                first_step_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.02f,
                async_workspace,
                async_step_gpu.data(),
                first_step_gpu.data(),
                async_accumulated_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto async_result =
        plapoint::gpu::detail::copyAlignmentStepResultFromReservedWorkspace<float>(
            async_workspace,
            0);

    ASSERT_TRUE(async_result.step_valid);
    EXPECT_EQ(async_result.active_count, baseline.active_count);
    EXPECT_EQ(async_result.invalid_source_count, baseline.invalid_source_count);
    EXPECT_NEAR(async_result.step.delta, baseline.step.delta, 1.0e-6f);
    EXPECT_NEAR(async_result.residual_sq_sum, baseline.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toCpu();
    const auto async_accumulated_cpu = async_accumulated_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.getValue(row, col),
                baseline_accumulated_cpu.getValue(row, col),
                1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, TwoStepAlignmentAsyncLaunchUsesSpatialGridAndCopiesResultsWithOneSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_first_workspace;
    baseline_first_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_first_step_gpu(4, 4);
    const auto baseline_first = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.02f,
        baseline_first_workspace,
        baseline_first_step_gpu.data());
    ASSERT_TRUE(baseline_first.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_second_workspace;
    baseline_second_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_second_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> baseline_accumulated_gpu(4, 4);
    const auto baseline_second =
        plapoint::gpu::detail::
            computeTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                baseline_first_step_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.02f,
                baseline_second_workspace,
                baseline_second_step_gpu.data(),
                baseline_first_step_gpu.data(),
                baseline_accumulated_gpu.data());
    ASSERT_TRUE(baseline_second.step_valid);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_first_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace async_second_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_first_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_second_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> async_accumulated_gpu(4, 4);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const bool launched =
        plapoint::gpu::detail::launchIcpTwoStepAlignmentColumnMajorWithReservedWorkspaces(
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.02f,
            async_first_workspace,
            async_second_workspace,
            async_first_step_gpu.data(),
            async_second_step_gpu.data(),
            async_accumulated_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto result =
        plapoint::gpu::detail::copyIcpTwoStepAlignmentResultFromReservedWorkspaces<float>(
            async_first_workspace,
            async_second_workspace,
            0);

    ASSERT_TRUE(result.launched);
    ASSERT_TRUE(result.first_alignment_step.step_valid);
    ASSERT_TRUE(result.second_alignment_step.step_valid);
    EXPECT_EQ(result.first_alignment_step.active_count, baseline_first.active_count);
    EXPECT_EQ(result.first_alignment_step.invalid_source_count, baseline_first.invalid_source_count);
    EXPECT_NEAR(result.first_alignment_step.step.delta, baseline_first.step.delta, 1.0e-6f);
    EXPECT_NEAR(result.first_alignment_step.residual_sq_sum, baseline_first.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(result.second_alignment_step.active_count, baseline_second.active_count);
    EXPECT_EQ(result.second_alignment_step.invalid_source_count, baseline_second.invalid_source_count);
    EXPECT_NEAR(result.second_alignment_step.step.delta, baseline_second.step.delta, 1.0e-6f);
    EXPECT_NEAR(result.second_alignment_step.residual_sq_sum, baseline_second.residual_sq_sum, 1.0e-5);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toCpu();
    const auto async_accumulated_cpu = async_accumulated_gpu.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.getValue(row, col),
                baseline_accumulated_cpu.getValue(row, col),
                1.0e-5f);
        }
    }
}

TEST(ICPGpuPathTest, TwoStepAlignmentAsyncLaunchWritesEmptySecondStepWhenFirstStepInvalid)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeGridPoints(4096);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> source_cpu(2, 3);
    for (int row = 0; row < 2; ++row)
    {
        source_cpu.setValue(row, 0, target_cpu.getValue(row, 0) + 0.003f);
        source_cpu.setValue(row, 1, target_cpu.getValue(row, 1) - 0.002f);
        source_cpu.setValue(row, 2, target_cpu.getValue(row, 2) + 0.001f);
    }
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace second_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> second_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_gpu(4, 4);
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    const bool launched =
        plapoint::gpu::detail::launchIcpTwoStepAlignmentColumnMajorWithReservedWorkspaces(
            source_gpu.data(),
            static_cast<int>(source_gpu.rows()),
            target_gpu.data(),
            static_cast<int>(target_gpu.rows()),
            0.02f,
            first_workspace,
            second_workspace,
            first_step_gpu.data(),
            second_step_gpu.data(),
            accumulated_gpu.data());

    ASSERT_TRUE(launched);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 0);

    const auto result =
        plapoint::gpu::detail::copyIcpTwoStepAlignmentResultFromReservedWorkspaces<float>(
            first_workspace,
            second_workspace,
            0);

    ASSERT_TRUE(result.launched);
    EXPECT_EQ(result.first_alignment_step.active_count, 2);
    EXPECT_LT(result.first_alignment_step.active_count, 3);
    EXPECT_EQ(result.second_alignment_step.active_count, 0);
    EXPECT_EQ(result.second_alignment_step.invalid_source_count, 0);
    EXPECT_FALSE(result.second_alignment_step.step_valid);
    EXPECT_EQ(result.second_alignment_step.residual_sq_sum, 0.0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
}

TEST(ICPGpuPathTest, AlignUsesFusedSmallTargetKernelForTransformedFiniteRadiusStep)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTransformedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpAccumulatedAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_LE(plapoint::gpu::icpAccumulatedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignSmallFiniteRadiusFinalMetricsAvoidExtraHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    target_cpu_points.setValue(1, 0, target_cpu_points.getValue(1, 0) + 0.03f);
    target_cpu_points.setValue(2, 1, target_cpu_points.getValue(2, 1) - 0.02f);
    target_cpu_points.setValue(3, 2, target_cpu_points.getValue(3, 2) + 0.015f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_LT(icp.getFinalRmse(), 0.08f);
    EXPECT_TRUE(std::isfinite(icp.getFinalRmse()));
}

TEST(ICPGpuPathTest, AlignSmallFiniteRadiusFinalMetricsWithOutputAvoidExtraHostSynchronization)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    target_cpu_points.setValue(1, 0, target_cpu_points.getValue(1, 0) + 0.03f);
    target_cpu_points.setValue(2, 1, target_cpu_points.getValue(2, 1) - 0.02f);
    target_cpu_points.setValue(3, 2, target_cpu_points.getValue(3, 2) + 0.015f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(output.size(), source->size());
    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_LT(icp.getFinalRmse(), 0.08f);
    EXPECT_TRUE(std::isfinite(icp.getFinalRmse()));

    const auto output_cpu = output.toCpu();
    for (plamatrix::Index row = 0; row < output_cpu.points().rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_cpu.points().cols(); ++col)
        {
            EXPECT_TRUE(std::isfinite(output_cpu.points().getValue(row, col)));
        }
    }
}

TEST(ICPGpuPathTest, AlignSmallFiniteRadiusFinalMetricsWithTargetAliasOutputAvoidsScratchCopy)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    target_cpu_points.setValue(1, 0, target_cpu_points.getValue(1, 0) + 0.03f);
    target_cpu_points.setValue(2, 1, target_cpu_points.getValue(2, 1) - 0.02f);
    target_cpu_points.setValue(3, 2, target_cpu_points.getValue(3, 2) + 0.015f);

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points = target->points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    plapoint::gpu::resetIcpTransformResidualOutputPointWriteCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    icp.align(*target);

    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformResidualOutputPointWriteCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 1);
    EXPECT_EQ(target->points().data(), target_points);
    EXPECT_EQ(target->size(), source->size());
    EXPECT_EQ(icp._gpu_points_a, nullptr);
    EXPECT_EQ(icp._gpu_points_b, nullptr);
    EXPECT_EQ(icp._gpu_output_device_to_device_copy_async_count, 0);
    EXPECT_GT(icp.getFinalRmse(), 0.0f);
    EXPECT_LT(icp.getFinalRmse(), 0.08f);
    EXPECT_TRUE(std::isfinite(icp.getFinalRmse()));

    const auto output_cpu = target->toCpu();
    for (plamatrix::Index row = 0; row < output_cpu.points().rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < output_cpu.points().cols(); ++col)
        {
            EXPECT_TRUE(std::isfinite(output_cpu.points().getValue(row, col)));
        }
    }
}

TEST(ICPGpuPathTest, AlignmentStepUsesFusedSmallTargetKernelAtTargetCountThreshold)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackTileBoundKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpFallbackUnboundedKernelLaunchCountForTesting();
    const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        step_gpu.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFallbackTileBoundKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpFallbackUnboundedKernelLaunchCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignmentStepUsesTargetSpatialGridForLargeFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto source_cpu = makeTranslatedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.02f,
        stats_workspace,
        step_gpu.data());

    EXPECT_EQ(result.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
}

TEST(ICPGpuPathTest, StatsAndStepCanUseOrderedCorrespondencesForFiniteRadiusTranslation)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_cpu = makeNonCollinearPoints();
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = source_cpu.toGpu();
    auto target_gpu = target_cpu.toGpu();

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        2.0f,
        stats_workspace,
        step_gpu.data(),
        step_workspace,
        0,
        true);

    EXPECT_EQ(result.stats.active_count, static_cast<int>(source_gpu.rows()));
    EXPECT_TRUE(result.step_valid);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);

    const auto step_cpu = step_gpu.toCpu();
    EXPECT_NEAR(step_cpu.getValue(0, 3), -0.1f, 1.0e-5f);
    EXPECT_NEAR(step_cpu.getValue(1, 3), 0.05f, 1.0e-5f);
    EXPECT_NEAR(step_cpu.getValue(2, 3), -0.025f, 1.0e-5f);
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

TEST(ICPGpuPathTest, AlignDoesNotPopulateGpuPointCpuCaches)
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

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
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

    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f);
    auto target_points = makeBinaryGridPoints(4096);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridReserveCountForTesting();
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
}

TEST(ICPGpuPathTest, AlignReusesGpuWorkspacesAcrossRepeatedCalls)
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* source_points = static_cast<const GpuCloud&>(*source).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

TEST(ICPGpuPathTest, AlignUsesScratchForTerminalTransformWhenOutputAliasesTargetWithoutSpatialGridSnapshot)
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
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu =
        std::make_shared<CpuCloud>(makeTranslatedBinaryGridPoints(4096, 0.03125f, -0.015625f, 0.0078125f));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points_ptr = static_cast<const GpuCloud&>(*target).points().data();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto target_points = makeNonCollinearPoints();
    auto source_points = makeTranslatedNonCollinearPoints(target_points, 0.1f, -0.05f, 0.025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* stale_target_points = static_cast<const GpuCloud&>(*target).points().data();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> stale_normals(target->size(), 3);
    target->setNormals(std::move(stale_normals));
    target->setMaterialLibraryFile("stale.mtl");

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
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

    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud first_output;
    icp.align(first_output);

    auto& mutable_target_points = target->points();
    mutable_target_points.setValue(0, 0, mutable_target_points.getValue(0, 0) + 0.01f);
    GpuCloud second_output;
    icp.align(second_output);

    EXPECT_EQ(first_output.size(), source->size());
    EXPECT_EQ(second_output.size(), source->size());
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 2);
}

TEST(ICPGpuPathTest, AlignDoesNotIncrementTargetPointsVersionForSameBufferNoWriteOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto cpu_cloud = std::make_shared<CpuCloud>(makeNonCollinearPoints());
    auto cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    const auto initial_version = cloud->pointsVersion();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto first_target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto second_target_cpu =
        std::make_shared<CpuCloud>(makeTranslatedGridPoints(4096, 0.003f, -0.002f, 0.001f));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto first_target = std::make_shared<GpuCloud>(first_target_cpu->toGpu());
    auto second_target = std::make_shared<GpuCloud>(second_target_cpu->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
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

    auto source_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeBinaryGridPoints(4096));
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

    auto source_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
    auto target_cpu = std::make_shared<CpuCloud>(makeGridPoints(4096));
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

TEST(ICPGpuPathTest, AlignSkipsMetricUpdateForNonTerminalGpuIterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.2f, -0.1f, 0.05f);

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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.2f, -0.1f, 0.05f);

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

    using CpuCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<float, plamatrix::Device::GPU>;

    auto source_points = makeNonCollinearPoints();
    auto target_points = makeTranslatedNonCollinearPoints(source_points, 0.2f, -0.1f, 0.05f);

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
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);

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
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-8f);
    icp.setComputeFinalMetrics(false);

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

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto transform_gpu = transform.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
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

    const auto step_cpu = step_transform.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(step_cpu.getValue(row, col), expected, 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, TransformedAccumulatedAlignmentStepSkipsSpatialGridPrepareForExactPointwiseMatches)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto transform_gpu = transform.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_transform(4, 4);
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

    const auto accumulated_cpu = accumulated_transform.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(accumulated_cpu.getValue(row, col), transform.getValue(row, col), 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, TransformedExactPointwiseAlignmentStepFallsBackToSpatialGridOnMismatch)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(-0.25f, 0.125f, -0.0625f);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto transform_gpu = transform.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
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

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto cache_build_transform = makeTranslationTransform(-0.25f, 0.125f, -0.0625f);
    auto exact_transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto cache_build_transform_gpu = cache_build_transform.toGpu();
    auto exact_transform_gpu = exact_transform.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
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

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto cache_build_transform = makeTranslationTransform(-0.25f, 0.125f, -0.0625f);
    auto exact_transform = makeTranslationTransform(-0.5f, 0.25f, -0.125f);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto cache_build_transform_gpu = cache_build_transform.toGpu();
    auto exact_transform_gpu = exact_transform.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
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

    auto target_points = makeBinaryGridPoints(4096);
    auto source_points = makeTranslatedBinaryGridPoints(4096, 0.5f, -0.25f, 0.125f);
    auto transform = makeTranslationTransform(10.0f, -10.0f, 5.0f);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> accumulated_sentinel(4, 4);
    accumulated_sentinel.fill(7.0f);

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    auto transform_gpu = transform.toGpu();
    auto accumulated_transform = accumulated_sentinel.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
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

    const auto accumulated_cpu = accumulated_transform.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                accumulated_cpu.getValue(row, col),
                accumulated_sentinel.getValue(row, col),
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
    icp.setGpuProbeExactPointwiseOnFiniteRadius(true);

    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpExactPointwiseStepCallCountForTesting();
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
}

TEST(ICPGpuPathTest, AlignCanUseOrderedPointwiseCorrespondencesForFiniteRadiusTranslation)
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
    icp.setComputeFinalMetrics(false);
    icp.setGpuAssumeOrderedCorrespondences(true);

    plapoint::gpu::resetIcpExactPointwiseStepCallCountForTesting();
    plapoint::gpu::resetIcpFullDistanceEvaluationCountForTesting();
    plapoint::gpu::resetIcpTargetCandidateVisitCountForTesting();
    plapoint::gpu::resetIcpGridCellLookupCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpExactPointwiseStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpFullDistanceEvaluationCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetCandidateVisitCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpGridCellLookupCountForTesting(), 0ull);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 0);
    const auto& final_transform = icp.getFinalTransformation();
    EXPECT_NEAR(final_transform.getValue(0, 3), -0.1f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(1, 3), 0.05f, 1.0e-5f);
    EXPECT_NEAR(final_transform.getValue(2, 3), -0.025f, 1.0e-5f);
    EXPECT_EQ(output.size(), source->size());
}

TEST(ICPGpuPathTest, AlignmentStepPrefersSameBufferExactIdentityWhenOrdered)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    auto points = makeNonCollinearPoints().toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
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

    const auto step_cpu = step_transform.toCpu();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(step_cpu.getValue(row, col), expected, 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, AlignPropagatesOrderedCorrespondencesToTransformedIterations)
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
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_TRUE(icp.hasConverged());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-6f);
}

TEST(ICPGpuPathTest, AlignSkipsNextAccumulatedTransformBufferForLastTransformedIdentityStep)
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
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpTransformedAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseResidualCallCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(icp._gpu_next_T_acc, nullptr);
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
    GpuCloud output;
    icp.align(output);

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
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

    auto source_points = makeNonCollinearPoints();
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

TEST(ICPGpuPathTest, AlignReusesFullCoverageGridTranslationResultAcrossSeparateBufferCalls)
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

    plapoint::gpu::resetIcpAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTransformPointsCallCountForTesting();
    GpuCloud output;
    icp.align(output);
    icp.align(output);

    EXPECT_EQ(output.size(), source->size());
    EXPECT_NEAR(icp.getFinalRmse(), 0.0f, 1.0e-5f);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
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
    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
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

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
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

    EXPECT_EQ(plapoint::gpu::icpHostSynchronizationCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepHostResultCopyCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpAlignmentStepCallCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpResidualStatsCallCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridPrepareCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridBuildCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTransformPointsCallCountForTesting(), 0);
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
