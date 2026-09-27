#include "icp_gpu_path_test_support.h"
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_matrix.h>

#ifdef PLAPOINT_WITH_CUDA

TEST(ICPGpuPathTest, MultiplyTransform4x4UsesColumnMajorTransformComposition)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    plamatrix::MatrixXf A(4, 4);
    plamatrix::MatrixXf B(4, 4);
    A.setConstant(0.0f);
    B.setConstant(0.0f);

    A.operator()(0, 0) = 0.0f;  A.operator()(0, 1) = -1.0f; A.operator()(0, 2) = 0.0f; A.operator()(0, 3) = 2.0f;
    A.operator()(1, 0) = 1.0f;  A.operator()(1, 1) = 0.0f;  A.operator()(1, 2) = 0.0f; A.operator()(1, 3) = -1.0f;
    A.operator()(2, 0) = 0.0f;  A.operator()(2, 1) = 0.0f;  A.operator()(2, 2) = 1.0f; A.operator()(2, 3) = 0.5f;
    A.operator()(3, 0) = 0.0f;  A.operator()(3, 1) = 0.0f;  A.operator()(3, 2) = 0.0f; A.operator()(3, 3) = 1.0f;

    B.operator()(0, 0) = 1.0f;  B.operator()(0, 1) = 0.0f;  B.operator()(0, 2) = 0.0f; B.operator()(0, 3) = -3.0f;
    B.operator()(1, 0) = 0.0f;  B.operator()(1, 1) = 1.0f;  B.operator()(1, 2) = 0.0f; B.operator()(1, 3) = 4.0f;
    B.operator()(2, 0) = 0.0f;  B.operator()(2, 1) = 0.0f;  B.operator()(2, 2) = 1.0f; B.operator()(2, 3) = 1.5f;
    B.operator()(3, 0) = 0.0f;  B.operator()(3, 1) = 0.0f;  B.operator()(3, 2) = 0.0f; B.operator()(3, 3) = 1.0f;

    auto A_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(A, matrix_context);
    auto B_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(B, matrix_context);
    plamatrix::internal::ResidentMatrix<float> C_gpu(4, 4, matrix_context);

    plapoint::gpu::multiplyTransform4x4(A_gpu.data(), B_gpu.data(), C_gpu.data());
    auto C = C_gpu.toHostMatrix();
    auto expected = multiplyCpu4x4(A, B);

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(C.operator()(row, col), expected.operator()(row, col), 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, MultiplyTransform4x4AsyncUsesCallerStream)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    plamatrix::MatrixXf A(4, 4);
    plamatrix::MatrixXf B(4, 4);
    A.setConstant(0.0f);
    B.setConstant(0.0f);

    A.operator()(0, 0) = 1.0f; A.operator()(0, 3) = 1.5f;
    A.operator()(1, 1) = 1.0f; A.operator()(1, 3) = -2.0f;
    A.operator()(2, 2) = 1.0f; A.operator()(2, 3) = 0.25f;
    A.operator()(3, 3) = 1.0f;

    B.operator()(0, 0) = 0.0f;  B.operator()(0, 1) = -1.0f; B.operator()(0, 3) = 3.0f;
    B.operator()(1, 0) = 1.0f;  B.operator()(1, 1) = 0.0f;  B.operator()(1, 3) = 4.0f;
    B.operator()(2, 2) = 1.0f;  B.operator()(2, 3) = -1.0f;
    B.operator()(3, 3) = 1.0f;

    auto A_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(A, matrix_context);
    auto B_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(B, matrix_context);
    plamatrix::internal::ResidentMatrix<float> C_gpu(4, 4, matrix_context);

    cudaStream_t stream{};
    PLAPOINT_CHECK_CUDA(cudaStreamCreate(&stream));
    plapoint::gpu::multiplyTransform4x4Async(A_gpu.data(), B_gpu.data(), C_gpu.data(), stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));

    auto C = C_gpu.toHostMatrix();
    auto expected = multiplyCpu4x4(A, B);

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(C.operator()(row, col), expected.operator()(row, col), 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, SetIdentityTransform4x4WritesColumnMajorIdentity)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    plamatrix::internal::ResidentMatrix<float> identity_gpu(4, 4, matrix_context);
    plapoint::gpu::setIdentityTransform4x4(identity_gpu.data());
    auto identity = identity_gpu.toHostMatrix();

    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            const float expected = row == col ? 1.0f : 0.0f;
            EXPECT_NEAR(identity.operator()(row, col), expected, 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, StepTransformFromStatsWritesDeviceTransform)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source = makeNonCollinearPoints();
    plamatrix::MatrixXf expected(4, 4);
    expected.setConstant(0.0f);
    expected.operator()(0, 0) = 0.0f;
    expected.operator()(0, 1) = -1.0f;
    expected.operator()(0, 2) = 0.0f;
    expected.operator()(0, 3) = 2.0f;
    expected.operator()(1, 0) = 1.0f;
    expected.operator()(1, 1) = 0.0f;
    expected.operator()(1, 2) = 0.0f;
    expected.operator()(1, 3) = -1.0f;
    expected.operator()(2, 0) = 0.0f;  expected.operator()(2, 1) = 0.0f;  expected.operator()(2, 2) = 1.0f; expected.operator()(2, 3) = 0.5f;
    expected.operator()(3, 0) = 0.0f;  expected.operator()(3, 1) = 0.0f;  expected.operator()(3, 2) = 0.0f; expected.operator()(3, 3) = 1.0f;

    auto target = transformedPoints(expected, source);
    const auto stats = makeMatchedStats(source, target);
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

    const auto result = plapoint::gpu::computeIcpStepTransformFromStats(stats, step_gpu.data());
    auto step = step_gpu.toHostMatrix();

    float expected_delta = 0.0f;
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            expected_delta += std::abs(expected.operator()(row, col) - (row == col ? 1.0f : 0.0f));
        }
        expected_delta += std::abs(expected.operator()(row, 3));
    }

    EXPECT_NEAR(result.delta, expected_delta, 1.0e-5f);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(step.operator()(row, col), expected.operator()(row, col), 1.0e-5f);
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeNonCollinearPoints();
    auto target_cpu = makeTranslatedNonCollinearPoints(source_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace full_stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace full_step_workspace;
    plamatrix::internal::ResidentMatrix<float> full_step_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> compact_step_gpu(4, 4, matrix_context);
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

    auto full_step = full_step_gpu.toHostMatrix();
    auto compact_step = compact_step_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(compact_step.operator()(row, col), full_step.operator()(row, col), 1.0e-6f);
        }
    }
}

TEST(ICPGpuPathTest, StatsAndStepTransformCopiesOneHostResult)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeNonCollinearPoints();
    auto target_cpu = makeTranslatedNonCollinearPoints(source_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeNonCollinearPoints();
    plamatrix::MatrixXf source_cpu(4, 3);
    for (int col = 0; col < 3; ++col)
    {
        source_cpu.operator()(0, col) = target_cpu.operator()(1, col);
        source_cpu.operator()(1, col) = target_cpu.operator()(2, col);
        source_cpu.operator()(2, col) = target_cpu.operator()(3, col);
        source_cpu.operator()(3, col) = target_cpu.operator()(0, col);
    }
    target_cpu = padTargetWithNonFiniteRows(target_cpu);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeNonCollinearPoints();
    auto target_cpu = makeTranslatedNonCollinearPoints(source_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeNonCollinearPoints();
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridKeyInitKernelLaunchCountForTesting();
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
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridKeyInitKernelLaunchCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignmentStepSkipsTargetSpatialGridBelowTargetCountThreshold)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridKeyInitKernelLaunchCountForTesting();
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
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridKeyInitKernelLaunchCountForTesting(), 0);
}

TEST(ICPGpuPathTest, AlignmentStepSkipsTargetTileBoundsForSmallFiniteRadiusTarget)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto transform_cpu = makeTranslationTransform(-0.01f, 0.005f, -0.0025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);
    auto transform_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(transform_cpu, matrix_context);
    plamatrix::internal::ResidentMatrix<float> output_gpu(source_gpu.rows(), 3, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.operator()(1, 0) = target_cpu.operator()(1, 0) + 0.03f;
    target_cpu.operator()(2, 1) = target_cpu.operator()(2, 1) - 0.02f;
    target_cpu.operator()(3, 2) = target_cpu.operator()(3, 2) + 0.015f;
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> first_step_gpu(4, 4, matrix_context);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        first_step_gpu.data());
    ASSERT_TRUE(first_step.step_valid);

    plamatrix::internal::ResidentMatrix<float> fused_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> fused_accumulated_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> fused_output_gpu(source_gpu.rows(), 3, matrix_context);
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

    plamatrix::internal::ResidentMatrix<float> baseline_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> baseline_accumulated_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> baseline_output_gpu(source_gpu.rows(), 3, matrix_context);
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

    const auto fused_output_cpu = fused_output_gpu.toHostMatrix();
    const auto baseline_output_cpu = baseline_output_gpu.toHostMatrix();
    ASSERT_EQ(fused_output_cpu.rows(), baseline_output_cpu.rows());
    ASSERT_EQ(fused_output_cpu.cols(), baseline_output_cpu.cols());
    for (plamatrix::Index row = 0; row < fused_output_cpu.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < fused_output_cpu.cols(); ++col)
        {
            EXPECT_NEAR(
                fused_output_cpu.operator()(row, col),
                baseline_output_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.operator()(1, 0) = target_cpu.operator()(1, 0) + 0.03f;
    target_cpu.operator()(2, 1) = target_cpu.operator()(2, 1) - 0.02f;
    target_cpu.operator()(3, 2) = target_cpu.operator()(3, 2) + 0.015f;
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::internal::ResidentMatrix<float> baseline_step_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> async_step_gpu(4, 4, matrix_context);
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

    const auto baseline_step_cpu = baseline_step_gpu.toHostMatrix();
    const auto async_step_cpu = async_step_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_step_cpu.operator()(row, col),
                baseline_step_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::internal::ResidentMatrix<float> baseline_step_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> async_step_gpu(4, 4, matrix_context);
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

    const auto baseline_step_cpu = baseline_step_gpu.toHostMatrix();
    const auto async_step_cpu = async_step_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_step_cpu.operator()(row, col),
                baseline_step_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto exact_source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(exact_source_cpu, matrix_context);
    auto exact_target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(exact_target_cpu, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.operator()(1, 0) = target_cpu.operator()(1, 0) + 0.03f;
    target_cpu.operator()(2, 1) = target_cpu.operator()(2, 1) - 0.02f;
    target_cpu.operator()(3, 2) = target_cpu.operator()(3, 2) + 0.015f;
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> first_step_gpu(4, 4, matrix_context);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        first_step_gpu.data());
    ASSERT_TRUE(first_step.step_valid);

    plamatrix::internal::ResidentMatrix<float> fused_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> fused_accumulated_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> fused_output_gpu(source_gpu.rows(), 3, matrix_context);
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.operator()(1, 0) = target_cpu.operator()(1, 0) + 0.03f;
    target_cpu.operator()(2, 1) = target_cpu.operator()(2, 1) - 0.02f;
    target_cpu.operator()(3, 2) = target_cpu.operator()(3, 2) + 0.015f;
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_workspace;
    baseline_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::internal::ResidentMatrix<float> baseline_step_gpu(4, 4, matrix_context);
    const auto baseline_step = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        baseline_workspace,
        baseline_step_gpu.data());
    ASSERT_TRUE(baseline_step.step_valid);

    plamatrix::internal::ResidentMatrix<float> baseline_output_gpu(source_gpu.rows(), 3, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> async_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_output_gpu(source_gpu.rows(), 3, matrix_context);
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

    const auto baseline_step_cpu = baseline_step_gpu.toHostMatrix();
    const auto async_step_cpu = async_step_gpu.toHostMatrix();
    const auto baseline_output_cpu = baseline_output_gpu.toHostMatrix();
    const auto async_output_cpu = async_output_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_step_cpu.operator()(row, col),
                baseline_step_cpu.operator()(row, col),
                1.0e-5f);
        }
    }
    for (plamatrix::Index row = 0; row < async_output_cpu.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < async_output_cpu.cols(); ++col)
        {
            EXPECT_NEAR(
                async_output_cpu.operator()(row, col),
                baseline_output_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.operator()(1, 0) = target_cpu.operator()(1, 0) + 0.03f;
    target_cpu.operator()(2, 1) = target_cpu.operator()(2, 1) - 0.02f;
    target_cpu.operator()(3, 2) = target_cpu.operator()(3, 2) + 0.015f;
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_first_workspace;
    baseline_first_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::internal::ResidentMatrix<float> baseline_first_step_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> baseline_terminal_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> baseline_accumulated_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> baseline_output_gpu(source_gpu.rows(), 3, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> async_first_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_terminal_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_accumulated_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_output_gpu(source_gpu.rows(), 3, matrix_context);
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

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toHostMatrix();
    const auto async_accumulated_cpu = async_accumulated_gpu.toHostMatrix();
    const auto baseline_output_cpu = baseline_output_gpu.toHostMatrix();
    const auto async_output_cpu = async_output_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.operator()(row, col),
                baseline_accumulated_cpu.operator()(row, col),
                1.0e-5f);
        }
    }
    for (plamatrix::Index row = 0; row < async_output_cpu.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < async_output_cpu.cols(); ++col)
        {
            EXPECT_NEAR(
                async_output_cpu.operator()(row, col),
                baseline_output_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    plamatrix::MatrixXf target_cpu(4, 3);
    plamatrix::MatrixXf source_cpu(4, 3);
    target_cpu.operator()(0, 0) = 0.0f;
    target_cpu.operator()(0, 1) = 0.0f;
    target_cpu.operator()(0, 2) = 0.0f;
    target_cpu.operator()(1, 0) = 0.1f;
    target_cpu.operator()(1, 1) = 0.0f;
    target_cpu.operator()(1, 2) = 0.0f;
    target_cpu.operator()(2, 0) = 0.0f;
    target_cpu.operator()(2, 1) = 0.1f;
    target_cpu.operator()(2, 2) = 0.0f;
    target_cpu.operator()(3, 0) = 0.0f; target_cpu.operator()(3, 1) = 0.0f; target_cpu.operator()(3, 2) = 0.1f;
    source_cpu.operator()(0, 0) = 0.005f; source_cpu.operator()(0, 1) = 0.0f; source_cpu.operator()(0, 2) = 0.0f;
    source_cpu.operator()(1, 0) = 0.105f; source_cpu.operator()(1, 1) = 0.0f; source_cpu.operator()(1, 2) = 0.0f;
    source_cpu.operator()(2, 0) = 10.0f; source_cpu.operator()(2, 1) = 10.0f; source_cpu.operator()(2, 2) = 10.0f;
    source_cpu.operator()(3, 0) = 20.0f; source_cpu.operator()(3, 1) = 20.0f; source_cpu.operator()(3, 2) = 20.0f;
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace terminal_workspace;
    plamatrix::internal::ResidentMatrix<float> first_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> terminal_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> accumulated_gpu(4, 4, matrix_context);
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    target_cpu.operator()(1, 0) = target_cpu.operator()(1, 0) + 0.03f;
    target_cpu.operator()(2, 1) = target_cpu.operator()(2, 1) - 0.02f;
    target_cpu.operator()(3, 2) = target_cpu.operator()(3, 2) + 0.015f;
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_step_workspace;
    plamatrix::internal::ResidentMatrix<float> first_step_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> baseline_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> baseline_accumulated_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> async_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_accumulated_gpu(4, 4, matrix_context);
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

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toHostMatrix();
    const auto async_accumulated_cpu = async_accumulated_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.operator()(row, col),
                baseline_accumulated_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_step_workspace;
    plamatrix::internal::ResidentMatrix<float> first_step_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> baseline_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> baseline_accumulated_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> async_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_accumulated_gpu(4, 4, matrix_context);
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

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toHostMatrix();
    const auto async_accumulated_cpu = async_accumulated_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.operator()(row, col),
                baseline_accumulated_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeTranslatedPerturbedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace baseline_first_workspace;
    baseline_first_workspace.reserveFloatAlignmentStep(static_cast<int>(source_gpu.rows()));
    plamatrix::internal::ResidentMatrix<float> baseline_first_step_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> baseline_second_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> baseline_accumulated_gpu(4, 4, matrix_context);
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
    plamatrix::internal::ResidentMatrix<float> async_first_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_second_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> async_accumulated_gpu(4, 4, matrix_context);
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

    const auto baseline_accumulated_cpu = baseline_accumulated_gpu.toHostMatrix();
    const auto async_accumulated_cpu = async_accumulated_gpu.toHostMatrix();
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            EXPECT_NEAR(
                async_accumulated_cpu.operator()(row, col),
                baseline_accumulated_cpu.operator()(row, col),
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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeGridPoints(4096);
    plamatrix::MatrixXf source_cpu(2, 3);
    for (int row = 0; row < 2; ++row)
    {
        source_cpu.operator()(row, 0) = target_cpu.operator()(row, 0) + 0.003f;
        source_cpu.operator()(row, 1) = target_cpu.operator()(row, 1) - 0.002f;
        source_cpu.operator()(row, 2) = target_cpu.operator()(row, 2) + 0.001f;
    }
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace second_workspace;
    plamatrix::internal::ResidentMatrix<float> first_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> second_step_gpu(4, 4, matrix_context);
    plamatrix::internal::ResidentMatrix<float> accumulated_gpu(4, 4, matrix_context);
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

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
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

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    target_cpu_points.operator()(1, 0) = target_cpu_points.operator()(1, 0) + 0.03f;
    target_cpu_points.operator()(2, 1) = target_cpu_points.operator()(2, 1) - 0.02f;
    target_cpu_points.operator()(3, 2) = target_cpu_points.operator()(3, 2) + 0.015f;

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);

    plapoint::gpu::resetIcpHostSynchronizationCountForTesting();
    plapoint::gpu::resetIcpAlignmentStepHostResultCopyCountForTesting();
    plapoint::gpu::resetIcpSmallAlignmentStepKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallResidualStatsKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpSmallTerminalAlignmentResidualKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTransformedExactPointwiseAlignmentStepCallCountForTesting();
    plapoint::gpu::resetIcpResidualStatsCallCountForTesting();
    icp.align();

    EXPECT_EQ(plapoint::gpu::icpSmallAlignmentStepKernelLaunchCountForTesting(), 2);
    EXPECT_EQ(plapoint::gpu::icpSmallTerminalAlignmentResidualKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpSmallResidualStatsKernelLaunchCountForTesting(), 0);
    EXPECT_EQ(plapoint::gpu::icpTransformedExactPointwiseAlignmentStepCallCountForTesting(), 0);
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

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    target_cpu_points.operator()(1, 0) = target_cpu_points.operator()(1, 0) + 0.03f;
    target_cpu_points.operator()(2, 1) = target_cpu_points.operator()(2, 1) - 0.02f;
    target_cpu_points.operator()(3, 2) = target_cpu_points.operator()(3, 2) + 0.015f;

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
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
            EXPECT_TRUE(std::isfinite(output_cpu.points().operator()(row, col)));
        }
    }
}

TEST(ICPGpuPathTest, AlignSmallFiniteRadiusFinalMetricsWithTargetAliasOutputAvoidsScratchCopy)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }

    using CpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

    auto target_cpu_points = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting - 1);
    auto source_cpu_points = makeTranslatedNonCollinearPoints(target_cpu_points, 0.01f, -0.005f, 0.0025f);
    target_cpu_points.operator()(1, 0) = target_cpu_points.operator()(1, 0) + 0.03f;
    target_cpu_points.operator()(2, 1) = target_cpu_points.operator()(2, 1) - 0.02f;
    target_cpu_points.operator()(3, 2) = target_cpu_points.operator()(3, 2) + 0.015f;

    auto source_cpu = std::make_shared<CpuCloud>(std::move(source_cpu_points));
    auto target_cpu = std::make_shared<CpuCloud>(std::move(target_cpu_points));
    auto source = std::make_shared<GpuCloud>(source_cpu->toGpu());
    auto target = std::make_shared<GpuCloud>(target_cpu->toGpu());
    const auto* target_points = target->points().data();

    plapoint::MatrixIterativeClosestPoint<float, plamatrix::internal::Device::GPU> icp;
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
            EXPECT_TRUE(std::isfinite(output_cpu.points().operator()(row, col)));
        }
    }
}

TEST(ICPGpuPathTest, AlignmentStepUsesFusedSmallTargetKernelAtTargetCountThreshold)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeCompactNonCollinearGridPoints(kMinTargetSpatialGridRowsForTesting);
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.01f, -0.005f, 0.0025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto source_cpu = makeTranslatedGridPoints(4096, 0.003f, -0.002f, 0.001f);
    auto target_cpu = makeGridPoints(4096);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

    plapoint::gpu::resetIcpTargetSpatialGridPrepareCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridBuildCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridKeyInitKernelLaunchCountForTesting();
    plapoint::gpu::resetIcpTargetSpatialGridRunLengthEncodeCountForTesting();
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
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridKeyInitKernelLaunchCountForTesting(), 1);
    EXPECT_EQ(plapoint::gpu::icpTargetSpatialGridRunLengthEncodeCountForTesting(), 1);
}

TEST(ICPGpuPathTest, StatsAndStepCanUseOrderedCorrespondencesForFiniteRadiusTranslation)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto target_cpu = makeNonCollinearPoints();
    auto source_cpu = makeTranslatedNonCollinearPoints(target_cpu, 0.1f, -0.05f, 0.025f);
    auto source_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(source_cpu, matrix_context);
    auto target_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(target_cpu, matrix_context);

    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::internal::ResidentMatrix<float> step_gpu(4, 4, matrix_context);

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

    const auto step_cpu = step_gpu.toHostMatrix();
    EXPECT_NEAR(step_cpu.operator()(0, 3), -0.1f, 1.0e-5f);
    EXPECT_NEAR(step_cpu.operator()(1, 3), 0.05f, 1.0e-5f);
    EXPECT_NEAR(step_cpu.operator()(2, 3), -0.025f, 1.0e-5f);
}

TEST(ICPGpuPathTest, TransformPointsColumnMajorWritesCallerOwnedOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU ICP path test";
    }
    auto matrix_context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});

    auto points_cpu = makeNonCollinearPoints();
    plamatrix::MatrixXf T(4, 4);
    T.setConstant(0.0f);
    T.operator()(0, 0) = 0.0f;
    T.operator()(0, 1) = -1.0f;
    T.operator()(0, 2) = 0.0f;
    T.operator()(0, 3) = 2.0f;
    T.operator()(1, 0) = 1.0f;
    T.operator()(1, 1) = 0.0f;
    T.operator()(1, 2) = 0.0f;
    T.operator()(1, 3) = -1.0f;
    T.operator()(2, 0) = 0.0f;  T.operator()(2, 1) = 0.0f;  T.operator()(2, 2) = 1.0f; T.operator()(2, 3) = 0.5f;
    T.operator()(3, 0) = 0.0f;  T.operator()(3, 1) = 0.0f;  T.operator()(3, 2) = 0.0f; T.operator()(3, 3) = 1.0f;

    auto points_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(points_cpu, matrix_context);
    auto T_gpu = plamatrix::internal::ResidentMatrix<float>::copyFrom(T, matrix_context);
    plamatrix::internal::ResidentMatrix<float> transformed_gpu(points_cpu.rows(), 3, matrix_context);

    plapoint::gpu::transformPointsColumnMajor(
        T_gpu.data(),
        points_gpu.data(),
        static_cast<int>(points_cpu.rows()),
        transformed_gpu.data());

    auto transformed = transformed_gpu.toHostMatrix();
    auto expected = transformedPoints(T, points_cpu);
    for (plamatrix::Index row = 0; row < points_cpu.rows(); ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_NEAR(transformed.operator()(row, col), expected.operator()(row, col), 1.0e-6f);
        }
    }
}

#endif // PLAPOINT_WITH_CUDA
