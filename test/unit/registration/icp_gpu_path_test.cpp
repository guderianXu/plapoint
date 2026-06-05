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

} // namespace

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

#endif // PLAPOINT_WITH_CUDA
