#include <gtest/gtest.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cmath>
#include <limits>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>

static bool hasCudaDeviceForUniformDownsample()
{
    return plapoint::gpu::hasUsableCudaDevice();
}
#endif

TEST(UniformDownsampleTest, KeepsEveryNthPoint)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(10, 3);
    for (int i = 0; i < 10; ++i) { pts.setValue(i, 0, Scalar(i)); pts.setValue(i, 1, 0); pts.setValue(i, 2, 0); }
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(3);

    Cloud output;
    ud.filter(output);
    // Every 3rd: indices 0, 3, 6, 9 = 4 points
    EXPECT_EQ(output.size(), 4u);
}

TEST(UniformDownsampleTest, EmptyInputReturnsEmptyOutput)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(3);

    Cloud output;
    ud.filter(output);

    EXPECT_EQ(output.size(), 0u);
    EXPECT_EQ(output.points().rows(), 0);
    EXPECT_EQ(output.points().cols(), 3);
    EXPECT_FALSE(output.hasNormals());
}

TEST(UniformDownsampleTest, StepLessThanOneClampsToOne)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i + 10));
        pts.setValue(i, 2, Scalar(i + 20));
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(0);

    Cloud output;
    ud.filter(output);

    ASSERT_EQ(output.size(), 3u);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ(output.points().getValue(i, 0), Scalar(i));
        EXPECT_FLOAT_EQ(output.points().getValue(i, 1), Scalar(i + 10));
        EXPECT_FLOAT_EQ(output.points().getValue(i, 2), Scalar(i + 20));
    }
}

TEST(UniformDownsampleTest, CopiesNormalsForCpuOutput)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    Matrix normals(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, 0);
        pts.setValue(i, 2, 0);
        normals.setValue(i, 0, Scalar(i + 1));
        normals.setValue(i, 1, Scalar(i + 2));
        normals.setValue(i, 2, Scalar(i + 3));
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(normals));

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(2);

    Cloud output;
    ud.filter(output);

    ASSERT_TRUE(output.hasNormals());
    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output.normals()->getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.normals()->getValue(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(output.normals()->getValue(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.normals()->getValue(1, 2), 5.0f);
}

TEST(UniformDownsampleTest, PreservesKeptNonFiniteCoordinates)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(3, 3);
    pts.setValue(0, 0, std::numeric_limits<Scalar>::quiet_NaN());
    pts.setValue(0, 1, 0);
    pts.setValue(0, 2, 0);
    pts.setValue(1, 0, 1);
    pts.setValue(1, 1, 1);
    pts.setValue(1, 2, 1);
    pts.setValue(2, 0, std::numeric_limits<Scalar>::infinity());
    pts.setValue(2, 1, 2);
    pts.setValue(2, 2, 2);
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(2);

    Cloud output;
    ud.filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_TRUE(std::isnan(output.points().getValue(0, 0)));
    EXPECT_TRUE(std::isinf(output.points().getValue(1, 0)));
}

#ifdef PLAPOINT_WITH_CUDA
TEST(UniformDownsampleTest, GpuInputKeepsEveryNthPointAndNormals)
{
    if (!hasCudaDeviceForUniformDownsample())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU uniform downsample test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(5, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, 0);
        pts.setValue(i, 2, 0);
        normals.setValue(i, 0, 0);
        normals.setValue(i, 1, 0);
        normals.setValue(i, 2, Scalar(i + 1));
    }
    CpuCloud cpu_cloud(std::move(pts));
    cpu_cloud.setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::UniformDownsample<Scalar, plamatrix::Device::GPU> ud;
    ud.setInputCloud(gpu_cloud);
    ud.setStep(2);

    GpuCloud output;
    ud.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 3u);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(2, 0), 4.0f);
    ASSERT_TRUE(cpu_output.hasNormals());
    EXPECT_FLOAT_EQ(cpu_output.normals()->getValue(2, 2), 5.0f);
}

TEST(UniformDownsampleTest, GpuMatchesCpuForStepGreaterThanPointCount)
{
    if (!hasCudaDeviceForUniformDownsample())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU uniform downsample test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(2, 3);
    pts.setValue(0, 0, 3.0f); pts.setValue(0, 1, 4.0f); pts.setValue(0, 2, 5.0f);
    pts.setValue(1, 0, 6.0f); pts.setValue(1, 1, 7.0f); pts.setValue(1, 2, 8.0f);
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::UniformDownsample<Scalar, plamatrix::Device::CPU> cpu_ud;
    cpu_ud.setInputCloud(cpu_cloud);
    cpu_ud.setStep(5);
    CpuCloud cpu_output;
    cpu_ud.filter(cpu_output);

    plapoint::UniformDownsample<Scalar, plamatrix::Device::GPU> gpu_ud;
    gpu_ud.setInputCloud(gpu_cloud);
    gpu_ud.setStep(5);
    GpuCloud gpu_output;
    gpu_ud.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    ASSERT_EQ(gpu_output_cpu.size(), 1u);
    EXPECT_FLOAT_EQ(gpu_output_cpu.points().getValue(0, 0), cpu_output.points().getValue(0, 0));
    EXPECT_FLOAT_EQ(gpu_output_cpu.points().getValue(0, 1), cpu_output.points().getValue(0, 1));
    EXPECT_FLOAT_EQ(gpu_output_cpu.points().getValue(0, 2), cpu_output.points().getValue(0, 2));
}
#endif
