#include <gtest/gtest.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

    Matrix pts(10, 3);
    for (int i = 0; i < 10; ++i)
    {
        pts.operator()(i, 0) = Scalar(i);
        pts.operator()(i, 1) = 0;
        pts.operator()(i, 2) = 0;
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> ud;
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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> ud;
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
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

    Matrix pts(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        pts.operator()(i, 0) = Scalar(i);
        pts.operator()(i, 1) = Scalar(i + 10);
        pts.operator()(i, 2) = Scalar(i + 20);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(0);

    Cloud output;
    ud.filter(output);

    ASSERT_EQ(output.size(), 3u);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ(output.points().operator()(i, 0), Scalar(i));
        EXPECT_FLOAT_EQ(output.points().operator()(i, 1), Scalar(i + 10));
        EXPECT_FLOAT_EQ(output.points().operator()(i, 2), Scalar(i + 20));
    }
}

TEST(UniformDownsampleTest, CopiesNormalsForCpuOutput)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

    Matrix pts(4, 3);
    Matrix normals(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        pts.operator()(i, 0) = Scalar(i);
        pts.operator()(i, 1) = 0;
        pts.operator()(i, 2) = 0;
        normals.operator()(i, 0) = Scalar(i + 1);
        normals.operator()(i, 1) = Scalar(i + 2);
        normals.operator()(i, 2) = Scalar(i + 3);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(normals));

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(2);

    Cloud output;
    ud.filter(output);

    ASSERT_TRUE(output.hasNormals());
    ASSERT_EQ(output.size(), 2u);
    EXPECT_FLOAT_EQ(output.normals()->operator()(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.normals()->operator()(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(output.normals()->operator()(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.normals()->operator()(1, 2), 5.0f);
}

TEST(UniformDownsampleTest, CopiesColorsAndIntensitiesForKeptPoints)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

    Matrix pts(5, 3);
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(5, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(5, 1);
    for (int i = 0; i < 5; ++i)
    {
        pts.operator()(i, 0) = Scalar(i);
        pts.operator()(i, 1) = 0;
        pts.operator()(i, 2) = 0;
        colors.operator()(i, 0) = static_cast<std::uint8_t>(10 + i);
        colors.operator()(i, 1) = static_cast<std::uint8_t>(20 + i);
        colors.operator()(i, 2) = static_cast<std::uint8_t>(30 + i);
        intensities.operator()(i, 0) = static_cast<std::uint16_t>(1000 + i);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setColors(std::move(colors));
    cloud->setIntensities(std::move(intensities));

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(2);

    Cloud output;
    ud.filter(output);

    ASSERT_EQ(output.size(), 3u);
    ASSERT_TRUE(output.hasColors());
    ASSERT_TRUE(output.hasIntensities());
    EXPECT_EQ(output.colors()->operator()(0, 0), 10);
    EXPECT_EQ(output.colors()->operator()(1, 0), 12);
    EXPECT_EQ(output.colors()->operator()(2, 2), 34);
    EXPECT_EQ(output.intensities()->operator()(0, 0), 1000);
    EXPECT_EQ(output.intensities()->operator()(1, 0), 1002);
    EXPECT_EQ(output.intensities()->operator()(2, 0), 1004);
}

TEST(UniformDownsampleTest, CopiesScalarFieldsForKeptPoints)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

    Matrix pts(5, 3);
    Matrix scalar_fields(5, 2);
    for (int i = 0; i < 5; ++i)
    {
        pts.operator()(i, 0) = Scalar(i);
        pts.operator()(i, 1) = 0;
        pts.operator()(i, 2) = 0;
        scalar_fields.operator()(i, 0) = Scalar(0.1f * i);
        scalar_fields.operator()(i, 1) = Scalar(10 + i);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setScalarFields({"error", "confidence"}, std::move(scalar_fields));

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(2);

    Cloud output;
    ud.filter(output);

    ASSERT_EQ(output.size(), 3u);
    ASSERT_TRUE(output.hasScalarFields());
    EXPECT_EQ(output.scalarFieldNames(), (std::vector<std::string>{"error", "confidence"}));
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(1, 0), 0.2f);
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(2, 0), 0.4f);
    EXPECT_FLOAT_EQ(output.scalarFields()->operator()(1, 1), 12.0f);
}

TEST(UniformDownsampleTest, PreservesKeptNonFiniteCoordinates)
{
    using Scalar = float;
    using Cloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

    Matrix pts(3, 3);
    pts.operator()(0, 0) = std::numeric_limits<Scalar>::quiet_NaN();
    pts.operator()(0, 1) = 0;
    pts.operator()(0, 2) = 0;
    pts.operator()(1, 0) = 1;
    pts.operator()(1, 1) = 1;
    pts.operator()(1, 2) = 1;
    pts.operator()(2, 0) = std::numeric_limits<Scalar>::infinity();
    pts.operator()(2, 1) = 2;
    pts.operator()(2, 2) = 2;
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> ud;
    ud.setInputCloud(cloud);
    ud.setStep(2);

    Cloud output;
    ud.filter(output);

    ASSERT_EQ(output.size(), 2u);
    EXPECT_TRUE(std::isnan(output.points().operator()(0, 0)));
    EXPECT_TRUE(std::isinf(output.points().operator()(1, 0)));
}

#ifdef PLAPOINT_WITH_CUDA
TEST(UniformDownsampleTest, GpuInputKeepsEveryNthPointAndNormals)
{
    if (!hasCudaDeviceForUniformDownsample())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU uniform downsample test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> pts(5, 3);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        pts.operator()(i, 0) = Scalar(i);
        pts.operator()(i, 1) = 0;
        pts.operator()(i, 2) = 0;
        normals.operator()(i, 0) = 0;
        normals.operator()(i, 1) = 0;
        normals.operator()(i, 2) = Scalar(i + 1);
    }
    CpuCloud cpu_cloud(std::move(pts));
    cpu_cloud.setNormals(std::move(normals));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::GPU> ud;
    ud.setInputCloud(gpu_cloud);
    ud.setStep(2);

    GpuCloud output;
    ud.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 3u);
    EXPECT_FLOAT_EQ(cpu_output.points().operator()(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(cpu_output.points().operator()(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(cpu_output.points().operator()(2, 0), 4.0f);
    ASSERT_TRUE(cpu_output.hasNormals());
    EXPECT_FLOAT_EQ(cpu_output.normals()->operator()(2, 2), 5.0f);
}

TEST(UniformDownsampleTest, GpuMatchesCpuForStepGreaterThanPointCount)
{
    if (!hasCudaDeviceForUniformDownsample())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU uniform downsample test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;
    using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> pts(2, 3);
    pts.operator()(0, 0) = 3.0f;
    pts.operator()(0, 1) = 4.0f;
    pts.operator()(0, 2) = 5.0f;
    pts.operator()(1, 0) = 6.0f;
    pts.operator()(1, 1) = 7.0f;
    pts.operator()(1, 2) = 8.0f;
    auto cpu_cloud = std::make_shared<CpuCloud>(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::CPU> cpu_ud;
    cpu_ud.setInputCloud(cpu_cloud);
    cpu_ud.setStep(5);
    CpuCloud cpu_output;
    cpu_ud.filter(cpu_output);

    plapoint::UniformDownsample<Scalar, plamatrix::internal::Device::GPU> gpu_ud;
    gpu_ud.setInputCloud(gpu_cloud);
    gpu_ud.setStep(5);
    GpuCloud gpu_output;
    gpu_ud.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    ASSERT_EQ(gpu_output_cpu.size(), 1u);
    EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(0, 0), cpu_output.points().operator()(0, 0));
    EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(0, 1), cpu_output.points().operator()(0, 1));
    EXPECT_FLOAT_EQ(gpu_output_cpu.points().operator()(0, 2), cpu_output.points().operator()(0, 2));
}
#endif
