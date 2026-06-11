#include <gtest/gtest.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>

#include <plamatrix/plamatrix.h>

#include <cstdint>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif

namespace
{

plapoint::PointCloud<float, plamatrix::Device::CPU> makeClusterWithOutlier()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(5, 3);
    for (int i = 0; i < 4; ++i)
    {
        points.setValue(i, 0, static_cast<float>(i) * 0.02f);
        points.setValue(i, 1, 0.0f);
        points.setValue(i, 2, 0.0f);
    }
    points.setValue(4, 0, 10.0f);
    points.setValue(4, 1, 0.0f);
    points.setValue(4, 2, 0.0f);

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        colors.setValue(i, 0, static_cast<std::uint8_t>(10 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(20 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(30 + i));
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setColors(std::move(colors));
    return cloud;
}

} // namespace

TEST(PreprocessingApiTest, VoxelDownsampleCpuInputPreservesAveragedColors)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(3, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, 0.2f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 2.0f); points.setValue(2, 1, 0.0f); points.setValue(2, 2, 0.0f);

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(3, 3);
    colors.setValue(0, 0, 40); colors.setValue(0, 1, 50); colors.setValue(0, 2, 60);
    colors.setValue(1, 0, 44); colors.setValue(1, 1, 54); colors.setValue(1, 2, 64);
    colors.setValue(2, 0, 100); colors.setValue(2, 1, 110); colors.setValue(2, 2, 120);

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    cloud.setColors(std::move(colors));

    plapoint::ProcessingReport report;
    const auto output = plapoint::voxelDownsample(
        cloud, 1.0f, plapoint::ProcessingDevice::CPU, &report);

    ASSERT_EQ(output.size(), 2u);
    ASSERT_TRUE(output.hasColors());
    EXPECT_EQ(report.requestedDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_EQ(report.usedDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_FALSE(report.usedFallback);
    EXPECT_EQ(output.colors()->getValue(0, 0), 42);
    EXPECT_EQ(output.colors()->getValue(0, 2), 62);
    EXPECT_EQ(output.colors()->getValue(1, 0), 100);
}

TEST(PreprocessingApiTest, StatisticalOutlierRemovalBuildsSearchInternally)
{
    const auto cloud = makeClusterWithOutlier();

    std::vector<int> removed;
    const auto output = plapoint::statisticalOutlierRemoval(
        cloud, 2, 0.5f, plapoint::ProcessingDevice::CPU, &removed);

    ASSERT_EQ(output.size(), 4u);
    ASSERT_EQ(removed.size(), 1u);
    EXPECT_EQ(removed.front(), 4);
    ASSERT_TRUE(output.hasColors());
    EXPECT_EQ(output.colors()->getValue(0, 0), 10);
    EXPECT_EQ(output.colors()->getValue(3, 2), 33);
}

TEST(PreprocessingApiTest, RadiusOutlierRemovalBuildsFilterInternally)
{
    const auto cloud = makeClusterWithOutlier();

    std::vector<int> removed;
    const auto output = plapoint::radiusOutlierRemoval(
        cloud, 0.1f, 2, plapoint::ProcessingDevice::CPU, &removed);

    ASSERT_EQ(output.size(), 4u);
    ASSERT_EQ(removed.size(), 1u);
    EXPECT_EQ(removed.front(), 4);
    ASSERT_TRUE(output.hasColors());
    EXPECT_EQ(output.colors()->getValue(3, 0), 13);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(PreprocessingApiTest, AutoDeviceUsesGpuWhenAvailableAndReturnsCpuCloud)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping preprocessing auto-device test";
    }

    const auto cloud = makeClusterWithOutlier();

    plapoint::ProcessingReport report;
    const auto output = plapoint::radiusOutlierRemoval(
        cloud, 0.1f, 2, plapoint::ProcessingDevice::Auto, nullptr, &report);

    ASSERT_EQ(output.size(), 4u);
    ASSERT_TRUE(output.hasColors());
    EXPECT_EQ(report.requestedDevice, plapoint::ProcessingDevice::Auto);
    EXPECT_EQ(report.usedDevice, plapoint::ProcessingDevice::GPU);
    EXPECT_FALSE(report.usedFallback);
}

TEST(PreprocessingApiTest, VoxelAutoReportsCpuFallbackForAttributedCpuInput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA device, skipping preprocessing auto-device test";
    }

    const auto cloud = makeClusterWithOutlier();

    plapoint::ProcessingReport report;
    const auto output = plapoint::voxelDownsample(
        cloud, 1.0f, plapoint::ProcessingDevice::Auto, &report);

    ASSERT_TRUE(output.hasColors());
    EXPECT_EQ(report.requestedDevice, plapoint::ProcessingDevice::Auto);
    EXPECT_EQ(report.usedDevice, plapoint::ProcessingDevice::CPU);
    EXPECT_TRUE(report.usedFallback);
    EXPECT_NE(report.fallbackReason.find("preserve attributes"), std::string::npos);
}
#endif
