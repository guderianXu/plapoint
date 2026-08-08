#include <gtest/gtest.h>

#include <stdexcept>

#ifdef PLAPOINT_WITH_OPENCL

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include <plapoint/filters/preprocessing.h>
#include <plapoint/opencl/opencl_runtime.h>
#include <plapoint/opencl/preprocessing.h>

namespace
{

using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

Cloud makeAttributedCloud()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(6, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(6, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(6, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(6, 1);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> scalar_fields(6, 2);
    for (int row = 0; row < 5; ++row)
    {
        points(row, 0) = static_cast<float>(row) * 0.02f;
        points(row, 1) = static_cast<float>(row % 2) * 0.01f;
        points(row, 2) = 0.0f;
        normals(row, 0) = static_cast<float>(row + 1);
        normals(row, 1) = 0.0f;
        normals(row, 2) = 1.0f;
        colors(row, 0) = static_cast<std::uint8_t>(10 + row + (row == 4 ? 3 : 0));
        colors(row, 1) = static_cast<std::uint8_t>(20 + row);
        colors(row, 2) = static_cast<std::uint8_t>(30 + row);
        intensities(row, 0) = static_cast<std::uint16_t>(1000 + row * 3 + (row == 4 ? 3 : 0));
        scalar_fields(row, 0) = static_cast<float>(row) * 0.25f;
        scalar_fields(row, 1) = 10.0f + static_cast<float>(row) * 1.5f;
    }
    points(5, 0) = 20.0f;
    points(5, 1) = 20.0f;
    points(5, 2) = 20.0f;
    normals(5, 0) = 99.0f;
    normals(5, 1) = 99.0f;
    normals(5, 2) = 99.0f;
    colors(5, 0) = 100;
    colors(5, 1) = 110;
    colors(5, 2) = 120;
    intensities(5, 0) = 3000;
    scalar_fields(5, 0) = 20.0f;
    scalar_fields(5, 1) = 100.0f;
    Cloud cloud(std::move(points));
    cloud.setNormals(std::move(normals));
    cloud.setColors(std::move(colors));
    cloud.setIntensities(std::move(intensities));
    cloud.setScalarFields({"error", "confidence"}, std::move(scalar_fields));
    return cloud;
}

bool hasOpenCl()
{
    return plapoint::opencl::hasUsableOpenClDevice();
}

} // namespace

TEST(PreprocessingOpenClTest, EnumeratesAndReportsSelectedGpu)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    const auto devices = plapoint::opencl::enumerateOpenClGpuDevices();
    ASSERT_FALSE(devices.empty());
    const int selected = plapoint::opencl::selectedOpenClDeviceIndex();
    ASSERT_GE(selected, 0);
    const auto found = std::find_if(devices.begin(), devices.end(), [&](const auto& device)
    {
        return device.index == static_cast<std::size_t>(selected);
    });
    ASSERT_NE(found, devices.end());
    EXPECT_EQ(found->name, plapoint::opencl::selectedOpenClDeviceName());
    EXPECT_TRUE(found->available);
    EXPECT_TRUE(found->compilerAvailable);
}

TEST(PreprocessingOpenClTest, EmptyVoxelInputPreservesAttributeSchemaWithoutDeviceUse)
{
    Cloud cloud(0);
    cloud.setNormals(plamatrix::DenseMatrix<float, plamatrix::Device::CPU>(0, 3));
    cloud.setColors(plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>(0, 3));
    cloud.setIntensities(plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>(0, 1));
    cloud.setScalarFields(
        {"error", "confidence"},
        plamatrix::DenseMatrix<float, plamatrix::Device::CPU>(0, 2));

    const auto result = plapoint::opencl::voxelDownsample(cloud, 1.0f, 1.0f, 1.0f);

    EXPECT_EQ(result.size(), 0u);
    ASSERT_TRUE(result.hasNormals());
    EXPECT_EQ(result.normals()->rows(), 0);
    EXPECT_EQ(result.normals()->cols(), 3);
    ASSERT_TRUE(result.hasColors());
    EXPECT_EQ(result.colors()->rows(), 0);
    EXPECT_EQ(result.colors()->cols(), 3);
    ASSERT_TRUE(result.hasIntensities());
    EXPECT_EQ(result.intensities()->rows(), 0);
    EXPECT_EQ(result.intensities()->cols(), 1);
    ASSERT_TRUE(result.hasScalarFields());
    EXPECT_EQ(result.scalarFieldNames(), (std::vector<std::string>{"error", "confidence"}));
    EXPECT_EQ(result.scalarFields()->rows(), 0);
    EXPECT_EQ(result.scalarFields()->cols(), 2);
}

TEST(PreprocessingOpenClTest, VoxelMatchesCpuAndPreservesAveragedAttributes)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    const Cloud cloud = makeAttributedCloud();
    const auto cpu = plapoint::voxelDownsample(
        cloud, 1.0f, plapoint::ProcessingDevice::CPU);
    plapoint::ProcessingReport report;
    const auto result = plapoint::voxelDownsample(
        cloud, 1.0f, plapoint::ProcessingDevice::OpenCL, &report);

    ASSERT_EQ(result.size(), cpu.size());
    ASSERT_TRUE(result.hasNormals());
    ASSERT_TRUE(result.hasColors());
    ASSERT_TRUE(result.hasIntensities());
    ASSERT_TRUE(result.hasScalarFields());
    EXPECT_EQ(result.scalarFieldNames(), cpu.scalarFieldNames());
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::OpenCL);
    EXPECT_FALSE(report.usedFallback);
    for (std::size_t row = 0; row < result.size(); ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            EXPECT_NEAR(
                result.points().getValue(static_cast<plamatrix::Index>(row), column),
                cpu.points().getValue(static_cast<plamatrix::Index>(row), column),
                1e-6f);
            EXPECT_FLOAT_EQ(
                result.normals()->getValue(static_cast<plamatrix::Index>(row), column),
                cpu.normals()->getValue(static_cast<plamatrix::Index>(row), column));
            EXPECT_EQ(
                result.colors()->getValue(static_cast<plamatrix::Index>(row), column),
                cpu.colors()->getValue(static_cast<plamatrix::Index>(row), column));
        }
        EXPECT_EQ(
            result.intensities()->getValue(static_cast<plamatrix::Index>(row), 0),
            cpu.intensities()->getValue(static_cast<plamatrix::Index>(row), 0));
        for (plamatrix::Index column = 0; column < result.scalarFields()->cols(); ++column)
        {
            EXPECT_FLOAT_EQ(
                result.scalarFields()->getValue(static_cast<plamatrix::Index>(row), column),
                cpu.scalarFields()->getValue(static_cast<plamatrix::Index>(row), column));
        }
    }
    EXPECT_EQ(result.colors()->getValue(0, 0), 13);
    EXPECT_EQ(result.intensities()->getValue(0, 0), 1007);
    EXPECT_FLOAT_EQ(result.normals()->getValue(0, 0), 3.0f);
    EXPECT_FLOAT_EQ(result.scalarFields()->getValue(0, 0), 0.5f);
}

TEST(PreprocessingOpenClTest, RadiusMatchesCpuAndPreservesSelectedAttributes)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    const Cloud cloud = makeAttributedCloud();
    std::vector<int> cpu_removed;
    std::vector<int> opencl_removed;
    const auto cpu = plapoint::radiusOutlierRemoval(
        cloud, 0.1f, 2, plapoint::ProcessingDevice::CPU, &cpu_removed);
    plapoint::ProcessingReport report;
    const auto result = plapoint::radiusOutlierRemoval(
        cloud, 0.1f, 2, plapoint::ProcessingDevice::OpenCL, &opencl_removed, &report);

    ASSERT_EQ(result.size(), cpu.size());
    EXPECT_EQ(opencl_removed, cpu_removed);
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::OpenCL);
    EXPECT_EQ(report.neighborBackend, plapoint::ProcessingNeighborBackend::OpenClUniformGrid);
    ASSERT_TRUE(result.hasColors());
    ASSERT_TRUE(result.hasNormals());
    for (std::size_t row = 0; row < result.size(); ++row)
    {
        EXPECT_EQ(result.colors()->getValue(static_cast<plamatrix::Index>(row), 0),
                  cpu.colors()->getValue(static_cast<plamatrix::Index>(row), 0));
        EXPECT_FLOAT_EQ(result.normals()->getValue(static_cast<plamatrix::Index>(row), 0),
                        cpu.normals()->getValue(static_cast<plamatrix::Index>(row), 0));
    }
}

TEST(PreprocessingOpenClTest, RadiusHandlesMaximumReachableGridSpan)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    using DoubleCloud = plapoint::PointCloud<double, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<double, plamatrix::Device::CPU> points(2, 3);
    points.fill(0.0);
    // These raw cells normalize to [0, INT64_MAX], so the upper query must skip +1.
    points(0, 0) = -1023.0;
    points(1, 0) = std::ldexp(1.0, 63) - 1024.0;
    const DoubleCloud cloud(std::move(points));

    try
    {
        const auto keep_mask = plapoint::opencl::radiusOutlierKeepMask(cloud, 1.0, 2);
        EXPECT_EQ(keep_mask, (std::vector<std::uint8_t>{0, 0}));
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("double precision") != std::string::npos)
        {
            GTEST_SKIP() << error.what();
        }
        throw;
    }
}

TEST(PreprocessingOpenClTest, StatisticalMatchesCpuKeepDecision)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    const Cloud cloud = makeAttributedCloud();
    std::vector<int> cpu_removed;
    std::vector<int> opencl_removed;
    const auto cpu = plapoint::statisticalOutlierRemoval(
        cloud, 3, 0.5f, plapoint::ProcessingDevice::CPU, &cpu_removed);
    plapoint::ProcessingReport report;
    const auto result = plapoint::statisticalOutlierRemoval(
        cloud, 3, 0.5f, plapoint::ProcessingDevice::OpenCL, &opencl_removed, &report);

    EXPECT_EQ(result.size(), cpu.size());
    EXPECT_EQ(opencl_removed, cpu_removed);
    EXPECT_EQ(report.actualDevice, plapoint::ProcessingDevice::OpenCL);
    EXPECT_EQ(report.neighborBackend, plapoint::ProcessingNeighborBackend::OpenClUniformGrid);
    EXPECT_FALSE(report.usedFallback);
}

TEST(PreprocessingOpenClTest, ExplicitUnsupportedKThrowsWithoutCpuFallback)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(40, 3);
    for (int row = 0; row < 40; ++row)
    {
        points(row, 0) = static_cast<float>(row);
        points(row, 1) = 0.0f;
        points(row, 2) = 0.0f;
    }
    const Cloud cloud(std::move(points));
    EXPECT_THROW(
        plapoint::statisticalOutlierRemoval(
            cloud, 32, 0.5f, plapoint::ProcessingDevice::OpenCL),
        std::invalid_argument);
}

TEST(PreprocessingOpenClTest, DoublePrecisionBackendsMatchCpuWhenSupported)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    using DoubleCloud = plapoint::PointCloud<double, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<double, plamatrix::Device::CPU> points(6, 3);
    for (int row = 0; row < 5; ++row)
    {
        points(row, 0) = static_cast<double>(row) * 0.02;
        points(row, 1) = static_cast<double>(row % 2) * 0.01;
        points(row, 2) = 0.0;
    }
    points(5, 0) = 20.0;
    points(5, 1) = 20.0;
    points(5, 2) = 20.0;
    const DoubleCloud cloud(std::move(points));

    try
    {
        const auto cpu_voxels = plapoint::voxelDownsample(
            cloud, 1.0, plapoint::ProcessingDevice::CPU);
        const auto opencl_voxels = plapoint::voxelDownsample(
            cloud, 1.0, plapoint::ProcessingDevice::OpenCL);
        ASSERT_EQ(opencl_voxels.size(), cpu_voxels.size());
        for (std::size_t row = 0; row < opencl_voxels.size(); ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                EXPECT_NEAR(
                    opencl_voxels.points().getValue(
                        static_cast<plamatrix::Index>(row), column),
                    cpu_voxels.points().getValue(
                        static_cast<plamatrix::Index>(row), column),
                    1.0e-12);
            }
        }

        std::vector<int> cpu_removed;
        std::vector<int> opencl_removed;
        static_cast<void>(plapoint::statisticalOutlierRemoval(
            cloud, 3, 0.5, plapoint::ProcessingDevice::CPU, &cpu_removed));
        static_cast<void>(plapoint::statisticalOutlierRemoval(
            cloud, 3, 0.5, plapoint::ProcessingDevice::OpenCL, &opencl_removed));
        EXPECT_EQ(opencl_removed, cpu_removed);
        cpu_removed.clear();
        opencl_removed.clear();
        static_cast<void>(plapoint::radiusOutlierRemoval(
            cloud, 0.1, 2, plapoint::ProcessingDevice::CPU, &cpu_removed));
        static_cast<void>(plapoint::radiusOutlierRemoval(
            cloud, 0.1, 2, plapoint::ProcessingDevice::OpenCL, &opencl_removed));
        EXPECT_EQ(opencl_removed, cpu_removed);
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("double precision") != std::string::npos)
        {
            GTEST_SKIP() << error.what();
        }
        throw;
    }
}

TEST(PreprocessingOpenClTest, PathologicalUniformGridWorkIsRejectedBeforeLaunch)
{
    if (!hasOpenCl()) GTEST_SKIP() << "No usable OpenCL GPU device";
    constexpr int point_count = 10001;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(point_count, 3);
    points.fill(0.0f);
    const Cloud cloud(std::move(points));

    EXPECT_THROW(
        plapoint::opencl::statisticalOutlierKeepMask(cloud, 3, 0.5f),
        std::runtime_error);
    EXPECT_THROW(
        plapoint::opencl::radiusOutlierKeepMask(cloud, 0.1f, 2),
        std::runtime_error);
}

TEST(PreprocessingOpenClTest, ExplicitInvalidDeviceIndexReportsSelectionError)
{
    const char* requested_index = std::getenv("PLAPOINT_OPENCL_DEVICE_INDEX");
    if (!requested_index || std::string(requested_index) != "999999")
    {
        GTEST_SKIP() << "Run this test with PLAPOINT_OPENCL_DEVICE_INDEX=999999";
    }
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(1, 3);
    points.fill(0.0f);
    const Cloud cloud(std::move(points));
    try
    {
        static_cast<void>(plapoint::voxelDownsample(
            cloud, 1.0f, plapoint::ProcessingDevice::OpenCL));
        FAIL() << "Expected invalid OpenCL device selection to throw";
    }
    catch (const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("PLAPOINT_OPENCL_DEVICE_INDEX"), std::string::npos);
        EXPECT_NE(message.find("999999"), std::string::npos);
    }
}

#endif // PLAPOINT_WITH_OPENCL

#ifndef PLAPOINT_WITH_OPENCL

#include <plapoint/opencl/height_grid.h>
#include <plapoint/opencl/opencl_runtime.h>
#include <plapoint/opencl/preprocessing.h>

TEST(OpenClDisabledTest, PublicApisHaveRuntimeStubs)
{
    EXPECT_TRUE(plapoint::opencl::enumerateOpenClGpuDevices().empty());
    EXPECT_FALSE(plapoint::opencl::hasUsableOpenClDevice());
    EXPECT_THROW(plapoint::opencl::requireUsableOpenClDevice(), std::runtime_error);
    EXPECT_EQ(plapoint::opencl::selectedOpenClDeviceIndex(), -1);
    EXPECT_TRUE(plapoint::opencl::selectedOpenClDeviceName().empty());
    EXPECT_EQ(plapoint::opencl::heightGridOpenClExecutionCount(), 0u);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(1, 3);
    points.fill(0.0f);
    const plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(points));
    EXPECT_THROW(
        plapoint::opencl::voxelDownsample(cloud, 1.0f, 1.0f, 1.0f),
        std::runtime_error);
}

#endif // PLAPOINT_WITH_OPENCL
