#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_OPENCL

#include <cstdint>
#include <limits>

#include <plapoint/opencl/height_grid.h>
#include <plapoint/opencl/opencl_runtime.h>

namespace
{

using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

Cloud makeHeightPoints()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(5, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(5, 3);
    const float values[5][3] = {
        {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 2.0f}, {0.0f, 1.0f, 3.0f},
        {1.0f, 1.0f, 4.0f}, {0.45f, 0.6f, 2.5f}};
    for (int row = 0; row < 5; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            points(row, column) = values[row][column];
            colors(row, column) = static_cast<std::uint8_t>(20 * row + column);
        }
    }
    Cloud cloud(std::move(points));
    cloud.setColors(std::move(colors));
    return cloud;
}

} // namespace

TEST(HeightGridOpenClTest, MatchesCpuBilinearMeanAndColors)
{
    if (!plapoint::opencl::hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU device";
    }
    const Cloud cloud = makeHeightPoints();
    plapoint::mesh::HeightGridOptions<float> options;
    options.width = 5;
    options.height = 4;
    options.useBilinearSplat = true;
    options.elevationAggregation = plapoint::mesh::ElevationAggregation::Mean;
    const auto cpu = plapoint::mesh::buildHeightGrid(cloud, options);
    const auto executions_before = plapoint::opencl::heightGridOpenClExecutionCount();
    const auto result = plapoint::opencl::buildHeightGrid(cloud, options);
    EXPECT_GT(plapoint::opencl::heightGridOpenClExecutionCount(), executions_before);

    EXPECT_EQ(result.width, cpu.width);
    EXPECT_EQ(result.height, cpu.height);
    EXPECT_EQ(result.valid, cpu.valid);
    EXPECT_EQ(result.colors, cpu.colors);
    ASSERT_EQ(result.heights.size(), cpu.heights.size());
    for (std::size_t cell = 0; cell < result.heights.size(); ++cell)
    {
        EXPECT_NEAR(result.heights[cell], cpu.heights[cell], 1e-5f);
        EXPECT_NEAR(result.weights[cell], cpu.weights[cell], 1e-6f);
    }
}

TEST(HeightGridOpenClTest, MatchesCpuNearestMinimumAggregation)
{
    if (!plapoint::opencl::hasUsableOpenClDevice())
    {
        GTEST_SKIP() << "No usable OpenCL GPU device";
    }
    const Cloud cloud = makeHeightPoints();
    plapoint::mesh::HeightGridOptions<float> options;
    options.width = 3;
    options.height = 3;
    options.useBilinearSplat = false;
    options.elevationAggregation = plapoint::mesh::ElevationAggregation::Min;
    const auto cpu = plapoint::mesh::buildHeightGrid(cloud, options);
    const auto result = plapoint::opencl::buildHeightGrid(cloud, options);

    EXPECT_EQ(result.valid, cpu.valid);
    EXPECT_EQ(result.colors, cpu.colors);
    EXPECT_EQ(result.heights, cpu.heights);
    EXPECT_EQ(result.weights, cpu.weights);
}

TEST(HeightGridOpenClTest, RejectsNonFiniteDerivedGeometryBeforeDeviceUse)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(2, 3);
    points(0, 0) = std::numeric_limits<float>::lowest();
    points(0, 1) = 0.0f;
    points(0, 2) = 0.0f;
    points(1, 0) = std::numeric_limits<float>::max();
    points(1, 1) = 1.0f;
    points(1, 2) = 0.0f;
    const Cloud extreme_cloud(std::move(points));

    plapoint::mesh::HeightGridOptions<float> options;
    options.width = 2;
    options.height = 2;
    EXPECT_THROW(
        static_cast<void>(plapoint::opencl::buildHeightGrid(extreme_cloud, options)),
        std::overflow_error);

    auto padded_cloud = makeHeightPoints();
    options.padding = std::numeric_limits<float>::max();
    EXPECT_THROW(
        static_cast<void>(plapoint::opencl::buildHeightGrid(padded_cloud, options)),
        std::overflow_error);
}

#endif // PLAPOINT_WITH_OPENCL
