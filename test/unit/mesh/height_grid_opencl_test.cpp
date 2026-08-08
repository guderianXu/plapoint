#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_OPENCL

#include <cstdint>

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

#endif // PLAPOINT_WITH_OPENCL
