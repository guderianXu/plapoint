#include <gtest/gtest.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/height_grid.h>
#include <plamatrix/plamatrix.h>

#include <cstdint>

namespace
{

using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using FloatMatrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

Cloud makeColoredPlaneCloud()
{
    FloatMatrix points(4, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 1.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 2.0f);
    points.setValue(2, 0, 0.0f); points.setValue(2, 1, 1.0f); points.setValue(2, 2, 3.0f);
    points.setValue(3, 0, 1.0f); points.setValue(3, 1, 1.0f); points.setValue(3, 2, 4.0f);

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(4, 3);
    colors.setValue(0, 0, 10); colors.setValue(0, 1, 20); colors.setValue(0, 2, 30);
    colors.setValue(1, 0, 40); colors.setValue(1, 1, 50); colors.setValue(1, 2, 60);
    colors.setValue(2, 0, 70); colors.setValue(2, 1, 80); colors.setValue(2, 2, 90);
    colors.setValue(3, 0, 100); colors.setValue(3, 1, 110); colors.setValue(3, 2, 120);

    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(4, 1);
    intensities.setValue(0, 0, 100);
    intensities.setValue(1, 0, 200);
    intensities.setValue(2, 0, 300);
    intensities.setValue(3, 0, 400);

    Cloud cloud(std::move(points));
    cloud.setColors(std::move(colors));
    cloud.setIntensities(std::move(intensities));
    return cloud;
}

} // namespace

TEST(HeightGridTest, BuildHeightGridUsesBilinearPointSplat)
{
    auto cloud = makeColoredPlaneCloud();
    plapoint::mesh::HeightGridOptions<float> options;
    options.width = 3;
    options.height = 3;
    options.padding = 0.0f;

    const auto grid = plapoint::mesh::buildHeightGrid(cloud, options);

    ASSERT_EQ(grid.width, 3);
    ASSERT_EQ(grid.height, 3);
    EXPECT_TRUE(grid.isValid(0, 0));
    EXPECT_TRUE(grid.isValid(2, 0));
    EXPECT_TRUE(grid.isValid(0, 2));
    EXPECT_TRUE(grid.isValid(2, 2));
    EXPECT_NEAR(grid.at(0, 0), 1.0f, 1.0e-6f);
    EXPECT_NEAR(grid.at(2, 0), 2.0f, 1.0e-6f);
    EXPECT_NEAR(grid.at(0, 2), 3.0f, 1.0e-6f);
    EXPECT_NEAR(grid.at(2, 2), 4.0f, 1.0e-6f);
    EXPECT_FALSE(grid.isValid(1, 1));
}

TEST(HeightGridTest, FillHolesUsesNeighborMeanAndRecordsPass)
{
    plapoint::mesh::HeightGrid<float> grid;
    grid.width = 3;
    grid.height = 3;
    grid.minX = 0.0f;
    grid.minY = 0.0f;
    grid.stepX = 1.0f;
    grid.stepY = 1.0f;
    grid.heights.assign(9, 0.0f);
    grid.weights.assign(9, 0.0f);
    grid.valid.assign(9, 0);
    grid.fillPass.assign(9, 0);
    grid.at(0, 0) = 1.0f; grid.setValid(0, 0, true);
    grid.at(2, 0) = 3.0f; grid.setValid(2, 0, true);
    grid.at(0, 2) = 5.0f; grid.setValid(0, 2, true);
    grid.at(2, 2) = 7.0f; grid.setValid(2, 2, true);

    plapoint::mesh::fillHoles(grid, 1);

    EXPECT_TRUE(grid.isValid(1, 1));
    EXPECT_NEAR(grid.at(1, 1), 4.0f, 1.0e-6f);
    EXPECT_EQ(grid.fillPassAt(1, 1), 1);
}

TEST(HeightGridTest, HeightGridToMeshCreatesColoredIntensityTriangles)
{
    auto cloud = makeColoredPlaneCloud();
    plapoint::mesh::HeightGridOptions<float> options;
    options.width = 3;
    options.height = 3;
    options.padding = 0.0f;
    options.maxFillPassForFaces = 1;

    auto grid = plapoint::mesh::buildHeightGrid(cloud, options);
    plapoint::mesh::fillHoles(grid, 1);

    auto mesh = plapoint::mesh::heightGridToMesh(grid, cloud, options);

    ASSERT_TRUE(mesh.hasFaces());
    ASSERT_TRUE(mesh.hasColors());
    ASSERT_TRUE(mesh.hasIntensities());
    EXPECT_EQ(mesh.size(), 9u);
    EXPECT_GT(mesh.faces()->rows(), 0);
    EXPECT_EQ(mesh.colors()->getValue(0, 0), 10);
    EXPECT_EQ(mesh.colors()->getValue(2, 0), 40);
    EXPECT_EQ(mesh.intensities()->getValue(0, 0), 100);
    EXPECT_EQ(mesh.intensities()->getValue(2, 0), 200);
}
