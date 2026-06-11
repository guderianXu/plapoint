#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/height_grid.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/height_grid.h>
#endif

namespace
{

using Scalar = float;
using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
using CpuMatrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

#ifdef PLAPOINT_WITH_CUDA
using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

bool hasCudaDevice()
{
    return plapoint::gpu::hasUsableCudaDevice();
}

#define SKIP_IF_NO_GPU() \
    do \
    { \
        if (!hasCudaDevice()) \
        { \
            GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU height grid test"; \
        } \
    } while (0)
#endif

CpuCloud makeSmallTerrainCloud()
{
    CpuMatrix points(5, 3);
    points.setValue(0, 0, 0.0f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 1.0f);
    points.setValue(1, 0, 1.0f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 2.0f);
    points.setValue(2, 0, 0.0f); points.setValue(2, 1, 1.0f); points.setValue(2, 2, 3.0f);
    points.setValue(3, 0, 1.0f); points.setValue(3, 1, 1.0f); points.setValue(3, 2, 4.0f);
    points.setValue(4, 0, 0.25f); points.setValue(4, 1, 0.25f); points.setValue(4, 2, 5.0f);

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(5, 3);
    colors.setValue(0, 0, 10); colors.setValue(0, 1, 20); colors.setValue(0, 2, 30);
    colors.setValue(1, 0, 40); colors.setValue(1, 1, 50); colors.setValue(1, 2, 60);
    colors.setValue(2, 0, 70); colors.setValue(2, 1, 80); colors.setValue(2, 2, 90);
    colors.setValue(3, 0, 100); colors.setValue(3, 1, 110); colors.setValue(3, 2, 120);
    colors.setValue(4, 0, 130); colors.setValue(4, 1, 140); colors.setValue(4, 2, 150);

    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(5, 1);
    intensities.setValue(0, 0, 1000);
    intensities.setValue(1, 0, 2000);
    intensities.setValue(2, 0, 3000);
    intensities.setValue(3, 0, 4000);
    intensities.setValue(4, 0, 5000);

    CpuCloud cloud(std::move(points));
    cloud.setColors(std::move(colors));
    cloud.setIntensities(std::move(intensities));
    return cloud;
}

plapoint::mesh::HeightGridOptions<Scalar> smallGridOptions()
{
    plapoint::mesh::HeightGridOptions<Scalar> options;
    options.width = 3;
    options.height = 3;
    options.padding = 0.0f;
    options.maxFillPassForFaces = 1;
    return options;
}

void expectGridNear(
    const plapoint::mesh::HeightGrid<Scalar>& actual,
    const plapoint::mesh::HeightGrid<Scalar>& expected)
{
    ASSERT_EQ(actual.width, expected.width);
    ASSERT_EQ(actual.height, expected.height);
    EXPECT_NEAR(actual.minX, expected.minX, 1.0e-6f);
    EXPECT_NEAR(actual.minY, expected.minY, 1.0e-6f);
    EXPECT_NEAR(actual.stepX, expected.stepX, 1.0e-6f);
    EXPECT_NEAR(actual.stepY, expected.stepY, 1.0e-6f);
    ASSERT_EQ(actual.heights.size(), expected.heights.size());
    ASSERT_EQ(actual.weights.size(), expected.weights.size());
    ASSERT_EQ(actual.valid.size(), expected.valid.size());
    ASSERT_EQ(actual.fillPass.size(), expected.fillPass.size());

    for (std::size_t i = 0; i < expected.heights.size(); ++i)
    {
        EXPECT_NEAR(actual.heights[i], expected.heights[i], 1.0e-5f) << "cell " << i;
        EXPECT_NEAR(actual.weights[i], expected.weights[i], 1.0e-5f) << "cell " << i;
        EXPECT_EQ(actual.valid[i], expected.valid[i]) << "cell " << i;
        EXPECT_EQ(actual.fillPass[i], expected.fillPass[i]) << "cell " << i;
    }
}

} // namespace

#ifdef PLAPOINT_WITH_CUDA

TEST(HeightGridGpuTest, BuildHeightGridMatchesCpuOnSmallCloud)
{
    SKIP_IF_NO_GPU();

    const CpuCloud cpu_cloud = makeSmallTerrainCloud();
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    const auto options = smallGridOptions();

    const auto expected = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
    const auto actual = plapoint::gpu::buildHeightGrid(gpu_cloud, options);

    expectGridNear(actual, expected);
}

TEST(HeightGridGpuTest, HeightGridToMeshPreservesColorsAndIntensitiesThroughGpuPath)
{
    SKIP_IF_NO_GPU();

    const CpuCloud cpu_cloud = makeSmallTerrainCloud();
    const GpuCloud gpu_cloud = cpu_cloud.toGpu();
    const auto options = smallGridOptions();

    auto expected_grid = plapoint::mesh::buildHeightGrid(cpu_cloud, options);
    plapoint::mesh::fillHoles(expected_grid, 1);
    const auto expected_mesh = plapoint::mesh::heightGridToMesh(expected_grid, cpu_cloud, options);

    auto actual_grid = plapoint::gpu::buildHeightGrid(gpu_cloud, options);
    plapoint::gpu::fillHoles(actual_grid, 1);
    const auto actual_mesh = plapoint::gpu::heightGridToMesh(actual_grid, gpu_cloud, options);

    ASSERT_TRUE(actual_mesh.hasColors());
    ASSERT_TRUE(actual_mesh.hasIntensities());
    ASSERT_TRUE(actual_mesh.hasFaces());
    ASSERT_TRUE(expected_mesh.hasColors());
    ASSERT_TRUE(expected_mesh.hasIntensities());
    ASSERT_TRUE(expected_mesh.hasFaces());
    ASSERT_EQ(actual_mesh.size(), expected_mesh.size());
    ASSERT_EQ(actual_mesh.faces()->rows(), expected_mesh.faces()->rows());

    for (std::size_t i = 0; i < expected_mesh.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        EXPECT_NEAR(actual_mesh.points().getValue(row, 0), expected_mesh.points().getValue(row, 0), 1.0e-5f);
        EXPECT_NEAR(actual_mesh.points().getValue(row, 1), expected_mesh.points().getValue(row, 1), 1.0e-5f);
        EXPECT_NEAR(actual_mesh.points().getValue(row, 2), expected_mesh.points().getValue(row, 2), 1.0e-5f);
        EXPECT_EQ(actual_mesh.colors()->getValue(row, 0), expected_mesh.colors()->getValue(row, 0));
        EXPECT_EQ(actual_mesh.colors()->getValue(row, 1), expected_mesh.colors()->getValue(row, 1));
        EXPECT_EQ(actual_mesh.colors()->getValue(row, 2), expected_mesh.colors()->getValue(row, 2));
        EXPECT_EQ(actual_mesh.intensities()->getValue(row, 0), expected_mesh.intensities()->getValue(row, 0));
    }
}

TEST(HeightGridGpuTest, BuildHeightGridHandlesEmptyCloudAndRejectsInvalidOptions)
{
    SKIP_IF_NO_GPU();

    const CpuCloud empty_cpu;
    const GpuCloud empty_gpu = empty_cpu.toGpu();
    const auto empty_grid = plapoint::gpu::buildHeightGrid(empty_gpu, smallGridOptions());
    EXPECT_EQ(empty_grid.width, 0);
    EXPECT_EQ(empty_grid.height, 0);
    EXPECT_TRUE(empty_grid.heights.empty());
    EXPECT_TRUE(empty_grid.valid.empty());

    const GpuCloud gpu_cloud = makeSmallTerrainCloud().toGpu();

    auto bad_padding = smallGridOptions();
    bad_padding.padding = -0.25f;
    EXPECT_THROW(
        (void)plapoint::gpu::buildHeightGrid(gpu_cloud, bad_padding),
        std::invalid_argument);

    auto bad_size = smallGridOptions();
    bad_size.width = 1;
    EXPECT_THROW(
        (void)plapoint::gpu::buildHeightGrid(gpu_cloud, bad_size),
        std::invalid_argument);
}

#else

TEST(HeightGridGpuTest, SkipsWhenBuiltWithoutCuda)
{
    GTEST_SKIP() << "PlaPoint was built without CUDA support";
}

#endif // PLAPOINT_WITH_CUDA
