#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cmath>
#include <cstdint>
#include <limits>

#include <plamatrix/plamatrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_cluster.h>
#include <plapoint/mesh/mesh_processing.h>

namespace
{

bool hasCudaDeviceForVoxelCluster()
{
    return plapoint::gpu::hasUsableCudaDevice();
}

using CpuCloudF = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using GpuCloudF = plapoint::PointCloud<float, plamatrix::Device::GPU>;

CpuCloudF makeClusteredMeshWithAttributes()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(6, 3);
    points.setValue(0, 0, -0.20f); points.setValue(0, 1, 0.00f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, -0.10f); points.setValue(1, 1, 0.10f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 0.80f); points.setValue(2, 1, 0.00f); points.setValue(2, 2, 0.0f);
    points.setValue(3, 0, 0.90f); points.setValue(3, 1, 0.10f); points.setValue(3, 2, 0.0f);
    points.setValue(4, 0, 0.80f); points.setValue(4, 1, 0.60f); points.setValue(4, 2, 0.0f);
    points.setValue(5, 0, 0.90f); points.setValue(5, 1, 0.70f); points.setValue(5, 2, 0.0f);

    CpuCloudF mesh(std::move(points));

    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(6, 3);
    colors.setValue(0, 0, 10); colors.setValue(0, 1, 20); colors.setValue(0, 2, 30);
    colors.setValue(1, 0, 14); colors.setValue(1, 1, 24); colors.setValue(1, 2, 34);
    colors.setValue(2, 0, 50); colors.setValue(2, 1, 60); colors.setValue(2, 2, 70);
    colors.setValue(3, 0, 52); colors.setValue(3, 1, 62); colors.setValue(3, 2, 72);
    colors.setValue(4, 0, 90); colors.setValue(4, 1, 100); colors.setValue(4, 2, 110);
    colors.setValue(5, 0, 94); colors.setValue(5, 1, 104); colors.setValue(5, 2, 114);
    mesh.setColors(std::move(colors));

    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(6, 1);
    intensities.setValue(0, 0, 1000);
    intensities.setValue(1, 0, 1004);
    intensities.setValue(2, 0, 2000);
    intensities.setValue(3, 0, 2002);
    intensities.setValue(4, 0, 3000);
    intensities.setValue(5, 0, 3004);
    mesh.setIntensities(std::move(intensities));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(3, 3);
    faces.setValue(0, 0, 0); faces.setValue(0, 1, 2); faces.setValue(0, 2, 4);
    faces.setValue(1, 0, 1); faces.setValue(1, 1, 3); faces.setValue(1, 2, 5);
    faces.setValue(2, 0, 0); faces.setValue(2, 1, 1); faces.setValue(2, 2, 2);
    mesh.setFaces(std::move(faces));

    return mesh;
}

void expectSameMeshAsCpu(const CpuCloudF& actual, const CpuCloudF& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t row = 0; row < expected.size(); ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_NEAR(
                actual.points().getValue(static_cast<plamatrix::Index>(row), col),
                expected.points().getValue(static_cast<plamatrix::Index>(row), col),
                1.0e-6f);
        }
    }

    ASSERT_EQ(actual.hasColors(), expected.hasColors());
    ASSERT_EQ(actual.hasIntensities(), expected.hasIntensities());
    ASSERT_TRUE(actual.hasColors());
    ASSERT_TRUE(actual.hasIntensities());
    for (std::size_t row = 0; row < expected.size(); ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_EQ(
                actual.colors()->getValue(static_cast<plamatrix::Index>(row), col),
                expected.colors()->getValue(static_cast<plamatrix::Index>(row), col));
        }
        EXPECT_EQ(
            actual.intensities()->getValue(static_cast<plamatrix::Index>(row), 0),
            expected.intensities()->getValue(static_cast<plamatrix::Index>(row), 0));
    }

    ASSERT_EQ(actual.hasFaces(), expected.hasFaces());
    ASSERT_TRUE(actual.hasFaces());
    ASSERT_EQ(actual.faces()->rows(), expected.faces()->rows());
    ASSERT_EQ(actual.faces()->cols(), 3);
    for (plamatrix::Index row = 0; row < actual.faces()->rows(); ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_EQ(actual.faces()->getValue(row, col), expected.faces()->getValue(row, col));
            EXPECT_GE(actual.faces()->getValue(row, col), 0);
            EXPECT_LT(actual.faces()->getValue(row, col), static_cast<int>(actual.size()));
        }
    }
}

} // namespace

TEST(VoxelClusterGpuTest, MatchesCpuForSmallMeshAndPreservesColorAndIntensity)
{
    if (!hasCudaDeviceForVoxelCluster())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel cluster test";
    }

    auto cpu_mesh = makeClusteredMeshWithAttributes();
    const auto expected = plapoint::mesh::voxelClusterSimplify(cpu_mesh, 0.5f);

    const GpuCloudF gpu_mesh = cpu_mesh.toGpu();
    const auto actual = plapoint::mesh::voxelClusterSimplify(gpu_mesh, 0.5f).toCpu();

    expectSameMeshAsCpu(actual, expected);
}

TEST(VoxelClusterGpuTest, RejectsDegenerateClusterSizeLikeCpu)
{
    if (!hasCudaDeviceForVoxelCluster())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel cluster test";
    }

    const auto cpu_mesh = makeClusteredMeshWithAttributes();
    const GpuCloudF gpu_mesh = cpu_mesh.toGpu();

    EXPECT_THROW(
        (void)plapoint::mesh::voxelClusterSimplify(gpu_mesh, 0.0f),
        std::invalid_argument);
    EXPECT_THROW(
        (void)plapoint::mesh::voxelClusterSimplify(gpu_mesh, -0.5f),
        std::invalid_argument);
    EXPECT_THROW(
        (void)plapoint::mesh::voxelClusterSimplify(
            gpu_mesh,
            std::numeric_limits<float>::quiet_NaN()),
        std::invalid_argument);
}

#endif // PLAPOINT_WITH_CUDA
