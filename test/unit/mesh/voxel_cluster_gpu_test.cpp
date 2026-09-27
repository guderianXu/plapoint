#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cmath>
#include <cstdint>
#include <limits>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/detail/geometry_bridge.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_cluster.h>
#include <plapoint/mesh/mesh_processing.h>

namespace
{

bool hasCudaDeviceForVoxelCluster()
{
    return plapoint::gpu::hasUsableCudaDevice();
}

using CpuCloudF = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
using GpuCloudF = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>;

CpuCloudF makeClusteredMeshWithAttributes()
{
    plamatrix::MatrixXf points(6, 3);
    points.operator()(0, 0) = -0.20f;
    points.operator()(0, 1) = 0.00f;
    points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = -0.10f;
    points.operator()(1, 1) = 0.10f;
    points.operator()(1, 2) = 0.0f;
    points.operator()(2, 0) = 0.80f;
    points.operator()(2, 1) = 0.00f;
    points.operator()(2, 2) = 0.0f;
    points.operator()(3, 0) = 0.90f; points.operator()(3, 1) = 0.10f; points.operator()(3, 2) = 0.0f;
    points.operator()(4, 0) = 0.80f; points.operator()(4, 1) = 0.60f; points.operator()(4, 2) = 0.0f;
    points.operator()(5, 0) = 0.90f; points.operator()(5, 1) = 0.70f; points.operator()(5, 2) = 0.0f;

    CpuCloudF mesh(std::move(points));

    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(6, 3);
    colors.operator()(0, 0) = 10;
    colors.operator()(0, 1) = 20;
    colors.operator()(0, 2) = 30;
    colors.operator()(1, 0) = 14;
    colors.operator()(1, 1) = 24;
    colors.operator()(1, 2) = 34;
    colors.operator()(2, 0) = 50;
    colors.operator()(2, 1) = 60;
    colors.operator()(2, 2) = 70;
    colors.operator()(3, 0) = 52; colors.operator()(3, 1) = 62; colors.operator()(3, 2) = 72;
    colors.operator()(4, 0) = 90; colors.operator()(4, 1) = 100; colors.operator()(4, 2) = 110;
    colors.operator()(5, 0) = 94;
    colors.operator()(5, 1) = 104;
    colors.operator()(5, 2) = 114;
    mesh.setColors(std::move(colors));

    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(6, 1);
    intensities.operator()(0, 0) = 1000;
    intensities.operator()(1, 0) = 1004;
    intensities.operator()(2, 0) = 2000;
    intensities.operator()(3, 0) = 2002;
    intensities.operator()(4, 0) = 3000;
    intensities.operator()(5, 0) = 3004;
    mesh.setIntensities(std::move(intensities));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(3, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 2;
    faces.operator()(0, 2) = 4;
    faces.operator()(1, 0) = 1;
    faces.operator()(1, 1) = 3;
    faces.operator()(1, 2) = 5;
    faces.operator()(2, 0) = 0;
    faces.operator()(2, 1) = 1;
    faces.operator()(2, 2) = 2;
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
                actual.points().operator()(static_cast<plamatrix::Index>(row), col),
                expected.points().operator()(static_cast<plamatrix::Index>(row), col),
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
                actual.colors()->operator()(static_cast<plamatrix::Index>(row), col),
                expected.colors()->operator()(static_cast<plamatrix::Index>(row), col));
        }
        EXPECT_EQ(
            actual.intensities()->operator()(static_cast<plamatrix::Index>(row), 0),
            expected.intensities()->operator()(static_cast<plamatrix::Index>(row), 0));
    }

    ASSERT_EQ(actual.hasFaces(), expected.hasFaces());
    ASSERT_TRUE(actual.hasFaces());
    ASSERT_EQ(actual.faces()->rows(), expected.faces()->rows());
    ASSERT_EQ(actual.faces()->cols(), 3);
    for (plamatrix::Index row = 0; row < actual.faces()->rows(); ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_EQ(actual.faces()->operator()(row, col), expected.faces()->operator()(row, col));
            EXPECT_GE(actual.faces()->operator()(row, col), 0);
            EXPECT_LT(actual.faces()->operator()(row, col), static_cast<int>(actual.size()));
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
    const auto expected = plapoint::detail::toDeviceCloud(
        plapoint::mesh::voxelClusterSimplify(plapoint::detail::fromDeviceCloud(cpu_mesh), 0.5f));

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

    EXPECT_THROW((void)plapoint::mesh::voxelClusterSimplify(gpu_mesh, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)plapoint::mesh::voxelClusterSimplify(gpu_mesh, -0.5f), std::invalid_argument);
    EXPECT_THROW((void)plapoint::mesh::voxelClusterSimplify(gpu_mesh, std::numeric_limits<float>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(VoxelClusterGpuTest, RejectsNonFinitePoints)
{
    if (!hasCudaDeviceForVoxelCluster())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel cluster test";
    }

    plamatrix::MatrixXf points(2, 3);
    points.operator()(0, 0) = 0.0f;
    points.operator()(0, 1) = 0.0f;
    points.operator()(0, 2) = 0.0f;
    points.operator()(1, 0) = 1.0f;
    points.operator()(1, 1) = std::numeric_limits<float>::quiet_NaN();
    points.operator()(1, 2) = 1.0f;
    const GpuCloudF gpu_mesh(CpuCloudF(std::move(points)).toGpu());

    EXPECT_THROW((void)plapoint::mesh::voxelClusterSimplify(gpu_mesh, 0.5f), std::invalid_argument);
}

#endif // PLAPOINT_WITH_CUDA
