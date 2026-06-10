#include <gtest/gtest.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <limits>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>

static bool hasCudaDeviceForVoxelGrid()
{
    return plapoint::gpu::hasUsableCudaDevice();
}
#endif

namespace
{

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> makeSignedVoxelPoints()
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(5, 3);
    points.setValue(0, 0, 1.2f); points.setValue(0, 1, 0.0f); points.setValue(0, 2, 0.0f);
    points.setValue(1, 0, -0.8f); points.setValue(1, 1, 0.0f); points.setValue(1, 2, 0.0f);
    points.setValue(2, 0, 0.2f); points.setValue(2, 1, 0.0f); points.setValue(2, 2, 0.0f);
    points.setValue(3, 0, -1.2f); points.setValue(3, 1, 0.0f); points.setValue(3, 2, 0.0f);
    points.setValue(4, 0, 1.8f); points.setValue(4, 1, 0.0f); points.setValue(4, 2, 0.0f);
    return points;
}

} // namespace

TEST(VoxelGridTest, DownsamplesUniformGrid)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    // 8 points forming a 2x2x2 cube
    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(8, 3);
    int idx = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
            {
                mat.setValue(idx, 0, Scalar(x));
                mat.setValue(idx, 1, Scalar(y));
                mat.setValue(idx, 2, Scalar(z));
                ++idx;
            }
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(2.0), Scalar(2.0), Scalar(2.0));

    Cloud output;
    vg.filter(output);
    // All 8 points in one voxel => 1 centroid
    EXPECT_EQ(output.size(), 1u);
}

TEST(VoxelGridTest, PreservesSinglePoint)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(1, 3);
    mat.setValue(0, 0, 1.0f);
    mat.setValue(0, 1, 2.0f);
    mat.setValue(0, 2, 3.0f);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(0.1), Scalar(0.1), Scalar(0.1));

    Cloud output;
    vg.filter(output);
    EXPECT_EQ(output.size(), 1u);
}

TEST(VoxelGridTest, EmptyInputReturnsEmptyOutput)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto cloud = std::make_shared<Cloud>(0);

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(Scalar(0.5), Scalar(0.5), Scalar(0.5));

    Cloud output;
    vg.filter(output);

    EXPECT_EQ(output.size(), 0u);
    EXPECT_EQ(output.points().rows(), 0);
    EXPECT_EQ(output.points().cols(), 3);
}

TEST(VoxelGridTest, AveragesRepeatedPointsIntoSingleCentroid)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(3, 3);
    mat.setValue(0, 0, 0.0f); mat.setValue(0, 1, 0.0f); mat.setValue(0, 2, 0.0f);
    mat.setValue(1, 0, 0.0f); mat.setValue(1, 1, 0.0f); mat.setValue(1, 2, 0.0f);
    mat.setValue(2, 0, 0.3f); mat.setValue(2, 1, 0.6f); mat.setValue(2, 2, 0.9f);
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 1u);
    EXPECT_NEAR(output.points().getValue(0, 0), 0.1f, 1e-6f);
    EXPECT_NEAR(output.points().getValue(0, 1), 0.2f, 1e-6f);
    EXPECT_NEAR(output.points().getValue(0, 2), 0.3f, 1e-6f);
}

TEST(VoxelGridTest, OutputsCentroidsInDeterministicVoxelKeyOrder)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    auto mat = makeSignedVoxelPoints();
    auto cloud = std::make_shared<Cloud>(std::move(mat));

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    Cloud output;
    vg.filter(output);

    ASSERT_EQ(output.size(), 4u);
    EXPECT_FLOAT_EQ(output.points().getValue(0, 0), -1.2f);
    EXPECT_FLOAT_EQ(output.points().getValue(1, 0), -0.8f);
    EXPECT_FLOAT_EQ(output.points().getValue(2, 0), 0.2f);
    EXPECT_FLOAT_EQ(output.points().getValue(3, 0), 1.5f);
}

TEST(VoxelGridTest, ThrowsOnZeroLeafSize)
{
    plapoint::VoxelGrid<float, plamatrix::Device::CPU> vg;
    EXPECT_THROW(vg.setLeafSize(0, 1, 1), std::invalid_argument);
}

TEST(VoxelGridTest, RejectsNegativeNaNAndInfiniteLeafSizes)
{
    plapoint::VoxelGrid<float, plamatrix::Device::CPU> vg;

    EXPECT_THROW(vg.setLeafSize(-1.0f, 1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(1.0f, -1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(1.0f, 1.0f, -1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(vg.setLeafSize(1.0f, std::numeric_limits<float>::infinity(), 1.0f), std::invalid_argument);
}

#ifdef PLAPOINT_WITH_CUDA
TEST(VoxelGridTest, GpuInputProducesGpuOutput)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(3, 3);
    pts.setValue(0, 0, 0.0f); pts.setValue(0, 1, 0.0f); pts.setValue(0, 2, 0.0f);
    pts.setValue(1, 0, 0.2f); pts.setValue(1, 1, 0.0f); pts.setValue(1, 2, 0.0f);
    pts.setValue(2, 0, 2.0f); pts.setValue(2, 1, 0.0f); pts.setValue(2, 2, 0.0f);

    CpuCloud cpu_cloud(std::move(pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::Device::GPU> vg;
    vg.setInputCloud(gpu_cloud);
    vg.setLeafSize(1.0f, 1.0f, 1.0f);

    GpuCloud output;
    vg.filter(output);
    auto cpu_output = output.toCpu();

    ASSERT_EQ(cpu_output.size(), 2u);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(0, 0), 0.1f);
    EXPECT_FLOAT_EQ(cpu_output.points().getValue(1, 0), 2.0f);
}

TEST(VoxelGridTest, GpuMatchesCpuForNegativeCoordinatesAndSortedOutput)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto cpu_cloud = std::make_shared<CpuCloud>(makeSignedVoxelPoints());
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> cpu_vg;
    cpu_vg.setInputCloud(cpu_cloud);
    cpu_vg.setLeafSize(1.0f, 1.0f, 1.0f);
    CpuCloud cpu_output;
    cpu_vg.filter(cpu_output);

    plapoint::VoxelGrid<Scalar, plamatrix::Device::GPU> gpu_vg;
    gpu_vg.setInputCloud(gpu_cloud);
    gpu_vg.setLeafSize(1.0f, 1.0f, 1.0f);
    GpuCloud gpu_output;
    gpu_vg.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    ASSERT_EQ(gpu_output_cpu.size(), cpu_output.size());
    for (std::size_t i = 0; i < cpu_output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().getValue(static_cast<plamatrix::Index>(i), 0),
                        cpu_output.points().getValue(static_cast<plamatrix::Index>(i), 0));
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().getValue(static_cast<plamatrix::Index>(i), 1),
                        cpu_output.points().getValue(static_cast<plamatrix::Index>(i), 1));
        EXPECT_FLOAT_EQ(gpu_output_cpu.points().getValue(static_cast<plamatrix::Index>(i), 2),
                        cpu_output.points().getValue(static_cast<plamatrix::Index>(i), 2));
    }
}

TEST(VoxelGridTest, GpuEmptyInputMatchesCpu)
{
    if (!hasCudaDeviceForVoxelGrid())
    {
        GTEST_SKIP() << "No CUDA device, skipping GPU voxel grid test";
    }

    using Scalar = float;
    using CpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    auto cpu_cloud = std::make_shared<CpuCloud>(0);
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());

    plapoint::VoxelGrid<Scalar, plamatrix::Device::CPU> cpu_vg;
    cpu_vg.setInputCloud(cpu_cloud);
    cpu_vg.setLeafSize(0.5f, 0.5f, 0.5f);
    CpuCloud cpu_output;
    cpu_vg.filter(cpu_output);

    plapoint::VoxelGrid<Scalar, plamatrix::Device::GPU> gpu_vg;
    gpu_vg.setInputCloud(gpu_cloud);
    gpu_vg.setLeafSize(0.5f, 0.5f, 0.5f);
    GpuCloud gpu_output;
    gpu_vg.filter(gpu_output);
    auto gpu_output_cpu = gpu_output.toCpu();

    EXPECT_EQ(gpu_output_cpu.size(), cpu_output.size());
    EXPECT_EQ(gpu_output_cpu.points().rows(), cpu_output.points().rows());
    EXPECT_EQ(gpu_output_cpu.points().cols(), cpu_output.points().cols());
}
#endif
