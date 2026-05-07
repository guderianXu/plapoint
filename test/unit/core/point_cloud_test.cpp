#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointCloudTest, CpuCreation)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(100);
    EXPECT_EQ(cloud.size(), 100);
    EXPECT_EQ(cloud.points().rows(), 100);
    EXPECT_EQ(cloud.points().cols(), 3);
}

TEST(PointCloudTest, GpuTransfer)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cpu_cloud(10);
    cpu_cloud.points().fill(1.0f);

    auto gpu_cloud = cpu_cloud.toGpu();
    EXPECT_EQ(gpu_cloud.size(), 10);

    auto cpu_cloud_back = gpu_cloud.toCpu();
    EXPECT_FLOAT_EQ(cpu_cloud_back.points().getValue(0, 0), 1.0f);
}

TEST(PointCloudTest, MoveFromMatrix)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> mat(50, 3);
    mat.fill(3.0f);
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(std::move(mat));
    EXPECT_EQ(cloud.size(), 50);
    EXPECT_FLOAT_EQ(cloud.points().getValue(0, 0), 3.0f);
}

TEST(PointCloudTest, RejectsNonNx3Matrix)
{
    using CloudType = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> bad(5, 4);
    EXPECT_THROW(CloudType cloud(std::move(bad)), std::runtime_error);
}
