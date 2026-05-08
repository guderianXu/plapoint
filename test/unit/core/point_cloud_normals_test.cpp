#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointCloudTest, OptionalNormals)
{
    using Scalar = float;
    plapoint::PointCloud<Scalar, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasNormals());

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> n(10, 3);
    n.fill(1.0f);
    cloud.setNormals(std::move(n));
    EXPECT_TRUE(cloud.hasNormals());
    EXPECT_EQ(cloud.normals()->rows(), 10);
    EXPECT_FLOAT_EQ(cloud.normals()->getValue(0, 0), 1.0f);
}

TEST(PointCloudTest, NormalsWrongSize)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(5);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> bad(3, 3);
    bad.fill(0);
    EXPECT_THROW(cloud.setNormals(bad), std::runtime_error);
}
