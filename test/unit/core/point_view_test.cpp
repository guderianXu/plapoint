#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointViewTest, AccessXYZ)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    cloud.points().setValue(0, 0, 1.0f);
    cloud.points().setValue(0, 1, 2.0f);
    cloud.points().setValue(0, 2, 3.0f);

    auto pt = cloud[0];
    EXPECT_FLOAT_EQ(pt.x(), 1.0f);
    EXPECT_FLOAT_EQ(pt.y(), 2.0f);
    EXPECT_FLOAT_EQ(pt.z(), 3.0f);
}

TEST(PointViewTest, AccessColors)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(3, 3);
    colors.setValue(0, 0, 10); colors.setValue(0, 1, 20); colors.setValue(0, 2, 30);
    cloud.setColors(std::move(colors));

    auto pt = cloud[0];
    EXPECT_EQ(pt.r(), 10);
    EXPECT_EQ(pt.g(), 20);
    EXPECT_EQ(pt.b(), 30);
}

TEST(PointViewTest, AccessNormals)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(3, 3);
    normals.setValue(0, 0, 0.0f); normals.setValue(0, 1, 0.0f); normals.setValue(0, 2, 1.0f);
    cloud.setNormals(std::move(normals));

    auto pt = cloud[0];
    EXPECT_FLOAT_EQ(pt.nx(), 0.0f);
    EXPECT_FLOAT_EQ(pt.ny(), 0.0f);
    EXPECT_FLOAT_EQ(pt.nz(), 1.0f);
}

TEST(PointViewTest, AccessTextureCoords)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(3, 2);
    tex.setValue(0, 0, 0.25f); tex.setValue(0, 1, 0.75f);
    cloud.setTextureCoords(std::move(tex));

    auto pt = cloud[0];
    EXPECT_FLOAT_EQ(pt.u(), 0.25f);
    EXPECT_FLOAT_EQ(pt.v(), 0.75f);
}
