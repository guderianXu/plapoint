#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>

TEST(PointCloudAttributesTest, NoColorsByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasColors());
    EXPECT_EQ(cloud.colors(), nullptr);
}

TEST(PointCloudAttributesTest, SetColorsCopy)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(10, 3);
    colors.fill(128);

    cloud.setColors(colors);

    ASSERT_TRUE(cloud.hasColors());
    EXPECT_EQ(cloud.colors()->getValue(0, 0), 128);
    EXPECT_EQ(cloud.colors()->getValue(0, 1), 128);
    EXPECT_EQ(cloud.colors()->getValue(0, 2), 128);
}

TEST(PointCloudAttributesTest, SetColorsMove)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(10, 3);
    colors.setValue(0, 0, 255);

    cloud.setColors(std::move(colors));

    ASSERT_TRUE(cloud.hasColors());
    EXPECT_EQ(cloud.colors()->getValue(0, 0), 255);
}

TEST(PointCloudAttributesTest, SetColorsRejectsWrongSize)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(5, 3);
    EXPECT_THROW(cloud.setColors(colors), std::runtime_error);
}

TEST(PointCloudAttributesTest, NoTextureCoordsByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasTextureCoords());
    EXPECT_EQ(cloud.textureCoords(), nullptr);
}

TEST(PointCloudAttributesTest, SetTextureCoords)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(10, 2);
    tex.setValue(0, 0, 0.5f);
    tex.setValue(0, 1, 0.75f);

    cloud.setTextureCoords(std::move(tex));

    ASSERT_TRUE(cloud.hasTextureCoords());
    EXPECT_FLOAT_EQ(cloud.textureCoords()->getValue(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(cloud.textureCoords()->getValue(0, 1), 0.75f);
}

TEST(PointCloudAttributesTest, SetTextureCoordsRejectsWrongSize)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(5, 2);
    EXPECT_THROW(cloud.setTextureCoords(tex), std::runtime_error);
}
