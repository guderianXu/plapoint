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

TEST(PointCloudAttributesTest, NoFacesByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasFaces());
    EXPECT_EQ(cloud.faces(), nullptr);
}

TEST(PointCloudAttributesTest, SetFaces)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces.setValue(0, 0, 0); faces.setValue(0, 1, 1); faces.setValue(0, 2, 2);
    faces.setValue(1, 0, 3); faces.setValue(1, 1, 4); faces.setValue(1, 2, 5);

    cloud.setFaces(std::move(faces));

    ASSERT_TRUE(cloud.hasFaces());
    EXPECT_EQ(cloud.faces()->rows(), 2);
    EXPECT_EQ(cloud.faces()->cols(), 3);
    EXPECT_EQ(cloud.faces()->getValue(0, 0), 0);
    EXPECT_EQ(cloud.faces()->getValue(1, 2), 5);
}

TEST(PointCloudAttributesTest, SetFacesWithTextureIndices)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0); faces.setValue(0, 1, 1); faces.setValue(0, 2, 2);
    cloud.setFaces(faces);

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> texFaces(1, 3);
    texFaces.setValue(0, 0, 5); texFaces.setValue(0, 1, 6); texFaces.setValue(0, 2, 7);
    cloud.setFaceTextureIndices(std::move(texFaces));

    ASSERT_TRUE(cloud.hasFaceTextureIndices());
    EXPECT_EQ(cloud.faceTextureIndices()->getValue(0, 0), 5);
}

TEST(PointCloudAttributesTest, SetFacesRejectsNonNx3)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 2);
    EXPECT_THROW(cloud.setFaces(faces), std::runtime_error);
}

TEST(PointCloudAttributesTest, MaterialLibraryFileEmptyByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_TRUE(cloud.materialLibraryFile().empty());
}

TEST(PointCloudAttributesTest, SetMaterialLibraryFile)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    cloud.setMaterialLibraryFile("materials.mtl");
    EXPECT_EQ(cloud.materialLibraryFile(), "materials.mtl");
}

TEST(PointCloudAttributesTest, TextureImageFileEmptyByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_TRUE(cloud.textureImageFile().empty());
}

TEST(PointCloudAttributesTest, SetTextureImageFile)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    cloud.setTextureImageFile("diffuse.png");
    EXPECT_EQ(cloud.textureImageFile(), "diffuse.png");
}
