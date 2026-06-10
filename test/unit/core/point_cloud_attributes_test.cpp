#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <string>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif

static bool hasCudaDeviceForAttributes()
{
#ifdef PLAPOINT_WITH_CUDA
    return plapoint::gpu::hasUsableCudaDevice();
#else
    return false;
#endif
}

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
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(10, 3);
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

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(10, 2);
    tex.fill(0.0f);
    cloud.setTextureCoords(std::move(tex));

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

TEST(PointCloudAttributesTest, SetFacesRejectsOutOfRangeVertexIndex)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 3);

    EXPECT_THROW(cloud.setFaces(faces), std::out_of_range);
}

TEST(PointCloudAttributesTest, SetFacesRejectsNegativeVertexIndex)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, -1);
    faces.setValue(0, 2, 2);

    EXPECT_THROW(cloud.setFaces(faces), std::out_of_range);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsMismatchedFaceCount)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(4);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(2, 3);
    face_texture_indices.fill(0);

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::runtime_error);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsNonNx3)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(4);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(4, 2);
    tex.fill(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 2);
    face_texture_indices.fill(0);

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::runtime_error);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsMissingTextureCoords)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(4);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.fill(0);

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::runtime_error);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsOutOfRangeTextureIndex)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(4);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(4, 2);
    tex.fill(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.setValue(0, 0, 0);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 4);

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::out_of_range);
}

TEST(PointCloudAttributesTest, ReplacingFacesRevalidatesFaceTextureIndices)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(4);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(4, 2);
    tex.fill(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.setValue(0, 0, 0);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 3);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> too_many_faces(2, 3);
    too_many_faces.fill(0);
    EXPECT_THROW(cloud.setFaces(std::move(too_many_faces)), std::runtime_error);
}

TEST(PointCloudAttributesTest, ReplacingTextureCoordsRevalidatesFaceTextureIndices)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(4);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> tex(4, 2);
    tex.fill(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.setValue(0, 0, 0);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 3);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> too_few_texture_coords(3, 2);
    too_few_texture_coords.fill(0.0f);
    EXPECT_THROW(cloud.setTextureCoords(std::move(too_few_texture_coords)), std::out_of_range);
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

#ifdef PLAPOINT_WITH_CUDA
TEST(PointCloudAttributesTest, CpuGpuRoundtripPreservesOptionalAttributes)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud attribute transfer test";
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    for (int i = 0; i < 3; ++i)
    {
        cloud.points().setValue(i, 0, static_cast<float>(i));
        cloud.points().setValue(i, 1, static_cast<float>(i + 10));
        cloud.points().setValue(i, 2, static_cast<float>(i + 20));
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(3, 3);
    normals.fill(0.0f);
    normals.setValue(1, 2, 1.0f);
    cloud.setNormals(std::move(normals));

    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(3, 3);
    colors.setValue(0, 0, 10);
    colors.setValue(0, 1, 20);
    colors.setValue(0, 2, 30);
    colors.setValue(1, 0, 40);
    colors.setValue(1, 1, 50);
    colors.setValue(1, 2, 60);
    colors.setValue(2, 0, 70);
    colors.setValue(2, 1, 80);
    colors.setValue(2, 2, 90);
    cloud.setColors(std::move(colors));

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coords(3, 2);
    texture_coords.setValue(0, 0, 0.1f);
    texture_coords.setValue(0, 1, 0.2f);
    texture_coords.setValue(1, 0, 0.3f);
    texture_coords.setValue(1, 1, 0.4f);
    texture_coords.setValue(2, 0, 0.5f);
    texture_coords.setValue(2, 1, 0.6f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.setValue(0, 0, 2);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 0);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    cloud.setMaterialLibraryFile("materials.mtl");
    cloud.setTextureImageFile("diffuse.png");

    auto roundtrip = cloud.toGpu().toCpu();

    ASSERT_TRUE(roundtrip.hasNormals());
    EXPECT_FLOAT_EQ(roundtrip.normals()->getValue(1, 2), 1.0f);

    ASSERT_TRUE(roundtrip.hasColors());
    EXPECT_EQ(roundtrip.colors()->getValue(2, 1), 80);

    ASSERT_TRUE(roundtrip.hasTextureCoords());
    EXPECT_FLOAT_EQ(roundtrip.textureCoords()->getValue(2, 1), 0.6f);

    ASSERT_TRUE(roundtrip.hasFaces());
    EXPECT_EQ(roundtrip.faces()->getValue(0, 2), 2);

    ASSERT_TRUE(roundtrip.hasFaceTextureIndices());
    EXPECT_EQ(roundtrip.faceTextureIndices()->getValue(0, 0), 2);

    EXPECT_EQ(roundtrip.materialLibraryFile(), "materials.mtl");
    EXPECT_EQ(roundtrip.textureImageFile(), "diffuse.png");
}

TEST(PointCloudAttributesTest, ToGpuRejectsMutableInvalidFaceTextureIndices)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coords(3, 2);
    texture_coords.fill(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.setValue(0, 0, 0);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 2);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    cloud.faceTextureIndices()->setValue(0, 2, 3);

    EXPECT_THROW((void)cloud.toGpu(), std::out_of_range);
}

TEST(PointCloudAttributesTest, ToGpuRejectsMutableInvalidPointShapeBeforeTransfer)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    cloud.points() = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>(3, 4);

    try
    {
        (void)cloud.toGpu();
        FAIL() << "Expected invalid point shape to be rejected";
    }
    catch (const std::runtime_error& ex)
    {
        EXPECT_NE(std::string(ex.what()).find("PointCloud points must be Nx3"), std::string::npos);
    }
}

TEST(PointCloudAttributesTest, ToGpuRejectsMutableInvalidFaceTextureShape)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coords(3, 2);
    texture_coords.fill(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.fill(0);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.fill(0);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> invalid_face_texture_indices(1, 2);
    invalid_face_texture_indices.fill(0);
    *cloud.faceTextureIndices() = std::move(invalid_face_texture_indices);

    EXPECT_THROW((void)cloud.toGpu(), std::runtime_error);
}

TEST(PointCloudAttributesTest, ToCpuRejectsMutableInvalidGpuFaceTextureIndices)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coords(3, 2);
    texture_coords.fill(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.setValue(0, 0, 0);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 2);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    auto gpu_cloud = cloud.toGpu();
    gpu_cloud.faceTextureIndices()->setValue(0, 1, 3);

    EXPECT_THROW((void)gpu_cloud.toCpu(), std::out_of_range);
}

TEST(PointCloudAttributesTest, ToCpuRejectsMutableInvalidGpuFaceTextureShape)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coords(3, 2);
    texture_coords.fill(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.fill(0);
    cloud.setFaces(std::move(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.fill(0);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    auto gpu_cloud = cloud.toGpu();
    auto invalid_face_texture_indices =
        plamatrix::DenseMatrix<int, plamatrix::Device::GPU>(1, 2);
    invalid_face_texture_indices.fill(0);
    *gpu_cloud.faceTextureIndices() = std::move(invalid_face_texture_indices);

    EXPECT_THROW((void)gpu_cloud.toCpu(), std::runtime_error);
}

TEST(PointCloudAttributesTest, ToCpuRejectsMutableInvalidGpuPointShapeBeforeTransfer)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    auto gpu_cloud = cloud.toGpu();
    gpu_cloud.points() = plamatrix::DenseMatrix<float, plamatrix::Device::GPU>(3, 4);

    try
    {
        (void)gpu_cloud.toCpu();
        FAIL() << "Expected invalid point shape to be rejected";
    }
    catch (const std::runtime_error& ex)
    {
        EXPECT_NE(std::string(ex.what()).find("PointCloud points must be Nx3"), std::string::npos);
    }
}
#endif
