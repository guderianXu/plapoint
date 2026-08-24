#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <string>
#include <utility>

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

TEST(PointCloudAttributesTest, NoIntensitiesByDefault)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasIntensities());
    EXPECT_EQ(cloud.intensities(), nullptr);
}

TEST(PointCloudAttributesTest, SetIntensitiesCopy)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(10, 1);
    intensities.fill(1024);

    cloud.setIntensities(intensities);

    ASSERT_TRUE(cloud.hasIntensities());
    EXPECT_EQ(cloud.intensities()->getValue(0, 0), 1024);
}

TEST(PointCloudAttributesTest, SetIntensitiesMove)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(10, 1);
    intensities.setValue(0, 0, 65535);

    cloud.setIntensities(std::move(intensities));

    ASSERT_TRUE(cloud.hasIntensities());
    EXPECT_EQ(cloud.intensities()->getValue(0, 0), 65535);
}

TEST(PointCloudAttributesTest, SetIntensitiesRejectsWrongSize)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(10);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> wrong_rows(5, 1);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> wrong_cols(10, 2);

    EXPECT_THROW(cloud.setIntensities(wrong_rows), std::runtime_error);
    EXPECT_THROW(cloud.setIntensities(wrong_cols), std::runtime_error);
}

TEST(PointCloudAttributesTest, CopySettersOwnIndependentAttributeStorage)
{
    using Matrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);

    Matrix normals(3, 3);
    normals.fill(0.0f);
    normals.setValue(2, 2, 1.0f);
    cloud.setNormals(std::as_const(normals));

    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(3, 3);
    colors.fill(0);
    colors.setValue(2, 1, 80);
    cloud.setColors(std::as_const(colors));

    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(3, 1);
    intensities.fill(0);
    intensities.setValue(2, 0, 4096);
    cloud.setIntensities(std::as_const(intensities));

    Matrix scalar_fields(3, 2);
    scalar_fields.fill(0.0f);
    scalar_fields.setValue(1, 1, 0.75f);
    cloud.setScalarFields({"error", "confidence"}, std::as_const(scalar_fields));

    Matrix texture_coords(3, 2);
    texture_coords.fill(0.0f);
    texture_coords.setValue(2, 1, 0.6f);
    cloud.setTextureCoords(std::as_const(texture_coords));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(1, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    cloud.setFaces(std::as_const(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(1, 3);
    face_texture_indices.setValue(0, 0, 2);
    face_texture_indices.setValue(0, 1, 1);
    face_texture_indices.setValue(0, 2, 0);
    cloud.setFaceTextureIndices(std::as_const(face_texture_indices));

    EXPECT_NE(cloud.normals()->data(), normals.data());
    EXPECT_NE(cloud.colors()->data(), colors.data());
    EXPECT_NE(cloud.intensities()->data(), intensities.data());
    EXPECT_NE(cloud.scalarFields()->data(), scalar_fields.data());
    EXPECT_NE(cloud.textureCoords()->data(), texture_coords.data());
    EXPECT_NE(cloud.faces()->data(), faces.data());
    EXPECT_NE(cloud.faceTextureIndices()->data(), face_texture_indices.data());

    normals.setValue(2, 2, 9.0f);
    colors.setValue(2, 1, 9);
    intensities.setValue(2, 0, 9);
    scalar_fields.setValue(1, 1, 9.0f);
    texture_coords.setValue(2, 1, 0.9f);
    faces.setValue(0, 2, 0);
    face_texture_indices.setValue(0, 0, 0);

    EXPECT_FLOAT_EQ(cloud.normals()->getValue(2, 2), 1.0f);
    EXPECT_EQ(cloud.colors()->getValue(2, 1), 80);
    EXPECT_EQ(cloud.intensities()->getValue(2, 0), 4096);
    EXPECT_FLOAT_EQ(cloud.scalarFields()->getValue(1, 1), 0.75f);
    EXPECT_FLOAT_EQ(cloud.textureCoords()->getValue(2, 1), 0.6f);
    EXPECT_EQ(cloud.faces()->getValue(0, 2), 2);
    EXPECT_EQ(cloud.faceTextureIndices()->getValue(0, 0), 2);
}

TEST(PointCloudAttributesTest, CopySettersPreserveEmptyAttributeShapes)
{
    using Matrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud;
    Matrix normals(0, 3);
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors(0, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(0, 1);
    Matrix scalar_fields(0, 1);
    Matrix texture_coords(0, 2);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(0, 3);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices(0, 3);

    cloud.setNormals(std::as_const(normals));
    cloud.setColors(std::as_const(colors));
    cloud.setIntensities(std::as_const(intensities));
    cloud.setScalarFields({"confidence"}, std::as_const(scalar_fields));
    cloud.setTextureCoords(std::as_const(texture_coords));
    cloud.setFaces(std::as_const(faces));
    cloud.setFaceTextureIndices(std::as_const(face_texture_indices));

    ASSERT_NE(cloud.normals(), nullptr);
    EXPECT_EQ(cloud.normals()->rows(), 0);
    EXPECT_EQ(cloud.normals()->cols(), 3);
    ASSERT_NE(cloud.colors(), nullptr);
    EXPECT_EQ(cloud.colors()->rows(), 0);
    EXPECT_EQ(cloud.colors()->cols(), 3);
    ASSERT_NE(cloud.intensities(), nullptr);
    EXPECT_EQ(cloud.intensities()->rows(), 0);
    EXPECT_EQ(cloud.intensities()->cols(), 1);
    ASSERT_NE(cloud.scalarFields(), nullptr);
    EXPECT_EQ(cloud.scalarFields()->rows(), 0);
    EXPECT_EQ(cloud.scalarFields()->cols(), 1);
    ASSERT_NE(cloud.textureCoords(), nullptr);
    EXPECT_EQ(cloud.textureCoords()->rows(), 0);
    EXPECT_EQ(cloud.textureCoords()->cols(), 2);
    ASSERT_NE(cloud.faces(), nullptr);
    EXPECT_EQ(cloud.faces()->rows(), 0);
    EXPECT_EQ(cloud.faces()->cols(), 3);
    ASSERT_NE(cloud.faceTextureIndices(), nullptr);
    EXPECT_EQ(cloud.faceTextureIndices()->rows(), 0);
    EXPECT_EQ(cloud.faceTextureIndices()->cols(), 3);
}

TEST(PointCloudAttributesTest, SetNamedScalarFields)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> fields(3, 2);
    fields.setValue(0, 0, 0.1f);
    fields.setValue(1, 0, 0.2f);
    fields.setValue(2, 0, 0.3f);
    fields.setValue(0, 1, 10.0f);
    fields.setValue(1, 1, 20.0f);
    fields.setValue(2, 1, 30.0f);

    cloud.setScalarFields({"error", "confidence"}, std::move(fields));

    ASSERT_TRUE(cloud.hasScalarFields());
    ASSERT_NE(cloud.scalarFields(), nullptr);
    EXPECT_TRUE(cloud.hasScalarField("error"));
    EXPECT_EQ(cloud.scalarFieldIndex("confidence"), 1);
    EXPECT_EQ(cloud.scalarFieldNames().at(0), "error");
    EXPECT_FLOAT_EQ(cloud.scalarFields()->getValue(1, 0), 0.2f);
    EXPECT_FLOAT_EQ(cloud[2].scalar("confidence"), 30.0f);
}

TEST(PointCloudAttributesTest, SetNamedScalarFieldsRejectsBadShapeAndNames)
{
    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> wrongRows(2, 1);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> wrongCols(3, 2);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> ok(3, 1);

    EXPECT_THROW(cloud.setScalarFields({"error"}, wrongRows), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({"error"}, wrongCols), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({""}, ok), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({"error", "error"}, wrongCols), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({"x"}, ok), std::runtime_error);
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

    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(3, 1);
    intensities.setValue(0, 0, 17);
    intensities.setValue(1, 0, 1024);
    intensities.setValue(2, 0, 65535);
    cloud.setIntensities(std::move(intensities));

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

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> scalar_fields(3, 1);
    scalar_fields.setValue(0, 0, 0.1f);
    scalar_fields.setValue(1, 0, 0.2f);
    scalar_fields.setValue(2, 0, 0.3f);
    cloud.setScalarFields({"error"}, std::move(scalar_fields));

    auto roundtrip = cloud.toGpu().toCpu();

    ASSERT_TRUE(roundtrip.hasNormals());
    EXPECT_FLOAT_EQ(roundtrip.normals()->getValue(1, 2), 1.0f);

    ASSERT_TRUE(roundtrip.hasColors());
    EXPECT_EQ(roundtrip.colors()->getValue(2, 1), 80);

    ASSERT_TRUE(roundtrip.hasIntensities());
    EXPECT_EQ(roundtrip.intensities()->getValue(2, 0), 65535);

    ASSERT_TRUE(roundtrip.hasTextureCoords());
    EXPECT_FLOAT_EQ(roundtrip.textureCoords()->getValue(2, 1), 0.6f);

    ASSERT_TRUE(roundtrip.hasFaces());
    EXPECT_EQ(roundtrip.faces()->getValue(0, 2), 2);

    ASSERT_TRUE(roundtrip.hasFaceTextureIndices());
    EXPECT_EQ(roundtrip.faceTextureIndices()->getValue(0, 0), 2);

    EXPECT_EQ(roundtrip.materialLibraryFile(), "materials.mtl");
    EXPECT_EQ(roundtrip.textureImageFile(), "diffuse.png");

    ASSERT_TRUE(roundtrip.hasScalarFields());
    EXPECT_EQ(roundtrip.scalarFieldNames().at(0), "error");
    EXPECT_FLOAT_EQ(roundtrip.scalarFields()->getValue(2, 0), 0.3f);
}

TEST(PointCloudAttributesTest, GpuCopySettersOwnIndependentDeviceStorage)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud GPU copy setter test";
    }

    using CpuMatrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

    plapoint::PointCloud<float, plamatrix::Device::CPU> cpu_cloud(3);
    auto cloud = cpu_cloud.toGpu();

    CpuMatrix normals_cpu(3, 3);
    normals_cpu.fill(0.0f);
    normals_cpu.setValue(2, 2, 1.0f);
    auto normals = normals_cpu.toGpu();
    cloud.setNormals(std::as_const(normals));

    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors_cpu(3, 3);
    colors_cpu.fill(0);
    colors_cpu.setValue(2, 1, 80);
    auto colors = colors_cpu.toGpu();
    cloud.setColors(std::as_const(colors));

    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities_cpu(3, 1);
    intensities_cpu.fill(0);
    intensities_cpu.setValue(2, 0, 4096);
    auto intensities = intensities_cpu.toGpu();
    cloud.setIntensities(std::as_const(intensities));

    CpuMatrix scalar_fields_cpu(3, 1);
    scalar_fields_cpu.fill(0.0f);
    scalar_fields_cpu.setValue(1, 0, 0.75f);
    auto scalar_fields = scalar_fields_cpu.toGpu();
    cloud.setScalarFields({"confidence"}, std::as_const(scalar_fields));

    CpuMatrix texture_coords_cpu(3, 2);
    texture_coords_cpu.fill(0.0f);
    texture_coords_cpu.setValue(2, 1, 0.6f);
    auto texture_coords = texture_coords_cpu.toGpu();
    cloud.setTextureCoords(std::as_const(texture_coords));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces_cpu(1, 3);
    faces_cpu.setValue(0, 0, 0);
    faces_cpu.setValue(0, 1, 1);
    faces_cpu.setValue(0, 2, 2);
    auto faces = faces_cpu.toGpu();
    cloud.setFaces(std::as_const(faces));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices_cpu(1, 3);
    face_texture_indices_cpu.setValue(0, 0, 2);
    face_texture_indices_cpu.setValue(0, 1, 1);
    face_texture_indices_cpu.setValue(0, 2, 0);
    auto face_texture_indices = face_texture_indices_cpu.toGpu();
    cloud.setFaceTextureIndices(std::as_const(face_texture_indices));

    EXPECT_NE(cloud.normals()->data(), normals.data());
    EXPECT_NE(cloud.colors()->data(), colors.data());
    EXPECT_NE(cloud.intensities()->data(), intensities.data());
    EXPECT_NE(cloud.scalarFields()->data(), scalar_fields.data());
    EXPECT_NE(cloud.textureCoords()->data(), texture_coords.data());
    EXPECT_NE(cloud.faces()->data(), faces.data());
    EXPECT_NE(cloud.faceTextureIndices()->data(), face_texture_indices.data());

    normals.setValue(2, 2, 9.0f);
    colors.setValue(2, 1, 9);
    intensities.setValue(2, 0, 9);
    scalar_fields.setValue(1, 0, 9.0f);
    texture_coords.setValue(2, 1, 0.9f);
    faces.setValue(0, 2, 0);
    face_texture_indices.setValue(0, 0, 0);

    const auto stored = cloud.toCpu();
    EXPECT_FLOAT_EQ(stored.normals()->getValue(2, 2), 1.0f);
    EXPECT_EQ(stored.colors()->getValue(2, 1), 80);
    EXPECT_EQ(stored.intensities()->getValue(2, 0), 4096);
    EXPECT_FLOAT_EQ(stored.scalarFields()->getValue(1, 0), 0.75f);
    EXPECT_FLOAT_EQ(stored.textureCoords()->getValue(2, 1), 0.6f);
    EXPECT_EQ(stored.faces()->getValue(0, 2), 2);
    EXPECT_EQ(stored.faceTextureIndices()->getValue(0, 0), 2);
}

TEST(PointCloudAttributesTest, GpuCopySettersPreserveEmptyAttributeShapes)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping empty point-cloud GPU copy setter test";
    }

    using CpuMatrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

    plapoint::PointCloud<float, plamatrix::Device::CPU> cpu_cloud;
    auto cloud = cpu_cloud.toGpu();

    CpuMatrix normals_cpu(0, 3);
    auto normals = normals_cpu.toGpu();
    plamatrix::DenseMatrix<uint8_t, plamatrix::Device::CPU> colors_cpu(0, 3);
    auto colors = colors_cpu.toGpu();
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities_cpu(0, 1);
    auto intensities = intensities_cpu.toGpu();
    CpuMatrix scalar_fields_cpu(0, 1);
    auto scalar_fields = scalar_fields_cpu.toGpu();
    CpuMatrix texture_coords_cpu(0, 2);
    auto texture_coords = texture_coords_cpu.toGpu();
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces_cpu(0, 3);
    auto faces = faces_cpu.toGpu();
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> face_texture_indices_cpu(0, 3);
    auto face_texture_indices = face_texture_indices_cpu.toGpu();

    cloud.setNormals(std::as_const(normals));
    cloud.setColors(std::as_const(colors));
    cloud.setIntensities(std::as_const(intensities));
    cloud.setScalarFields({"confidence"}, std::as_const(scalar_fields));
    cloud.setTextureCoords(std::as_const(texture_coords));
    cloud.setFaces(std::as_const(faces));
    cloud.setFaceTextureIndices(std::as_const(face_texture_indices));

    const auto stored = cloud.toCpu();
    ASSERT_NE(stored.normals(), nullptr);
    EXPECT_EQ(stored.normals()->rows(), 0);
    EXPECT_EQ(stored.normals()->cols(), 3);
    ASSERT_NE(stored.colors(), nullptr);
    EXPECT_EQ(stored.colors()->rows(), 0);
    EXPECT_EQ(stored.colors()->cols(), 3);
    ASSERT_NE(stored.intensities(), nullptr);
    EXPECT_EQ(stored.intensities()->rows(), 0);
    EXPECT_EQ(stored.intensities()->cols(), 1);
    ASSERT_NE(stored.scalarFields(), nullptr);
    EXPECT_EQ(stored.scalarFields()->rows(), 0);
    EXPECT_EQ(stored.scalarFields()->cols(), 1);
    ASSERT_NE(stored.textureCoords(), nullptr);
    EXPECT_EQ(stored.textureCoords()->rows(), 0);
    EXPECT_EQ(stored.textureCoords()->cols(), 2);
    ASSERT_NE(stored.faces(), nullptr);
    EXPECT_EQ(stored.faces()->rows(), 0);
    EXPECT_EQ(stored.faces()->cols(), 3);
    ASSERT_NE(stored.faceTextureIndices(), nullptr);
    EXPECT_EQ(stored.faceTextureIndices()->rows(), 0);
    EXPECT_EQ(stored.faceTextureIndices()->cols(), 3);
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
