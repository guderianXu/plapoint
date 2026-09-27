#include <gtest/gtest.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_matrix.h>
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
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasColors());
    EXPECT_EQ(cloud.colors(), nullptr);
}

TEST(PointCloudAttributesTest, SetColorsCopy)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(10, 3);
    colors.setConstant(128);

    cloud.setColors(colors);

    ASSERT_TRUE(cloud.hasColors());
    EXPECT_EQ(cloud.colors()->operator()(0, 0), 128);
    EXPECT_EQ(cloud.colors()->operator()(0, 1), 128);
    EXPECT_EQ(cloud.colors()->operator()(0, 2), 128);
}

TEST(PointCloudAttributesTest, SetColorsMove)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(10, 3);
    colors.operator()(0, 0) = 255;

    cloud.setColors(std::move(colors));

    ASSERT_TRUE(cloud.hasColors());
    EXPECT_EQ(cloud.colors()->operator()(0, 0), 255);
}

TEST(PointCloudAttributesTest, SetColorsRejectsWrongSize)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(5, 3);
    EXPECT_THROW(cloud.setColors(colors), std::runtime_error);
}

TEST(PointCloudAttributesTest, NoIntensitiesByDefault)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasIntensities());
    EXPECT_EQ(cloud.intensities(), nullptr);
}

TEST(PointCloudAttributesTest, SetIntensitiesCopy)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(10, 1);
    intensities.setConstant(1024);

    cloud.setIntensities(intensities);

    ASSERT_TRUE(cloud.hasIntensities());
    EXPECT_EQ(cloud.intensities()->operator()(0, 0), 1024);
}

TEST(PointCloudAttributesTest, SetIntensitiesMove)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(10, 1);
    intensities.operator()(0, 0) = 65535;

    cloud.setIntensities(std::move(intensities));

    ASSERT_TRUE(cloud.hasIntensities());
    EXPECT_EQ(cloud.intensities()->operator()(0, 0), 65535);
}

TEST(PointCloudAttributesTest, SetIntensitiesRejectsWrongSize)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> wrong_rows(5, 1);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> wrong_cols(10, 2);

    EXPECT_THROW(cloud.setIntensities(wrong_rows), std::runtime_error);
    EXPECT_THROW(cloud.setIntensities(wrong_cols), std::runtime_error);
}

TEST(PointCloudAttributesTest, CopySettersOwnIndependentAttributeStorage)
{
    using Matrix = plamatrix::MatrixXf;

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);

    Matrix normals(3, 3);
    normals.setConstant(0.0f);
    normals.operator()(2, 2) = 1.0f;
    cloud.setNormals(std::as_const(normals));

    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(3, 3);
    colors.setConstant(0);
    colors.operator()(2, 1) = 80;
    cloud.setColors(std::as_const(colors));

    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(3, 1);
    intensities.setConstant(0);
    intensities.operator()(2, 0) = 4096;
    cloud.setIntensities(std::as_const(intensities));

    Matrix scalar_fields(3, 2);
    scalar_fields.setConstant(0.0f);
    scalar_fields.operator()(1, 1) = 0.75f;
    cloud.setScalarFields({"error", "confidence"}, std::as_const(scalar_fields));

    Matrix texture_coords(3, 2);
    texture_coords.setConstant(0.0f);
    texture_coords.operator()(2, 1) = 0.6f;
    cloud.setTextureCoords(std::as_const(texture_coords));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::as_const(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.operator()(0, 0) = 2;
    face_texture_indices.operator()(0, 1) = 1;
    face_texture_indices.operator()(0, 2) = 0;
    cloud.setFaceTextureIndices(std::as_const(face_texture_indices));

    EXPECT_NE(cloud.normals()->data(), normals.data());
    EXPECT_NE(cloud.colors()->data(), colors.data());
    EXPECT_NE(cloud.intensities()->data(), intensities.data());
    EXPECT_NE(cloud.scalarFields()->data(), scalar_fields.data());
    EXPECT_NE(cloud.textureCoords()->data(), texture_coords.data());
    EXPECT_NE(cloud.faces()->data(), faces.data());
    EXPECT_NE(cloud.faceTextureIndices()->data(), face_texture_indices.data());

    normals.operator()(2, 2) = 9.0f;
    colors.operator()(2, 1) = 9;
    intensities.operator()(2, 0) = 9;
    scalar_fields.operator()(1, 1) = 9.0f;
    texture_coords.operator()(2, 1) = 0.9f;
    faces.operator()(0, 2) = 0;
    face_texture_indices.operator()(0, 0) = 0;

    EXPECT_FLOAT_EQ(cloud.normals()->operator()(2, 2), 1.0f);
    EXPECT_EQ(cloud.colors()->operator()(2, 1), 80);
    EXPECT_EQ(cloud.intensities()->operator()(2, 0), 4096);
    EXPECT_FLOAT_EQ(cloud.scalarFields()->operator()(1, 1), 0.75f);
    EXPECT_FLOAT_EQ(cloud.textureCoords()->operator()(2, 1), 0.6f);
    EXPECT_EQ(cloud.faces()->operator()(0, 2), 2);
    EXPECT_EQ(cloud.faceTextureIndices()->operator()(0, 0), 2);
}

TEST(PointCloudAttributesTest, CopySettersPreserveEmptyAttributeShapes)
{
    using Matrix = plamatrix::MatrixXf;

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud;
    Matrix normals(0, 3);
    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(0, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(0, 1);
    Matrix scalar_fields(0, 1);
    Matrix texture_coords(0, 2);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(0, 3);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(0, 3);

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
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::MatrixXf fields(3, 2);
    fields.operator()(0, 0) = 0.1f;
    fields.operator()(1, 0) = 0.2f;
    fields.operator()(2, 0) = 0.3f;
    fields.operator()(0, 1) = 10.0f;
    fields.operator()(1, 1) = 20.0f;
    fields.operator()(2, 1) = 30.0f;

    cloud.setScalarFields({"error", "confidence"}, std::move(fields));

    ASSERT_TRUE(cloud.hasScalarFields());
    ASSERT_NE(cloud.scalarFields(), nullptr);
    EXPECT_TRUE(cloud.hasScalarField("error"));
    EXPECT_EQ(cloud.scalarFieldIndex("confidence"), 1);
    EXPECT_EQ(cloud.scalarFieldNames().at(0), "error");
    EXPECT_FLOAT_EQ(cloud.scalarFields()->operator()(1, 0), 0.2f);
    EXPECT_FLOAT_EQ(cloud[2].scalar("confidence"), 30.0f);
}

TEST(PointCloudAttributesTest, SetNamedScalarFieldsRejectsBadShapeAndNames)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::MatrixXf wrongRows(2, 1);
    plamatrix::MatrixXf wrongCols(3, 2);
    plamatrix::MatrixXf ok(3, 1);

    EXPECT_THROW(cloud.setScalarFields({"error"}, wrongRows), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({"error"}, wrongCols), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({""}, ok), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({"error", "error"}, wrongCols), std::runtime_error);
    EXPECT_THROW(cloud.setScalarFields({"x"}, ok), std::runtime_error);
}

TEST(PointCloudAttributesTest, NoTextureCoordsByDefault)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasTextureCoords());
    EXPECT_EQ(cloud.textureCoords(), nullptr);
}

TEST(PointCloudAttributesTest, SetTextureCoords)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::MatrixXf tex(10, 2);
    tex.operator()(0, 0) = 0.5f;
    tex.operator()(0, 1) = 0.75f;

    cloud.setTextureCoords(std::move(tex));

    ASSERT_TRUE(cloud.hasTextureCoords());
    EXPECT_FLOAT_EQ(cloud.textureCoords()->operator()(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(cloud.textureCoords()->operator()(0, 1), 0.75f);
}

TEST(PointCloudAttributesTest, SetTextureCoordsRejectsWrongSize)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::MatrixXf tex(10, 3);
    EXPECT_THROW(cloud.setTextureCoords(tex), std::runtime_error);
}

TEST(PointCloudAttributesTest, NoFacesByDefault)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    EXPECT_FALSE(cloud.hasFaces());
    EXPECT_EQ(cloud.faces(), nullptr);
}

TEST(PointCloudAttributesTest, SetFaces)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(2, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    faces.operator()(1, 0) = 3;
    faces.operator()(1, 1) = 4;
    faces.operator()(1, 2) = 5;

    cloud.setFaces(std::move(faces));

    ASSERT_TRUE(cloud.hasFaces());
    EXPECT_EQ(cloud.faces()->rows(), 2);
    EXPECT_EQ(cloud.faces()->cols(), 3);
    EXPECT_EQ(cloud.faces()->operator()(0, 0), 0);
    EXPECT_EQ(cloud.faces()->operator()(1, 2), 5);
}

TEST(PointCloudAttributesTest, SetFacesWithTextureIndices)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(faces);

    plamatrix::MatrixXf tex(10, 2);
    tex.setConstant(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> texFaces(1, 3);
    texFaces.operator()(0, 0) = 5;
    texFaces.operator()(0, 1) = 6;
    texFaces.operator()(0, 2) = 7;
    cloud.setFaceTextureIndices(std::move(texFaces));

    ASSERT_TRUE(cloud.hasFaceTextureIndices());
    EXPECT_EQ(cloud.faceTextureIndices()->operator()(0, 0), 5);
}

TEST(PointCloudAttributesTest, SetFacesRejectsNonNx3)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(2, 2);
    EXPECT_THROW(cloud.setFaces(faces), std::runtime_error);
}

TEST(PointCloudAttributesTest, SetFacesRejectsOutOfRangeVertexIndex)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 3;

    EXPECT_THROW(cloud.setFaces(faces), std::out_of_range);
}

TEST(PointCloudAttributesTest, SetFacesRejectsNegativeVertexIndex)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = -1;
    faces.operator()(0, 2) = 2;

    EXPECT_THROW(cloud.setFaces(faces), std::out_of_range);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsMismatchedFaceCount)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(4);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(2, 3);
    face_texture_indices.setConstant(0);

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::runtime_error);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsNonNx3)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(4);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::MatrixXf tex(4, 2);
    tex.setConstant(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 2);
    face_texture_indices.setConstant(0);

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::runtime_error);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsMissingTextureCoords)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(4);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.setConstant(0);

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::runtime_error);
}

TEST(PointCloudAttributesTest, SetFaceTextureIndicesRejectsOutOfRangeTextureIndex)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(4);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::MatrixXf tex(4, 2);
    tex.setConstant(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.operator()(0, 0) = 0;
    face_texture_indices.operator()(0, 1) = 1;
    face_texture_indices.operator()(0, 2) = 4;

    EXPECT_THROW(cloud.setFaceTextureIndices(face_texture_indices), std::out_of_range);
}

TEST(PointCloudAttributesTest, ReplacingFacesRevalidatesFaceTextureIndices)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(4);
    plamatrix::MatrixXf tex(4, 2);
    tex.setConstant(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.operator()(0, 0) = 0;
    face_texture_indices.operator()(0, 1) = 1;
    face_texture_indices.operator()(0, 2) = 3;
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> too_many_faces(2, 3);
    too_many_faces.setConstant(0);
    EXPECT_THROW(cloud.setFaces(std::move(too_many_faces)), std::runtime_error);
}

TEST(PointCloudAttributesTest, ReplacingTextureCoordsRevalidatesFaceTextureIndices)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(4);
    plamatrix::MatrixXf tex(4, 2);
    tex.setConstant(0.0f);
    cloud.setTextureCoords(std::move(tex));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.operator()(0, 0) = 0;
    face_texture_indices.operator()(0, 1) = 1;
    face_texture_indices.operator()(0, 2) = 3;
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    plamatrix::MatrixXf too_few_texture_coords(3, 2);
    too_few_texture_coords.setConstant(0.0f);
    EXPECT_THROW(cloud.setTextureCoords(std::move(too_few_texture_coords)), std::out_of_range);
}

TEST(PointCloudAttributesTest, PublicValidationRejectsAttributeShapeChangedThroughMutableAlias)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(3, 3);
    colors.setConstant(0);
    cloud.setColors(std::move(colors));
    ASSERT_NO_THROW(cloud.validate());

    *cloud.colors() = plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>(1, 3);

    EXPECT_THROW(cloud.validate(), std::runtime_error);
}

TEST(PointCloudAttributesTest, MaterialLibraryFileEmptyByDefault)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    EXPECT_TRUE(cloud.materialLibraryFile().empty());
}

TEST(PointCloudAttributesTest, SetMaterialLibraryFile)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    cloud.setMaterialLibraryFile("materials.mtl");
    EXPECT_EQ(cloud.materialLibraryFile(), "materials.mtl");
}

TEST(PointCloudAttributesTest, TextureImageFileEmptyByDefault)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
    EXPECT_TRUE(cloud.textureImageFile().empty());
}

TEST(PointCloudAttributesTest, SetTextureImageFile)
{
    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(10);
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

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    for (int i = 0; i < 3; ++i)
    {
        cloud.points().operator()(i, 0) = static_cast<float>(i);
        cloud.points().operator()(i, 1) = static_cast<float>(i + 10);
        cloud.points().operator()(i, 2) = static_cast<float>(i + 20);
    }

    plamatrix::MatrixXf normals(3, 3);
    normals.setConstant(0.0f);
    normals.operator()(1, 2) = 1.0f;
    cloud.setNormals(std::move(normals));

    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(3, 3);
    colors.operator()(0, 0) = 10;
    colors.operator()(0, 1) = 20;
    colors.operator()(0, 2) = 30;
    colors.operator()(1, 0) = 40;
    colors.operator()(1, 1) = 50;
    colors.operator()(1, 2) = 60;
    colors.operator()(2, 0) = 70;
    colors.operator()(2, 1) = 80;
    colors.operator()(2, 2) = 90;
    cloud.setColors(std::move(colors));

    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(3, 1);
    intensities.operator()(0, 0) = 17;
    intensities.operator()(1, 0) = 1024;
    intensities.operator()(2, 0) = 65535;
    cloud.setIntensities(std::move(intensities));

    plamatrix::MatrixXf texture_coords(3, 2);
    texture_coords.operator()(0, 0) = 0.1f;
    texture_coords.operator()(0, 1) = 0.2f;
    texture_coords.operator()(1, 0) = 0.3f;
    texture_coords.operator()(1, 1) = 0.4f;
    texture_coords.operator()(2, 0) = 0.5f;
    texture_coords.operator()(2, 1) = 0.6f;
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.operator()(0, 0) = 2;
    face_texture_indices.operator()(0, 1) = 1;
    face_texture_indices.operator()(0, 2) = 0;
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    cloud.setMaterialLibraryFile("materials.mtl");
    cloud.setTextureImageFile("diffuse.png");

    plamatrix::MatrixXf scalar_fields(3, 1);
    scalar_fields.operator()(0, 0) = 0.1f;
    scalar_fields.operator()(1, 0) = 0.2f;
    scalar_fields.operator()(2, 0) = 0.3f;
    cloud.setScalarFields({"error"}, std::move(scalar_fields));

    auto roundtrip = cloud.toGpu().toCpu();

    ASSERT_TRUE(roundtrip.hasNormals());
    EXPECT_FLOAT_EQ(roundtrip.normals()->operator()(1, 2), 1.0f);

    ASSERT_TRUE(roundtrip.hasColors());
    EXPECT_EQ(roundtrip.colors()->operator()(2, 1), 80);

    ASSERT_TRUE(roundtrip.hasIntensities());
    EXPECT_EQ(roundtrip.intensities()->operator()(2, 0), 65535);

    ASSERT_TRUE(roundtrip.hasTextureCoords());
    EXPECT_FLOAT_EQ(roundtrip.textureCoords()->operator()(2, 1), 0.6f);

    ASSERT_TRUE(roundtrip.hasFaces());
    EXPECT_EQ(roundtrip.faces()->operator()(0, 2), 2);

    ASSERT_TRUE(roundtrip.hasFaceTextureIndices());
    EXPECT_EQ(roundtrip.faceTextureIndices()->operator()(0, 0), 2);

    EXPECT_EQ(roundtrip.materialLibraryFile(), "materials.mtl");
    EXPECT_EQ(roundtrip.textureImageFile(), "diffuse.png");

    ASSERT_TRUE(roundtrip.hasScalarFields());
    EXPECT_EQ(roundtrip.scalarFieldNames().at(0), "error");
    EXPECT_FLOAT_EQ(roundtrip.scalarFields()->operator()(2, 0), 0.3f);
}

TEST(PointCloudAttributesTest, GpuCopySettersOwnIndependentDeviceStorage)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud GPU copy setter test";
    }

    using CpuMatrix = plamatrix::MatrixXf;

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cpu_cloud(3);
    auto cloud = cpu_cloud.toGpu();

    CpuMatrix normals_cpu(3, 3);
    normals_cpu.setConstant(0.0f);
    normals_cpu.operator()(2, 2) = 1.0f;
    auto normals = plamatrix::internal::ResidentMatrix<float>::copyFrom(normals_cpu, cloud.executionContext());
    cloud.setNormals(std::as_const(normals));

    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors_cpu(3, 3);
    colors_cpu.setConstant(0);
    colors_cpu.operator()(2, 1) = 80;
    auto colors = plamatrix::internal::ResidentMatrix<uint8_t>::copyFrom(colors_cpu, cloud.executionContext());
    cloud.setColors(std::as_const(colors));

    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities_cpu(3, 1);
    intensities_cpu.setConstant(0);
    intensities_cpu.operator()(2, 0) = 4096;
    auto intensities =
        plamatrix::internal::ResidentMatrix<std::uint16_t>::copyFrom(intensities_cpu, cloud.executionContext());
    cloud.setIntensities(std::as_const(intensities));

    CpuMatrix scalar_fields_cpu(3, 1);
    scalar_fields_cpu.setConstant(0.0f);
    scalar_fields_cpu.operator()(1, 0) = 0.75f;
    auto scalar_fields = plamatrix::internal::ResidentMatrix<float>::copyFrom(
        scalar_fields_cpu, cloud.executionContext());
    cloud.setScalarFields({"confidence"}, std::as_const(scalar_fields));

    CpuMatrix texture_coords_cpu(3, 2);
    texture_coords_cpu.setConstant(0.0f);
    texture_coords_cpu.operator()(2, 1) = 0.6f;
    auto texture_coords =
        plamatrix::internal::ResidentMatrix<float>::copyFrom(texture_coords_cpu, cloud.executionContext());
    cloud.setTextureCoords(std::as_const(texture_coords));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces_cpu(1, 3);
    faces_cpu.operator()(0, 0) = 0;
    faces_cpu.operator()(0, 1) = 1;
    faces_cpu.operator()(0, 2) = 2;
    auto faces = plamatrix::internal::ResidentMatrix<int>::copyFrom(faces_cpu, cloud.executionContext());
    cloud.setFaces(std::as_const(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices_cpu(1, 3);
    face_texture_indices_cpu.operator()(0, 0) = 2;
    face_texture_indices_cpu.operator()(0, 1) = 1;
    face_texture_indices_cpu.operator()(0, 2) = 0;
    auto face_texture_indices = plamatrix::internal::ResidentMatrix<int>::copyFrom(
        face_texture_indices_cpu, cloud.executionContext());
    cloud.setFaceTextureIndices(std::as_const(face_texture_indices));

    EXPECT_NE(cloud.normals()->data(), normals.data());
    EXPECT_NE(cloud.colors()->data(), colors.data());
    EXPECT_NE(cloud.intensities()->data(), intensities.data());
    EXPECT_NE(cloud.scalarFields()->data(), scalar_fields.data());
    EXPECT_NE(cloud.textureCoords()->data(), texture_coords.data());
    EXPECT_NE(cloud.faces()->data(), faces.data());
    EXPECT_NE(cloud.faceTextureIndices()->data(), face_texture_indices.data());

    normals_cpu(2, 2) = 9.0f;
    colors_cpu(2, 1) = 9;
    intensities_cpu(2, 0) = 9;
    scalar_fields_cpu(1, 0) = 9.0f;
    texture_coords_cpu(2, 1) = 0.9f;
    faces_cpu(0, 2) = 0;
    face_texture_indices_cpu(0, 0) = 0;
    normals.copyFromHost(normals_cpu.data(), normals_cpu.size());
    colors.copyFromHost(colors_cpu.data(), colors_cpu.size());
    intensities.copyFromHost(intensities_cpu.data(), intensities_cpu.size());
    scalar_fields.copyFromHost(scalar_fields_cpu.data(), scalar_fields_cpu.size());
    texture_coords.copyFromHost(texture_coords_cpu.data(), texture_coords_cpu.size());
    faces.copyFromHost(faces_cpu.data(), faces_cpu.size());
    face_texture_indices.copyFromHost(
        face_texture_indices_cpu.data(), face_texture_indices_cpu.size());

    const auto stored = cloud.toCpu();
    EXPECT_FLOAT_EQ(stored.normals()->operator()(2, 2), 1.0f);
    EXPECT_EQ(stored.colors()->operator()(2, 1), 80);
    EXPECT_EQ(stored.intensities()->operator()(2, 0), 4096);
    EXPECT_FLOAT_EQ(stored.scalarFields()->operator()(1, 0), 0.75f);
    EXPECT_FLOAT_EQ(stored.textureCoords()->operator()(2, 1), 0.6f);
    EXPECT_EQ(stored.faces()->operator()(0, 2), 2);
    EXPECT_EQ(stored.faceTextureIndices()->operator()(0, 0), 2);
}

TEST(PointCloudAttributesTest, GpuCopySettersPreserveEmptyAttributeShapes)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping empty point-cloud GPU copy setter test";
    }

    using CpuMatrix = plamatrix::MatrixXf;

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cpu_cloud;
    auto cloud = cpu_cloud.toGpu();

    CpuMatrix normals_cpu(0, 3);
    auto normals = plamatrix::internal::ResidentMatrix<float>::copyFrom(normals_cpu, cloud.executionContext());
    plamatrix::Matrix<uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors_cpu(0, 3);
    auto colors = plamatrix::internal::ResidentMatrix<uint8_t>::copyFrom(colors_cpu, cloud.executionContext());
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities_cpu(0, 1);
    auto intensities =
        plamatrix::internal::ResidentMatrix<std::uint16_t>::copyFrom(intensities_cpu, cloud.executionContext());
    CpuMatrix scalar_fields_cpu(0, 1);
    auto scalar_fields = plamatrix::internal::ResidentMatrix<float>::copyFrom(
        scalar_fields_cpu, cloud.executionContext());
    CpuMatrix texture_coords_cpu(0, 2);
    auto texture_coords =
        plamatrix::internal::ResidentMatrix<float>::copyFrom(texture_coords_cpu, cloud.executionContext());
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces_cpu(0, 3);
    auto faces = plamatrix::internal::ResidentMatrix<int>::copyFrom(faces_cpu, cloud.executionContext());
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices_cpu(0, 3);
    auto face_texture_indices =
        plamatrix::internal::ResidentMatrix<int>::copyFrom(face_texture_indices_cpu, cloud.executionContext());

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

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::MatrixXf texture_coords(3, 2);
    texture_coords.setConstant(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.operator()(0, 0) = 0;
    face_texture_indices.operator()(0, 1) = 1;
    face_texture_indices.operator()(0, 2) = 2;
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    cloud.faceTextureIndices()->operator()(0, 2) = 3;

    EXPECT_THROW((void)cloud.toGpu(), std::out_of_range);
}

TEST(PointCloudAttributesTest, ToGpuRejectsMutableInvalidPointShapeBeforeTransfer)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    cloud.points() = plamatrix::MatrixXf(3, 4);

    try
    {
        (void)cloud.toGpu();
        FAIL() << "Expected invalid point shape to be rejected";
    }
    catch (const std::runtime_error& ex)
    {
        EXPECT_NE(std::string(ex.what()).find("DeviceCloud points must be Nx3"), std::string::npos);
    }
}

TEST(PointCloudAttributesTest, ToGpuRejectsMutableInvalidFaceTextureShape)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::MatrixXf texture_coords(3, 2);
    texture_coords.setConstant(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.setConstant(0);
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.setConstant(0);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> invalid_face_texture_indices(1, 2);
    invalid_face_texture_indices.setConstant(0);
    *cloud.faceTextureIndices() = std::move(invalid_face_texture_indices);

    EXPECT_THROW((void)cloud.toGpu(), std::runtime_error);
}

TEST(PointCloudAttributesTest, ToCpuRejectsMutableInvalidGpuFaceTextureIndices)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::MatrixXf texture_coords(3, 2);
    texture_coords.setConstant(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.operator()(0, 0) = 0;
    faces.operator()(0, 1) = 1;
    faces.operator()(0, 2) = 2;
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.operator()(0, 0) = 0;
    face_texture_indices.operator()(0, 1) = 1;
    face_texture_indices.operator()(0, 2) = 2;
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    auto gpu_cloud = cloud.toGpu();
    auto invalid_indices = gpu_cloud.faceTextureIndices()->toHostMatrix();
    invalid_indices(0, 1) = 3;
    *gpu_cloud.faceTextureIndices() = plamatrix::internal::ResidentMatrix<int>::copyFrom(
        invalid_indices, gpu_cloud.executionContext());

    EXPECT_THROW((void)gpu_cloud.toCpu(), std::out_of_range);
}

TEST(PointCloudAttributesTest, ToCpuRejectsMutableInvalidGpuFaceTextureShape)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    plamatrix::MatrixXf texture_coords(3, 2);
    texture_coords.setConstant(0.0f);
    cloud.setTextureCoords(std::move(texture_coords));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> faces(1, 3);
    faces.setConstant(0);
    cloud.setFaces(std::move(faces));

    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> face_texture_indices(1, 3);
    face_texture_indices.setConstant(0);
    cloud.setFaceTextureIndices(std::move(face_texture_indices));

    auto gpu_cloud = cloud.toGpu();
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> invalid_host_indices(1, 2);
    invalid_host_indices.setZero();
    auto invalid_face_texture_indices =
        plamatrix::internal::ResidentMatrix<int>::copyFrom(invalid_host_indices, gpu_cloud.executionContext());
    *gpu_cloud.faceTextureIndices() = std::move(invalid_face_texture_indices);

    EXPECT_THROW((void)gpu_cloud.toCpu(), std::runtime_error);
}

TEST(PointCloudAttributesTest, ToCpuRejectsMutableInvalidGpuPointShapeBeforeTransfer)
{
    if (!hasCudaDeviceForAttributes())
    {
        GTEST_SKIP() << "No CUDA device, skipping point-cloud validation transfer test";
    }

    plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> cloud(3);
    auto gpu_cloud = cloud.toGpu();
    gpu_cloud.points() = plamatrix::internal::ResidentMatrix<float>(
        3, 4, gpu_cloud.executionContext());

    try
    {
        (void)gpu_cloud.toCpu();
        FAIL() << "Expected invalid point shape to be rejected";
    }
    catch (const std::runtime_error& ex)
    {
        EXPECT_NE(std::string(ex.what()).find("DeviceCloud points must be Nx3"), std::string::npos);
    }
}

TEST(PointCloudAttributesTest, GpuCloudRejectsCpuResidentContext)
{
    auto context = plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cpu, 0});
    plamatrix::MatrixXf host_points(1, 3);
    host_points.setZero();
    auto resident_points = plamatrix::internal::ResidentMatrix<float>::copyFrom(host_points, context);

    EXPECT_THROW((plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>(
        std::move(resident_points), context)), std::invalid_argument);
}
#endif
