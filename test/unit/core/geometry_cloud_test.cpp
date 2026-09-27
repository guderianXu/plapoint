#include <gtest/gtest.h>

#include <plapoint/filters/detail/geometry_bridge.h>
#include <plapoint/geometry_cloud.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

TEST(GeometryCloudTest, OwnsAttributesTopologyAndMaterialsAcrossMoveAndBackendRoundTrip)
{
    using Cloud = plapoint::GeometryCloud<float>;
    using FloatMatrix = Cloud::MatrixType;

    FloatMatrix points(3, 3);
    points << 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;
    Cloud cloud(std::move(points));

    FloatMatrix normals(3, 3);
    normals.setZero();
    normals(0, 2) = 1.0f;
    cloud.setNormals(normals);
    normals(0, 2) = 7.0f;

    Cloud::AttributeMatrix<std::uint8_t> colors(3, 3);
    colors.setConstant(20);
    cloud.setColors(std::move(colors));
    Cloud::AttributeMatrix<std::uint16_t> intensities(3, 1);
    intensities << 100, 200, 300;
    cloud.setIntensities(std::move(intensities));
    FloatMatrix fields(3, 1);
    fields << 0.25f, 0.5f, 0.75f;
    cloud.setScalarFields({"confidence"}, std::move(fields));

    FloatMatrix uv(4, 2);
    uv << 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f;
    cloud.setTextureCoords(uv);
    uv(0, 0) = 9.0f;

    Cloud::AttributeMatrix<int> faces(1, 3);
    faces << 0, 1, 2;
    cloud.setFaces(faces);
    faces(0, 0) = 2;
    Cloud::AttributeMatrix<int> face_uv(1, 3);
    face_uv << 2, 1, 3;
    cloud.setFaceTextureIndices(std::move(face_uv));
    cloud.setMaterialLibraryFile("surface.mtl");
    cloud.setTextureImageFile("surface.png");

    Cloud moved;
    moved = std::move(cloud);
    moved.validate();
    EXPECT_FLOAT_EQ((*moved.normals())(0, 2), 1.0f);
    EXPECT_FLOAT_EQ((*moved.textureCoords())(0, 0), 0.0f);
    EXPECT_EQ((*moved.faces())(0, 0), 0);
    EXPECT_FALSE(moved.hasPointAlignedTextureCoords());

    const auto restored = plapoint::detail::fromDeviceCloud(plapoint::detail::toDeviceCloud(moved));
    restored.validate();
    ASSERT_EQ(restored.size(), 3u);
    ASSERT_TRUE(restored.hasNormals());
    ASSERT_TRUE(restored.hasColors());
    ASSERT_TRUE(restored.hasIntensities());
    ASSERT_TRUE(restored.hasScalarFields());
    ASSERT_TRUE(restored.hasTextureCoords());
    ASSERT_TRUE(restored.hasFaces());
    ASSERT_TRUE(restored.hasFaceTextureIndices());
    EXPECT_EQ(restored.scalarFieldNames(), (std::vector<std::string>{"confidence"}));
    EXPECT_FLOAT_EQ((*restored.scalarFields())(2, 0), 0.75f);
    EXPECT_EQ((*restored.colors())(1, 0), 20);
    EXPECT_EQ((*restored.intensities())(2, 0), 300);
    EXPECT_EQ((*restored.faceTextureIndices())(0, 0), 2);
    EXPECT_EQ((*restored.faceTextureIndices())(0, 2), 3);
    EXPECT_EQ(restored.materialLibraryFile(), "surface.mtl");
    EXPECT_EQ(restored.textureImageFile(), "surface.png");
}
