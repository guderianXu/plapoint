#include <gtest/gtest.h>
#include "temp_file.h"
#include <plapoint/io/obj_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdio>

TEST(ObjIoTest, WriteAndReadBackVertexOnly)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(4, 3);
    pts.setValue(0, 0, 0.0f); pts.setValue(0, 1, 0.0f); pts.setValue(0, 2, 0.0f);
    pts.setValue(1, 0, 1.0f); pts.setValue(1, 1, 0.0f); pts.setValue(1, 2, 0.0f);
    pts.setValue(2, 0, 0.5f); pts.setValue(2, 1, 1.0f); pts.setValue(2, 2, 0.0f);
    pts.setValue(3, 0, 0.5f); pts.setValue(3, 1, 0.5f); pts.setValue(3, 2, 1.0f);

    Cloud cloud(std::move(pts));

    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    plapoint::io::writeObj<float>(path, cloud);

    auto read_back = plapoint::io::readObj<float>(path);
    ASSERT_NE(read_back, nullptr);
    EXPECT_EQ(read_back->size(), 4);
    EXPECT_FLOAT_EQ(read_back->points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(read_back->points().getValue(3, 2), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, WriteAndReadBackWithFaces)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    Cloud cloud(6);
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces.setValue(0, 0, 0); faces.setValue(0, 1, 1); faces.setValue(0, 2, 2);
    faces.setValue(1, 0, 3); faces.setValue(1, 1, 4); faces.setValue(1, 2, 5);
    cloud.setFaces(std::move(faces));

    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    plapoint::io::writeObj<float>(path, cloud);

    auto read_back = plapoint::io::readObj<float>(path);
    ASSERT_NE(read_back, nullptr);
    EXPECT_EQ(read_back->size(), 6);
    ASSERT_TRUE(read_back->hasFaces());
    EXPECT_EQ(read_back->faces()->rows(), 2);
    EXPECT_EQ(read_back->faces()->getValue(0, 0), 0);

    std::remove(path.c_str());
}

TEST(ObjIoTest, WriteAndReadBackWithNormals)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    Cloud cloud(3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(3, 3);
    normals.setValue(0, 0, 0.0f); normals.setValue(0, 1, 0.0f); normals.setValue(0, 2, 1.0f);
    normals.setValue(1, 0, 1.0f); normals.setValue(1, 1, 0.0f); normals.setValue(1, 2, 0.0f);
    normals.setValue(2, 0, 0.0f); normals.setValue(2, 1, 1.0f); normals.setValue(2, 2, 0.0f);
    cloud.setNormals(std::move(normals));

    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    plapoint::io::writeObj<float>(path, cloud);

    auto read_back = plapoint::io::readObj<float>(path);
    ASSERT_NE(read_back, nullptr);
    EXPECT_EQ(read_back->size(), 3);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 2), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadsFaceNormalIndicesIntoPointNormals)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vn 1 0 0\n";
        f << "vn 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "f 1//3 2//2 3//1\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(2, 0), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadsFaceNormalIndicesWithUnreferencedVertex)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "v 0 0 1\n";
        f << "vn 1 0 0\n";
        f << "vn 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "vn -1 0 0\n";
        f << "f 1//3 2//2 3//1\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(2, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(3, 0), -1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadsRelativeFaceNormalIndices)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vn 1 0 0\n";
        f << "vn 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "f 1//-1 2//-2 3//-3\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(2, 0), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, FallsBackToOrderedNormalsWhenFaceNormalIndicesConflict)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vn 1 0 0\n";
        f << "vn 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "f 1//1 2//2 3//3\n";
        f << "f 1//2 2//2 3//3\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(2, 2), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, DoesNotFallbackToOrderedNormalsWhenConflictLeavesReferencedVertexMissingNormal)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vn 1 0 0\n";
        f << "vn 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "f 1//1 2 3//3\n";
        f << "f 1//2 2 3//3\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    EXPECT_FALSE(read_back->hasNormals());

    std::remove(path.c_str());
}

TEST(ObjIoTest, RejectsZeroObjIndices)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vt 0 0\n";
        f << "vn 0 0 1\n";
        f << "f 1/0/0 2/1/1 3/1/1\n";
    }

    EXPECT_THROW((void)plapoint::io::readObj<float>(path), std::out_of_range);

    std::remove(path.c_str());
}

TEST(ObjIoTest, RejectsFacesWithFewerThanThreeVertices)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "f 1 2\n";
    }

    EXPECT_THROW((void)plapoint::io::readObj<float>(path), std::invalid_argument);

    std::remove(path.c_str());
}

TEST(ObjIoTest, IgnoresPartialFaceTextureIndicesWithoutShiftingCorners)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vt 0 0\n";
        f << "vt 1 0\n";
        f << "vt 0 1\n";
        f << "f 1 2/2 3/3\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasTextureCoords());
    EXPECT_FALSE(read_back->hasFaceTextureIndices());

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadsFaceNormalIndicesWhenSomeFacesOmitNormals)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vn 1 0 0\n";
        f << "vn 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "f 1//3 2//2 3//1\n";
        f << "f 1 2 3\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(2, 0), 1.0f);

    std::remove(path.c_str());
}

TEST(ObjIoTest, DoesNotInventNormalsForCornersThatOmitNormalIndices)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "vn 9 9 9\n";
        f << "vn 1 0 0\n";
        f << "f 1//1 2 3//3\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    EXPECT_FALSE(read_back->hasNormals());

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadsIndependentTextureCoordinateTable)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vt 0 0\n";
        f << "vt 1 0\n";
        f << "vt 0 1\n";
        f << "vt 1 1\n";
        f << "f 1/1 2/2 3/4\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);
    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasTextureCoords());
    EXPECT_EQ(read_back->textureCoords()->rows(), 4);
    ASSERT_TRUE(read_back->hasFaceTextureIndices());
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 2), 3);

    std::remove(path.c_str());
}

TEST(ObjIoTest, PreservesFirstTriangleTextureIndicesForQuad)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 1 1 0\n";
        f << "v 0 1 0\n";
        f << "vt 0 0\n";
        f << "vt 1 0\n";
        f << "vt 1 1\n";
        f << "vt 0 1\n";
        f << "f 1/1 2/2 3/3 4/4\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasFaces());
    EXPECT_EQ(read_back->faces()->getValue(0, 0), 0);
    EXPECT_EQ(read_back->faces()->getValue(0, 1), 1);
    EXPECT_EQ(read_back->faces()->getValue(0, 2), 2);
    ASSERT_TRUE(read_back->hasTextureCoords());
    EXPECT_EQ(read_back->textureCoords()->rows(), 4);
    ASSERT_TRUE(read_back->hasFaceTextureIndices());
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 0), 0);
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 1), 1);
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 2), 2);

    std::remove(path.c_str());
}

TEST(ObjIoTest, WritesAllTextureCoordinates)
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    Cloud cloud(3);

    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> texture_coords(4, 2);
    texture_coords.setValue(0, 0, 0.0f); texture_coords.setValue(0, 1, 0.0f);
    texture_coords.setValue(1, 0, 1.0f); texture_coords.setValue(1, 1, 0.0f);
    texture_coords.setValue(2, 0, 0.0f); texture_coords.setValue(2, 1, 1.0f);
    texture_coords.setValue(3, 0, 1.0f); texture_coords.setValue(3, 1, 1.0f);
    cloud.setTextureCoords(std::move(texture_coords));

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

    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    plapoint::io::writeObj<float>(path, cloud);
    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_TRUE(read_back->hasTextureCoords());
    EXPECT_EQ(read_back->textureCoords()->rows(), 4);
    ASSERT_TRUE(read_back->hasFaceTextureIndices());
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 2), 3);

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadsCompleteFaceTextureAndNormalIndices)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vt 0 0\n";
        f << "vt 1 0\n";
        f << "vt 0 1\n";
        f << "vt 1 1\n";
        f << "vn 1 0 0\n";
        f << "vn 0 1 0\n";
        f << "vn 0 0 1\n";
        f << "f 1/4/3 2/2/2 3/1/1\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_TRUE(read_back->hasFaces());
    EXPECT_EQ(read_back->faces()->getValue(0, 0), 0);
    EXPECT_EQ(read_back->faces()->getValue(0, 1), 1);
    EXPECT_EQ(read_back->faces()->getValue(0, 2), 2);
    ASSERT_TRUE(read_back->hasTextureCoords());
    EXPECT_EQ(read_back->textureCoords()->rows(), 4);
    ASSERT_TRUE(read_back->hasFaceTextureIndices());
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 0), 3);
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 1), 1);
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 2), 0);
    ASSERT_TRUE(read_back->hasNormals());
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(read_back->normals()->getValue(2, 0), 1.0f);

    std::filesystem::remove(path);
}

TEST(ObjIoTest, ReadsVertexLinesWithExtraAttributes)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "o colored_vertices\n";
        f << "v 1 2 3 1.0 255 0 0\n";
        f << "v 4 5 6 1.0 0 255 0\n";
        f << "usemtl ignored_material\n";
    }

    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_NE(read_back, nullptr);
    ASSERT_EQ(read_back->size(), 2u);
    EXPECT_FLOAT_EQ(read_back->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(read_back->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(read_back->points().getValue(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(read_back->points().getValue(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(read_back->points().getValue(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(read_back->points().getValue(1, 2), 6.0f);

    std::filesystem::remove(path);
}

TEST(ObjIoTest, RejectsOutOfRangeFaceVertexIndex)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "f 1 2 4\n";
    }

    EXPECT_THROW((void)plapoint::io::readObj<float>(path), std::out_of_range);

    std::filesystem::remove(path);
}

TEST(ObjIoTest, RejectsOutOfRangeRelativeFaceVertexIndex)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "f -4 -2 -1\n";
    }

    EXPECT_THROW((void)plapoint::io::readObj<float>(path), std::out_of_range);

    std::filesystem::remove(path);
}

TEST(ObjIoTest, RejectsOutOfRangeFaceTextureIndex)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vt 0 0\n";
        f << "f 1/1 2/2 3/1\n";
    }

    EXPECT_THROW((void)plapoint::io::readObj<float>(path), std::out_of_range);

    std::filesystem::remove(path);
}

TEST(ObjIoTest, RejectsMalformedFaceIndexToken)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "v 0 0 0\n";
        f << "v 1 0 0\n";
        f << "v 0 1 0\n";
        f << "vt 0 0\n";
        f << "f 1/abc 2/1 3/1\n";
    }

    EXPECT_THROW((void)plapoint::io::readObj<float>(path), std::invalid_argument);

    std::filesystem::remove(path);
}

TEST(ObjIoTest, ReadNonExistentFileThrows)
{
    const plapoint::test::TempFile temp_file(".obj");
    const auto path = temp_file.string();
    std::filesystem::remove(path);

    EXPECT_THROW(plapoint::io::readObj<float>(path), std::runtime_error);
}
