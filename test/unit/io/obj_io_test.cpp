#include <gtest/gtest.h>
#include <plapoint/io/obj_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <fstream>
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

    std::string path = "/tmp/plapoint_test_vertex.obj";
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

    std::string path = "/tmp/plapoint_test_faces.obj";
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

    std::string path = "/tmp/plapoint_test_normals.obj";
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
    const std::string path = "/tmp/plapoint_test_face_normal_indices.obj";
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
    const std::string path = "/tmp/plapoint_test_face_normal_unreferenced_vertex.obj";
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
    const std::string path = "/tmp/plapoint_test_relative_face_normal_indices.obj";
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
    const std::string path = "/tmp/plapoint_test_conflicting_face_normals.obj";
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
    const std::string path = "/tmp/plapoint_test_conflict_and_missing_face_normal.obj";
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
    const std::string path = "/tmp/plapoint_test_zero_indices.obj";
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
    const std::string path = "/tmp/plapoint_test_short_face.obj";
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
    const std::string path = "/tmp/plapoint_test_partial_face_texture_indices.obj";
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
    const std::string path = "/tmp/plapoint_test_mixed_face_normal_indices.obj";
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
    const std::string path = "/tmp/plapoint_test_missing_corner_normal.obj";
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
    const std::string path = "/tmp/plapoint_test_independent_uv.obj";
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
    const std::string path = "/tmp/plapoint_test_quad_face_texture_indices.obj";
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

    const std::string path = "/tmp/plapoint_test_write_independent_uv.obj";
    plapoint::io::writeObj<float>(path, cloud);
    auto read_back = plapoint::io::readObj<float>(path);

    ASSERT_TRUE(read_back->hasTextureCoords());
    EXPECT_EQ(read_back->textureCoords()->rows(), 4);
    ASSERT_TRUE(read_back->hasFaceTextureIndices());
    EXPECT_EQ(read_back->faceTextureIndices()->getValue(0, 2), 3);

    std::remove(path.c_str());
}

TEST(ObjIoTest, ReadNonExistentFileThrows)
{
    EXPECT_THROW(plapoint::io::readObj<float>("/tmp/nonexistent_plapoint.obj"), std::runtime_error);
}
