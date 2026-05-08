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

TEST(ObjIoTest, ReadNonExistentFileThrows)
{
    EXPECT_THROW(plapoint::io::readObj<float>("/tmp/nonexistent_plapoint.obj"), std::runtime_error);
}
