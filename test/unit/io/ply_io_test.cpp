#include <gtest/gtest.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdio>
#include <fstream>
#include <string>

TEST(PlyIOTest, RoundtripASCII)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(5, 3);
    Matrix nrm(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i*2));
        pts.setValue(i, 2, Scalar(i*3));
        nrm.setValue(i, 0, 0); nrm.setValue(i, 1, 0); nrm.setValue(i, 2, 1);
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));
    cloud->setNormals(std::move(nrm));

    std::string path = "/tmp/plapoint_test_ascii.ply";
    EXPECT_NO_THROW(plapoint::io::writePly(path, *cloud));

    auto loaded = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(loaded->size(), 5u);
    EXPECT_TRUE(loaded->hasNormals());
    EXPECT_FLOAT_EQ(loaded->normals()->getValue(0, 2), 1.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadOnlyPositions)
{
    using Scalar = float;

    std::string path = "/tmp/plapoint_test_minimal.ply";
    {
        std::ofstream f(path);
        f << "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        f << "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(cloud->size(), 3u);
    EXPECT_FALSE(cloud->hasNormals());
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(2, 2), 9.0f);

    std::remove(path.c_str());
}
