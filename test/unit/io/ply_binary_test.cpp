#include <gtest/gtest.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdio>
#include <string>

TEST(PlyIOTest, BinaryLERoundtrip)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i*2));
        pts.setValue(i, 2, Scalar(i*3));
    }
    Cloud cloud(std::move(pts));

    std::string path = "/tmp/plapoint_test_ble.ply";
    plapoint::io::writePly(path, cloud, plapoint::io::PlyFormat::BinaryLE);

    auto loaded = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(loaded->size(), 5u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(loaded->points().getValue(4, 2), 12.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, BinaryBERoundtrip)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        pts.setValue(i, 0, Scalar(i+1));
        pts.setValue(i, 1, Scalar(i+2));
        pts.setValue(i, 2, Scalar(i+3));
    }
    Cloud cloud(std::move(pts));

    std::string path = "/tmp/plapoint_test_bbe.ply";
    plapoint::io::writePly(path, cloud, plapoint::io::PlyFormat::BinaryBE);

    auto loaded = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(loaded->size(), 3u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(0, 0), 1.0f);

    std::remove(path.c_str());
}
