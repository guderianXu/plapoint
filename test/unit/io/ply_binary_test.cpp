#include <gtest/gtest.h>
#include "temp_file.h"
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

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    plapoint::io::writePly(path, cloud, plapoint::io::PlyFormat::BinaryLE);

    auto loaded = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(loaded->size(), 5u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(loaded->points().getValue(4, 2), 12.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, BinaryLERoundtripPreservesDoubleLargeCoordinateDeltas)
{
    using Scalar = double;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(2, 3);
    pts.setValue(0, 0, 100000000.01);
    pts.setValue(0, 1, -100000000.02);
    pts.setValue(0, 2, 123456789.125);
    pts.setValue(1, 0, 100000000.02);
    pts.setValue(1, 1, -100000000.03);
    pts.setValue(1, 2, 123456789.25);
    Cloud cloud(std::move(pts));

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    plapoint::io::writePly(path, cloud, plapoint::io::PlyFormat::BinaryLE);

    auto loaded = plapoint::io::readPly<Scalar>(path);

    ASSERT_EQ(loaded->size(), 2u);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(0, 0), 100000000.01);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(0, 1), -100000000.02);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(0, 2), 123456789.125);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(1, 0), 100000000.02);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(1, 1), -100000000.03);
    EXPECT_DOUBLE_EQ(loaded->points().getValue(1, 2), 123456789.25);

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

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    plapoint::io::writePly(path, cloud, plapoint::io::PlyFormat::BinaryBE);

    auto loaded = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(loaded->size(), 3u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(0, 0), 1.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, BinaryLERoundtripPreservesFaces)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    pts.setValue(0, 0, 0); pts.setValue(0, 1, 0); pts.setValue(0, 2, 0);
    pts.setValue(1, 0, 1); pts.setValue(1, 1, 0); pts.setValue(1, 2, 0);
    pts.setValue(2, 0, 0); pts.setValue(2, 1, 1); pts.setValue(2, 2, 0);
    pts.setValue(3, 0, 0); pts.setValue(3, 1, 0); pts.setValue(3, 2, 1);
    Cloud cloud(std::move(pts));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(2, 3);
    faces.setValue(0, 0, 0);
    faces.setValue(0, 1, 1);
    faces.setValue(0, 2, 2);
    faces.setValue(1, 0, 0);
    faces.setValue(1, 1, 2);
    faces.setValue(1, 2, 3);
    cloud.setFaces(std::move(faces));

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    plapoint::io::writePly(path, cloud, plapoint::io::PlyFormat::BinaryLE);

    auto loaded = plapoint::io::readPly<Scalar>(path);

    ASSERT_TRUE(loaded->hasFaces());
    ASSERT_EQ(loaded->faces()->rows(), 2);
    EXPECT_EQ(loaded->faces()->getValue(0, 0), 0);
    EXPECT_EQ(loaded->faces()->getValue(0, 1), 1);
    EXPECT_EQ(loaded->faces()->getValue(0, 2), 2);
    EXPECT_EQ(loaded->faces()->getValue(1, 0), 0);
    EXPECT_EQ(loaded->faces()->getValue(1, 1), 2);
    EXPECT_EQ(loaded->faces()->getValue(1, 2), 3);

    std::remove(path.c_str());
}
