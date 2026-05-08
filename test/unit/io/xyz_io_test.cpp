#include <gtest/gtest.h>
#include <plapoint/io/xyz_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdio>
#include <fstream>
#include <string>

TEST(XyzIOTest, Roundtrip)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;

    Matrix pts(4, 3);
    for (int i = 0; i < 4; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i*2));
        pts.setValue(i, 2, Scalar(i*3));
    }
    Cloud cloud(std::move(pts));

    std::string path = "/tmp/plapoint_test.xyz";
    plapoint::io::writeXyz(path, cloud);

    auto loaded = plapoint::io::readXyz<Scalar>(path);
    EXPECT_EQ(loaded->size(), 4u);
    EXPECT_FLOAT_EQ(loaded->points().getValue(2, 1), 4.0f);

    std::remove(path.c_str());
}
