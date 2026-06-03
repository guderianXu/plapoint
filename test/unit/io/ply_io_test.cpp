#include <gtest/gtest.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace {

void writeBigEndianFloat(std::ofstream& f, float value)
{
    static_assert(sizeof(float) == sizeof(std::uint32_t), "float must be 32-bit");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((bits >> 24) & 0xff),
        static_cast<unsigned char>((bits >> 16) & 0xff),
        static_cast<unsigned char>((bits >> 8) & 0xff),
        static_cast<unsigned char>(bits & 0xff),
    };
    f.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

} // namespace

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

TEST(PlyIOTest, ReadsPointOffsetComment)
{
    using Scalar = float;

    std::string path = "/tmp/plapoint_test_point_offset.ply";
    {
        std::ofstream f(path);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "comment POINT_OFFSET 1000000.0 -2000000.0 3000000.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        f << "1.25 2.5 3.75\n-4.0 -5.0 -6.0\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1000001.25f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), -1999997.5f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 2999994.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBinaryLittleEndianPointOffsetComment)
{
    using Scalar = float;

    std::string path = "/tmp/plapoint_test_binary_point_offset.ply";
    {
        std::ofstream f(path, std::ios::binary);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "comment POINT_OFFSET 1000000.0 -2000000.0 3000000.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        const float p0[3] = {1.25f, 2.5f, 3.75f};
        const float p1[3] = {-4.0f, -5.0f, -6.0f};
        f.write(reinterpret_cast<const char*>(p0), sizeof(p0));
        f.write(reinterpret_cast<const char*>(p1), sizeof(p1));
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1000001.25f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), -1999997.5f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 2999994.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBinaryBigEndianPointOffsetComment)
{
    using Scalar = float;

    std::string path = "/tmp/plapoint_test_binary_be_point_offset.ply";
    {
        std::ofstream f(path, std::ios::binary);
        f << "ply\n"
          << "format binary_big_endian 1.0\n"
          << "comment POINT_OFFSET 1000000.0 -2000000.0 3000000.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        const float values[6] = {1.25f, 2.5f, 3.75f, -4.0f, -5.0f, -6.0f};
        for (float value : values)
        {
            writeBigEndianFloat(f, value);
        }
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    EXPECT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1000001.25f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), -1999997.5f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 2999994.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsPointOffsetAsLocalCoordinatesWhenRequested)
{
    using Scalar = float;

    std::string path = "/tmp/plapoint_test_local_point_offset.ply";
    {
        std::ofstream f(path);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "comment POINT_OFFSET 100000000.0 -200000000.0 300000000.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        f << "0.03125 0.0625 0.09375\n"
          << "0.125 0.15625 0.1875\n";
    }

    std::array<double, 3> offset = {0.0, 0.0, 0.0};
    bool hasOffset = false;
    auto cloud = plapoint::io::readPlyLocal<Scalar>(path, &offset, &hasOffset);
    EXPECT_EQ(cloud->size(), 2u);
    EXPECT_TRUE(hasOffset);
    EXPECT_DOUBLE_EQ(offset[0], 100000000.0);
    EXPECT_DOUBLE_EQ(offset[1], -200000000.0);
    EXPECT_DOUBLE_EQ(offset[2], 300000000.0);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 0.03125f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 0.0625f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 0.09375f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 0), 0.125f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 1), 0.15625f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 0.1875f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBinaryPointOffsetAsLocalCoordinatesWhenRequested)
{
    using Scalar = float;

    const auto writeAndRead = [](const std::string& path, bool bigEndian) {
        {
            std::ofstream f(path, std::ios::binary);
            f << "ply\n"
              << (bigEndian ? "format binary_big_endian 1.0\n"
                            : "format binary_little_endian 1.0\n")
              << "comment POINT_OFFSET 100000000.0 -200000000.0 300000000.0\n"
              << "element vertex 2\n"
              << "property float x\n"
              << "property float y\n"
              << "property float z\n"
              << "end_header\n";
            const float values[6] = {0.03125f, 0.0625f, 0.09375f, 0.125f, 0.15625f, 0.1875f};
            for (float value : values)
            {
                if (bigEndian)
                {
                    writeBigEndianFloat(f, value);
                }
                else
                {
                    f.write(reinterpret_cast<const char*>(&value), sizeof(float));
                }
            }
        }

        std::array<double, 3> offset = {0.0, 0.0, 0.0};
        bool hasOffset = false;
        auto cloud = plapoint::io::readPlyLocal<Scalar>(path, &offset, &hasOffset);
        EXPECT_EQ(cloud->size(), 2u);
        EXPECT_TRUE(hasOffset);
        EXPECT_DOUBLE_EQ(offset[0], 100000000.0);
        EXPECT_DOUBLE_EQ(offset[1], -200000000.0);
        EXPECT_DOUBLE_EQ(offset[2], 300000000.0);
        EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 0.03125f);
        EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 0.0625f);
        EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 0.09375f);
        EXPECT_FLOAT_EQ(cloud->points().getValue(1, 0), 0.125f);
        EXPECT_FLOAT_EQ(cloud->points().getValue(1, 1), 0.15625f);
        EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 0.1875f);
        std::remove(path.c_str());
    };

    writeAndRead("/tmp/plapoint_test_local_binary_le_point_offset.ply", false);
    writeAndRead("/tmp/plapoint_test_local_binary_be_point_offset.ply", true);
}
