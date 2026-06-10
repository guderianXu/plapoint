#include <gtest/gtest.h>
#include "temp_file.h"
#include <plapoint/io/ply_io.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/plamatrix.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

template <typename Exception, typename Fn>
void expectThrowsMessageContaining(Fn&& fn, const std::string& expected_message)
{
    try
    {
        fn();
        FAIL() << "Expected exception containing: " << expected_message;
    }
    catch (const Exception& e)
    {
        EXPECT_NE(std::string(e.what()).find(expected_message), std::string::npos)
            << "Actual exception: " << e.what();
    }
}

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

void writeBigEndianUint16(std::ofstream& f, std::uint16_t value)
{
    const unsigned char bytes[2] = {
        static_cast<unsigned char>((value >> 8) & 0xff),
        static_cast<unsigned char>(value & 0xff),
    };
    f.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void writeBigEndianInt32(std::ofstream& f, std::int32_t value)
{
    const auto bits = static_cast<std::uint32_t>(value);
    const unsigned char bytes[4] = {
        static_cast<unsigned char>((bits >> 24) & 0xff),
        static_cast<unsigned char>((bits >> 16) & 0xff),
        static_cast<unsigned char>((bits >> 8) & 0xff),
        static_cast<unsigned char>(bits & 0xff),
    };
    f.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

std::shared_ptr<plapoint::PointCloud<float, plamatrix::Device::CPU>> makeCloudWithColors()
{
    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using PointMatrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;
    using ColorMatrix = plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>;

    PointMatrix points(2, 3);
    points.setValue(0, 0, 1.0f);
    points.setValue(0, 1, 2.0f);
    points.setValue(0, 2, 3.0f);
    points.setValue(1, 0, 4.0f);
    points.setValue(1, 1, 5.0f);
    points.setValue(1, 2, 6.0f);

    ColorMatrix colors(2, 3);
    colors.setValue(0, 0, 10);
    colors.setValue(0, 1, 20);
    colors.setValue(0, 2, 30);
    colors.setValue(1, 0, 200);
    colors.setValue(1, 1, 210);
    colors.setValue(1, 2, 220);

    auto cloud = std::make_shared<Cloud>(std::move(points));
    cloud->setColors(std::move(colors));
    return cloud;
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

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
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

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
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

TEST(PlyIOTest, ReadsAsciiIntensityAsGrayscaleColors)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "property uchar intensity\n"
          << "end_header\n";
        f << "1.0 2.0 3.0 17\n"
          << "4.0 5.0 6.0 240\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);

    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->getValue(0, 0), 17);
    EXPECT_EQ(cloud->colors()->getValue(0, 1), 17);
    EXPECT_EQ(cloud->colors()->getValue(0, 2), 17);
    EXPECT_EQ(cloud->colors()->getValue(1, 0), 240);
    EXPECT_EQ(cloud->colors()->getValue(1, 1), 240);
    EXPECT_EQ(cloud->colors()->getValue(1, 2), 240);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsAsciiRgbColorsUsingHeaderOrder)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 2\n"
          << "property uchar blue\n"
          << "property float z\n"
          << "property uchar red\n"
          << "property float x\n"
          << "property uchar green\n"
          << "property float y\n"
          << "end_header\n";
        f << "30 3.0 10 1.0 20 2.0\n"
          << "60 6.0 40 4.0 50 5.0\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);

    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->getValue(0, 0), 10);
    EXPECT_EQ(cloud->colors()->getValue(0, 1), 20);
    EXPECT_EQ(cloud->colors()->getValue(0, 2), 30);
    EXPECT_EQ(cloud->colors()->getValue(1, 0), 40);
    EXPECT_EQ(cloud->colors()->getValue(1, 1), 50);
    EXPECT_EQ(cloud->colors()->getValue(1, 2), 60);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 3.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBinaryIntensityAsGrayscaleColors)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path, std::ios::binary);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "property uchar intensity\n"
          << "end_header\n";
        const float p0[3] = {1.0f, 2.0f, 3.0f};
        const unsigned char i0 = 18;
        const float p1[3] = {4.0f, 5.0f, 6.0f};
        const unsigned char i1 = 241;
        f.write(reinterpret_cast<const char*>(p0), sizeof(p0));
        f.write(reinterpret_cast<const char*>(&i0), sizeof(i0));
        f.write(reinterpret_cast<const char*>(p1), sizeof(p1));
        f.write(reinterpret_cast<const char*>(&i1), sizeof(i1));
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);

    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->getValue(0, 0), 18);
    EXPECT_EQ(cloud->colors()->getValue(0, 1), 18);
    EXPECT_EQ(cloud->colors()->getValue(0, 2), 18);
    EXPECT_EQ(cloud->colors()->getValue(1, 0), 241);
    EXPECT_EQ(cloud->colors()->getValue(1, 1), 241);
    EXPECT_EQ(cloud->colors()->getValue(1, 2), 241);

    std::remove(path.c_str());
}

TEST(PlyIOTest, RoundtripAsciiPreservesColors)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();

    const auto cloud = makeCloudWithColors();
    EXPECT_NO_THROW(plapoint::io::writePly(path, *cloud, plapoint::io::PlyFormat::ASCII));

    auto loaded = plapoint::io::readPly<float>(path);

    ASSERT_EQ(loaded->size(), 2u);
    ASSERT_TRUE(loaded->hasColors());
    EXPECT_EQ(loaded->colors()->getValue(0, 0), 10);
    EXPECT_EQ(loaded->colors()->getValue(0, 1), 20);
    EXPECT_EQ(loaded->colors()->getValue(0, 2), 30);
    EXPECT_EQ(loaded->colors()->getValue(1, 0), 200);
    EXPECT_EQ(loaded->colors()->getValue(1, 1), 210);
    EXPECT_EQ(loaded->colors()->getValue(1, 2), 220);

    std::remove(path.c_str());
}

TEST(PlyIOTest, RoundtripBinaryPreservesColors)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();

    const auto cloud = makeCloudWithColors();
    EXPECT_NO_THROW(plapoint::io::writePly(path, *cloud, plapoint::io::PlyFormat::BinaryLE));

    auto loaded = plapoint::io::readPly<float>(path);

    ASSERT_EQ(loaded->size(), 2u);
    ASSERT_TRUE(loaded->hasColors());
    EXPECT_EQ(loaded->colors()->getValue(0, 0), 10);
    EXPECT_EQ(loaded->colors()->getValue(0, 1), 20);
    EXPECT_EQ(loaded->colors()->getValue(0, 2), 30);
    EXPECT_EQ(loaded->colors()->getValue(1, 0), 200);
    EXPECT_EQ(loaded->colors()->getValue(1, 1), 210);
    EXPECT_EQ(loaded->colors()->getValue(1, 2), 220);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsAsciiVertexPropertiesUsingHeaderOrder)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 2\n"
          << "property float intensity\n"
          << "property list uchar int adjacent_vertices\n"
          << "property float z\n"
          << "property float x\n"
          << "property float y\n"
          << "property float nx\n"
          << "property float ny\n"
          << "property float nz\n"
          << "end_header\n";
        f << "99 2 7 8 3.0 1.0 2.0 0.0 0.0 1.0\n"
          << "42 0 6.0 4.0 5.0 1.0 0.0 0.0\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasNormals());
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 6.0f);
    EXPECT_FLOAT_EQ(cloud->normals()->getValue(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(cloud->normals()->getValue(1, 0), 1.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBinaryVertexPropertiesUsingHeaderOrder)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path, std::ios::binary);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "element vertex 2\n"
          << "property uchar red\n"
          << "property float intensity\n"
          << "property list uchar int adjacent_vertices\n"
          << "property float z\n"
          << "property float x\n"
          << "property float y\n"
          << "property float nx\n"
          << "property float ny\n"
          << "property float nz\n"
          << "end_header\n";

        const unsigned char red0 = 9;
        const float intensity0 = 99.0f;
        const unsigned char listCount0 = 2;
        const int list0[2] = {7, 8};
        const float values0[6] = {3.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f};

        const unsigned char red1 = 4;
        const float intensity1 = 42.0f;
        const unsigned char listCount1 = 0;
        const float values1[6] = {6.0f, 4.0f, 5.0f, 1.0f, 0.0f, 0.0f};

        f.write(reinterpret_cast<const char*>(&red0), sizeof(red0));
        f.write(reinterpret_cast<const char*>(&intensity0), sizeof(intensity0));
        f.write(reinterpret_cast<const char*>(&listCount0), sizeof(listCount0));
        f.write(reinterpret_cast<const char*>(list0), sizeof(list0));
        f.write(reinterpret_cast<const char*>(values0), sizeof(values0));

        f.write(reinterpret_cast<const char*>(&red1), sizeof(red1));
        f.write(reinterpret_cast<const char*>(&intensity1), sizeof(intensity1));
        f.write(reinterpret_cast<const char*>(&listCount1), sizeof(listCount1));
        f.write(reinterpret_cast<const char*>(values1), sizeof(values1));
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasNormals());
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 6.0f);
    EXPECT_FLOAT_EQ(cloud->normals()->getValue(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(cloud->normals()->getValue(1, 0), 1.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsPointOffsetComment)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
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

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
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

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
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

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
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

    const plapoint::test::TempFile little_endian_file(".ply");
    const plapoint::test::TempFile big_endian_file(".ply");
    writeAndRead(little_endian_file.string(), false);
    writeAndRead(big_endian_file.string(), true);
}

TEST(PlyIOTest, FaceElementDoesNotOverrideVertexCount)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 3\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "element face 1\n"
          << "property list uchar int vertex_indices\n"
          << "end_header\n"
          << "1.0 2.0 3.0\n"
          << "4.0 5.0 6.0\n"
          << "7.0 8.0 9.0\n"
          << "3 0 1 2\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    ASSERT_EQ(cloud->size(), 3u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(2, 2), 9.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBinaryVertexPropertiesWithUcharColors)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path, std::ios::binary);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "comment POINT_OFFSET 1000.0 -2000.0 3000.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "property uchar red\n"
          << "property uchar green\n"
          << "property uchar blue\n"
          << "element face 1\n"
          << "property list uchar int vertex_indices\n"
          << "end_header\n";
        const float p0[3] = {1.0f, 2.0f, 3.0f};
        const unsigned char c0[3] = {10, 20, 30};
        const float p1[3] = {4.0f, 5.0f, 6.0f};
        const unsigned char c1[3] = {40, 50, 60};
        f.write(reinterpret_cast<const char*>(p0), sizeof(p0));
        f.write(reinterpret_cast<const char*>(c0), sizeof(c0));
        f.write(reinterpret_cast<const char*>(p1), sizeof(p1));
        f.write(reinterpret_cast<const char*>(c1), sizeof(c1));
        const unsigned char faceCount = 3;
        const int indices[3] = {0, 1, 2};
        f.write(reinterpret_cast<const char*>(&faceCount), sizeof(faceCount));
        f.write(reinterpret_cast<const char*>(indices), sizeof(indices));
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1001.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), -1998.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 3003.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 0), 1004.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 1), -1995.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 3006.0f);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->getValue(0, 0), 10);
    EXPECT_EQ(cloud->colors()->getValue(0, 1), 20);
    EXPECT_EQ(cloud->colors()->getValue(0, 2), 30);
    EXPECT_EQ(cloud->colors()->getValue(1, 0), 40);
    EXPECT_EQ(cloud->colors()->getValue(1, 1), 50);
    EXPECT_EQ(cloud->colors()->getValue(1, 2), 60);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsAsciiWhenFaceElementPrecedesVertexElement)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element face 1\n"
          << "property list uchar int vertex_indices\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n"
          << "3 0 1 0\n"
          << "1.0 2.0 3.0\n"
          << "4.0 5.0 6.0\n";
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 6.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBinaryWhenFaceElementPrecedesVertexElement)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path, std::ios::binary);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "comment POINT_OFFSET 100.0 -200.0 300.0\n"
          << "element face 1\n"
          << "property list uchar int vertex_indices\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        const unsigned char faceCount = 3;
        const int indices[3] = {0, 1, 0};
        const float p0[3] = {1.0f, 2.0f, 3.0f};
        const float p1[3] = {4.0f, 5.0f, 6.0f};
        f.write(reinterpret_cast<const char*>(&faceCount), sizeof(faceCount));
        f.write(reinterpret_cast<const char*>(indices), sizeof(indices));
        f.write(reinterpret_cast<const char*>(p0), sizeof(p0));
        f.write(reinterpret_cast<const char*>(p1), sizeof(p1));
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 101.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), -198.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 306.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, ReadsBigEndianBinaryWhenFaceElementPrecedesVertexElement)
{
    using Scalar = float;

    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    {
        std::ofstream f(path, std::ios::binary);
        f << "ply\n"
          << "format binary_big_endian 1.0\n"
          << "comment POINT_OFFSET 100.0 -200.0 300.0\n"
          << "element face 1\n"
          << "property list ushort int vertex_indices\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        writeBigEndianUint16(f, 3);
        writeBigEndianInt32(f, 0);
        writeBigEndianInt32(f, 1);
        writeBigEndianInt32(f, 0);
        for (float value : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
        {
            writeBigEndianFloat(f, value);
        }
    }

    auto cloud = plapoint::io::readPly<Scalar>(path);
    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 101.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), -198.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 306.0f);

    std::remove(path.c_str());
}

TEST(PlyIOTest, RejectsInvalidMagicHeader)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "not_ply\n"
          << "format ascii 1.0\n"
          << "element vertex 1\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n"
          << "1 2 3\n";
    }

    EXPECT_THROW((void)plapoint::io::readPly<float>(path), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(PlyIOTest, RejectsHeaderWithoutEndHeader)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 1\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n";
    }

    expectThrowsMessageContaining<std::runtime_error>(
        [&]() { (void)plapoint::io::readPly<float>(path); },
        "end_header");

    std::filesystem::remove(path);
}

TEST(PlyIOTest, RejectsNegativeElementCount)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex -1\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
    }

    expectThrowsMessageContaining<std::runtime_error>(
        [&]() { (void)plapoint::io::readPly<float>(path); },
        "negative element count");

    std::filesystem::remove(path);
}

TEST(PlyIOTest, RejectsMissingRequiredVertexCoordinates)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 1\n"
          << "property float x\n"
          << "property float y\n"
          << "end_header\n"
          << "1 2\n";
    }

    expectThrowsMessageContaining<std::runtime_error>(
        [&]() { (void)plapoint::io::readPly<float>(path); },
        "x, y, and z");

    std::filesystem::remove(path);
}

TEST(PlyIOTest, RejectsTruncatedAsciiVertexRow)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 2\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n"
          << "1 2 3\n"
          << "4 5\n";
    }

    expectThrowsMessageContaining<std::runtime_error>(
        [&]() { (void)plapoint::io::readPly<float>(path); },
        "ASCII vertex row");

    std::filesystem::remove(path);
}

TEST(PlyIOTest, RejectsTruncatedBinaryVertexPayload)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path, std::ios::binary);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "element vertex 1\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        const float xy[2] = {1.0f, 2.0f};
        f.write(reinterpret_cast<const char*>(xy), sizeof(xy));
    }

    expectThrowsMessageContaining<std::runtime_error>(
        [&]() { (void)plapoint::io::readPly<float>(path); },
        "binary data");

    std::filesystem::remove(path);
}

TEST(PlyIOTest, IgnoresIncompleteNormalTriplet)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format ascii 1.0\n"
          << "element vertex 1\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "property float nx\n"
          << "property float ny\n"
          << "end_header\n"
          << "1 2 3 0.5 0.25\n";
    }

    auto cloud = plapoint::io::readPly<float>(path);

    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_FALSE(cloud->hasNormals());
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 3.0f);

    std::filesystem::remove(path);
}

TEST(PlyIOTest, RejectsUnsupportedBinaryVertexPropertyType)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path, std::ios::binary);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "element vertex 1\n"
          << "property long x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        const std::int64_t x = 1;
        const float yz[2] = {2.0f, 3.0f};
        f.write(reinterpret_cast<const char*>(&x), sizeof(x));
        f.write(reinterpret_cast<const char*>(yz), sizeof(yz));
    }

    EXPECT_THROW((void)plapoint::io::readPly<float>(path), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(PlyIOTest, RejectsNegativeBinaryListCount)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path, std::ios::binary);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "element face 1\n"
          << "property list char int vertex_indices\n"
          << "element vertex 1\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        const std::int8_t negative_count = -1;
        const float point[3] = {1.0f, 2.0f, 3.0f};
        f.write(reinterpret_cast<const char*>(&negative_count), sizeof(negative_count));
        f.write(reinterpret_cast<const char*>(point), sizeof(point));
    }

    EXPECT_THROW((void)plapoint::io::readPly<float>(path), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(PlyIOTest, SkipsBinaryScalarPropertiesOnNonVertexElementsBeforeVertices)
{
    const plapoint::test::TempFile temp_file(".ply");
    const auto path = temp_file.string();
    std::filesystem::remove(path);
    {
        std::ofstream f(path, std::ios::binary);
        ASSERT_TRUE(f);
        f << "ply\n"
          << "format binary_little_endian 1.0\n"
          << "element edge 1\n"
          << "property int vertex1\n"
          << "property int vertex2\n"
          << "property float confidence\n"
          << "element vertex 1\n"
          << "property float x\n"
          << "property float y\n"
          << "property float z\n"
          << "end_header\n";
        const std::int32_t edge_vertices[2] = {10, 11};
        const float confidence = 0.5f;
        const float point[3] = {7.0f, 8.0f, 9.0f};
        f.write(reinterpret_cast<const char*>(edge_vertices), sizeof(edge_vertices));
        f.write(reinterpret_cast<const char*>(&confidence), sizeof(confidence));
        f.write(reinterpret_cast<const char*>(point), sizeof(point));
    }

    auto cloud = plapoint::io::readPly<float>(path);

    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 7.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 8.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 9.0f);

    std::filesystem::remove(path);
}
