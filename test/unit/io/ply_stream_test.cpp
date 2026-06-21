#include <gtest/gtest.h>

#include "temp_file.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <plamatrix/plamatrix.h>

#include <cstdint>
#include <fstream>
#include <string>

namespace
{

using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
using FloatMatrix = plamatrix::DenseMatrix<float, plamatrix::Device::CPU>;

void writeStreamFixture(const std::string& path)
{
    FloatMatrix points(4, 3);
    FloatMatrix normals(4, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(4, 3);

    for (int i = 0; i < 4; ++i)
    {
        points.setValue(i, 0, static_cast<float>(i + 1));
        points.setValue(i, 1, static_cast<float>(10 + i));
        points.setValue(i, 2, static_cast<float>(20 + i));

        normals.setValue(i, 0, 0.0f);
        normals.setValue(i, 1, 0.0f);
        normals.setValue(i, 2, 1.0f);

        colors.setValue(i, 0, static_cast<std::uint8_t>(10 + i));
        colors.setValue(i, 1, static_cast<std::uint8_t>(20 + i));
        colors.setValue(i, 2, static_cast<std::uint8_t>(30 + i));
    }

    Cloud cloud(std::move(points));
    cloud.setNormals(std::move(normals));
    cloud.setColors(std::move(colors));

    plapoint::io::writePly(path, cloud, plapoint::io::PlyFormat::BinaryLE);
}

} // namespace

TEST(PlyStreamTest, ParsesBinaryLittleEndianVertexLayout)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    writeStreamFixture(path);

    plapoint::io::PlyVertexStreamHeader header;
    std::string error;
    ASSERT_TRUE(plapoint::io::parseBinaryPlyVertexStreamHeader(path, &header, &error)) << error;

    EXPECT_TRUE(header.valid);
    EXPECT_TRUE(header.binaryLittleEndian);
    EXPECT_EQ(header.vertexCount, 4u);
    EXPECT_EQ(header.vertexStride, 27);
    EXPECT_GT(header.dataStartOffset, std::streamoff{0});
    EXPECT_EQ(header.xProperty, 0);
    EXPECT_EQ(header.yProperty, 1);
    EXPECT_EQ(header.zProperty, 2);
    EXPECT_EQ(header.redProperty, 3);
    EXPECT_EQ(header.greenProperty, 4);
    EXPECT_EQ(header.blueProperty, 5);
    EXPECT_EQ(header.nxProperty, 6);
    EXPECT_EQ(header.nyProperty, 7);
    EXPECT_EQ(header.nzProperty, 8);
    EXPECT_TRUE(header.hasColors());
    EXPECT_TRUE(header.hasNormals());
}

TEST(PlyStreamTest, ReadsChunksAndDecodesVertexRecords)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    writeStreamFixture(path);

    plapoint::io::PlyVertexStreamHeader header;
    ASSERT_TRUE(plapoint::io::parseBinaryPlyVertexStreamHeader(path, &header));

    const auto chunks = plapoint::io::makePlyVertexChunks(header, 54);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].startVertex, 0u);
    EXPECT_EQ(chunks[0].vertexCount, 2u);
    EXPECT_EQ(chunks[1].startVertex, 2u);
    EXPECT_EQ(chunks[1].vertexCount, 2u);

    std::ifstream file(path, std::ios::binary);
    std::vector<char> buffer;
    std::string error;
    ASSERT_TRUE(plapoint::io::readPlyVertexChunk(file, header, chunks[1], &buffer, &error)) << error;
    ASSERT_EQ(buffer.size(), static_cast<std::size_t>(header.vertexStride * chunks[1].vertexCount));

    const auto first = plapoint::io::readPlyVertexPoint(buffer.data(), header);
    EXPECT_FLOAT_EQ(first.x, 3.0f);
    EXPECT_FLOAT_EQ(first.y, 12.0f);
    EXPECT_FLOAT_EQ(first.z, 22.0f);
    EXPECT_EQ(first.r, 12);
    EXPECT_EQ(first.g, 22);
    EXPECT_EQ(first.b, 32);
    EXPECT_TRUE(first.hasNormal);
    EXPECT_FLOAT_EQ(first.nx, 0.0f);
    EXPECT_FLOAT_EQ(first.ny, 0.0f);
    EXPECT_FLOAT_EQ(first.nz, 1.0f);
}

TEST(PlyStreamTest, SamplesBinaryVerticesWithoutLoadingWholeCloud)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    writeStreamFixture(path);

    plapoint::io::PlyVertexStreamHeader header;
    ASSERT_TRUE(plapoint::io::parseBinaryPlyVertexStreamHeader(path, &header));

    std::string error;
    const auto points = plapoint::io::sampleBinaryPlyVertices(path, header, 2, &error);
    ASSERT_EQ(points.size(), 2u) << error;
    EXPECT_FLOAT_EQ(points[0].x, 1.0f);
    EXPECT_FLOAT_EQ(points[1].x, 3.0f);
    EXPECT_EQ(points[1].r, 12);
}
