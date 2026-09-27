#include <gtest/gtest.h>

#include "temp_file.h"

#include <plapoint/core/point_cloud.h>
#include <plapoint/io/ply_io.h>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace
{

    using Cloud = plapoint::GeometryCloud<float>;
    using FloatMatrix = plamatrix::MatrixXf;

    void writeStreamFixture(const std::string& path)
    {
        FloatMatrix points(4, 3);
        FloatMatrix normals(4, 3);
        plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(4, 3);

        for (int i = 0; i < 4; ++i)
        {
            points.operator()(i, 0) = static_cast<float>(i + 1);
            points.operator()(i, 1) = static_cast<float>(10 + i);
            points.operator()(i, 2) = static_cast<float>(20 + i);

            normals.operator()(i, 0) = 0.0f;
            normals.operator()(i, 1) = 0.0f;
            normals.operator()(i, 2) = 1.0f;

            colors.operator()(i, 0) = static_cast<std::uint8_t>(10 + i);
            colors.operator()(i, 1) = static_cast<std::uint8_t>(20 + i);
            colors.operator()(i, 2) = static_cast<std::uint8_t>(30 + i);
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

TEST(PlyStreamTest, VectorDecodersRejectShortRecords)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    writeStreamFixture(path);

    plapoint::io::PlyVertexStreamHeader header;
    ASSERT_TRUE(plapoint::io::parseBinaryPlyVertexStreamHeader(path, &header));
    ASSERT_GT(header.vertexStride, 0);

    const std::vector<char> empty_record;
    const std::vector<char> short_record(
        static_cast<std::size_t>(header.vertexStride - 1));
    EXPECT_THROW(plapoint::io::readPlyVertexPoint(empty_record, header), std::invalid_argument);
    EXPECT_THROW(plapoint::io::readPlyVertexPoint(short_record, header), std::invalid_argument);
    EXPECT_THROW(plapoint::io::readPlyVertexPoint64(empty_record, header), std::invalid_argument);
    EXPECT_THROW(plapoint::io::readPlyVertexPoint64(short_record, header), std::invalid_argument);
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

TEST(PlyStreamTest, HighLevelVisitorStreamsChunksAndReportsProgress)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    writeStreamFixture(path);

    plapoint::io::PlyVertexStreamOptions options;
    options.chunkBytes = 54;
    std::vector<std::uint64_t> progress;
    options.progress = [&](std::uint64_t processed, std::uint64_t total)
    {
        EXPECT_EQ(total, 4u);
        progress.push_back(processed);
    };

    std::vector<float> xs;
    const auto status = plapoint::io::forEachBinaryPlyVertex(
        path,
        options,
        [&](const char* record,
            const plapoint::io::PlyVertexPoint64& point,
            std::uint64_t index)
        {
            ASSERT_NE(record, nullptr);
            EXPECT_EQ(index, xs.size());
            xs.push_back(point.x);
        });

    EXPECT_EQ(status, plapoint::io::PlyVertexStreamStatus::Completed);
    EXPECT_EQ(xs, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(progress, (std::vector<std::uint64_t>{2u, 4u}));
}

TEST(PlyStreamTest, HighLevelVisitorSupportsStopAndCancellation)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    writeStreamFixture(path);

    plapoint::io::PlyVertexStreamOptions options;
    const auto stopped = plapoint::io::forEachBinaryPlyVertex(
        path,
        options,
        [](const char*, const plapoint::io::PlyVertexPoint64&, std::uint64_t index)
        {
            return index < 1;
        });
    EXPECT_EQ(stopped, plapoint::io::PlyVertexStreamStatus::StoppedByVisitor);

    std::atomic_bool cancelled{true};
    options.cancellationFlag = &cancelled;
    int visits = 0;
    const auto cancelled_status = plapoint::io::forEachBinaryPlyVertex(
        path,
        options,
        [&](const char*, const plapoint::io::PlyVertexPoint64&, std::uint64_t)
        {
            ++visits;
        });
    EXPECT_EQ(cancelled_status, plapoint::io::PlyVertexStreamStatus::Cancelled);
    EXPECT_EQ(visits, 0);
}

TEST(PlyStreamTest, HighLevelVisitorAcceptsAnEmptyVertexElement)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    {
        std::ofstream output(path, std::ios::binary);
        output << "ply\n"
               << "format binary_little_endian 1.0\n"
               << "element vertex 0\n"
               << "property float x\n"
               << "property float y\n"
               << "property float z\n"
               << "end_header\n";
    }

    int visits = 0;
    const auto status = plapoint::io::forEachBinaryPlyVertex(
        path,
        {},
        [&](const char*, const plapoint::io::PlyVertexPoint64&, std::uint64_t)
        {
            ++visits;
        });

    EXPECT_EQ(status, plapoint::io::PlyVertexStreamStatus::Completed);
    EXPECT_EQ(visits, 0);
}

TEST(PlyStreamTest, HeaderRejectsMissingOrMalformedVertexCounts)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    const std::vector<std::string> malformed_elements{
        "element vertex",
        "element vertex nope",
        "element vertex -1",
        "element vertex 0 trailing"};

    for (const std::string& element : malformed_elements)
    {
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output << "ply\n"
                   << "format binary_little_endian 1.0\n"
                   << element << "\n"
                   << "property float x\n"
                   << "property float y\n"
                   << "property float z\n"
                   << "end_header\n";
        }

        plapoint::io::PlyVertexStreamHeader header;
        std::string error;
        EXPECT_FALSE(plapoint::io::parseBinaryPlyVertexStreamHeader(path, &header, &error))
            << element;
        EXPECT_FALSE(error.empty()) << element;
    }
}

TEST(PlyStreamTest, HeaderRejectsNonEmptyElementsBeforeVertices)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    {
        std::ofstream output(path, std::ios::binary);
        output << "ply\n"
               << "format binary_little_endian 1.0\n"
               << "element edge 1\n"
               << "property uchar flag\n"
               << "element vertex 1\n"
               << "property float x\n"
               << "property float y\n"
               << "property float z\n"
               << "end_header\n";
    }

    plapoint::io::PlyVertexStreamHeader header;
    std::string error;
    EXPECT_FALSE(plapoint::io::parseBinaryPlyVertexStreamHeader(path, &header, &error));
    EXPECT_NE(error.find("first non-empty element"), std::string::npos);
}

TEST(PlyStreamTest, HighLevelVisitorPreservesDoublePrecision)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    const double coordinates[3]{
        123456789.123456789,
        -987654321.654321,
        0.123456789012345};
    {
        std::ofstream output(path, std::ios::binary);
        output << "ply\n"
               << "format binary_little_endian 1.0\n"
               << "element vertex 1\n"
               << "property double x\n"
               << "property double y\n"
               << "property double z\n"
               << "end_header\n";
        output.write(reinterpret_cast<const char*>(coordinates), sizeof(coordinates));
    }

    int visits = 0;
    const auto status = plapoint::io::forEachBinaryPlyVertex(
        path,
        {},
        [&](const char*, const plapoint::io::PlyVertexPoint64& point, std::uint64_t index)
        {
            ++visits;
            EXPECT_EQ(index, 0u);
            EXPECT_DOUBLE_EQ(point.x, coordinates[0]);
            EXPECT_DOUBLE_EQ(point.y, coordinates[1]);
            EXPECT_DOUBLE_EQ(point.z, coordinates[2]);
            EXPECT_NE(point.x, static_cast<double>(static_cast<float>(coordinates[0])));
        });

    EXPECT_EQ(status, plapoint::io::PlyVertexStreamStatus::Completed);
    EXPECT_EQ(visits, 1);
}

TEST(PlyStreamTest, HighLevelVisitorDoesNotMaterializeChunkMetadata)
{
    const plapoint::test::TempFile temp_file(".ply");
    const std::string path = temp_file.string();
    {
        std::ofstream output(path, std::ios::binary);
        output << "ply\n"
               << "format binary_little_endian 1.0\n"
               << "element vertex 1000000000\n"
               << "property float x\n"
               << "property float y\n"
               << "property float z\n"
               << "end_header\n";
        const float point[3]{1.0f, 2.0f, 3.0f};
        output.write(reinterpret_cast<const char*>(point), sizeof(point));
    }

    plapoint::io::PlyVertexStreamOptions options;
    options.chunkBytes = static_cast<int>(sizeof(float) * 3);
    int visits = 0;
    const auto status = plapoint::io::forEachBinaryPlyVertex(
        path,
        options,
        [&](const char*, const plapoint::io::PlyVertexPoint64& point, std::uint64_t index)
        {
            ++visits;
            EXPECT_EQ(index, 0u);
            EXPECT_FLOAT_EQ(point.x, 1.0f);
            return false;
        });

    EXPECT_EQ(status, plapoint::io::PlyVertexStreamStatus::StoppedByVisitor);
    EXPECT_EQ(visits, 1);
}
