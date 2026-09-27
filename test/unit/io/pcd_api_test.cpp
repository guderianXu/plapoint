#include <gtest/gtest.h>

#include <plapoint/io/pcd_io.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace
{

    template <typename Save> void checkIntensityRoundTrip(const std::filesystem::path& path, Save save)
    {
        plapoint::PointCloud<plapoint::PointXYZI> input;
        input.header.frame_id = "sensor";
        input.sensor_origin_ = Eigen::Vector4f(1.0f, 2.0f, 3.0f, 0.0f);
        input.sensor_orientation_ = Eigen::Quaternionf(Eigen::AngleAxisf(0.25f, Eigen::Vector3f::UnitZ()));
        input.emplace_back(1.0f, 2.0f, 3.0f, 0.25f);
        input.emplace_back(4.0f, 5.0f, 6.0f, 0.75f);

        ASSERT_EQ(save(path.string(), input), 0);
        plapoint::PointCloud<plapoint::PointXYZI> output;
        ASSERT_EQ(plapoint::io::loadPCDFile(path.string(), output), 0);
        ASSERT_EQ(output.size(), 2u);
        EXPECT_FLOAT_EQ(output[0].intensity, 0.25f);
        EXPECT_FLOAT_EQ(output[1].z, 6.0f);
        EXPECT_TRUE(output.sensor_origin_.isApprox(input.sensor_origin_));
        EXPECT_TRUE(output.sensor_orientation_.isApprox(input.sensor_orientation_));
        std::filesystem::remove(path);
    }

} // namespace

TEST(PcdApiTest, ReadsAndWritesAsciiBinaryAndCompressedFiles)
{
    const std::filesystem::path directory(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(directory);
    checkIntensityRoundTrip(directory / "point_xyzi_ascii.pcd",
                            [](const std::string& path, const auto& cloud)
                            { return plapoint::io::savePCDFileASCII(path, cloud); });
    checkIntensityRoundTrip(directory / "point_xyzi_binary.pcd",
                            [](const std::string& path, const auto& cloud)
                            { return plapoint::io::savePCDFileBinary(path, cloud); });
    checkIntensityRoundTrip(directory / "point_xyzi_compressed.pcd",
                            [](const std::string& path, const auto& cloud)
                            { return plapoint::io::savePCDFileBinaryCompressed(path, cloud); });
}

TEST(PcdApiTest, HeaderAndBlobOverloadsPreserveFields)
{
    const std::filesystem::path directory(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(directory);
    const auto path = directory / "point_xyzrgba_blob.pcd";

    plapoint::PointCloud<plapoint::PointXYZRGBA> input;
    input.emplace_back(1.0f, 2.0f, 3.0f, 10, 20, 30, 40);
    ASSERT_EQ(plapoint::io::savePCDFileBinaryCompressed(path.string(), input), 0);

    plapoint::PCDReader reader;
    plapoint::PCLPointCloud2 header;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    int version = -1;
    int data_type = -1;
    unsigned int data_index = 0;
    ASSERT_EQ(reader.readHeader(path.string(), header, origin, orientation, version, data_type, data_index), 0);
    EXPECT_EQ(version, plapoint::PCDReader::PCD_V7);
    EXPECT_EQ(data_type, 2);
    EXPECT_GT(data_index, 0u);
    ASSERT_EQ(header.fields.size(), 4u);
    EXPECT_EQ(header.fields.back().name, "rgba");

    plapoint::PCLPointCloud2 blob;
    ASSERT_EQ(plapoint::io::loadPCDFile(path.string(), blob), 0);
    plapoint::PointCloud<plapoint::PointXYZRGBA> output;
    plapoint::fromPCLPointCloud2(blob, output);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].rgba, input[0].rgba);
    std::filesystem::remove(path);
}

TEST(PcdApiTest, StreamHeaderAndBodyOverloadsMatchFileOffsets)
{
    const std::filesystem::path directory(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(directory);
    const auto plain_path = directory / "point_xyzi_stream_ascii.pcd";
    const auto offset_path = directory / "point_xyzi_stream_offset.pcd";

    plapoint::PointCloud<plapoint::PointXYZI> input;
    input.emplace_back(1.0f, 2.0f, 3.0f, 0.25f);
    input.emplace_back(4.0f, 5.0f, 6.0f, 0.75f);
    ASSERT_EQ(plapoint::io::savePCDFileASCII(plain_path.string(), input), 0);

    std::ifstream file(plain_path, std::ios::binary);
    ASSERT_TRUE(file);
    const std::string payload((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::istringstream stream(payload);

    plapoint::PCDReader reader;
    plapoint::PCLPointCloud2 cloud;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    int version = -1;
    int data_type = -1;
    unsigned int data_index = 0;
    ASSERT_EQ(reader.readHeader(stream, cloud, origin, orientation, version, data_type, data_index), 0);
    EXPECT_EQ(data_type, 0);
    EXPECT_EQ(static_cast<std::streamoff>(data_index), stream.tellg());
    ASSERT_EQ(reader.readBodyASCII(stream, cloud, version), 0);

    plapoint::PointCloud<plapoint::PointXYZI> output;
    plapoint::fromPCLPointCloud2(cloud, output);
    ASSERT_EQ(output.size(), input.size());
    EXPECT_FLOAT_EQ(output[1].intensity, input[1].intensity);

    const std::string prefix = "archive-prefix\n";
    {
        std::ofstream offset_file(offset_path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(offset_file);
        offset_file << prefix << payload;
    }
    plapoint::PCLPointCloud2 offset_cloud;
    ASSERT_EQ(reader.readHeader(offset_path.string(),
                                offset_cloud,
                                origin,
                                orientation,
                                version,
                                data_type,
                                data_index,
                                static_cast<int>(prefix.size())),
              0);
    const auto data_line = payload.find("DATA ascii\n");
    ASSERT_NE(data_line, std::string::npos);
    const auto body = data_line + std::string("DATA ascii\n").size();
    EXPECT_EQ(data_index, prefix.size() + body);

    std::filesystem::remove(plain_path);
    std::filesystem::remove(offset_path);
}

TEST(PcdApiTest, BinaryBodyOverloadReadsCompressedMemory)
{
    const std::filesystem::path directory(PLAPOINT_TEST_TMP_DIR);
    std::filesystem::create_directories(directory);
    const auto path = directory / "point_xyzrgba_body_compressed.pcd";

    plapoint::PointCloud<plapoint::PointXYZRGBA> input;
    input.emplace_back(1.0f, 2.0f, 3.0f, 10, 20, 30, 40);
    ASSERT_EQ(plapoint::io::savePCDFileBinaryCompressed(path.string(), input), 0);
    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    plapoint::PCDReader reader;
    plapoint::PCLPointCloud2 cloud;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    int version = -1;
    int data_type = -1;
    unsigned int data_index = 0;
    ASSERT_EQ(reader.readHeader(path.string(), cloud, origin, orientation, version, data_type, data_index), 0);
    ASSERT_EQ(data_type, 2);
    ASSERT_EQ(reader.readBodyBinary(bytes.data(), cloud, version, true, data_index), 0);

    plapoint::PointCloud<plapoint::PointXYZRGBA> output;
    plapoint::fromPCLPointCloud2(cloud, output);
    ASSERT_EQ(output.size(), 1u);
    EXPECT_EQ(output[0].rgba, input[0].rgba);
    std::filesystem::remove(path);
}
