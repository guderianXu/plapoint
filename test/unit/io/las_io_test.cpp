#include <gtest/gtest.h>
#include "temp_file.h"
#include <plapoint/io/las_io.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {

void writeI32(std::array<char, 28>& record, std::size_t offset, int32_t value)
{
    std::memcpy(record.data() + offset, &value, sizeof(value));
}

std::array<char, 28> makeLasFormat1Record(int32_t x, int32_t y, int32_t z)
{
    std::array<char, 28> record{};
    writeI32(record, 0, x);
    writeI32(record, 4, y);
    writeI32(record, 8, z);
    return record;
}

} // namespace

TEST(LasIOTest, ReadsLasFormat1UsingHeaderRecordLength)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = sizeof(plapoint::io::LasHeader);
    header.point_data_offset = sizeof(plapoint::io::LasHeader);
    header.point_data_format = 1;
    header.point_data_record_length = 28;
    header.num_point_records = 2;
    header.num_points_by_return[0] = 2;
    header.x_scale_factor = 1.0;
    header.y_scale_factor = 1.0;
    header.z_scale_factor = 1.0;

    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        const auto first = makeLasFormat1Record(1, 2, 3);
        const auto second = makeLasFormat1Record(4, 5, 6);
        out.write(first.data(), static_cast<std::streamsize>(first.size()));
        out.write(second.data(), static_cast<std::streamsize>(second.size()));
    }

    auto cloud = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(cloud->points().getValue(1, 2), 6.0f);
}

TEST(LasIOTest, WriteLasUsesPointFormatMatchingRecordLength)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::PointCloud<float, plamatrix::Device::CPU> cloud(1);
    cloud.points().setValue(0, 0, 1.0f);
    cloud.points().setValue(0, 1, 2.0f);
    cloud.points().setValue(0, 2, 3.0f);

    plapoint::io::writeLas(path, cloud, 0.01);

    plapoint::io::LasHeader header{};
    {
        std::ifstream in(path, std::ios::binary);
        ASSERT_TRUE(in);
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
    }

    std::filesystem::remove(path);

    EXPECT_EQ(header.point_data_format, 0);
    EXPECT_EQ(header.point_data_record_length, 20);
}
