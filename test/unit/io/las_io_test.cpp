#include <gtest/gtest.h>
#include "temp_file.h"
#include <plapoint/io/las_io.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {

template <std::size_t Size>
void writeI32(std::array<char, Size>& record, std::size_t offset, int32_t value)
{
    std::memcpy(record.data() + offset, &value, sizeof(value));
}

template <std::size_t Size>
void writeU16(std::array<char, Size>& record, std::size_t offset, std::uint16_t value)
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

std::array<char, 20> makeLasFormat0Record(int32_t x,
                                          int32_t y,
                                          int32_t z,
                                          std::uint16_t intensity)
{
    std::array<char, 20> record{};
    writeI32(record, 0, x);
    writeI32(record, 4, y);
    writeI32(record, 8, z);
    writeU16(record, 12, intensity);
    return record;
}

std::array<char, 26> makeLasFormat2Record(int32_t x,
                                          int32_t y,
                                          int32_t z,
                                          std::uint8_t r,
                                          std::uint8_t g,
                                          std::uint8_t b)
{
    std::array<char, 26> record{};
    writeI32(record, 0, x);
    writeI32(record, 4, y);
    writeI32(record, 8, z);
    writeU16(record, 20, static_cast<std::uint16_t>(r) * 257u);
    writeU16(record, 22, static_cast<std::uint16_t>(g) * 257u);
    writeU16(record, 24, static_cast<std::uint16_t>(b) * 257u);
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

TEST(LasIOTest, RejectsUnsupportedPointDataFormat)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = sizeof(plapoint::io::LasHeader);
    header.point_data_offset = sizeof(plapoint::io::LasHeader);
    header.point_data_format = 9;
    header.point_data_record_length = 20;
    header.num_point_records = 0;
    header.x_scale_factor = 1.0;
    header.y_scale_factor = 1.0;
    header.z_scale_factor = 1.0;

    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    }

    EXPECT_THROW((void)plapoint::io::readLas<float>(path), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(LasIOTest, RejectsPointFormatWithTooShortRecordLength)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    auto write_header = [&](std::uint8_t point_format, std::uint16_t record_length) {
        plapoint::io::LasHeader header{};
        std::memcpy(header.file_signature, "LASF", 4);
        header.version_major = 1;
        header.version_minor = 2;
        header.header_size = sizeof(plapoint::io::LasHeader);
        header.point_data_offset = sizeof(plapoint::io::LasHeader);
        header.point_data_format = point_format;
        header.point_data_record_length = record_length;
        header.num_point_records = 0;
        header.x_scale_factor = 1.0;
        header.y_scale_factor = 1.0;
        header.z_scale_factor = 1.0;

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    };

    write_header(1, 20);
    EXPECT_THROW((void)plapoint::io::readLas<float>(path), std::runtime_error);

    write_header(5, 34);
    EXPECT_THROW((void)plapoint::io::readLas<float>(path), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(LasIOTest, ReadsLasIntensityValues)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = sizeof(plapoint::io::LasHeader);
    header.point_data_offset = sizeof(plapoint::io::LasHeader);
    header.point_data_format = 0;
    header.point_data_record_length = 20;
    header.num_point_records = 2;
    header.num_points_by_return[0] = 2;
    header.x_scale_factor = 1.0;
    header.y_scale_factor = 1.0;
    header.z_scale_factor = 1.0;

    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        const auto first = makeLasFormat0Record(1, 2, 3, 17);
        const auto second = makeLasFormat0Record(4, 5, 6, 4096);
        out.write(first.data(), static_cast<std::streamsize>(first.size()));
        out.write(second.data(), static_cast<std::streamsize>(second.size()));
    }

    auto cloud = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasIntensities());
    EXPECT_EQ(cloud->intensities()->getValue(0, 0), 17);
    EXPECT_EQ(cloud->intensities()->getValue(1, 0), 4096);
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

TEST(LasIOTest, ReadsLasFormat2RgbColors)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = sizeof(plapoint::io::LasHeader);
    header.point_data_offset = sizeof(plapoint::io::LasHeader);
    header.point_data_format = 2;
    header.point_data_record_length = 26;
    header.num_point_records = 2;
    header.num_points_by_return[0] = 2;
    header.x_scale_factor = 1.0;
    header.y_scale_factor = 1.0;
    header.z_scale_factor = 1.0;

    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        const auto first = makeLasFormat2Record(1, 2, 3, 10, 20, 30);
        const auto second = makeLasFormat2Record(4, 5, 6, 200, 210, 220);
        out.write(first.data(), static_cast<std::streamsize>(first.size()));
        out.write(second.data(), static_cast<std::streamsize>(second.size()));
    }

    auto cloud = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->getValue(0, 0), 10);
    EXPECT_EQ(cloud->colors()->getValue(0, 1), 20);
    EXPECT_EQ(cloud->colors()->getValue(0, 2), 30);
    EXPECT_EQ(cloud->colors()->getValue(1, 0), 200);
    EXPECT_EQ(cloud->colors()->getValue(1, 1), 210);
    EXPECT_EQ(cloud->colors()->getValue(1, 2), 220);
}

TEST(LasIOTest, WriteLasWithColorsUsesFormat2AndRoundtripsColors)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using ColorMatrix = plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>;

    Cloud cloud(2);
    cloud.points().setValue(0, 0, 1.0f);
    cloud.points().setValue(0, 1, 2.0f);
    cloud.points().setValue(0, 2, 3.0f);
    cloud.points().setValue(1, 0, 4.0f);
    cloud.points().setValue(1, 1, 5.0f);
    cloud.points().setValue(1, 2, 6.0f);

    ColorMatrix colors(2, 3);
    colors.setValue(0, 0, 11);
    colors.setValue(0, 1, 22);
    colors.setValue(0, 2, 33);
    colors.setValue(1, 0, 201);
    colors.setValue(1, 1, 211);
    colors.setValue(1, 2, 221);
    cloud.setColors(std::move(colors));

    plapoint::io::writeLas(path, cloud, 0.01);

    plapoint::io::LasHeader header{};
    {
        std::ifstream in(path, std::ios::binary);
        ASSERT_TRUE(in);
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
    }
    auto loaded = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    EXPECT_EQ(header.point_data_format, 2);
    EXPECT_EQ(header.point_data_record_length, 26);
    ASSERT_TRUE(loaded->hasColors());
    EXPECT_EQ(loaded->colors()->getValue(0, 0), 11);
    EXPECT_EQ(loaded->colors()->getValue(0, 1), 22);
    EXPECT_EQ(loaded->colors()->getValue(0, 2), 33);
    EXPECT_EQ(loaded->colors()->getValue(1, 0), 201);
    EXPECT_EQ(loaded->colors()->getValue(1, 1), 211);
    EXPECT_EQ(loaded->colors()->getValue(1, 2), 221);
}

TEST(LasIOTest, WriteLasRoundtripsIntensityValues)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    using Cloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;
    using IntensityMatrix = plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>;

    Cloud cloud(2);
    cloud.points().setValue(0, 0, 1.0f);
    cloud.points().setValue(0, 1, 2.0f);
    cloud.points().setValue(0, 2, 3.0f);
    cloud.points().setValue(1, 0, 4.0f);
    cloud.points().setValue(1, 1, 5.0f);
    cloud.points().setValue(1, 2, 6.0f);

    IntensityMatrix intensities(2, 1);
    intensities.setValue(0, 0, 17);
    intensities.setValue(1, 0, 4096);
    cloud.setIntensities(std::move(intensities));

    plapoint::io::writeLas(path, cloud, 0.01);

    plapoint::io::LasHeader header{};
    {
        std::ifstream in(path, std::ios::binary);
        ASSERT_TRUE(in);
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
    }
    auto loaded = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    EXPECT_EQ(header.point_data_format, 0);
    EXPECT_EQ(header.point_data_record_length, 20);
    ASSERT_TRUE(loaded->hasIntensities());
    EXPECT_EQ(loaded->intensities()->getValue(0, 0), 17);
    EXPECT_EQ(loaded->intensities()->getValue(1, 0), 4096);
}

TEST(LasIOTest, WriteLasRejectsNonFiniteCoordinates)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::PointCloud<double, plamatrix::Device::CPU> cloud(2);
    cloud.points().setValue(0, 0, 0.0);
    cloud.points().setValue(0, 1, 1.0);
    cloud.points().setValue(0, 2, 2.0);
    cloud.points().setValue(1, 0, std::numeric_limits<double>::quiet_NaN());
    cloud.points().setValue(1, 1, 3.0);
    cloud.points().setValue(1, 2, 4.0);

    EXPECT_THROW(plapoint::io::writeLas(path, cloud, 0.001), std::invalid_argument);

    std::filesystem::remove(path);
}

TEST(LasIOTest, WriteLasRejectsQuantizedCoordinatesOutsideInt32Range)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::PointCloud<double, plamatrix::Device::CPU> cloud(2);
    cloud.points().setValue(0, 0, 0.0);
    cloud.points().setValue(0, 1, 0.0);
    cloud.points().setValue(0, 2, 0.0);
    cloud.points().setValue(1, 0, 3000000.0);
    cloud.points().setValue(1, 1, 0.0);
    cloud.points().setValue(1, 2, 0.0);

    EXPECT_THROW(plapoint::io::writeLas(path, cloud, 0.001), std::out_of_range);

    std::filesystem::remove(path);
}
