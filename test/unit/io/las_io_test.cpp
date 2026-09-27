#include <gtest/gtest.h>
#include "temp_file.h"
#include <plapoint/io/las_io.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <plamatrix/internal/core/device.h>

namespace {

template <std::size_t Size>
void writeI32(std::array<char, Size>& record, std::size_t offset, int32_t value)
{
    plapoint::io::detail::writeLeI32(record.data(), offset, value);
}

template <std::size_t Size>
void writeU16(std::array<char, Size>& record, std::size_t offset, std::uint16_t value)
{
    plapoint::io::detail::writeLeU16(record.data(), offset, value);
}

void writeLasHeader(std::ofstream& output, const plapoint::io::LasHeader& header)
{
    const auto bytes = plapoint::io::detail::encodeLasHeader(header);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

plapoint::io::LasHeader readLasHeader(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("Cannot open LAS test file");
    }
    std::array<char, plapoint::io::kLasHeaderSize> bytes{};
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input)
    {
        throw std::runtime_error("Cannot read LAS test header");
    }
    return plapoint::io::detail::decodeLasHeader(bytes);
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
    header.header_size = plapoint::io::kLasHeaderSize;
    header.point_data_offset = plapoint::io::kLasHeaderSize;
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
        writeLasHeader(out, header);

        const auto first = makeLasFormat1Record(1, 2, 3);
        const auto second = makeLasFormat1Record(4, 5, 6);
        out.write(first.data(), static_cast<std::streamsize>(first.size()));
        out.write(second.data(), static_cast<std::streamsize>(second.size()));
    }

    auto cloud = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points().operator()(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(cloud->points().operator()(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(cloud->points().operator()(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(cloud->points().operator()(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(cloud->points().operator()(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(cloud->points().operator()(1, 2), 6.0f);
}

TEST(LasIOTest, RejectsUnsupportedPointDataFormat)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = plapoint::io::kLasHeaderSize;
    header.point_data_offset = plapoint::io::kLasHeaderSize;
    header.point_data_format = 9;
    header.point_data_record_length = 20;
    header.num_point_records = 0;
    header.x_scale_factor = 1.0;
    header.y_scale_factor = 1.0;
    header.z_scale_factor = 1.0;

    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        writeLasHeader(out, header);
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
        header.header_size = plapoint::io::kLasHeaderSize;
        header.point_data_offset = plapoint::io::kLasHeaderSize;
        header.point_data_format = point_format;
        header.point_data_record_length = record_length;
        header.num_point_records = 0;
        header.x_scale_factor = 1.0;
        header.y_scale_factor = 1.0;
        header.z_scale_factor = 1.0;

        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out);
        writeLasHeader(out, header);
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
    header.header_size = plapoint::io::kLasHeaderSize;
    header.point_data_offset = plapoint::io::kLasHeaderSize;
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
        writeLasHeader(out, header);

        const auto first = makeLasFormat0Record(1, 2, 3, 17);
        const auto second = makeLasFormat0Record(4, 5, 6, 4096);
        out.write(first.data(), static_cast<std::streamsize>(first.size()));
        out.write(second.data(), static_cast<std::streamsize>(second.size()));
    }

    auto cloud = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasIntensities());
    EXPECT_EQ(cloud->intensities()->operator()(0, 0), 17);
    EXPECT_EQ(cloud->intensities()->operator()(1, 0), 4096);
}

TEST(LasIOTest, WriteLasUsesPointFormatMatchingRecordLength)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::GeometryCloud<float> cloud(1);
    cloud.points().operator()(0, 0) = 1.0f;
    cloud.points().operator()(0, 1) = 2.0f;
    cloud.points().operator()(0, 2) = 3.0f;

    plapoint::io::writeLas(path, cloud, 0.01);

    const plapoint::io::LasHeader header = readLasHeader(path);

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
    header.header_size = plapoint::io::kLasHeaderSize;
    header.point_data_offset = plapoint::io::kLasHeaderSize;
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
        writeLasHeader(out, header);

        const auto first = makeLasFormat2Record(1, 2, 3, 10, 20, 30);
        const auto second = makeLasFormat2Record(4, 5, 6, 200, 210, 220);
        out.write(first.data(), static_cast<std::streamsize>(first.size()));
        out.write(second.data(), static_cast<std::streamsize>(second.size()));
    }

    auto cloud = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    ASSERT_EQ(cloud->size(), 2u);
    ASSERT_TRUE(cloud->hasColors());
    EXPECT_EQ(cloud->colors()->operator()(0, 0), 10);
    EXPECT_EQ(cloud->colors()->operator()(0, 1), 20);
    EXPECT_EQ(cloud->colors()->operator()(0, 2), 30);
    EXPECT_EQ(cloud->colors()->operator()(1, 0), 200);
    EXPECT_EQ(cloud->colors()->operator()(1, 1), 210);
    EXPECT_EQ(cloud->colors()->operator()(1, 2), 220);
}

TEST(LasIOTest, WriteLasWithColorsUsesFormat2AndRoundtripsColors)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    using Cloud = plapoint::GeometryCloud<float>;
    using ColorMatrix = plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>;

    Cloud cloud(2);
    cloud.points().operator()(0, 0) = 1.0f;
    cloud.points().operator()(0, 1) = 2.0f;
    cloud.points().operator()(0, 2) = 3.0f;
    cloud.points().operator()(1, 0) = 4.0f;
    cloud.points().operator()(1, 1) = 5.0f;
    cloud.points().operator()(1, 2) = 6.0f;

    ColorMatrix colors(2, 3);
    colors.operator()(0, 0) = 11;
    colors.operator()(0, 1) = 22;
    colors.operator()(0, 2) = 33;
    colors.operator()(1, 0) = 201;
    colors.operator()(1, 1) = 211;
    colors.operator()(1, 2) = 221;
    cloud.setColors(std::move(colors));

    plapoint::io::writeLas(path, cloud, 0.01);

    const plapoint::io::LasHeader header = readLasHeader(path);
    auto loaded = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    EXPECT_EQ(header.point_data_format, 2);
    EXPECT_EQ(header.point_data_record_length, 26);
    ASSERT_TRUE(loaded->hasColors());
    EXPECT_EQ(loaded->colors()->operator()(0, 0), 11);
    EXPECT_EQ(loaded->colors()->operator()(0, 1), 22);
    EXPECT_EQ(loaded->colors()->operator()(0, 2), 33);
    EXPECT_EQ(loaded->colors()->operator()(1, 0), 201);
    EXPECT_EQ(loaded->colors()->operator()(1, 1), 211);
    EXPECT_EQ(loaded->colors()->operator()(1, 2), 221);
}

TEST(LasIOTest, WriteLasRoundtripsIntensityValues)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    using Cloud = plapoint::GeometryCloud<float>;
    using IntensityMatrix = plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>;

    Cloud cloud(2);
    cloud.points().operator()(0, 0) = 1.0f;
    cloud.points().operator()(0, 1) = 2.0f;
    cloud.points().operator()(0, 2) = 3.0f;
    cloud.points().operator()(1, 0) = 4.0f;
    cloud.points().operator()(1, 1) = 5.0f;
    cloud.points().operator()(1, 2) = 6.0f;

    IntensityMatrix intensities(2, 1);
    intensities.operator()(0, 0) = 17;
    intensities.operator()(1, 0) = 4096;
    cloud.setIntensities(std::move(intensities));

    plapoint::io::writeLas(path, cloud, 0.01);

    const plapoint::io::LasHeader header = readLasHeader(path);
    auto loaded = plapoint::io::readLas<float>(path);

    std::filesystem::remove(path);

    EXPECT_EQ(header.point_data_format, 0);
    EXPECT_EQ(header.point_data_record_length, 20);
    ASSERT_TRUE(loaded->hasIntensities());
    EXPECT_EQ(loaded->intensities()->operator()(0, 0), 17);
    EXPECT_EQ(loaded->intensities()->operator()(1, 0), 4096);
}

TEST(LasIOTest, WriteLasRejectsNonFiniteCoordinates)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::GeometryCloud<double> cloud(2);
    cloud.points().operator()(0, 0) = 0.0;
    cloud.points().operator()(0, 1) = 1.0;
    cloud.points().operator()(0, 2) = 2.0;
    cloud.points().operator()(1, 0) = std::numeric_limits<double>::quiet_NaN();
    cloud.points().operator()(1, 1) = 3.0;
    cloud.points().operator()(1, 2) = 4.0;

    EXPECT_THROW(plapoint::io::writeLas(path, cloud, 0.001), std::invalid_argument);

    std::filesystem::remove(path);
}

TEST(LasIOTest, WriteLasRejectsQuantizedCoordinatesOutsideInt32Range)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::GeometryCloud<double> cloud(2);
    cloud.points().operator()(0, 0) = 0.0;
    cloud.points().operator()(0, 1) = 0.0;
    cloud.points().operator()(0, 2) = 0.0;
    cloud.points().operator()(1, 0) = 3000000.0;
    cloud.points().operator()(1, 1) = 0.0;
    cloud.points().operator()(1, 2) = 0.0;

    EXPECT_THROW(plapoint::io::writeLas(path, cloud, 0.001), std::out_of_range);

    std::filesystem::remove(path);
}

TEST(LasIOTest, HeaderEncodingUsesFixedLittleEndianWireOffsets)
{
    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.file_source_id = 0x1234u;
    header.project_id_1 = 0x89abcdefu;
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = plapoint::io::kLasHeaderSize;
    header.point_data_offset = 0x01020304u;
    header.point_data_record_length = 0x0506u;
    header.num_point_records = 0x0708090au;
    header.x_scale_factor = 0.25;

    const auto bytes = plapoint::io::detail::encodeLasHeader(header);

    EXPECT_EQ(bytes.size(), 227u);
    EXPECT_EQ(std::string(bytes.data(), 4), "LASF");
    EXPECT_EQ(static_cast<unsigned char>(bytes[4]), 0x34u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[5]), 0x12u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[8]), 0xefu);
    EXPECT_EQ(static_cast<unsigned char>(bytes[11]), 0x89u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[96]), 0x04u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[99]), 0x01u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[105]), 0x06u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[106]), 0x05u);
    EXPECT_EQ(static_cast<unsigned char>(bytes[107]), 0x0au);
    EXPECT_EQ(static_cast<unsigned char>(bytes[110]), 0x07u);
    EXPECT_DOUBLE_EQ(plapoint::io::detail::readLeDouble(bytes.data(), 131), 0.25);
}

TEST(LasIOTest, RejectsTruncatedHugePointCountBeforeAllocation)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = plapoint::io::kLasHeaderSize;
    header.point_data_offset = plapoint::io::kLasHeaderSize;
    header.point_data_format = 0;
    header.point_data_record_length = plapoint::io::kLasPointFormat0Size;
    header.num_point_records = std::numeric_limits<std::uint32_t>::max();
    header.x_scale_factor = 1.0;
    header.y_scale_factor = 1.0;
    header.z_scale_factor = 1.0;
    {
        std::ofstream output(path, std::ios::binary);
        ASSERT_TRUE(output);
        writeLasHeader(output, header);
    }

    EXPECT_THROW((void)plapoint::io::readLas<float>(path), std::runtime_error);
}

TEST(LasIOTest, RejectsInvalidVersionOffsetAndScale)
{
    const plapoint::test::TempFile temp_file(".las");
    const auto path = temp_file.string();

    plapoint::io::LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    header.header_size = plapoint::io::kLasHeaderSize;
    header.point_data_offset = plapoint::io::kLasHeaderSize;
    header.point_data_format = 0;
    header.point_data_record_length = plapoint::io::kLasPointFormat0Size;
    header.x_scale_factor = 1.0;
    header.y_scale_factor = 1.0;
    header.z_scale_factor = 1.0;

    const auto write_and_reject = [&](const plapoint::io::LasHeader& invalid_header)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(output);
        writeLasHeader(output, invalid_header);
        output.close();
        EXPECT_THROW((void)plapoint::io::readLas<float>(path), std::runtime_error);
    };

    auto invalid_header = header;
    invalid_header.version_minor = 4;
    write_and_reject(invalid_header);

    invalid_header = header;
    invalid_header.point_data_offset = plapoint::io::kLasHeaderSize - 1;
    write_and_reject(invalid_header);

    invalid_header = header;
    invalid_header.x_scale_factor = 0.0;
    write_and_reject(invalid_header);

    invalid_header = header;
    invalid_header.min_x = 1.0;
    invalid_header.max_x = 0.0;
    write_and_reject(invalid_header);
}

TEST(LasIOTest, WriteReportsLateDeviceFailures)
{
    if (!std::filesystem::exists("/dev/full"))
    {
        GTEST_SKIP() << "/dev/full is unavailable on this platform";
    }

    plapoint::GeometryCloud<float> cloud(1);
    EXPECT_THROW(plapoint::io::writeLas("/dev/full", cloud), std::runtime_error);
}
