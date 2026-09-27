#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/geometry_cloud.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/core/device.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace plapoint {
namespace io {

inline constexpr std::size_t kLasHeaderSize = 227;
inline constexpr std::size_t kLasPointFormat0Size = 20;
inline constexpr std::size_t kLasPointFormat2Size = 26;

/// Logical representation of a LAS 1.0--1.3 public header block.
/// Serialization is handled explicitly so this type's native layout is irrelevant.
struct LasHeader
{
    char file_signature[4] = {};
    std::uint16_t file_source_id = 0;
    std::uint16_t global_encoding = 0;
    std::uint32_t project_id_1 = 0;
    std::uint16_t project_id_2 = 0;
    std::uint16_t project_id_3 = 0;
    char project_id_4[8] = {};
    std::uint8_t version_major = 0;
    std::uint8_t version_minor = 0;
    char system_identifier[32] = {};
    char generating_software[32] = {};
    std::uint16_t creation_day = 0;
    std::uint16_t creation_year = 0;
    std::uint16_t header_size = 0;
    std::uint32_t point_data_offset = 0;
    std::uint32_t num_variable_length_records = 0;
    std::uint8_t point_data_format = 0;
    std::uint16_t point_data_record_length = 0;
    std::uint32_t num_point_records = 0;
    std::uint32_t num_points_by_return[5] = {};
    double x_scale_factor = 0.0;
    double y_scale_factor = 0.0;
    double z_scale_factor = 0.0;
    double x_offset = 0.0;
    double y_offset = 0.0;
    double z_offset = 0.0;
    double max_x = 0.0;
    double min_x = 0.0;
    double max_y = 0.0;
    double min_y = 0.0;
    double max_z = 0.0;
    double min_z = 0.0;
};

struct LasPointFormat0
{
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;
    std::uint16_t intensity = 0;
    std::uint8_t return_flags = 0;
    std::uint8_t classification = 0;
    std::int8_t scan_angle_rank = 0;
    std::uint8_t user_data = 0;
    std::uint16_t point_source_id = 0;
};

struct LasPointFormat2
{
    LasPointFormat0 base;
    std::uint16_t red = 0;
    std::uint16_t green = 0;
    std::uint16_t blue = 0;
};

using LasPoint = LasPointFormat0;

namespace detail {

inline std::uint16_t readLeU16(const char* data, std::size_t offset)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(data + offset);
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8);
}

inline std::uint32_t readLeU32(const char* data, std::size_t offset)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(data + offset);
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

inline std::uint64_t readLeU64(const char* data, std::size_t offset)
{
    std::uint64_t value = 0;
    for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    {
        value |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(data[offset + byte]))
                 << (byte * 8);
    }
    return value;
}

inline std::int32_t readLeI32(const char* data, std::size_t offset)
{
    const std::uint32_t bits = readLeU32(data, offset);
    std::int32_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline double readLeDouble(const char* data, std::size_t offset)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    const std::uint64_t bits = readLeU64(data, offset);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void writeLeU16(char* data, std::size_t offset, std::uint16_t value)
{
    data[offset] = static_cast<char>(value & 0xffu);
    data[offset + 1] = static_cast<char>((value >> 8) & 0xffu);
}

inline void writeLeU32(char* data, std::size_t offset, std::uint32_t value)
{
    for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    {
        data[offset + byte] = static_cast<char>((value >> (byte * 8)) & 0xffu);
    }
}

inline void writeLeU64(char* data, std::size_t offset, std::uint64_t value)
{
    for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    {
        data[offset + byte] = static_cast<char>((value >> (byte * 8)) & 0xffu);
    }
}

inline void writeLeI32(char* data, std::size_t offset, std::int32_t value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    writeLeU32(data, offset, bits);
}

inline void writeLeDouble(char* data, std::size_t offset, double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    writeLeU64(data, offset, bits);
}

inline std::array<char, kLasHeaderSize> encodeLasHeader(const LasHeader& header)
{
    std::array<char, kLasHeaderSize> bytes{};
    std::memcpy(bytes.data(), header.file_signature, sizeof(header.file_signature));
    writeLeU16(bytes.data(), 4, header.file_source_id);
    writeLeU16(bytes.data(), 6, header.global_encoding);
    writeLeU32(bytes.data(), 8, header.project_id_1);
    writeLeU16(bytes.data(), 12, header.project_id_2);
    writeLeU16(bytes.data(), 14, header.project_id_3);
    std::memcpy(bytes.data() + 16, header.project_id_4, sizeof(header.project_id_4));
    bytes[24] = static_cast<char>(header.version_major);
    bytes[25] = static_cast<char>(header.version_minor);
    std::memcpy(bytes.data() + 26, header.system_identifier, sizeof(header.system_identifier));
    std::memcpy(bytes.data() + 58, header.generating_software, sizeof(header.generating_software));
    writeLeU16(bytes.data(), 90, header.creation_day);
    writeLeU16(bytes.data(), 92, header.creation_year);
    writeLeU16(bytes.data(), 94, header.header_size);
    writeLeU32(bytes.data(), 96, header.point_data_offset);
    writeLeU32(bytes.data(), 100, header.num_variable_length_records);
    bytes[104] = static_cast<char>(header.point_data_format);
    writeLeU16(bytes.data(), 105, header.point_data_record_length);
    writeLeU32(bytes.data(), 107, header.num_point_records);
    for (std::size_t return_number = 0; return_number < 5; ++return_number)
    {
        writeLeU32(bytes.data(), 111 + return_number * 4,
                   header.num_points_by_return[return_number]);
    }
    writeLeDouble(bytes.data(), 131, header.x_scale_factor);
    writeLeDouble(bytes.data(), 139, header.y_scale_factor);
    writeLeDouble(bytes.data(), 147, header.z_scale_factor);
    writeLeDouble(bytes.data(), 155, header.x_offset);
    writeLeDouble(bytes.data(), 163, header.y_offset);
    writeLeDouble(bytes.data(), 171, header.z_offset);
    writeLeDouble(bytes.data(), 179, header.max_x);
    writeLeDouble(bytes.data(), 187, header.min_x);
    writeLeDouble(bytes.data(), 195, header.max_y);
    writeLeDouble(bytes.data(), 203, header.min_y);
    writeLeDouble(bytes.data(), 211, header.max_z);
    writeLeDouble(bytes.data(), 219, header.min_z);
    return bytes;
}

inline LasHeader decodeLasHeader(const char* bytes)
{
    LasHeader header{};
    std::memcpy(header.file_signature, bytes, sizeof(header.file_signature));
    header.file_source_id = readLeU16(bytes, 4);
    header.global_encoding = readLeU16(bytes, 6);
    header.project_id_1 = readLeU32(bytes, 8);
    header.project_id_2 = readLeU16(bytes, 12);
    header.project_id_3 = readLeU16(bytes, 14);
    std::memcpy(header.project_id_4, bytes + 16, sizeof(header.project_id_4));
    header.version_major = static_cast<std::uint8_t>(bytes[24]);
    header.version_minor = static_cast<std::uint8_t>(bytes[25]);
    std::memcpy(header.system_identifier, bytes + 26, sizeof(header.system_identifier));
    std::memcpy(header.generating_software, bytes + 58, sizeof(header.generating_software));
    header.creation_day = readLeU16(bytes, 90);
    header.creation_year = readLeU16(bytes, 92);
    header.header_size = readLeU16(bytes, 94);
    header.point_data_offset = readLeU32(bytes, 96);
    header.num_variable_length_records = readLeU32(bytes, 100);
    header.point_data_format = static_cast<std::uint8_t>(bytes[104]);
    header.point_data_record_length = readLeU16(bytes, 105);
    header.num_point_records = readLeU32(bytes, 107);
    for (std::size_t return_number = 0; return_number < 5; ++return_number)
    {
        header.num_points_by_return[return_number] =
            readLeU32(bytes, 111 + return_number * 4);
    }
    header.x_scale_factor = readLeDouble(bytes, 131);
    header.y_scale_factor = readLeDouble(bytes, 139);
    header.z_scale_factor = readLeDouble(bytes, 147);
    header.x_offset = readLeDouble(bytes, 155);
    header.y_offset = readLeDouble(bytes, 163);
    header.z_offset = readLeDouble(bytes, 171);
    header.max_x = readLeDouble(bytes, 179);
    header.min_x = readLeDouble(bytes, 187);
    header.max_y = readLeDouble(bytes, 195);
    header.min_y = readLeDouble(bytes, 203);
    header.max_z = readLeDouble(bytes, 211);
    header.min_z = readLeDouble(bytes, 219);
    return header;
}

inline LasHeader decodeLasHeader(const std::array<char, kLasHeaderSize>& bytes)
{
    return decodeLasHeader(bytes.data());
}

inline void requireFiniteLasCoordinate(double value)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument("LAS coordinates must be finite");
    }
}

inline std::int32_t quantizeLasCoordinate(double coordinate, double offset, double scale)
{
    requireFiniteLasCoordinate(coordinate);
    const double quantized = std::round((coordinate - offset) / scale);
    if (!std::isfinite(quantized) ||
        quantized < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        quantized > static_cast<double>(std::numeric_limits<std::int32_t>::max()))
    {
        throw std::out_of_range("LAS quantized coordinate is outside int32 range");
    }
    return static_cast<std::int32_t>(quantized);
}

inline int lasRgbOffset(std::uint8_t point_data_format)
{
    if (point_data_format == 2)
    {
        return 20;
    }
    if (point_data_format == 3 || point_data_format == 5)
    {
        return 28;
    }
    return -1;
}

inline bool isSupportedLasPointDataFormat(std::uint8_t point_data_format)
{
    return point_data_format == 0 || point_data_format == 1 ||
           point_data_format == 2 || point_data_format == 3 ||
           point_data_format == 5;
}

inline std::uint16_t minLasPointRecordLength(std::uint8_t point_data_format)
{
    switch (point_data_format)
    {
    case 0:
        return 20;
    case 1:
        return 28;
    case 2:
        return 26;
    case 3:
        return 34;
    case 5:
        return 63;
    default:
        return 0;
    }
}

inline std::uint8_t lasColorByte(std::uint16_t value)
{
    return static_cast<std::uint8_t>((static_cast<unsigned int>(value) + 128u) / 257u);
}

inline std::uint16_t lasColorWord(std::uint8_t value)
{
    return static_cast<std::uint16_t>(value) * 257u;
}

inline void validateLasHeader(const LasHeader& header,
                              std::uint64_t file_size,
                              const std::string& path)
{
    const auto invalid = [&](const std::string& reason)
    {
        throw std::runtime_error("Invalid LAS header in " + path + ": " + reason);
    };

    if (std::memcmp(header.file_signature, "LASF", 4) != 0)
    {
        invalid("missing LASF signature");
    }
    if (header.version_major != 1 || header.version_minor > 3)
    {
        invalid("only LAS versions 1.0 through 1.3 are supported");
    }
    if (header.header_size < kLasHeaderSize)
    {
        invalid("header size is smaller than the LAS public header");
    }
    if (header.point_data_offset < header.header_size || header.point_data_offset > file_size)
    {
        invalid("point data offset is outside the file");
    }
    if (!isSupportedLasPointDataFormat(header.point_data_format))
    {
        invalid("unsupported point data format");
    }
    const std::uint16_t minimum_record_length =
        minLasPointRecordLength(header.point_data_format);
    if (header.point_data_record_length < minimum_record_length)
    {
        invalid("point record is shorter than its declared format");
    }
    const int rgb_offset = lasRgbOffset(header.point_data_format);
    if (rgb_offset >= 0 &&
        header.point_data_record_length < static_cast<std::uint16_t>(rgb_offset + 6))
    {
        invalid("RGB point record is too short");
    }
    if (!std::isfinite(header.x_scale_factor) || header.x_scale_factor <= 0.0 ||
        !std::isfinite(header.y_scale_factor) || header.y_scale_factor <= 0.0 ||
        !std::isfinite(header.z_scale_factor) || header.z_scale_factor <= 0.0)
    {
        invalid("coordinate scales must be finite and positive");
    }
    const std::array<double, 12> finite_values = {
        header.x_offset, header.y_offset, header.z_offset,
        header.max_x, header.min_x, header.max_y, header.min_y,
        header.max_z, header.min_z,
        header.x_scale_factor, header.y_scale_factor, header.z_scale_factor,
    };
    if (!std::all_of(finite_values.begin(), finite_values.end(),
                     [](double value) { return std::isfinite(value); }))
    {
        invalid("coordinate metadata must be finite");
    }
    if (header.max_x < header.min_x || header.max_y < header.min_y ||
        header.max_z < header.min_z)
    {
        invalid("coordinate bounds are inverted");
    }

    const std::uint64_t available_bytes = file_size - header.point_data_offset;
    const std::uint64_t available_records =
        available_bytes / header.point_data_record_length;
    if (header.num_point_records > available_records)
    {
        invalid("declared point count exceeds available point records");
    }
    if (header.num_point_records >
        static_cast<std::uint64_t>(std::numeric_limits<plamatrix::Index>::max()))
    {
        invalid("point count exceeds the matrix index range");
    }
}

} // namespace detail

template <typename Scalar>
std::shared_ptr<GeometryCloud<Scalar>>
readLas(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot open LAS file: " + path);
    }

    file.seekg(0, std::ios::end);
    const std::streampos end_position = file.tellg();
    if (end_position < std::streampos(0))
    {
        throw std::runtime_error("Cannot determine LAS file size: " + path);
    }
    const auto file_size = static_cast<std::uint64_t>(
        static_cast<std::streamoff>(end_position));
    if (file_size < kLasHeaderSize)
    {
        throw std::runtime_error("Truncated LAS header: " + path);
    }
    file.seekg(0, std::ios::beg);

    std::array<char, kLasHeaderSize> header_bytes{};
    file.read(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));
    if (!file)
    {
        throw std::runtime_error("Cannot read LAS header: " + path);
    }
    const LasHeader header = detail::decodeLasHeader(header_bytes);
    detail::validateLasHeader(header, file_size, path);

    file.seekg(static_cast<std::streamoff>(header.point_data_offset), std::ios::beg);
    if (!file)
    {
        throw std::runtime_error("Cannot seek to LAS point data: " + path);
    }

    const auto point_count = static_cast<plamatrix::Index>(header.num_point_records);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> points(point_count, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(point_count, 1);
    const int rgb_offset = detail::lasRgbOffset(header.point_data_format);
    const bool have_colors = rgb_offset >= 0;
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors;
    if (have_colors)
    {
        colors = plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>(point_count, 3);
    }

    std::vector<char> record(header.point_data_record_length);
    for (plamatrix::Index row = 0; row < point_count; ++row)
    {
        file.read(record.data(), static_cast<std::streamsize>(record.size()));
        if (!file)
        {
            throw std::runtime_error("Truncated LAS point record in: " + path);
        }

        const std::int32_t encoded_x = detail::readLeI32(record.data(), 0);
        const std::int32_t encoded_y = detail::readLeI32(record.data(), 4);
        const std::int32_t encoded_z = detail::readLeI32(record.data(), 8);
        const Scalar x = static_cast<Scalar>(
            encoded_x * header.x_scale_factor + header.x_offset);
        const Scalar y = static_cast<Scalar>(
            encoded_y * header.y_scale_factor + header.y_offset);
        const Scalar z = static_cast<Scalar>(
            encoded_z * header.z_scale_factor + header.z_offset);
        if (!std::isfinite(static_cast<double>(x)) ||
            !std::isfinite(static_cast<double>(y)) ||
            !std::isfinite(static_cast<double>(z)))
        {
            throw std::runtime_error("LAS coordinate is outside the requested scalar range: " + path);
        }
        points(row, 0) = x;
        points(row, 1) = y;
        points(row, 2) = z;
        intensities(row, 0) = detail::readLeU16(record.data(), 12);

        if (have_colors)
        {
            colors(row, 0) = detail::lasColorByte(
                detail::readLeU16(record.data(), static_cast<std::size_t>(rgb_offset)));
            colors(row, 1) = detail::lasColorByte(
                detail::readLeU16(record.data(), static_cast<std::size_t>(rgb_offset + 2)));
            colors(row, 2) = detail::lasColorByte(
                detail::readLeU16(record.data(), static_cast<std::size_t>(rgb_offset + 4)));
        }
    }

    auto cloud = std::make_shared<GeometryCloud<Scalar>>(
        std::move(points));
    cloud->setIntensities(std::move(intensities));
    if (have_colors)
    {
        cloud->setColors(std::move(colors));
    }
    return cloud;
}

template <typename Scalar>
void writeLas(const std::string& path,
              const GeometryCloud<Scalar>& cloud,
              double scale = 0.001)
{
    cloud.validate();
    if (!std::isfinite(scale) || scale <= 0.0)
    {
        throw std::invalid_argument("LAS scale must be finite and positive");
    }
    if (cloud.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("LAS 1.2 point count exceeds uint32 range");
    }

    const auto point_count = static_cast<plamatrix::Index>(cloud.size());
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    for (plamatrix::Index row = 0; row < point_count; ++row)
    {
        const double x = static_cast<double>(cloud.points().operator()(row, 0));
        const double y = static_cast<double>(cloud.points().operator()(row, 1));
        const double z = static_cast<double>(cloud.points().operator()(row, 2));
        detail::requireFiniteLasCoordinate(x);
        detail::requireFiniteLasCoordinate(y);
        detail::requireFiniteLasCoordinate(z);
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);
    }
    if (point_count == 0)
    {
        min_x = min_y = min_z = 0.0;
        max_x = max_y = max_z = 0.0;
    }

    std::vector<std::array<std::int32_t, 3>> quantized_points(
        static_cast<std::size_t>(point_count));
    for (plamatrix::Index row = 0; row < point_count; ++row)
    {
        quantized_points[static_cast<std::size_t>(row)] = {
            detail::quantizeLasCoordinate(
                static_cast<double>(cloud.points().operator()(row, 0)), min_x, scale),
            detail::quantizeLasCoordinate(
                static_cast<double>(cloud.points().operator()(row, 1)), min_y, scale),
            detail::quantizeLasCoordinate(
                static_cast<double>(cloud.points().operator()(row, 2)), min_z, scale),
        };
    }

    const bool with_colors = cloud.hasColors();
    LasHeader header{};
    std::memcpy(header.file_signature, "LASF", 4);
    header.version_major = 1;
    header.version_minor = 2;
    std::memcpy(header.system_identifier, "PlaPoint", 8);
    std::memcpy(header.generating_software, "PlaPoint", 8);
    header.header_size = static_cast<std::uint16_t>(kLasHeaderSize);
    header.point_data_offset = static_cast<std::uint32_t>(kLasHeaderSize);
    header.point_data_format = with_colors ? 2 : 0;
    header.point_data_record_length = static_cast<std::uint16_t>(
        with_colors ? kLasPointFormat2Size : kLasPointFormat0Size);
    header.num_point_records = static_cast<std::uint32_t>(point_count);
    header.num_points_by_return[0] = static_cast<std::uint32_t>(point_count);
    header.x_scale_factor = scale;
    header.y_scale_factor = scale;
    header.z_scale_factor = scale;
    header.x_offset = min_x;
    header.y_offset = min_y;
    header.z_offset = min_z;
    header.max_x = max_x;
    header.min_x = min_x;
    header.max_y = max_y;
    header.min_y = min_y;
    header.max_z = max_z;
    header.min_z = min_z;

    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot write LAS file: " + path);
    }
    const auto header_bytes = detail::encodeLasHeader(header);
    file.write(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));

    std::vector<char> record(header.point_data_record_length);
    for (plamatrix::Index row = 0; row < point_count; ++row)
    {
        std::fill(record.begin(), record.end(), 0);
        const auto& point = quantized_points[static_cast<std::size_t>(row)];
        detail::writeLeI32(record.data(), 0, point[0]);
        detail::writeLeI32(record.data(), 4, point[1]);
        detail::writeLeI32(record.data(), 8, point[2]);
        const auto* intensities = cloud.intensities();
        detail::writeLeU16(
            record.data(), 12,
            intensities ? intensities->operator()(row, 0) : static_cast<std::uint16_t>(255));
        record[14] = 1;
        record[15] = 1;
        if (with_colors)
        {
            const auto* colors = cloud.colors();
            detail::writeLeU16(record.data(), 20,
                               detail::lasColorWord(colors->operator()(row, 0)));
            detail::writeLeU16(record.data(), 22,
                               detail::lasColorWord(colors->operator()(row, 1)));
            detail::writeLeU16(record.data(), 24,
                               detail::lasColorWord(colors->operator()(row, 2)));
        }
        file.write(record.data(), static_cast<std::streamsize>(record.size()));
    }

    file.close();
    if (!file)
    {
        throw std::runtime_error("Failed to write LAS file: " + path);
    }
}

} // namespace io
} // namespace plapoint
