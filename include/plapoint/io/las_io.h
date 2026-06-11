#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace plapoint {
namespace io {

// LAS 1.2 Public Header Block (227 bytes)
#pragma pack(push, 1)
struct LasHeader
{
    char     file_signature[4];    // "LASF"
    uint16_t file_source_id;
    uint16_t global_encoding;
    uint32_t project_id_1;
    uint16_t project_id_2;
    uint16_t project_id_3;
    char     project_id_4[8];
    uint8_t  version_major;
    uint8_t  version_minor;
    char     system_identifier[32];
    char     generating_software[32];
    uint16_t creation_day;
    uint16_t creation_year;
    uint16_t header_size;
    uint32_t point_data_offset;
    uint32_t num_variable_length_records;
    uint8_t  point_data_format;
    uint16_t point_data_record_length;
    uint32_t num_point_records;
    uint32_t num_points_by_return[5];
    double   x_scale_factor;
    double   y_scale_factor;
    double   z_scale_factor;
    double   x_offset;
    double   y_offset;
    double   z_offset;
    double   max_x;
    double   min_x;
    double   max_y;
    double   min_y;
    double   max_z;
    double   min_z;
};

struct LasPointFormat0
{
    int32_t x, y, z;
    uint16_t intensity;
    uint8_t  return_flags;
    uint8_t  classification;
    int8_t   scan_angle_rank;
    uint8_t  user_data;
    uint16_t point_source_id;
};

struct LasPointFormat2
{
    LasPointFormat0 base;
    uint16_t red;
    uint16_t green;
    uint16_t blue;
};
#pragma pack(pop)

static_assert(sizeof(LasHeader) == 227, "LAS 1.2 public header must be 227 bytes");
static_assert(sizeof(LasPointFormat0) == 20, "LAS point format 0 record must be 20 bytes");
static_assert(sizeof(LasPointFormat2) == 26, "LAS point format 2 record must be 26 bytes");

using LasPoint = LasPointFormat0;

namespace detail {

inline void requireFiniteLasCoordinate(double value)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument("LAS coordinates must be finite");
    }
}

inline int32_t quantizeLasCoordinate(double coordinate, double offset, double scale)
{
    requireFiniteLasCoordinate(coordinate);
    const double quantized = std::round((coordinate - offset) / scale);
    if (quantized < static_cast<double>(std::numeric_limits<int32_t>::min()) ||
        quantized > static_cast<double>(std::numeric_limits<int32_t>::max()))
    {
        throw std::out_of_range("LAS quantized coordinate is outside int32 range");
    }
    return static_cast<int32_t>(quantized);
}

inline int lasRgbOffset(std::uint8_t pointDataFormat)
{
    if (pointDataFormat == 2) return 20;
    if (pointDataFormat == 3 || pointDataFormat == 5) return 28;
    return -1;
}

inline bool isSupportedLasPointDataFormat(std::uint8_t pointDataFormat)
{
    return pointDataFormat == 0 || pointDataFormat == 1 ||
           pointDataFormat == 2 || pointDataFormat == 3 ||
           pointDataFormat == 5;
}

inline std::uint16_t minLasPointRecordLength(std::uint8_t pointDataFormat)
{
    switch (pointDataFormat)
    {
    case 0: return 20;
    case 1: return 28;
    case 2: return 26;
    case 3: return 34;
    case 5: return 63;
    default: return 0;
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

} // namespace detail

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readLas(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open LAS file: " + path);

    LasHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(LasHeader));
    if (!f) throw std::runtime_error("Cannot read LAS header: " + path);

    if (std::strncmp(hdr.file_signature, "LASF", 4) != 0)
        throw std::runtime_error("Not a LAS file: " + path);
    if (!detail::isSupportedLasPointDataFormat(hdr.point_data_format))
        throw std::runtime_error("Unsupported LAS point data format in: " + path);
    if (hdr.point_data_record_length < detail::minLasPointRecordLength(hdr.point_data_format))
        throw std::runtime_error("Unsupported LAS point record length in: " + path);

    // Skip VLRs
    f.seekg(static_cast<std::streamoff>(hdr.point_data_offset), std::ios::beg);
    if (!f) throw std::runtime_error("Cannot seek to LAS point data: " + path);

    int n = static_cast<int>(hdr.num_point_records);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(n, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(n, 1);
    const int rgb_offset = detail::lasRgbOffset(hdr.point_data_format);
    if (rgb_offset >= 0 &&
        hdr.point_data_record_length < static_cast<std::uint16_t>(rgb_offset + 3 * sizeof(std::uint16_t)))
    {
        throw std::runtime_error("Unsupported LAS RGB point record length in: " + path);
    }
    const bool have_colors = rgb_offset >= 0;
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(n, 3);

    std::vector<char> record(hdr.point_data_record_length);
    for (int i = 0; i < n; ++i)
    {
        f.read(record.data(), static_cast<std::streamsize>(record.size()));
        if (!f) throw std::runtime_error("Truncated LAS point record in: " + path);

        LasPointFormat0 lp{};
        std::memcpy(&lp, record.data(), sizeof(lp));

        Scalar x = static_cast<Scalar>(lp.x * hdr.x_scale_factor + hdr.x_offset);
        Scalar y = static_cast<Scalar>(lp.y * hdr.y_scale_factor + hdr.y_offset);
        Scalar z = static_cast<Scalar>(lp.z * hdr.z_scale_factor + hdr.z_offset);
        pts(i, 0) = x;
        pts(i, 1) = y;
        pts(i, 2) = z;
        intensities(i, 0) = lp.intensity;

        if (have_colors)
        {
            std::uint16_t rgb[3] = {};
            std::memcpy(rgb, record.data() + rgb_offset, sizeof(rgb));
            colors(i, 0) = detail::lasColorByte(rgb[0]);
            colors(i, 1) = detail::lasColorByte(rgb[1]);
            colors(i, 2) = detail::lasColorByte(rgb[2]);
        }
    }

    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
    cloud->setIntensities(std::move(intensities));
    if (have_colors)
    {
        cloud->setColors(std::move(colors));
    }
    return cloud;
}

template <typename Scalar>
void writeLas(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
              double scale = 0.001)
{
    if (!std::isfinite(scale) || scale <= 0)
    {
        throw std::invalid_argument("LAS scale must be positive");
    }

    int n = static_cast<int>(cloud.size());
    const bool with_colors = cloud.hasColors();

    // Compute bounds
    double min_x = 1e100, max_x = -1e100, min_y = 1e100, max_y = -1e100, min_z = 1e100, max_z = -1e100;
    for (int i = 0; i < n; ++i)
    {
        double x = cloud.points().getValue(i, 0);
        double y = cloud.points().getValue(i, 1);
        double z = cloud.points().getValue(i, 2);
        detail::requireFiniteLasCoordinate(x);
        detail::requireFiniteLasCoordinate(y);
        detail::requireFiniteLasCoordinate(z);
        if (x < min_x) min_x = x; if (x > max_x) max_x = x;
        if (y < min_y) min_y = y; if (y > max_y) max_y = y;
        if (z < min_z) min_z = z; if (z > max_z) max_z = z;
    }
    if (n == 0)
    {
        min_x = max_x = min_y = max_y = min_z = max_z = 0.0;
    }

    double offset_x = min_x, offset_y = min_y, offset_z = min_z;

    LasHeader hdr{};
    std::memcpy(hdr.file_signature, "LASF", 4);
    hdr.version_major = 1;
    hdr.version_minor = 2;
    std::strncpy(hdr.system_identifier, "PlaPoint", 31);
    std::strncpy(hdr.generating_software, "PlaPoint", 31);
    hdr.header_size = sizeof(LasHeader);
    hdr.point_data_offset = sizeof(LasHeader);
    hdr.point_data_format = with_colors ? 2 : 0;
    hdr.point_data_record_length = with_colors ? sizeof(LasPointFormat2) : sizeof(LasPointFormat0);
    hdr.num_point_records = static_cast<uint32_t>(n);
    hdr.num_points_by_return[0] = static_cast<uint32_t>(n);

    hdr.x_scale_factor = scale;
    hdr.y_scale_factor = scale;
    hdr.z_scale_factor = scale;
    hdr.x_offset = offset_x;
    hdr.y_offset = offset_y;
    hdr.z_offset = offset_z;
    hdr.max_x = max_x; hdr.min_x = min_x;
    hdr.max_y = max_y; hdr.min_y = min_y;
    hdr.max_z = max_z; hdr.min_z = min_z;

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write LAS file: " + path);

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(LasHeader));

    for (int i = 0; i < n; ++i)
    {
        LasPointFormat0 pt{};
        const auto* intensities = cloud.intensities();
        double x = cloud.points().getValue(i, 0);
        double y = cloud.points().getValue(i, 1);
        double z = cloud.points().getValue(i, 2);
        pt.x = detail::quantizeLasCoordinate(x, offset_x, scale);
        pt.y = detail::quantizeLasCoordinate(y, offset_y, scale);
        pt.z = detail::quantizeLasCoordinate(z, offset_z, scale);
        pt.intensity = intensities ? intensities->getValue(i, 0) : 255;
        pt.return_flags = 1;
        pt.classification = 1;  // unclassified
        f.write(reinterpret_cast<const char*>(&pt), sizeof(LasPointFormat0));
        if (with_colors)
        {
            auto* c = cloud.colors();
            const std::uint16_t rgb[3] = {
                detail::lasColorWord(c->getValue(i, 0)),
                detail::lasColorWord(c->getValue(i, 1)),
                detail::lasColorWord(c->getValue(i, 2)),
            };
            f.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
        }
    }
}

} // namespace io
} // namespace plapoint
