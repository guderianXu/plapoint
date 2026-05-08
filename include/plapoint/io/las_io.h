#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <cstring>
#include <fstream>
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
#pragma pack(pop)

struct LasPoint
{
    int32_t x, y, z;
    uint16_t intensity;
    uint8_t  return_number : 3;
    uint8_t  number_of_returns : 3;
    uint8_t  scan_direction_flag : 1;
    uint8_t  edge_of_flight_line : 1;
    uint8_t  classification;
    int8_t   scan_angle_rank;
    uint8_t  user_data;
    uint16_t point_source_id;
};

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readLas(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open LAS file: " + path);

    LasHeader hdr{};
    f.read(reinterpret_cast<char*>(&hdr), sizeof(LasHeader));

    if (std::strncmp(hdr.file_signature, "LASF", 4) != 0)
        throw std::runtime_error("Not a LAS file: " + path);

    // Skip VLRs
    f.seekg(hdr.point_data_offset);

    int n = static_cast<int>(hdr.num_point_records);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(n, 3);

    for (int i = 0; i < n; ++i)
    {
        LasPoint lp{};
        f.read(reinterpret_cast<char*>(&lp), sizeof(LasPoint));

        Scalar x = static_cast<Scalar>(lp.x * hdr.x_scale_factor + hdr.x_offset);
        Scalar y = static_cast<Scalar>(lp.y * hdr.y_scale_factor + hdr.y_offset);
        Scalar z = static_cast<Scalar>(lp.z * hdr.z_scale_factor + hdr.z_offset);
        pts(i, 0) = x;
        pts(i, 1) = y;
        pts(i, 2) = z;
    }

    return std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
}

template <typename Scalar>
void writeLas(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
              double scale = 0.001)
{
    int n = static_cast<int>(cloud.size());

    // Compute bounds
    double min_x = 1e100, max_x = -1e100, min_y = 1e100, max_y = -1e100, min_z = 1e100, max_z = -1e100;
    for (int i = 0; i < n; ++i)
    {
        double x = cloud.points().getValue(i, 0);
        double y = cloud.points().getValue(i, 1);
        double z = cloud.points().getValue(i, 2);
        if (x < min_x) min_x = x; if (x > max_x) max_x = x;
        if (y < min_y) min_y = y; if (y > max_y) max_y = y;
        if (z < min_z) min_z = z; if (z > max_z) max_z = z;
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
    hdr.point_data_format = 1;  // format 1: xyz + intensity + return + classification
    hdr.point_data_record_length = sizeof(LasPoint);
    hdr.num_point_records = static_cast<uint32_t>(n);

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
        LasPoint pt{};
        double x = cloud.points().getValue(i, 0);
        double y = cloud.points().getValue(i, 1);
        double z = cloud.points().getValue(i, 2);
        pt.x = static_cast<int32_t>(std::round((x - offset_x) / scale));
        pt.y = static_cast<int32_t>(std::round((y - offset_y) / scale));
        pt.z = static_cast<int32_t>(std::round((z - offset_z) / scale));
        pt.intensity = 255;
        pt.classification = 1;  // unclassified
        f.write(reinterpret_cast<const char*>(&pt), sizeof(LasPoint));
    }
}

} // namespace io
} // namespace plapoint
