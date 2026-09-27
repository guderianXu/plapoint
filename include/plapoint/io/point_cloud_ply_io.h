#pragma once

#include <cstdint>
#include <exception>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>

#include <plapoint/io/detail/ply_cloud_blob.h>

namespace plapoint
{

    class PLYReader
    {
    public:
        enum
        {
            PLY_V0 = 0,
            PLY_V1 = 1
        };

        int readHeader(const std::string& file_name,
                       PCLPointCloud2& cloud,
                       Eigen::Vector4f& origin,
                       Eigen::Quaternionf& orientation,
                       int& ply_version,
                       int& data_type,
                       unsigned int& data_idx,
                       const int offset = 0)
        {
            try
            {
                auto document = io::detail::readPlyBlob(file_name, offset);
                cloud = std::move(document.cloud);
                origin = document.origin;
                orientation = document.orientation;
                ply_version = PLY_V1;
                data_type = 0;
                data_idx = 0;
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PLYReader::readHeader: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

        int read(const std::string& file_name,
                 PCLPointCloud2& cloud,
                 Eigen::Vector4f& origin,
                 Eigen::Quaternionf& orientation,
                 int& ply_version,
                 const int offset = 0)
        {
            try
            {
                auto document = io::detail::readPlyBlob(file_name, offset);
                cloud = std::move(document.cloud);
                origin = document.origin;
                orientation = document.orientation;
                ply_version = PLY_V1;
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PLYReader::read: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

        int read(const std::string& file_name, PCLPointCloud2& cloud, const int offset = 0)
        {
            Eigen::Vector4f origin;
            Eigen::Quaternionf orientation;
            int version = 0;
            return read(file_name, cloud, origin, orientation, version, offset);
        }

        template <typename PointT>
        int read(const std::string& file_name, PointCloud<PointT>& cloud, const int offset = 0)
        {
            PCLPointCloud2 blob;
            Eigen::Vector4f origin;
            Eigen::Quaternionf orientation;
            int version = 0;
            const int result = read(file_name, blob, origin, orientation, version, offset);
            if (result == 0)
            {
                fromPCLPointCloud2(blob, cloud);
                cloud.sensor_origin_ = origin;
                cloud.sensor_orientation_ = orientation;
            }
            return result;
        }

        int read(const std::string& file_name,
                 PolygonMesh& mesh,
                 Eigen::Vector4f& origin,
                 Eigen::Quaternionf& orientation,
                 int& ply_version,
                 const int offset = 0)
        {
            try
            {
                auto document = io::detail::readPlyBlob(file_name, offset);
                mesh.header = document.cloud.header;
                mesh.cloud = std::move(document.cloud);
                mesh.polygons = std::move(document.polygons);
                origin = document.origin;
                orientation = document.orientation;
                ply_version = PLY_V1;
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PLYReader::read: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

        int read(const std::string& file_name, PolygonMesh& mesh, const int offset = 0)
        {
            Eigen::Vector4f origin;
            Eigen::Quaternionf orientation;
            int version = 0;
            return read(file_name, mesh, origin, orientation, version, offset);
        }
    };

    class PLYWriter
    {
    public:
        std::string generateHeaderBinary(const PCLPointCloud2& cloud,
                                         const Eigen::Vector4f& origin,
                                         const Eigen::Quaternionf& orientation,
                                         int valid_points,
                                         bool use_camera = true)
        {
            return generateHeader(cloud, origin, orientation, true, use_camera, valid_points);
        }

        std::string generateHeaderASCII(const PCLPointCloud2& cloud,
                                        const Eigen::Vector4f& origin,
                                        const Eigen::Quaternionf& orientation,
                                        int valid_points,
                                        bool use_camera = true)
        {
            return generateHeader(cloud, origin, orientation, false, use_camera, valid_points);
        }

        int writeASCII(const std::string& file_name,
                       const PCLPointCloud2& cloud,
                       const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                       const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                       int precision = 8,
                       bool use_camera = true)
        {
            return writeImpl(file_name,
                             cloud,
                             {},
                             origin,
                             orientation,
                             false,
                             use_camera,
                             static_cast<unsigned int>(std::max(precision, 1)));
        }

        int writeBinary(const std::string& file_name,
                        const PCLPointCloud2& cloud,
                        const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                        const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                        bool use_camera = true)
        {
            return writeImpl(file_name, cloud, {}, origin, orientation, true, use_camera, 8);
        }

        int writeBinary(std::ostream& stream,
                        const PCLPointCloud2& cloud,
                        const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                        const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                        bool use_camera = true)
        {
            try
            {
                io::detail::writePlyBlob(stream, cloud, {}, origin, orientation, true, use_camera, 8);
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PLYWriter::writeBinary: " << error.what() << '\n';
                return -1;
            }
        }

        int write(const std::string& file_name,
                  const PCLPointCloud2& cloud,
                  const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                  const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                  bool binary = false)
        {
            return write(file_name, cloud, origin, orientation, binary, true);
        }

        int write(const std::string& file_name,
                  const PCLPointCloud2& cloud,
                  const Eigen::Vector4f& origin,
                  const Eigen::Quaternionf& orientation,
                  bool binary,
                  bool use_camera)
        {
            return binary ? writeBinary(file_name, cloud, origin, orientation, use_camera)
                          : writeASCII(file_name, cloud, origin, orientation, 8, use_camera);
        }

        int write(const std::string& file_name,
                  const PCLPointCloud2::ConstPtr& cloud,
                  const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                  const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                  bool binary = false,
                  bool use_camera = true)
        {
            return cloud ? write(file_name, *cloud, origin, orientation, binary, use_camera) : -1;
        }

        template <typename PointT>
        int write(const std::string& file_name,
                  const PointCloud<PointT>& cloud,
                  bool binary = false,
                  bool use_camera = true)
        {
            PCLPointCloud2 blob;
            toPCLPointCloud2(cloud, blob);
            return write(file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_, binary, use_camera);
        }

    private:
        static std::string generateHeader(const PCLPointCloud2& cloud,
                                          const Eigen::Vector4f& origin,
                                          const Eigen::Quaternionf&,
                                          bool binary,
                                          bool use_camera,
                                          int valid_points)
        {
            std::ostringstream output;
            output.imbue(std::locale::classic());
            output << "ply\nformat ";
            if (!binary)
            {
                output << "ascii";
            }
            else
            {
                output << (cloud.is_bigendian ? "binary_big_endian" : "binary_little_endian");
            }
            output << " 1.0\ncomment PlaPoint generated\n";
            if (!use_camera)
            {
                output << "obj_info is_cyberware_data 0\n"
                          "obj_info is_mesh 0\n"
                          "obj_info is_warped 0\n"
                          "obj_info is_interlaced 0\n"
                       << "obj_info num_cols " << cloud.width << '\n'
                       << "obj_info num_rows " << cloud.height << '\n'
                       << "obj_info echo_rgb_offset_x " << origin[0] << '\n'
                       << "obj_info echo_rgb_offset_y " << origin[1] << '\n'
                       << "obj_info echo_rgb_offset_z " << origin[2] << '\n'
                       << "obj_info echo_rgb_frontfocus 0.0\n"
                          "obj_info echo_rgb_backfocus 0.0\n"
                          "obj_info echo_rgb_pixelsize 0.0\n"
                          "obj_info echo_rgb_centerpixel 0\n"
                          "obj_info echo_frames 1\n"
                          "obj_info echo_lgincr 0.0\n";
            }
            output << "element vertex " << valid_points << '\n';
            for (const auto& field : io::detail::validPcdFields(cloud))
            {
                if (field.name == "rgb" || field.name == "rgba")
                {
                    output << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
                    if (field.name == "rgba")
                    {
                        output << "property uchar alpha\n";
                    }
                }
                else if (field.count == 1)
                {
                    output << "property " << io::detail::plyBlobTypeName(field.datatype) << ' '
                           << io::detail::plyOutputName(field.name) << '\n';
                }
                else
                {
                    output << "property list uint " << io::detail::plyBlobTypeName(field.datatype) << ' '
                           << io::detail::plyOutputName(field.name) << '\n';
                }
            }
            output << "element face 0\n";
            if (use_camera)
            {
                io::detail::appendPlyCameraHeader(output);
            }
            else if (cloud.height > 1)
            {
                output << "element range_grid " << static_cast<std::uint64_t>(cloud.width) * cloud.height
                       << "\nproperty list uchar int vertex_indices\n";
            }
            output << "end_header\n";
            return output.str();
        }

        static int writeImpl(const std::string& file_name,
                             const PCLPointCloud2& cloud,
                             const std::vector<Vertices>& polygons,
                             const Eigen::Vector4f& origin,
                             const Eigen::Quaternionf& orientation,
                             bool binary,
                             bool use_camera,
                             unsigned int precision)
        {
            try
            {
                io::detail::writePlyBlob(
                    file_name, cloud, polygons, origin, orientation, binary, use_camera, precision);
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PLYWriter::write: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }
    };

    namespace io
    {

        inline int loadPLYFile(const std::string& file_name, PCLPointCloud2& cloud)
        {
            return PLYReader().read(file_name, cloud);
        }

        inline int loadPLYFile(const std::string& file_name,
                               PCLPointCloud2& cloud,
                               Eigen::Vector4f& origin,
                               Eigen::Quaternionf& orientation)
        {
            int version = 0;
            return PLYReader().read(file_name, cloud, origin, orientation, version);
        }

        template <typename PointT> int loadPLYFile(const std::string& file_name, PointCloud<PointT>& cloud)
        {
            return PLYReader().read(file_name, cloud);
        }

        inline int loadPLYFile(const std::string& file_name, PolygonMesh& mesh)
        {
            return PLYReader().read(file_name, mesh);
        }

        inline int savePLYFile(const std::string& file_name,
                               const PCLPointCloud2& cloud,
                               const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                               const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                               bool binary_mode = false,
                               bool use_camera = true)
        {
            return PLYWriter().write(file_name, cloud, origin, orientation, binary_mode, use_camera);
        }

        template <typename PointT>
        int savePLYFile(const std::string& file_name, const PointCloud<PointT>& cloud, bool binary_mode = false)
        {
            return PLYWriter().write(file_name, cloud, binary_mode);
        }

        template <typename PointT> int savePLYFileASCII(const std::string& file_name, const PointCloud<PointT>& cloud)
        {
            return PLYWriter().write(file_name, cloud, false);
        }

        template <typename PointT> int savePLYFileBinary(const std::string& file_name, const PointCloud<PointT>& cloud)
        {
            return PLYWriter().write(file_name, cloud, true);
        }

        template <typename PointT>
        int savePLYFile(const std::string& file_name,
                        const PointCloud<PointT>& cloud,
                        const Indices& indices,
                        bool binary_mode = false)
        {
            try
            {
                return savePLYFile(file_name, PointCloud<PointT>(cloud, indices), binary_mode);
            }
            catch (const std::exception& error)
            {
                std::cerr << "savePLYFile: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

        inline int savePLYFile(const std::string& file_name, const PolygonMesh& mesh, unsigned int precision = 5)
        {
            try
            {
                io::detail::writePlyBlob(file_name,
                                         mesh.cloud,
                                         mesh.polygons,
                                         Eigen::Vector4f::Zero(),
                                         Eigen::Quaternionf::Identity(),
                                         false,
                                         false,
                                         precision);
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "savePLYFile: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

        inline int savePLYFileBinary(const std::string& file_name, const PolygonMesh& mesh)
        {
            try
            {
                io::detail::writePlyBlob(file_name,
                                         mesh.cloud,
                                         mesh.polygons,
                                         Eigen::Vector4f::Zero(),
                                         Eigen::Quaternionf::Identity(),
                                         true,
                                         false,
                                         8);
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "savePLYFileBinary: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

    } // namespace io
} // namespace plapoint
