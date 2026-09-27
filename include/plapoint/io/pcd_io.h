#pragma once

#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include <plapoint/io/detail/pcd_format.h>

namespace plapoint
{

    class PCDReader
    {
    public:
        enum
        {
            PCD_V6 = 0,
            PCD_V7 = 1
        };

        int readHeader(std::istream& binary_stream,
                       PCLPointCloud2& cloud,
                       Eigen::Vector4f& origin,
                       Eigen::Quaternionf& orientation,
                       int& pcd_version,
                       int& data_type,
                       unsigned int& data_idx)
        {
            try
            {
                const auto header = io::detail::readPcdHeader(binary_stream);
                io::detail::configurePcdCloud(header, cloud);
                setHeaderOutputs(header, origin, orientation, pcd_version, data_type, data_idx);
                _lastHeader = header;
                return 0;
            }
            catch (const std::exception& error)
            {
                _lastHeader.reset();
                std::cerr << "PCDReader::readHeader: " << error.what() << '\n';
                return -1;
            }
        }

        int readHeader(const std::string& file_name,
                       PCLPointCloud2& cloud,
                       Eigen::Vector4f& origin,
                       Eigen::Quaternionf& orientation,
                       int& pcd_version,
                       int& data_type,
                       unsigned int& data_idx,
                       const int offset = 0)
        {
            try
            {
                std::ifstream stream(file_name, std::ios::binary);
                if (!stream || offset < 0)
                {
                    return -1;
                }
                stream.seekg(offset);
                return readHeader(stream, cloud, origin, orientation, pcd_version, data_type, data_idx);
            }
            catch (const std::exception& error)
            {
                std::cerr << "PCDReader::readHeader: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

        int readHeader(const std::string& file_name, PCLPointCloud2& cloud, const int offset = 0)
        {
            Eigen::Vector4f origin;
            Eigen::Quaternionf orientation;
            int version = 0;
            int data_type = 0;
            unsigned int data_index = 0;
            return readHeader(file_name, cloud, origin, orientation, version, data_type, data_index, offset);
        }

        int readBodyASCII(std::istream& stream, PCLPointCloud2& cloud, int)
        {
            try
            {
                if (_lastHeader && headerMatchesCloud(*_lastHeader, cloud))
                {
                    io::detail::readPcdAscii(stream, *_lastHeader, cloud);
                }
                else
                {
                    readPcdAsciiFromCloud(stream, cloud);
                }
                updateDenseFlag(cloud);
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PCDReader::readBodyASCII: " << error.what() << '\n';
                return -1;
            }
        }

        int
        readBodyBinary(const unsigned char* data, PCLPointCloud2& cloud, int, bool compressed, unsigned int data_idx)
        {
            if (!data)
            {
                return -1;
            }
            try
            {
                if (!compressed)
                {
                    if (!cloud.data.empty())
                    {
                        std::memcpy(cloud.data.data(), data + data_idx, cloud.data.size());
                    }
                }
                else
                {
                    const auto compressed_size = readLittleEndianU32(data + data_idx);
                    const auto uncompressed_size = readLittleEndianU32(data + data_idx + 4);
                    std::vector<std::uint8_t> compressed_data(compressed_size);
                    if (compressed_size != 0)
                    {
                        std::memcpy(compressed_data.data(), data + data_idx + 8, compressed_size);
                    }
                    const auto decoded = io::detail::lzfDecode(compressed_data, uncompressed_size);
                    io::detail::expandFieldMajorPcdData(decoded, cloud);
                }
                updateDenseFlag(cloud);
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PCDReader::readBodyBinary: " << error.what() << '\n';
                return -1;
            }
        }

        int read(const std::string& file_name,
                 PCLPointCloud2& cloud,
                 Eigen::Vector4f& origin,
                 Eigen::Quaternionf& orientation,
                 int& pcd_version,
                 const int offset = 0)
        {
            try
            {
                std::ifstream stream(file_name, std::ios::binary);
                if (!stream || offset < 0)
                {
                    return -1;
                }
                stream.seekg(offset);
                const auto header = io::detail::readPcdHeader(stream);
                io::detail::configurePcdCloud(header, cloud);
                origin = header.origin;
                orientation = header.orientation;
                pcd_version = header.version == 7 ? PCD_V7 : PCD_V6;
                if (header.mode == io::detail::PcdDataMode::Ascii)
                {
                    io::detail::readPcdAscii(stream, header, cloud);
                }
                else if (header.mode == io::detail::PcdDataMode::Binary)
                {
                    if (!stream.read(reinterpret_cast<char*>(cloud.data.data()),
                                     static_cast<std::streamsize>(cloud.data.size())))
                    {
                        throw std::runtime_error("binary point data is truncated");
                    }
                }
                else
                {
                    const std::uint32_t compressed_size = io::detail::readPcdU32(stream);
                    const std::uint32_t uncompressed_size = io::detail::readPcdU32(stream);
                    std::vector<std::uint8_t> compressed(compressed_size);
                    if (!stream.read(reinterpret_cast<char*>(compressed.data()), compressed.size()))
                    {
                        throw std::runtime_error("compressed point data is truncated");
                    }
                    const auto decoded = io::detail::lzfDecode(compressed, uncompressed_size);
                    io::detail::expandFieldMajorPcdData(decoded, cloud);
                }
                updateDenseFlag(cloud);
                return 0;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PCDReader::read: " << file_name << ": " << error.what() << '\n';
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

    private:
        static void setHeaderOutputs(const io::detail::PcdHeader& header,
                                     Eigen::Vector4f& origin,
                                     Eigen::Quaternionf& orientation,
                                     int& pcd_version,
                                     int& data_type,
                                     unsigned int& data_idx)
        {
            origin = header.origin;
            orientation = header.orientation;
            pcd_version = header.version == 7 ? PCD_V7 : PCD_V6;
            data_type = header.mode == io::detail::PcdDataMode::Ascii
                            ? 0
                            : (header.mode == io::detail::PcdDataMode::Binary ? 1 : 2);
            if (header.data_position < 0 ||
                static_cast<std::uint64_t>(header.data_position) > std::numeric_limits<unsigned int>::max())
            {
                throw std::overflow_error("PCD: data offset exceeds 32-bit range");
            }
            data_idx = static_cast<unsigned int>(header.data_position);
        }

        static bool headerMatchesCloud(const io::detail::PcdHeader& header, const PCLPointCloud2& cloud)
        {
            return header.width == cloud.width && header.height == cloud.height &&
                   header.points == static_cast<std::uint64_t>(cloud.width) * cloud.height;
        }

        static void readPcdAsciiFromCloud(std::istream& stream, PCLPointCloud2& cloud)
        {
            const std::size_t points = static_cast<std::size_t>(cloud.width) * cloud.height;
            for (std::size_t point = 0; point < points; ++point)
            {
                for (const auto& field : cloud.fields)
                {
                    const std::size_t element_size = getFieldSize(field.datatype);
                    if (element_size == 0 || field.offset + element_size * field.count > cloud.point_step)
                    {
                        throw std::invalid_argument("PCD: field exceeds point step");
                    }
                    for (std::uint32_t element = 0; element < field.count; ++element)
                    {
                        std::string token;
                        if (!(stream >> token))
                        {
                            throw std::runtime_error("PCD: unexpected end of ASCII point data");
                        }
                        io::detail::writePcdToken(cloud.data.data() + point * cloud.point_step + field.offset +
                                                      element * element_size,
                                                  field.datatype,
                                                  token,
                                                  field.name == "rgb" && field.datatype == PCLPointField::FLOAT32);
                    }
                }
            }
        }

        static std::uint32_t readLittleEndianU32(const unsigned char* data)
        {
            return static_cast<std::uint32_t>(data[0]) | (static_cast<std::uint32_t>(data[1]) << 8) |
                   (static_cast<std::uint32_t>(data[2]) << 16) | (static_cast<std::uint32_t>(data[3]) << 24);
        }

        static void updateDenseFlag(PCLPointCloud2& cloud)
        {
            cloud.is_dense = 1;
            const std::size_t points = static_cast<std::size_t>(cloud.width) * cloud.height;
            for (const auto& field : cloud.fields)
            {
                if (field.datatype != PCLPointField::FLOAT32 && field.datatype != PCLPointField::FLOAT64)
                {
                    continue;
                }
                const std::size_t element_size = getFieldSize(field.datatype);
                for (std::size_t point = 0; point < points; ++point)
                {
                    for (std::uint32_t element = 0; element < field.count; ++element)
                    {
                        const auto* value =
                            cloud.data.data() + point * cloud.point_step + field.offset + element * element_size;
                        if (!std::isfinite(
                                static_cast<double>(plapoint::detail::readNumericField(value, field.datatype))))
                        {
                            cloud.is_dense = 0;
                            return;
                        }
                    }
                }
            }
        }

        std::optional<io::detail::PcdHeader> _lastHeader;
    };

    class PCDWriter
    {
    public:
        void setMapSynchronization(bool sync)
        {
            _mapSynchronization = sync;
        }

        std::string generateHeaderBinary(const PCLPointCloud2& cloud,
                                         const Eigen::Vector4f& origin,
                                         const Eigen::Quaternionf& orientation)
        {
            auto header = io::detail::pcdHeaderText(cloud, origin, orientation, io::detail::PcdDataMode::Binary);
            return header.substr(0, header.rfind("DATA binary\n"));
        }

        int generateHeaderBinaryCompressed(std::ostream& stream,
                                           const PCLPointCloud2& cloud,
                                           const Eigen::Vector4f& origin,
                                           const Eigen::Quaternionf& orientation)
        {
            try
            {
                auto header =
                    io::detail::pcdHeaderText(cloud, origin, orientation, io::detail::PcdDataMode::BinaryCompressed);
                stream << header.substr(0, header.rfind("DATA binary_compressed\n"));
                return stream ? 0 : -1;
            }
            catch (const std::exception&)
            {
                return -1;
            }
        }

        std::string generateHeaderBinaryCompressed(const PCLPointCloud2& cloud,
                                                   const Eigen::Vector4f& origin,
                                                   const Eigen::Quaternionf& orientation)
        {
            std::ostringstream stream;
            if (generateHeaderBinaryCompressed(stream, cloud, origin, orientation) != 0)
            {
                return {};
            }
            return stream.str();
        }

        std::string generateHeaderASCII(const PCLPointCloud2& cloud,
                                        const Eigen::Vector4f& origin,
                                        const Eigen::Quaternionf& orientation)
        {
            auto header = io::detail::pcdHeaderText(cloud, origin, orientation, io::detail::PcdDataMode::Ascii);
            return header.substr(0, header.rfind("DATA ascii\n"));
        }

        template <typename PointT>
        static std::string generateHeader(const PointCloud<PointT>& cloud,
                                          const int point_count = std::numeric_limits<int>::max())
        {
            PCLPointCloud2 blob;
            toPCLPointCloud2(cloud, blob);
            if (point_count != std::numeric_limits<int>::max())
            {
                blob.width = static_cast<std::uint32_t>(point_count);
                blob.height = point_count == 0 ? 0u : 1u;
            }
            PCDWriter writer;
            return writer.generateHeaderASCII(blob, cloud.sensor_origin_, cloud.sensor_orientation_);
        }

        int writeASCII(const std::string& file_name,
                       const PCLPointCloud2& cloud,
                       const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                       const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                       const int precision = 8)
        {
            try
            {
                std::ofstream stream(file_name, std::ios::binary | std::ios::trunc);
                if (!stream)
                    return -1;
                stream << io::detail::pcdHeaderText(cloud, origin, orientation, io::detail::PcdDataMode::Ascii);
                stream << std::setprecision(precision);
                const auto fields = io::detail::validPcdFields(cloud);
                const std::size_t points = static_cast<std::size_t>(cloud.width) * cloud.height;
                for (std::size_t point = 0; point < points; ++point)
                {
                    bool first = true;
                    for (const auto& field : fields)
                    {
                        const std::size_t element_size = getFieldSize(field.datatype);
                        for (std::uint32_t element = 0; element < field.count; ++element)
                        {
                            if (!first)
                                stream << ' ';
                            first = false;
                            io::detail::writePcdAsciiValue(stream,
                                                           cloud.data.data() + point * cloud.point_step + field.offset +
                                                               element * element_size,
                                                           field);
                        }
                    }
                    stream << '\n';
                }
                return stream ? 0 : -1;
            }
            catch (const std::exception& error)
            {
                std::cerr << "PCDWriter::writeASCII: " << file_name << ": " << error.what() << '\n';
                return -1;
            }
        }

        int writeBinary(std::ostream& stream,
                        const PCLPointCloud2& cloud,
                        const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                        const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity())
        {
            try
            {
                stream << io::detail::pcdHeaderText(cloud, origin, orientation, io::detail::PcdDataMode::Binary);
                const auto data = io::detail::compactPcdData(cloud, io::detail::validPcdFields(cloud));
                stream.write(reinterpret_cast<const char*>(data.data()), data.size());
                return stream ? 0 : -1;
            }
            catch (const std::exception&)
            {
                return -1;
            }
        }

        int writeBinary(const std::string& file_name,
                        const PCLPointCloud2& cloud,
                        const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                        const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity())
        {
            std::ofstream stream(file_name, std::ios::binary | std::ios::trunc);
            return stream ? writeBinary(stream, cloud, origin, orientation) : -1;
        }

        int writeBinaryCompressed(std::ostream& stream,
                                  const PCLPointCloud2& cloud,
                                  const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                                  const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity())
        {
            try
            {
                const auto data = io::detail::fieldMajorPcdData(cloud, io::detail::validPcdFields(cloud));
                if (data.size() > std::numeric_limits<std::uint32_t>::max())
                    return -2;
                const auto compressed = io::detail::lzfLiteralEncode(data);
                if (compressed.size() > std::numeric_limits<std::uint32_t>::max())
                    return -2;
                stream << io::detail::pcdHeaderText(
                    cloud, origin, orientation, io::detail::PcdDataMode::BinaryCompressed);
                io::detail::writePcdU32(stream, static_cast<std::uint32_t>(compressed.size()));
                io::detail::writePcdU32(stream, static_cast<std::uint32_t>(data.size()));
                stream.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                return stream ? 0 : -1;
            }
            catch (const std::exception&)
            {
                return -1;
            }
        }

        int writeBinaryCompressed(const std::string& file_name,
                                  const PCLPointCloud2& cloud,
                                  const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                                  const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity())
        {
            std::ofstream stream(file_name, std::ios::binary | std::ios::trunc);
            return stream ? writeBinaryCompressed(stream, cloud, origin, orientation) : -1;
        }

        int write(const std::string& file_name,
                  const PCLPointCloud2& cloud,
                  const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                  const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                  const bool binary = false)
        {
            return binary ? writeBinary(file_name, cloud, origin, orientation)
                          : writeASCII(file_name, cloud, origin, orientation);
        }

        int write(const std::string& file_name,
                  const PCLPointCloud2::ConstPtr& cloud,
                  const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                  const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                  const bool binary = false)
        {
            return cloud ? write(file_name, *cloud, origin, orientation, binary) : -1;
        }

        template <typename PointT> int writeBinary(const std::string& file_name, const PointCloud<PointT>& cloud)
        {
            PCLPointCloud2 blob;
            toPCLPointCloud2(cloud, blob);
            return writeBinary(file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_);
        }

        template <typename PointT>
        int writeBinaryCompressed(const std::string& file_name, const PointCloud<PointT>& cloud)
        {
            PCLPointCloud2 blob;
            toPCLPointCloud2(cloud, blob);
            return writeBinaryCompressed(file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_);
        }

        template <typename PointT>
        int writeBinary(const std::string& file_name, const PointCloud<PointT>& cloud, const Indices& indices)
        {
            return writeBinary(file_name, PointCloud<PointT>(cloud, indices));
        }

        template <typename PointT>
        int writeASCII(const std::string& file_name, const PointCloud<PointT>& cloud, const int precision = 8)
        {
            PCLPointCloud2 blob;
            toPCLPointCloud2(cloud, blob);
            return writeASCII(file_name, blob, cloud.sensor_origin_, cloud.sensor_orientation_, precision);
        }

        template <typename PointT>
        int writeASCII(const std::string& file_name,
                       const PointCloud<PointT>& cloud,
                       const Indices& indices,
                       const int precision = 8)
        {
            return writeASCII(file_name, PointCloud<PointT>(cloud, indices), precision);
        }

        template <typename PointT>
        int write(const std::string& file_name, const PointCloud<PointT>& cloud, const bool binary = false)
        {
            return binary ? writeBinary(file_name, cloud) : writeASCII(file_name, cloud);
        }

        template <typename PointT>
        int write(const std::string& file_name,
                  const PointCloud<PointT>& cloud,
                  const Indices& indices,
                  bool binary = false)
        {
            return binary ? writeBinary(file_name, cloud, indices) : writeASCII(file_name, cloud, indices);
        }

    private:
        bool _mapSynchronization = false;
    };

    namespace io
    {

        inline int loadPCDFile(const std::string& file_name, PCLPointCloud2& cloud)
        {
            return PCDReader().read(file_name, cloud);
        }

        inline int loadPCDFile(const std::string& file_name,
                               PCLPointCloud2& cloud,
                               Eigen::Vector4f& origin,
                               Eigen::Quaternionf& orientation)
        {
            int version = 0;
            return PCDReader().read(file_name, cloud, origin, orientation, version);
        }

        template <typename PointT> int loadPCDFile(const std::string& file_name, PointCloud<PointT>& cloud)
        {
            return PCDReader().read(file_name, cloud);
        }

        inline int savePCDFile(const std::string& file_name,
                               const PCLPointCloud2& cloud,
                               const Eigen::Vector4f& origin = Eigen::Vector4f::Zero(),
                               const Eigen::Quaternionf& orientation = Eigen::Quaternionf::Identity(),
                               const bool binary_mode = false)
        {
            return PCDWriter().write(file_name, cloud, origin, orientation, binary_mode);
        }

        template <typename PointT>
        int savePCDFile(const std::string& file_name, const PointCloud<PointT>& cloud, bool binary_mode = false)
        {
            return PCDWriter().write(file_name, cloud, binary_mode);
        }

        template <typename PointT> int savePCDFileASCII(const std::string& file_name, const PointCloud<PointT>& cloud)
        {
            return PCDWriter().writeASCII(file_name, cloud);
        }

        template <typename PointT> int savePCDFileBinary(const std::string& file_name, const PointCloud<PointT>& cloud)
        {
            return PCDWriter().writeBinary(file_name, cloud);
        }

        template <typename PointT>
        int savePCDFile(const std::string& file_name,
                        const PointCloud<PointT>& cloud,
                        const Indices& indices,
                        const bool binary_mode = false)
        {
            return PCDWriter().write(file_name, cloud, indices, binary_mode);
        }

        template <typename PointT>
        int savePCDFileBinaryCompressed(const std::string& file_name, const PointCloud<PointT>& cloud)
        {
            return PCDWriter().writeBinaryCompressed(file_name, cloud);
        }

    } // namespace io
} // namespace plapoint
