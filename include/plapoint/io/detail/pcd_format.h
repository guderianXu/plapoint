#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <plapoint/core/point_cloud_blob.h>

namespace plapoint::io::detail
{

    enum class PcdDataMode
    {
        Ascii,
        Binary,
        BinaryCompressed
    };

    struct PcdHeader
    {
        std::vector<std::string> names;
        std::vector<int> sizes;
        std::vector<char> types;
        std::vector<std::uint32_t> counts;
        std::uint32_t width = 0;
        std::uint32_t height = 1;
        std::uint64_t points = 0;
        Eigen::Vector4f origin = Eigen::Vector4f::Zero();
        Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();
        PcdDataMode mode = PcdDataMode::Ascii;
        std::streamoff data_position = 0;
        int version = 7;
    };

    inline std::vector<std::string> splitPcdLine(const std::string& line)
    {
        std::istringstream stream(line);
        std::vector<std::string> tokens;
        for (std::string token; stream >> token;)
        {
            tokens.push_back(std::move(token));
        }
        return tokens;
    }

    inline std::string uppercasePcdToken(std::string value)
    {
        std::transform(value.begin(),
                       value.end(),
                       value.begin(),
                       [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
        return value;
    }

    inline std::uint8_t pcdDatatype(char type, int size)
    {
        if (type == 'F' && size == 4)
            return PCLPointField::FLOAT32;
        if (type == 'F' && size == 8)
            return PCLPointField::FLOAT64;
        if (type == 'I' && size == 1)
            return PCLPointField::INT8;
        if (type == 'I' && size == 2)
            return PCLPointField::INT16;
        if (type == 'I' && size == 4)
            return PCLPointField::INT32;
        if (type == 'I' && size == 8)
            return PCLPointField::INT64;
        if (type == 'U' && size == 1)
            return PCLPointField::UINT8;
        if (type == 'U' && size == 2)
            return PCLPointField::UINT16;
        if (type == 'U' && size == 4)
            return PCLPointField::UINT32;
        if (type == 'U' && size == 8)
            return PCLPointField::UINT64;
        throw std::invalid_argument("PCD: unsupported TYPE/SIZE combination");
    }

    inline char pcdFieldType(std::uint8_t datatype)
    {
        switch (datatype)
        {
        case PCLPointField::FLOAT32:
        case PCLPointField::FLOAT64:
            return 'F';
        case PCLPointField::INT8:
        case PCLPointField::INT16:
        case PCLPointField::INT32:
        case PCLPointField::INT64:
            return 'I';
        case PCLPointField::BOOL:
        case PCLPointField::UINT8:
        case PCLPointField::UINT16:
        case PCLPointField::UINT32:
        case PCLPointField::UINT64:
            return 'U';
        default:
            throw std::invalid_argument("PCD: unsupported point field datatype");
        }
    }

    inline PcdHeader readPcdHeader(std::istream& stream)
    {
        PcdHeader header;
        std::string line;
        bool found_data = false;
        while (std::getline(stream, line))
        {
            const auto tokens = splitPcdLine(line);
            if (tokens.empty() || tokens[0][0] == '#')
            {
                continue;
            }
            const std::string key = uppercasePcdToken(tokens[0]);
            if (key == "VERSION" && tokens.size() >= 2)
            {
                header.version = tokens[1].find('7') != std::string::npos ? 7 : 6;
            }
            else if (key == "FIELDS" || key == "COLUMNS")
            {
                header.names.assign(tokens.begin() + 1, tokens.end());
            }
            else if (key == "SIZE")
            {
                header.sizes.clear();
                for (std::size_t index = 1; index < tokens.size(); ++index)
                {
                    header.sizes.push_back(std::stoi(tokens[index]));
                }
            }
            else if (key == "TYPE")
            {
                header.types.clear();
                for (std::size_t index = 1; index < tokens.size(); ++index)
                {
                    header.types.push_back(static_cast<char>(std::toupper(tokens[index].at(0))));
                }
            }
            else if (key == "COUNT")
            {
                header.counts.clear();
                for (std::size_t index = 1; index < tokens.size(); ++index)
                {
                    header.counts.push_back(static_cast<std::uint32_t>(std::stoul(tokens[index])));
                }
            }
            else if (key == "WIDTH" && tokens.size() >= 2)
            {
                header.width = static_cast<std::uint32_t>(std::stoul(tokens[1]));
            }
            else if (key == "HEIGHT" && tokens.size() >= 2)
            {
                header.height = static_cast<std::uint32_t>(std::stoul(tokens[1]));
            }
            else if (key == "POINTS" && tokens.size() >= 2)
            {
                header.points = std::stoull(tokens[1]);
            }
            else if (key == "VIEWPOINT" && tokens.size() >= 8)
            {
                header.origin = Eigen::Vector4f(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]), 0.0f);
                header.orientation = Eigen::Quaternionf(
                    std::stof(tokens[4]), std::stof(tokens[5]), std::stof(tokens[6]), std::stof(tokens[7]));
            }
            else if (key == "DATA" && tokens.size() >= 2)
            {
                const std::string mode = uppercasePcdToken(tokens[1]);
                if (mode == "ASCII")
                    header.mode = PcdDataMode::Ascii;
                else if (mode == "BINARY")
                    header.mode = PcdDataMode::Binary;
                else if (mode == "BINARY_COMPRESSED")
                    header.mode = PcdDataMode::BinaryCompressed;
                else
                    throw std::invalid_argument("PCD: unsupported DATA mode");
                header.data_position = stream.tellg();
                found_data = true;
                break;
            }
        }
        if (!found_data || header.names.empty() || header.sizes.size() != header.names.size() ||
            header.types.size() != header.names.size())
        {
            throw std::invalid_argument("PCD: incomplete header");
        }
        if (header.counts.empty())
        {
            header.counts.assign(header.names.size(), 1);
        }
        if (header.counts.size() != header.names.size())
        {
            throw std::invalid_argument("PCD: COUNT does not match FIELDS");
        }
        if (header.points == 0)
        {
            header.points = static_cast<std::uint64_t>(header.width) * header.height;
        }
        if (header.width == 0 && header.points <= std::numeric_limits<std::uint32_t>::max())
        {
            header.width = static_cast<std::uint32_t>(header.points);
            header.height = header.points == 0 ? 0u : 1u;
        }
        if (static_cast<std::uint64_t>(header.width) * header.height != header.points ||
            header.points > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument("PCD: inconsistent or oversized point dimensions");
        }
        return header;
    }

    inline void configurePcdCloud(const PcdHeader& header, PCLPointCloud2& cloud)
    {
        cloud = {};
        cloud.width = header.width;
        cloud.height = header.height;
        cloud.is_bigendian = static_cast<std::uint8_t>(plapoint::detail::hostIsBigEndian());
        std::uint64_t offset = 0;
        for (std::size_t index = 0; index < header.names.size(); ++index)
        {
            const std::uint8_t datatype = pcdDatatype(header.types[index], header.sizes[index]);
            const std::uint64_t bytes = static_cast<std::uint64_t>(header.sizes[index]) * header.counts[index];
            if (offset + bytes > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::overflow_error("PCD: point step exceeds 32-bit range");
            }
            if (header.names[index] != "_")
            {
                cloud.fields.push_back(
                    {header.names[index], static_cast<std::uint32_t>(offset), datatype, header.counts[index]});
            }
            offset += bytes;
        }
        cloud.point_step = static_cast<std::uint32_t>(offset);
        cloud.row_step = cloud.point_step * cloud.width;
        if (header.points != 0 && cloud.point_step > std::numeric_limits<std::size_t>::max() / header.points)
        {
            throw std::overflow_error("PCD: point data exceeds addressable memory");
        }
        cloud.data.assign(static_cast<std::size_t>(header.points) * cloud.point_step, 0);
        cloud.is_dense = 1;
    }

    template <typename T> inline void writePcdScalar(std::uint8_t* destination, long double value)
    {
        const T converted = static_cast<T>(value);
        std::memcpy(destination, &converted, sizeof(converted));
    }

    inline void
    writePcdToken(std::uint8_t* destination, std::uint8_t datatype, const std::string& token, bool packed_rgb)
    {
        if (packed_rgb)
        {
            const std::uint32_t bits = static_cast<std::uint32_t>(std::stoull(token));
            std::memcpy(destination, &bits, sizeof(bits));
            return;
        }
        const long double value = std::stold(token);
        switch (datatype)
        {
        case PCLPointField::INT8:
            writePcdScalar<std::int8_t>(destination, value);
            break;
        case PCLPointField::UINT8:
            writePcdScalar<std::uint8_t>(destination, value);
            break;
        case PCLPointField::INT16:
            writePcdScalar<std::int16_t>(destination, value);
            break;
        case PCLPointField::UINT16:
            writePcdScalar<std::uint16_t>(destination, value);
            break;
        case PCLPointField::INT32:
            writePcdScalar<std::int32_t>(destination, value);
            break;
        case PCLPointField::UINT32:
            writePcdScalar<std::uint32_t>(destination, value);
            break;
        case PCLPointField::FLOAT32:
            writePcdScalar<float>(destination, value);
            break;
        case PCLPointField::FLOAT64:
            writePcdScalar<double>(destination, value);
            break;
        case PCLPointField::INT64:
            writePcdScalar<std::int64_t>(destination, value);
            break;
        case PCLPointField::UINT64:
            writePcdScalar<std::uint64_t>(destination, value);
            break;
        default:
            throw std::invalid_argument("PCD: unsupported ASCII datatype");
        }
    }

    inline void readPcdAscii(std::istream& stream, const PcdHeader& header, PCLPointCloud2& cloud)
    {
        std::vector<PCLPointField> all_fields;
        std::uint32_t offset = 0;
        for (std::size_t index = 0; index < header.names.size(); ++index)
        {
            all_fields.push_back({header.names[index],
                                  offset,
                                  pcdDatatype(header.types[index], header.sizes[index]),
                                  header.counts[index]});
            offset += static_cast<std::uint32_t>(header.sizes[index]) * header.counts[index];
        }
        for (std::uint64_t point = 0; point < header.points; ++point)
        {
            for (const auto& field : all_fields)
            {
                const std::size_t element_size = getFieldSize(field.datatype);
                for (std::uint32_t element = 0; element < field.count; ++element)
                {
                    std::string token;
                    if (!(stream >> token))
                    {
                        throw std::runtime_error("PCD: unexpected end of ASCII point data");
                    }
                    writePcdToken(cloud.data.data() + point * cloud.point_step + field.offset + element * element_size,
                                  field.datatype,
                                  token,
                                  field.name == "rgb" && field.datatype == PCLPointField::FLOAT32);
                }
            }
        }
    }

    inline std::vector<std::uint8_t> lzfLiteralEncode(const std::vector<std::uint8_t>& input)
    {
        std::vector<std::uint8_t> output;
        output.reserve(input.size() + (input.size() + 31) / 32);
        for (std::size_t offset = 0; offset < input.size();)
        {
            const std::size_t count = std::min<std::size_t>(32, input.size() - offset);
            output.push_back(static_cast<std::uint8_t>(count - 1));
            output.insert(output.end(),
                          input.begin() + static_cast<std::ptrdiff_t>(offset),
                          input.begin() + static_cast<std::ptrdiff_t>(offset + count));
            offset += count;
        }
        return output;
    }

    inline std::vector<std::uint8_t> lzfDecode(const std::vector<std::uint8_t>& input, std::size_t output_size)
    {
        std::vector<std::uint8_t> output(output_size);
        std::size_t input_position = 0;
        std::size_t output_position = 0;
        while (input_position < input.size() && output_position < output.size())
        {
            const unsigned int control = input[input_position++];
            if (control < 32)
            {
                const std::size_t length = control + 1;
                if (input_position + length > input.size() || output_position + length > output.size())
                {
                    throw std::invalid_argument("PCD: invalid LZF literal run");
                }
                std::memcpy(output.data() + output_position, input.data() + input_position, length);
                input_position += length;
                output_position += length;
                continue;
            }
            std::size_t length = control >> 5;
            std::size_t reference_distance = (control & 0x1fu) << 8;
            if (length == 7)
            {
                if (input_position >= input.size())
                    throw std::invalid_argument("PCD: invalid LZF length");
                length += input[input_position++];
            }
            if (input_position >= input.size())
                throw std::invalid_argument("PCD: invalid LZF reference");
            reference_distance += input[input_position++];
            if (reference_distance + 1 > output_position || output_position + length + 2 > output.size())
            {
                throw std::invalid_argument("PCD: LZF reference is outside decoded data");
            }
            std::size_t reference = output_position - reference_distance - 1;
            length += 2;
            while (length-- != 0)
            {
                output[output_position++] = output[reference++];
            }
        }
        if (output_position != output.size())
        {
            throw std::invalid_argument("PCD: decoded LZF size does not match header");
        }
        return output;
    }

    inline std::uint32_t readPcdU32(std::istream& stream)
    {
        std::array<std::uint8_t, 4> bytes{};
        if (!stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        {
            throw std::runtime_error("PCD: missing compressed size header");
        }
        return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
               (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
    }

    inline void writePcdU32(std::ostream& stream, std::uint32_t value)
    {
        const std::array<std::uint8_t, 4> bytes = {static_cast<std::uint8_t>(value),
                                                   static_cast<std::uint8_t>(value >> 8),
                                                   static_cast<std::uint8_t>(value >> 16),
                                                   static_cast<std::uint8_t>(value >> 24)};
        stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }

    inline std::vector<PCLPointField> validPcdFields(const PCLPointCloud2& cloud)
    {
        std::vector<PCLPointField> fields;
        for (const auto& field : cloud.fields)
        {
            if (field.name != "_")
            {
                fields.push_back(field);
            }
        }
        std::sort(
            fields.begin(), fields.end(), [](const auto& lhs, const auto& rhs) { return lhs.offset < rhs.offset; });
        return fields;
    }

    inline std::vector<std::uint8_t> compactPcdData(const PCLPointCloud2& cloud,
                                                    const std::vector<PCLPointField>& fields)
    {
        const std::size_t points = static_cast<std::size_t>(cloud.width) * cloud.height;
        std::size_t compact_step = 0;
        for (const auto& field : fields)
        {
            const std::size_t bytes = getFieldSize(field.datatype) * field.count;
            if (field.offset + bytes > cloud.point_step)
            {
                throw std::invalid_argument("PCD: field exceeds point step");
            }
            compact_step += bytes;
        }
        if (points != 0 && (cloud.data.size() < points * cloud.point_step ||
                            compact_step > std::numeric_limits<std::size_t>::max() / points))
        {
            throw std::invalid_argument("PCD: inconsistent cloud data size");
        }
        std::vector<std::uint8_t> output(points * compact_step);
        for (std::size_t point = 0; point < points; ++point)
        {
            std::size_t destination_offset = point * compact_step;
            for (const auto& field : fields)
            {
                const std::size_t bytes = getFieldSize(field.datatype) * field.count;
                std::memcpy(output.data() + destination_offset,
                            cloud.data.data() + point * cloud.point_step + field.offset,
                            bytes);
                destination_offset += bytes;
            }
        }
        return output;
    }

    inline std::vector<std::uint8_t> fieldMajorPcdData(const PCLPointCloud2& cloud,
                                                       const std::vector<PCLPointField>& fields)
    {
        const std::size_t points = static_cast<std::size_t>(cloud.width) * cloud.height;
        std::vector<std::uint8_t> output;
        for (const auto& field : fields)
        {
            output.resize(output.size() + getFieldSize(field.datatype) * field.count * points);
        }
        std::size_t field_block = 0;
        for (const auto& field : fields)
        {
            const std::size_t bytes = getFieldSize(field.datatype) * field.count;
            for (std::size_t point = 0; point < points; ++point)
            {
                std::memcpy(output.data() + field_block + point * bytes,
                            cloud.data.data() + point * cloud.point_step + field.offset,
                            bytes);
            }
            field_block += bytes * points;
        }
        return output;
    }

    inline void expandFieldMajorPcdData(const std::vector<std::uint8_t>& field_major, PCLPointCloud2& cloud)
    {
        const std::size_t points = static_cast<std::size_t>(cloud.width) * cloud.height;
        std::size_t field_block = 0;
        for (const auto& field : cloud.fields)
        {
            const std::size_t bytes = getFieldSize(field.datatype) * field.count;
            if (field_block + bytes * points > field_major.size())
            {
                throw std::invalid_argument("PCD: compressed field data is truncated");
            }
            for (std::size_t point = 0; point < points; ++point)
            {
                std::memcpy(cloud.data.data() + point * cloud.point_step + field.offset,
                            field_major.data() + field_block + point * bytes,
                            bytes);
            }
            field_block += bytes * points;
        }
        if (field_block != field_major.size())
        {
            throw std::invalid_argument("PCD: compressed field data size does not match fields");
        }
    }

    inline std::string pcdHeaderText(const PCLPointCloud2& cloud,
                                     const Eigen::Vector4f& origin,
                                     const Eigen::Quaternionf& orientation,
                                     PcdDataMode mode)
    {
        const auto fields = validPcdFields(cloud);
        if (fields.empty())
        {
            throw std::invalid_argument("PCD: cloud has no fields");
        }
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS";
        for (const auto& field : fields)
            output << ' ' << field.name;
        output << "\nSIZE";
        for (const auto& field : fields)
            output << ' ' << getFieldSize(field.datatype);
        output << "\nTYPE";
        for (const auto& field : fields)
            output << ' ' << pcdFieldType(field.datatype);
        output << "\nCOUNT";
        for (const auto& field : fields)
            output << ' ' << std::max<std::uint32_t>(field.count, 1);
        output << "\nWIDTH " << cloud.width << "\nHEIGHT " << cloud.height << "\nVIEWPOINT " << origin[0] << ' '
               << origin[1] << ' ' << origin[2] << ' ' << orientation.w() << ' ' << orientation.x() << ' '
               << orientation.y() << ' ' << orientation.z() << "\nPOINTS "
               << static_cast<std::uint64_t>(cloud.width) * cloud.height << "\nDATA ";
        if (mode == PcdDataMode::Ascii)
            output << "ascii\n";
        else if (mode == PcdDataMode::Binary)
            output << "binary\n";
        else
            output << "binary_compressed\n";
        return output.str();
    }

    inline void writePcdAsciiValue(std::ostream& stream, const std::uint8_t* source, const PCLPointField& field)
    {
        if (field.name == "rgb" && field.datatype == PCLPointField::FLOAT32)
        {
            std::uint32_t value{};
            std::memcpy(&value, source, sizeof(value));
            stream << value;
            return;
        }
        switch (field.datatype)
        {
#define PLAPOINT_WRITE_PCD_VALUE(code, type, cast_type)                                                                \
    case code:                                                                                                         \
    {                                                                                                                  \
        type value{};                                                                                                  \
        std::memcpy(&value, source, sizeof(value));                                                                    \
        stream << static_cast<cast_type>(value);                                                                       \
        return;                                                                                                        \
    }
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::INT8, std::int8_t, int)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::UINT8, std::uint8_t, unsigned int)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::INT16, std::int16_t, int)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::UINT16, std::uint16_t, unsigned int)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::INT32, std::int32_t, std::int32_t)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::UINT32, std::uint32_t, std::uint32_t)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::FLOAT32, float, float)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::FLOAT64, double, double)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::INT64, std::int64_t, std::int64_t)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::UINT64, std::uint64_t, std::uint64_t)
            PLAPOINT_WRITE_PCD_VALUE(PCLPointField::BOOL, bool, int)
#undef PLAPOINT_WRITE_PCD_VALUE
        default:
            throw std::invalid_argument("PCD: unsupported ASCII datatype");
        }
    }

} // namespace plapoint::io::detail
