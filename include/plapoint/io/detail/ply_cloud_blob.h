#pragma once

#include <algorithm>
#include <array>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <plapoint/core/mesh_types.h>
#include <plapoint/io/detail/pcd_format.h>
#include <plapoint/io/ply_io.h>

namespace plapoint::io::detail
{

    struct BlobPlyProperty
    {
        bool list = false;
        std::string count_type;
        std::string type;
        std::string name;
    };

    struct BlobPlyElement
    {
        std::string name;
        std::size_t count = 0;
        std::vector<BlobPlyProperty> properties;
    };

    struct BlobPlyDocument
    {
        PCLPointCloud2 cloud;
        std::vector<Vertices> polygons;
        Eigen::Vector4f origin = Eigen::Vector4f::Zero();
        Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();
        int version = 1;
    };

    inline std::uint8_t plyBlobDatatype(const std::string& type)
    {
        if (type == "char" || type == "int8")
            return PCLPointField::INT8;
        if (type == "uchar" || type == "uint8" || type == "unsigned_char")
            return PCLPointField::UINT8;
        if (type == "short" || type == "int16")
            return PCLPointField::INT16;
        if (type == "ushort" || type == "uint16" || type == "unsigned_short")
            return PCLPointField::UINT16;
        if (type == "int" || type == "int32")
            return PCLPointField::INT32;
        if (type == "uint" || type == "uint32" || type == "unsigned_int")
            return PCLPointField::UINT32;
        if (type == "float" || type == "float32")
            return PCLPointField::FLOAT32;
        if (type == "double" || type == "float64")
            return PCLPointField::FLOAT64;
        throw std::invalid_argument("PLY: unsupported scalar type " + type);
    }

    inline const char* plyBlobTypeName(std::uint8_t datatype)
    {
        switch (datatype)
        {
        case PCLPointField::INT8:
            return "char";
        case PCLPointField::UINT8:
        case PCLPointField::BOOL:
            return "uchar";
        case PCLPointField::INT16:
            return "short";
        case PCLPointField::UINT16:
            return "ushort";
        case PCLPointField::INT32:
            return "int";
        case PCLPointField::UINT32:
            return "uint";
        case PCLPointField::FLOAT32:
            return "float";
        case PCLPointField::FLOAT64:
            return "double";
        default:
            throw std::invalid_argument("PLY: unsupported point field datatype");
        }
    }

    inline std::string normalizedPlyFieldName(const std::string& name)
    {
        if (name == "nx")
            return "normal_x";
        if (name == "ny")
            return "normal_y";
        if (name == "nz")
            return "normal_z";
        return name;
    }

    inline bool isPlyRed(const std::string& name)
    {
        return name == "red" || name == "diffuse_red";
    }
    inline bool isPlyGreen(const std::string& name)
    {
        return name == "green" || name == "diffuse_green";
    }
    inline bool isPlyBlue(const std::string& name)
    {
        return name == "blue" || name == "diffuse_blue";
    }
    inline bool isPlyAlpha(const std::string& name)
    {
        return name == "alpha";
    }

    template <typename T> inline long double readPlyNumber(std::istream& stream, bool swap_bytes)
    {
        T value{};
        if (!stream.read(reinterpret_cast<char*>(&value), sizeof(value)))
        {
            throw std::runtime_error("PLY: binary scalar data is truncated");
        }
        if (swap_bytes && sizeof(value) > 1)
        {
            plapoint::io::detail::swapEndian(value);
        }
        return static_cast<long double>(value);
    }

    inline long double readPlyNumber(std::istream& stream, const std::string& type, PlyFormat format)
    {
        if (format == PlyFormat::ASCII)
        {
            std::string token;
            if (!(stream >> token))
            {
                throw std::runtime_error("PLY: ASCII scalar data is truncated");
            }
            return std::stold(token);
        }
        const bool swap_bytes = (format == PlyFormat::BinaryBE) == plapoint::io::detail::isLittleEndian();
        const auto datatype = plyBlobDatatype(type);
        switch (datatype)
        {
        case PCLPointField::INT8:
            return readPlyNumber<std::int8_t>(stream, false);
        case PCLPointField::UINT8:
            return readPlyNumber<std::uint8_t>(stream, false);
        case PCLPointField::INT16:
            return readPlyNumber<std::int16_t>(stream, swap_bytes);
        case PCLPointField::UINT16:
            return readPlyNumber<std::uint16_t>(stream, swap_bytes);
        case PCLPointField::INT32:
            return readPlyNumber<std::int32_t>(stream, swap_bytes);
        case PCLPointField::UINT32:
            return readPlyNumber<std::uint32_t>(stream, swap_bytes);
        case PCLPointField::FLOAT32:
            return readPlyNumber<float>(stream, swap_bytes);
        case PCLPointField::FLOAT64:
            return readPlyNumber<double>(stream, swap_bytes);
        default:
            throw std::invalid_argument("PLY: unsupported scalar datatype");
        }
    }

    inline void storePlyNumber(std::uint8_t* destination, std::uint8_t datatype, long double value)
    {
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
        default:
            throw std::invalid_argument("PLY: unsupported scalar datatype");
        }
    }

    inline std::uint8_t plyBlobColor(long double value, const std::string& type)
    {
        return plapoint::io::detail::plyColorByteForProperty(static_cast<double>(value), type);
    }

    inline std::size_t plyListCount(std::istream& stream, const BlobPlyProperty& property, PlyFormat format)
    {
        const long double value = readPlyNumber(stream, property.count_type, format);
        if (!std::isfinite(static_cast<double>(value)) || value < 0.0L || std::floor(value) != value ||
            value > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::invalid_argument("PLY: invalid list count");
        }
        return static_cast<std::size_t>(value);
    }

    inline BlobPlyDocument readPlyBlob(const std::string& path, int file_offset = 0)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream || file_offset < 0)
        {
            throw std::runtime_error("Cannot open PLY file: " + path);
        }
        stream.seekg(file_offset);
        std::string line;
        if (!std::getline(stream, line) || (line != "ply" && line != "ply\r"))
        {
            throw std::invalid_argument("PLY: missing magic header");
        }
        if (!std::getline(stream, line))
        {
            throw std::invalid_argument("PLY: missing format line");
        }
        PlyFormat format;
        if (line.find("format ascii") != std::string::npos)
            format = PlyFormat::ASCII;
        else if (line.find("binary_little_endian") != std::string::npos)
            format = PlyFormat::BinaryLE;
        else if (line.find("binary_big_endian") != std::string::npos)
            format = PlyFormat::BinaryBE;
        else
            throw std::invalid_argument("PLY: unsupported format");

        std::vector<BlobPlyElement> elements;
        BlobPlyElement* current = nullptr;
        std::array<long double, 3> point_offset{};
        bool found_end = false;
        while (std::getline(stream, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line == "end_header")
            {
                found_end = true;
                break;
            }
            std::istringstream parser(line);
            std::string keyword;
            parser >> keyword;
            if (keyword == "element")
            {
                BlobPlyElement element;
                if (!(parser >> element.name >> element.count))
                {
                    throw std::invalid_argument("PLY: malformed element declaration");
                }
                elements.push_back(std::move(element));
                current = &elements.back();
            }
            else if (keyword == "property")
            {
                if (!current)
                    throw std::invalid_argument("PLY: property precedes element");
                BlobPlyProperty property;
                if (!(parser >> property.type))
                    throw std::invalid_argument("PLY: malformed property");
                if (property.type == "list")
                {
                    property.list = true;
                    if (!(parser >> property.count_type >> property.type >> property.name))
                    {
                        throw std::invalid_argument("PLY: malformed list property");
                    }
                }
                else if (!(parser >> property.name))
                {
                    throw std::invalid_argument("PLY: malformed scalar property");
                }
                current->properties.push_back(std::move(property));
            }
            else if (keyword == "comment")
            {
                std::string name;
                parser >> name;
                if (name == "POINT_OFFSET")
                    parser >> point_offset[0] >> point_offset[1] >> point_offset[2];
            }
        }
        if (!found_end)
            throw std::invalid_argument("PLY: missing end_header");

        const auto vertex_iterator = std::find_if(
            elements.begin(), elements.end(), [](const auto& element) { return element.name == "vertex"; });
        const std::size_t point_count = vertex_iterator == elements.end() ? 0 : vertex_iterator->count;
        if (point_count > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error("PLY: point count exceeds 32-bit range");
        }

        BlobPlyDocument document;
        document.cloud.width = static_cast<std::uint32_t>(point_count);
        document.cloud.height = point_count == 0 ? 0u : 1u;
        document.cloud.is_bigendian = static_cast<std::uint8_t>(plapoint::detail::hostIsBigEndian());
        bool has_red = false;
        bool has_green = false;
        bool has_blue = false;
        bool has_alpha = false;
        if (vertex_iterator != elements.end())
        {
            std::uint32_t field_offset = 0;
            for (const auto& property : vertex_iterator->properties)
            {
                if (property.list)
                    continue;
                has_red = has_red || isPlyRed(property.name);
                has_green = has_green || isPlyGreen(property.name);
                has_blue = has_blue || isPlyBlue(property.name);
                has_alpha = has_alpha || isPlyAlpha(property.name);
                if (isPlyRed(property.name) || isPlyGreen(property.name) || isPlyBlue(property.name) ||
                    isPlyAlpha(property.name))
                {
                    continue;
                }
                const std::string name = normalizedPlyFieldName(property.name);
                if (std::any_of(document.cloud.fields.begin(),
                                document.cloud.fields.end(),
                                [&name](const auto& field) { return field.name == name; }))
                {
                    throw std::invalid_argument("PLY: duplicate vertex property " + name);
                }
                const auto datatype = plyBlobDatatype(property.type);
                document.cloud.fields.push_back({name, field_offset, datatype, 1});
                field_offset += static_cast<std::uint32_t>(getFieldSize(datatype));
            }
            if (has_red && has_green && has_blue)
            {
                document.cloud.fields.push_back({has_alpha ? "rgba" : "rgb",
                                                 field_offset,
                                                 has_alpha ? PCLPointField::UINT32 : PCLPointField::FLOAT32,
                                                 1});
                field_offset += 4;
            }
            document.cloud.point_step = field_offset;
            document.cloud.row_step = field_offset * document.cloud.width;
            document.cloud.data.assign(point_count * field_offset, 0);
        }
        document.cloud.is_dense = 1;

        std::unordered_map<std::string, PCLPointField> output_fields;
        for (const auto& field : document.cloud.fields)
            output_fields.emplace(field.name, field);
        std::array<float, 9> camera_rotation = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        bool camera_seen = false;
        for (const auto& element : elements)
        {
            for (std::size_t row = 0; row < element.count; ++row)
            {
                std::array<std::uint8_t, 4> color = {0, 0, 0, 255};
                Vertices polygon;
                for (const auto& property : element.properties)
                {
                    if (property.list)
                    {
                        const std::size_t count = plyListCount(stream, property, format);
                        for (std::size_t item = 0; item < count; ++item)
                        {
                            const long double value = readPlyNumber(stream, property.type, format);
                            if (element.name == "face" &&
                                (property.name == "vertex_indices" || property.name == "vertex_index"))
                            {
                                if (value < 0.0L ||
                                    value > static_cast<long double>(std::numeric_limits<index_t>::max()) ||
                                    std::floor(value) != value)
                                {
                                    throw std::invalid_argument("PLY: invalid face vertex index");
                                }
                                polygon.vertices.push_back(static_cast<index_t>(value));
                            }
                        }
                        continue;
                    }
                    long double value = readPlyNumber(stream, property.type, format);
                    if (element.name == "vertex")
                    {
                        if (property.name == "x")
                            value += point_offset[0];
                        if (property.name == "y")
                            value += point_offset[1];
                        if (property.name == "z")
                            value += point_offset[2];
                        if (isPlyRed(property.name))
                            color[2] = plyBlobColor(value, property.type);
                        else if (isPlyGreen(property.name))
                            color[1] = plyBlobColor(value, property.type);
                        else if (isPlyBlue(property.name))
                            color[0] = plyBlobColor(value, property.type);
                        else if (isPlyAlpha(property.name))
                            color[3] = plyBlobColor(value, property.type);
                        else
                        {
                            const auto field = output_fields.find(normalizedPlyFieldName(property.name));
                            if (field != output_fields.end())
                            {
                                storePlyNumber(document.cloud.data.data() + row * document.cloud.point_step +
                                                   field->second.offset,
                                               field->second.datatype,
                                               value);
                                if ((property.name == "x" || property.name == "y" || property.name == "z") &&
                                    !std::isfinite(static_cast<double>(value)))
                                {
                                    document.cloud.is_dense = 0;
                                }
                            }
                        }
                    }
                    else if (element.name == "camera")
                    {
                        camera_seen = true;
                        if (property.name == "view_px")
                            document.origin[0] = static_cast<float>(value);
                        else if (property.name == "view_py")
                            document.origin[1] = static_cast<float>(value);
                        else if (property.name == "view_pz")
                            document.origin[2] = static_cast<float>(value);
                        else if (property.name == "x_axisx")
                            camera_rotation[0] = static_cast<float>(value);
                        else if (property.name == "x_axisy")
                            camera_rotation[1] = static_cast<float>(value);
                        else if (property.name == "x_axisz")
                            camera_rotation[2] = static_cast<float>(value);
                        else if (property.name == "y_axisx")
                            camera_rotation[3] = static_cast<float>(value);
                        else if (property.name == "y_axisy")
                            camera_rotation[4] = static_cast<float>(value);
                        else if (property.name == "y_axisz")
                            camera_rotation[5] = static_cast<float>(value);
                        else if (property.name == "z_axisx")
                            camera_rotation[6] = static_cast<float>(value);
                        else if (property.name == "z_axisy")
                            camera_rotation[7] = static_cast<float>(value);
                        else if (property.name == "z_axisz")
                            camera_rotation[8] = static_cast<float>(value);
                    }
                }
                if (element.name == "vertex" && has_red && has_green && has_blue)
                {
                    const auto& field = document.cloud.fields.back();
                    const std::uint32_t packed =
                        static_cast<std::uint32_t>(color[0]) | (static_cast<std::uint32_t>(color[1]) << 8) |
                        (static_cast<std::uint32_t>(color[2]) << 16) | (static_cast<std::uint32_t>(color[3]) << 24);
                    std::memcpy(document.cloud.data.data() + row * document.cloud.point_step + field.offset,
                                &packed,
                                sizeof(packed));
                }
                if (element.name == "face" && !polygon.vertices.empty())
                {
                    for (const int index : polygon.vertices)
                    {
                        if (index < 0 || static_cast<std::size_t>(index) >= point_count)
                        {
                            throw std::out_of_range("PLY: face vertex index is outside the cloud");
                        }
                    }
                    document.polygons.push_back(std::move(polygon));
                }
            }
        }
        if (camera_seen)
        {
            Eigen::Matrix3f rotation;
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    rotation(row, column) = camera_rotation[static_cast<std::size_t>(row * 3 + column)];
                }
            }
            document.orientation = Eigen::Quaternionf(rotation);
        }
        return document;
    }

    inline std::string plyOutputName(const std::string& name)
    {
        if (name == "normal_x")
            return "nx";
        if (name == "normal_y")
            return "ny";
        if (name == "normal_z")
            return "nz";
        return name;
    }

    template <typename T> inline void writePlyBinaryNumber(std::ostream& stream, T value)
    {
        if (!plapoint::io::detail::isLittleEndian() && sizeof(value) > 1)
        {
            plapoint::io::detail::swapEndian(value);
        }
        stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    inline void
    writePlyStoredNumber(std::ostream& stream, const std::uint8_t* source, std::uint8_t datatype, bool binary)
    {
        if (!binary)
        {
            PCLPointField field;
            field.datatype = datatype;
            writePcdAsciiValue(stream, source, field);
            return;
        }
        switch (datatype)
        {
#define PLAPOINT_WRITE_PLY_BINARY(code, type)                                                                          \
    case code:                                                                                                         \
    {                                                                                                                  \
        type value{};                                                                                                  \
        std::memcpy(&value, source, sizeof(value));                                                                    \
        writePlyBinaryNumber(stream, value);                                                                           \
        return;                                                                                                        \
    }
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::INT8, std::int8_t)
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::UINT8, std::uint8_t)
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::INT16, std::int16_t)
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::UINT16, std::uint16_t)
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::INT32, std::int32_t)
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::UINT32, std::uint32_t)
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::FLOAT32, float)
            PLAPOINT_WRITE_PLY_BINARY(PCLPointField::FLOAT64, double)
#undef PLAPOINT_WRITE_PLY_BINARY
        default:
            throw std::invalid_argument("PLY: unsupported binary datatype");
        }
    }

    inline void appendPlyCameraHeader(std::ostream& stream)
    {
        stream << "element camera 1\n"
                  "property float view_px\nproperty float view_py\nproperty float view_pz\n"
                  "property float x_axisx\nproperty float x_axisy\nproperty float x_axisz\n"
                  "property float y_axisx\nproperty float y_axisy\nproperty float y_axisz\n"
                  "property float z_axisx\nproperty float z_axisy\nproperty float z_axisz\n"
                  "property float focal\nproperty float scalex\nproperty float scaley\n"
                  "property float centerx\nproperty float centery\n"
                  "property int viewportx\nproperty int viewporty\n"
                  "property float k1\nproperty float k2\n";
    }

    inline void writePlyBlob(std::ostream& stream,
                             const PCLPointCloud2& cloud,
                             const std::vector<Vertices>& polygons,
                             const Eigen::Vector4f& origin,
                             const Eigen::Quaternionf& orientation,
                             bool binary,
                             bool use_camera,
                             unsigned int precision)
    {
        const std::size_t point_count = static_cast<std::size_t>(cloud.width) * cloud.height;
        if (cloud.data.size() < point_count * cloud.point_step)
        {
            throw std::invalid_argument("PLY: inconsistent point cloud data size");
        }
        const auto fields = validPcdFields(cloud);
        stream.imbue(std::locale::classic());
        stream << "ply\nformat " << (binary ? "binary_little_endian" : "ascii") << " 1.0\n"
               << "element vertex " << point_count << '\n';
        for (const auto& field : fields)
        {
            if (field.name == "rgb" || field.name == "rgba")
            {
                stream << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
                if (field.name == "rgba")
                    stream << "property uchar alpha\n";
                continue;
            }
            if (field.count == 1)
            {
                stream << "property " << plyBlobTypeName(field.datatype) << ' ' << plyOutputName(field.name) << '\n';
            }
            else
            {
                stream << "property list uint " << plyBlobTypeName(field.datatype) << ' ' << plyOutputName(field.name)
                       << '\n';
            }
        }
        stream << "element face " << polygons.size() << "\nproperty list uchar int vertex_indices\n";
        if (use_camera)
            appendPlyCameraHeader(stream);
        stream << "end_header\n";
        stream << std::setprecision(static_cast<int>(precision));

        for (std::size_t point = 0; point < point_count; ++point)
        {
            bool first = true;
            for (const auto& field : fields)
            {
                const auto* source = cloud.data.data() + point * cloud.point_step + field.offset;
                if (field.name == "rgb" || field.name == "rgba")
                {
                    std::uint32_t packed{};
                    std::memcpy(&packed, source, sizeof(packed));
                    const std::array<std::uint8_t, 4> color = {static_cast<std::uint8_t>(packed >> 16),
                                                               static_cast<std::uint8_t>(packed >> 8),
                                                               static_cast<std::uint8_t>(packed),
                                                               static_cast<std::uint8_t>(packed >> 24)};
                    const int color_count = field.name == "rgba" ? 4 : 3;
                    for (int component = 0; component < color_count; ++component)
                    {
                        if (binary)
                            stream.write(reinterpret_cast<const char*>(&color[component]), 1);
                        else
                        {
                            if (!first)
                                stream << ' ';
                            stream << static_cast<unsigned int>(color[component]);
                        }
                        first = false;
                    }
                    continue;
                }
                const std::size_t element_size = getFieldSize(field.datatype);
                if (field.count != 1)
                {
                    if (binary)
                        writePlyBinaryNumber(stream, field.count);
                    else
                    {
                        if (!first)
                            stream << ' ';
                        stream << field.count;
                        first = false;
                    }
                }
                for (std::uint32_t element = 0; element < field.count; ++element)
                {
                    if (!binary && !first)
                        stream << ' ';
                    writePlyStoredNumber(stream, source + element * element_size, field.datatype, binary);
                    first = false;
                }
            }
            if (!binary)
                stream << '\n';
        }
        for (const auto& polygon : polygons)
        {
            if (polygon.vertices.size() > 255)
            {
                throw std::overflow_error("PLY: polygon has more than 255 vertices");
            }
            if (binary)
            {
                const auto count = static_cast<std::uint8_t>(polygon.vertices.size());
                stream.write(reinterpret_cast<const char*>(&count), 1);
                for (const int vertex : polygon.vertices)
                    writePlyBinaryNumber(stream, static_cast<std::int32_t>(vertex));
            }
            else
            {
                stream << polygon.vertices.size();
                for (const int vertex : polygon.vertices)
                    stream << ' ' << vertex;
                stream << '\n';
            }
        }
        if (use_camera)
        {
            const Eigen::Matrix3f rotation = orientation.toRotationMatrix();
            const std::array<float, 21> values = {origin[0],
                                                  origin[1],
                                                  origin[2],
                                                  rotation(0, 0),
                                                  rotation(0, 1),
                                                  rotation(0, 2),
                                                  rotation(1, 0),
                                                  rotation(1, 1),
                                                  rotation(1, 2),
                                                  rotation(2, 0),
                                                  rotation(2, 1),
                                                  rotation(2, 2),
                                                  0.f,
                                                  0.f,
                                                  0.f,
                                                  0.f,
                                                  0.f,
                                                  static_cast<float>(cloud.width),
                                                  static_cast<float>(cloud.height),
                                                  0.f,
                                                  0.f};
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                if (binary)
                {
                    if (index == 17 || index == 18)
                        writePlyBinaryNumber(stream, static_cast<std::int32_t>(values[index]));
                    else
                        writePlyBinaryNumber(stream, values[index]);
                }
                else
                {
                    if (index != 0)
                        stream << ' ';
                    stream << values[index];
                }
            }
            if (!binary)
                stream << '\n';
        }
        if (!stream)
            throw std::runtime_error("Failed to write PLY stream");
    }

    inline void writePlyBlob(const std::string& path,
                             const PCLPointCloud2& cloud,
                             const std::vector<Vertices>& polygons,
                             const Eigen::Vector4f& origin,
                             const Eigen::Quaternionf& orientation,
                             bool binary,
                             bool use_camera,
                             unsigned int precision)
    {
        std::ofstream stream(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode{}));
        if (!stream)
        {
            throw std::runtime_error("Cannot open PLY file: " + path);
        }
        writePlyBlob(stream, cloud, polygons, origin, orientation, binary, use_camera, precision);
    }

} // namespace plapoint::io::detail
