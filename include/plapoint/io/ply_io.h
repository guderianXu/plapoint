#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>

#include <plapoint/core/point_cloud.h>

namespace plapoint {
namespace io {

namespace detail {

inline bool isLittleEndian()
{
    int n = 1;
    return (*reinterpret_cast<char*>(&n) == 1);
}

template <typename T>
void swapEndian(T& val)
{
    char* p = reinterpret_cast<char*>(&val);
    for (std::size_t i = 0; i < sizeof(T) / 2; ++i)
        std::swap(p[i], p[sizeof(T) - 1 - i]);
}

} // namespace detail

enum class PlyFormat { ASCII, BinaryLE, BinaryBE };

namespace detail {

enum class PlyVertexPropertyRole { Ignore, X, Y, Z, NX, NY, NZ, Red, Green, Blue, Intensity };

struct PlyScalarProperty
{
    bool isList = false;
    std::string countType;
    std::string type;
    std::string name;
    PlyVertexPropertyRole role = PlyVertexPropertyRole::Ignore;
    int extraScalarIndex = -1;
};

struct PlyElement
{
    std::string name;
    int count = 0;
    std::vector<PlyScalarProperty> properties;
};

template <typename T>
T readBinaryValue(std::istream& stream, bool swapBytes)
{
    T value{};
    if (!stream.read(reinterpret_cast<char*>(&value), sizeof(T)))
    {
        throw std::runtime_error("Unexpected end of PLY binary data");
    }
    if (swapBytes && sizeof(T) > 1)
    {
        swapEndian(value);
    }
    return value;
}

inline double readBinaryScalar(std::istream& stream,
                               const std::string& type,
                               bool swapBytes)
{
    if (type == "char" || type == "int8")
    {
        return static_cast<double>(readBinaryValue<std::int8_t>(stream, false));
    }
    if (type == "uchar" || type == "uint8" || type == "unsigned_char")
    {
        return static_cast<double>(readBinaryValue<std::uint8_t>(stream, false));
    }
    if (type == "short" || type == "int16")
    {
        return static_cast<double>(readBinaryValue<std::int16_t>(stream, swapBytes));
    }
    if (type == "ushort" || type == "uint16" || type == "unsigned_short")
    {
        return static_cast<double>(readBinaryValue<std::uint16_t>(stream, swapBytes));
    }
    if (type == "int" || type == "int32")
    {
        return static_cast<double>(readBinaryValue<std::int32_t>(stream, swapBytes));
    }
    if (type == "uint" || type == "uint32" || type == "unsigned_int")
    {
        return static_cast<double>(readBinaryValue<std::uint32_t>(stream, swapBytes));
    }
    if (type == "float" || type == "float32")
    {
        return static_cast<double>(readBinaryValue<float>(stream, swapBytes));
    }
    if (type == "double" || type == "float64")
    {
        return readBinaryValue<double>(stream, swapBytes);
    }
    throw std::runtime_error("Unsupported PLY vertex property type: " + type);
}

inline bool isPlyIntegerScalarType(const std::string& type)
{
    return type == "char" || type == "int8" ||
           type == "uchar" || type == "uint8" || type == "unsigned_char" ||
           type == "short" || type == "int16" ||
           type == "ushort" || type == "uint16" || type == "unsigned_short" ||
           type == "int" || type == "int32" ||
           type == "uint" || type == "uint32" || type == "unsigned_int";
}

inline std::size_t readBinaryListCount(std::istream& stream,
                                       const std::string& type,
                                       bool swapBytes)
{
    if (!isPlyIntegerScalarType(type))
    {
        throw std::runtime_error("PLY list count type must be an integer type");
    }
    const double count = readBinaryScalar(stream, type, swapBytes);
    if (count < 0.0)
    {
        throw std::runtime_error("PLY list property has a negative item count");
    }
    return static_cast<std::size_t>(count);
}

inline int parseAsciiListCountToken(const std::string& token, const std::string& type)
{
    if (!isPlyIntegerScalarType(type))
    {
        throw std::runtime_error("PLY list count type must be an integer type");
    }
    if (token.empty())
    {
        throw std::runtime_error("PLY ASCII list count is missing");
    }
    char* end = nullptr;
    errno = 0;
    const long long count = std::strtoll(token.c_str(), &end, 10);
    if (errno == ERANGE || end == token.c_str() || *end != '\0')
    {
        throw std::runtime_error("PLY ASCII list count must be an integer token");
    }
    if (count < 0)
    {
        throw std::runtime_error("PLY list property has a negative item count");
    }
    if (count > static_cast<long long>(std::numeric_limits<int>::max()))
    {
        throw std::out_of_range("PLY ASCII list count exceeds int range");
    }
    return static_cast<int>(count);
}

inline int readAsciiListCount(std::istream& stream, const std::string& type)
{
    std::string token;
    if (!(stream >> token))
    {
        throw std::runtime_error("PLY ASCII list property is missing an item count");
    }
    return parseAsciiListCountToken(token, type);
}

inline void skipAsciiList(std::istream& stream, const PlyScalarProperty& prop)
{
    const int count = readAsciiListCount(stream, prop.countType);
    for (int i = 0; i < count; ++i)
    {
        std::string ignored;
        if (!(stream >> ignored))
        {
            throw std::runtime_error("PLY ASCII list property is missing items");
        }
    }
}

inline void skipBinaryList(std::istream& stream,
                           const PlyScalarProperty& prop,
                           bool swapBytes)
{
    const std::size_t count = readBinaryListCount(stream, prop.countType, swapBytes);
    for (std::size_t i = 0; i < count; ++i)
    {
        (void)readBinaryScalar(stream, prop.type, swapBytes);
    }
}

inline PlyVertexPropertyRole vertexPropertyRole(const std::string& name)
{
    if (name == "x") return PlyVertexPropertyRole::X;
    if (name == "y") return PlyVertexPropertyRole::Y;
    if (name == "z") return PlyVertexPropertyRole::Z;
    if (name == "nx") return PlyVertexPropertyRole::NX;
    if (name == "ny") return PlyVertexPropertyRole::NY;
    if (name == "nz") return PlyVertexPropertyRole::NZ;
    if (name == "red") return PlyVertexPropertyRole::Red;
    if (name == "green") return PlyVertexPropertyRole::Green;
    if (name == "blue") return PlyVertexPropertyRole::Blue;
    if (name == "intensity") return PlyVertexPropertyRole::Intensity;
    return PlyVertexPropertyRole::Ignore;
}

inline std::uint8_t plyColorByte(double value)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("PLY color property value must be finite");
    }
    if (value <= 0.0) return 0;
    if (value >= 255.0) return 255;
    return static_cast<std::uint8_t>(std::lround(value));
}

inline bool isPlyUInt16Type(const std::string& type)
{
    return type == "ushort" || type == "uint16" || type == "unsigned_short";
}

inline bool isPlyFloatingScalarType(const std::string& type)
{
    return type == "float" || type == "float32" ||
           type == "double" || type == "float64";
}

inline std::uint8_t plyColorByteForProperty(double value, const std::string& type)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("PLY color property value must be finite");
    }
    if (isPlyFloatingScalarType(type) && value >= 0.0 && value <= 1.0)
    {
        return plyColorByte(value * 255.0);
    }
    if (isPlyUInt16Type(type))
    {
        if (value <= 0.0) return 0;
        if (value >= static_cast<double>(std::numeric_limits<std::uint16_t>::max())) return 255;
        return static_cast<std::uint8_t>(
            (static_cast<unsigned int>(std::lround(value)) + 128u) / 257u);
    }
    return plyColorByte(value);
}

inline std::uint16_t plyIntensityWord(double value)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("PLY intensity property value must be finite");
    }
    if (value <= 0.0) return 0;
    if (value >= static_cast<double>(std::numeric_limits<std::uint16_t>::max()))
    {
        return std::numeric_limits<std::uint16_t>::max();
    }
    return static_cast<std::uint16_t>(std::lround(value));
}

inline std::uint16_t plyIntensityWordForProperty(double value, const std::string& type)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("PLY intensity property value must be finite");
    }
    if (isPlyFloatingScalarType(type) && value >= 0.0 && value <= 1.0)
    {
        return static_cast<std::uint16_t>(
            std::lround(value * static_cast<double>(std::numeric_limits<std::uint16_t>::max())));
    }
    return plyIntensityWord(value);
}

inline bool isFaceVertexIndicesProperty(const std::string& name)
{
    return name == "vertex_indices" || name == "vertex_index";
}

inline int plyFaceIndex(double value)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("PLY face index must be finite");
    }
    const double rounded = std::round(value);
    if (std::abs(value - rounded) > 1e-6)
    {
        throw std::runtime_error("PLY face index must be an integer");
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max()))
    {
        throw std::out_of_range("PLY face index out of int range");
    }
    return static_cast<int>(rounded);
}

inline std::vector<int> readAsciiFaceIndexList(std::istream& stream,
                                               const PlyScalarProperty& prop)
{
    const int count = readAsciiListCount(stream, prop.countType);
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        double value = 0.0;
        if (!(stream >> value))
        {
            throw std::runtime_error("PLY ASCII face list property is missing items");
        }
        indices.push_back(plyFaceIndex(value));
    }
    return indices;
}

inline std::vector<int> readBinaryFaceIndexList(std::istream& stream,
                                               const PlyScalarProperty& prop,
                                               bool swapBytes)
{
    const std::size_t count = readBinaryListCount(stream, prop.countType, swapBytes);
    std::vector<int> indices;
    indices.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        indices.push_back(plyFaceIndex(readBinaryScalar(stream, prop.type, swapBytes)));
    }
    return indices;
}

inline void appendTriangulatedFace(const std::vector<int>& polygon, std::vector<int>& triangles)
{
    if (polygon.empty())
    {
        return;
    }
    if (polygon.size() < 3)
    {
        throw std::runtime_error("PLY face must contain at least 3 vertices");
    }
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
    {
        triangles.push_back(polygon[0]);
        triangles.push_back(polygon[i]);
        triangles.push_back(polygon[i + 1]);
    }
}

template <typename Scalar>
constexpr const char* plyFloatingPointTypeName()
{
    static_assert(std::is_floating_point<Scalar>::value,
                  "PLY writer requires a floating-point point scalar type");
    return std::is_same<Scalar, double>::value ? "double" : "float";
}

[[noreturn]] inline void throwPlyParseError(const std::string& path, const std::string& message)
{
    throw std::runtime_error("PLY parse error in " + path + ": " + message);
}

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readPlyImpl(const std::string& path,
            bool applyPointOffset,
            std::array<double, 3>* pointOffsetOut,
            bool* hasPointOffsetOut)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PLY file: " + path);

    std::string line;
    if (!std::getline(f, line))
    {
        throwPlyParseError(path, "missing magic header");
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply") throw std::runtime_error("Not a PLY file: " + path);

    if (!std::getline(f, line))
    {
        throwPlyParseError(path, "missing format line");
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    PlyFormat fmt = PlyFormat::ASCII;
    if (line.find("format ascii") != std::string::npos) fmt = PlyFormat::ASCII;
    else if (line.find("binary_little_endian") != std::string::npos) fmt = PlyFormat::BinaryLE;
    else if (line.find("binary_big_endian") != std::string::npos) fmt = PlyFormat::BinaryBE;
    else throwPlyParseError(path, "unsupported or missing format line");

    int n_verts = 0;
    std::vector<PlyElement> elements;
    PlyElement* currentElement = nullptr;
    bool saw_vertex_element = false;
    bool has_x = false, has_y = false, has_z = false;
    bool has_nx = false, has_ny = false, has_nz = false;
    bool has_red = false, has_green = false, has_blue = false, has_intensity = false;
    std::vector<std::string> extra_scalar_names;
    std::array<double, 3> pointOffset = {0.0, 0.0, 0.0};
    bool hasPointOffset = false;
    bool saw_end_header = false;

    while (std::getline(f, line))
    {
        // Trim \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header")
        {
            saw_end_header = true;
            break;
        }
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element")
        {
            std::string elem;
            int count = 0;
            if (!(iss >> elem >> count))
            {
                throwPlyParseError(path, "malformed element declaration");
            }
            if (count < 0)
            {
                throwPlyParseError(path, "negative element count for " + elem);
            }
            elements.push_back({elem, count, {}});
            currentElement = &elements.back();
            if (elem == "vertex")
            {
                n_verts = count;
                saw_vertex_element = true;
            }
        }
        else if (token == "comment")
        {
            std::string key;
            iss >> key;
            if (key == "POINT_OFFSET")
            {
                if (iss >> pointOffset[0] >> pointOffset[1] >> pointOffset[2])
                {
                    hasPointOffset = true;
                }
            }
        }
        else if (token == "property")
        {
            if (!currentElement)
            {
                throwPlyParseError(path, "property declared before any element");
            }
            std::string type, name;
            if (!(iss >> type >> name))
            {
                throwPlyParseError(path, "malformed property declaration");
            }
            if (type == "list")
            {
                std::string itemType, listName;
                if (!(iss >> itemType >> listName))
                {
                    throwPlyParseError(path, "malformed list property declaration");
                }
                const std::string countType = name;
                currentElement->properties.push_back(
                    {true, countType, itemType, listName, PlyVertexPropertyRole::Ignore});
                continue;
            }
            const PlyVertexPropertyRole role =
                currentElement->name == "vertex"
                    ? vertexPropertyRole(name)
                    : PlyVertexPropertyRole::Ignore;
            currentElement->properties.push_back({false, {}, type, name, role});
            if (currentElement->name == "vertex")
            {
                if (role == PlyVertexPropertyRole::X) has_x = true;
                if (role == PlyVertexPropertyRole::Y) has_y = true;
                if (role == PlyVertexPropertyRole::Z) has_z = true;
                if (role == PlyVertexPropertyRole::NX) has_nx = true;
                if (role == PlyVertexPropertyRole::NY) has_ny = true;
                if (role == PlyVertexPropertyRole::NZ) has_nz = true;
                if (role == PlyVertexPropertyRole::Red) has_red = true;
                if (role == PlyVertexPropertyRole::Green) has_green = true;
                if (role == PlyVertexPropertyRole::Blue) has_blue = true;
                if (role == PlyVertexPropertyRole::Intensity) has_intensity = true;
                if (role == PlyVertexPropertyRole::Ignore)
                {
                    if (std::find(extra_scalar_names.begin(), extra_scalar_names.end(), name) !=
                        extra_scalar_names.end())
                    {
                        throwPlyParseError(path, "duplicate extra vertex scalar property: " + name);
                    }
                    currentElement->properties.back().extraScalarIndex =
                        static_cast<int>(extra_scalar_names.size());
                    extra_scalar_names.push_back(name);
                }
            }
        }
    }
    if (!saw_end_header)
    {
        throwPlyParseError(path, "missing end_header");
    }
    if (saw_vertex_element && (!has_x || !has_y || !has_z))
    {
        throwPlyParseError(path, "vertex element must declare x, y, and z properties");
    }
    if (pointOffsetOut)
    {
        *pointOffsetOut = pointOffset;
    }
    if (hasPointOffsetOut)
    {
        *hasPointOffsetOut = hasPointOffset;
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(n_verts, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(n_verts, 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(n_verts, 3);
    plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(n_verts, 1);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> extra_scalars(
        n_verts, static_cast<plamatrix::Index>(extra_scalar_names.size()));
    std::vector<int> face_indices;
    bool have_normals = has_nx && has_ny && has_nz;
    bool have_colors = has_intensity || (has_red && has_green && has_blue);

    if (fmt == PlyFormat::ASCII)
    {
        int vertexIndex = 0;
        for (const PlyElement& element : elements)
        {
            for (int i = 0; i < element.count; ++i)
            {
                if (!std::getline(f, line))
                {
                    throwPlyParseError(path, "truncated ASCII element rows");
                }
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (element.name != "vertex" && element.name != "face")
                {
                    continue;
                }
                std::istringstream iss(line);
                if (element.name == "face")
                {
                    for (const auto& prop : element.properties)
                    {
                        if (prop.isList)
                        {
                            if (isFaceVertexIndicesProperty(prop.name))
                            {
                                appendTriangulatedFace(readAsciiFaceIndexList(iss, prop), face_indices);
                            }
                            else
                            {
                                skipAsciiList(iss, prop);
                            }
                            continue;
                        }
                        double ignored = 0.0;
                        if (!(iss >> ignored))
                        {
                            throwPlyParseError(path, "truncated ASCII face row");
                        }
                    }
                    continue;
                }
                std::array<std::uint8_t, 3> rgb = {0, 0, 0};
                std::uint16_t intensity = 0;
                std::uint8_t intensity_color = 0;
                for (const auto& prop : element.properties)
                {
                    if (prop.isList)
                    {
                        skipAsciiList(iss, prop);
                        continue;
                    }
                    double val = 0.0;
                    if (!(iss >> val))
                    {
                        throwPlyParseError(path, "truncated ASCII vertex row");
                    }
                    switch (prop.role)
                    {
                    case PlyVertexPropertyRole::X:
                        pts(vertexIndex, 0) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[0] : 0.0));
                        break;
                    case PlyVertexPropertyRole::Y:
                        pts(vertexIndex, 1) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[1] : 0.0));
                        break;
                    case PlyVertexPropertyRole::Z:
                        pts(vertexIndex, 2) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[2] : 0.0));
                        break;
                    case PlyVertexPropertyRole::NX:
                        nrm(vertexIndex, 0) = static_cast<Scalar>(val);
                        break;
                    case PlyVertexPropertyRole::NY:
                        nrm(vertexIndex, 1) = static_cast<Scalar>(val);
                        break;
                    case PlyVertexPropertyRole::NZ:
                        nrm(vertexIndex, 2) = static_cast<Scalar>(val);
                        break;
                    case PlyVertexPropertyRole::Red:
                        rgb[0] = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Green:
                        rgb[1] = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Blue:
                        rgb[2] = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Intensity:
                        intensity = plyIntensityWordForProperty(val, prop.type);
                        intensity_color = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Ignore:
                        if (prop.extraScalarIndex >= 0)
                        {
                            extra_scalars(vertexIndex, prop.extraScalarIndex) = static_cast<Scalar>(val);
                        }
                        break;
                    }
                }
                if (have_colors)
                {
                    if (has_red && has_green && has_blue)
                    {
                        colors(vertexIndex, 0) = rgb[0];
                        colors(vertexIndex, 1) = rgb[1];
                        colors(vertexIndex, 2) = rgb[2];
                    }
                    else
                    {
                        colors(vertexIndex, 0) = intensity_color;
                        colors(vertexIndex, 1) = intensity_color;
                        colors(vertexIndex, 2) = intensity_color;
                    }
                }
                if (has_intensity)
                {
                    intensities(vertexIndex, 0) = intensity;
                }
                ++vertexIndex;
            }
        }
    }
    else
    {
        bool swap_bytes = (fmt == PlyFormat::BinaryBE) == detail::isLittleEndian();
        int vertexIndex = 0;
        for (const PlyElement& element : elements)
        {
            for (int i = 0; i < element.count; ++i)
            {
                std::array<std::uint8_t, 3> rgb = {0, 0, 0};
                std::uint16_t intensity = 0;
                std::uint8_t intensity_color = 0;
                std::vector<int> face_polygon;
                for (const auto& prop : element.properties)
                {
                    if (prop.isList)
                    {
                        if (element.name == "face" && isFaceVertexIndicesProperty(prop.name))
                        {
                            face_polygon = readBinaryFaceIndexList(f, prop, swap_bytes);
                        }
                        else
                        {
                            skipBinaryList(f, prop, swap_bytes);
                        }
                        continue;
                    }
                    const double val = readBinaryScalar(f, prop.type, swap_bytes);
                    if (element.name != "vertex")
                    {
                        continue;
                    }
                    switch (prop.role)
                    {
                    case PlyVertexPropertyRole::X:
                        pts(vertexIndex, 0) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[0] : 0.0));
                        break;
                    case PlyVertexPropertyRole::Y:
                        pts(vertexIndex, 1) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[1] : 0.0));
                        break;
                    case PlyVertexPropertyRole::Z:
                        pts(vertexIndex, 2) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[2] : 0.0));
                        break;
                    case PlyVertexPropertyRole::NX:
                        nrm(vertexIndex, 0) = static_cast<Scalar>(val);
                        break;
                    case PlyVertexPropertyRole::NY:
                        nrm(vertexIndex, 1) = static_cast<Scalar>(val);
                        break;
                    case PlyVertexPropertyRole::NZ:
                        nrm(vertexIndex, 2) = static_cast<Scalar>(val);
                        break;
                    case PlyVertexPropertyRole::Red:
                        rgb[0] = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Green:
                        rgb[1] = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Blue:
                        rgb[2] = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Intensity:
                        intensity = plyIntensityWordForProperty(val, prop.type);
                        intensity_color = plyColorByteForProperty(val, prop.type);
                        break;
                    case PlyVertexPropertyRole::Ignore:
                        if (prop.extraScalarIndex >= 0)
                        {
                            extra_scalars(vertexIndex, prop.extraScalarIndex) = static_cast<Scalar>(val);
                        }
                        break;
                    }
                }
                if (element.name == "face")
                {
                    appendTriangulatedFace(face_polygon, face_indices);
                }
                if (element.name == "vertex")
                {
                    if (have_colors)
                    {
                        if (has_red && has_green && has_blue)
                        {
                            colors(vertexIndex, 0) = rgb[0];
                            colors(vertexIndex, 1) = rgb[1];
                            colors(vertexIndex, 2) = rgb[2];
                        }
                        else
                        {
                            colors(vertexIndex, 0) = intensity_color;
                            colors(vertexIndex, 1) = intensity_color;
                            colors(vertexIndex, 2) = intensity_color;
                        }
                    }
                    if (has_intensity)
                    {
                        intensities(vertexIndex, 0) = intensity;
                    }
                    ++vertexIndex;
                }
            }
        }
    }

    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
    if (have_normals) cloud->setNormals(std::move(nrm));
    if (have_colors) cloud->setColors(std::move(colors));
    if (has_intensity) cloud->setIntensities(std::move(intensities));
    if (!extra_scalar_names.empty()) cloud->setScalarFields(std::move(extra_scalar_names), std::move(extra_scalars));
    if (!face_indices.empty())
    {
        const auto face_count = static_cast<plamatrix::Index>(face_indices.size() / 3);
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(face_count, 3);
        for (plamatrix::Index i = 0; i < face_count; ++i)
        {
            faces(i, 0) = face_indices[static_cast<std::size_t>(i * 3)];
            faces(i, 1) = face_indices[static_cast<std::size_t>(i * 3 + 1)];
            faces(i, 2) = face_indices[static_cast<std::size_t>(i * 3 + 2)];
        }
        cloud->setFaces(std::move(faces));
    }
    return cloud;
}

} // namespace detail

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readPly(const std::string& path)
{
    return detail::readPlyImpl<Scalar>(path, true, nullptr, nullptr);
}

// Reads vertex coordinates as stored in the PLY payload without applying
// `comment POINT_OFFSET`; returns that offset separately when requested.
template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readPlyLocal(const std::string& path,
             std::array<double, 3>* pointOffsetOut = nullptr,
             bool* hasPointOffsetOut = nullptr)
{
    return detail::readPlyImpl<Scalar>(path, false, pointOffsetOut, hasPointOffsetOut);
}

template <typename Scalar>
void writePly(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud,
              PlyFormat fmt = PlyFormat::ASCII)
{
    std::ios::openmode mode = std::ios::out;
    if (fmt != PlyFormat::ASCII) mode |= std::ios::binary;
    std::ofstream f(path, mode);
    if (!f) throw std::runtime_error("Cannot write PLY file: " + path);

    bool with_normals = cloud.hasNormals();
    bool with_colors = cloud.hasColors();
    bool with_intensities = cloud.hasIntensities();
    bool with_faces = cloud.hasFaces();
    bool with_scalar_fields = cloud.hasScalarFields();
    const char* scalar_type = detail::plyFloatingPointTypeName<Scalar>();

    f << "ply\n";
    if (fmt == PlyFormat::ASCII)
        f << "format ascii 1.0\n";
    else if (fmt == PlyFormat::BinaryLE)
        f << "format binary_little_endian 1.0\n";
    else
        f << "format binary_big_endian 1.0\n";

    f << "element vertex " << cloud.size() << "\n";
    f << "property " << scalar_type << " x\n"
      << "property " << scalar_type << " y\n"
      << "property " << scalar_type << " z\n";
    if (with_colors)
        f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    if (with_intensities)
        f << "property ushort intensity\n";
    if (with_normals)
        f << "property " << scalar_type << " nx\n"
          << "property " << scalar_type << " ny\n"
          << "property " << scalar_type << " nz\n";
    if (with_scalar_fields)
    {
        for (const auto& name : cloud.scalarFieldNames())
        {
            f << "property " << scalar_type << " " << name << "\n";
        }
    }
    if (with_faces)
        f << "element face " << cloud.faces()->rows() << "\n"
          << "property list uchar int vertex_indices\n";
    f << "end_header\n";

    if (fmt == PlyFormat::ASCII)
    {
        f << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
        for (std::size_t i = 0; i < cloud.size(); ++i)
        {
            f << cloud.points().getValue(static_cast<plamatrix::Index>(i), 0) << " "
              << cloud.points().getValue(static_cast<plamatrix::Index>(i), 1) << " "
              << cloud.points().getValue(static_cast<plamatrix::Index>(i), 2);
            if (with_colors)
            {
                auto* c = cloud.colors();
                f << " " << static_cast<int>(c->getValue(static_cast<plamatrix::Index>(i), 0))
                  << " " << static_cast<int>(c->getValue(static_cast<plamatrix::Index>(i), 1))
                  << " " << static_cast<int>(c->getValue(static_cast<plamatrix::Index>(i), 2));
            }
            if (with_intensities)
            {
                auto* intensities = cloud.intensities();
                f << " " << intensities->getValue(static_cast<plamatrix::Index>(i), 0);
            }
            if (with_normals)
            {
                auto* n = cloud.normals();
                f << " " << n->getValue(static_cast<plamatrix::Index>(i), 0)
                  << " " << n->getValue(static_cast<plamatrix::Index>(i), 1)
                  << " " << n->getValue(static_cast<plamatrix::Index>(i), 2);
            }
            if (with_scalar_fields)
            {
                auto* fields = cloud.scalarFields();
                for (plamatrix::Index col = 0; col < fields->cols(); ++col)
                {
                    f << " " << fields->getValue(static_cast<plamatrix::Index>(i), col);
                }
            }
            f << "\n";
        }
        if (with_faces)
        {
            auto* faces = cloud.faces();
            for (plamatrix::Index i = 0; i < faces->rows(); ++i)
            {
                f << "3 "
                  << faces->getValue(i, 0) << " "
                  << faces->getValue(i, 1) << " "
                  << faces->getValue(i, 2) << "\n";
            }
        }
    }
    else
    {
        bool swap_bytes = (fmt == PlyFormat::BinaryBE) == detail::isLittleEndian();
        auto write_scalar = [&](Scalar value) {
            if (swap_bytes) detail::swapEndian(value);
            f.write(reinterpret_cast<const char*>(&value), sizeof(Scalar));
        };
        auto write_int32 = [&](std::int32_t value) {
            if (swap_bytes) detail::swapEndian(value);
            f.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        auto write_uint16 = [&](std::uint16_t value) {
            if (swap_bytes) detail::swapEndian(value);
            f.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        for (std::size_t i = 0; i < cloud.size(); ++i)
        {
            write_scalar(cloud.points().getValue(static_cast<plamatrix::Index>(i), 0));
            write_scalar(cloud.points().getValue(static_cast<plamatrix::Index>(i), 1));
            write_scalar(cloud.points().getValue(static_cast<plamatrix::Index>(i), 2));
            if (with_colors)
            {
                auto* c = cloud.colors();
                const std::uint8_t color[3] = {
                    c->getValue(static_cast<plamatrix::Index>(i), 0),
                    c->getValue(static_cast<plamatrix::Index>(i), 1),
                    c->getValue(static_cast<plamatrix::Index>(i), 2),
                };
                f.write(reinterpret_cast<const char*>(color), sizeof(color));
            }
            if (with_intensities)
            {
                auto* intensities = cloud.intensities();
                write_uint16(intensities->getValue(static_cast<plamatrix::Index>(i), 0));
            }
            if (with_normals)
            {
                auto* n = cloud.normals();
                write_scalar(n->getValue(static_cast<plamatrix::Index>(i), 0));
                write_scalar(n->getValue(static_cast<plamatrix::Index>(i), 1));
                write_scalar(n->getValue(static_cast<plamatrix::Index>(i), 2));
            }
            if (with_scalar_fields)
            {
                auto* fields = cloud.scalarFields();
                for (plamatrix::Index col = 0; col < fields->cols(); ++col)
                {
                    write_scalar(fields->getValue(static_cast<plamatrix::Index>(i), col));
                }
            }
        }
        if (with_faces)
        {
            auto* faces = cloud.faces();
            for (plamatrix::Index i = 0; i < faces->rows(); ++i)
            {
                const std::uint8_t count = 3;
                f.write(reinterpret_cast<const char*>(&count), sizeof(count));
                write_int32(static_cast<std::int32_t>(faces->getValue(i, 0)));
                write_int32(static_cast<std::int32_t>(faces->getValue(i, 1)));
                write_int32(static_cast<std::int32_t>(faces->getValue(i, 2)));
            }
        }
    }
}

} // namespace io
} // namespace plapoint
