#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

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

enum class PlyVertexPropertyRole { Ignore, X, Y, Z, NX, NY, NZ };

struct PlyScalarProperty
{
    bool isList = false;
    std::string countType;
    std::string type;
    std::string name;
    PlyVertexPropertyRole role = PlyVertexPropertyRole::Ignore;
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
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
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

inline std::size_t readBinaryListCount(std::istream& stream,
                                       const std::string& type,
                                       bool swapBytes)
{
    const double count = readBinaryScalar(stream, type, swapBytes);
    if (count < 0.0)
    {
        throw std::runtime_error("PLY list property has a negative item count");
    }
    return static_cast<std::size_t>(count);
}

inline void skipAsciiList(std::istream& stream)
{
    int count = 0;
    stream >> count;
    for (int i = 0; i < count; ++i)
    {
        std::string ignored;
        stream >> ignored;
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
    return PlyVertexPropertyRole::Ignore;
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
    std::getline(f, line);
    if (line != "ply") throw std::runtime_error("Not a PLY file: " + path);

    std::getline(f, line);
    PlyFormat fmt = PlyFormat::ASCII;
    if (line.find("binary_little_endian") != std::string::npos) fmt = PlyFormat::BinaryLE;
    else if (line.find("binary_big_endian") != std::string::npos) fmt = PlyFormat::BinaryBE;

    int n_verts = 0;
    std::vector<PlyElement> elements;
    PlyElement* currentElement = nullptr;
    bool has_nx = false, has_ny = false, has_nz = false;
    std::array<double, 3> pointOffset = {0.0, 0.0, 0.0};
    bool hasPointOffset = false;

    while (std::getline(f, line))
    {
        // Trim \r
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") break;
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element")
        {
            std::string elem;
            int count = 0;
            iss >> elem >> count;
            elements.push_back({elem, count, {}});
            currentElement = &elements.back();
            if (elem == "vertex")
            {
                n_verts = count;
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
                continue;
            }
            std::string type, name;
            iss >> type >> name;
            if (type == "list")
            {
                std::string itemType, listName;
                iss >> itemType >> listName;
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
                if (role == PlyVertexPropertyRole::NX) has_nx = true;
                if (role == PlyVertexPropertyRole::NY) has_ny = true;
                if (role == PlyVertexPropertyRole::NZ) has_nz = true;
            }
        }
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
    bool have_normals = has_nx && has_ny && has_nz;

    if (fmt == PlyFormat::ASCII)
    {
        int vertexIndex = 0;
        for (const PlyElement& element : elements)
        {
            for (int i = 0; i < element.count; ++i)
            {
                std::getline(f, line);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (element.name != "vertex")
                {
                    continue;
                }
                std::istringstream iss(line);
                for (const auto& prop : element.properties)
                {
                    if (prop.isList)
                    {
                        skipAsciiList(iss);
                        continue;
                    }
                    double val = 0.0;
                    iss >> val;
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
                    case PlyVertexPropertyRole::Ignore:
                        break;
                    }
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
                for (const auto& prop : element.properties)
                {
                    if (prop.isList)
                    {
                        skipBinaryList(f, prop, swap_bytes);
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
                    case PlyVertexPropertyRole::Ignore:
                        break;
                    }
                }
                if (element.name == "vertex")
                {
                    ++vertexIndex;
                }
            }
        }
    }

    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
    if (have_normals) cloud->setNormals(std::move(nrm));
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

    f << "ply\n";
    if (fmt == PlyFormat::ASCII)
        f << "format ascii 1.0\n";
    else if (fmt == PlyFormat::BinaryLE)
        f << "format binary_little_endian 1.0\n";
    else
        f << "format binary_big_endian 1.0\n";

    f << "element vertex " << cloud.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    if (with_normals)
        f << "property float nx\nproperty float ny\nproperty float nz\n";
    f << "end_header\n";

    if (fmt == PlyFormat::ASCII)
    {
        for (std::size_t i = 0; i < cloud.size(); ++i)
        {
            f << cloud.points().getValue(static_cast<plamatrix::Index>(i), 0) << " "
              << cloud.points().getValue(static_cast<plamatrix::Index>(i), 1) << " "
              << cloud.points().getValue(static_cast<plamatrix::Index>(i), 2);
            if (with_normals)
            {
                auto* n = cloud.normals();
                f << " " << n->getValue(static_cast<plamatrix::Index>(i), 0)
                  << " " << n->getValue(static_cast<plamatrix::Index>(i), 1)
                  << " " << n->getValue(static_cast<plamatrix::Index>(i), 2);
            }
            f << "\n";
        }
    }
    else
    {
        bool swap_bytes = (fmt == PlyFormat::BinaryBE) == detail::isLittleEndian();
        for (std::size_t i = 0; i < cloud.size(); ++i)
        {
            float vx = static_cast<float>(cloud.points().getValue(static_cast<plamatrix::Index>(i), 0));
            float vy = static_cast<float>(cloud.points().getValue(static_cast<plamatrix::Index>(i), 1));
            float vz = static_cast<float>(cloud.points().getValue(static_cast<plamatrix::Index>(i), 2));
            if (swap_bytes) { detail::swapEndian(vx); detail::swapEndian(vy); detail::swapEndian(vz); }
            f.write(reinterpret_cast<const char*>(&vx), 4);
            f.write(reinterpret_cast<const char*>(&vy), 4);
            f.write(reinterpret_cast<const char*>(&vz), 4);
            if (with_normals)
            {
                auto* n = cloud.normals();
                float nx = static_cast<float>(n->getValue(static_cast<plamatrix::Index>(i), 0));
                float ny = static_cast<float>(n->getValue(static_cast<plamatrix::Index>(i), 1));
                float nz = static_cast<float>(n->getValue(static_cast<plamatrix::Index>(i), 2));
                if (swap_bytes) { detail::swapEndian(nx); detail::swapEndian(ny); detail::swapEndian(nz); }
                f.write(reinterpret_cast<const char*>(&nx), 4);
                f.write(reinterpret_cast<const char*>(&ny), 4);
                f.write(reinterpret_cast<const char*>(&nz), 4);
            }
        }
    }
}

} // namespace io
} // namespace plapoint
