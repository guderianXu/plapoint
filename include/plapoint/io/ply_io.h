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

struct PlyScalarProperty
{
    std::string type;
    std::string name;
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
    std::string currentElement;
    std::vector<PlyScalarProperty> props;
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
            currentElement = elem;
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
            std::string type, name;
            iss >> type >> name;
            if (currentElement != "vertex")
            {
                continue;
            }
            if (type == "list")
            {
                throw std::runtime_error("Unsupported PLY list property on vertex element: " + line);
            }
            props.push_back({type, name});
            if (name == "nx") has_nx = true;
            if (name == "ny") has_ny = true;
            if (name == "nz") has_nz = true;
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
        for (int i = 0; i < n_verts; ++i)
        {
            std::getline(f, line);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream iss(line);
            for (const auto& prop : props)
            {
                double val = 0.0;
                iss >> val;
                const auto& p = prop.name;
                if (p == "x")      pts(i, 0) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[0] : 0.0));
                else if (p == "y") pts(i, 1) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[1] : 0.0));
                else if (p == "z") pts(i, 2) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[2] : 0.0));
                else if (p == "nx") nrm(i, 0) = static_cast<Scalar>(val);
                else if (p == "ny") nrm(i, 1) = static_cast<Scalar>(val);
                else if (p == "nz") nrm(i, 2) = static_cast<Scalar>(val);
            }
        }
    }
    else
    {
        bool swap_bytes = (fmt == PlyFormat::BinaryBE) == detail::isLittleEndian();
        for (int i = 0; i < n_verts; ++i)
        {
            for (const auto& prop : props)
            {
                const double val = readBinaryScalar(f, prop.type, swap_bytes);
                const auto& p = prop.name;
                if (p == "x")      pts(i, 0) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[0] : 0.0));
                else if (p == "y") pts(i, 1) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[1] : 0.0));
                else if (p == "z") pts(i, 2) = static_cast<Scalar>(val + (applyPointOffset ? pointOffset[2] : 0.0));
                else if (p == "nx") nrm(i, 0) = static_cast<Scalar>(val);
                else if (p == "ny") nrm(i, 1) = static_cast<Scalar>(val);
                else if (p == "nz") nrm(i, 2) = static_cast<Scalar>(val);
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
