#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plapoint {
namespace io {

enum class XyzReadMode
{
    Strict,
    Permissive
};

namespace detail {

inline std::uint8_t xyzColorByte(double value)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("XYZ color value must be finite");
    }
    if (value <= 0.0) return 0;
    if (value >= 255.0) return 255;
    return static_cast<std::uint8_t>(std::lround(value));
}

inline std::string trimAsciiWhitespace(const std::string& value)
{
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0)
    {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])) != 0)
    {
        --end;
    }
    return value.substr(begin, end - begin);
}

inline std::runtime_error xyzParseError(
    const std::string& path,
    int line_number,
    const std::string& reason)
{
    std::ostringstream message;
    message << "Invalid XYZ file: " << path << " line " << line_number << ": " << reason;
    return std::runtime_error(message.str());
}

template <typename Number>
inline bool parseFiniteNumberToken(const std::string& token, Number& value)
{
    std::istringstream token_stream(token);
    Number parsed{};
    if (!(token_stream >> parsed))
    {
        return false;
    }
    token_stream >> std::ws;
    if (!token_stream.eof() || !std::isfinite(static_cast<double>(parsed)))
    {
        return false;
    }
    value = parsed;
    return true;
}

} // namespace detail

/// Read XYZ file (one point per line: x y z)
template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readXyz(const std::string& path, XyzReadMode mode = XyzReadMode::Strict)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open XYZ file: " + path);

    // First pass: count points
    std::vector<Scalar> buf;
    std::vector<std::uint8_t> color_buf;
    buf.reserve(100000 * 3);
    color_buf.reserve(100000 * 3);
    bool saw_color = false;
    bool saw_uncolored = false;
    std::string line;
    int line_number = 0;
    while (std::getline(f, line))
    {
        ++line_number;
        const std::string trimmed = detail::trimAsciiWhitespace(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        std::istringstream iss(trimmed);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token)
        {
            tokens.push_back(token);
        }

        Scalar x, y, z;
        const bool has_xyz = tokens.size() >= 3 &&
            detail::parseFiniteNumberToken(tokens[0], x) &&
            detail::parseFiniteNumberToken(tokens[1], y) &&
            detail::parseFiniteNumberToken(tokens[2], z);
        if (!has_xyz)
        {
            if (mode == XyzReadMode::Strict)
            {
                throw detail::xyzParseError(path, line_number, "expected finite x y z coordinates");
            }
            continue;
        }

        if (mode == XyzReadMode::Strict && tokens.size() != 3 && tokens.size() != 6)
        {
            throw detail::xyzParseError(path, line_number, "expected either x y z or x y z r g b columns");
        }

        buf.push_back(x);
        buf.push_back(y);
        buf.push_back(z);
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        const bool has_color = tokens.size() >= 6 &&
            detail::parseFiniteNumberToken(tokens[3], r) &&
            detail::parseFiniteNumberToken(tokens[4], g) &&
            detail::parseFiniteNumberToken(tokens[5], b);
        if (has_color)
        {
            color_buf.push_back(detail::xyzColorByte(r));
            color_buf.push_back(detail::xyzColorByte(g));
            color_buf.push_back(detail::xyzColorByte(b));
            saw_color = true;
        }
        else
        {
            if (mode == XyzReadMode::Strict && tokens.size() == 6)
            {
                throw detail::xyzParseError(path, line_number, "expected finite r g b color values");
            }
            saw_uncolored = true;
        }
    }

    int n = static_cast<int>(buf.size()) / 3;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(n, 3);
    for (int i = 0; i < n; ++i)
    {
        pts(i, 0) = buf[static_cast<std::size_t>(i * 3)];
        pts(i, 1) = buf[static_cast<std::size_t>(i * 3 + 1)];
        pts(i, 2) = buf[static_cast<std::size_t>(i * 3 + 2)];
    }

    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
    if (saw_color && !saw_uncolored &&
        color_buf.size() == static_cast<std::size_t>(n * 3))
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(n, 3);
        for (int i = 0; i < n; ++i)
        {
            colors(i, 0) = color_buf[static_cast<std::size_t>(i * 3)];
            colors(i, 1) = color_buf[static_cast<std::size_t>(i * 3 + 1)];
            colors(i, 2) = color_buf[static_cast<std::size_t>(i * 3 + 2)];
        }
        cloud->setColors(std::move(colors));
    }

    return cloud;
}

/// Write XYZ file
template <typename Scalar>
void writeXyz(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write XYZ file: " + path);
    f << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
    const bool with_colors = cloud.hasColors();

    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        f << cloud.points().getValue(static_cast<plamatrix::Index>(i), 0) << " "
          << cloud.points().getValue(static_cast<plamatrix::Index>(i), 1) << " "
          << cloud.points().getValue(static_cast<plamatrix::Index>(i), 2);
        if (with_colors)
        {
            const auto* colors = cloud.colors();
            f << " " << static_cast<int>(colors->getValue(static_cast<plamatrix::Index>(i), 0))
              << " " << static_cast<int>(colors->getValue(static_cast<plamatrix::Index>(i), 1))
              << " " << static_cast<int>(colors->getValue(static_cast<plamatrix::Index>(i), 2));
        }
        f << "\n";
    }
}

} // namespace io
} // namespace plapoint
