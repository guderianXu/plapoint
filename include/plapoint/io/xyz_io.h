#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
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

} // namespace detail

/// Read XYZ file (one point per line: x y z)
template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readXyz(const std::string& path)
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
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Scalar x, y, z;
        if (iss >> x >> y >> z)
        {
            buf.push_back(x);
            buf.push_back(y);
            buf.push_back(z);
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            if (iss >> r >> g >> b)
            {
                color_buf.push_back(detail::xyzColorByte(r));
                color_buf.push_back(detail::xyzColorByte(g));
                color_buf.push_back(detail::xyzColorByte(b));
                saw_color = true;
            }
            else
            {
                saw_uncolored = true;
            }
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
