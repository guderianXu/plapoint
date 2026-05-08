#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plapoint {
namespace io {

/// Read XYZ file (one point per line: x y z)
template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readXyz(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open XYZ file: " + path);

    // First pass: count points
    std::vector<Scalar> buf;
    buf.reserve(100000 * 3);
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

    return std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
}

/// Write XYZ file
template <typename Scalar>
void writeXyz(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write XYZ file: " + path);

    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        f << cloud.points().getValue(static_cast<plamatrix::Index>(i), 0) << " "
          << cloud.points().getValue(static_cast<plamatrix::Index>(i), 1) << " "
          << cloud.points().getValue(static_cast<plamatrix::Index>(i), 2) << "\n";
    }
}

} // namespace io
} // namespace plapoint
