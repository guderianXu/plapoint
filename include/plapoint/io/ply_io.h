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

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readPly(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open PLY file: " + path);

    std::string line;
    std::getline(f, line);
    if (line != "ply") throw std::runtime_error("Not a PLY file: " + path);

    std::getline(f, line);

    int n_verts = 0;
    std::vector<std::string> props;
    bool has_nx = false, has_ny = false, has_nz = false;

    while (std::getline(f, line))
    {
        if (line == "end_header") break;
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "element")
        {
            std::string elem;
            iss >> elem >> n_verts;
        }
        else if (token == "property")
        {
            std::string type, name;
            iss >> type >> name;
            props.push_back(name);
            if (name == "nx") has_nx = true;
            if (name == "ny") has_ny = true;
            if (name == "nz") has_nz = true;
        }
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(n_verts, 3);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(n_verts, 3);
    bool have_normals = has_nx && has_ny && has_nz;

    for (int i = 0; i < n_verts; ++i)
    {
        std::getline(f, line);
        std::istringstream iss(line);
        Scalar val;
        for (const auto& p : props)
        {
            iss >> val;
            if (p == "x")      pts(i, 0) = val;
            else if (p == "y") pts(i, 1) = val;
            else if (p == "z") pts(i, 2) = val;
            else if (p == "nx") nrm(i, 0) = val;
            else if (p == "ny") nrm(i, 1) = val;
            else if (p == "nz") nrm(i, 2) = val;
        }
    }

    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(std::move(pts));
    if (have_normals) cloud->setNormals(std::move(nrm));
    return cloud;
}

template <typename Scalar>
void writePly(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write PLY file: " + path);

    bool with_normals = cloud.hasNormals();

    f << "ply\nformat ascii 1.0\nelement vertex " << cloud.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    if (with_normals)
        f << "property float nx\nproperty float ny\nproperty float nz\n";
    f << "end_header\n";

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

} // namespace io
} // namespace plapoint
