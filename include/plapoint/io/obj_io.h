#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

namespace plapoint {
namespace io {

namespace detail {

struct ObjVertexIndices { int v = -1; int t = -1; int n = -1; };

inline ObjVertexIndices parseFaceVertex(const std::string& s)
{
    ObjVertexIndices idx;
    size_t slash1 = s.find('/');
    if (slash1 == std::string::npos)
    {
        idx.v = std::stoi(s) - 1;
        return idx;
    }
    idx.v = std::stoi(s.substr(0, slash1)) - 1;
    size_t slash2 = s.find('/', slash1 + 1);
    if (slash2 == std::string::npos || slash2 == slash1 + 1)
    {
        return idx;
    }
    if (slash2 > slash1 + 1)
    {
        idx.t = std::stoi(s.substr(slash1 + 1, slash2 - slash1 - 1)) - 1;
    }
    if (slash2 + 1 < s.size())
    {
        idx.n = std::stoi(s.substr(slash2 + 1)) - 1;
    }
    return idx;
}

} // namespace detail

template <typename Scalar>
std::shared_ptr<PointCloud<Scalar, plamatrix::Device::CPU>>
readObj(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
    {
        throw std::runtime_error("Cannot open OBJ file: " + path);
    }

    std::vector<Scalar> vx, vy, vz;
    std::vector<Scalar> nx, ny, nz;
    std::vector<Scalar> tx, ty;
    std::vector<std::vector<int>> face_verts;
    std::vector<std::vector<int>> face_tex;
    std::string mtlLib;

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v")
        {
            Scalar x, y, z;
            iss >> x >> y >> z;
            vx.push_back(x);
            vy.push_back(y);
            vz.push_back(z);
        }
        else if (token == "vn")
        {
            Scalar x, y, z;
            iss >> x >> y >> z;
            nx.push_back(x);
            ny.push_back(y);
            nz.push_back(z);
        }
        else if (token == "vt")
        {
            Scalar u, v;
            iss >> u >> v;
            tx.push_back(u);
            ty.push_back(v);
        }
        else if (token == "f")
        {
            std::vector<int> fv, ft;
            std::string s;
            while (iss >> s)
            {
                auto idx = detail::parseFaceVertex(s);
                fv.push_back(idx.v);
                if (idx.t >= 0)
                {
                    ft.push_back(idx.t);
                }
            }
            face_verts.push_back(fv);
            if (!ft.empty())
            {
                face_tex.push_back(ft);
            }
        }
        else if (token == "mtllib")
        {
            iss >> mtlLib;
        }
    }

    size_t n = vx.size();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
        static_cast<plamatrix::Index>(n), 3);
    for (size_t i = 0; i < n; ++i)
    {
        pts(static_cast<plamatrix::Index>(i), 0) = vx[i];
        pts(static_cast<plamatrix::Index>(i), 1) = vy[i];
        pts(static_cast<plamatrix::Index>(i), 2) = vz[i];
    }
    auto cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(
        std::move(pts));

    if (!nx.empty() && nx.size() == n)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(
            static_cast<plamatrix::Index>(n), 3);
        for (size_t i = 0; i < n; ++i)
        {
            nrm(static_cast<plamatrix::Index>(i), 0) = nx[i];
            nrm(static_cast<plamatrix::Index>(i), 1) = ny[i];
            nrm(static_cast<plamatrix::Index>(i), 2) = nz[i];
        }
        cloud->setNormals(std::move(nrm));
    }

    if (!tx.empty() && tx.size() == n)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> tex(
            static_cast<plamatrix::Index>(n), 2);
        for (size_t i = 0; i < n; ++i)
        {
            tex(static_cast<plamatrix::Index>(i), 0) = tx[i];
            tex(static_cast<plamatrix::Index>(i), 1) = ty[i];
        }
        cloud->setTextureCoords(std::move(tex));
    }

    if (!face_verts.empty())
    {
        plamatrix::Index nf = static_cast<plamatrix::Index>(face_verts.size());
        plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(nf, 3);
        for (plamatrix::Index fi = 0; fi < nf; ++fi)
        {
            for (int c = 0; c < 3 && c < static_cast<int>(face_verts[fi].size()); ++c)
            {
                faces(fi, c) = face_verts[fi][c];
            }
        }
        cloud->setFaces(std::move(faces));

        if (!face_tex.empty() && face_tex.size() == face_verts.size())
        {
            plamatrix::DenseMatrix<int, plamatrix::Device::CPU> ft(nf, 3);
            for (plamatrix::Index fi = 0; fi < nf; ++fi)
            {
                for (int c = 0; c < 3 && c < static_cast<int>(face_tex[fi].size()); ++c)
                {
                    ft(fi, c) = face_tex[fi][c];
                }
            }
            cloud->setFaceTextureIndices(std::move(ft));
        }
    }

    if (!mtlLib.empty())
    {
        cloud->setMaterialLibraryFile(mtlLib);
    }

    return cloud;
}

template <typename Scalar>
void writeObj(const std::string& path,
              const PointCloud<Scalar, plamatrix::Device::CPU>& cloud)
{
    std::ofstream f(path);
    if (!f)
    {
        throw std::runtime_error("Cannot write OBJ file: " + path);
    }

    const auto& mtlLib = cloud.materialLibraryFile();
    if (!mtlLib.empty())
    {
        f << "mtllib " << mtlLib << "\n";
    }

    for (size_t i = 0; i < cloud.size(); ++i)
    {
        f << "v " << cloud.points()(static_cast<plamatrix::Index>(i), 0)
          << " " << cloud.points()(static_cast<plamatrix::Index>(i), 1)
          << " " << cloud.points()(static_cast<plamatrix::Index>(i), 2) << "\n";
    }

    if (cloud.hasNormals())
    {
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            f << "vn " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 0)
              << " " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 1)
              << " " << cloud.normals()->getValue(static_cast<plamatrix::Index>(i), 2) << "\n";
        }
    }

    if (cloud.hasTextureCoords())
    {
        for (size_t i = 0; i < cloud.size(); ++i)
        {
            f << "vt " << cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(i), 0)
              << " " << cloud.textureCoords()->getValue(static_cast<plamatrix::Index>(i), 1) << "\n";
        }
    }

    if (cloud.hasFaces())
    {
        bool use_tex = cloud.hasFaceTextureIndices();
        bool use_nrm = cloud.hasNormals();
        for (plamatrix::Index fi = 0; fi < cloud.faces()->rows(); ++fi)
        {
            f << "f";
            for (int c = 0; c < 3; ++c)
            {
                int vi = cloud.faces()->getValue(fi, c) + 1; // OBJ 1-indexed
                f << " " << vi;
                if (use_tex || use_nrm)
                {
                    f << "/";
                    if (use_tex)
                    {
                        f << cloud.faceTextureIndices()->getValue(fi, c) + 1;
                    }
                    if (use_nrm)
                    {
                        f << "/" << vi;
                    }
                }
            }
            f << "\n";
        }
    }
}

} // namespace io
} // namespace plapoint
