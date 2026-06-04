#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <algorithm>
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

struct ObjVertexIndices
{
    int v = 0;
    int t = 0;
    int n = 0;
    bool has_t = false;
    bool has_n = false;
};

inline int parseObjIndexComponent(const std::string& s)
{
    const int value = std::stoi(s);
    if (value == 0)
    {
        throw std::out_of_range("OBJ indices are 1-based and must not be zero");
    }
    return value > 0 ? value - 1 : value;
}

inline int resolveObjIndex(int idx, size_t count, const char* label)
{
    const int resolved = idx < 0
        ? static_cast<int>(count) + idx
        : idx;
    if (resolved < 0 || static_cast<size_t>(resolved) >= count)
    {
        throw std::out_of_range(std::string(label) + " index out of range");
    }
    return resolved;
}

inline ObjVertexIndices parseFaceVertex(const std::string& s)
{
    ObjVertexIndices idx;
    size_t slash1 = s.find('/');
    if (slash1 == std::string::npos)
    {
        idx.v = parseObjIndexComponent(s);
        return idx;
    }
    idx.v = parseObjIndexComponent(s.substr(0, slash1));
    size_t slash2 = s.find('/', slash1 + 1);
    if (slash2 == std::string::npos)
    {
        if (slash1 + 1 < s.size())
        {
            idx.t = parseObjIndexComponent(s.substr(slash1 + 1));
            idx.has_t = true;
        }
        return idx;
    }
    if (slash2 == slash1 + 1)
    {
        if (slash2 + 1 < s.size())
        {
            idx.n = parseObjIndexComponent(s.substr(slash2 + 1));
            idx.has_n = true;
        }
        return idx;
    }
    idx.t = parseObjIndexComponent(s.substr(slash1 + 1, slash2 - slash1 - 1));
    idx.has_t = true;
    if (slash2 + 1 < s.size())
    {
        idx.n = parseObjIndexComponent(s.substr(slash2 + 1));
        idx.has_n = true;
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
    std::vector<std::vector<int>> face_normals;
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
            std::vector<int> fv, ft, fn;
            std::string s;
            while (iss >> s)
            {
                auto idx = detail::parseFaceVertex(s);
                fv.push_back(detail::resolveObjIndex(idx.v, vx.size(), "OBJ face vertex"));
                ft.push_back(-1);
                fn.push_back(-1);
                if (idx.has_t)
                {
                    ft.back() = detail::resolveObjIndex(idx.t, tx.size(), "OBJ face texture");
                }
                if (idx.has_n)
                {
                    fn.back() = detail::resolveObjIndex(idx.n, nx.size(), "OBJ face normal");
                }
            }
            if (fv.size() < 3)
            {
                throw std::invalid_argument("OBJ face must contain at least 3 vertices");
            }
            face_verts.push_back(fv);
            face_tex.push_back(ft);
            face_normals.push_back(fn);
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

    std::vector<int> vertex_normal_indices(n, -1);
    std::vector<bool> vertex_referenced_by_face(n, false);
    const bool has_face_normal_indices = std::any_of(
        face_normals.begin(), face_normals.end(),
        [](const std::vector<int>& row) {
            return std::any_of(row.begin(), row.end(), [](int idx) { return idx >= 0; });
        });
    bool face_normal_indices_consistent = !nx.empty() && has_face_normal_indices;
    if (!nx.empty() && has_face_normal_indices)
    {
        for (size_t fi = 0; fi < face_verts.size(); ++fi)
        {
            if (face_normals[fi].size() != face_verts[fi].size())
            {
                face_normal_indices_consistent = false;
                continue;
            }
            for (size_t c = 0; c < face_verts[fi].size(); ++c)
            {
                const int vi = face_verts[fi][c];
                if (vi >= 0 && static_cast<size_t>(vi) < n)
                {
                    vertex_referenced_by_face[static_cast<size_t>(vi)] = true;
                }
                const int ni = face_normals[fi][c];
                if (ni < 0)
                {
                    continue;
                }
                if (vi < 0 || static_cast<size_t>(vi) >= n ||
                    ni < 0 || static_cast<size_t>(ni) >= nx.size())
                {
                    throw std::out_of_range("OBJ face normal index out of range");
                }
                int& assigned = vertex_normal_indices[static_cast<size_t>(vi)];
                if (assigned >= 0 && assigned != ni)
                {
                    face_normal_indices_consistent = false;
                    continue;
                }
                if (assigned < 0)
                {
                    assigned = ni;
                }
            }
        }
    }

    const bool use_ordered_normals = !nx.empty() && nx.size() == n;
    const bool all_vertices_have_face_normals =
        std::all_of(vertex_normal_indices.begin(), vertex_normal_indices.end(),
                    [](int idx) { return idx >= 0; });
    bool ordered_normals_only_fill_unreferenced_vertices = use_ordered_normals;
    for (size_t i = 0; i < n && ordered_normals_only_fill_unreferenced_vertices; ++i)
    {
        if (vertex_normal_indices[i] < 0 && vertex_referenced_by_face[i])
        {
            ordered_normals_only_fill_unreferenced_vertices = false;
        }
    }
    const bool can_assign_normals =
        (face_normal_indices_consistent &&
         (all_vertices_have_face_normals || ordered_normals_only_fill_unreferenced_vertices)) ||
        (!face_normal_indices_consistent && ordered_normals_only_fill_unreferenced_vertices) ||
        (!has_face_normal_indices && use_ordered_normals);

    if (can_assign_normals)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(
            static_cast<plamatrix::Index>(n), 3);
        for (size_t i = 0; i < n; ++i)
        {
            const size_t ni =
                (face_normal_indices_consistent && vertex_normal_indices[i] >= 0)
                    ? static_cast<size_t>(vertex_normal_indices[i])
                    : i;
            nrm(static_cast<plamatrix::Index>(i), 0) = nx[ni];
            nrm(static_cast<plamatrix::Index>(i), 1) = ny[ni];
            nrm(static_cast<plamatrix::Index>(i), 2) = nz[ni];
        }
        cloud->setNormals(std::move(nrm));
    }

    if (!tx.empty())
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> tex(
            static_cast<plamatrix::Index>(tx.size()), 2);
        for (size_t i = 0; i < tx.size(); ++i)
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

        const bool has_complete_face_texture_indices = !face_tex.empty() &&
            face_tex.size() == face_verts.size() &&
            std::all_of(face_tex.begin(), face_tex.end(), [](const std::vector<int>& row) {
                return row.size() >= 3 &&
                    std::all_of(row.begin(), row.begin() + 3, [](int idx) { return idx >= 0; });
            });

        if (has_complete_face_texture_indices)
        {
            plamatrix::DenseMatrix<int, plamatrix::Device::CPU> ft(nf, 3);
            for (plamatrix::Index fi = 0; fi < nf; ++fi)
            {
                for (int c = 0; c < 3; ++c)
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
        for (plamatrix::Index i = 0; i < cloud.textureCoords()->rows(); ++i)
        {
            f << "vt " << cloud.textureCoords()->getValue(i, 0)
              << " " << cloud.textureCoords()->getValue(i, 1) << "\n";
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
