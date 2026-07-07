#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <plapoint/core/point_cloud.h>

namespace plapoint::mesh
{

namespace detail
{

template <typename Scalar>
using CpuCloud = PointCloud<Scalar, plamatrix::Device::CPU>;

template <typename Scalar>
struct Vec3
{
    long double x = 0;
    long double y = 0;
    long double z = 0;
};

struct VoxelKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator<(const VoxelKey& other) const
    {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

inline std::uint64_t edgeKey(int a, int b)
{
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

template <typename Scalar>
Vec3<Scalar> pointAt(const CpuCloud<Scalar>& mesh, int idx)
{
    return {
        static_cast<long double>(mesh.points().getValue(idx, 0)),
        static_cast<long double>(mesh.points().getValue(idx, 1)),
        static_cast<long double>(mesh.points().getValue(idx, 2))
    };
}

template <typename Scalar>
Vec3<Scalar> cross(const Vec3<Scalar>& a, const Vec3<Scalar>& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

template <typename Scalar>
long double norm(const Vec3<Scalar>& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

template <typename Scalar>
long double triangleArea(const CpuCloud<Scalar>& mesh, int ia, int ib, int ic)
{
    const auto a = pointAt(mesh, ia);
    const auto b = pointAt(mesh, ib);
    const auto c = pointAt(mesh, ic);
    const Vec3<Scalar> ab{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3<Scalar> ac{c.x - a.x, c.y - a.y, c.z - a.z};
    return norm(cross<Scalar>(ab, ac)) * 0.5L;
}

template <typename Scalar>
plamatrix::DenseMatrix<int, plamatrix::Device::CPU>
facesToMatrix(const std::vector<std::array<int, 3>>& faces)
{
    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> matrix(
        static_cast<plamatrix::Index>(faces.size()), 3);
    for (std::size_t r = 0; r < faces.size(); ++r)
    {
        matrix.setValue(static_cast<plamatrix::Index>(r), 0, faces[r][0]);
        matrix.setValue(static_cast<plamatrix::Index>(r), 1, faces[r][1]);
        matrix.setValue(static_cast<plamatrix::Index>(r), 2, faces[r][2]);
    }
    return matrix;
}

template <typename Scalar>
std::vector<std::array<int, 3>> matrixToFaces(const plamatrix::DenseMatrix<int, plamatrix::Device::CPU>& faces)
{
    std::vector<std::array<int, 3>> out;
    out.reserve(static_cast<std::size_t>(faces.rows()));
    for (plamatrix::Index r = 0; r < faces.rows(); ++r)
    {
        out.push_back({
            faces.getValue(r, 0),
            faces.getValue(r, 1),
            faces.getValue(r, 2)
        });
    }
    return out;
}

template <typename Scalar>
void copyPointwiseAttributes(
    const CpuCloud<Scalar>& src,
    const std::vector<int>& source_indices,
    CpuCloud<Scalar>& dst,
    bool copy_normals = true)
{
    if (copy_normals && src.hasNormals())
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(
            static_cast<plamatrix::Index>(source_indices.size()), 3);
        for (std::size_t i = 0; i < source_indices.size(); ++i)
        {
            const int src_idx = source_indices[i];
            normals.setValue(static_cast<plamatrix::Index>(i), 0, src.normals()->getValue(src_idx, 0));
            normals.setValue(static_cast<plamatrix::Index>(i), 1, src.normals()->getValue(src_idx, 1));
            normals.setValue(static_cast<plamatrix::Index>(i), 2, src.normals()->getValue(src_idx, 2));
        }
        dst.setNormals(std::move(normals));
    }

    if (src.hasColors())
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
            static_cast<plamatrix::Index>(source_indices.size()), 3);
        for (std::size_t i = 0; i < source_indices.size(); ++i)
        {
            const int src_idx = source_indices[i];
            colors.setValue(static_cast<plamatrix::Index>(i), 0, src.colors()->getValue(src_idx, 0));
            colors.setValue(static_cast<plamatrix::Index>(i), 1, src.colors()->getValue(src_idx, 1));
            colors.setValue(static_cast<plamatrix::Index>(i), 2, src.colors()->getValue(src_idx, 2));
        }
        dst.setColors(std::move(colors));
    }

    if (src.hasIntensities())
    {
        plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(
            static_cast<plamatrix::Index>(source_indices.size()), 1);
        for (std::size_t i = 0; i < source_indices.size(); ++i)
        {
            intensities.setValue(
                static_cast<plamatrix::Index>(i),
                0,
                src.intensities()->getValue(source_indices[i], 0));
        }
        dst.setIntensities(std::move(intensities));
    }

    if (src.hasScalarFields())
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> scalar_fields(
            static_cast<plamatrix::Index>(source_indices.size()),
            static_cast<plamatrix::Index>(src.scalarFieldNames().size()));
        for (std::size_t i = 0; i < source_indices.size(); ++i)
        {
            const int src_idx = source_indices[i];
            for (plamatrix::Index c = 0; c < scalar_fields.cols(); ++c)
            {
                scalar_fields.setValue(
                    static_cast<plamatrix::Index>(i),
                    c,
                    src.scalarFields()->getValue(src_idx, c));
            }
        }
        dst.setScalarFields(src.scalarFieldNames(), std::move(scalar_fields));
    }

    if (src.hasTextureCoords() &&
        src.textureCoords()->rows() == static_cast<plamatrix::Index>(src.size()))
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> texture_coords(
            static_cast<plamatrix::Index>(source_indices.size()), 2);
        for (std::size_t i = 0; i < source_indices.size(); ++i)
        {
            const int src_idx = source_indices[i];
            texture_coords.setValue(static_cast<plamatrix::Index>(i), 0, src.textureCoords()->getValue(src_idx, 0));
            texture_coords.setValue(static_cast<plamatrix::Index>(i), 1, src.textureCoords()->getValue(src_idx, 1));
        }
        dst.setTextureCoords(std::move(texture_coords));
    }
}

template <typename Scalar>
CpuCloud<Scalar> copyWithFaces(
    const CpuCloud<Scalar>& src,
    const std::vector<std::array<int, 3>>& faces,
    bool copy_normals = true)
{
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(src.size()), 3);
    std::vector<int> source_indices;
    source_indices.reserve(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        source_indices.push_back(static_cast<int>(i));
        for (int c = 0; c < 3; ++c)
        {
            points.setValue(static_cast<plamatrix::Index>(i), c, src.points().getValue(static_cast<plamatrix::Index>(i), c));
        }
    }

    CpuCloud<Scalar> out(std::move(points));
    copyPointwiseAttributes(src, source_indices, out, copy_normals);
    out.setFaces(facesToMatrix<Scalar>(faces));
    return out;
}

template <typename Scalar>
CpuCloud<Scalar> compactWithFaces(
    const CpuCloud<Scalar>& src,
    const std::vector<std::array<int, 3>>& kept_faces,
    bool copy_normals = true)
{
    std::vector<int> old_to_new(src.size(), -1);
    std::vector<int> source_indices;
    for (const auto& face : kept_faces)
    {
        for (int idx : face)
        {
            if (old_to_new[static_cast<std::size_t>(idx)] < 0)
            {
                old_to_new[static_cast<std::size_t>(idx)] = static_cast<int>(source_indices.size());
                source_indices.push_back(idx);
            }
        }
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(source_indices.size()), 3);
    for (std::size_t i = 0; i < source_indices.size(); ++i)
    {
        const int src_idx = source_indices[i];
        for (int c = 0; c < 3; ++c)
        {
            points.setValue(static_cast<plamatrix::Index>(i), c, src.points().getValue(src_idx, c));
        }
    }

    std::vector<std::array<int, 3>> remapped_faces;
    remapped_faces.reserve(kept_faces.size());
    for (const auto& face : kept_faces)
    {
        remapped_faces.push_back({
            old_to_new[static_cast<std::size_t>(face[0])],
            old_to_new[static_cast<std::size_t>(face[1])],
            old_to_new[static_cast<std::size_t>(face[2])]
        });
    }

    CpuCloud<Scalar> out(std::move(points));
    copyPointwiseAttributes(src, source_indices, out, copy_normals);
    out.setFaces(facesToMatrix<Scalar>(remapped_faces));
    return out;
}

template <typename Scalar>
std::vector<std::vector<int>> buildVertexNeighbors(const CpuCloud<Scalar>& mesh)
{
    std::vector<std::vector<int>> neighbors(mesh.size());
    if (!mesh.hasFaces())
    {
        return neighbors;
    }
    const auto& faces = *mesh.faces();
    for (plamatrix::Index r = 0; r < faces.rows(); ++r)
    {
        const int a = faces.getValue(r, 0);
        const int b = faces.getValue(r, 1);
        const int c = faces.getValue(r, 2);
        neighbors[static_cast<std::size_t>(a)].push_back(b);
        neighbors[static_cast<std::size_t>(a)].push_back(c);
        neighbors[static_cast<std::size_t>(b)].push_back(a);
        neighbors[static_cast<std::size_t>(b)].push_back(c);
        neighbors[static_cast<std::size_t>(c)].push_back(a);
        neighbors[static_cast<std::size_t>(c)].push_back(b);
    }
    for (auto& row : neighbors)
    {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return neighbors;
}

template <typename Scalar>
void applyLaplacianStep(
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& points,
    const std::vector<std::vector<int>>& neighbors,
    Scalar factor)
{
    if (factor == Scalar(0))
    {
        return;
    }
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> next(points.rows(), points.cols());
    for (plamatrix::Index r = 0; r < points.rows(); ++r)
    {
        for (int c = 0; c < points.cols(); ++c)
        {
            next.setValue(r, c, points.getValue(r, c));
        }
    }
    for (std::size_t i = 0; i < neighbors.size(); ++i)
    {
        if (neighbors[i].empty())
        {
            continue;
        }
        for (int c = 0; c < 3; ++c)
        {
            long double mean = 0;
            for (int nb : neighbors[i])
            {
                mean += static_cast<long double>(points.getValue(nb, c));
            }
            mean /= static_cast<long double>(neighbors[i].size());
            const long double current = static_cast<long double>(points.getValue(static_cast<plamatrix::Index>(i), c));
            const long double updated = current + static_cast<long double>(factor) * (mean - current);
            next.setValue(static_cast<plamatrix::Index>(i), c, static_cast<Scalar>(updated));
        }
    }
    points = std::move(next);
}

} // namespace detail

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> voxelClusterSimplify(
    const PointCloud<Scalar, plamatrix::Device::CPU>& mesh,
    Scalar cluster_size)
{
    if (!std::isfinite(cluster_size) || cluster_size <= Scalar(0))
    {
        throw std::invalid_argument("voxelClusterSimplify: cluster size must be finite and positive");
    }
    if (mesh.size() == 0)
    {
        return detail::compactWithFaces(mesh, {});
    }

    Scalar min_x = mesh.points().getValue(0, 0);
    Scalar min_y = mesh.points().getValue(0, 1);
    Scalar min_z = mesh.points().getValue(0, 2);
    for (std::size_t i = 0; i < mesh.size(); ++i)
    {
        min_x = std::min(min_x, mesh.points().getValue(static_cast<plamatrix::Index>(i), 0));
        min_y = std::min(min_y, mesh.points().getValue(static_cast<plamatrix::Index>(i), 1));
        min_z = std::min(min_z, mesh.points().getValue(static_cast<plamatrix::Index>(i), 2));
    }

    struct Accum
    {
        long double x = 0;
        long double y = 0;
        long double z = 0;
        long double r = 0;
        long double g = 0;
        long double b = 0;
        long double intensity = 0;
        int count = 0;
        int new_index = -1;
    };

    std::map<detail::VoxelKey, Accum> clusters;
    std::vector<detail::VoxelKey> point_keys(mesh.size());
    for (std::size_t i = 0; i < mesh.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        const Scalar x = mesh.points().getValue(row, 0);
        const Scalar y = mesh.points().getValue(row, 1);
        const Scalar z = mesh.points().getValue(row, 2);
        detail::VoxelKey key{
            static_cast<int>(std::floor((x - min_x) / cluster_size)),
            static_cast<int>(std::floor((y - min_y) / cluster_size)),
            static_cast<int>(std::floor((z - min_z) / cluster_size))
        };
        point_keys[i] = key;
        auto& acc = clusters[key];
        acc.x += x;
        acc.y += y;
        acc.z += z;
        if (mesh.hasColors())
        {
            acc.r += mesh.colors()->getValue(row, 0);
            acc.g += mesh.colors()->getValue(row, 1);
            acc.b += mesh.colors()->getValue(row, 2);
        }
        if (mesh.hasIntensities())
        {
            acc.intensity += mesh.intensities()->getValue(row, 0);
        }
        ++acc.count;
    }

    int next_index = 0;
    for (auto& entry : clusters)
    {
        entry.second.new_index = next_index++;
    }

    std::vector<int> point_remap(mesh.size(), -1);
    for (std::size_t i = 0; i < point_keys.size(); ++i)
    {
        point_remap[i] = clusters[point_keys[i]].new_index;
    }

    std::vector<std::array<int, 3>> simplified_faces;
    if (mesh.hasFaces())
    {
        const auto& faces = *mesh.faces();
        simplified_faces.reserve(static_cast<std::size_t>(faces.rows()));
        for (plamatrix::Index r = 0; r < faces.rows(); ++r)
        {
            const int a = point_remap[static_cast<std::size_t>(faces.getValue(r, 0))];
            const int b = point_remap[static_cast<std::size_t>(faces.getValue(r, 1))];
            const int c = point_remap[static_cast<std::size_t>(faces.getValue(r, 2))];
            if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c)
            {
                continue;
            }
            simplified_faces.push_back({a, b, c});
        }
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(clusters.size()), 3);
    std::unique_ptr<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>> colors;
    std::unique_ptr<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>> intensities;
    if (mesh.hasColors())
    {
        colors = std::make_unique<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>>(
            static_cast<plamatrix::Index>(clusters.size()), 3);
    }
    if (mesh.hasIntensities())
    {
        intensities = std::make_unique<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>>(
            static_cast<plamatrix::Index>(clusters.size()), 1);
    }

    for (const auto& entry : clusters)
    {
        const auto& acc = entry.second;
        const auto row = static_cast<plamatrix::Index>(acc.new_index);
        const long double inv = 1.0L / static_cast<long double>(acc.count);
        points.setValue(row, 0, static_cast<Scalar>(acc.x * inv));
        points.setValue(row, 1, static_cast<Scalar>(acc.y * inv));
        points.setValue(row, 2, static_cast<Scalar>(acc.z * inv));
        if (colors)
        {
            colors->setValue(row, 0, static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(std::lround(acc.r * inv)), 0, 255)));
            colors->setValue(row, 1, static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(std::lround(acc.g * inv)), 0, 255)));
            colors->setValue(row, 2, static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(std::lround(acc.b * inv)), 0, 255)));
        }
        if (intensities)
        {
            intensities->setValue(row, 0, static_cast<std::uint16_t>(
                std::clamp<int>(static_cast<int>(std::lround(acc.intensity * inv)), 0, 65535)));
        }
    }

    PointCloud<Scalar, plamatrix::Device::CPU> out(std::move(points));
    if (colors) out.setColors(std::move(*colors));
    if (intensities) out.setIntensities(std::move(*intensities));
    out.setFaces(detail::facesToMatrix<Scalar>(simplified_faces));
    return out;
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> removeDegenerateFaces(
    const PointCloud<Scalar, plamatrix::Device::CPU>& mesh,
    Scalar min_area = Scalar(0))
{
    if (!std::isfinite(min_area) || min_area < Scalar(0))
    {
        throw std::invalid_argument("removeDegenerateFaces: min area must be finite and non-negative");
    }
    if (!mesh.hasFaces())
    {
        return detail::copyWithFaces(mesh, {});
    }

    std::vector<std::array<int, 3>> kept;
    const auto& faces = *mesh.faces();
    kept.reserve(static_cast<std::size_t>(faces.rows()));
    for (plamatrix::Index r = 0; r < faces.rows(); ++r)
    {
        const int a = faces.getValue(r, 0);
        const int b = faces.getValue(r, 1);
        const int c = faces.getValue(r, 2);
        if (a == b || b == c || a == c)
        {
            continue;
        }
        if (detail::triangleArea(mesh, a, b, c) <= static_cast<long double>(min_area))
        {
            continue;
        }
        kept.push_back({a, b, c});
    }
    return detail::copyWithFaces(mesh, kept);
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> removeSmallConnectedComponents(
    const PointCloud<Scalar, plamatrix::Device::CPU>& mesh,
    std::size_t min_faces)
{
    if (!mesh.hasFaces())
    {
        return detail::compactWithFaces(mesh, {});
    }

    const auto faces = detail::matrixToFaces<Scalar>(*mesh.faces());
    std::unordered_map<std::uint64_t, std::vector<int>> edge_to_faces;
    edge_to_faces.reserve(faces.size() * 3);
    for (std::size_t face_idx = 0; face_idx < faces.size(); ++face_idx)
    {
        const auto& face = faces[face_idx];
        edge_to_faces[detail::edgeKey(face[0], face[1])].push_back(static_cast<int>(face_idx));
        edge_to_faces[detail::edgeKey(face[1], face[2])].push_back(static_cast<int>(face_idx));
        edge_to_faces[detail::edgeKey(face[2], face[0])].push_back(static_cast<int>(face_idx));
    }

    std::vector<std::vector<int>> adjacency(faces.size());
    for (const auto& entry : edge_to_faces)
    {
        const auto& edge_faces = entry.second;
        for (std::size_t i = 0; i < edge_faces.size(); ++i)
        {
            for (std::size_t j = i + 1; j < edge_faces.size(); ++j)
            {
                adjacency[static_cast<std::size_t>(edge_faces[i])].push_back(edge_faces[j]);
                adjacency[static_cast<std::size_t>(edge_faces[j])].push_back(edge_faces[i]);
            }
        }
    }

    std::vector<std::uint8_t> visited(faces.size(), 0);
    std::vector<std::array<int, 3>> kept_faces;
    for (std::size_t start = 0; start < faces.size(); ++start)
    {
        if (visited[start])
        {
            continue;
        }

        std::vector<int> component;
        std::queue<int> queue;
        visited[start] = 1;
        queue.push(static_cast<int>(start));
        while (!queue.empty())
        {
            const int face_idx = queue.front();
            queue.pop();
            component.push_back(face_idx);
            for (int neighbor_face : adjacency[static_cast<std::size_t>(face_idx)])
            {
                if (!visited[static_cast<std::size_t>(neighbor_face)])
                {
                    visited[static_cast<std::size_t>(neighbor_face)] = 1;
                    queue.push(neighbor_face);
                }
            }
        }

        if (component.size() >= min_faces)
        {
            for (int face_idx : component)
            {
                kept_faces.push_back(faces[static_cast<std::size_t>(face_idx)]);
            }
        }
    }

    return detail::compactWithFaces(mesh, kept_faces);
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> recomputeVertexNormals(
    const PointCloud<Scalar, plamatrix::Device::CPU>& mesh)
{
    const auto faces = mesh.hasFaces()
        ? detail::matrixToFaces<Scalar>(*mesh.faces())
        : std::vector<std::array<int, 3>>{};
    auto out = detail::copyWithFaces(mesh, faces, false);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(
        static_cast<plamatrix::Index>(mesh.size()), 3);
    normals.fill(0);

    std::vector<detail::Vec3<Scalar>> accum(mesh.size());
    for (const auto& face : faces)
    {
        const auto a = detail::pointAt(mesh, face[0]);
        const auto b = detail::pointAt(mesh, face[1]);
        const auto c = detail::pointAt(mesh, face[2]);
        const detail::Vec3<Scalar> ab{b.x - a.x, b.y - a.y, b.z - a.z};
        const detail::Vec3<Scalar> ac{c.x - a.x, c.y - a.y, c.z - a.z};
        const auto n = detail::cross<Scalar>(ab, ac);
        for (int vertex : face)
        {
            auto& dst = accum[static_cast<std::size_t>(vertex)];
            dst.x += n.x;
            dst.y += n.y;
            dst.z += n.z;
        }
    }

    for (std::size_t i = 0; i < accum.size(); ++i)
    {
        const long double length = detail::norm(accum[i]);
        if (length <= std::numeric_limits<long double>::epsilon())
        {
            normals.setValue(static_cast<plamatrix::Index>(i), 0, Scalar(0));
            normals.setValue(static_cast<plamatrix::Index>(i), 1, Scalar(0));
            normals.setValue(static_cast<plamatrix::Index>(i), 2, Scalar(1));
            continue;
        }
        normals.setValue(static_cast<plamatrix::Index>(i), 0, static_cast<Scalar>(accum[i].x / length));
        normals.setValue(static_cast<plamatrix::Index>(i), 1, static_cast<Scalar>(accum[i].y / length));
        normals.setValue(static_cast<plamatrix::Index>(i), 2, static_cast<Scalar>(accum[i].z / length));
    }
    out.setNormals(std::move(normals));
    return out;
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> orientNormalsOutwardFromCentroid(
    const PointCloud<Scalar, plamatrix::Device::CPU>& mesh)
{
    const auto faces = mesh.hasFaces()
        ? detail::matrixToFaces<Scalar>(*mesh.faces())
        : std::vector<std::array<int, 3>>{};
    auto out = detail::copyWithFaces(mesh, faces);
    if (!out.hasNormals() || out.size() == 0)
    {
        return out;
    }

    long double cx = 0;
    long double cy = 0;
    long double cz = 0;
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        cx += out.points().getValue(static_cast<plamatrix::Index>(i), 0);
        cy += out.points().getValue(static_cast<plamatrix::Index>(i), 1);
        cz += out.points().getValue(static_cast<plamatrix::Index>(i), 2);
    }
    const long double inv_count = 1.0L / static_cast<long double>(out.size());
    cx *= inv_count;
    cy *= inv_count;
    cz *= inv_count;

    long double dot_sum = 0;
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        dot_sum += static_cast<long double>(out.normals()->getValue(row, 0)) *
                   (static_cast<long double>(out.points().getValue(row, 0)) - cx);
        dot_sum += static_cast<long double>(out.normals()->getValue(row, 1)) *
                   (static_cast<long double>(out.points().getValue(row, 1)) - cy);
        dot_sum += static_cast<long double>(out.normals()->getValue(row, 2)) *
                   (static_cast<long double>(out.points().getValue(row, 2)) - cz);
    }

    if (dot_sum >= 0)
    {
        return out;
    }

    auto* normals = out.normals();
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        normals->setValue(row, 0, -normals->getValue(row, 0));
        normals->setValue(row, 1, -normals->getValue(row, 1));
        normals->setValue(row, 2, -normals->getValue(row, 2));
    }
    if (out.hasFaces())
    {
        auto* out_faces = out.faces();
        for (plamatrix::Index r = 0; r < out_faces->rows(); ++r)
        {
            const int tmp = out_faces->getValue(r, 1);
            out_faces->setValue(r, 1, out_faces->getValue(r, 2));
            out_faces->setValue(r, 2, tmp);
        }
    }
    return out;
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> taubinSmooth(
    const PointCloud<Scalar, plamatrix::Device::CPU>& mesh,
    int iterations,
    Scalar lambda = Scalar(0.5),
    Scalar mu = Scalar(-0.53))
{
    if (iterations < 0)
    {
        throw std::invalid_argument("taubinSmooth: iterations must be non-negative");
    }
    if (!std::isfinite(lambda) || !std::isfinite(mu))
    {
        throw std::invalid_argument("taubinSmooth: factors must be finite");
    }

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(mesh.size()), 3);
    std::vector<int> source_indices;
    source_indices.reserve(mesh.size());
    for (std::size_t i = 0; i < mesh.size(); ++i)
    {
        source_indices.push_back(static_cast<int>(i));
        for (int c = 0; c < 3; ++c)
        {
            points.setValue(static_cast<plamatrix::Index>(i), c, mesh.points().getValue(static_cast<plamatrix::Index>(i), c));
        }
    }

    const auto neighbors = detail::buildVertexNeighbors(mesh);
    for (int iter = 0; iter < iterations; ++iter)
    {
        detail::applyLaplacianStep(points, neighbors, lambda);
        detail::applyLaplacianStep(points, neighbors, mu);
    }

    const auto faces = mesh.hasFaces()
        ? detail::matrixToFaces<Scalar>(*mesh.faces())
        : std::vector<std::array<int, 3>>{};
    PointCloud<Scalar, plamatrix::Device::CPU> out(std::move(points));
    detail::copyPointwiseAttributes(mesh, source_indices, out, false);
    out.setFaces(detail::facesToMatrix<Scalar>(faces));
    if (mesh.hasNormals())
    {
        return recomputeVertexNormals(out);
    }
    return out;
}

} // namespace plapoint::mesh
