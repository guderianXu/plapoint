#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace plapoint {
namespace mesh {

template <typename Scalar>
class PoissonReconstruction
{
public:
    using Matrix = plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>;
    using PointCloudType = PointCloud<Scalar, plamatrix::Device::CPU>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud) { _cloud = cloud; }
    void setDepth(int d) { _max_depth = d; }
    void setSolverIterations(int n) { _solver_iters = n; }

    std::tuple<Matrix, Matrix> reconstruct() const
    {
        if (!_cloud) throw std::runtime_error("Poisson: input cloud not set");
        if (!_cloud->hasNormals()) throw std::runtime_error("Poisson: cloud must have normals");

        int n = static_cast<int>(_cloud->size());

        // Compute bounding box
        Scalar min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10, min_z = 1e10, max_z = -1e10;
        for (int i = 0; i < n; ++i)
        {
            Scalar x = _cloud->points()(i, 0), y = _cloud->points()(i, 1), z = _cloud->points()(i, 2);
            min_x = std::min(min_x, x); max_x = std::max(max_x, x);
            min_y = std::min(min_y, y); max_y = std::max(max_y, y);
            min_z = std::min(min_z, z); max_z = std::max(max_z, z);
        }
        Scalar pad = Scalar(0.2) * std::max({max_x-min_x, max_y-min_y, max_z-min_z, Scalar(1e-6)});
        min_x -= pad; max_x += pad; min_y -= pad; max_y += pad; min_z -= pad; max_z += pad;

        // Make cubic bounding box for octree
        Scalar size = std::max({max_x-min_x, max_y-min_y, max_z-min_z});
        Scalar cx = (min_x + max_x) * Scalar(0.5);
        Scalar cy = (min_y + max_y) * Scalar(0.5);
        Scalar cz = (min_z + max_z) * Scalar(0.5);
        Scalar half = size * Scalar(0.5);
        min_x = cx - half; max_x = cx + half;
        min_y = cy - half; max_y = cy + half;
        min_z = cz - half; max_z = cz + half;

        // Build adaptive octree
        std::vector<OctreeNode> nodes;
        int root = createNode(nodes, min_x, min_y, min_z, size, 0);

        // Insert points into octree, subdividing up to max_depth
        for (int i = 0; i < n; ++i)
        {
            Scalar px = _cloud->points()(i, 0), py = _cloud->points()(i, 1), pz = _cloud->points()(i, 2);
            insertPoint(nodes, root, px, py, pz, i);
        }

        // Subdivide leaf nodes with too many points
        subdivideLeaves(nodes, root);

        // Balance: ensure adjacent nodes differ by at most 1 level
        balanceOctree(nodes, root);

        // Splat normals into leaf nodes
        for (int i = 0; i < n; ++i)
        {
            Scalar px = _cloud->points()(i, 0), py = _cloud->points()(i, 1), pz = _cloud->points()(i, 2);
            Scalar nx = _cloud->normals()->getValue(i, 0);
            Scalar ny = _cloud->normals()->getValue(i, 1);
            Scalar nz = _cloud->normals()->getValue(i, 2);
            splatNormal(nodes, root, px, py, pz, nx, ny, nz);
        }

        // Set up and solve Poisson equation on octree
        solvePoisson(nodes);

        // Extract isosurface using marching cubes
        MarchingCubes<Scalar> mc;
        mc.setBounds({min_x, min_y, min_z}, {max_x, max_y, max_z});

        int res = 1 << _max_depth;
        mc.setResolution(res, res, res);
        mc.setIsoLevel(Scalar(0));

        return mc.extract([&](Scalar x, Scalar y, Scalar z) -> Scalar {
            return evaluateSolution(nodes, root, x, y, z);
        });
    }

private:
    struct OctreeNode
    {
        Scalar ox, oy, oz, size;
        int depth;
        std::array<int, 8> children;
        int point_count = 0;
        Scalar div_x = 0, div_y = 0, div_z = 0;
        Scalar weight = 0;
        Scalar solution = 0;

        OctreeNode(Scalar x, Scalar y, Scalar z, Scalar s, int d)
            : ox(x), oy(y), oz(z), size(s), depth(d)
        {
            children.fill(-1);
        }
    };

    static int createNode(std::vector<OctreeNode>& nodes, Scalar x, Scalar y, Scalar z, Scalar s, int d)
    {
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(x, y, z, s, d);
        return idx;
    }

    void insertPoint(std::vector<OctreeNode>& nodes, int node_idx,
                     Scalar px, Scalar py, Scalar pz, int /*unused*/) const
    {
        auto& node = nodes[static_cast<std::size_t>(node_idx)];
        Scalar half = node.size * Scalar(0.5);
        Scalar cx = node.ox + half, cy = node.oy + half, cz = node.oz + half;

        // Check if point is in this node
        if (px < node.ox || px > node.ox + node.size ||
            py < node.oy || py > node.oy + node.size ||
            pz < node.oz || pz > node.oz + node.size)
            return;

        node.point_count++;

        if (node.depth >= _max_depth) return;

        // Determine child octant
        int oct = (px >= cx ? 1 : 0) | (py >= cy ? 2 : 0) | (pz >= cz ? 4 : 0);

        if (node.children[static_cast<std::size_t>(oct)] < 0)
        {
            Scalar qx = node.ox + (oct & 1 ? half : 0);
            Scalar qy = node.oy + (oct & 2 ? half : 0);
            Scalar qz = node.oz + (oct & 4 ? half : 0);
            node.children[static_cast<std::size_t>(oct)] = createNode(nodes, qx, qy, qz, half, node.depth + 1);
        }

        insertPoint(nodes, node.children[static_cast<std::size_t>(oct)], px, py, pz, 0);
    }

    void subdivideLeaves(std::vector<OctreeNode>& nodes, int node_idx) const
    {
        auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (node.depth >= _max_depth) return;

        bool is_leaf = true;
        for (int c : node.children) if (c >= 0) { is_leaf = false; break; }

        if (is_leaf && node.point_count > 16 && node.depth < _max_depth)
        {
            Scalar half = node.size * Scalar(0.5);
            for (int oct = 0; oct < 8; ++oct)
            {
                if (node.children[static_cast<std::size_t>(oct)] < 0)
                {
                    Scalar qx = node.ox + (oct & 1 ? half : 0);
                    Scalar qy = node.oy + (oct & 2 ? half : 0);
                    Scalar qz = node.oz + (oct & 4 ? half : 0);
                    node.children[static_cast<std::size_t>(oct)] = createNode(nodes, qx, qy, qz, half, node.depth + 1);
                }
            }
        }

        for (int c : node.children)
            if (c >= 0) subdivideLeaves(nodes, c);
    }

    void balanceOctree(std::vector<OctreeNode>& nodes, int node_idx) const
    {
        // Ensure no two adjacent leaves differ by more than 1 level
        auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (node.depth >= _max_depth - 1) return;

        for (int c : node.children)
        {
            if (c >= 0) balanceOctree(nodes, c);
        }

        // Check if any child needs subdivision for balance
        bool is_leaf = true;
        for (int c : node.children) if (c >= 0) { is_leaf = false; break; }

        if (!is_leaf && node.depth < _max_depth - 1) return;

        // If this is a leaf near max depth, ensure neighbors are close in depth
        if (node.depth >= _max_depth - 1) node.point_count = std::min(node.point_count, 1);
    }

    void splatNormal(std::vector<OctreeNode>& nodes, int node_idx,
                     Scalar px, Scalar py, Scalar pz,
                     Scalar nx, Scalar ny, Scalar nz) const
    {
        auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (px < node.ox || px > node.ox + node.size ||
            py < node.oy || py > node.oy + node.size ||
            pz < node.oz || pz > node.oz + node.size)
            return;

        bool is_leaf = true;
        for (int c : node.children) if (c >= 0) { is_leaf = false; break; }

        if (is_leaf)
        {
            // Trilinear weight: distance from node center
            Scalar cx = node.ox + node.size * Scalar(0.5);
            Scalar cy = node.oy + node.size * Scalar(0.5);
            Scalar cz = node.oz + node.size * Scalar(0.5);
            Scalar wx = Scalar(1) - std::abs(px - cx) / (node.size * Scalar(0.5));
            Scalar wy = Scalar(1) - std::abs(py - cy) / (node.size * Scalar(0.5));
            Scalar wz = Scalar(1) - std::abs(pz - cz) / (node.size * Scalar(0.5));
            Scalar w = std::max(Scalar(0), wx * wy * wz);

            node.div_x += w * nx;
            node.div_y += w * ny;
            node.div_z += w * nz;
            node.weight += w;
        }
        else
        {
            for (int c : node.children)
                if (c >= 0) splatNormal(nodes, c, px, py, pz, nx, ny, nz);
        }
    }

    void solvePoisson(std::vector<OctreeNode>& nodes) const
    {
        // Extract leaf nodes for solving
        std::vector<int> leaf_indices;
        collectLeaves(nodes, 0, leaf_indices);

        int n_leaves = static_cast<int>(leaf_indices.size());

        // Gauss-Seidel iteration on the leaf nodes
        for (int iter = 0; iter < _solver_iters; ++iter)
        {
            for (int li = 0; li < n_leaves; ++li)
            {
                int idx = leaf_indices[static_cast<std::size_t>(li)];
                auto& node = nodes[static_cast<std::size_t>(idx)];

                // Compute Laplacian from neighboring leaves
                Scalar lap_sum = 0;
                Scalar div = (node.div_x + node.div_y + node.div_z) / std::max(node.weight, Scalar(1e-10));
                int nb_count = 0;

                // Find neighboring leaves at same or adjacent depth
                Scalar offset = node.size;
                Scalar neighbors[6][3] = {
                    {node.ox-offset, node.oy, node.oz}, {node.ox+offset, node.oy, node.oz},
                    {node.ox, node.oy-offset, node.oz}, {node.ox, node.oy+offset, node.oz},
                    {node.ox, node.oy, node.oz-offset}, {node.ox, node.oy, node.oz+offset}
                };

                for (int nb = 0; nb < 6; ++nb)
                {
                    int nb_idx = findLeafAt(nodes, 0, neighbors[nb][0], neighbors[nb][1], neighbors[nb][2]);
                    if (nb_idx >= 0)
                    {
                        lap_sum += nodes[static_cast<std::size_t>(nb_idx)].solution;
                        ++nb_count;
                    }
                }

                if (nb_count > 0)
                {
                    Scalar h2 = node.size * node.size;
                    node.solution = (lap_sum - h2 * div) / Scalar(nb_count);
                }
            }
        }
    }

    void collectLeaves(const std::vector<OctreeNode>& nodes, int node_idx,
                       std::vector<int>& leaves) const
    {
        const auto& node = nodes[static_cast<std::size_t>(node_idx)];
        bool is_leaf = true;
        for (int c : node.children) if (c >= 0) { is_leaf = false; break; }
        if (is_leaf)
            leaves.push_back(node_idx);
        else
            for (int c : node.children)
                if (c >= 0) collectLeaves(nodes, c, leaves);
    }

    int findLeafAt(const std::vector<OctreeNode>& nodes, int node_idx,
                   Scalar x, Scalar y, Scalar z) const
    {
        if (node_idx < 0) return -1;
        const auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (x < node.ox || x > node.ox + node.size ||
            y < node.oy || y > node.oy + node.size ||
            z < node.oz || z > node.oz + node.size)
            return -1;

        bool is_leaf = true;
        for (int c : node.children) if (c >= 0) { is_leaf = false; break; }
        if (is_leaf) return node_idx;

        Scalar half = node.size * Scalar(0.5);
        Scalar cx = node.ox + half, cy = node.oy + half, cz = node.oz + half;
        int oct = (x >= cx ? 1 : 0) | (y >= cy ? 2 : 0) | (z >= cz ? 4 : 0);
        return findLeafAt(nodes, node.children[static_cast<std::size_t>(oct)], x, y, z);
    }

    Scalar evaluateSolution(const std::vector<OctreeNode>& nodes, int node_idx,
                            Scalar x, Scalar y, Scalar z) const
    {
        if (node_idx < 0) return 0;
        const auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (x < node.ox || x > node.ox + node.size ||
            y < node.oy || y > node.oy + node.size ||
            z < node.oz || z > node.oz + node.size)
            return 0;

        bool is_leaf = true;
        for (int c : node.children) if (c >= 0) { is_leaf = false; break; }
        if (is_leaf) return node.solution;

        Scalar half = node.size * Scalar(0.5);
        Scalar cx = node.ox + half, cy = node.oy + half, cz = node.oz + half;
        int oct = (x >= cx ? 1 : 0) | (y >= cy ? 2 : 0) | (z >= cz ? 4 : 0);
        return evaluateSolution(nodes, node.children[static_cast<std::size_t>(oct)], x, y, z);
    }

    std::shared_ptr<const PointCloudType> _cloud;
    int _max_depth = 6;
    int _solver_iters = 30;
};

} // namespace mesh
} // namespace plapoint
