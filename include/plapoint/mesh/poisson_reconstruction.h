#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/opencl/iterative_solver.h>
#include <plamatrix/internal/ops/point_cloud.h>
#include <plamatrix/sparse/conjugate_gradient.h>
#include <plamatrix/internal/sparse/iterative_solver.h>
#include <plamatrix/internal/sparse/sparse_ops.h>
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_csr_matrix.h>
#include <plamatrix/internal/device/device_vector.h>

#include <plapoint/core/processing_policy.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/geometry_cloud.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plapoint/mesh/poisson_system.h>
#include <plapoint/search/kdtree.h>

namespace plapoint {
namespace mesh {

struct PoissonProcessingReport
{
    ProcessingDevice requestedDevice = ProcessingDevice::CPU;
    /// Device used by the full reconstruction chain. Field evaluation and extraction are CPU.
    ProcessingDevice actualDevice = ProcessingDevice::CPU;
    /// Device used only for the sparse PCG solve.
    ProcessingDevice solverDevice = ProcessingDevice::CPU;
    bool usedFallback = false;
    std::string fallbackReason;
    /// Number of populated leaves assembled into the sparse Poisson system.
    std::size_t leafCount = 0;
    plamatrix::internal::IterativeSolverReport solver;
    /// Total number of leaves in the complete, balanced octree, including empty leaves.
    std::size_t octreeLeafCount = 0;
    double fieldMinimum = 0.0;
    double fieldMaximum = 0.0;
    double isoLevel = 0.0;
    /// Maximum depth difference among face-, edge-, or corner-adjacent octree leaves.
    int maximumLeafNeighborDepthDifference = 0;
    /// Number of extraction-grid samples evaluated during reconstruction.
    std::size_t fieldGridSampleCount = 0;
    /// Number of spatial indices built while orienting the extracted faces.
    std::size_t orientationIndexBuildCount = 0;
    /// Number of nearest-normal queries issued while orienting non-degenerate faces.
    std::size_t orientationQueryCount = 0;
};

/// Reconstruct a triangle mesh from a point cloud with normals using a Poisson-style field solve.
template <typename Scalar> class PoissonReconstruction
{
public:
    using Matrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;
    using Vector = plamatrix::Matrix<Scalar, plamatrix::Dynamic, 1>;
    using PointCloudType = GeometryCloud<Scalar>;

    /// Set the CPU point cloud with normals used as reconstruction input.
    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud) { _cloud = cloud; }

    /// Set octree depth in the supported range [1, 8].
    void setDepth(int d)
    {
        if (d <= 0 || d > kMaxDepth)
        {
            throw std::invalid_argument("Poisson: depth must be in [1, 8]");
        }
        _max_depth = d;
    }

    /// Set the positive maximum PCG iteration count.
    void setSolverIterations(int n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("Poisson: solver iterations must be positive");
        }
        _solver_iters = n;
    }

    void setSolverTolerance(double tolerance)
    {
        if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance > 1.0)
        {
            throw std::invalid_argument("Poisson: solver tolerance must be finite and in (0, 1]");
        }
        _solver_tolerance = tolerance;
    }

    void setProcessingDevice(ProcessingDevice device) noexcept
    {
        _processing_device = device;
    }

    const PoissonProcessingReport& lastReport() const noexcept { return _lastReport; }

    /// Return the most recently assembled system. The reference is invalidated by the next
    /// reconstruct() call on this object or by destruction.
    const PoissonSystem<Scalar>& lastSystem() const
    {
        if (!_lastSystem)
        {
            throw std::logic_error("Poisson: no assembled system is available before reconstruct()");
        }
        return *_lastSystem;
    }

    /// Reconstruct vertices and triangular faces from the configured input cloud.
    std::tuple<Matrix, Matrix> reconstruct() const
    {
        _lastSystem.reset();
        _lastReport = {};
        _lastReport.requestedDevice = _processing_device;
        if (!_cloud) throw std::runtime_error("Poisson: input cloud not set");
        if (!_cloud->hasNormals()) throw std::runtime_error("Poisson: cloud must have normals");
        validateInputCloud();

        int n = static_cast<int>(_cloud->size());

        // Compute bounding box from input values, avoiding fixed sentinels that reject large coordinates.
        Scalar min_x = _cloud->points()(0, 0);
        Scalar max_x = min_x;
        Scalar min_y = _cloud->points()(0, 1);
        Scalar max_y = min_y;
        Scalar min_z = _cloud->points()(0, 2);
        Scalar max_z = min_z;
        for (int i = 1; i < n; ++i)
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
        _lastReport.maximumLeafNeighborDepthDifference = balanceOctree(nodes, root);
        _lastReport.octreeLeafCount = countOctreeLeaves(nodes, root);

        // Splat normals into leaf nodes
        for (int i = 0; i < n; ++i)
        {
            Scalar px = _cloud->points()(i, 0), py = _cloud->points()(i, 1), pz = _cloud->points()(i, 2);
            Scalar nx = _cloud->normals()->operator()(i, 0);
            Scalar ny = _cloud->normals()->operator()(i, 1);
            Scalar nz = _cloud->normals()->operator()(i, 2);
            splatNormal(nodes, root, px, py, pz, nx, ny, nz);
        }

        // Set up and solve Poisson equation on octree
        solvePoisson(nodes);

        // Extract isosurface using marching cubes
        MarchingCubes<Scalar> mc;
        mc.setBounds({min_x, min_y, min_z}, {max_x, max_y, max_z});

        int res = 1 << _max_depth;
        mc.setResolution(res, res, res);
        long double iso_sum = 0.0L;
        for (int i = 0; i < n; ++i)
        {
            iso_sum += static_cast<long double>(evaluateSolution(
                nodes,
                root,
                _cloud->points()(i, 0),
                _cloud->points()(i, 1),
                _cloud->points()(i, 2)));
        }
        const auto field = mc.sampleField([&](Scalar x, Scalar y, Scalar z) -> Scalar
        {
            return evaluateSolution(nodes, root, x, y, z);
        });
        _lastReport.fieldGridSampleCount = field.size();
        const auto field_extrema = std::minmax_element(field.begin(), field.end());
        const Scalar field_min = *field_extrema.first;
        const Scalar field_max = *field_extrema.second;
        Scalar iso_level = static_cast<Scalar>(iso_sum / static_cast<long double>(n));
        if (!(iso_level > field_min && iso_level < field_max))
        {
            iso_level = field_min + (field_max - field_min) * Scalar(0.5);
        }
        _lastReport.fieldMinimum = static_cast<double>(field_min);
        _lastReport.fieldMaximum = static_cast<double>(field_max);
        _lastReport.isoLevel = static_cast<double>(iso_level);
        mc.setIsoLevel(iso_level);

        auto [vertices, faces] = mc.extractSampledField(field);
        orientFacesWithInputNormals(vertices, faces);
        return {std::move(vertices), std::move(faces)};
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
        Scalar divergence = 0;
        Scalar solution = 0;

        OctreeNode(Scalar x, Scalar y, Scalar z, Scalar s, int d)
            : ox(x), oy(y), oz(z), size(s), depth(d)
        {
            children.fill(-1);
        }
    };

    void validateInputCloud() const
    {
        if (_cloud->size() == 0)
        {
            throw std::invalid_argument("Poisson: input cloud must not be empty");
        }
        if (_cloud->size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("Poisson: point count exceeds int range");
        }

        const auto* normals = _cloud->normals();
        for (std::size_t i = 0; i < _cloud->size(); ++i)
        {
            long double normal_components[3] = {0, 0, 0};
            for (int c = 0; c < 3; ++c)
            {
                const auto row = static_cast<plamatrix::Index>(i);
                if (!std::isfinite(_cloud->points()(row, c)))
                {
                    throw std::invalid_argument("Poisson: points must be finite");
                }
                if (!std::isfinite(normals->operator()(row, c)))
                {
                    throw std::invalid_argument("Poisson: normals must be finite");
                }
                const Scalar component = normals->operator()(row, c);
                normal_components[c] = static_cast<long double>(component);
            }
            const long double normal_norm = std::hypot(
                std::hypot(normal_components[0], normal_components[1]),
                normal_components[2]);
            if (!std::isfinite(normal_norm) || normal_norm <= 0 ||
                normal_norm > static_cast<long double>(std::numeric_limits<Scalar>::max()))
            {
                throw std::invalid_argument("Poisson: normals must have finite non-zero length");
            }
        }
    }

    static int createNode(std::vector<OctreeNode>& nodes, Scalar x, Scalar y, Scalar z, Scalar s, int d)
    {
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(x, y, z, s, d);
        return idx;
    }

    void insertPoint(std::vector<OctreeNode>& nodes, int node_idx,
                     Scalar px, Scalar py, Scalar pz, int /*unused*/) const
    {
        const auto node_pos = static_cast<std::size_t>(node_idx);
        auto& node = nodes[node_pos];
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
            const int child = createNode(nodes, qx, qy, qz, half, node.depth + 1);
            nodes[node_pos].children[static_cast<std::size_t>(oct)] = child;
        }

        insertPoint(nodes, nodes[node_pos].children[static_cast<std::size_t>(oct)], px, py, pz, 0);
    }

    void subdivideLeaves(std::vector<OctreeNode>& nodes, int node_idx) const
    {
        const auto node_pos = static_cast<std::size_t>(node_idx);
        auto& node = nodes[node_pos];
        if (node.depth >= _max_depth) return;

        bool is_leaf = true;
        for (int c : node.children) if (c >= 0) { is_leaf = false; break; }

        if (is_leaf && node.point_count > 16 && node.depth < _max_depth)
        {
            const Scalar half = node.size * Scalar(0.5);
            const Scalar ox = node.ox;
            const Scalar oy = node.oy;
            const Scalar oz = node.oz;
            const int child_depth = node.depth + 1;
            for (int oct = 0; oct < 8; ++oct)
            {
                if (nodes[node_pos].children[static_cast<std::size_t>(oct)] < 0)
                {
                    Scalar qx = ox + (oct & 1 ? half : 0);
                    Scalar qy = oy + (oct & 2 ? half : 0);
                    Scalar qz = oz + (oct & 4 ? half : 0);
                    const int child = createNode(nodes, qx, qy, qz, half, child_depth);
                    nodes[node_pos].children[static_cast<std::size_t>(oct)] = child;
                }
            }
        }

        const auto children = nodes[node_pos].children;
        for (int c : children)
            if (c >= 0) subdivideLeaves(nodes, c);
    }

    struct LeafCell
    {
        int nodeIndex = -1;
        int x = 0;
        int y = 0;
        int z = 0;
        int span = 0;
    };

    static bool isLeaf(const OctreeNode& node)
    {
        return std::all_of(node.children.begin(), node.children.end(), [](int child)
        {
            return child < 0;
        });
    }

    void subdivideLeaf(std::vector<OctreeNode>& nodes, int node_idx) const
    {
        const auto node_pos = static_cast<std::size_t>(node_idx);
        if (!isLeaf(nodes[node_pos]))
        {
            return;
        }
        if (nodes[node_pos].depth >= _max_depth)
        {
            throw std::logic_error("Poisson: cannot subdivide an octree leaf beyond maximum depth");
        }

        const Scalar half = nodes[node_pos].size * Scalar(0.5);
        const Scalar ox = nodes[node_pos].ox;
        const Scalar oy = nodes[node_pos].oy;
        const Scalar oz = nodes[node_pos].oz;
        const int child_depth = nodes[node_pos].depth + 1;
        for (int oct = 0; oct < 8; ++oct)
        {
            const Scalar qx = ox + (oct & 1 ? half : Scalar(0));
            const Scalar qy = oy + (oct & 2 ? half : Scalar(0));
            const Scalar qz = oz + (oct & 4 ? half : Scalar(0));
            const int child = createNode(nodes, qx, qy, qz, half, child_depth);
            nodes[node_pos].children[static_cast<std::size_t>(oct)] = child;
        }
    }

    void completeOctree(std::vector<OctreeNode>& nodes, int node_idx) const
    {
        const auto node_pos = static_cast<std::size_t>(node_idx);
        if (isLeaf(nodes[node_pos]))
        {
            return;
        }
        if (nodes[node_pos].depth >= _max_depth)
        {
            throw std::logic_error("Poisson: internal octree node exceeds maximum depth");
        }

        const Scalar half = nodes[node_pos].size * Scalar(0.5);
        const Scalar ox = nodes[node_pos].ox;
        const Scalar oy = nodes[node_pos].oy;
        const Scalar oz = nodes[node_pos].oz;
        const int child_depth = nodes[node_pos].depth + 1;
        for (int oct = 0; oct < 8; ++oct)
        {
            if (nodes[node_pos].children[static_cast<std::size_t>(oct)] >= 0)
            {
                continue;
            }
            const Scalar qx = ox + (oct & 1 ? half : Scalar(0));
            const Scalar qy = oy + (oct & 2 ? half : Scalar(0));
            const Scalar qz = oz + (oct & 4 ? half : Scalar(0));
            const int child = createNode(nodes, qx, qy, qz, half, child_depth);
            nodes[node_pos].children[static_cast<std::size_t>(oct)] = child;
        }

        const auto children = nodes[node_pos].children;
        for (int child : children)
        {
            completeOctree(nodes, child);
        }
    }

    void collectLeafCells(const std::vector<OctreeNode>& nodes, int node_idx,
                          int x, int y, int z, int span,
                          std::vector<LeafCell>& leaves) const
    {
        const auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (isLeaf(node))
        {
            leaves.push_back({node_idx, x, y, z, span});
            return;
        }

        const int child_span = span / 2;
        for (int oct = 0; oct < 8; ++oct)
        {
            const int child = node.children[static_cast<std::size_t>(oct)];
            if (child < 0)
            {
                throw std::logic_error("Poisson: incomplete octree encountered during balancing");
            }
            collectLeafCells(
                nodes,
                child,
                x + (oct & 1 ? child_span : 0),
                y + (oct & 2 ? child_span : 0),
                z + (oct & 4 ? child_span : 0),
                child_span,
                leaves);
        }
    }

    int findLeafForGridCell(const std::vector<OctreeNode>& nodes, int node_idx,
                            int x, int y, int z) const
    {
        int current = node_idx;
        while (current >= 0)
        {
            const auto& node = nodes[static_cast<std::size_t>(current)];
            if (isLeaf(node))
            {
                return current;
            }

            const int bit = 1 << (_max_depth - node.depth - 1);
            const int oct = ((x & bit) != 0 ? 1 : 0)
                | ((y & bit) != 0 ? 2 : 0)
                | ((z & bit) != 0 ? 4 : 0);
            current = node.children[static_cast<std::size_t>(oct)];
        }
        return -1;
    }

    int inspectLeafBalance(const std::vector<OctreeNode>& nodes, int root,
                           std::set<int>* leaves_to_subdivide) const
    {
        const int grid_resolution = 1 << _max_depth;
        std::vector<LeafCell> leaves;
        collectLeafCells(nodes, root, 0, 0, 0, grid_resolution, leaves);

        int maximum_difference = 0;
        for (const auto& leaf : leaves)
        {
            const int center_x = leaf.x + leaf.span / 2;
            const int center_y = leaf.y + leaf.span / 2;
            const int center_z = leaf.z + leaf.span / 2;
            for (int dz = -1; dz <= 1; ++dz)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dx == 0 && dy == 0 && dz == 0)
                        {
                            continue;
                        }
                        const int query_x = dx < 0 ? leaf.x - 1
                            : (dx > 0 ? leaf.x + leaf.span : center_x);
                        const int query_y = dy < 0 ? leaf.y - 1
                            : (dy > 0 ? leaf.y + leaf.span : center_y);
                        const int query_z = dz < 0 ? leaf.z - 1
                            : (dz > 0 ? leaf.z + leaf.span : center_z);
                        if (query_x < 0 || query_x >= grid_resolution
                            || query_y < 0 || query_y >= grid_resolution
                            || query_z < 0 || query_z >= grid_resolution)
                        {
                            continue;
                        }

                        const int neighbor_idx = findLeafForGridCell(
                            nodes, root, query_x, query_y, query_z);
                        if (neighbor_idx < 0 || neighbor_idx == leaf.nodeIndex)
                        {
                            continue;
                        }
                        const int leaf_depth = nodes[static_cast<std::size_t>(leaf.nodeIndex)].depth;
                        const int neighbor_depth = nodes[static_cast<std::size_t>(neighbor_idx)].depth;
                        const int difference = std::abs(leaf_depth - neighbor_depth);
                        maximum_difference = std::max(maximum_difference, difference);
                        if (difference > 1 && leaves_to_subdivide != nullptr)
                        {
                            leaves_to_subdivide->insert(
                                leaf_depth < neighbor_depth ? leaf.nodeIndex : neighbor_idx);
                        }
                    }
                }
            }
        }
        return maximum_difference;
    }

    int balanceOctree(std::vector<OctreeNode>& nodes, int node_idx) const
    {
        // Materialize empty siblings first so the leaves cover the root cube. Balance all
        // face-, edge-, and corner-adjacent leaves on the integer grid at maximum depth.
        completeOctree(nodes, node_idx);
        while (true)
        {
            std::set<int> leaves_to_subdivide;
            const int maximum_difference = inspectLeafBalance(
                nodes, node_idx, &leaves_to_subdivide);
            if (leaves_to_subdivide.empty())
            {
                return maximum_difference;
            }
            for (int leaf : leaves_to_subdivide)
            {
                subdivideLeaf(nodes, leaf);
            }
        }
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
            Scalar cx = node.ox + node.size * Scalar(0.5);
            Scalar cy = node.oy + node.size * Scalar(0.5);
            Scalar cz = node.oz + node.size * Scalar(0.5);
            Scalar wx = Scalar(1) - std::abs(px - cx) / (node.size * Scalar(0.5));
            Scalar wy = Scalar(1) - std::abs(py - cy) / (node.size * Scalar(0.5));
            Scalar wz = Scalar(1) - std::abs(pz - cz) / (node.size * Scalar(0.5));
            Scalar w = std::max(Scalar(0), wx * wy * wz);

            // Store normal components (weighted)
            node.div_x += w * nx;
            node.div_y += w * ny;
            node.div_z += w * nz;
            node.weight += w;
        }
        else
        {
            const Scalar half = node.size * Scalar(0.5);
            const Scalar cx = node.ox + half;
            const Scalar cy = node.oy + half;
            const Scalar cz = node.oz + half;
            const int oct = (px >= cx ? 1 : 0) | (py >= cy ? 2 : 0) | (pz >= cz ? 4 : 0);
            const int child = node.children[static_cast<std::size_t>(oct)];
            if (child >= 0)
            {
                splatNormal(nodes, child, px, py, pz, nx, ny, nz);
            }
        }
    }

    void solvePoisson(std::vector<OctreeNode>& nodes) const
    {
        std::vector<int> leaf_indices;
        collectPopulatedLeaves(nodes, 0, leaf_indices);

        int n_leaves = static_cast<int>(leaf_indices.size());

        // Compute divergence at each leaf using finite-difference on the octree
        // div(n) = dnx/dx + dny/dy + dnz/dz
        for (int li = 0; li < n_leaves; ++li)
        {
            int idx = leaf_indices[static_cast<std::size_t>(li)];
            auto& node = nodes[static_cast<std::size_t>(idx)];

            Scalar h = node.size;
            Scalar w = std::max(node.weight, Scalar(1e-10));
            // Normal field at this leaf
            Scalar vx = node.div_x / w;
            Scalar vy = node.div_y / w;
            Scalar vz = node.div_z / w;

            // Finite-difference divergence using face neighbors
            Scalar div = 0;
            Scalar off = h;  // neighbor at distance h
            Scalar nbs[6][3] = {
                {node.ox-off, node.oy, node.oz}, {node.ox+off, node.oy, node.oz},
                {node.ox, node.oy-off, node.oz}, {node.ox, node.oy+off, node.oz},
                {node.ox, node.oy, node.oz-off}, {node.ox, node.oy, node.oz+off}
            };

            for (int d = 0; d < 3; ++d)
            {
                // Forward difference: look at neighbor +h in direction d
                // For d=0 (x): find neighbor at (ox+h, oy, oz)
                // For d=1 (y): (ox, oy+h, oz)
                // For d=2 (z): (ox, oy, oz+h)
                int nb_p = findLeafAt(nodes, 0, nbs[2*d+1][0], nbs[2*d+1][1], nbs[2*d+1][2]);
                int nb_m = findLeafAt(nodes, 0, nbs[2*d][0], nbs[2*d][1], nbs[2*d][2]);

                if (nb_p >= 0 && nb_m >= 0)
                {
                    auto& np = nodes[static_cast<std::size_t>(nb_p)];
                    auto& nm = nodes[static_cast<std::size_t>(nb_m)];
                    Scalar vp = (d==0 ? np.div_x : (d==1 ? np.div_y : np.div_z)) / std::max(np.weight, Scalar(1e-10));
                    Scalar vm = (d==0 ? nm.div_x : (d==1 ? nm.div_y : nm.div_z)) / std::max(nm.weight, Scalar(1e-10));
                    div += (vp - vm) / (np.size + node.size);  // central difference
                }
                else if (nb_p >= 0)
                {
                    auto& np = nodes[static_cast<std::size_t>(nb_p)];
                    Scalar vp = (d==0 ? np.div_x : (d==1 ? np.div_y : np.div_z)) / std::max(np.weight, Scalar(1e-10));
                    div += (vp - (d==0 ? vx : (d==1 ? vy : vz))) / (np.size + node.size);
                }
                else if (nb_m >= 0)
                {
                    auto& nm = nodes[static_cast<std::size_t>(nb_m)];
                    Scalar vm = (d==0 ? nm.div_x : (d==1 ? nm.div_y : nm.div_z)) / std::max(nm.weight, Scalar(1e-10));
                    div += ((d==0 ? vx : (d==1 ? vy : vz)) - vm) / (nm.size + node.size);
                }
                // else: both neighbors missing, no contribution
            }

            node.divergence = div;
        }

        std::vector<int> node_to_row(nodes.size(), -1);
        for (int row = 0; row < n_leaves; ++row)
        {
            node_to_row[static_cast<std::size_t>(leaf_indices[static_cast<std::size_t>(row)])] = row;
        }

        std::set<std::pair<int, int>> edges;
        for (int row = 0; row < n_leaves; ++row)
        {
            const auto& node = nodes[static_cast<std::size_t>(
                leaf_indices[static_cast<std::size_t>(row)])];
            const Scalar off = node.size;
            const Scalar neighbors[6][3] = {
                {node.ox-off, node.oy, node.oz}, {node.ox+off, node.oy, node.oz},
                {node.ox, node.oy-off, node.oz}, {node.ox, node.oy+off, node.oz},
                {node.ox, node.oy, node.oz-off}, {node.ox, node.oy, node.oz+off}
            };
            for (const auto& position : neighbors)
            {
                const int neighbor_node = findLeafAt(
                    nodes, 0, position[0], position[1], position[2]);
                if (neighbor_node < 0)
                {
                    continue;
                }
                const int neighbor_row = node_to_row[static_cast<std::size_t>(neighbor_node)];
                if (neighbor_row >= 0 && neighbor_row != row)
                {
                    edges.emplace(std::min(row, neighbor_row), std::max(row, neighbor_row));
                }
            }
        }

        std::vector<int> degrees(static_cast<std::size_t>(n_leaves), 0);
        for (const auto& edge : edges)
        {
            ++degrees[static_cast<std::size_t>(edge.first)];
            ++degrees[static_cast<std::size_t>(edge.second)];
        }

        // Each connected graph component has one constant Laplacian null mode. Deterministically
        // choose its lowest row as an anchor so the assembled matrix is strictly positive definite.
        std::vector<int> parents(static_cast<std::size_t>(n_leaves));
        for (int row = 0; row < n_leaves; ++row)
        {
            parents[static_cast<std::size_t>(row)] = row;
        }
        auto find_root = [&](int row)
        {
            int root = row;
            while (parents[static_cast<std::size_t>(root)] != root)
            {
                root = parents[static_cast<std::size_t>(root)];
            }
            while (parents[static_cast<std::size_t>(row)] != row)
            {
                const int next = parents[static_cast<std::size_t>(row)];
                parents[static_cast<std::size_t>(row)] = root;
                row = next;
            }
            return root;
        };
        for (const auto& edge : edges)
        {
            const int first_root = find_root(edge.first);
            const int second_root = find_root(edge.second);
            if (first_root != second_root)
            {
                const int anchor = std::min(first_root, second_root);
                const int merged = std::max(first_root, second_root);
                parents[static_cast<std::size_t>(merged)] = anchor;
            }
        }
        for (int row = 0; row < n_leaves; ++row)
        {
            parents[static_cast<std::size_t>(row)] = find_root(row);
        }

        std::vector<plamatrix::Index> coo_rows;
        std::vector<plamatrix::Index> coo_columns;
        std::vector<Scalar> coo_values;
        coo_rows.reserve(static_cast<std::size_t>(n_leaves) + edges.size() * 2);
        coo_columns.reserve(coo_rows.capacity());
        coo_values.reserve(coo_rows.capacity());
        for (int row = 0; row < n_leaves; ++row)
        {
            const int degree = degrees[static_cast<std::size_t>(row)];
            Scalar diagonal = static_cast<Scalar>(degree);
            if (parents[static_cast<std::size_t>(row)] == row)
            {
                diagonal += static_cast<Scalar>(std::max(1, degree));
            }
            coo_rows.push_back(row);
            coo_columns.push_back(row);
            coo_values.push_back(diagonal);
        }
        for (const auto& edge : edges)
        {
            coo_rows.push_back(edge.first);
            coo_columns.push_back(edge.second);
            coo_values.push_back(Scalar(-1));
            coo_rows.push_back(edge.second);
            coo_columns.push_back(edge.first);
            coo_values.push_back(Scalar(-1));
        }

        Vector rhs(n_leaves);
        for (int row = 0; row < n_leaves; ++row)
        {
            const auto& node = nodes[static_cast<std::size_t>(
                leaf_indices[static_cast<std::size_t>(row)])];
            const Scalar value = -node.size * node.size * node.divergence;
            rhs(row, 0) = value;
        }

        plamatrix::SparseMatrix<Scalar, plamatrix::RowMajor, plamatrix::Index> matrix(n_leaves, n_leaves);
        std::vector<plamatrix::Triplet<Scalar, plamatrix::Index>> triplets;
        triplets.reserve(coo_values.size());
        for (std::size_t entry = 0; entry < coo_values.size(); ++entry)
        {
            triplets.emplace_back(coo_rows[entry], coo_columns[entry], coo_values[entry]);
        }
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        PoissonSystem<Scalar> system{
            std::move(matrix), std::move(rhs), leaf_indices};
        Vector solution(n_leaves);
        solution.setZero();
        plamatrix::internal::SolveOptions solver_options;
        solver_options.maxIterations = _solver_iters;
        solver_options.relativeTolerance = _solver_tolerance;
        solver_options.absoluteTolerance = 0.0;
        solver_options.useJacobiPreconditioner = true;
        solver_options.requireConvergence = false;

        const bool try_cuda = _processing_device == ProcessingDevice::CUDA
            || _processing_device == ProcessingDevice::Auto;
#ifdef PLAPOINT_WITH_OPENCL
        const bool try_opencl = _processing_device == ProcessingDevice::OpenCL
            || _processing_device == ProcessingDevice::Auto;
#endif
        auto append_fallback = [&](const std::string& backend, const std::string& reason)
        {
            if (!_lastReport.fallbackReason.empty())
            {
                _lastReport.fallbackReason += "; ";
            }
            _lastReport.fallbackReason += backend + ": " + reason;
            _lastReport.usedFallback = true;
        };
        bool solved = false;
        bool accelerator_solve_nonconverged = false;
        double fallback_absolute_tolerance = solver_options.absoluteTolerance;
#ifdef PLAPOINT_WITH_CUDA
        if (try_cuda)
        {
            try
            {
                auto context = plamatrix::internal::ExecutionContext::create({plamatrix::internal::Backend::Cuda, 0});
                auto matrix_gpu = plamatrix::internal::ResidentCsrMatrix<Scalar>::copyFrom(system.matrix, context);
                auto rhs_gpu = plamatrix::internal::ResidentVector<Scalar>::copyFrom(system.rhs, context);
                auto solution_gpu = plamatrix::internal::ResidentVector<Scalar>::copyFrom(solution, context);
                const auto gpu_report = plamatrix::internal::pcg(
                    matrix_gpu, rhs_gpu, solution_gpu, solver_options, context);
                _lastReport.solver = gpu_report;
                solution = solution_gpu.toHostVector();
                if (!gpu_report.converged)
                {
                    const std::string reason =
                        std::string("Poisson GPU PCG did not converge after ") +
                        std::to_string(gpu_report.iterations) + " iterations; residual=" +
                        std::to_string(gpu_report.finalResidual);
                    if (_processing_device == ProcessingDevice::CUDA)
                    {
                        throw std::runtime_error(reason);
                    }
                    accelerator_solve_nonconverged = true;
                    // Resume on CPU from the CUDA iterate. Preserve the convergence target
                    // from the original zero guess and grant the fallback its own solve budget.
                    fallback_absolute_tolerance = std::max(
                        fallback_absolute_tolerance,
                        solver_options.relativeTolerance * gpu_report.initialResidual);
                    append_fallback("CUDA", reason);
                }
                else
                {
                    _lastReport.solverDevice = ProcessingDevice::CUDA;
                    solved = true;
                }
            }
            catch (const std::exception& error)
            {
                if (_processing_device == ProcessingDevice::CUDA)
                {
                    throw std::runtime_error(
                        std::string("Poisson GPU PCG failed for ") +
                        std::to_string(n_leaves) + " leaves and " +
                        std::to_string(_solver_iters) + " iterations: " + error.what());
                }
                append_fallback("CUDA", error.what());
            }
        }
#else
        if (_processing_device == ProcessingDevice::CUDA)
        {
            throw std::runtime_error("Poisson GPU PCG requires PLAPOINT_WITH_CUDA=ON");
        }
        if (_processing_device == ProcessingDevice::Auto)
        {
            append_fallback("CUDA", "PlaPoint was built without CUDA support");
        }
#endif
#ifdef PLAPOINT_WITH_OPENCL
        if (!solved && try_opencl)
        {
            try
            {
                auto context = plamatrix::internal::ExecutionContext::create({plamatrix::internal::Backend::OpenCl, 0});
                auto matrix_opencl = plamatrix::internal::ResidentCsrMatrix<Scalar>::copyFrom(system.matrix, context);
                auto rhs_opencl = plamatrix::internal::ResidentVector<Scalar>::copyFrom(system.rhs, context);
                auto solution_opencl = plamatrix::internal::ResidentVector<Scalar>::copyFrom(solution, context);
                const auto opencl_report = plamatrix::internal::pcg(
                    matrix_opencl, rhs_opencl, solution_opencl, solver_options, context);
                solution = solution_opencl.toHostVector();
                _lastReport.solver = opencl_report;
                if (!opencl_report.converged)
                {
                    const std::string reason =
                        std::string("Poisson OpenCL PCG did not converge after ") +
                        std::to_string(opencl_report.iterations) + " iterations; residual=" +
                        std::to_string(opencl_report.finalResidual);
                    if (_processing_device == ProcessingDevice::OpenCL)
                    {
                        throw std::runtime_error(reason);
                    }
                    accelerator_solve_nonconverged = true;
                    fallback_absolute_tolerance = std::max(
                        fallback_absolute_tolerance,
                        solver_options.relativeTolerance * opencl_report.initialResidual);
                    append_fallback("OpenCL", reason);
                }
                else
                {
                    _lastReport.solverDevice = ProcessingDevice::OpenCL;
                    solved = true;
                }
            }
            catch (const std::exception& error)
            {
                if (_processing_device == ProcessingDevice::OpenCL)
                {
                    throw std::runtime_error(
                        std::string("Poisson OpenCL PCG failed for ") +
                        std::to_string(n_leaves) + " leaves and " +
                        std::to_string(_solver_iters) + " iterations: " + error.what());
                }
                append_fallback("OpenCL", error.what());
            }
        }
#else
        if (_processing_device == ProcessingDevice::OpenCL)
        {
            throw std::runtime_error("Poisson OpenCL PCG requires PLAPOINT_WITH_OPENCL=ON");
        }
        if (!solved && _processing_device == ProcessingDevice::Auto)
        {
            append_fallback("OpenCL", "PlaPoint was built without OpenCL support");
        }
#endif
        if (!solved)
        {
            auto cpu_solver_options = solver_options;
            cpu_solver_options.absoluteTolerance = fallback_absolute_tolerance;
            if (accelerator_solve_nonconverged)
            {
                cpu_solver_options.maxIterations = static_cast<int>(std::min<long long>(
                    std::numeric_limits<int>::max(),
                    2LL * solver_options.maxIterations));
            }
            using SparseMatrix = decltype(system.matrix);
            plamatrix::ConjugateGradient<SparseMatrix, plamatrix::Lower | plamatrix::Upper> solver;
            solver.setMaxIterations(cpu_solver_options.maxIterations);
            const double rhs_norm = static_cast<double>(system.rhs.norm());
            const double initial_residual = static_cast<double>((system.rhs - system.matrix * solution).norm());
            const double target_residual = std::max(
                cpu_solver_options.absoluteTolerance,
                cpu_solver_options.relativeTolerance * initial_residual);
            solver.setTolerance(rhs_norm > 0.0 ? static_cast<Scalar>(target_residual / rhs_norm) : Scalar(0));
            solver.compute(system.matrix);
            solution = solver.solveWithGuess(system.rhs, solution);
            _lastReport.solver.converged = solver.info() == plamatrix::Success;
            _lastReport.solver.iterations = static_cast<int>(solver.iterations());
            _lastReport.solver.initialResidual = initial_residual;
            _lastReport.solver.finalResidual =
                static_cast<double>((system.rhs - system.matrix * solution).norm());
            _lastReport.solverDevice = ProcessingDevice::CPU;
            if (accelerator_solve_nonconverged && !_lastReport.solver.converged)
            {
                _lastReport.leafCount = static_cast<std::size_t>(n_leaves);
                throw std::runtime_error(
                    std::string("Poisson Auto PCG failed; accelerator attempts: ") +
                    _lastReport.fallbackReason + "; CPU did not converge after " +
                    std::to_string(_lastReport.solver.iterations) +
                    " iterations; residual=" +
                    std::to_string(_lastReport.solver.finalResidual));
            }
        }
        _lastReport.leafCount = static_cast<std::size_t>(n_leaves);
        for (int row = 0; row < n_leaves; ++row)
        {
            nodes[static_cast<std::size_t>(leaf_indices[static_cast<std::size_t>(row)])].solution =
                solution(row, 0);
        }
        _lastSystem = std::make_shared<PoissonSystem<Scalar>>(std::move(system));
    }

    std::size_t countOctreeLeaves(const std::vector<OctreeNode>& nodes, int node_idx) const
    {
        const auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (isLeaf(node))
        {
            return 1;
        }

        std::size_t count = 0;
        for (int child : node.children)
        {
            if (child >= 0)
            {
                count += countOctreeLeaves(nodes, child);
            }
        }
        return count;
    }

    void collectPopulatedLeaves(const std::vector<OctreeNode>& nodes, int node_idx,
                                std::vector<int>& leaves) const
    {
        const auto& node = nodes[static_cast<std::size_t>(node_idx)];
        if (isLeaf(node))
        {
            if (node.point_count > 0)
            {
                leaves.push_back(node_idx);
            }
            return;
        }
        for (int child : node.children)
        {
            if (child >= 0)
            {
                collectPopulatedLeaves(nodes, child, leaves);
            }
        }
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
        if (is_leaf) return node.point_count > 0 ? node_idx : -1;

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

    void orientFacesWithInputNormals(const Matrix& vertices, Matrix& faces) const
    {
        if (!_cloud || !_cloud->hasNormals() || faces.rows() == 0) return;
        const auto* normals = _cloud->normals();
        constexpr double kDegenerateArea = 1e-20;
        constexpr double kOrientationEpsilon = 1e-12;
        using SearchPoint = std::conditional_t<std::is_same_v<Scalar, double>, PointXYZd, PointXYZ>;
        auto search_cloud = std::make_shared<PointCloud<SearchPoint>>();
        search_cloud->reserve(_cloud->size());
        const auto& source_points = _cloud->points();
        for (plamatrix::Index row = 0; row < source_points.rows(); ++row)
        {
            search_cloud->emplace_back(source_points(row, 0), source_points(row, 1), source_points(row, 2));
        }
        search::KdTree<SearchPoint> spatial_index;
        spatial_index.setInputCloud(search_cloud);
        ++_lastReport.orientationIndexBuildCount;

        for (plamatrix::Index f = 0; f < faces.rows(); ++f)
        {
            int idx[3] = {0, 0, 0};
            bool valid_face = true;
            for (int c = 0; c < 3; ++c)
            {
                const double raw = static_cast<double>(faces.operator()(f, c));
                const double rounded = std::round(raw);
                if (!std::isfinite(raw) || std::abs(raw - rounded) > 1e-4)
                {
                    valid_face = false;
                    break;
                }
                idx[c] = static_cast<int>(rounded);
                if (idx[c] < 0 || idx[c] >= vertices.rows())
                {
                    valid_face = false;
                    break;
                }
            }
            if (!valid_face) continue;

            const double ax = static_cast<double>(vertices.operator()(idx[0], 0));
            const double ay = static_cast<double>(vertices.operator()(idx[0], 1));
            const double az = static_cast<double>(vertices.operator()(idx[0], 2));
            const double bx = static_cast<double>(vertices.operator()(idx[1], 0));
            const double by = static_cast<double>(vertices.operator()(idx[1], 1));
            const double bz = static_cast<double>(vertices.operator()(idx[1], 2));
            const double cx = static_cast<double>(vertices.operator()(idx[2], 0));
            const double cy = static_cast<double>(vertices.operator()(idx[2], 1));
            const double cz = static_cast<double>(vertices.operator()(idx[2], 2));

            const double ux = bx - ax;
            const double uy = by - ay;
            const double uz = bz - az;
            const double vx = cx - ax;
            const double vy = cy - ay;
            const double vz = cz - az;
            const double face_nx = uy * vz - uz * vy;
            const double face_ny = uz * vx - ux * vz;
            const double face_nz = ux * vy - uy * vx;
            const double area2 = face_nx * face_nx + face_ny * face_ny + face_nz * face_nz;
            if (area2 <= kDegenerateArea) continue;

            const double center_x = (ax + bx + cx) / 3.0;
            const double center_y = (ay + by + cy) / 3.0;
            const double center_z = (az + bz + cz) / 3.0;

            std::vector<int> nearest;
            std::vector<float> squared_distances;
            spatial_index.nearestKSearch(
                SearchPoint(static_cast<Scalar>(center_x),
                            static_cast<Scalar>(center_y),
                            static_cast<Scalar>(center_z)),
                1, nearest, squared_distances);
            ++_lastReport.orientationQueryCount;
            if (nearest.empty())
            {
                continue;
            }

            const auto normal_row = static_cast<plamatrix::Index>(nearest.front());
            const double nx = static_cast<double>(normals->operator()(normal_row, 0));
            const double ny = static_cast<double>(normals->operator()(normal_row, 1));
            const double nz = static_cast<double>(normals->operator()(normal_row, 2));
            const double dot = face_nx * nx + face_ny * ny + face_nz * nz;
            if (dot < -kOrientationEpsilon)
            {
                const Scalar tmp = faces.operator()(f, 1);
                faces.operator()(f, 1) = faces.operator()(f, 2);
                faces.operator()(f, 2) = tmp;
            }
        }
    }

    std::shared_ptr<const PointCloudType> _cloud;
    static constexpr int kMaxDepth = 8;
    int _max_depth = 6;
    int _solver_iters = 30;
    double _solver_tolerance = 1.0e-6;
    ProcessingDevice _processing_device = ProcessingDevice::CPU;
    mutable PoissonProcessingReport _lastReport;
    mutable std::shared_ptr<PoissonSystem<Scalar>> _lastSystem;
};

} // namespace mesh
} // namespace plapoint
