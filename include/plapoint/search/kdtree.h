#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/knn.h>
#include <plapoint/search/gpu_kdtree_policy.h>
#include <plamatrix/ops/point_cloud.h>

#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace plapoint {
namespace search {

namespace detail {

inline std::size_t checkedSizeProduct(std::size_t lhs, std::size_t rhs, const char* label)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw std::overflow_error(std::string(label) + " size exceeds size_t range");
    }
    return lhs * rhs;
}

template <typename T>
inline std::size_t checkedByteCount(std::size_t count, const char* label)
{
    return checkedSizeProduct(count, sizeof(T), label);
}

template <typename Scalar>
inline bool pointCoordinateLess(Scalar lhs, int lhs_idx, Scalar rhs, int rhs_idx)
{
    const bool lhs_finite = std::isfinite(lhs);
    const bool rhs_finite = std::isfinite(rhs);
    if (lhs_finite != rhs_finite)
    {
        return lhs_finite;
    }
    if (lhs_finite && lhs != rhs)
    {
        return lhs < rhs;
    }
    return lhs_idx < rhs_idx;
}

} // namespace detail

template <typename Scalar>
struct KdTreeNode
{
    int point_idx;
    int left;
    int right;
    int split_dim;
    Scalar split_val;
};

template <typename Scalar, plamatrix::Device Dev>
class KdTree
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud)
    {
        _cloud = cloud;
        _nodes.clear();
        _host_points.reset();
        _host_points_identity.reset();
        _host_points_data = nullptr;
#ifdef PLAPOINT_WITH_CUDA
        _gpu_spatial_index = {};
        _gpu_host_occupied_cells.clear();
        _gpu_host_occupancy_valid = false;
        _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
        _built = false;
    }

    void build()
    {
        if (!_cloud)
        {
            throw std::runtime_error("KdTree: input cloud not set");
        }
        _nodes.clear();
        if constexpr (Dev == plamatrix::Device::GPU)
        {
            _host_points = std::make_shared<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(
                _cloud->points().toCpu());
            _host_points_revision = _cloud->pointsRevision();
            _host_points_identity = _cloud->pointsIdentity();
            _host_points_data = _cloud->points().data();
        }
        std::vector<int> indices(static_cast<std::size_t>(_cloud->size()));
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = checkedInt(i, "KdTree: point index");
        }
        _nodes.reserve(indices.size());
        buildRecursive(indices, 0, checkedInt(indices.size(), "KdTree: point count") - 1, 0);
        _built = true;
    }

    bool isBuilt() const noexcept
    {
        return _built;
    }

    struct DistComparator
    {
        bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b) const
        {
            return a.first < b.first;
        }
    };

    std::vector<int> nearestKSearch(const plamatrix::Vec3<Scalar>& query, int k) const
    {
        ensureBuilt();
        validateQuery(query);
        std::vector<int> result;
        if (_nodes.empty() || k <= 0) return result;

        std::vector<std::pair<double, int>> heap;
        heap.reserve(std::min(static_cast<std::size_t>(k), _nodes.size()));
        nearestKSearchInto(query, k, result, heap);
        return result;
    }

    std::vector<int> radiusSearch(const plamatrix::Vec3<Scalar>& query, Scalar radius) const
    {
        ensureBuilt();
        validateQuery(query);
        std::vector<int> result;
        if (!std::isfinite(radius) || radius < Scalar(0))
        {
            throw std::invalid_argument("KdTree: radius must be finite and non-negative");
        }
        if (_nodes.empty()) return result;
        radiusSearchRecursive(query, radius, 0, result);
        return result;
    }

    /// Batch K-nearest neighbor search for multiple query points.
    /// On GPU: uses brute-force CUDA kernel (fast for up to ~100K points).
    /// On CPU: loops over queries using the kd-tree.
    /// @param queries   M x 3 matrix of query points
    /// @param k         number of neighbors per query
    /// @return          vector of M vectors, each with up to K finite neighbor indices
    std::vector<std::vector<int>> batchNearestKSearch(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& queries, int k) const
    {
        ensureBuilt();
        if (queries.cols() != 3)
        {
            throw std::invalid_argument("KdTree: queries must be an Mx3 matrix");
        }
        validateQueries(queries);
        int M = checkedInt(static_cast<std::size_t>(queries.rows()), "KdTree: query count");
        std::vector<std::vector<int>> results(static_cast<std::size_t>(M));
        if (M <= 0 || k <= 0)
        {
#ifdef PLAPOINT_WITH_CUDA
            _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
            return results;
        }
        if (!_cloud || _cloud->size() == 0)
        {
#ifdef PLAPOINT_WITH_CUDA
            _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
            return results;
        }

        if constexpr (Dev == plamatrix::Device::CPU)
        {
            const int N = checkedInt(_cloud->size(), "KdTree: point count");
            std::vector<std::pair<double, int>> heap;
            heap.reserve(std::min(static_cast<std::size_t>(k), static_cast<std::size_t>(N)));
            for (int i = 0; i < M; ++i)
            {
                plamatrix::Vec3<Scalar> q{queries(i, 0), queries(i, 1), queries(i, 2)};
                nearestKSearchInto(q, k, results[static_cast<std::size_t>(i)], heap);
            }
        }
        else
        {
#ifndef PLAPOINT_WITH_CUDA
            throw std::runtime_error("PlaPoint was built without CUDA support");
#else
            int N = checkedInt(_cloud->size(), "KdTree: point count");
            if (N <= 0)
            {
                return results;
            }
            const int K_use = std::min(k, N);
            if (K_use > 32)
            {
                refreshGpuHostPoints();
                for (int i = 0; i < M; ++i)
                {
                    plamatrix::Vec3<Scalar> q{queries(i, 0), queries(i, 1), queries(i, 2)};
                    struct Candidate
                    {
                        DistanceOrderKey distance;
                        int index;
                    };
                    std::vector<Candidate> candidates;
                    candidates.reserve(static_cast<std::size_t>(N));
                    for (int point = 0; point < N; ++point)
                    {
                        const auto distance = makeDistanceOrderKey(q, pointVec(point));
                        if (distance.exponent != std::numeric_limits<int>::max())
                        {
                            candidates.push_back({distance, point});
                        }
                    }
                    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs)
                    {
                        return distanceKeyLess(lhs.distance, rhs.distance)
                            || (!distanceKeyLess(rhs.distance, lhs.distance)
                                && lhs.index < rhs.index);
                    });
                    auto& row = results[static_cast<std::size_t>(i)];
                    const auto count = std::min(
                        static_cast<std::size_t>(K_use), candidates.size());
                    row.reserve(count);
                    for (std::size_t candidate = 0; candidate < count; ++candidate)
                    {
                        row.push_back(candidates[candidate].index);
                    }
                }
                _last_neighbor_backend = gpu::GpuNeighborBackend::CpuBruteForce;
                return results;
            }

            auto gpu_queries = queries.toGpu();
            const bool enough_work = static_cast<std::size_t>(N)
                >= (detail::kIndexedKnnWorkThreshold + static_cast<std::size_t>(M) - 1)
                    / static_cast<std::size_t>(M);
            bool use_indexed = enough_work;
            bool rebuilt_index = false;
            Scalar cell_size = Scalar(1);
            if (use_indexed)
            {
                refreshGpuHostPoints();
                cell_size = detail::estimateKnnCellSize(*_host_points);
                if (!std::isfinite(cell_size) || cell_size <= Scalar(0))
                {
                    use_indexed = false;
                }
                else
                {
                    try
                    {
                        if (!_gpu_spatial_index.matches(*_cloud, cell_size))
                        {
                            _gpu_spatial_index.build(*_cloud, cell_size);
                            rebuilt_index = true;
                        }
                    }
                    catch (const std::overflow_error&)
                    {
                        use_indexed = false;
                    }
                    catch (const std::invalid_argument&)
                    {
                        use_indexed = false;
                    }
                }
                if (use_indexed && !detail::indexedGridIsPathological(_gpu_spatial_index))
                {
                    const auto identity = _cloud->pointsIdentity();
                    const auto revision = _cloud->pointsRevision();
                    if (rebuilt_index
                        || !_gpu_host_occupancy_valid
                        || _gpu_host_occupancy_identity != identity
                        || _gpu_host_occupancy_revision != revision
                        || _gpu_host_occupancy_cell_size != cell_size
                        || _cloud->hasUntrackedMutablePointAlias())
                    {
                        _gpu_host_occupancy_valid = detail::buildHostGridOccupancy(
                            *_host_points, cell_size, _gpu_host_occupied_cells);
                        _gpu_host_occupancy_identity = identity;
                        _gpu_host_occupancy_revision = revision;
                        _gpu_host_occupancy_cell_size = cell_size;
                    }
                    use_indexed = _gpu_host_occupancy_valid
                        && detail::queriesFitIndexedShells(
                            queries, cell_size, _gpu_host_occupied_cells);
                }
                else
                {
                    use_indexed = false;
                }
            }

            plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::CPU> indexed_indices;
            plamatrix::DenseMatrix<int, plamatrix::Device::CPU> brute_indices;
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> flat_dst;
            if (use_indexed)
            {
                auto indexed = _gpu_spatial_index.knnSearchAsync(
                    gpu_queries, K_use, _gpu_query_workspace, nullptr);
                indexed_indices = indexed.indices.toCpu();
                flat_dst = indexed.squaredDistances.toCpu();
                _last_neighbor_backend = gpu::GpuNeighborBackend::UniformGrid;
            }
            else
            {
                plamatrix::DenseMatrix<int, plamatrix::Device::GPU> gpu_indices(
                    queries.rows(), K_use);
                plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> gpu_dists(
                    queries.rows(), K_use);
                gpu::batchKnnDevice(gpu_queries, _cloud->points(), K_use, gpu_indices, gpu_dists);
                brute_indices = gpu_indices.toCpu();
                flat_dst = gpu_dists.toCpu();
                _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
            }

            for (int i = 0; i < M; ++i)
            {
                auto& row = results[static_cast<std::size_t>(i)];
                row.reserve(static_cast<std::size_t>(K_use));
                for (int j = 0; j < K_use; ++j)
                {
                    const auto indexed_value = use_indexed
                        ? indexed_indices(i, j)
                        : plamatrix::Index(-1);
                    const int idx = use_indexed
                        ? (indexed_value >= 0
                               && indexed_value <= std::numeric_limits<int>::max()
                               ? static_cast<int>(indexed_value)
                               : -1)
                        : brute_indices(i, j);
                    const Scalar dist = flat_dst(i, j);
                    if (idx >= 0 && idx < N && std::isfinite(dist))
                    {
                        row.push_back(idx);
                    }
                }
            }
#endif
        }

        return results;
    }

    gpu::GpuNeighborBackend lastNeighborBackend() const noexcept
    {
#ifdef PLAPOINT_WITH_CUDA
        return _last_neighbor_backend;
#else
        return gpu::GpuNeighborBackend::BruteForce;
#endif
    }

#if defined(PLAPOINT_ENABLE_TESTING) && defined(PLAPOINT_WITH_CUDA)
    std::size_t gpuBatchQueryScalarCapacityForTesting() const
    {
        return 0;
    }

    std::size_t gpuBatchResultCapacityForTesting() const
    {
        return 0;
    }
#endif

private:
#ifdef PLAPOINT_WITH_CUDA
    void refreshGpuHostPoints() const
    {
        if constexpr (Dev == plamatrix::Device::GPU)
        {
            const auto revision = _cloud->pointsRevision();
            const auto* data = _cloud->points().data();
            if (!_host_points
                || _host_points_revision != revision
                || _host_points_identity != _cloud->pointsIdentity()
                || _host_points_data != data
                || _cloud->hasUntrackedMutablePointAlias())
            {
                _host_points = std::make_shared<
                    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(
                    _cloud->points().toCpu());
                _host_points_revision = revision;
                _host_points_identity = _cloud->pointsIdentity();
                _host_points_data = data;
            }
        }
    }
#endif

    void ensureBuilt() const
    {
        if (!_built)
        {
            throw std::runtime_error("KdTree: build() must be called before search");
        }
    }

    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return _cloud->points()(idx, dim);
        }
        else
        {
            if (_host_points)
            {
                return (*_host_points)(idx, dim);
            }
            return _cloud->points().getValue(idx, dim);
        }
    }

    plamatrix::Vec3<Scalar> pointVec(int idx) const
    {
        return {pointCoord(idx, 0), pointCoord(idx, 1), pointCoord(idx, 2)};
    }

    std::vector<int> filterFiniteNeighbors(
        const plamatrix::Vec3<Scalar>& query,
        const std::vector<int>& neighbors,
        int point_count) const
    {
        std::vector<int> filtered;
        filtered.reserve(neighbors.size());
        for (int idx : neighbors)
        {
            if (idx < 0 || idx >= point_count)
            {
                continue;
            }
            const double d = finiteDistance(query, pointVec(idx));
            if (std::isfinite(d))
            {
                filtered.push_back(idx);
            }
        }
        return filtered;
    }

    Scalar distSq(const plamatrix::Vec3<Scalar>& a, const plamatrix::Vec3<Scalar>& b) const
    {
        Scalar dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    struct DistanceOrderKey
    {
        int exponent;
        double mantissa;
    };

    static DistanceOrderKey absoluteDifferenceKey(double lhs, double rhs)
    {
        if (lhs == rhs)
        {
            return {std::numeric_limits<int>::min(), 0.0};
        }
        if (std::signbit(lhs) == std::signbit(rhs) || lhs == 0.0 || rhs == 0.0)
        {
            int exponent = 0;
            const double mantissa = std::frexp(std::abs(lhs - rhs), &exponent);
            return {exponent, mantissa};
        }
        int lhs_exponent = 0;
        int rhs_exponent = 0;
        const double lhs_mantissa = std::frexp(std::abs(lhs), &lhs_exponent);
        const double rhs_mantissa = std::frexp(std::abs(rhs), &rhs_exponent);
        const int exponent = std::max(lhs_exponent, rhs_exponent);
        const double sum = std::ldexp(lhs_mantissa, lhs_exponent - exponent)
            + std::ldexp(rhs_mantissa, rhs_exponent - exponent);
        int adjustment = 0;
        const double mantissa = std::frexp(sum, &adjustment);
        return {exponent + adjustment, mantissa};
    }

    static DistanceOrderKey makeDistanceOrderKey(
        const plamatrix::Vec3<Scalar>& a,
        const plamatrix::Vec3<Scalar>& b)
    {
        const double coordinates[6] = {
            static_cast<double>(a.x), static_cast<double>(a.y), static_cast<double>(a.z),
            static_cast<double>(b.x), static_cast<double>(b.y), static_cast<double>(b.z)};
        for (double coordinate : coordinates)
        {
            if (!std::isfinite(coordinate))
            {
                return {std::numeric_limits<int>::max(),
                        std::numeric_limits<double>::max()};
            }
        }
        const DistanceOrderKey components[3] = {
            absoluteDifferenceKey(coordinates[0], coordinates[3]),
            absoluteDifferenceKey(coordinates[1], coordinates[4]),
            absoluteDifferenceKey(coordinates[2], coordinates[5])};
        const int exponent = std::max({
            components[0].exponent, components[1].exponent, components[2].exponent});
        if (exponent == std::numeric_limits<int>::min())
        {
            return {exponent, 0.0};
        }
        const auto scaled = [exponent](const DistanceOrderKey& component)
        {
            return component.mantissa == 0.0
                ? 0.0
                : std::ldexp(component.mantissa, component.exponent - exponent);
        };
        const double distance = std::hypot(
            std::hypot(scaled(components[0]), scaled(components[1])),
            scaled(components[2]));
        int adjustment = 0;
        const double mantissa = std::frexp(distance, &adjustment);
        return {exponent + adjustment, mantissa};
    }

    static bool distanceKeyLess(const DistanceOrderKey& lhs, const DistanceOrderKey& rhs)
    {
        return lhs.exponent < rhs.exponent
            || (lhs.exponent == rhs.exponent && lhs.mantissa < rhs.mantissa);
    }

    static double finiteDistance(
        const plamatrix::Vec3<Scalar>& a,
        const plamatrix::Vec3<Scalar>& b)
    {
        const auto distance = makeDistanceOrderKey(a, b);
        if (distance.exponent == std::numeric_limits<int>::max())
        {
            return std::numeric_limits<double>::infinity();
        }
        if (distance.exponent == std::numeric_limits<int>::min())
        {
            return 0.0;
        }
        if (distance.exponent > std::numeric_limits<double>::max_exponent)
        {
            return std::numeric_limits<double>::max();
        }
        return std::ldexp(distance.mantissa, distance.exponent);
    }

    static bool finiteDistanceWithinRadius(
        const plamatrix::Vec3<Scalar>& a,
        const plamatrix::Vec3<Scalar>& b,
        Scalar radius)
    {
        const auto distance = makeDistanceOrderKey(a, b);
        int radius_exponent = 0;
        const double radius_mantissa = std::frexp(static_cast<double>(radius), &radius_exponent);
        const DistanceOrderKey radius_key{
            radius_mantissa == 0.0 ? std::numeric_limits<int>::min() : radius_exponent,
            radius_mantissa};
        return distance.exponent != std::numeric_limits<int>::max()
            && !distanceKeyLess(radius_key, distance);
    }

    static void validateQuery(const plamatrix::Vec3<Scalar>& query)
    {
        if (!std::isfinite(query.x) || !std::isfinite(query.y) || !std::isfinite(query.z))
        {
            throw std::invalid_argument("KdTree: query coordinates must be finite");
        }
    }

    static void validateQueries(const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& queries)
    {
        for (plamatrix::Index r = 0; r < queries.rows(); ++r)
        {
            for (plamatrix::Index c = 0; c < queries.cols(); ++c)
            {
                if (!std::isfinite(queries(r, c)))
                {
                    throw std::invalid_argument("KdTree: query coordinates must be finite");
                }
            }
        }
    }

    int buildRecursive(std::vector<int>& indices, int start, int end, int depth)
    {
        if (start > end) return -1;

        int dim = depth % 3;
        int mid = start + (end - start) / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end + 1,
                         [&](int a, int b) {
                             return detail::pointCoordinateLess(pointCoord(a, dim), a, pointCoord(b, dim), b);
                         });

        int node_idx = static_cast<int>(_nodes.size());
        KdTreeNode<Scalar> node{};
        node.point_idx = indices[static_cast<std::size_t>(mid)];
        node.split_dim = dim;
        node.split_val = pointCoord(node.point_idx, dim);
        _nodes.push_back(node);

        _nodes[static_cast<std::size_t>(node_idx)].left  = buildRecursive(indices, start, mid - 1, depth + 1);
        _nodes[static_cast<std::size_t>(node_idx)].right = buildRecursive(indices, mid + 1, end, depth + 1);
        return node_idx;
    }

    void nearestKSearchInto(const plamatrix::Vec3<Scalar>& query, int k,
                            std::vector<int>& result,
                            std::vector<std::pair<double, int>>& heap) const
    {
        result.clear();
        heap.clear();
        if (_nodes.empty() || k <= 0)
        {
            return;
        }

        nearestKSearchRecursive(query, k, 0, heap);

        result.resize(heap.size());
        const DistComparator compare;
        for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i)
        {
            std::pop_heap(heap.begin(), heap.end(), compare);
            result[static_cast<std::size_t>(i)] = heap.back().second;
            heap.pop_back();
        }
    }

    void nearestKSearchRecursive(const plamatrix::Vec3<Scalar>& query, int k,
                                 int node_idx,
                                 std::vector<std::pair<double, int>>& heap) const
    {
        if (node_idx < 0) return;
        const auto& node = _nodes[static_cast<std::size_t>(node_idx)];

        auto pt = pointVec(node.point_idx);
        double d = finiteDistance(query, pt);

        const DistComparator compare;
        if (std::isfinite(d) && static_cast<int>(heap.size()) < k)
        {
            heap.push_back({d, node.point_idx});
            std::push_heap(heap.begin(), heap.end(), compare);
        }
        else if (std::isfinite(d) && !heap.empty() && d < heap.front().first)
        {
            std::pop_heap(heap.begin(), heap.end(), compare);
            heap.back() = {d, node.point_idx};
            std::push_heap(heap.begin(), heap.end(), compare);
        }

        int dim = node.split_dim;
        const double query_coord = static_cast<double>(dim == 0 ? query.x : (dim == 1 ? query.y : query.z));
        const double diff = query_coord - static_cast<double>(node.split_val);
        if (!std::isfinite(diff))
        {
            nearestKSearchRecursive(query, k, node.left, heap);
            nearestKSearchRecursive(query, k, node.right, heap);
            return;
        }
        int near_child = diff <= 0 ? node.left : node.right;
        int far_child  = diff <= 0 ? node.right : node.left;

        nearestKSearchRecursive(query, k, near_child, heap);

        const double max_dist = heap.empty() ? std::numeric_limits<double>::infinity() : heap.front().first;
        const double split_dist = std::abs(diff);
        if (!std::isfinite(split_dist) || split_dist <= max_dist || static_cast<int>(heap.size()) < k)
        {
            nearestKSearchRecursive(query, k, far_child, heap);
        }
    }

    void radiusSearchRecursive(const plamatrix::Vec3<Scalar>& query, Scalar radius,
                               int node_idx, std::vector<int>& result) const
    {
        if (node_idx < 0) return;
        const auto& node = _nodes[static_cast<std::size_t>(node_idx)];

        auto pt = pointVec(node.point_idx);
        if (finiteDistanceWithinRadius(query, pt, radius))
        {
            result.push_back(node.point_idx);
        }

        int dim = node.split_dim;
        const double query_coord = static_cast<double>(dim == 0 ? query.x : (dim == 1 ? query.y : query.z));
        const double diff = query_coord - static_cast<double>(node.split_val);
        if (!std::isfinite(diff))
        {
            radiusSearchRecursive(query, radius, node.left, result);
            radiusSearchRecursive(query, radius, node.right, result);
            return;
        }

        const int near_child = diff <= 0 ? node.left : node.right;
        const int far_child = diff <= 0 ? node.right : node.left;

        radiusSearchRecursive(query, radius, near_child, result);

        const double split_distance = std::abs(diff);
        if (!std::isfinite(split_distance) || split_distance <= static_cast<double>(radius))
        {
            radiusSearchRecursive(query, radius, far_child, result);
        }
    }

    static int checkedInt(std::size_t value, const char* label)
    {
        if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error(std::string(label) + " exceeds int range");
        }
        return static_cast<int>(value);
    }

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> copyCpuMatrix(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& matrix)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> copy(matrix.rows(), matrix.cols());
        for (plamatrix::Index r = 0; r < matrix.rows(); ++r)
            for (plamatrix::Index c = 0; c < matrix.cols(); ++c)
                copy(r, c) = matrix(r, c);
        return copy;
    }

    std::shared_ptr<const PointCloudType> _cloud;
    mutable std::shared_ptr<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>> _host_points;
    mutable std::uint64_t _host_points_revision = 0;
    mutable std::shared_ptr<const void> _host_points_identity;
    mutable const Scalar* _host_points_data = nullptr;
    std::vector<KdTreeNode<Scalar>> _nodes;
#ifdef PLAPOINT_WITH_CUDA
    mutable gpu::GpuSpatialIndex<Scalar> _gpu_spatial_index;
    mutable gpu::GpuSpatialQueryWorkspace<Scalar> _gpu_query_workspace;
    mutable detail::HostGridCellSet _gpu_host_occupied_cells;
    mutable std::shared_ptr<const void> _gpu_host_occupancy_identity;
    mutable std::uint64_t _gpu_host_occupancy_revision = 0;
    mutable Scalar _gpu_host_occupancy_cell_size = Scalar(0);
    mutable bool _gpu_host_occupancy_valid = false;
    mutable gpu::GpuNeighborBackend _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
    bool _built = false;
};

} // namespace search
} // namespace plapoint
