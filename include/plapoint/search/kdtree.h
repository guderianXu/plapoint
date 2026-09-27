#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/core/point_cloud_bridge.h>
#include <plapoint/core/point_representation.h>
#include <plapoint/search/search.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/knn.h>
#include <plapoint/search/gpu_kdtree_policy.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/ops/point_cloud.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>

#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
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

namespace internal
{

template <typename Scalar>
struct KdTreeNode
{
    int point_idx;
    int left;
    int right;
    int split_dim;
    Scalar split_val;
};

template <typename Scalar, plamatrix::internal::Device Dev = plamatrix::internal::Device::CPU>
class DeviceKdTree
{
public:
    using PointCloudType = plapoint::internal::DeviceCloud<Scalar, Dev>;
    using Indices = std::vector<int>;
    using IndicesConstPtr = std::shared_ptr<const Indices>;

    /// Set the searchable cloud and optional subset of source indices. Returns false for a null cloud.
    /// Queries build the index on demand; invalid subset indices throw before replacing the current input.
    bool setInputCloud(const std::shared_ptr<const PointCloudType>& cloud,
                       const IndicesConstPtr& indices = {})
    {
        if (cloud)
        {
            cloud->validate();
            if (indices)
            {
                for (const int index : *indices)
                {
                    if (index < 0 || static_cast<std::size_t>(index) >= cloud->size())
                    {
                        throw std::out_of_range("DeviceKdTree: subset index is outside the input cloud");
                    }
                }
            }
        }
        _cloud = cloud;
        _indices = cloud ? indices : nullptr;
        _nodes.clear();
        _host_points.reset();
        _host_points_identity.reset();
        _host_points_data = nullptr;
        _host_points_revision = 0;
        _host_point_count = 0;
#ifdef PLAPOINT_WITH_CUDA
        _gpu_tree_cloud.reset();
        _gpu_spatial_index = {};
        _gpu_host_occupied_cells.clear();
        _gpu_host_occupancy_valid = false;
        _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
        _built = false;
        return static_cast<bool>(_cloud);
    }

    std::shared_ptr<const PointCloudType> getInputCloud() const noexcept { return _cloud; }
    IndicesConstPtr getIndices() const noexcept { return _indices; }

    void build() const
    {
        if (!_cloud)
        {
            throw std::runtime_error("DeviceKdTree: input cloud not set");
        }
        _built = false;
        _cloud->validate();
        _nodes.clear();
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            _host_points = std::make_shared<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                copyCpuMatrix(_cloud->points()));
        }
        else
        {
            _host_points = std::make_shared<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
                _cloud->points().toHostMatrix());
#ifdef PLAPOINT_WITH_CUDA
            if (!_indices)
            {
                _gpu_tree_cloud = std::make_shared<
                    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>>(
                        _cloud->points().clone(), _cloud->executionContext());
            }
            else
            {
                _gpu_tree_cloud.reset();
            }
#endif
        }
        _host_points_revision = _cloud->pointsRevision();
        _host_points_identity = _cloud->pointsIdentity();
        _host_points_data = _cloud->points().data();
        _host_point_count = static_cast<std::size_t>(_host_points->rows());
#ifdef PLAPOINT_WITH_CUDA
        _gpu_spatial_index = {};
        _gpu_host_occupied_cells.clear();
        _gpu_host_occupancy_valid = false;
#endif

        std::vector<int> indices;
        if (_indices)
        {
            indices = *_indices;
            for (const int index : indices)
            {
                if (index < 0 || static_cast<std::size_t>(index) >= _host_point_count)
                {
                    throw std::out_of_range("DeviceKdTree: subset index is outside the input cloud");
                }
            }
        }
        else
        {
            indices.resize(_host_point_count);
        }
        for (std::size_t i = 0; !_indices && i < indices.size(); ++i)
        {
            indices[i] = checkedInt(i, "DeviceKdTree: point index");
        }
        _nodes.reserve(indices.size());
        buildRecursive(indices, 0, checkedInt(indices.size(), "DeviceKdTree: point count") - 1, 0);
        _built = true;
    }

    bool isBuilt() const noexcept
    {
        return _built && matchesBuiltCloud();
    }

    struct DistComparator
    {
        bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b) const
        {
            if (a.first != b.first)
            {
                return a.first < b.first;
            }
            return a.second < b.second;
        }
    };

    std::vector<int> nearestKSearch(const plamatrix::Matrix<Scalar, 3, 1>& query, int k) const
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

    /// Search for up to k neighbors and report squared float distances.
    /// Squared distances use float, including for double-precision point clouds.
    int nearestKSearch(const plamatrix::Matrix<Scalar, 3, 1>& query, int k,
                       std::vector<int>& indices, std::vector<float>& squared_distances) const
    {
        auto found_indices = nearestKSearch(query, k);
        std::vector<float> found_distances;
        found_distances.reserve(found_indices.size());
        for (const int index : found_indices)
        {
            found_distances.push_back(squaredDistanceForOutput(query, pointVec(index)));
        }
        indices.swap(found_indices);
        squared_distances.swap(found_distances);
        return checkedInt(indices.size(), "DeviceKdTree: neighbor count");
    }

    std::vector<int> radiusSearch(const plamatrix::Matrix<Scalar, 3, 1>& query, double radius) const
    {
        ensureBuilt();
        validateQuery(query);
        std::vector<int> result;
        if (!std::isfinite(radius) || radius < 0.0)
        {
            throw std::invalid_argument("DeviceKdTree: radius must be finite and non-negative");
        }
        if (_nodes.empty()) return result;
        radiusSearchRecursive(query, radius, 0, result);
        return result;
    }

    /// Search within an inclusive radius and report squared float distances.
    /// max_nn == 0 returns all matches. Invalid queries or radii throw without changing the outputs.
    int radiusSearch(const plamatrix::Matrix<Scalar, 3, 1>& query, double radius,
                     std::vector<int>& indices, std::vector<float>& squared_distances,
                     unsigned int max_nn = 0) const
    {
        auto found_indices = radiusSearch(query, radius);
        std::vector<std::pair<double, int>> ordered;
        ordered.reserve(found_indices.size());
        for (const int index : found_indices)
        {
            ordered.emplace_back(finiteDistance(query, pointVec(index)), index);
        }
        std::sort(ordered.begin(), ordered.end(), DistComparator{});
        if (max_nn != 0 && ordered.size() > max_nn)
        {
            ordered.resize(max_nn);
        }

        found_indices.clear();
        found_indices.reserve(ordered.size());
        std::vector<float> found_distances;
        found_distances.reserve(ordered.size());
        for (const auto& neighbor : ordered)
        {
            found_indices.push_back(neighbor.second);
            found_distances.push_back(squaredDistanceForOutput(neighbor.first));
        }
        indices.swap(found_indices);
        squared_distances.swap(found_distances);
        return checkedInt(indices.size(), "DeviceKdTree: neighbor count");
    }

    /// Batch K-nearest neighbor search for multiple query points.
    /// On GPU: uses brute-force CUDA kernel (fast for up to ~100K points).
    /// On CPU: loops over queries using the kd-tree.
    /// @param queries   M x 3 matrix of query points
    /// @param k         number of neighbors per query
    /// @return          vector of M vectors, each with up to K finite neighbor indices
    std::vector<std::vector<int>>
    batchNearestKSearch(const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& queries, int k) const
    {
        ensureBuilt();
        if (queries.cols() != 3)
        {
            throw std::invalid_argument("DeviceKdTree: queries must be an Mx3 matrix");
        }
        validateQueries(queries);
        int M = checkedInt(static_cast<std::size_t>(queries.rows()), "DeviceKdTree: query count");
        std::vector<std::vector<int>> results(static_cast<std::size_t>(M));
        if (M <= 0 || k <= 0)
        {
#ifdef PLAPOINT_WITH_CUDA
            _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
            return results;
        }
        if (_host_point_count == 0)
        {
#ifdef PLAPOINT_WITH_CUDA
            _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
            return results;
        }

        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            const int N = checkedInt(_host_point_count, "DeviceKdTree: point count");
            std::vector<std::pair<double, int>> heap;
            heap.reserve(std::min(static_cast<std::size_t>(k), static_cast<std::size_t>(N)));
            for (int i = 0; i < M; ++i)
            {
                plamatrix::Matrix<Scalar, 3, 1> q{queries(i, 0), queries(i, 1), queries(i, 2)};
                nearestKSearchInto(q, k, results[static_cast<std::size_t>(i)], heap);
            }
        }
        else
        {
#ifndef PLAPOINT_WITH_CUDA
            throw std::runtime_error("PlaPoint was built without CUDA support");
#else
            if (_indices)
            {
                std::vector<std::pair<double, int>> heap;
                heap.reserve(std::min(static_cast<std::size_t>(k), _nodes.size()));
                for (int i = 0; i < M; ++i)
                {
                    const plamatrix::Matrix<Scalar, 3, 1> query{
                        queries(i, 0), queries(i, 1), queries(i, 2)};
                    nearestKSearchInto(query, k, results[static_cast<std::size_t>(i)], heap);
                }
                _last_neighbor_backend = gpu::GpuNeighborBackend::CpuBruteForce;
                return results;
            }
            int N = checkedInt(_host_point_count, "DeviceKdTree: point count");
            if (N <= 0)
            {
                return results;
            }
            const int K_use = std::min(k, N);
            if (K_use > 32)
            {
                for (int i = 0; i < M; ++i)
                {
                    plamatrix::Matrix<Scalar, 3, 1> q{queries(i, 0), queries(i, 1), queries(i, 2)};
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

            auto gpu_queries = plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(
                queries, _gpu_tree_cloud->executionContext());
            const bool enough_work = static_cast<std::size_t>(N)
                >= (detail::kIndexedKnnWorkThreshold + static_cast<std::size_t>(M) - 1)
                    / static_cast<std::size_t>(M);
            bool use_indexed = enough_work;
            bool rebuilt_index = false;
            Scalar cell_size = Scalar(1);
            if (use_indexed)
            {
                cell_size = detail::estimateKnnCellSize(*_host_points);
                if (!std::isfinite(cell_size) || cell_size <= Scalar(0))
                {
                    use_indexed = false;
                }
                else
                {
                    try
                    {
                        if (!_gpu_spatial_index.matches(*_gpu_tree_cloud, cell_size))
                        {
                            _gpu_spatial_index.build(*_gpu_tree_cloud, cell_size);
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
                    const auto identity = _gpu_tree_cloud->pointsIdentity();
                    const auto revision = _gpu_tree_cloud->pointsRevision();
                    if (rebuilt_index
                        || !_gpu_host_occupancy_valid
                        || _gpu_host_occupancy_identity != identity
                        || _gpu_host_occupancy_revision != revision
                        || _gpu_host_occupancy_cell_size != cell_size)
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

            plamatrix::Matrix<plamatrix::Index, plamatrix::Dynamic, plamatrix::Dynamic> indexed_indices;
            plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> brute_indices;
            plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> flat_dst;
            if (use_indexed)
            {
                auto indexed = _gpu_spatial_index.knnSearchAsync(gpu_queries, K_use, _gpu_query_workspace, nullptr);
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
                indexed_indices = indexed.indices.toHostMatrix();
                flat_dst = indexed.squaredDistances.toHostMatrix();
                _last_neighbor_backend = gpu::GpuNeighborBackend::UniformGrid;
            }
            else
            {
                plamatrix::internal::ResidentMatrix<int> gpu_indices(
                    queries.rows(), K_use, _gpu_tree_cloud->executionContext());
                plamatrix::internal::ResidentMatrix<Scalar> gpu_dists(
                    queries.rows(), K_use, _gpu_tree_cloud->executionContext());
                gpu::batchKnnDevice(
                    gpu_queries, _gpu_tree_cloud->points(), K_use, gpu_indices, gpu_dists);
                brute_indices = gpu_indices.toHostMatrix();
                flat_dst = gpu_dists.toHostMatrix();
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
    bool matchesBuiltCloud() const noexcept
    {
        const bool metadata_matches = _cloud
            && _host_points
            && _host_points_revision == _cloud->pointsRevision()
            && _host_points_identity == _cloud->pointsIdentity()
            && _host_points_data == _cloud->points().data()
            && _host_point_count == _cloud->size();
        if (!metadata_matches || _cloud->hasActivePointEdit())
        {
            return false;
        }
        if (!_cloud->hasUntrackedMutablePointAlias())
        {
            return true;
        }
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            return cpuSnapshotMatchesCloud();
        }
        return false;
    }

    void ensureBuilt() const
    {
        if (!_built)
        {
            build();
        }
        else if (!matchesBuiltCloud())
        {
            // The derived index is a cache. Rebuild it from the current cloud instead of mixing
            // stale split planes with updated coordinates. GPU aliases are refreshed
            // conservatively on every query; CPU aliases are compared with the snapshot.
            build();
        }
    }

    bool cpuSnapshotMatchesCloud() const noexcept
    {
        if constexpr (Dev != plamatrix::internal::Device::CPU)
        {
            return false;
        }
        else
        {
            const auto& current = _cloud->points();
            if (current.rows() != _host_points->rows() || current.cols() != _host_points->cols())
            {
                return false;
            }
            const auto element_count = static_cast<std::size_t>(current.size());
            if (element_count == 0)
            {
                return true;
            }
            if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(Scalar))
            {
                return false;
            }
            return std::memcmp(
                       current.data(),
                       _host_points->data(),
                       element_count * sizeof(Scalar)) == 0;
        }
    }

    Scalar pointCoord(int idx, int dim) const
    {
        return (*_host_points)(idx, dim);
    }

    plamatrix::Matrix<Scalar, 3, 1> pointVec(int idx) const
    {
        return plamatrix::Matrix<Scalar, 3, 1>(pointCoord(idx, 0), pointCoord(idx, 1), pointCoord(idx, 2));
    }

    std::vector<int> filterFiniteNeighbors(
        const plamatrix::Matrix<Scalar, 3, 1>& query,
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

    Scalar distSq(const plamatrix::Matrix<Scalar, 3, 1>& a, const plamatrix::Matrix<Scalar, 3, 1>& b) const
    {
        Scalar dx = a(0) - b(0), dy = a(1) - b(1), dz = a(2) - b(2);
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
        const plamatrix::Matrix<Scalar, 3, 1>& a,
        const plamatrix::Matrix<Scalar, 3, 1>& b)
    {
        const double coordinates[6] = {
            static_cast<double>(a(0)), static_cast<double>(a(1)), static_cast<double>(a(2)),
            static_cast<double>(b(0)), static_cast<double>(b(1)), static_cast<double>(b(2))};
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
        const plamatrix::Matrix<Scalar, 3, 1>& a,
        const plamatrix::Matrix<Scalar, 3, 1>& b)
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

    static float squaredDistanceForOutput(double distance)
    {
        const double largest_finite_root = std::sqrt(
            static_cast<double>(std::numeric_limits<float>::max()));
        if (distance > largest_finite_root)
        {
            return std::numeric_limits<float>::infinity();
        }
        return static_cast<float>(distance * distance);
    }

    static float squaredDistanceForOutput(
        const plamatrix::Matrix<Scalar, 3, 1>& a,
        const plamatrix::Matrix<Scalar, 3, 1>& b)
    {
        return squaredDistanceForOutput(finiteDistance(a, b));
    }

    static bool finiteDistanceWithinRadius(
        const plamatrix::Matrix<Scalar, 3, 1>& a,
        const plamatrix::Matrix<Scalar, 3, 1>& b,
        double radius)
    {
        const auto distance = makeDistanceOrderKey(a, b);
        int radius_exponent = 0;
        const double radius_mantissa = std::frexp(radius, &radius_exponent);
        const DistanceOrderKey radius_key{
            radius_mantissa == 0.0 ? std::numeric_limits<int>::min() : radius_exponent,
            radius_mantissa};
        return distance.exponent != std::numeric_limits<int>::max()
            && !distanceKeyLess(radius_key, distance);
    }

    static void validateQuery(const plamatrix::Matrix<Scalar, 3, 1>& query)
    {
        if (!std::isfinite(query(0)) || !std::isfinite(query(1)) || !std::isfinite(query(2)))
        {
            throw std::invalid_argument("DeviceKdTree: query coordinates must be finite");
        }
    }

    static void validateQueries(const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& queries)
    {
        for (plamatrix::Index r = 0; r < queries.rows(); ++r)
        {
            for (plamatrix::Index c = 0; c < queries.cols(); ++c)
            {
                if (!std::isfinite(queries(r, c)))
                {
                    throw std::invalid_argument("DeviceKdTree: query coordinates must be finite");
                }
            }
        }
    }

    int buildRecursive(std::vector<int>& indices, int start, int end, int depth) const
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

    void nearestKSearchInto(const plamatrix::Matrix<Scalar, 3, 1>& query, int k,
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

    void nearestKSearchRecursive(const plamatrix::Matrix<Scalar, 3, 1>& query, int k,
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
        else if (std::isfinite(d) && !heap.empty() &&
                 compare({d, node.point_idx}, heap.front()))
        {
            std::pop_heap(heap.begin(), heap.end(), compare);
            heap.back() = {d, node.point_idx};
            std::push_heap(heap.begin(), heap.end(), compare);
        }

        int dim = node.split_dim;
        const double query_coord = static_cast<double>(dim == 0 ? query(0) : (dim == 1 ? query(1) : query(2)));
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

    void radiusSearchRecursive(const plamatrix::Matrix<Scalar, 3, 1>& query, double radius,
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
        const double query_coord = static_cast<double>(dim == 0 ? query(0) : (dim == 1 ? query(1) : query(2)));
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
        if (!std::isfinite(split_distance) || split_distance <= radius)
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

    static plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>
    copyCpuMatrix(const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& matrix)
    {
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> copy(matrix.rows(), matrix.cols());
        for (plamatrix::Index r = 0; r < matrix.rows(); ++r)
            for (plamatrix::Index c = 0; c < matrix.cols(); ++c)
                copy(r, c) = matrix(r, c);
        return copy;
    }

    std::shared_ptr<const PointCloudType> _cloud;
    IndicesConstPtr _indices;
    mutable std::shared_ptr<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>> _host_points;
    mutable std::uint64_t _host_points_revision = 0;
    mutable std::shared_ptr<const void> _host_points_identity;
    mutable const Scalar* _host_points_data = nullptr;
    mutable std::size_t _host_point_count = 0;
    mutable std::vector<KdTreeNode<Scalar>> _nodes;
#ifdef PLAPOINT_WITH_CUDA
    mutable std::shared_ptr<plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>> _gpu_tree_cloud;
    mutable gpu::GpuSpatialIndex<Scalar> _gpu_spatial_index;
    mutable gpu::GpuSpatialQueryWorkspace<Scalar> _gpu_query_workspace;
    mutable detail::HostGridCellSet _gpu_host_occupied_cells;
    mutable std::shared_ptr<const void> _gpu_host_occupancy_identity;
    mutable std::uint64_t _gpu_host_occupancy_revision = 0;
    mutable Scalar _gpu_host_occupancy_cell_size = Scalar(0);
    mutable bool _gpu_host_occupancy_valid = false;
    mutable gpu::GpuNeighborBackend _last_neighbor_backend = gpu::GpuNeighborBackend::BruteForce;
#endif
    mutable bool _built = false;
};

} // namespace internal

} // namespace search

template <typename PointT> class KdTree
{
public:
    using PointCloud = plapoint::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;
    using PointRepresentation = plapoint::PointRepresentation<PointT>;
    using PointRepresentationConstPtr = typename PointRepresentation::ConstPtr;
    using Ptr = std::shared_ptr<KdTree<PointT>>;
    using ConstPtr = std::shared_ptr<const KdTree<PointT>>;

    explicit KdTree(bool sorted = true)
        : sorted_(sorted), point_representation_(std::make_shared<DefaultPointRepresentation<PointT>>())
    {
    }

    virtual ~KdTree() = default;

    virtual void setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = {})
    {
        input_ = cloud;
        indices_ = indices;
    }

    IndicesConstPtr getIndices() const { return indices_; }
    PointCloudConstPtr getInputCloud() const { return input_; }

    void setPointRepresentation(const PointRepresentationConstPtr& representation)
    {
        if (!representation)
        {
            throw std::invalid_argument("KdTree: point representation must not be null");
        }
        point_representation_ = representation;
        default_representation_ = false;
        if (input_)
        {
            setInputCloud(input_, indices_);
        }
    }

    PointRepresentationConstPtr getPointRepresentation() const { return point_representation_; }

    virtual int nearestKSearch(const PointT& point, unsigned int k, Indices& indices,
                               std::vector<float>& squared_distances) const = 0;

    virtual int nearestKSearch(const PointCloud& cloud, int index, unsigned int k, Indices& indices,
                               std::vector<float>& squared_distances) const
    {
        return nearestKSearch(cloud.points.at(static_cast<std::size_t>(index)), k, indices, squared_distances);
    }

    template <typename PointTDiff>
    int nearestKSearchT(const PointTDiff& point, unsigned int k, Indices& indices,
                        std::vector<float>& squared_distances) const
    {
        PointT query{};
        query.x = point.x;
        query.y = point.y;
        query.z = point.z;
        return nearestKSearch(query, k, indices, squared_distances);
    }

    virtual int nearestKSearch(int index, unsigned int k, Indices& indices,
                               std::vector<float>& squared_distances) const
    {
        if (!input_)
        {
            throw std::runtime_error("KdTree: input cloud not set");
        }
        const int cloud_index = indices_ ? indices_->at(static_cast<std::size_t>(index)) : index;
        return nearestKSearch(*input_, cloud_index, k, indices, squared_distances);
    }

    virtual int radiusSearch(const PointT& point, double radius, Indices& indices,
                             std::vector<float>& squared_distances, unsigned int max_nn = 0) const = 0;

    virtual int radiusSearch(const PointCloud& cloud, int index, double radius, Indices& indices,
                             std::vector<float>& squared_distances, unsigned int max_nn = 0) const
    {
        return radiusSearch(cloud.points.at(static_cast<std::size_t>(index)), radius,
                            indices, squared_distances, max_nn);
    }

    template <typename PointTDiff>
    int radiusSearchT(const PointTDiff& point, double radius, Indices& indices,
                      std::vector<float>& squared_distances, unsigned int max_nn = 0) const
    {
        PointT query{};
        query.x = point.x;
        query.y = point.y;
        query.z = point.z;
        return radiusSearch(query, radius, indices, squared_distances, max_nn);
    }

    virtual int radiusSearch(int index, double radius, Indices& indices,
                             std::vector<float>& squared_distances, unsigned int max_nn = 0) const
    {
        if (!input_)
        {
            throw std::runtime_error("KdTree: input cloud not set");
        }
        const int cloud_index = indices_ ? indices_->at(static_cast<std::size_t>(index)) : index;
        return radiusSearch(*input_, cloud_index, radius, indices, squared_distances, max_nn);
    }

    virtual void setEpsilon(float epsilon)
    {
        if (!std::isfinite(epsilon) || epsilon < 0.0f)
        {
            throw std::invalid_argument("KdTree: epsilon must be non-negative and finite");
        }
        epsilon_ = epsilon;
    }

    float getEpsilon() const { return epsilon_; }
    void setMinPts(int min_points) { min_pts_ = min_points; }
    int getMinPts() const { return min_pts_; }
    bool getSortedResults() const { return sorted_; }

protected:
    virtual std::string getName() const = 0;

    PointCloudConstPtr input_;
    IndicesConstPtr indices_;
    float epsilon_ = 0.0f;
    int min_pts_ = 1;
    bool sorted_ = true;
    PointRepresentationConstPtr point_representation_;
    bool default_representation_ = true;
};

} // namespace plapoint

namespace flann
{
template <typename T> struct L2_Simple;
}

namespace plapoint
{

namespace detail
{

template <typename PointT> struct RepresentedNeighbor
{
    int index = -1;
    float squared_distance = 0.0f;
};

template <typename PointT>
float representedSquaredDistance(const PointRepresentation<PointT>& representation,
                                 const PointT& lhs, const PointT& rhs)
{
    const int dimensions = representation.getNumberOfDimensions();
    std::vector<float> lhs_values(static_cast<std::size_t>(dimensions));
    std::vector<float> rhs_values(static_cast<std::size_t>(dimensions));
    representation.vectorize(lhs, lhs_values);
    representation.vectorize(rhs, rhs_values);
    float squared_distance = 0.0f;
    for (int dimension = 0; dimension < dimensions; ++dimension)
    {
        const float delta = lhs_values[static_cast<std::size_t>(dimension)] -
                            rhs_values[static_cast<std::size_t>(dimension)];
        squared_distance += delta * delta;
    }
    return squared_distance;
}

} // namespace detail

template <typename PointT, typename Dist = ::flann::L2_Simple<float>>
class KdTreeFLANN : public KdTree<PointT>
{
public:
    using Base = KdTree<PointT>;
    using Scalar = std::decay_t<decltype(PointT::x)>;
    using PointCloud = typename Base::PointCloud;
    using PointCloudConstPtr = typename Base::PointCloudConstPtr;
    using PointRepresentationConstPtr = typename Base::PointRepresentationConstPtr;
    using Ptr = std::shared_ptr<KdTreeFLANN<PointT, Dist>>;
    using ConstPtr = std::shared_ptr<const KdTreeFLANN<PointT, Dist>>;
    using Base::nearestKSearch;
    using Base::radiusSearch;

    explicit KdTreeFLANN(bool sorted = true) : Base(sorted) {}
    KdTreeFLANN(const KdTreeFLANN&) = default;
    KdTreeFLANN& operator=(const KdTreeFLANN&) = default;

    Ptr makeShared() { return std::make_shared<KdTreeFLANN>(*this); }

    void setSortedResults(bool sorted)
    {
        this->sorted_ = sorted;
    }

    void setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = {}) override
    {
        Base::setInputCloud(cloud, indices);
        _matrix_cloud.reset();
        _tree.reset();
        _active_indices.clear();
        if (!cloud)
        {
            return;
        }
        if (cloud->size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("KdTreeFLANN: point count exceeds int index range");
        }

        const std::size_t candidate_count = indices ? indices->size() : cloud->size();
        _active_indices.reserve(candidate_count);
        for (std::size_t position = 0; position < candidate_count; ++position)
        {
            const int index = indices ? indices->at(position) : static_cast<int>(position);
            if (index < 0 || static_cast<std::size_t>(index) >= cloud->size())
            {
                throw std::out_of_range("KdTreeFLANN: input index is outside the cloud");
            }
            const auto& point = cloud->points[static_cast<std::size_t>(index)];
            const bool valid = this->default_representation_
                                   ? std::isfinite(static_cast<double>(point.x)) &&
                                         std::isfinite(static_cast<double>(point.y)) &&
                                         std::isfinite(static_cast<double>(point.z))
                                   : this->point_representation_->isValid(point);
            if (valid)
            {
                _active_indices.push_back(index);
            }
        }

        if (this->default_representation_)
        {
            _matrix_cloud = plapoint::detail::toMatrixCloud<Scalar>(*cloud);
            IndicesConstPtr effective_indices;
            if (_active_indices.size() != cloud->size() || indices)
            {
                effective_indices = std::make_shared<const Indices>(_active_indices);
            }
            _tree = std::make_shared<search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
            _tree->setInputCloud(_matrix_cloud, effective_indices);
            _tree->build();
        }
    }

    void build() const
    {
        if (!this->input_)
        {
            throw std::runtime_error("KdTreeFLANN: input cloud not set");
        }
        if (_tree)
        {
            _tree->build();
        }
    }

    int nearestKSearch(const PointT& point, unsigned int k, Indices& indices,
                       std::vector<float>& squared_distances) const override
    {
        if (!this->input_)
        {
            throw std::runtime_error("KdTreeFLANN: input cloud not set");
        }
        if (k > static_cast<unsigned int>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("KdTreeFLANN: neighbor count exceeds int range");
        }
        if (_tree)
        {
            return _tree->nearestKSearch(plapoint::detail::pointVector(point), static_cast<int>(k),
                                         indices, squared_distances);
        }
        return nearestRepresented(point, k, indices, squared_distances);
    }

    int radiusSearch(const PointT& point, double radius, Indices& indices,
                     std::vector<float>& squared_distances, unsigned int max_nn = 0) const override
    {
        if (!this->input_)
        {
            throw std::runtime_error("KdTreeFLANN: input cloud not set");
        }
        if (_tree)
        {
            return _tree->radiusSearch(plapoint::detail::pointVector(point), radius,
                                       indices, squared_distances, max_nn);
        }
        return radiusRepresented(point, radius, indices, squared_distances, max_nn);
    }

protected:
    std::string getName() const override { return "KdTreeFLANN"; }

private:
    int nearestRepresented(const PointT& point, unsigned int k, Indices& indices,
                           std::vector<float>& squared_distances) const
    {
        std::vector<detail::RepresentedNeighbor<PointT>> neighbors;
        neighbors.reserve(_active_indices.size());
        for (const int index : _active_indices)
        {
            const auto& candidate = this->input_->points[static_cast<std::size_t>(index)];
            neighbors.push_back({index, detail::representedSquaredDistance(*this->point_representation_, point, candidate)});
        }
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& lhs, const auto& rhs)
        {
            return lhs.squared_distance < rhs.squared_distance ||
                   (lhs.squared_distance == rhs.squared_distance && lhs.index < rhs.index);
        });
        const std::size_t count = std::min<std::size_t>(k, neighbors.size());
        indices.resize(count);
        squared_distances.resize(count);
        for (std::size_t position = 0; position < count; ++position)
        {
            indices[position] = neighbors[position].index;
            squared_distances[position] = neighbors[position].squared_distance;
        }
        return static_cast<int>(count);
    }

    int radiusRepresented(const PointT& point, double radius, Indices& indices,
                          std::vector<float>& squared_distances, unsigned int max_nn) const
    {
        if (!std::isfinite(radius) || radius < 0.0)
        {
            throw std::invalid_argument("KdTreeFLANN: radius must be non-negative and finite");
        }
        const double squared_radius = radius * radius;
        std::vector<detail::RepresentedNeighbor<PointT>> neighbors;
        for (const int index : _active_indices)
        {
            const auto& candidate = this->input_->points[static_cast<std::size_t>(index)];
            const float distance = detail::representedSquaredDistance(*this->point_representation_, point, candidate);
            if (distance <= squared_radius)
            {
                neighbors.push_back({index, distance});
            }
        }
        if (this->sorted_)
        {
            std::sort(neighbors.begin(), neighbors.end(), [](const auto& lhs, const auto& rhs)
            {
                return lhs.squared_distance < rhs.squared_distance ||
                       (lhs.squared_distance == rhs.squared_distance && lhs.index < rhs.index);
            });
        }
        if (max_nn != 0 && neighbors.size() > max_nn)
        {
            neighbors.resize(max_nn);
        }
        indices.resize(neighbors.size());
        squared_distances.resize(neighbors.size());
        for (std::size_t position = 0; position < neighbors.size(); ++position)
        {
            indices[position] = neighbors[position].index;
            squared_distances[position] = neighbors[position].squared_distance;
        }
        return static_cast<int>(neighbors.size());
    }

    Indices _active_indices;
    std::shared_ptr<plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>> _matrix_cloud;
    std::shared_ptr<search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>> _tree;
};

namespace search
{

template <typename PointT, class Tree = plapoint::KdTreeFLANN<PointT>>
class KdTree : public Search<PointT>
{
public:
    using Base = Search<PointT>;
    using PointCloud = typename Base::PointCloud;
    using PointCloudConstPtr = typename Base::PointCloudConstPtr;
    using KdTreePtr = typename Tree::Ptr;
    using KdTreeConstPtr = typename Tree::ConstPtr;
    using PointRepresentationConstPtr = typename PointRepresentation<PointT>::ConstPtr;
    using Ptr = std::shared_ptr<KdTree<PointT, Tree>>;
    using ConstPtr = std::shared_ptr<const KdTree<PointT, Tree>>;
    using Base::nearestKSearch;
    using Base::radiusSearch;

    explicit KdTree(bool sorted = true)
        : Base("KdTree", sorted), tree_(std::make_shared<Tree>(sorted))
    {
    }

    void setPointRepresentation(const PointRepresentationConstPtr& representation)
    {
        tree_->setPointRepresentation(representation);
    }

    PointRepresentationConstPtr getPointRepresentation() const
    {
        return tree_->getPointRepresentation();
    }

    void setSortedResults(bool sorted) override
    {
        Base::setSortedResults(sorted);
        tree_->setSortedResults(sorted);
    }

    void setEpsilon(float epsilon) { tree_->setEpsilon(epsilon); }
    float getEpsilon() const { return tree_->getEpsilon(); }

    bool setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = {}) override
    {
        tree_->setInputCloud(cloud, indices);
        return static_cast<bool>(cloud);
    }

    PointCloudConstPtr getInputCloud() const noexcept override { return tree_->getInputCloud(); }
    IndicesConstPtr getIndices() const noexcept override { return tree_->getIndices(); }

    void build() const { tree_->build(); }

    int nearestKSearch(const PointT& point, int k, Indices& indices,
                       std::vector<float>& squared_distances) const override
    {
        if (k < 0)
        {
            throw std::invalid_argument("search::KdTree: neighbor count must be non-negative");
        }
        return tree_->nearestKSearch(point, static_cast<unsigned int>(k), indices, squared_distances);
    }

    int radiusSearch(const PointT& point, double radius, Indices& indices,
                     std::vector<float>& squared_distances, unsigned int max_nn = 0) const override
    {
        return tree_->radiusSearch(point, radius, indices, squared_distances, max_nn);
    }

private:
    KdTreePtr tree_;
};

} // namespace search
} // namespace plapoint
