#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/knn.h>
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
                copyCpuMatrix(_cloud->pointsCpu()));
        }
        std::vector<int> indices(static_cast<std::size_t>(_cloud->size()));
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = checkedInt(i, "KdTree: point index");
        }
        _nodes.reserve(indices.size());
        buildRecursive(indices, 0, checkedInt(indices.size(), "KdTree: point count") - 1, 0);
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
        std::vector<int> result;
        if (_nodes.empty() || k <= 0) return result;

        std::vector<std::pair<double, int>> heap;
        heap.reserve(std::min(static_cast<std::size_t>(k), _nodes.size()));
        nearestKSearchInto(query, k, result, heap);
        return result;
    }

    std::vector<int> radiusSearch(const plamatrix::Vec3<Scalar>& query, Scalar radius) const
    {
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
        if (queries.cols() != 3)
        {
            throw std::invalid_argument("KdTree: queries must be an Mx3 matrix");
        }
        int M = checkedInt(static_cast<std::size_t>(queries.rows()), "KdTree: query count");
        std::vector<std::vector<int>> results(static_cast<std::size_t>(M));
        if (M <= 0 || k <= 0)
        {
            return results;
        }
        if (!_cloud || _cloud->size() == 0)
        {
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
                for (int i = 0; i < M; ++i)
                {
                    plamatrix::Vec3<Scalar> q{queries(i, 0), queries(i, 1), queries(i, 2)};
                    auto row = nearestKSearch(q, k);
                    results[static_cast<std::size_t>(i)] = filterFiniteNeighbors(q, row, N);
                }
                return results;
            }

            const std::size_t query_count = static_cast<std::size_t>(M);
            const std::size_t k_count = static_cast<std::size_t>(K_use);
            const std::size_t query_scalars =
                detail::checkedSizeProduct(query_count, 3, "KdTree: query buffer");
            const std::size_t result_count =
                detail::checkedSizeProduct(query_count, k_count, "KdTree: result buffer");
            const std::size_t query_bytes =
                detail::checkedByteCount<Scalar>(query_scalars, "KdTree: query buffer");
            const std::size_t index_bytes =
                detail::checkedByteCount<int>(result_count, "KdTree: index buffer");
            const std::size_t dist_bytes =
                detail::checkedByteCount<Scalar>(result_count, "KdTree: distance buffer");

            ensureGpuBatchWorkspace(query_scalars, result_count);

            // Copy host queries to GPU
            std::vector<Scalar> h_queries(query_scalars);
            for (int i = 0; i < M; ++i)
            {
                const std::size_t row_offset = static_cast<std::size_t>(i) * 3u;
                h_queries[row_offset]     = queries(i, 0);
                h_queries[row_offset + 1] = queries(i, 1);
                h_queries[row_offset + 2] = queries(i, 2);
            }
            PLAPOINT_CHECK_CUDA(cudaMemcpy(_gpu_queries.get(), h_queries.data(), query_bytes, cudaMemcpyHostToDevice));

            gpu::batchKnnDeviceColumnMajor(
                _gpu_queries.get(), M, _cloud->points().data(), N, K_use, _gpu_indices.get(), _gpu_dists.get());

            std::vector<int>    flat_idx(result_count);
            std::vector<Scalar> flat_dst(result_count);
            PLAPOINT_CHECK_CUDA(cudaMemcpy(flat_idx.data(), _gpu_indices.get(), index_bytes, cudaMemcpyDeviceToHost));
            PLAPOINT_CHECK_CUDA(cudaMemcpy(flat_dst.data(),  _gpu_dists.get(),  dist_bytes, cudaMemcpyDeviceToHost));

            for (int i = 0; i < M; ++i)
            {
                auto& row = results[static_cast<std::size_t>(i)];
                row.reserve(static_cast<std::size_t>(K_use));
                const std::size_t row_offset = static_cast<std::size_t>(i) * k_count;
                for (int j = 0; j < K_use; ++j)
                {
                    const std::size_t flat_offset = row_offset + static_cast<std::size_t>(j);
                    const int idx = flat_idx[flat_offset];
                    const Scalar dist = flat_dst[flat_offset];
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

private:
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

    static double finiteDistance(
        const plamatrix::Vec3<Scalar>& a,
        const plamatrix::Vec3<Scalar>& b)
    {
        const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
        const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
        const double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
        return std::hypot(std::hypot(dx, dy), dz);
    }

    static bool finiteDistanceWithinRadius(
        const plamatrix::Vec3<Scalar>& a,
        const plamatrix::Vec3<Scalar>& b,
        Scalar radius)
    {
        const double distance = finiteDistance(a, b);
        return std::isfinite(distance) && distance <= static_cast<double>(radius);
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
        int near = diff <= 0 ? node.left : node.right;
        int far  = diff <= 0 ? node.right : node.left;

        nearestKSearchRecursive(query, k, near, heap);

        const double max_dist = heap.empty() ? std::numeric_limits<double>::infinity() : heap.front().first;
        const double split_dist = std::abs(diff);
        if (!std::isfinite(split_dist) || split_dist <= max_dist || static_cast<int>(heap.size()) < k)
        {
            nearestKSearchRecursive(query, k, far, heap);
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

        const int near = diff <= 0 ? node.left : node.right;
        const int far = diff <= 0 ? node.right : node.left;

        radiusSearchRecursive(query, radius, near, result);

        const double split_distance = std::abs(diff);
        if (!std::isfinite(split_distance) || split_distance <= static_cast<double>(radius))
        {
            radiusSearchRecursive(query, radius, far, result);
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

#ifdef PLAPOINT_WITH_CUDA
    void ensureGpuBatchWorkspace(std::size_t query_scalars, std::size_t result_count) const
    {
        if (_gpu_queries.size() < query_scalars)
        {
            _gpu_queries.allocate(query_scalars);
        }
        if (_gpu_indices.size() < result_count)
        {
            _gpu_indices.allocate(result_count);
        }
        if (_gpu_dists.size() < result_count)
        {
            _gpu_dists.allocate(result_count);
        }
    }
#endif

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
    std::shared_ptr<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>> _host_points;
    std::vector<KdTreeNode<Scalar>> _nodes;
#ifdef PLAPOINT_WITH_CUDA
    mutable gpu::DeviceBuffer<Scalar> _gpu_queries;
    mutable gpu::DeviceBuffer<int> _gpu_indices;
    mutable gpu::DeviceBuffer<Scalar> _gpu_dists;
#endif
};

} // namespace search
} // namespace plapoint
