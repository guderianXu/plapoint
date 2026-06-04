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
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace plapoint {
namespace search {

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
                _cloud->points().toCpu());
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
        bool operator()(const std::pair<Scalar, int>& a, const std::pair<Scalar, int>& b) const
        {
            return a.first < b.first;
        }
    };

    std::vector<int> nearestKSearch(const plamatrix::Vec3<Scalar>& query, int k) const
    {
        std::vector<int> result;
        if (_nodes.empty() || k <= 0) return result;

        using DistIndex = std::pair<Scalar, int>;
        std::priority_queue<DistIndex, std::vector<DistIndex>, DistComparator> pq;

        nearestKSearchRecursive(query, k, 0, pq);

        result.resize(pq.size());
        for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i)
        {
            result[static_cast<std::size_t>(i)] = pq.top().second;
            pq.pop();
        }
        return result;
    }

    std::vector<int> radiusSearch(const plamatrix::Vec3<Scalar>& query, Scalar radius) const
    {
        std::vector<int> result;
        if (!std::isfinite(radius) || radius < Scalar(0))
        {
            throw std::invalid_argument("KdTree: radius must be non-negative");
        }
        if (_nodes.empty()) return result;
        radiusSearchRecursive(query, radius * radius, 0, result);
        return result;
    }

    /// Batch K-nearest neighbor search for multiple query points.
    /// On GPU: uses brute-force CUDA kernel (fast for up to ~100K points).
    /// On CPU: loops over queries using the kd-tree.
    /// @param queries   M x 3 matrix of query points
    /// @param k         number of neighbors per query
    /// @return          vector of M vectors, each with K indices
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
            for (int i = 0; i < M; ++i)
            {
                plamatrix::Vec3<Scalar> q{queries(i, 0), queries(i, 1), queries(i, 2)};
                results[static_cast<std::size_t>(i)] = nearestKSearch(q, k);
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
                throw std::invalid_argument("GPU KNN supports k in [1, 32]");
            }

            gpu::DeviceBuffer<Scalar> d_queries(static_cast<std::size_t>(M * 3));
            gpu::DeviceBuffer<int> d_indices(static_cast<std::size_t>(M * K_use));
            gpu::DeviceBuffer<Scalar> d_dists(static_cast<std::size_t>(M * K_use));

            // Copy host queries to GPU
            std::vector<Scalar> h_queries(static_cast<std::size_t>(M * 3));
            for (int i = 0; i < M; ++i)
            {
                h_queries[static_cast<std::size_t>(i * 3)]     = queries(i, 0);
                h_queries[static_cast<std::size_t>(i * 3 + 1)] = queries(i, 1);
                h_queries[static_cast<std::size_t>(i * 3 + 2)] = queries(i, 2);
            }
            PLAPOINT_CHECK_CUDA(cudaMemcpy(d_queries.get(), h_queries.data(), M * 3 * sizeof(Scalar), cudaMemcpyHostToDevice));

            gpu::batchKnnDeviceColumnMajor(
                d_queries.get(), M, _cloud->points().data(), N, K_use, d_indices.get(), d_dists.get());

            std::vector<int>    flat_idx(static_cast<std::size_t>(M * K_use));
            std::vector<Scalar> flat_dst(static_cast<std::size_t>(M * K_use));
            PLAPOINT_CHECK_CUDA(cudaMemcpy(flat_idx.data(), d_indices.get(), M * K_use * sizeof(int),    cudaMemcpyDeviceToHost));
            PLAPOINT_CHECK_CUDA(cudaMemcpy(flat_dst.data(),  d_dists.get(),   M * K_use * sizeof(Scalar), cudaMemcpyDeviceToHost));

            for (int i = 0; i < M; ++i)
            {
                results[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(K_use));
                for (int j = 0; j < K_use; ++j)
                    results[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        flat_idx[static_cast<std::size_t>(i * K_use + j)];
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

    Scalar distSq(const plamatrix::Vec3<Scalar>& a, const plamatrix::Vec3<Scalar>& b) const
    {
        Scalar dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    int buildRecursive(std::vector<int>& indices, int start, int end, int depth)
    {
        if (start > end) return -1;

        int dim = depth % 3;
        int mid = start + (end - start) / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end + 1,
                         [&](int a, int b) { return pointCoord(a, dim) < pointCoord(b, dim); });

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

    void nearestKSearchRecursive(const plamatrix::Vec3<Scalar>& query, int k,
                                 int node_idx,
                                 std::priority_queue<std::pair<Scalar, int>,
                                     std::vector<std::pair<Scalar, int>>,
                                     DistComparator>& pq) const
    {
        if (node_idx < 0) return;
        const auto& node = _nodes[static_cast<std::size_t>(node_idx)];

        auto pt = pointVec(node.point_idx);
        Scalar d = distSq(query, pt);

        if (static_cast<int>(pq.size()) < k)
        {
            pq.push({d, node.point_idx});
        }
        else if (d < pq.top().first)
        {
            pq.pop();
            pq.push({d, node.point_idx});
        }

        int dim = node.split_dim;
        Scalar diff = (dim == 0 ? query.x : (dim == 1 ? query.y : query.z)) - node.split_val;
        int near = diff <= 0 ? node.left : node.right;
        int far  = diff <= 0 ? node.right : node.left;

        nearestKSearchRecursive(query, k, near, pq);

        Scalar max_dist = pq.empty() ? std::numeric_limits<Scalar>::max() : pq.top().first;
        if (diff * diff < max_dist || static_cast<int>(pq.size()) < k)
        {
            nearestKSearchRecursive(query, k, far, pq);
        }
    }

    void radiusSearchRecursive(const plamatrix::Vec3<Scalar>& query, Scalar r2,
                               int node_idx, std::vector<int>& result) const
    {
        if (node_idx < 0) return;
        const auto& node = _nodes[static_cast<std::size_t>(node_idx)];

        auto pt = pointVec(node.point_idx);
        Scalar d = distSq(query, pt);
        if (d <= r2)
        {
            result.push_back(node.point_idx);
        }

        int dim = node.split_dim;
        Scalar diff = (dim == 0 ? query.x : (dim == 1 ? query.y : query.z)) - node.split_val;

        if (diff <= 0)
        {
            radiusSearchRecursive(query, r2, node.left, result);
            if (diff * diff <= r2) radiusSearchRecursive(query, r2, node.right, result);
        }
        else
        {
            radiusSearchRecursive(query, r2, node.right, result);
            if (diff * diff <= r2) radiusSearchRecursive(query, r2, node.left, result);
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

    std::shared_ptr<const PointCloudType> _cloud;
    std::shared_ptr<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>> _host_points;
    std::vector<KdTreeNode<Scalar>> _nodes;
};

} // namespace search
} // namespace plapoint
