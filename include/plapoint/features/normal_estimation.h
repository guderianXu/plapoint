#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class NormalEstimation
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud) { _cloud = cloud; }
    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree) { _tree = tree; }
    void setKSearch(int k) { _k = k; }

    plamatrix::DenseMatrix<Scalar, Dev> compute() const
    {
        if (!_cloud) throw std::runtime_error("NormalEstimation: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalEstimation: search method not set");

        int n = static_cast<int>(_cloud->size());
        plamatrix::DenseMatrix<Scalar, Dev> normals(n, 3);

        std::vector<Scalar> pts_host(static_cast<std::size_t>(n * 3));
        copyPointsToHost(pts_host);

        // Build queries for batch KNN
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(n, 3);
        for (int i = 0; i < n; ++i)
        {
            queries(i, 0) = pts_host[static_cast<std::size_t>(i * 3)];
            queries(i, 1) = pts_host[static_cast<std::size_t>(i * 3 + 1)];
            queries(i, 2) = pts_host[static_cast<std::size_t>(i * 3 + 2)];
        }

        // Batch KNN (uses GPU brute-force when Dev == GPU)
        auto all_neighbors = _tree->batchNearestKSearch(queries, _k);

        // Compute normals per point
        for (int i = 0; i < n; ++i)
        {
            const auto& neighbors = all_neighbors[static_cast<std::size_t>(i)];
            if (neighbors.size() < 3) continue;

            int nn = static_cast<int>(neighbors.size());
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nb(nn, 3);
            for (int j = 0; j < nn; ++j)
            {
                int idx = neighbors[static_cast<std::size_t>(j)];
                nb(j, 0) = pts_host[static_cast<std::size_t>(idx * 3)];
                nb(j, 1) = pts_host[static_cast<std::size_t>(idx * 3 + 1)];
                nb(j, 2) = pts_host[static_cast<std::size_t>(idx * 3 + 2)];
            }

            auto cov = plamatrix::covarianceMatrix(nb);
            auto [U, S, Vt] = plamatrix::svd(cov);

            Scalar nx = Vt.getValue(2, 0);
            Scalar ny = Vt.getValue(2, 1);
            Scalar nz = Vt.getValue(2, 2);

            normals.setValue(i, 0, nx);
            normals.setValue(i, 1, ny);
            normals.setValue(i, 2, nz);
        }
        return normals;
    }

private:
    void copyPointsToHost(std::vector<Scalar>& host) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            int n = static_cast<int>(_cloud->size());
            for (int i = 0; i < n; ++i)
            {
                host[static_cast<std::size_t>(i * 3)]     = _cloud->points()(i, 0);
                host[static_cast<std::size_t>(i * 3 + 1)] = _cloud->points()(i, 1);
                host[static_cast<std::size_t>(i * 3 + 2)] = _cloud->points()(i, 2);
            }
        }
        else
        {
#ifdef PLAPOINT_WITH_CUDA
            cudaMemcpy(host.data(), _cloud->points().data(),
                       static_cast<std::size_t>(_cloud->size() * 3) * sizeof(Scalar),
                       cudaMemcpyDeviceToHost);
#endif
        }
    }

    std::shared_ptr<const PointCloudType> _cloud;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
    int _k = 10;
};

} // namespace plapoint
