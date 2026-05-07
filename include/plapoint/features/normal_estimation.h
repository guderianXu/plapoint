#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <memory>
#include <stdexcept>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class NormalEstimation
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud)
    {
        _cloud = cloud;
    }

    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree)
    {
        _tree = tree;
    }

    void setKSearch(int k) { _k = k; }

    plamatrix::DenseMatrix<Scalar, Dev> compute() const
    {
        if (!_cloud) throw std::runtime_error("NormalEstimation: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalEstimation: search method not set");

        int n = static_cast<int>(_cloud->size());
        plamatrix::DenseMatrix<Scalar, Dev> normals(n, 3);

        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(i);
            auto neighbors = _tree->nearestKSearch(pt, _k);
            if (neighbors.size() < 3) continue;

            int nn = static_cast<int>(neighbors.size());
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nb(nn, 3);
            for (int j = 0; j < nn; ++j)
            {
                auto npt = pointVec(neighbors[static_cast<std::size_t>(j)]);
                nb(j, 0) = npt.x;
                nb(j, 1) = npt.y;
                nb(j, 2) = npt.z;
            }

            auto cov = plamatrix::covarianceMatrix(nb);
            auto [U, S, Vt] = plamatrix::svd(cov);

            // Smallest singular value => last row of Vt
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
    plamatrix::Vec3<Scalar> pointVec(int idx) const
    {
        return {pointCoord(idx, 0), pointCoord(idx, 1), pointCoord(idx, 2)};
    }

    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return _cloud->points()(idx, dim);
        }
        else
        {
            return _cloud->points().getValue(idx, dim);
        }
    }

    std::shared_ptr<const PointCloudType> _cloud;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
    int _k = 10;
};

} // namespace plapoint
