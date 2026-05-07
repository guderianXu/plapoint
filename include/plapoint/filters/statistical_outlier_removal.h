#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class StatisticalOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setMeanK(int k) { _mean_k = k; }

    void setStddevMulThresh(Scalar m) { _stddev_mul = m; }

    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree)
    {
        _tree = tree;
    }

protected:
    void applyFilter(PointCloudType& output) override
    {
        if (!_tree)
        {
            throw std::runtime_error("StatisticalOutlierRemoval: search method not set");
        }

        std::size_t n = this->_input->size();
        std::vector<Scalar> mean_dists(n, 0);

        for (std::size_t i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(static_cast<int>(i));
            auto neighbors = _tree->nearestKSearch(pt, _mean_k + 1);
            Scalar sum = 0;
            int count = 0;
            for (int nb : neighbors)
            {
                if (nb != static_cast<int>(i))
                {
                    auto pt_nb = pointVec(nb);
                    Scalar dx = pt.x - pt_nb.x, dy = pt.y - pt_nb.y, dz = pt.z - pt_nb.z;
                    sum += std::sqrt(dx * dx + dy * dy + dz * dz);
                    ++count;
                }
            }
            mean_dists[i] = (count > 0) ? sum / static_cast<Scalar>(count) : 0;
        }

        Scalar global_mean = 0;
        for (auto d : mean_dists) global_mean += d;
        global_mean /= static_cast<Scalar>(n);

        Scalar global_var = 0;
        for (auto d : mean_dists) { Scalar diff = d - global_mean; global_var += diff * diff; }
        global_var /= static_cast<Scalar>(n);
        Scalar global_stddev = std::sqrt(global_var);

        Scalar threshold = global_mean + _stddev_mul * global_stddev;

        std::vector<int> inliers;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (mean_dists[i] <= threshold)
            {
                inliers.push_back(static_cast<int>(i));
            }
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(inliers.size()), 3);
        for (std::size_t i = 0; i < inliers.size(); ++i)
        {
            int src = inliers[i];
            pts(static_cast<plamatrix::Index>(i), 0) = pointCoord(src, 0);
            pts(static_cast<plamatrix::Index>(i), 1) = pointCoord(src, 1);
            pts(static_cast<plamatrix::Index>(i), 2) = pointCoord(src, 2);
        }
        output = PointCloudType(std::move(pts));
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return this->_input->points()(idx, dim);
        }
        else
        {
            return this->_input->points().getValue(idx, dim);
        }
    }

    plamatrix::Vec3<Scalar> pointVec(int idx) const
    {
        return {pointCoord(idx, 0), pointCoord(idx, 1), pointCoord(idx, 2)};
    }

    int _mean_k = 8;
    Scalar _stddev_mul = 1;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
};

} // namespace plapoint
