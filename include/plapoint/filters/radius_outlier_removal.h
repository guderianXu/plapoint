#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <memory>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class RadiusOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setRadius(Scalar r) { _radius = r; }
    void setMinNeighbors(int n) { _min_pts = n; }

protected:
    void applyFilter(PointCloudType& output) override
    {
        auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
        tree->setInputCloud(this->_input);
        tree->build();

        std::size_t n = this->_input->size();
        std::vector<int> inliers;

        for (std::size_t i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(static_cast<int>(i));
            auto neighbors = tree->radiusSearch(pt, _radius);
            if (static_cast<int>(neighbors.size()) >= _min_pts)
                inliers.push_back(static_cast<int>(i));
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
            return this->_input->points()(idx, dim);
        else
            return this->_input->points().getValue(idx, dim);
    }
    plamatrix::Vec3<Scalar> pointVec(int idx) const
        { return {pointCoord(idx,0), pointCoord(idx,1), pointCoord(idx,2)}; }

    Scalar _radius = 0.1;
    int _min_pts = 2;
};

} // namespace plapoint
