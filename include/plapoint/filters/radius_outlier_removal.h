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
class RadiusOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setRadius(Scalar r)
    {
        if (!std::isfinite(r) || r < Scalar(0))
        {
            throw std::invalid_argument("RadiusOutlierRemoval: radius must be non-negative");
        }
        _radius = r;
    }

    void setMinNeighbors(int n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("RadiusOutlierRemoval: min neighbors must be positive");
        }
        _min_pts = n;
    }

protected:
    void applyFilter(PointCloudType& output) override
    {
        auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
        tree->setInputCloud(this->_input);
        tree->build();

        std::size_t n = this->_input->size();
        std::vector<int> inliers;
        const auto* cpu_points = static_cast<const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>*>(nullptr);
        auto staged_points = stagePointsIfNeeded();
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            cpu_points = &this->_input->points();
        }
        else
        {
            cpu_points = &staged_points;
        }
        auto make_point = [&](int idx) -> plamatrix::Vec3<Scalar> {
            return {
                (*cpu_points)(idx, 0),
                (*cpu_points)(idx, 1),
                (*cpu_points)(idx, 2)
            };
        };

        for (std::size_t i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = make_point(static_cast<int>(i));
            auto neighbors = tree->radiusSearch(pt, _radius);
            if (static_cast<int>(neighbors.size()) >= _min_pts)
                inliers.push_back(static_cast<int>(i));
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(inliers.size()), 3);
        for (std::size_t i = 0; i < inliers.size(); ++i)
        {
            int src = inliers[i];
            pts(static_cast<plamatrix::Index>(i), 0) = (*cpu_points)(src, 0);
            pts(static_cast<plamatrix::Index>(i), 1) = (*cpu_points)(src, 1);
            pts(static_cast<plamatrix::Index>(i), 2) = (*cpu_points)(src, 2);
        }
        output = this->makeOutputCloud(std::move(pts));
        this->copyNormalsForIndices(inliers, output);
    }

private:
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> stagePointsIfNeeded() const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(0, 3);
        else
            return this->_input->points().toCpu();
    }

    Scalar _radius = 0.1;
    int _min_pts = 2;
};

} // namespace plapoint
