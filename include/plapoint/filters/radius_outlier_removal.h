#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/filter.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/filter_indices.h>
#endif
#include <plapoint/search/kdtree.h>

namespace plapoint
{

/// Radius-based outlier filter that keeps points with enough neighbors in a fixed radius.
template <typename Scalar, plamatrix::Device Dev>
class RadiusOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;
    using Filter<Scalar, Dev>::filter;

    /// Set the finite, non-negative neighbor search radius.
    void setRadius(Scalar r)
    {
        if (!std::isfinite(r) || r < Scalar(0))
        {
            throw std::invalid_argument("RadiusOutlierRemoval: radius must be finite and non-negative");
        }
        _radius = r;
    }

    /// Set the minimum neighbor count required to keep a point.
    void setMinNeighbors(int n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("RadiusOutlierRemoval: min neighbors must be positive");
        }
        _min_pts = n;
    }

    void filter(std::vector<int>& removed_indices) override
    {
        PointCloudType output;
        filter(output, removed_indices);
    }

    void filter(PointCloudType& output, std::vector<int>& removed_indices) override
    {
        if (!this->_input)
        {
            throw std::runtime_error("Filter: input cloud not set");
        }
        const auto inliers = computeDiagnosticInlierIndices();
        this->copyPointsAndAttributesForIndices(inliers, output);
        removed_indices = this->removedIndicesFromKept(inliers);
    }

protected:
    void applyFilter(PointCloudType& output) override
    {
        const auto inliers = computeDiagnosticInlierIndices();
        this->copyPointsAndAttributesForIndices(inliers, output);
    }

private:
    std::vector<int> computeDiagnosticInlierIndices() const
    {
        if constexpr (Dev == plamatrix::Device::GPU)
        {
#ifdef PLAPOINT_WITH_CUDA
            const auto point_count = checkedGpuPointCount();
            const auto keep_mask = gpu::radiusOutlierRemovalKeepMaskDeviceColumnMajor(
                this->_input->points().data(), point_count, _radius, _min_pts);
            return gpu::keptIndicesFromKeepMask(keep_mask);
#else
            throw std::runtime_error("PlaPoint was built without CUDA support");
#endif
        }
        else
        {
            return computeInlierIndices();
        }
    }

    int checkedGpuPointCount() const
    {
        const auto n = this->_input->size();
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("RadiusOutlierRemoval: point count exceeds GPU int range");
        }
        return static_cast<int>(n);
    }

    std::vector<int> computeInlierIndices() const
    {
        auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
        tree->setInputCloud(this->_input);
        tree->build();

        std::size_t n = this->_input->size();
        std::vector<int> inliers;
        const auto& cpu_points = this->_input->pointsCpu();
        auto make_point = [&](int idx) -> plamatrix::Vec3<Scalar> {
            return {
                cpu_points(idx, 0),
                cpu_points(idx, 1),
                cpu_points(idx, 2)
            };
        };

        for (std::size_t i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = make_point(static_cast<int>(i));
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            {
                continue;
            }
            auto neighbors = tree->radiusSearch(pt, _radius);
            if (static_cast<int>(neighbors.size()) >= _min_pts)
                inliers.push_back(static_cast<int>(i));
        }

        return inliers;
    }

    Scalar _radius = 0.1;
    int _min_pts = 2;
};

} // namespace plapoint
