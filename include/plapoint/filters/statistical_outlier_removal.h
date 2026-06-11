#pragma once

#include <algorithm>
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

/// Statistical outlier filter based on each point's mean KNN distance.
template <typename Scalar, plamatrix::Device Dev>
class StatisticalOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;
    using Filter<Scalar, Dev>::filter;

    /// Set the positive KNN neighborhood size. Must leave room for the self-neighbor.
    void setMeanK(int k)
    {
        if (k <= 0 || k > std::numeric_limits<int>::max() - 1)
        {
            throw std::invalid_argument(
                "StatisticalOutlierRemoval: mean k must be positive and less than INT_MAX");
        }
        _mean_k = k;
    }

    /// Set the finite, non-negative standard-deviation multiplier for rejection thresholding.
    void setStddevMulThresh(Scalar m)
    {
        if (!std::isfinite(m) || m < Scalar(0))
        {
            throw std::invalid_argument("StatisticalOutlierRemoval: stddev multiplier must be non-negative");
        }
        _stddev_mul = m;
    }

    /// Set the search structure used to compute each point's neighbor distances.
    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree)
    {
        _tree = tree;
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
        if (!_tree)
        {
            throw std::runtime_error("StatisticalOutlierRemoval: search method not set");
        }

        if constexpr (Dev == plamatrix::Device::GPU)
        {
#ifdef PLAPOINT_WITH_CUDA
            const auto point_count = checkedGpuPointCount();
            const auto k_use = point_count == 0 ? 0 : std::min(_mean_k + 1, point_count);
            if (k_use > 32)
            {
                return computeInlierIndices();
            }

            const auto keep_mask = gpu::statisticalOutlierRemovalKeepMaskDeviceColumnMajor(
                this->_input->points().data(), point_count, _mean_k, _stddev_mul);
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
            throw std::overflow_error("StatisticalOutlierRemoval: point count exceeds GPU int range");
        }
        return static_cast<int>(n);
    }

    std::vector<int> computeInlierIndices() const
    {
        if (!_tree)
        {
            throw std::runtime_error("StatisticalOutlierRemoval: search method not set");
        }

        std::size_t n = this->_input->size();
        if (n == 0)
        {
            return {};
        }

        std::vector<long double> mean_dists(n, 0);
        const auto& cpu_points = this->_input->pointsCpu();
        std::vector<int> finite_indices;
        finite_indices.reserve(n);
        auto make_point = [&](int idx) -> plamatrix::Vec3<Scalar> {
            return {
                cpu_points(idx, 0),
                cpu_points(idx, 1),
                cpu_points(idx, 2)
            };
        };
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto pt = make_point(static_cast<int>(i));
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
            {
                finite_indices.push_back(static_cast<int>(i));
            }
        }
        if (finite_indices.empty())
        {
            return {};
        }

        std::vector<int> inliers;
        inliers.reserve(finite_indices.size());
        if (finite_indices.size() < n)
        {
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> finite_points(
                static_cast<plamatrix::Index>(finite_indices.size()), 3);
            for (std::size_t i = 0; i < finite_indices.size(); ++i)
            {
                const int src = finite_indices[i];
                finite_points(static_cast<plamatrix::Index>(i), 0) = cpu_points(src, 0);
                finite_points(static_cast<plamatrix::Index>(i), 1) = cpu_points(src, 1);
                finite_points(static_cast<plamatrix::Index>(i), 2) = cpu_points(src, 2);
            }
            auto finite_cloud = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(
                std::move(finite_points));
            search::KdTree<Scalar, plamatrix::Device::CPU> finite_tree;
            finite_tree.setInputCloud(finite_cloud);
            finite_tree.build();

            const auto all_neighbors = finite_tree.batchNearestKSearch(
                finite_cloud->points(), _mean_k + 1);
            std::vector<long double> finite_mean_dists(finite_indices.size(), 0);
            for (std::size_t i = 0; i < finite_indices.size(); ++i)
            {
                plamatrix::Vec3<Scalar> pt{
                    finite_cloud->points()(static_cast<plamatrix::Index>(i), 0),
                    finite_cloud->points()(static_cast<plamatrix::Index>(i), 1),
                    finite_cloud->points()(static_cast<plamatrix::Index>(i), 2)
                };
                const auto& neighbors = all_neighbors[i];
                long double sum = 0;
                int count = 0;
                for (int nb : neighbors)
                {
                    if (nb != static_cast<int>(i))
                    {
                        auto pt_nb = plamatrix::Vec3<Scalar>{
                            finite_cloud->points()(nb, 0),
                            finite_cloud->points()(nb, 1),
                            finite_cloud->points()(nb, 2)
                        };
                        sum += finiteDistance(pt, pt_nb);
                        ++count;
                    }
                }
                finite_mean_dists[i] = (count > 0) ? sum / static_cast<long double>(count) : 0;
            }

            long double global_mean = 0;
            for (auto d : finite_mean_dists) global_mean += d;
            global_mean /= static_cast<long double>(finite_mean_dists.size());

            long double global_var = 0;
            for (auto d : finite_mean_dists) { long double diff = d - global_mean; global_var += diff * diff; }
            global_var /= static_cast<long double>(finite_mean_dists.size());
            long double global_stddev = std::sqrt(global_var);

            long double threshold = global_mean + static_cast<long double>(_stddev_mul) * global_stddev;
            for (std::size_t i = 0; i < finite_indices.size(); ++i)
            {
                if (finite_mean_dists[i] <= threshold)
                {
                    inliers.push_back(finite_indices[i]);
                }
            }
        }
        else
        {
            const auto all_neighbors = _tree->batchNearestKSearch(cpu_points, _mean_k + 1);
            for (std::size_t i = 0; i < n; ++i)
            {
                plamatrix::Vec3<Scalar> pt = make_point(static_cast<int>(i));
                const auto& neighbors = all_neighbors[i];
                long double sum = 0;
                int count = 0;
                for (int nb : neighbors)
                {
                    if (nb != static_cast<int>(i))
                    {
                        auto pt_nb = make_point(nb);
                        sum += finiteDistance(pt, pt_nb);
                        ++count;
                    }
                }
                mean_dists[i] = (count > 0) ? sum / static_cast<long double>(count) : 0;
            }

            long double global_mean = 0;
            for (auto d : mean_dists) global_mean += d;
            global_mean /= static_cast<long double>(n);

            long double global_var = 0;
            for (auto d : mean_dists) { long double diff = d - global_mean; global_var += diff * diff; }
            global_var /= static_cast<long double>(n);
            long double global_stddev = std::sqrt(global_var);

            long double threshold = global_mean + static_cast<long double>(_stddev_mul) * global_stddev;
            for (std::size_t i = 0; i < n; ++i)
            {
                if (mean_dists[i] <= threshold)
                {
                    inliers.push_back(static_cast<int>(i));
                }
            }
        }

        return inliers;
    }

    static long double finiteDistance(
        const plamatrix::Vec3<Scalar>& a,
        const plamatrix::Vec3<Scalar>& b)
    {
        const long double dx = static_cast<long double>(a.x) - static_cast<long double>(b.x);
        const long double dy = static_cast<long double>(a.y) - static_cast<long double>(b.y);
        const long double dz = static_cast<long double>(a.z) - static_cast<long double>(b.z);
        return std::hypot(std::hypot(dx, dy), dz);
    }

    int _mean_k = 8;
    Scalar _stddev_mul = 1;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
};

} // namespace plapoint
