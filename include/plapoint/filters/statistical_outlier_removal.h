#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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
        if constexpr (Dev == plamatrix::Device::GPU)
        {
#ifdef PLAPOINT_WITH_CUDA
            applyGpuFilter(output, &removed_indices);
#else
            throw std::runtime_error("PlaPoint was built without CUDA support");
#endif
        }
        else
        {
            const auto inliers = computeInlierIndices();
            this->copyPointsAndAttributesForIndices(inliers, output);
            removed_indices = this->removedIndicesFromKept(inliers);
        }
    }

#ifdef PLAPOINT_WITH_CUDA
    gpu::GpuOutlierRemovalBackend lastGpuBackend() const noexcept
    {
        return _lastGpuBackend;
    }

    const std::string& lastGpuFallbackReason() const noexcept
    {
        return _lastGpuFallbackReason;
    }

    std::size_t gpuIndexBuildCount() const noexcept
    {
        return _gpuWorkspace ? _gpuWorkspace->indexBuildCount() : 0;
    }
#endif

protected:
    void applyFilter(PointCloudType& output) override
    {
        if constexpr (Dev == plamatrix::Device::GPU)
        {
#ifdef PLAPOINT_WITH_CUDA
            applyGpuFilter(output, nullptr);
#else
            throw std::runtime_error("PlaPoint was built without CUDA support");
#endif
        }
        else
        {
            const auto inliers = computeInlierIndices();
            this->copyPointsAndAttributesForIndices(inliers, output);
        }
    }

private:
#ifdef PLAPOINT_WITH_CUDA
    gpu::OutlierRemovalGpuWorkspace<Scalar>& gpuWorkspace()
    {
        if (!_gpuWorkspace)
        {
            _gpuWorkspace = std::make_shared<gpu::OutlierRemovalGpuWorkspace<Scalar>>();
        }
        return *_gpuWorkspace;
    }

    void applyGpuFilter(PointCloudType& output, std::vector<int>* removed_indices)
    {
        if constexpr (Dev != plamatrix::Device::GPU)
        {
            throw std::logic_error("StatisticalOutlierRemoval: GPU helper used for CPU filter");
        }
        else
        {
            _lastGpuBackend = gpu::GpuOutlierRemovalBackend::None;
            _lastGpuFallbackReason.clear();
            const int point_count = checkedGpuPointCount();
            const int k_use = point_count == 0 ? 0 : std::min(_mean_k + 1, point_count);
            if (k_use > 32)
            {
                if (!_tree)
                {
                    throw std::runtime_error(
                        "StatisticalOutlierRemoval: search method required for CPU compatibility fallback");
                }
                _lastGpuBackend = gpu::GpuOutlierRemovalBackend::CpuCompatibility;
                _lastGpuFallbackReason = "mean_k + 1 exceeds indexed KNN limit 32";
                const auto inliers = computeInlierIndices();
                this->copyPointsAndAttributesForIndices(inliers, output);
                if (removed_indices)
                {
                    *removed_indices = this->removedIndicesFromKept(inliers);
                }
                return;
            }

            auto keep_mask = gpu::statisticalOutlierRemovalKeepMaskDevice(
                *this->_input, _mean_k, _stddev_mul, gpuWorkspace());
            output = gpu::compactPointCloudByKeepMask(*this->_input, keep_mask);
            _lastGpuBackend = gpu::GpuOutlierRemovalBackend::UniformGrid;
            if (removed_indices)
            {
                *removed_indices = gpu::removedIndicesFromKeepMaskDevice(keep_mask);
            }
        }
    }
#endif

    struct NormalizedDistanceStats
    {
        long double scale = 0;
        long double threshold = 0;
        bool valid = false;
    };

    NormalizedDistanceStats normalizedDistanceStats(
        const std::vector<long double>& distances) const
    {
        NormalizedDistanceStats stats;
        for (const long double distance : distances)
        {
            if (!std::isfinite(distance))
            {
                return stats;
            }
            stats.scale = std::max(stats.scale, distance);
        }
        if (!std::isfinite(stats.scale))
        {
            return stats;
        }
        if (stats.scale == 0)
        {
            stats.valid = true;
            return stats;
        }

        long double normalized_mean = 0;
        for (const long double distance : distances)
        {
            normalized_mean += distance / stats.scale;
        }
        normalized_mean /= static_cast<long double>(distances.size());

        long double normalized_variance = 0;
        for (const long double distance : distances)
        {
            const long double difference = distance / stats.scale - normalized_mean;
            normalized_variance += difference * difference;
        }
        normalized_variance /= static_cast<long double>(distances.size());
        stats.threshold = normalized_mean
            + static_cast<long double>(_stddev_mul) * std::sqrt(normalized_variance);
        stats.valid = std::isfinite(stats.threshold);
        return stats;
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
                long double mean_distance = 0;
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
                        ++count;
                        const long double distance = finiteDistance(pt, pt_nb);
                        if (!std::isfinite(distance))
                        {
                            return {};
                        }
                        mean_distance += (distance - mean_distance)
                            / static_cast<long double>(count);
                    }
                }
                finite_mean_dists[i] = count > 0 ? mean_distance : 0;
            }

            const auto stats = normalizedDistanceStats(finite_mean_dists);
            if (!stats.valid)
            {
                return {};
            }
            for (std::size_t i = 0; i < finite_indices.size(); ++i)
            {
                const long double normalized_distance = stats.scale == 0
                    ? 0
                    : finite_mean_dists[i] / stats.scale;
                if (normalized_distance <= stats.threshold)
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
                long double mean_distance = 0;
                int count = 0;
                for (int nb : neighbors)
                {
                    if (nb != static_cast<int>(i))
                    {
                        auto pt_nb = make_point(nb);
                        ++count;
                        const long double distance = finiteDistance(pt, pt_nb);
                        if (!std::isfinite(distance))
                        {
                            return {};
                        }
                        mean_distance += (distance - mean_distance)
                            / static_cast<long double>(count);
                    }
                }
                mean_dists[i] = count > 0 ? mean_distance : 0;
            }

            const auto stats = normalizedDistanceStats(mean_dists);
            if (!stats.valid)
            {
                return {};
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                const long double normalized_distance = stats.scale == 0
                    ? 0
                    : mean_dists[i] / stats.scale;
                if (normalized_distance <= stats.threshold)
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
        const long double distance = std::hypot(std::hypot(dx, dy), dz);
        if (distance > static_cast<long double>(std::numeric_limits<double>::max()))
        {
            return std::numeric_limits<long double>::infinity();
        }
        return distance;
    }

    int _mean_k = 8;
    Scalar _stddev_mul = 1;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
#ifdef PLAPOINT_WITH_CUDA
    std::shared_ptr<gpu::OutlierRemovalGpuWorkspace<Scalar>> _gpuWorkspace;
    gpu::GpuOutlierRemovalBackend _lastGpuBackend = gpu::GpuOutlierRemovalBackend::None;
    std::string _lastGpuFallbackReason;
#endif
};

} // namespace plapoint
