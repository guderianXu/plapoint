#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/ops/point_cloud.h>
#include <plamatrix/internal/core/device.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/core/point_cloud_bridge.h>
#include <plapoint/filters/filter.h>
#include <plapoint/filters/detail/outlier_adapter.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/filter_indices.h>
#endif
#include <plapoint/search/kdtree.h>

namespace plapoint
{

/// Statistical outlier filter based on each point's mean KNN distance.
template <typename Scalar, plamatrix::internal::Device Dev = plamatrix::internal::Device::CPU,
          typename Enable = void>
class StatisticalOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = plapoint::internal::DeviceCloud<Scalar, Dev>;
    using Vector3 = plamatrix::Matrix<Scalar, 3, 1>;
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
    void setSearchMethod(std::shared_ptr<search::internal::DeviceKdTree<Scalar, Dev>> tree)
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
        if constexpr (Dev == plamatrix::internal::Device::GPU)
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
        if constexpr (Dev == plamatrix::internal::Device::GPU)
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
        if constexpr (Dev != plamatrix::internal::Device::GPU)
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
        auto make_point = [&](int idx) -> Vector3 {
            Vector3 point;
            point(0) = cpu_points(idx, 0);
            point(1) = cpu_points(idx, 1);
            point(2) = cpu_points(idx, 2);
            return point;
        };
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto pt = make_point(static_cast<int>(i));
            if (std::isfinite(pt(0)) && std::isfinite(pt(1)) && std::isfinite(pt(2)))
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
            plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> finite_points(
                static_cast<plamatrix::Index>(finite_indices.size()), 3);
            for (std::size_t i = 0; i < finite_indices.size(); ++i)
            {
                const int src = finite_indices[i];
                finite_points(static_cast<plamatrix::Index>(i), 0) = cpu_points(src, 0);
                finite_points(static_cast<plamatrix::Index>(i), 1) = cpu_points(src, 1);
                finite_points(static_cast<plamatrix::Index>(i), 2) = cpu_points(src, 2);
            }
            auto finite_cloud = std::make_shared<plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>>(
                std::move(finite_points));
            search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU> finite_tree;
            finite_tree.setInputCloud(finite_cloud);
            finite_tree.build();

            const auto& finite_cloud_points =
                static_cast<const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>&>(
                    *finite_cloud).points();
            const auto all_neighbors = finite_tree.batchNearestKSearch(
                finite_cloud_points, _mean_k + 1);
            std::vector<long double> finite_mean_dists(finite_indices.size(), 0);
            for (std::size_t i = 0; i < finite_indices.size(); ++i)
            {
                Vector3 pt;
                pt(0) = finite_cloud_points(static_cast<plamatrix::Index>(i), 0);
                pt(1) = finite_cloud_points(static_cast<plamatrix::Index>(i), 1);
                pt(2) = finite_cloud_points(static_cast<plamatrix::Index>(i), 2);
                const auto& neighbors = all_neighbors[i];
                long double mean_distance = 0;
                int count = 0;
                for (int nb : neighbors)
                {
                    if (nb != static_cast<int>(i))
                    {
                        Vector3 pt_nb;
                        pt_nb(0) = finite_cloud_points(nb, 0);
                        pt_nb(1) = finite_cloud_points(nb, 1);
                        pt_nb(2) = finite_cloud_points(nb, 2);
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
                Vector3 pt = make_point(static_cast<int>(i));
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
        const Vector3& a,
        const Vector3& b)
    {
        const long double dx = static_cast<long double>(a(0)) - static_cast<long double>(b(0));
        const long double dy = static_cast<long double>(a(1)) - static_cast<long double>(b(1));
        const long double dz = static_cast<long double>(a(2)) - static_cast<long double>(b(2));
        const long double distance = std::hypot(std::hypot(dx, dy), dz);
        if (distance > static_cast<long double>(std::numeric_limits<double>::max()))
        {
            return std::numeric_limits<long double>::infinity();
        }
        return distance;
    }

    int _mean_k = 8;
    Scalar _stddev_mul = 1;
    std::shared_ptr<search::internal::DeviceKdTree<Scalar, Dev>> _tree;
#ifdef PLAPOINT_WITH_CUDA
    std::shared_ptr<gpu::OutlierRemovalGpuWorkspace<Scalar>> _gpuWorkspace;
    gpu::GpuOutlierRemovalBackend _lastGpuBackend = gpu::GpuOutlierRemovalBackend::None;
    std::string _lastGpuFallbackReason;
#endif
};

namespace detail
{

template <typename PointT, typename FilterT>
class PointStatisticalOutlierAdapter : public PointOutlierAdapter<PointT, FilterT>
{
public:
    using Base = PointOutlierAdapter<PointT, FilterT>;
    using Scalar = std::decay_t<decltype(PointT::x)>;
    using Base::Base;

    void setMeanK(int k)
    {
        if (k <= 0 || k == std::numeric_limits<int>::max())
        {
            throw std::invalid_argument("StatisticalOutlierRemoval: mean k must be positive");
        }
        _mean_k = k;
    }
    int getMeanK() const noexcept { return _mean_k; }

    void setStddevMulThresh(double multiplier)
    {
        if (!std::isfinite(multiplier) || multiplier < 0.0)
        {
            throw std::invalid_argument("StatisticalOutlierRemoval: threshold multiplier must be non-negative");
        }
        _stddev_multiplier = multiplier;
    }
    double getStddevMulThresh() const noexcept { return _stddev_multiplier; }

    using SearcherPtr = typename search::Search<PointT>::Ptr;

    void setSearchMethod(const SearcherPtr& searcher)
    {
        _searcher = searcher;
    }

    Indices computeRejectedIndices()
    {
        const auto cloud = this->inputCloud();
        const auto selected = this->inputIndices();
        const std::size_t count = selected ? selected->size() : cloud->size();
        auto searcher = _searcher;
        if (!searcher)
        {
            searcher = std::make_shared<search::KdTree<PointT>>(false);
        }
        if (searcher->getInputCloud() != cloud && !searcher->setInputCloud(cloud))
        {
            throw std::runtime_error("StatisticalOutlierRemoval: search input could not be initialized");
        }

        std::vector<double> distances(count, std::numeric_limits<double>::quiet_NaN());
        double sum = 0.0;
        std::size_t valid_count = 0;
        for (std::size_t row = 0; row < count; ++row)
        {
            const int index = selected ? selected->at(row) : static_cast<int>(row);
            if (index < 0 || static_cast<std::size_t>(index) >= cloud->size())
            {
                throw std::out_of_range("StatisticalOutlierRemoval: input index is outside the cloud");
            }
            const auto& point = cloud->points[static_cast<std::size_t>(index)];
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            {
                continue;
            }

            Indices neighbors;
            std::vector<float> squared_distances;
            searcher->nearestKSearch(point, _mean_k + 1, neighbors, squared_distances);
            double distance_sum = 0.0;
            std::size_t neighbor_count = 0;
            for (std::size_t neighbor = 0; neighbor < neighbors.size(); ++neighbor)
            {
                if (neighbors[neighbor] == index)
                {
                    continue;
                }
                const double distance = std::sqrt(static_cast<double>(squared_distances[neighbor]));
                if (std::isfinite(distance))
                {
                    distance_sum += distance;
                    ++neighbor_count;
                }
            }
            if (neighbor_count == 0)
            {
                continue;
            }
            distances[row] = distance_sum / static_cast<double>(neighbor_count);
            sum += distances[row];
            ++valid_count;
        }

        const double mean = valid_count == 0 ? 0.0 : sum / static_cast<double>(valid_count);
        double squared_deviation_sum = 0.0;
        for (const double distance : distances)
        {
            if (std::isfinite(distance))
            {
                const double difference = distance - mean;
                squared_deviation_sum += difference * difference;
            }
        }
        const double deviation = valid_count < 2
                                     ? 0.0
                                     : std::sqrt(squared_deviation_sum / static_cast<double>(valid_count - 1));
        const double threshold = mean + _stddev_multiplier * deviation;
        Indices rejected;
        for (std::size_t row = 0; row < count; ++row)
        {
            if (!std::isfinite(distances[row]) || distances[row] > threshold)
            {
                rejected.push_back(selected ? selected->at(row) : static_cast<int>(row));
            }
        }
        return rejected;
    }

private:
    int _mean_k = 2;
    double _stddev_multiplier = 0.0;
    SearcherPtr _searcher;
};

} // namespace detail

template <typename PointT>
class StatisticalOutlierRemoval<PointT, plamatrix::internal::Device::CPU,
                                std::enable_if_t<!std::is_arithmetic_v<PointT>>>
    : public detail::PointStatisticalOutlierAdapter<PointT, StatisticalOutlierRemoval<PointT>>
{
public:
    using detail::PointStatisticalOutlierAdapter<
        PointT, StatisticalOutlierRemoval<PointT>>::PointStatisticalOutlierAdapter;
};

} // namespace plapoint
