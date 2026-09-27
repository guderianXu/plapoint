#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

/// Radius-based outlier filter that keeps points with enough neighbors in a fixed radius.
template <typename Scalar, plamatrix::internal::Device Dev = plamatrix::internal::Device::CPU,
          typename Enable = void>
class RadiusOutlierRemoval : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = plapoint::internal::DeviceCloud<Scalar, Dev>;
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
        if constexpr (Dev == plamatrix::internal::Device::GPU)
        {
#ifdef PLAPOINT_WITH_CUDA
            auto keep_mask = gpu::radiusOutlierRemovalKeepMaskDevice(
                *this->_input, _radius, _min_pts, gpuWorkspace());
            output = gpu::compactPointCloudByKeepMask(*this->_input, keep_mask);
            removed_indices = gpu::removedIndicesFromKeepMaskDevice(keep_mask);
            return;
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
        return _gpuWorkspace ? _gpuWorkspace->lastBackend()
            : gpu::GpuOutlierRemovalBackend::None;
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
            auto keep_mask = gpu::radiusOutlierRemovalKeepMaskDevice(
                *this->_input, _radius, _min_pts, gpuWorkspace());
            output = gpu::compactPointCloudByKeepMask(*this->_input, keep_mask);
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
    gpu::OutlierRemovalGpuWorkspace<Scalar>& gpuWorkspace() const
    {
        if (!_gpuWorkspace)
        {
            _gpuWorkspace = std::make_shared<gpu::OutlierRemovalGpuWorkspace<Scalar>>();
        }
        return *_gpuWorkspace;
    }
#endif

    std::vector<int> computeInlierIndices() const
    {
        auto tree = std::make_shared<search::internal::DeviceKdTree<Scalar, Dev>>();
        tree->setInputCloud(this->_input);
        tree->build();

        std::size_t n = this->_input->size();
        std::vector<int> inliers;
        const auto& cpu_points = this->_input->pointsCpu();
        auto make_point = [&](int idx) -> plamatrix::Matrix<Scalar, 3, 1> {
            return plamatrix::Matrix<Scalar, 3, 1>{
                cpu_points(idx, 0),
                cpu_points(idx, 1),
                cpu_points(idx, 2)
            };
        };

        for (std::size_t i = 0; i < n; ++i)
        {
            plamatrix::Matrix<Scalar, 3, 1> pt = make_point(static_cast<int>(i));
            if (!pt.allFinite())
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
#ifdef PLAPOINT_WITH_CUDA
    mutable std::shared_ptr<gpu::OutlierRemovalGpuWorkspace<Scalar>> _gpuWorkspace;
#endif
};

namespace detail
{

template <typename PointT, typename FilterT>
class PointRadiusOutlierAdapter : public PointOutlierAdapter<PointT, FilterT>
{
public:
    using Base = PointOutlierAdapter<PointT, FilterT>;
    using Scalar = std::decay_t<decltype(PointT::x)>;
    using Base::Base;

    void setRadiusSearch(double radius)
    {
        if (!std::isfinite(radius) || radius < 0.0)
        {
            throw std::invalid_argument("RadiusOutlierRemoval: radius must be non-negative and finite");
        }
        _radius = radius;
    }
    double getRadiusSearch() const noexcept { return _radius; }

    void setMinNeighborsInRadius(int count)
    {
        if (count <= 0)
        {
            throw std::invalid_argument("RadiusOutlierRemoval: neighbor count must be positive");
        }
        _min_neighbors = count;
    }
    int getMinNeighborsInRadius() const noexcept { return _min_neighbors; }

    using SearcherPtr = typename search::Search<PointT>::Ptr;

    void setSearchMethod(const SearcherPtr& searcher)
    {
        _searcher = searcher;
    }

    void setNumberOfThreads(unsigned int thread_count = 0)
    {
#ifdef _OPENMP
        _thread_count = thread_count == 0 ? static_cast<unsigned int>(omp_get_num_procs()) : thread_count;
#else
        (void)thread_count;
        _thread_count = 1;
#endif
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
            throw std::runtime_error("RadiusOutlierRemoval: search input could not be initialized");
        }

        Indices candidates(count);
        for (std::size_t row = 0; row < count; ++row)
        {
            const int index = selected ? selected->at(row) : static_cast<int>(row);
            if (index < 0 || static_cast<std::size_t>(index) >= cloud->size())
            {
                throw std::out_of_range("RadiusOutlierRemoval: input index is outside the cloud");
            }
            candidates[row] = index;
        }

        std::vector<std::uint8_t> rejected_mask(count, 0);
#ifdef _OPENMP
#pragma omp parallel for num_threads(_thread_count)
#endif
        for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(count); ++row)
        {
            const int index = candidates[static_cast<std::size_t>(row)];
            const auto& point = cloud->points[static_cast<std::size_t>(index)];
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            {
                rejected_mask[static_cast<std::size_t>(row)] = 1;
                continue;
            }
            Indices neighbors;
            std::vector<float> squared_distances;
            const int found = searcher->radiusSearch(point, _radius, neighbors, squared_distances,
                                                     static_cast<unsigned int>(_min_neighbors) + 1u);
            if (found <= _min_neighbors)
            {
                rejected_mask[static_cast<std::size_t>(row)] = 1;
            }
        }

        Indices rejected;
        rejected.reserve(count);
        for (std::size_t row = 0; row < count; ++row)
        {
            if (rejected_mask[row])
            {
                rejected.push_back(candidates[row]);
            }
        }
        return rejected;
    }

private:
    double _radius = 0.0;
    int _min_neighbors = 1;
    SearcherPtr _searcher;
    unsigned int _thread_count = 1;
};

} // namespace detail

template <typename PointT>
class RadiusOutlierRemoval<PointT, plamatrix::internal::Device::CPU,
                           std::enable_if_t<!std::is_arithmetic_v<PointT>>>
    : public detail::PointRadiusOutlierAdapter<PointT, RadiusOutlierRemoval<PointT>>
{
public:
    using detail::PointRadiusOutlierAdapter<PointT, RadiusOutlierRemoval<PointT>>::PointRadiusOutlierAdapter;
};

} // namespace plapoint
