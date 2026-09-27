#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>
#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/normal_refinement.h>
#endif
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::internal::Device Dev>
class NormalRefinement
{
public:
    using PointCloudType = plapoint::internal::DeviceCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<PointCloudType>& cloud) { _cloud = cloud; }
    void setSearchMethod(std::shared_ptr<search::internal::DeviceKdTree<Scalar, Dev>> tree) { _tree = tree; }

    /// Smooth existing normals by averaging each point's k nearest neighbor normals.
    /// Throws if the cloud, search method, normals, or k are invalid.
    void smooth(int k)
    {
        if (!_cloud) throw std::runtime_error("NormalRefinement: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalRefinement: search method not set");
        if (!_cloud->hasNormals()) throw std::runtime_error("NormalRefinement: cloud has no normals");
        if (k <= 0) throw std::invalid_argument("NormalRefinement: k must be positive");

#ifdef PLAPOINT_WITH_CUDA
        if constexpr (Dev == plamatrix::internal::Device::GPU)
        {
            if (k <= 32)
            {
                if (!_tree->isBuilt())
                {
                    throw std::runtime_error("NormalRefinement: search method is not built");
                }
                gpu::GpuSpatialIndex<Scalar> index;
                index.buildAdaptive(*_cloud);
                gpu::NormalRefinementGpuWorkspace<Scalar> workspace;
                plamatrix::internal::ResidentMatrix<Scalar> refined_normals(
                    static_cast<plamatrix::Index>(_cloud->size()), 3, _cloud->executionContext());
                gpu::smoothNormalsAsync(
                    *_cloud, index, k, refined_normals, workspace, nullptr);
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
                workspace.resetStream();
                _cloud->setNormals(std::move(refined_normals));
                return;
            }
        }
#endif

        int n = static_cast<int>(_cloud->size());
        const auto& points_cpu = _cloud->pointsCpu();
        auto normals_cpu = toCpuCopy(*_cloud->normals());

        std::vector<plamatrix::Matrix<Scalar, 3, 1>> temp(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            temp[static_cast<std::size_t>(i)] = plamatrix::Matrix<Scalar, 3, 1>(
                normals_cpu(i, 0),
                normals_cpu(i, 1),
                normals_cpu(i, 2));

        const auto all_neighbors = _tree->batchNearestKSearch(points_cpu, k);
        for (int i = 0; i < n; ++i)
        {
            const auto& neighbors = all_neighbors[static_cast<std::size_t>(i)];
            Scalar sx = 0, sy = 0, sz = 0;
            for (int nb : neighbors)
            {
                sx += temp[static_cast<std::size_t>(nb)](0);
                sy += temp[static_cast<std::size_t>(nb)](1);
                sz += temp[static_cast<std::size_t>(nb)](2);
            }
            Scalar len = std::sqrt(sx*sx + sy*sy + sz*sz);
            if (len > Scalar(1e-10))
            {
                normals_cpu(i, 0) = sx / len;
                normals_cpu(i, 1) = sy / len;
                normals_cpu(i, 2) = sz / len;
            }
        }
        setCloudNormals(std::move(normals_cpu));
    }

    /// Flip normals in place so they point toward the supplied viewpoint.
    void orientConsistently(const plamatrix::Matrix<Scalar, 3, 1>& viewpoint)
    {
        if (!_cloud || !_cloud->hasNormals()) return;
#ifdef PLAPOINT_WITH_CUDA
        if constexpr (Dev == plamatrix::internal::Device::GPU)
        {
            gpu::orientNormalsTowardViewpointAsync(*_cloud, viewpoint, nullptr);
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
            return;
        }
#endif
        int n = static_cast<int>(_cloud->size());
        const auto& points_cpu = _cloud->pointsCpu();
        auto normals_cpu = toCpuCopy(*_cloud->normals());
        for (int i = 0; i < n; ++i)
        {
            const auto pt = pointVec(points_cpu, i);
            Scalar dx = viewpoint(0) - pt(0), dy = viewpoint(1) - pt(1), dz = viewpoint(2) - pt(2);
            Scalar nx = normals_cpu(i, 0), ny = normals_cpu(i, 1), nz = normals_cpu(i, 2);
            if (dx * nx + dy * ny + dz * nz < 0)
            {
                normals_cpu(i, 0) = -nx;
                normals_cpu(i, 1) = -ny;
                normals_cpu(i, 2) = -nz;
            }
        }
        setCloudNormals(std::move(normals_cpu));
    }

private:
    static plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>
    toCpuCopy(const typename PointCloudType::MatrixType& m)
    {
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            return m;
        }
        else
        {
            return m.toHostMatrix();
        }
    }

    static plamatrix::Matrix<Scalar, 3, 1>
    pointVec(const plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>& points, int idx)
    {
        return plamatrix::Matrix<Scalar, 3, 1>(points(idx, 0), points(idx, 1), points(idx, 2));
    }

    void setCloudNormals(plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>&& normals_cpu)
    {
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            _cloud->setNormals(std::move(normals_cpu));
        }
        else
        {
            _cloud->setNormals(plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(
                normals_cpu, _cloud->executionContext()));
        }
    }

    std::shared_ptr<PointCloudType> _cloud;
    std::shared_ptr<search::internal::DeviceKdTree<Scalar, Dev>> _tree;
};

} // namespace plapoint
