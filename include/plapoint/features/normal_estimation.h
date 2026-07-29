#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/gpu/cuda_check.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/normal_estimation.h>
#endif
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class NormalEstimation
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud) { _cloud = cloud; }
    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree) { _tree = tree; }
    void setKSearch(int k)
    {
        if (k < 3)
        {
            throw std::invalid_argument("NormalEstimation: k must be at least 3");
        }
        _k = k;
    }

    plamatrix::DenseMatrix<Scalar, Dev> compute() const
    {
        if (!_cloud) throw std::runtime_error("NormalEstimation: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalEstimation: search method not set");

        if constexpr (Dev == plamatrix::Device::GPU)
        {
#ifndef PLAPOINT_WITH_CUDA
            throw std::runtime_error("NormalEstimation GPU requires PLAPOINT_WITH_CUDA=ON");
#else
            if (!_tree->isBuilt())
            {
                throw std::runtime_error("KdTree: build() must be called before search");
            }
            if (_k > 32)
            {
                return computeHostNormals().toGpu();
            }
            gpu::GpuSpatialIndex<Scalar> index;
            index.buildAdaptive(*_cloud);
            gpu::NormalEstimationGpuWorkspace<Scalar> workspace;
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> normals;
            gpu::estimateNormalsAsync(*_cloud, index, _k, normals, workspace, nullptr);
            PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
            workspace.checkStatus();
            return normals;
#endif
        }
        else
        {
            return computeHostNormals();
        }
    }

private:
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> computeHostNormals() const
    {
        int n = static_cast<int>(_cloud->size());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> normals(n, 3);
        normals.fill(0);

        const auto& points_cpu = _cloud->pointsCpu();

        // Batch KNN (uses GPU brute-force when Dev == GPU)
        auto all_neighbors = _tree->batchNearestKSearch(points_cpu, _k);

        // Compute normals per point. The neighbor search can run on CUDA for GPU trees;
        // the small per-point covariance/SVD stage is CPU-parallelized.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i)
        {
            const auto& neighbors = all_neighbors[static_cast<std::size_t>(i)];
            if (neighbors.size() < 3) continue;

            int nn = static_cast<int>(neighbors.size());
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nb(nn, 3);
            for (int j = 0; j < nn; ++j)
            {
                int idx = neighbors[static_cast<std::size_t>(j)];
                nb(j, 0) = points_cpu(idx, 0);
                nb(j, 1) = points_cpu(idx, 1);
                nb(j, 2) = points_cpu(idx, 2);
            }

            auto cov = plamatrix::covarianceMatrix(nb);
            auto [U, S, Vt] = plamatrix::svd(cov);

            Scalar nx = Vt.getValue(2, 0);
            Scalar ny = Vt.getValue(2, 1);
            Scalar nz = Vt.getValue(2, 2);

            normals.setValue(i, 0, nx);
            normals.setValue(i, 1, ny);
            normals.setValue(i, 2, nz);
        }
        return normals;
    }

    std::shared_ptr<const PointCloudType> _cloud;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
    int _k = 10;
};

/// Estimate normals for a CPU-owned cloud using the requested device where available.
/// The GPU path accelerates batched KNN and returns CPU normals for PlaScan-facing APIs.
template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> estimateNormals(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    int k,
    ProcessingDevice device,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    const bool try_gpu = detail::shouldTryGpu(device, input.size());
    if (try_gpu)
    {
        if (k > 32)
        {
            fallback_reason = "indexed GPU normal estimation requires k <= 32";
            if (device == ProcessingDevice::GPU)
            {
                throw std::invalid_argument(fallback_reason);
            }
        }
        else
        {
#ifdef PLAPOINT_WITH_CUDA
            if (detail::gpuIsAvailable())
            {
                try
                {
                    auto gpu_cloud_value = input.toGpu();
                    auto gpu_cloud = std::make_shared<
                        const PointCloud<Scalar, plamatrix::Device::GPU>>(
                            std::move(gpu_cloud_value));
                    auto tree = std::make_shared<
                        search::KdTree<Scalar, plamatrix::Device::GPU>>();
                    tree->setInputCloud(gpu_cloud);
                    tree->build();

                    NormalEstimation<Scalar, plamatrix::Device::GPU> estimator;
                    estimator.setInputCloud(gpu_cloud);
                    estimator.setSearchMethod(tree);
                    estimator.setKSearch(k);
                    auto gpu_normals = estimator.compute();
                    detail::setReport(
                        report, device, ProcessingDevice::GPU, false, {},
                        ProcessingNeighborBackend::GpuUniformGrid);
                    return gpu_normals.toCpu();
                }
                catch (const std::exception& ex)
                {
                    if (device == ProcessingDevice::GPU)
                    {
                        throw;
                    }
                    fallback_reason = ex.what();
                }
            }
            else
            {
                fallback_reason = "CUDA device is not available";
                if (device == ProcessingDevice::GPU)
                {
                    throw std::runtime_error(fallback_reason);
                }
            }
#else
            fallback_reason = "PlaPoint was built without CUDA support";
            if (device == ProcessingDevice::GPU)
            {
                throw std::runtime_error(fallback_reason);
            }
#endif
        }
    }

    auto cloud = detail::nonOwningCloudPtr(input);
    auto tree = std::make_shared<search::KdTree<Scalar, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    NormalEstimation<Scalar, plamatrix::Device::CPU> estimator;
    estimator.setInputCloud(cloud);
    estimator.setSearchMethod(tree);
    estimator.setKSearch(k);
    auto normals = estimator.compute();
    detail::setReport(report, device, ProcessingDevice::CPU,
                      try_gpu && !fallback_reason.empty(),
                      fallback_reason,
                      ProcessingNeighborBackend::CpuKdTree);
    return normals;
}

} // namespace plapoint
