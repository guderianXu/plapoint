#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/gpu/cuda_check.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/normal_estimation.h>
#endif
#ifdef PLAPOINT_WITH_OPENCL
#include <plapoint/opencl/normal_estimation.h>
#endif
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/small_matrix.h>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
        int failure_slot_count = 1;
#ifdef _OPENMP
        failure_slot_count = omp_get_max_threads();
        if (failure_slot_count < 1)
        {
            failure_slot_count = 1;
        }
#endif
        std::vector<int> failure_rows(
            static_cast<std::size_t>(failure_slot_count), n);
        std::vector<std::exception_ptr> failures(
            static_cast<std::size_t>(failure_slot_count));
        std::atomic<int> first_failed_row{n};
        const auto record_failure = [&](int row) noexcept
        {
            int worker_index = 0;
#ifdef _OPENMP
            worker_index = omp_get_thread_num();
#endif
            const auto failure_slot = static_cast<std::size_t>(worker_index);
            if (row < failure_rows[failure_slot])
            {
                failures[failure_slot] = std::current_exception();
                failure_rows[failure_slot] = row;
            }

            int previous = first_failed_row.load(std::memory_order_relaxed);
            while (row < previous
                   && !first_failed_row.compare_exchange_weak(
                       previous,
                       row,
                       std::memory_order_relaxed,
                       std::memory_order_relaxed))
            {
            }
        };
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i)
        {
            if (i > first_failed_row.load(std::memory_order_relaxed))
            {
                continue;
            }
            try
            {
                const auto& neighbors = all_neighbors[static_cast<std::size_t>(i)];
                if (neighbors.size() < 3)
                {
                    continue;
                }

                std::array<Scalar, 3> centroid{};
                for (const int index : neighbors)
                {
                    centroid[0] += points_cpu(index, 0);
                    centroid[1] += points_cpu(index, 1);
                    centroid[2] += points_cpu(index, 2);
                }
                const Scalar inverse_count = Scalar(1) /
                    static_cast<Scalar>(neighbors.size());
                for (Scalar& value : centroid)
                {
                    value *= inverse_count;
                }

                std::array<Scalar, 6> covariance{};
                for (const int index : neighbors)
                {
                    const Scalar x = points_cpu(index, 0) - centroid[0];
                    const Scalar y = points_cpu(index, 1) - centroid[1];
                    const Scalar z = points_cpu(index, 2) - centroid[2];
                    covariance[0] += x * x;
                    covariance[1] += x * y;
                    covariance[2] += x * z;
                    covariance[3] += y * y;
                    covariance[4] += y * z;
                    covariance[5] += z * z;
                }
                for (Scalar& value : covariance)
                {
                    value *= inverse_count;
                }

                std::array<Scalar, 3> eigenvalues{};
                std::array<Scalar, 9> eigenvectors{};
                plamatrix::symmetricEigh3x3(
                    covariance, &eigenvalues, &eigenvectors);

                normals.setValue(i, 0, eigenvectors[0]);
                normals.setValue(i, 1, eigenvectors[1]);
                normals.setValue(i, 2, eigenvectors[2]);
            }
            catch (...)
            {
                record_failure(i);
            }
        }

        int selected_failure_row = n;
        std::exception_ptr selected_failure;
        for (std::size_t failure_slot = 0; failure_slot < failures.size(); ++failure_slot)
        {
            if (failure_rows[failure_slot] < selected_failure_row)
            {
                selected_failure_row = failure_rows[failure_slot];
                selected_failure = failures[failure_slot];
            }
        }
        if (selected_failure)
        {
            std::rethrow_exception(selected_failure);
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
    if (device == ProcessingDevice::CUDA || device == ProcessingDevice::Auto)
    {
        if (k > 32)
        {
            const std::string reason = "indexed normal estimation requires k <= 32";
            if (device == ProcessingDevice::CUDA)
            {
                throw std::invalid_argument("CUDA " + reason);
            }
            detail::appendFallbackReason(fallback_reason, "CUDA", reason);
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
                        report, device, ProcessingDevice::CUDA,
                        !fallback_reason.empty(), fallback_reason,
                        ProcessingNeighborBackend::GpuUniformGrid);
                    return gpu_normals.toCpu();
                }
                catch (const std::exception& ex)
                {
                    if (device == ProcessingDevice::CUDA)
                    {
                        throw;
                    }
                    detail::appendFallbackReason(fallback_reason, "CUDA", ex.what());
                }
            }
            else
            {
                const std::string reason = "device is not available";
                if (device == ProcessingDevice::CUDA)
                {
                    throw std::runtime_error("CUDA " + reason);
                }
                detail::appendFallbackReason(fallback_reason, "CUDA", reason);
            }
#else
            const std::string reason = "PlaPoint was built without CUDA support";
            if (device == ProcessingDevice::CUDA)
            {
                throw std::runtime_error(reason);
            }
            detail::appendFallbackReason(fallback_reason, "CUDA", reason);
#endif
        }
    }

    if (device == ProcessingDevice::OpenCL || device == ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_OPENCL
        if (k > 32)
        {
            const std::string reason = "uniform-grid normal estimation requires k <= 32";
            if (device == ProcessingDevice::OpenCL)
            {
                throw std::invalid_argument("OpenCL " + reason);
            }
            detail::appendFallbackReason(fallback_reason, "OpenCL", reason);
        }
        else
        {
            try
            {
                opencl::requireUsableOpenClDevice();
                auto normals = opencl::estimateNormals(input, k);
                detail::setReport(
                    report, device, ProcessingDevice::OpenCL,
                    !fallback_reason.empty(), fallback_reason,
                    ProcessingNeighborBackend::OpenClUniformGrid);
                return normals;
            }
            catch (const std::exception& ex)
            {
                if (device == ProcessingDevice::OpenCL)
                {
                    throw;
                }
                detail::appendFallbackReason(fallback_reason, "OpenCL", ex.what());
            }
        }
#else
        const std::string reason = "PlaPoint was built without OpenCL support";
        if (device == ProcessingDevice::OpenCL)
        {
            throw std::runtime_error(reason);
        }
        detail::appendFallbackReason(fallback_reason, "OpenCL", reason);
#endif
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
                      !fallback_reason.empty(),
                      fallback_reason,
                      ProcessingNeighborBackend::CpuKdTree);
    return normals;
}

} // namespace plapoint
