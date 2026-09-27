#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/features/feature.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/gpu/cuda_check.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/normal_estimation.h>
#endif
#ifdef PLAPOINT_WITH_OPENCL
#include <plapoint/opencl/normal_estimation.h>
#endif
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/matrix.h>
#include <plamatrix/dense/matrix_self_adjoint.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace plapoint {

template <typename Scalar, plamatrix::internal::Device Dev>
class MatrixNormalEstimation
{
public:
    using PointCloudType = plapoint::internal::DeviceCloud<Scalar, Dev>;
    using MatrixType = typename PointCloudType::MatrixType;

    void setInputCloud(const std::shared_ptr<const PointCloudType>& cloud) { _cloud = cloud; }
    void setSearchMethod(std::shared_ptr<search::internal::DeviceKdTree<Scalar, Dev>> tree) { _tree = tree; }
    void setKSearch(int k)
    {
        if (k < 3)
        {
            throw std::invalid_argument("MatrixNormalEstimation: k must be at least 3");
        }
        _k = k;
    }

    MatrixType compute() const
    {
        if (!_cloud) throw std::runtime_error("MatrixNormalEstimation: input cloud not set");
        if (!_tree)  throw std::runtime_error("MatrixNormalEstimation: search method not set");

        if constexpr (Dev == plamatrix::internal::Device::GPU)
        {
#ifndef PLAPOINT_WITH_CUDA
            throw std::runtime_error("MatrixNormalEstimation GPU requires PLAPOINT_WITH_CUDA=ON");
#else
            if (_k > 32)
            {
                return plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(
                    computeHostNormals(), _cloud->executionContext());
            }
            gpu::GpuSpatialIndex<Scalar> index;
            index.buildAdaptive(*_cloud);
            gpu::NormalEstimationGpuWorkspace<Scalar> workspace;
            plamatrix::internal::ResidentMatrix<Scalar> normals(
                static_cast<plamatrix::Index>(_cloud->size()), 3, _cloud->executionContext());
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
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> computeHostNormals() const
    {
        int n = static_cast<int>(_cloud->size());
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normals(n, 3);
        normals.setZero();

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

                plamatrix::Matrix<Scalar, 3, 3> covariance_matrix;
                covariance_matrix << covariance[0], covariance[1], covariance[2], covariance[1], covariance[3],
                    covariance[4], covariance[2], covariance[4], covariance[5];
                const plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix<Scalar, 3, 3>> solver(covariance_matrix);
                const auto& eigenvectors = solver.eigenvectors();

                normals.operator()(i, 0) = eigenvectors(0, 0);
                normals.operator()(i, 1) = eigenvectors(1, 0);
                normals.operator()(i, 2) = eigenvectors(2, 0);
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
    std::shared_ptr<search::internal::DeviceKdTree<Scalar, Dev>> _tree;
    int _k = 10;
};

namespace detail
{

/// Execute normal estimation against the internal device-backed cloud.
template <typename Scalar>
plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> estimateNormalsDeviceCloud(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& input,
    int k,
    ProcessingDevice device,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    const auto estimated_neighbors = k > 0 ? static_cast<std::size_t>(k) : std::size_t(0);
    if (device == ProcessingDevice::Auto &&
        !ProcessingPolicy::autoPrefersNeighborhoodGpu(input.size(), estimated_neighbors))
    {
        auto cloud = detail::nonOwningCloudPtr(input);
        auto tree = std::make_shared<search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
        tree->setInputCloud(cloud);
        tree->build();

        MatrixNormalEstimation<Scalar, plamatrix::internal::Device::CPU> estimator;
        estimator.setInputCloud(cloud);
        estimator.setSearchMethod(tree);
        estimator.setKSearch(k);
        auto normals = estimator.compute();
        detail::setAutoCpuReport(
            report,
            ProcessingNeighborBackend::CpuKdTree,
            "Auto selected CPU because normal-estimation work is below the accelerator threshold");
        return normals;
    }

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
                        const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>>(
                            std::move(gpu_cloud_value));
                    auto tree = std::make_shared<
                        search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::GPU>>();
                    tree->setInputCloud(gpu_cloud);
                    tree->build();

                    MatrixNormalEstimation<Scalar, plamatrix::internal::Device::GPU> estimator;
                    estimator.setInputCloud(gpu_cloud);
                    estimator.setSearchMethod(tree);
                    estimator.setKSearch(k);
                    auto gpu_normals = estimator.compute();
                    detail::setReport(
                        report, device, ProcessingDevice::CUDA,
                        !fallback_reason.empty(), fallback_reason,
                        ProcessingNeighborBackend::GpuUniformGrid);
                    return gpu_normals.toHostMatrix();
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
    auto tree = std::make_shared<search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    MatrixNormalEstimation<Scalar, plamatrix::internal::Device::CPU> estimator;
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

} // namespace detail

/// Estimate normals for CPU-owned geometry using the requested device where available.
template <typename Scalar>
typename GeometryCloud<Scalar>::MatrixType estimateNormals(
    const GeometryCloud<Scalar>& input,
    int k,
    ProcessingDevice device,
    ProcessingReport* report = nullptr)
{
    return detail::estimateNormalsDeviceCloud(detail::toDeviceCloud(input), k, device, report);
}

/// Estimate normals for point records while keeping the public feature contract compatible.
template <typename PointInT, typename PointOutT = Normal>
class NormalEstimation : public Feature<PointInT, PointOutT>
{
    static_assert(std::is_same_v<PointOutT, Normal> || std::is_same_v<PointOutT, PointNormal>,
                  "NormalEstimation output must be Normal or PointNormal");

public:
    using Base = Feature<PointInT, PointOutT>;
    using PointCloudIn = typename Base::PointCloudIn;
    using PointCloudOut = typename Base::PointCloudOut;
    using PointCloudConstPtr = typename Base::PointCloudInConstPtr;
    using Ptr = std::shared_ptr<NormalEstimation<PointInT, PointOutT>>;
    using ConstPtr = std::shared_ptr<const NormalEstimation<PointInT, PointOutT>>;

    NormalEstimation()
    {
        this->feature_name_ = "NormalEstimation";
    }

    bool computePointNormal(const PointCloudIn& cloud, const Indices& indices,
                            Eigen::Vector4f& plane_parameters, float& curvature)
    {
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 0.0f;
        plamatrix::Vector3d centroid;
        if (!solveNormal(cloud, indices, nx, ny, nz, curvature, &centroid))
        {
            plane_parameters.setConstant(std::numeric_limits<float>::quiet_NaN());
            return false;
        }
        plane_parameters << nx, ny, nz,
            static_cast<float>(-(static_cast<double>(nx) * centroid(0) +
                                 static_cast<double>(ny) * centroid(1) +
                                 static_cast<double>(nz) * centroid(2)));
        return true;
    }

    bool computePointNormal(const PointCloudIn& cloud, const Indices& indices,
                            float& nx, float& ny, float& nz, float& curvature)
    {
        return solveNormal(cloud, indices, nx, ny, nz, curvature, nullptr);
    }

    void setInputCloud(const PointCloudConstPtr& cloud) override
    {
        Base::setInputCloud(cloud);
        if (_use_sensor_origin)
        {
            if (cloud)
            {
                _view_x = cloud->sensor_origin_.x();
                _view_y = cloud->sensor_origin_.y();
                _view_z = cloud->sensor_origin_.z();
            }
            else
            {
                _view_x = _view_y = _view_z = 0.0f;
            }
        }
    }

    void setViewPoint(float x, float y, float z)
    {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            throw std::invalid_argument("NormalEstimation: viewpoint must be finite");
        }
        _view_x = x;
        _view_y = y;
        _view_z = z;
        _use_sensor_origin = false;
    }

    void getViewPoint(float& x, float& y, float& z)
    {
        x = _view_x;
        y = _view_y;
        z = _view_z;
    }

    void useSensorOriginAsViewPoint()
    {
        _use_sensor_origin = true;
        setSensorViewPoint(this->input_);
    }

protected:
    void computeFeature(PointCloudOut& output) override
    {
        for (std::size_t row = 0; row < output.size(); ++row)
        {
            const int input_index = this->indices_->at(row);
            if (input_index < 0 || static_cast<std::size_t>(input_index) >= this->input_->size())
            {
                throw std::out_of_range("NormalEstimation: input index is outside the cloud");
            }
            const auto& point = this->input_->points[static_cast<std::size_t>(input_index)];
            auto& normal = output.points[row];
            if constexpr (std::is_same_v<PointOutT, PointNormal>)
            {
                normal.x = point.x;
                normal.y = point.y;
                normal.z = point.z;
            }
            if (!std::isfinite(static_cast<double>(point.x)) ||
                !std::isfinite(static_cast<double>(point.y)) ||
                !std::isfinite(static_cast<double>(point.z)))
            {
                setInvalidNormal(normal);
                output.is_dense = false;
                continue;
            }

            Indices neighbors;
            std::vector<float> squared_distances;
            if (this->search_radius_ != 0.0)
            {
                this->tree_->radiusSearch(point, this->search_radius_, neighbors, squared_distances);
            }
            else
            {
                this->tree_->nearestKSearch(point, this->k_, neighbors, squared_distances);
            }

            float nx = 0.0f;
            float ny = 0.0f;
            float nz = 0.0f;
            if (!computePointNormal(*this->surface_, neighbors, nx, ny, nz, normal.curvature))
            {
                setInvalidNormal(normal);
                output.is_dense = false;
                continue;
            }
            const double view_x = static_cast<double>(_view_x) - point.x;
            const double view_y = static_cast<double>(_view_y) - point.y;
            const double view_z = static_cast<double>(_view_z) - point.z;
            if (nx * view_x + ny * view_y + nz * view_z < 0.0)
            {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            normal.normal_x = nx;
            normal.normal_y = ny;
            normal.normal_z = nz;
        }
    }

private:
    static bool solveNormal(const PointCloudIn& cloud, const Indices& indices,
                            float& nx, float& ny, float& nz, float& curvature,
                            plamatrix::Vector3d* centroid_output)
    {
        if (indices.size() < 3)
        {
            nx = ny = nz = curvature = std::numeric_limits<float>::quiet_NaN();
            return false;
        }
        plamatrix::Vector3d centroid = plamatrix::Vector3d::Zero();
        for (const int index : indices)
        {
            if (index < 0 || static_cast<std::size_t>(index) >= cloud.size())
            {
                nx = ny = nz = curvature = std::numeric_limits<float>::quiet_NaN();
                return false;
            }
            const auto& point = cloud.points[static_cast<std::size_t>(index)];
            if (!std::isfinite(static_cast<double>(point.x)) ||
                !std::isfinite(static_cast<double>(point.y)) ||
                !std::isfinite(static_cast<double>(point.z)))
            {
                nx = ny = nz = curvature = std::numeric_limits<float>::quiet_NaN();
                return false;
            }
            centroid(0) += static_cast<double>(point.x);
            centroid(1) += static_cast<double>(point.y);
            centroid(2) += static_cast<double>(point.z);
        }
        centroid /= static_cast<double>(indices.size());

        plamatrix::Matrix3d covariance = plamatrix::Matrix3d::Zero();
        for (const int index : indices)
        {
            const auto& point = cloud.points[static_cast<std::size_t>(index)];
            const plamatrix::Vector3d delta(static_cast<double>(point.x) - centroid(0),
                                            static_cast<double>(point.y) - centroid(1),
                                            static_cast<double>(point.z) - centroid(2));
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    covariance(row, column) += delta(row) * delta(column);
                }
            }
        }
        covariance /= static_cast<double>(indices.size());

        const plamatrix::SelfAdjointEigenSolver<plamatrix::Matrix3d> solver(covariance);
        const auto& values = solver.eigenvalues();
        const auto& vectors = solver.eigenvectors();
        const double trace = values(0) + values(1) + values(2);
        if (!std::isfinite(trace) || trace <= 0.0)
        {
            nx = ny = nz = curvature = std::numeric_limits<float>::quiet_NaN();
            return false;
        }
        nx = static_cast<float>(vectors(0, 0));
        ny = static_cast<float>(vectors(1, 0));
        nz = static_cast<float>(vectors(2, 0));
        curvature = static_cast<float>(std::max(0.0, values(0)) / trace);
        if (centroid_output)
        {
            *centroid_output = centroid;
        }
        return std::isfinite(nx) && std::isfinite(ny) && std::isfinite(nz) && std::isfinite(curvature);
    }

    void setSensorViewPoint(const PointCloudConstPtr& cloud)
    {
        if (cloud)
        {
            _view_x = cloud->sensor_origin_.x();
            _view_y = cloud->sensor_origin_.y();
            _view_z = cloud->sensor_origin_.z();
        }
        else
        {
            _view_x = _view_y = _view_z = 0.0f;
        }
    }

    static void setInvalidNormal(PointOutT& normal)
    {
        const float missing = std::numeric_limits<float>::quiet_NaN();
        normal.normal_x = missing;
        normal.normal_y = missing;
        normal.normal_z = missing;
        normal.curvature = missing;
    }

    float _view_x = 0.0f;
    float _view_y = 0.0f;
    float _view_z = 0.0f;
    bool _use_sensor_origin = true;
};

} // namespace plapoint
