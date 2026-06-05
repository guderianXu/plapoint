#pragma once

#include <plapoint/core/point_cloud.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/icp.h>
#endif
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class IterativeClosestPoint
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;
    using Matrix4 = plamatrix::DenseMatrix<Scalar, Dev>;

    void setInputSource(const std::shared_ptr<const PointCloudType>& cloud) { _source = cloud; }
    void setInputTarget(const std::shared_ptr<const PointCloudType>& cloud) { _target = cloud; }

    /// Set the maximum number of ICP iterations. Throws if n is not positive.
    void setMaxIterations(int n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("ICP: max iterations must be positive");
        }
        _max_iter = n;
    }

    /// Set convergence tolerance on the incremental transform. Throws if eps is not finite and positive.
    void setTransformationEpsilon(Scalar eps)
    {
        if (!std::isfinite(eps) || eps <= Scalar(0))
        {
            throw std::invalid_argument("ICP: transformation epsilon must be positive");
        }
        _eps = eps;
    }

    /// Set the largest accepted source-target correspondence distance. Positive infinity disables the limit.
    void setMaxCorrespondenceDistance(Scalar distance)
    {
        if (std::isnan(distance) || distance <= Scalar(0))
        {
            throw std::invalid_argument("ICP: max correspondence distance must be positive");
        }
        _max_corr_dist = distance;
    }

    /// Set the minimum required inlier correspondence ratio for convergence. Throws unless score is in [0, 1].
    void setMinFitnessScore(Scalar score)
    {
        if (!std::isfinite(score) || score < Scalar(0) || score > Scalar(1))
        {
            throw std::invalid_argument("ICP: minimum fitness score must be in [0, 1]");
        }
        _min_fitness_score = score;
    }

    /// Align the source cloud to the target cloud and write the transformed source to output.
    /// Throws for missing/empty clouds, too few valid correspondences, or degenerate correspondence geometry.
    void align(PointCloudType& output)
    {
        if (!_source) throw std::runtime_error("ICP: source cloud not set");
        if (!_target) throw std::runtime_error("ICP: target cloud not set");
        if (_source->size() == 0) throw std::invalid_argument("ICP: source cloud must not be empty");
        if (_target->size() == 0) throw std::invalid_argument("ICP: target cloud must not be empty");

        if constexpr (Dev == plamatrix::Device::GPU)
        {
#ifndef PLAPOINT_WITH_CUDA
            throw std::runtime_error("PlaPoint was built without CUDA support");
#else
            alignGpu(output);
            return;
#endif
        }

        // Build KD-tree on target
        auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
        tree->setInputCloud(_target);
        tree->build();

        int n = checkedInt(_source->size(), "ICP: source point count exceeds int range");
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> src = copyCpuMatrix(_source->pointsCpu());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> tgt = copyCpuMatrix(_target->pointsCpu());
        validateFinitePointMatrix(src, "ICP: source cloud contains non-finite point");

        // Accumulate transform as 4x4 identity
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> T_acc = identity4x4();

        // Copy src into cur (DenseMatrix is move-only)
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cur(src.rows(), src.cols());
        for (plamatrix::Index r = 0; r < src.rows(); ++r)
            for (int c = 0; c < 3; ++c)
                cur(r, c) = src(r, c);

        _converged = false;
        _fitness_score = Scalar(0);
        _final_rmse = std::numeric_limits<Scalar>::infinity();

        for (int iter = 0; iter < _max_iter; ++iter)
        {
            // Find correspondences: batch KNN (GPU-accelerated when Dev == GPU)
            std::vector<int> corr(static_cast<std::size_t>(n), -1);
            std::vector<int> active_indices;
            active_indices.reserve(static_cast<std::size_t>(n));
            collectCorrespondences(cur, tgt, *tree, corr, active_indices);

            if (active_indices.size() < 3)
            {
                throw std::runtime_error("ICP: fewer than 3 correspondences within max distance");
            }
            if (!hasNonCollinearGeometry(cur, active_indices))
            {
                throw std::runtime_error("ICP: correspondence geometry is degenerate");
            }
            std::vector<int> active_target_indices;
            active_target_indices.reserve(active_indices.size());
            for (int i : active_indices)
            {
                active_target_indices.push_back(corr[static_cast<std::size_t>(i)]);
            }
            if (!hasNonCollinearGeometry(tgt, active_target_indices))
            {
                throw std::runtime_error("ICP: target correspondence geometry is degenerate");
            }
            const int active_n = static_cast<int>(active_indices.size());
            updateResidualMetrics(cur, tgt, corr, active_indices, n);

            // Compute centroids in double to avoid overflowing float-scale accumulators.
            double src_ct[3]{0.0, 0.0, 0.0};
            double tgt_ct[3]{0.0, 0.0, 0.0};
            for (int i : active_indices)
            {
                int j = corr[static_cast<std::size_t>(i)];
                src_ct[0] += static_cast<double>(cur(i, 0));
                src_ct[1] += static_cast<double>(cur(i, 1));
                src_ct[2] += static_cast<double>(cur(i, 2));
                tgt_ct[0] += static_cast<double>(tgt(j, 0));
                tgt_ct[1] += static_cast<double>(tgt(j, 1));
                tgt_ct[2] += static_cast<double>(tgt(j, 2));
            }
            for (int c = 0; c < 3; ++c)
            {
                src_ct[c] /= static_cast<double>(active_n);
                tgt_ct[c] /= static_cast<double>(active_n);
                (void)finiteScalarFromDouble(src_ct[c], "ICP: correspondence centroid is not representable");
                (void)finiteScalarFromDouble(tgt_ct[c], "ICP: correspondence centroid is not representable");
            }

            // Cross-covariance H (3x3)
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> H(3, 3);
            H.fill(0);
            double h_acc[3][3]{};
            for (int i : active_indices)
            {
                int j = corr[static_cast<std::size_t>(i)];
                const double sx = static_cast<double>(cur(i, 0)) - src_ct[0];
                const double sy = static_cast<double>(cur(i, 1)) - src_ct[1];
                const double sz = static_cast<double>(cur(i, 2)) - src_ct[2];
                const double tx = static_cast<double>(tgt(j, 0)) - tgt_ct[0];
                const double ty = static_cast<double>(tgt(j, 1)) - tgt_ct[1];
                const double tz = static_cast<double>(tgt(j, 2)) - tgt_ct[2];
                h_acc[0][0] += sx * tx; h_acc[0][1] += sx * ty; h_acc[0][2] += sx * tz;
                h_acc[1][0] += sy * tx; h_acc[1][1] += sy * ty; h_acc[1][2] += sy * tz;
                h_acc[2][0] += sz * tx; h_acc[2][1] += sz * ty; h_acc[2][2] += sz * tz;
            }
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    H(r, c) = finiteScalarFromDouble(h_acc[r][c], "ICP: cross-covariance is not representable");
                }
            }

            auto [U, S, Vt] = plamatrix::svd(H);

            // R = V * U^T
            Scalar r00 = Vt.getValue(0,0)*U.getValue(0,0) + Vt.getValue(1,0)*U.getValue(0,1) + Vt.getValue(2,0)*U.getValue(0,2);
            Scalar r01 = Vt.getValue(0,0)*U.getValue(1,0) + Vt.getValue(1,0)*U.getValue(1,1) + Vt.getValue(2,0)*U.getValue(1,2);
            Scalar r02 = Vt.getValue(0,0)*U.getValue(2,0) + Vt.getValue(1,0)*U.getValue(2,1) + Vt.getValue(2,0)*U.getValue(2,2);
            Scalar r10 = Vt.getValue(0,1)*U.getValue(0,0) + Vt.getValue(1,1)*U.getValue(0,1) + Vt.getValue(2,1)*U.getValue(0,2);
            Scalar r11 = Vt.getValue(0,1)*U.getValue(1,0) + Vt.getValue(1,1)*U.getValue(1,1) + Vt.getValue(2,1)*U.getValue(1,2);
            Scalar r12 = Vt.getValue(0,1)*U.getValue(2,0) + Vt.getValue(1,1)*U.getValue(2,1) + Vt.getValue(2,1)*U.getValue(2,2);
            Scalar r20 = Vt.getValue(0,2)*U.getValue(0,0) + Vt.getValue(1,2)*U.getValue(0,1) + Vt.getValue(2,2)*U.getValue(0,2);
            Scalar r21 = Vt.getValue(0,2)*U.getValue(1,0) + Vt.getValue(1,2)*U.getValue(1,1) + Vt.getValue(2,2)*U.getValue(1,2);
            Scalar r22 = Vt.getValue(0,2)*U.getValue(2,0) + Vt.getValue(1,2)*U.getValue(2,1) + Vt.getValue(2,2)*U.getValue(2,2);

            // Handle reflection case
            Scalar det = r00*(r11*r22 - r12*r21) - r01*(r10*r22 - r12*r20) + r02*(r10*r21 - r11*r20);
            if (det < 0)
            {
                // Flip sign on last column of V
                r00 = Vt.getValue(0,0)*U.getValue(0,0) + Vt.getValue(1,0)*U.getValue(0,1) - Vt.getValue(2,0)*U.getValue(0,2);
                r01 = Vt.getValue(0,0)*U.getValue(1,0) + Vt.getValue(1,0)*U.getValue(1,1) - Vt.getValue(2,0)*U.getValue(1,2);
                r02 = Vt.getValue(0,0)*U.getValue(2,0) + Vt.getValue(1,0)*U.getValue(2,1) - Vt.getValue(2,0)*U.getValue(2,2);
                r10 = Vt.getValue(0,1)*U.getValue(0,0) + Vt.getValue(1,1)*U.getValue(0,1) - Vt.getValue(2,1)*U.getValue(0,2);
                r11 = Vt.getValue(0,1)*U.getValue(1,0) + Vt.getValue(1,1)*U.getValue(1,1) - Vt.getValue(2,1)*U.getValue(1,2);
                r12 = Vt.getValue(0,1)*U.getValue(2,0) + Vt.getValue(1,1)*U.getValue(2,1) - Vt.getValue(2,1)*U.getValue(2,2);
                r20 = Vt.getValue(0,2)*U.getValue(0,0) + Vt.getValue(1,2)*U.getValue(0,1) - Vt.getValue(2,2)*U.getValue(0,2);
                r21 = Vt.getValue(0,2)*U.getValue(1,0) + Vt.getValue(1,2)*U.getValue(1,1) - Vt.getValue(2,2)*U.getValue(1,2);
                r22 = Vt.getValue(0,2)*U.getValue(2,0) + Vt.getValue(1,2)*U.getValue(2,1) - Vt.getValue(2,2)*U.getValue(2,2);
            }

            const double tx_d = tgt_ct[0] - (static_cast<double>(r00) * src_ct[0]
                                           + static_cast<double>(r01) * src_ct[1]
                                           + static_cast<double>(r02) * src_ct[2]);
            const double ty_d = tgt_ct[1] - (static_cast<double>(r10) * src_ct[0]
                                           + static_cast<double>(r11) * src_ct[1]
                                           + static_cast<double>(r12) * src_ct[2]);
            const double tz_d = tgt_ct[2] - (static_cast<double>(r20) * src_ct[0]
                                           + static_cast<double>(r21) * src_ct[1]
                                           + static_cast<double>(r22) * src_ct[2]);
            Scalar tx = finiteScalarFromDouble(tx_d, "ICP: transform step is not representable");
            Scalar ty = finiteScalarFromDouble(ty_d, "ICP: transform step is not representable");
            Scalar tz = finiteScalarFromDouble(tz_d, "ICP: transform step is not representable");

            // Build step transform 4x4
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> T_step(4, 4);
            T_step.fill(0);
            T_step.setValue(0, 0, r00); T_step.setValue(0, 1, r01); T_step.setValue(0, 2, r02); T_step.setValue(0, 3, tx);
            T_step.setValue(1, 0, r10); T_step.setValue(1, 1, r11); T_step.setValue(1, 2, r12); T_step.setValue(1, 3, ty);
            T_step.setValue(2, 0, r20); T_step.setValue(2, 1, r21); T_step.setValue(2, 2, r22); T_step.setValue(2, 3, tz);
            T_step.setValue(3, 0, 0);   T_step.setValue(3, 1, 0);   T_step.setValue(3, 2, 0);   T_step.setValue(3, 3, 1);

            T_acc = multiply4x4(T_step, T_acc);
            cur = plamatrix::transformPoints(T_step, cur);
            validateFinitePointMatrix(cur, "ICP: transformed source contains non-finite point");
            std::vector<int> final_corr(static_cast<std::size_t>(n), -1);
            std::vector<int> final_active_indices;
            final_active_indices.reserve(static_cast<std::size_t>(n));
            collectCorrespondences(cur, tgt, *tree, final_corr, final_active_indices);
            if (final_active_indices.empty())
            {
                _fitness_score = Scalar(0);
                _final_rmse = std::numeric_limits<Scalar>::infinity();
            }
            else
            {
                updateResidualMetrics(cur, tgt, final_corr, final_active_indices, n);
            }

            // Convergence check
            Scalar delta = std::abs(r00-1) + std::abs(r11-1) + std::abs(r22-1)
                         + std::abs(r01) + std::abs(r02) + std::abs(r10)
                         + std::abs(r12) + std::abs(r20) + std::abs(r21)
                         + std::abs(tx) + std::abs(ty) + std::abs(tz);
            if (delta < _eps)
            {
                _converged = final_active_indices.size() >= 3 && _fitness_score >= _min_fitness_score;
                break;
            }
        }

        _final_T = std::move(T_acc);
        auto aligned = plamatrix::transformPoints(_final_T, src);
        validateFinitePointMatrix(aligned, "ICP: aligned output contains non-finite point");
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            output = PointCloudType(std::move(aligned));
        }
        else
        {
            output = PointCloudType(aligned.toGpu());
        }
    }

    /// Return the final 4x4 source-to-target transform on CPU.
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& getFinalTransformation() const { return _final_T; }

#ifdef PLAPOINT_WITH_CUDA
    /// Return the final 4x4 source-to-target transform on GPU after GPU align().
    /// Throws if align() has not populated a GPU final transform.
    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::GPU, const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>&>
    getFinalTransformationDevice() const
    {
        if (!_final_T_gpu)
        {
            throw std::runtime_error("ICP: GPU final transformation is not available");
        }
        return *_final_T_gpu;
    }
#endif

    /// Return true only if the transform converged and the final fitness meets the configured minimum.
    bool hasConverged() const { return _converged; }

    /// Return the final inlier correspondence ratio relative to the source point count.
    Scalar getFitnessScore() const { return _fitness_score; }

    /// Return the RMSE over the final accepted correspondences.
    Scalar getFinalRmse() const { return _final_rmse; }

private:
#ifdef PLAPOINT_WITH_CUDA
    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::GPU, void>
    alignGpu(PointCloudType& output)
    {
        const int source_count = checkedInt(_source->size(), "ICP: source point count exceeds int range");
        const int target_count = checkedInt(_target->size(), "ICP: target point count exceeds int range");

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> cur(source_count, 3);
        PLAPOINT_CHECK_CUDA(cudaMemcpy(cur.data(), _source->points().data(),
                                       static_cast<std::size_t>(source_count) * 3u * sizeof(Scalar),
                                       cudaMemcpyDeviceToDevice));

        gpu::DeviceBuffer<int> d_corr(static_cast<std::size_t>(source_count));
        auto T_acc_gpu = identity4x4().toGpu();

        _converged = false;
        _fitness_score = Scalar(0);
        _final_rmse = std::numeric_limits<Scalar>::infinity();
        _final_T_gpu.reset();

        for (int iter = 0; iter < _max_iter; ++iter)
        {
            auto stats = gpu::computeIcpCorrespondenceStatsColumnMajor(
                cur.data(),
                source_count,
                _target->points().data(),
                target_count,
                _max_corr_dist,
                d_corr.get());
            if (stats.invalid_source_count > 0)
            {
                throw std::invalid_argument(iter == 0
                    ? "ICP: source cloud contains non-finite point"
                    : "ICP: transformed source contains non-finite point");
            }
            if (stats.active_count < 3)
            {
                throw std::runtime_error("ICP: fewer than 3 correspondences within max distance");
            }
            if (!hasNonCollinearCovariance(stats.src_covariance))
            {
                throw std::runtime_error("ICP: correspondence geometry is degenerate");
            }
            if (!hasNonCollinearCovariance(stats.tgt_covariance))
            {
                throw std::runtime_error("ICP: target correspondence geometry is degenerate");
            }

            updateResidualMetricsFromGpuStats(stats, source_count);
            auto T_step = computeStepTransformFromGpuStats(stats);
            auto T_step_gpu = T_step.toGpu();
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> next_T_acc_gpu(4, 4);
            gpu::multiplyTransform4x4(T_step_gpu.data(), T_acc_gpu.data(), next_T_acc_gpu.data());
            T_acc_gpu = std::move(next_T_acc_gpu);
            cur = plamatrix::transformPoints(T_step_gpu, cur);

            auto final_stats = gpu::computeIcpCorrespondenceStatsColumnMajor(
                cur.data(),
                source_count,
                _target->points().data(),
                target_count,
                _max_corr_dist,
                d_corr.get());
            if (final_stats.invalid_source_count > 0)
            {
                throw std::invalid_argument("ICP: transformed source contains non-finite point");
            }
            if (final_stats.active_count == 0)
            {
                _fitness_score = Scalar(0);
                _final_rmse = std::numeric_limits<Scalar>::infinity();
            }
            else
            {
                updateResidualMetricsFromGpuStats(final_stats, source_count);
            }

            const Scalar r00 = T_step.getValue(0, 0);
            const Scalar r01 = T_step.getValue(0, 1);
            const Scalar r02 = T_step.getValue(0, 2);
            const Scalar r10 = T_step.getValue(1, 0);
            const Scalar r11 = T_step.getValue(1, 1);
            const Scalar r12 = T_step.getValue(1, 2);
            const Scalar r20 = T_step.getValue(2, 0);
            const Scalar r21 = T_step.getValue(2, 1);
            const Scalar r22 = T_step.getValue(2, 2);
            const Scalar tx = T_step.getValue(0, 3);
            const Scalar ty = T_step.getValue(1, 3);
            const Scalar tz = T_step.getValue(2, 3);
            const Scalar delta = std::abs(r00 - 1) + std::abs(r11 - 1) + std::abs(r22 - 1)
                               + std::abs(r01) + std::abs(r02) + std::abs(r10)
                               + std::abs(r12) + std::abs(r20) + std::abs(r21)
                               + std::abs(tx) + std::abs(ty) + std::abs(tz);
            if (delta < _eps)
            {
                _converged = final_stats.active_count >= 3 && _fitness_score >= _min_fitness_score;
                break;
            }
        }

        _final_T = T_acc_gpu.toCpu();
        _final_T_gpu =
            std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>>(std::move(T_acc_gpu));
        output = PointCloudType(std::move(cur));
    }

    static bool hasNonCollinearCovariance(const double covariance[9])
    {
        const double trace = covariance[0] + covariance[4] + covariance[8];
        if (!std::isfinite(trace) || trace <= 0.0)
        {
            return false;
        }

        plamatrix::DenseMatrix<double, plamatrix::Device::CPU> matrix(3, 3);
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                matrix(r, c) = covariance[r * 3 + c];
            }
        }
        auto [U, S, Vt] = plamatrix::svd(matrix);
        (void)U;
        (void)Vt;
        const double second_singular_value = S.getValue(1, 0);
        return std::isfinite(second_singular_value) &&
               second_singular_value > std::max(trace * 1.0e-12, 1.0e-30);
    }

    void updateResidualMetricsFromGpuStats(
        const gpu::IcpCorrespondenceStats<Scalar>& stats,
        int source_count)
    {
        if (stats.active_count <= 0)
        {
            _fitness_score = Scalar(0);
            _final_rmse = std::numeric_limits<Scalar>::infinity();
            return;
        }
        if (!std::isfinite(stats.residual_sq_sum))
        {
            throw std::runtime_error("ICP: residual distance is not finite");
        }

        const double rmse = std::sqrt(stats.residual_sq_sum / static_cast<double>(stats.active_count));
        _fitness_score = static_cast<Scalar>(stats.active_count) / static_cast<Scalar>(source_count);
        _final_rmse = metricScalarFromDouble(rmse);
    }

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> computeStepTransformFromGpuStats(
        const gpu::IcpCorrespondenceStats<Scalar>& stats)
    {
        for (int c = 0; c < 3; ++c)
        {
            (void)finiteScalarFromDouble(stats.src_centroid[c], "ICP: correspondence centroid is not representable");
            (void)finiteScalarFromDouble(stats.tgt_centroid[c], "ICP: correspondence centroid is not representable");
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> H(3, 3);
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                H(r, c) = finiteScalarFromDouble(
                    stats.cross_covariance[r * 3 + c],
                    "ICP: cross-covariance is not representable");
            }
        }

        auto [U, S, Vt] = plamatrix::svd(H);
        (void)S;

        Scalar r00 = Vt.getValue(0,0)*U.getValue(0,0) + Vt.getValue(1,0)*U.getValue(0,1) + Vt.getValue(2,0)*U.getValue(0,2);
        Scalar r01 = Vt.getValue(0,0)*U.getValue(1,0) + Vt.getValue(1,0)*U.getValue(1,1) + Vt.getValue(2,0)*U.getValue(1,2);
        Scalar r02 = Vt.getValue(0,0)*U.getValue(2,0) + Vt.getValue(1,0)*U.getValue(2,1) + Vt.getValue(2,0)*U.getValue(2,2);
        Scalar r10 = Vt.getValue(0,1)*U.getValue(0,0) + Vt.getValue(1,1)*U.getValue(0,1) + Vt.getValue(2,1)*U.getValue(0,2);
        Scalar r11 = Vt.getValue(0,1)*U.getValue(1,0) + Vt.getValue(1,1)*U.getValue(1,1) + Vt.getValue(2,1)*U.getValue(1,2);
        Scalar r12 = Vt.getValue(0,1)*U.getValue(2,0) + Vt.getValue(1,1)*U.getValue(2,1) + Vt.getValue(2,1)*U.getValue(2,2);
        Scalar r20 = Vt.getValue(0,2)*U.getValue(0,0) + Vt.getValue(1,2)*U.getValue(0,1) + Vt.getValue(2,2)*U.getValue(0,2);
        Scalar r21 = Vt.getValue(0,2)*U.getValue(1,0) + Vt.getValue(1,2)*U.getValue(1,1) + Vt.getValue(2,2)*U.getValue(1,2);
        Scalar r22 = Vt.getValue(0,2)*U.getValue(2,0) + Vt.getValue(1,2)*U.getValue(2,1) + Vt.getValue(2,2)*U.getValue(2,2);

        Scalar det = r00*(r11*r22 - r12*r21) - r01*(r10*r22 - r12*r20) + r02*(r10*r21 - r11*r20);
        if (det < 0)
        {
            r00 = Vt.getValue(0,0)*U.getValue(0,0) + Vt.getValue(1,0)*U.getValue(0,1) - Vt.getValue(2,0)*U.getValue(0,2);
            r01 = Vt.getValue(0,0)*U.getValue(1,0) + Vt.getValue(1,0)*U.getValue(1,1) - Vt.getValue(2,0)*U.getValue(1,2);
            r02 = Vt.getValue(0,0)*U.getValue(2,0) + Vt.getValue(1,0)*U.getValue(2,1) - Vt.getValue(2,0)*U.getValue(2,2);
            r10 = Vt.getValue(0,1)*U.getValue(0,0) + Vt.getValue(1,1)*U.getValue(0,1) - Vt.getValue(2,1)*U.getValue(0,2);
            r11 = Vt.getValue(0,1)*U.getValue(1,0) + Vt.getValue(1,1)*U.getValue(1,1) - Vt.getValue(2,1)*U.getValue(1,2);
            r12 = Vt.getValue(0,1)*U.getValue(2,0) + Vt.getValue(1,1)*U.getValue(2,1) - Vt.getValue(2,1)*U.getValue(2,2);
            r20 = Vt.getValue(0,2)*U.getValue(0,0) + Vt.getValue(1,2)*U.getValue(0,1) - Vt.getValue(2,2)*U.getValue(0,2);
            r21 = Vt.getValue(0,2)*U.getValue(1,0) + Vt.getValue(1,2)*U.getValue(1,1) - Vt.getValue(2,2)*U.getValue(1,2);
            r22 = Vt.getValue(0,2)*U.getValue(2,0) + Vt.getValue(1,2)*U.getValue(2,1) - Vt.getValue(2,2)*U.getValue(2,2);
        }

        const double tx_d = stats.tgt_centroid[0] - (static_cast<double>(r00) * stats.src_centroid[0]
                                                   + static_cast<double>(r01) * stats.src_centroid[1]
                                                   + static_cast<double>(r02) * stats.src_centroid[2]);
        const double ty_d = stats.tgt_centroid[1] - (static_cast<double>(r10) * stats.src_centroid[0]
                                                   + static_cast<double>(r11) * stats.src_centroid[1]
                                                   + static_cast<double>(r12) * stats.src_centroid[2]);
        const double tz_d = stats.tgt_centroid[2] - (static_cast<double>(r20) * stats.src_centroid[0]
                                                   + static_cast<double>(r21) * stats.src_centroid[1]
                                                   + static_cast<double>(r22) * stats.src_centroid[2]);
        Scalar tx = finiteScalarFromDouble(tx_d, "ICP: transform step is not representable");
        Scalar ty = finiteScalarFromDouble(ty_d, "ICP: transform step is not representable");
        Scalar tz = finiteScalarFromDouble(tz_d, "ICP: transform step is not representable");

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> T_step(4, 4);
        T_step.fill(0);
        T_step.setValue(0, 0, r00); T_step.setValue(0, 1, r01); T_step.setValue(0, 2, r02); T_step.setValue(0, 3, tx);
        T_step.setValue(1, 0, r10); T_step.setValue(1, 1, r11); T_step.setValue(1, 2, r12); T_step.setValue(1, 3, ty);
        T_step.setValue(2, 0, r20); T_step.setValue(2, 1, r21); T_step.setValue(2, 2, r22); T_step.setValue(2, 3, tz);
        T_step.setValue(3, 0, 0);   T_step.setValue(3, 1, 0);   T_step.setValue(3, 2, 0);   T_step.setValue(3, 3, 1);
        return T_step;
    }
#endif

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> identity4x4()
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> I(4, 4);
        I.fill(0);
        I(0, 0) = 1; I(1, 1) = 1; I(2, 2) = 1; I(3, 3) = 1;
        return I;
    }

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> copyCpuMatrix(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& matrix)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> copy(matrix.rows(), matrix.cols());
        for (plamatrix::Index i = 0; i < matrix.rows(); ++i)
            for (plamatrix::Index j = 0; j < matrix.cols(); ++j)
                copy(i, j) = matrix(i, j);
        return copy;
    }

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> multiply4x4(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& A,
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& B)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> C(4, 4);
        C.fill(0);
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k)
                {
                    sum += static_cast<double>(A(i, k)) * static_cast<double>(B(k, j));
                }
                C(i, j) = finiteScalarFromDouble(sum, "ICP: accumulated transform is not representable");
            }
        }
        return C;
    }

    static void validateFinitePointMatrix(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& points,
        const char* message)
    {
        for (plamatrix::Index r = 0; r < points.rows(); ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                if (!std::isfinite(points(r, c)))
                {
                    throw std::invalid_argument(message);
                }
            }
        }
    }

    static bool hasNonCollinearGeometry(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& points,
        const std::vector<int>& active_indices)
    {
        if (active_indices.size() < 3)
        {
            return false;
        }

        long double coord_scale = 0.0L;
        for (int idx : active_indices)
        {
            for (int c = 0; c < 3; ++c)
            {
                const long double value = static_cast<long double>(points(idx, c));
                if (!std::isfinite(value))
                {
                    return false;
                }
                coord_scale = std::max(coord_scale, std::abs(value));
            }
        }
        if (coord_scale <= 0.0L)
        {
            return false;
        }
        const auto normalized = [&](int idx, int c) -> long double {
            return static_cast<long double>(points(idx, c)) / coord_scale;
        };

        const int axis_a = active_indices.front();
        int axis_b = axis_a;
        long double max_len = 0.0L;
        for (int idx : active_indices)
        {
            const long double dx = normalized(idx, 0) - normalized(axis_a, 0);
            const long double dy = normalized(idx, 1) - normalized(axis_a, 1);
            const long double dz = normalized(idx, 2) - normalized(axis_a, 2);
            const long double len = std::hypot(dx, dy, dz);
            if (std::isfinite(len) && len > max_len)
            {
                max_len = len;
                axis_b = idx;
            }
        }
        if (max_len <= 0.0L)
        {
            return false;
        }

        const long double ax = normalized(axis_b, 0) - normalized(axis_a, 0);
        const long double ay = normalized(axis_b, 1) - normalized(axis_a, 1);
        const long double az = normalized(axis_b, 2) - normalized(axis_a, 2);
        const long double inv_a = 1.0L / max_len;
        const long double anx = ax * inv_a;
        const long double any = ay * inv_a;
        const long double anz = az * inv_a;
        long double max_cross = 0.0L;
        for (int idx : active_indices)
        {
            const long double bx = normalized(idx, 0) - normalized(axis_a, 0);
            const long double by = normalized(idx, 1) - normalized(axis_a, 1);
            const long double bz = normalized(idx, 2) - normalized(axis_a, 2);
            const long double b_len = std::hypot(bx, by, bz);
            if (!std::isfinite(b_len) || b_len <= 0.0L)
            {
                continue;
            }
            const long double inv_b = 1.0L / b_len;
            const long double bnx = bx * inv_b;
            const long double bny = by * inv_b;
            const long double bnz = bz * inv_b;
            const long double cx = any * bnz - anz * bny;
            const long double cy = anz * bnx - anx * bnz;
            const long double cz = anx * bny - any * bnx;
            max_cross = std::max(max_cross, std::hypot(cx, cy, cz));
        }

        constexpr long double kMinSineAngle = 1.0e-6L;
        return max_cross > kMinSineAngle;
    }

    void collectCorrespondences(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& source_points,
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& target_points,
        const search::KdTree<Scalar, Dev>& tree,
        std::vector<int>& corr,
        std::vector<int>& active_indices) const
    {
        const int n = static_cast<int>(source_points.rows());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(n, 3);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c)
                queries(i, c) = source_points(i, c);

        auto all_nn = tree.batchNearestKSearch(queries, 1);
        for (int i = 0; i < n; ++i)
        {
            if (all_nn[static_cast<std::size_t>(i)].empty())
            {
                continue;
            }

            const int j = all_nn[static_cast<std::size_t>(i)][0];
            if (j < 0 || j >= target_points.rows())
            {
                continue;
            }
            const double dx = static_cast<double>(source_points(i, 0)) - static_cast<double>(target_points(j, 0));
            const double dy = static_cast<double>(source_points(i, 1)) - static_cast<double>(target_points(j, 1));
            const double dz = static_cast<double>(source_points(i, 2)) - static_cast<double>(target_points(j, 2));
            if (withinMaxCorrespondenceDistance(dx, dy, dz))
            {
                corr[static_cast<std::size_t>(i)] = j;
                active_indices.push_back(i);
            }
        }
    }

    void updateResidualMetrics(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& source_points,
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& target_points,
        const std::vector<int>& corr,
        const std::vector<int>& active_indices,
        int source_count)
    {
        double scale = 0.0;
        double scaled_sq_sum = 0.0;
        for (int i : active_indices)
        {
            const int j = corr[static_cast<std::size_t>(i)];
            const double dx = static_cast<double>(source_points(i, 0)) - static_cast<double>(target_points(j, 0));
            const double dy = static_cast<double>(source_points(i, 1)) - static_cast<double>(target_points(j, 1));
            const double dz = static_cast<double>(source_points(i, 2)) - static_cast<double>(target_points(j, 2));
            const double dist = std::hypot(dx, dy, dz);
            if (!std::isfinite(dist))
            {
                throw std::runtime_error("ICP: residual distance is not finite");
            }
            if (dist == 0.0)
            {
                continue;
            }
            if (dist > scale)
            {
                const double ratio = scale / dist;
                scaled_sq_sum = scaled_sq_sum * ratio * ratio + 1.0;
                scale = dist;
            }
            else
            {
                const double ratio = dist / scale;
                scaled_sq_sum += ratio * ratio;
            }
        }
        const double rmse = (scale == 0.0)
            ? 0.0
            : scale * std::sqrt(scaled_sq_sum / static_cast<double>(active_indices.size()));
        _fitness_score = static_cast<Scalar>(active_indices.size()) / static_cast<Scalar>(source_count);
        _final_rmse = metricScalarFromDouble(rmse);
    }

    bool withinMaxCorrespondenceDistance(double dx, double dy, double dz) const
    {
        const double dist = std::hypot(dx, dy, dz);
        if (!std::isfinite(dist))
        {
            return false;
        }
        if (!std::isfinite(_max_corr_dist))
        {
            return true;
        }
        return dist <= static_cast<double>(_max_corr_dist);
    }

    static int checkedInt(std::size_t value, const char* message)
    {
        if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error(message);
        }
        return static_cast<int>(value);
    }

    static Scalar finiteScalarFromDouble(double value, const char* message)
    {
        if (!std::isfinite(value) ||
            std::abs(value) > static_cast<double>(std::numeric_limits<Scalar>::max()))
        {
            throw std::runtime_error(message);
        }
        return static_cast<Scalar>(value);
    }

    static Scalar metricScalarFromDouble(double value)
    {
        const Scalar max_value = std::numeric_limits<Scalar>::max();
        if (!std::isfinite(value) || value >= static_cast<double>(max_value))
        {
            return max_value;
        }
        return static_cast<Scalar>(value);
    }

    std::shared_ptr<const PointCloudType> _source;
    std::shared_ptr<const PointCloudType> _target;
    int _max_iter = 50;
    Scalar _eps = Scalar(1e-6);
    Scalar _max_corr_dist = std::numeric_limits<Scalar>::infinity();
    Scalar _min_fitness_score = Scalar(0);
    Scalar _fitness_score = Scalar(0);
    Scalar _final_rmse = std::numeric_limits<Scalar>::infinity();
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> _final_T;
#ifdef PLAPOINT_WITH_CUDA
    std::unique_ptr<plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>> _final_T_gpu;
#endif
    bool _converged = false;
};

} // namespace plapoint
