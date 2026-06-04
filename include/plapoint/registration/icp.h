#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
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

    /// Set the largest accepted source-target correspondence distance. Throws if distance is not finite and positive.
    void setMaxCorrespondenceDistance(Scalar distance)
    {
        if (!std::isfinite(distance) || distance <= Scalar(0))
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

        // Build KD-tree on target
        auto tree = std::make_shared<search::KdTree<Scalar, Dev>>();
        tree->setInputCloud(_target);
        tree->build();

        int n = static_cast<int>(_source->size());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> src = toCpuCopy(_source->points());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> tgt = toCpuCopy(_target->points());

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

            // Compute centroids
            plamatrix::Vec3<Scalar> src_ct{0, 0, 0}, tgt_ct{0, 0, 0};
            for (int i : active_indices)
            {
                int j = corr[static_cast<std::size_t>(i)];
                src_ct.x += cur(i, 0); src_ct.y += cur(i, 1); src_ct.z += cur(i, 2);
                tgt_ct.x += tgt(j, 0); tgt_ct.y += tgt(j, 1); tgt_ct.z += tgt(j, 2);
            }
            src_ct.x /= Scalar(active_n); src_ct.y /= Scalar(active_n); src_ct.z /= Scalar(active_n);
            tgt_ct.x /= Scalar(active_n); tgt_ct.y /= Scalar(active_n); tgt_ct.z /= Scalar(active_n);

            // Cross-covariance H (3x3)
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> H(3, 3);
            H.fill(0);
            for (int i : active_indices)
            {
                int j = corr[static_cast<std::size_t>(i)];
                Scalar sx = cur(i, 0) - src_ct.x;
                Scalar sy = cur(i, 1) - src_ct.y;
                Scalar sz = cur(i, 2) - src_ct.z;
                Scalar tx = tgt(j, 0) - tgt_ct.x;
                Scalar ty = tgt(j, 1) - tgt_ct.y;
                Scalar tz = tgt(j, 2) - tgt_ct.z;
                H(0, 0) += sx * tx; H(0, 1) += sx * ty; H(0, 2) += sx * tz;
                H(1, 0) += sy * tx; H(1, 1) += sy * ty; H(1, 2) += sy * tz;
                H(2, 0) += sz * tx; H(2, 1) += sz * ty; H(2, 2) += sz * tz;
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
                r00 = -Vt.getValue(0,0)*U.getValue(0,0) - Vt.getValue(1,0)*U.getValue(0,1) + Vt.getValue(2,0)*U.getValue(0,2);
                r01 = -Vt.getValue(0,0)*U.getValue(1,0) - Vt.getValue(1,0)*U.getValue(1,1) + Vt.getValue(2,0)*U.getValue(1,2);
                r02 = -Vt.getValue(0,0)*U.getValue(2,0) - Vt.getValue(1,0)*U.getValue(2,1) + Vt.getValue(2,0)*U.getValue(2,2);
                r10 = -Vt.getValue(0,1)*U.getValue(0,0) - Vt.getValue(1,1)*U.getValue(0,1) + Vt.getValue(2,1)*U.getValue(0,2);
                r11 = -Vt.getValue(0,1)*U.getValue(1,0) - Vt.getValue(1,1)*U.getValue(1,1) + Vt.getValue(2,1)*U.getValue(1,2);
                r12 = -Vt.getValue(0,1)*U.getValue(2,0) - Vt.getValue(1,1)*U.getValue(2,1) + Vt.getValue(2,1)*U.getValue(2,2);
                r20 = -Vt.getValue(0,2)*U.getValue(0,0) - Vt.getValue(1,2)*U.getValue(0,1) + Vt.getValue(2,2)*U.getValue(0,2);
                r21 = -Vt.getValue(0,2)*U.getValue(1,0) - Vt.getValue(1,2)*U.getValue(1,1) + Vt.getValue(2,2)*U.getValue(1,2);
                r22 = -Vt.getValue(0,2)*U.getValue(2,0) - Vt.getValue(1,2)*U.getValue(2,1) + Vt.getValue(2,2)*U.getValue(2,2);
            }

            Scalar tx = tgt_ct.x - (r00*src_ct.x + r01*src_ct.y + r02*src_ct.z);
            Scalar ty = tgt_ct.y - (r10*src_ct.x + r11*src_ct.y + r12*src_ct.z);
            Scalar tz = tgt_ct.z - (r20*src_ct.x + r21*src_ct.y + r22*src_ct.z);

            // Build step transform 4x4
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> T_step(4, 4);
            T_step.fill(0);
            T_step.setValue(0, 0, r00); T_step.setValue(0, 1, r01); T_step.setValue(0, 2, r02); T_step.setValue(0, 3, tx);
            T_step.setValue(1, 0, r10); T_step.setValue(1, 1, r11); T_step.setValue(1, 2, r12); T_step.setValue(1, 3, ty);
            T_step.setValue(2, 0, r20); T_step.setValue(2, 1, r21); T_step.setValue(2, 2, r22); T_step.setValue(2, 3, tz);
            T_step.setValue(3, 0, 0);   T_step.setValue(3, 1, 0);   T_step.setValue(3, 2, 0);   T_step.setValue(3, 3, 1);

            T_acc = multiply4x4(T_step, T_acc);
            cur = plamatrix::transformPoints(T_step, cur);
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

    /// Return true only if the transform converged and the final fitness meets the configured minimum.
    bool hasConverged() const { return _converged; }

    /// Return the final inlier correspondence ratio relative to the source point count.
    Scalar getFitnessScore() const { return _fitness_score; }

    /// Return the RMSE over the final accepted correspondences.
    Scalar getFinalRmse() const { return _final_rmse; }

private:
    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> identity4x4()
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> I(4, 4);
        I.fill(0);
        I(0, 0) = 1; I(1, 1) = 1; I(2, 2) = 1; I(3, 3) = 1;
        return I;
    }

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> toCpuCopy(
        const plamatrix::DenseMatrix<Scalar, Dev>& m)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> copy(m.rows(), m.cols());
            for (plamatrix::Index i = 0; i < m.rows(); ++i)
                for (plamatrix::Index j = 0; j < m.cols(); ++j)
                    copy(i, j) = m(i, j);
            return copy;
        }
        else
        {
            return m.toCpu();
        }
    }

    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> multiply4x4(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& A,
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& B)
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> C(4, 4);
        C.fill(0);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    C(i, j) += A(i, k) * B(k, j);
        return C;
    }

    static bool hasNonCollinearGeometry(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& points,
        const std::vector<int>& active_indices)
    {
        const int first = active_indices.front();
        const Scalar x0 = points(first, 0);
        const Scalar y0 = points(first, 1);
        const Scalar z0 = points(first, 2);
        Scalar ax = 0;
        Scalar ay = 0;
        Scalar az = 0;
        bool have_axis = false;
        constexpr Scalar EPS = Scalar(1e-8);

        for (int idx : active_indices)
        {
            ax = points(idx, 0) - x0;
            ay = points(idx, 1) - y0;
            az = points(idx, 2) - z0;
            if (ax * ax + ay * ay + az * az > EPS)
            {
                have_axis = true;
                break;
            }
        }
        if (!have_axis)
        {
            return false;
        }

        for (int idx : active_indices)
        {
            const Scalar bx = points(idx, 0) - x0;
            const Scalar by = points(idx, 1) - y0;
            const Scalar bz = points(idx, 2) - z0;
            const Scalar cx = ay * bz - az * by;
            const Scalar cy = az * bx - ax * bz;
            const Scalar cz = ax * by - ay * bx;
            if (cx * cx + cy * cy + cz * cz > EPS)
            {
                return true;
            }
        }
        return false;
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

        const Scalar max_corr_sq = _max_corr_dist * _max_corr_dist;
        auto all_nn = tree.batchNearestKSearch(queries, 1);
        for (int i = 0; i < n; ++i)
        {
            if (all_nn[static_cast<std::size_t>(i)].empty())
            {
                continue;
            }

            const int j = all_nn[static_cast<std::size_t>(i)][0];
            const Scalar dx = source_points(i, 0) - target_points(j, 0);
            const Scalar dy = source_points(i, 1) - target_points(j, 1);
            const Scalar dz = source_points(i, 2) - target_points(j, 2);
            const Scalar d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= max_corr_sq)
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
        Scalar sq_error_sum = 0;
        for (int i : active_indices)
        {
            const int j = corr[static_cast<std::size_t>(i)];
            const Scalar dx = source_points(i, 0) - target_points(j, 0);
            const Scalar dy = source_points(i, 1) - target_points(j, 1);
            const Scalar dz = source_points(i, 2) - target_points(j, 2);
            sq_error_sum += dx * dx + dy * dy + dz * dz;
        }
        _fitness_score = static_cast<Scalar>(active_indices.size()) / static_cast<Scalar>(source_count);
        _final_rmse = std::sqrt(sq_error_sum / static_cast<Scalar>(active_indices.size()));
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
    bool _converged = false;
};

} // namespace plapoint
