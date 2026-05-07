#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/dense/dense_matrix.h>
#include <plamatrix/ops/point_cloud.h>
#include <plamatrix/ops/decomposition.h>
#include <cmath>
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
    void setMaxIterations(int n) { _max_iter = n; }
    void setTransformationEpsilon(Scalar eps) { _eps = eps; }

    void align(PointCloudType& output)
    {
        if (!_source) throw std::runtime_error("ICP: source cloud not set");
        if (!_target) throw std::runtime_error("ICP: target cloud not set");

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

        for (int iter = 0; iter < _max_iter; ++iter)
        {
            // Find correspondences: for each source point, nearest target
            std::vector<int> corr(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                plamatrix::Vec3<Scalar> pt{cur(i, 0), cur(i, 1), cur(i, 2)};
                auto nn = tree->nearestKSearch(pt, 1);
                corr[static_cast<std::size_t>(i)] = nn.empty() ? 0 : nn[0];
            }

            // Compute centroids
            plamatrix::Vec3<Scalar> src_ct{0, 0, 0}, tgt_ct{0, 0, 0};
            for (int i = 0; i < n; ++i)
            {
                int j = corr[static_cast<std::size_t>(i)];
                src_ct.x += cur(i, 0); src_ct.y += cur(i, 1); src_ct.z += cur(i, 2);
                tgt_ct.x += tgt(j, 0); tgt_ct.y += tgt(j, 1); tgt_ct.z += tgt(j, 2);
            }
            src_ct.x /= Scalar(n); src_ct.y /= Scalar(n); src_ct.z /= Scalar(n);
            tgt_ct.x /= Scalar(n); tgt_ct.y /= Scalar(n); tgt_ct.z /= Scalar(n);

            // Cross-covariance H (3x3)
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> H(3, 3);
            H.fill(0);
            for (int i = 0; i < n; ++i)
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

            // Convergence check
            Scalar delta = std::abs(r00-1) + std::abs(r11-1) + std::abs(r22-1)
                         + std::abs(r01) + std::abs(r02) + std::abs(r10)
                         + std::abs(r12) + std::abs(r20) + std::abs(r21)
                         + std::abs(tx) + std::abs(ty) + std::abs(tz);
            if (delta < _eps)
            {
                _converged = true;
                break;
            }
        }

        _final_T = std::move(T_acc);
        output = PointCloudType(plamatrix::transformPoints(_final_T, src));
    }

    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& getFinalTransformation() const { return _final_T; }
    bool hasConverged() const { return _converged; }

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

    std::shared_ptr<const PointCloudType> _source;
    std::shared_ptr<const PointCloudType> _target;
    int _max_iter = 50;
    Scalar _eps = Scalar(1e-6);
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> _final_T;
    bool _converged = false;
};

} // namespace plapoint
