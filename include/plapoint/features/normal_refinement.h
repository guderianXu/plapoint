#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/ops/point_cloud.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class NormalRefinement
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<PointCloudType>& cloud) { _cloud = cloud; }
    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree) { _tree = tree; }

    /// Smooth existing normals by averaging each point's k nearest neighbor normals.
    /// Throws if the cloud, search method, normals, or k are invalid.
    void smooth(int k)
    {
        if (!_cloud) throw std::runtime_error("NormalRefinement: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalRefinement: search method not set");
        if (!_cloud->hasNormals()) throw std::runtime_error("NormalRefinement: cloud has no normals");
        if (k <= 0) throw std::invalid_argument("NormalRefinement: k must be positive");

        int n = static_cast<int>(_cloud->size());
        const auto& points_cpu = _cloud->pointsCpu();
        auto normals_cpu = toCpuCopy(*_cloud->normals());

        std::vector<plamatrix::Vec3<Scalar>> temp(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            temp[static_cast<std::size_t>(i)] = {
                normals_cpu(i, 0),
                normals_cpu(i, 1),
                normals_cpu(i, 2)
            };

        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(points_cpu, i);
            auto neighbors = _tree->nearestKSearch(pt, k);
            Scalar sx = 0, sy = 0, sz = 0;
            for (int nb : neighbors)
            {
                sx += temp[static_cast<std::size_t>(nb)].x;
                sy += temp[static_cast<std::size_t>(nb)].y;
                sz += temp[static_cast<std::size_t>(nb)].z;
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
    void orientConsistently(const plamatrix::Vec3<Scalar>& viewpoint)
    {
        if (!_cloud || !_cloud->hasNormals()) return;
        int n = static_cast<int>(_cloud->size());
        const auto& points_cpu = _cloud->pointsCpu();
        auto normals_cpu = toCpuCopy(*_cloud->normals());
        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(points_cpu, i);
            Scalar dx = viewpoint.x - pt.x, dy = viewpoint.y - pt.y, dz = viewpoint.z - pt.z;
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
    static plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> toCpuCopy(
        const plamatrix::DenseMatrix<Scalar, Dev>& m)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> copy(m.rows(), m.cols());
            for (plamatrix::Index r = 0; r < m.rows(); ++r)
                for (plamatrix::Index c = 0; c < m.cols(); ++c)
                    copy(r, c) = m(r, c);
            return copy;
        }
        else
        {
            return m.toCpu();
        }
    }

    static plamatrix::Vec3<Scalar> pointVec(
        const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>& points,
        int idx)
    {
        return {points(idx, 0), points(idx, 1), points(idx, 2)};
    }

    void setCloudNormals(plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>&& normals_cpu)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            _cloud->setNormals(std::move(normals_cpu));
        }
        else
        {
            _cloud->setNormals(normals_cpu.toGpu());
        }
    }

    std::shared_ptr<PointCloudType> _cloud;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
};

} // namespace plapoint
