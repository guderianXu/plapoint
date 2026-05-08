#pragma once

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/ops/point_cloud.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class NormalRefinement
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setInputCloud(const std::shared_ptr<PointCloudType>& cloud) { _cloud = cloud; }
    void setSearchMethod(std::shared_ptr<search::KdTree<Scalar, Dev>> tree) { _tree = tree; }

    void smooth(int k)
    {
        if (!_cloud) throw std::runtime_error("NormalRefinement: input cloud not set");
        if (!_tree)  throw std::runtime_error("NormalRefinement: search method not set");
        if (!_cloud->hasNormals()) throw std::runtime_error("NormalRefinement: cloud has no normals");

        auto* normals = _cloud->normals();
        int n = static_cast<int>(_cloud->size());

        std::vector<plamatrix::Vec3<Scalar>> temp(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            temp[static_cast<std::size_t>(i)] = {normals->getValue(i,0), normals->getValue(i,1), normals->getValue(i,2)};

        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(i);
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
                normals->setValue(i, 0, sx / len);
                normals->setValue(i, 1, sy / len);
                normals->setValue(i, 2, sz / len);
            }
        }
    }

    void orientConsistently(const plamatrix::Vec3<Scalar>& viewpoint)
    {
        if (!_cloud || !_cloud->hasNormals()) return;
        auto* normals = _cloud->normals();
        int n = static_cast<int>(_cloud->size());
        for (int i = 0; i < n; ++i)
        {
            plamatrix::Vec3<Scalar> pt = pointVec(i);
            Scalar dx = viewpoint.x - pt.x, dy = viewpoint.y - pt.y, dz = viewpoint.z - pt.z;
            Scalar nx = normals->getValue(i, 0), ny = normals->getValue(i, 1), nz = normals->getValue(i, 2);
            if (dx * nx + dy * ny + dz * nz < 0)
            {
                normals->setValue(i, 0, -nx);
                normals->setValue(i, 1, -ny);
                normals->setValue(i, 2, -nz);
            }
        }
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return _cloud->points()(idx, dim);
        else
            return _cloud->points().getValue(idx, dim);
    }
    plamatrix::Vec3<Scalar> pointVec(int idx) const
        { return {pointCoord(idx,0), pointCoord(idx,1), pointCoord(idx,2)}; }

    std::shared_ptr<PointCloudType> _cloud;
    std::shared_ptr<search::KdTree<Scalar, Dev>> _tree;
};

} // namespace plapoint
