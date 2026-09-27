#pragma once

#include <cmath>
#include <numeric>

#include <plapoint/features/normal_estimation.h>

namespace plapoint
{

    template <typename PointT>
    bool computePointNormal(const PointCloud<PointT>& cloud, Eigen::Vector4f& plane_parameters, float& curvature)
    {
        Indices indices(cloud.size());
        std::iota(indices.begin(), indices.end(), 0);
        NormalEstimation<PointT, Normal> estimation;
        return estimation.computePointNormal(cloud, indices, plane_parameters, curvature);
    }

    template <typename PointT>
    bool computePointNormal(const PointCloud<PointT>& cloud,
                            const Indices& indices,
                            Eigen::Vector4f& plane_parameters,
                            float& curvature)
    {
        NormalEstimation<PointT, Normal> estimation;
        return estimation.computePointNormal(cloud, indices, plane_parameters, curvature);
    }

    template <typename PointT, typename Scalar>
    void flipNormalTowardsViewpoint(
        const PointT& point, float view_x, float view_y, float view_z, Eigen::Matrix<Scalar, 4, 1>& normal)
    {
        const Eigen::Matrix<Scalar, 4, 1> direction(static_cast<Scalar>(view_x - point.x),
                                                    static_cast<Scalar>(view_y - point.y),
                                                    static_cast<Scalar>(view_z - point.z),
                                                    Scalar(0));
        if (direction.dot(normal) < Scalar(0))
        {
            normal *= Scalar(-1);
            normal[3] = -normal.template head<3>().dot(point.getVector3fMap().template cast<Scalar>());
        }
    }

    template <typename PointT, typename Scalar>
    void flipNormalTowardsViewpoint(
        const PointT& point, float view_x, float view_y, float view_z, Eigen::Matrix<Scalar, 3, 1>& normal)
    {
        const Eigen::Matrix<Scalar, 3, 1> direction(static_cast<Scalar>(view_x - point.x),
                                                    static_cast<Scalar>(view_y - point.y),
                                                    static_cast<Scalar>(view_z - point.z));
        if (direction.dot(normal) < Scalar(0))
        {
            normal *= Scalar(-1);
        }
    }

    template <typename PointT>
    void flipNormalTowardsViewpoint(const PointT& point,
                                    float view_x,
                                    float view_y,
                                    float view_z,
                                    float& normal_x,
                                    float& normal_y,
                                    float& normal_z)
    {
        view_x -= point.x;
        view_y -= point.y;
        view_z -= point.z;
        if (view_x * normal_x + view_y * normal_y + view_z * normal_z < 0.0f)
        {
            normal_x = -normal_x;
            normal_y = -normal_y;
            normal_z = -normal_z;
        }
    }

    template <typename PointNT>
    bool flipNormalTowardsNormalsMean(const PointCloud<PointNT>& normal_cloud,
                                      const Indices& normal_indices,
                                      Eigen::Vector3f& normal)
    {
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (const index_t index : normal_indices)
        {
            if (index < 0 || static_cast<std::size_t>(index) >= normal_cloud.size())
            {
                continue;
            }
            const auto& value = normal_cloud.points[static_cast<std::size_t>(index)];
            if (std::isfinite(value.normal_x) && std::isfinite(value.normal_y) && std::isfinite(value.normal_z))
            {
                mean += value.getNormalVector3fMap();
            }
        }
        if (mean.isZero())
        {
            return false;
        }
        mean.normalize();
        if (normal.dot(mean) < 0.0f)
        {
            normal = -normal;
        }
        return true;
    }

} // namespace plapoint
