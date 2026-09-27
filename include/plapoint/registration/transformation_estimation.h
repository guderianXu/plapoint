#pragma once

#include <memory>

#include <Eigen/Core>

#include <plapoint/core/point_cloud.h>
#include <plapoint/correspondence.h>

namespace plapoint::registration
{

    template <typename PointSource, typename PointTarget, typename Scalar = float> class TransformationEstimation
    {
    public:
        using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
        using Ptr = std::shared_ptr<TransformationEstimation<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const TransformationEstimation<PointSource, PointTarget, Scalar>>;

        virtual ~TransformationEstimation() = default;

        virtual void estimateRigidTransformation(const PointCloud<PointSource>& cloud_src,
                                                 const PointCloud<PointTarget>& cloud_tgt,
                                                 Matrix4& transformation_matrix) const = 0;
        virtual void estimateRigidTransformation(const PointCloud<PointSource>& cloud_src,
                                                 const Indices& indices_src,
                                                 const PointCloud<PointTarget>& cloud_tgt,
                                                 Matrix4& transformation_matrix) const = 0;
        virtual void estimateRigidTransformation(const PointCloud<PointSource>& cloud_src,
                                                 const Indices& indices_src,
                                                 const PointCloud<PointTarget>& cloud_tgt,
                                                 const Indices& indices_tgt,
                                                 Matrix4& transformation_matrix) const = 0;
        virtual void estimateRigidTransformation(const PointCloud<PointSource>& cloud_src,
                                                 const PointCloud<PointTarget>& cloud_tgt,
                                                 const Correspondences& correspondences,
                                                 Matrix4& transformation_matrix) const = 0;
    };

} // namespace plapoint::registration
