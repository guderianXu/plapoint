#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <plamatrix/dense/matrix.h>

#include <plapoint/registration/transformation_estimation.h>

namespace plapoint::registration
{

    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class TransformationEstimationSVD : public TransformationEstimation<PointSource, PointTarget, Scalar>
    {
    public:
        using Ptr = std::shared_ptr<TransformationEstimationSVD<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const TransformationEstimationSVD<PointSource, PointTarget, Scalar>>;
        using Matrix4 = typename TransformationEstimation<PointSource, PointTarget, Scalar>::Matrix4;

        explicit TransformationEstimationSVD(bool use_umeyama = true) : _useUmeyama(use_umeyama)
        {
        }

        void estimateRigidTransformation(const PointCloud<PointSource>& source,
                                         const PointCloud<PointTarget>& target,
                                         Matrix4& transformation) const override
        {
            if (source.size() != target.size())
            {
                throw std::invalid_argument("TransformationEstimationSVD: cloud sizes differ");
            }
            Indices source_indices(source.size());
            Indices target_indices(target.size());
            for (std::size_t index = 0; index < source.size(); ++index)
            {
                source_indices[index] = checkedIndex(index);
                target_indices[index] = checkedIndex(index);
            }
            estimate(source, source_indices, target, target_indices, transformation);
        }

        void estimateRigidTransformation(const PointCloud<PointSource>& source,
                                         const Indices& source_indices,
                                         const PointCloud<PointTarget>& target,
                                         Matrix4& transformation) const override
        {
            if (source_indices.size() != target.size())
            {
                throw std::invalid_argument("TransformationEstimationSVD: source indices and target size differ");
            }
            Indices target_indices(target.size());
            for (std::size_t index = 0; index < target.size(); ++index)
            {
                target_indices[index] = checkedIndex(index);
            }
            estimate(source, source_indices, target, target_indices, transformation);
        }

        void estimateRigidTransformation(const PointCloud<PointSource>& source,
                                         const Indices& source_indices,
                                         const PointCloud<PointTarget>& target,
                                         const Indices& target_indices,
                                         Matrix4& transformation) const override
        {
            estimate(source, source_indices, target, target_indices, transformation);
        }

        void estimateRigidTransformation(const PointCloud<PointSource>& source,
                                         const PointCloud<PointTarget>& target,
                                         const Correspondences& correspondences,
                                         Matrix4& transformation) const override
        {
            Indices source_indices;
            Indices target_indices;
            source_indices.reserve(correspondences.size());
            target_indices.reserve(correspondences.size());
            for (const auto& correspondence : correspondences)
            {
                source_indices.push_back(correspondence.index_query);
                target_indices.push_back(correspondence.index_match);
            }
            estimate(source, source_indices, target, target_indices, transformation);
        }

    protected:
        void getTransformationFromCorrelation(const Eigen::Matrix<Scalar, 3, 1>& source_centroid,
                                              const Eigen::Matrix<Scalar, 3, 1>& target_centroid,
                                              const Eigen::Matrix<Scalar, 3, 3>& correlation,
                                              Matrix4& transformation) const
        {
            plamatrix::Matrix<Scalar, 3, 3> covariance;
            for (plamatrix::Index row = 0; row < 3; ++row)
            {
                for (plamatrix::Index column = 0; column < 3; ++column)
                {
                    covariance(row, column) = correlation(row, column);
                }
            }
            const auto decomposition =
                covariance.template jacobiSvd<plamatrix::ComputeThinU | plamatrix::ComputeThinV>();
            const auto left = decomposition.matrixU();
            const auto right = decomposition.matrixV();

            Scalar rotation[3][3]{};
            multiplyRotation(right, left, Scalar{1}, rotation);
            if (determinant(rotation) < Scalar{})
            {
                multiplyRotation(right, left, Scalar{-1}, rotation);
            }

            transformation.setIdentity();
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    transformation(row, column) = rotation[row][column];
                }
                transformation(row, 3) = target_centroid(row) - (rotation[row][0] * source_centroid(0) +
                                                                 rotation[row][1] * source_centroid(1) +
                                                                 rotation[row][2] * source_centroid(2));
            }
        }

    private:
        static index_t checkedIndex(std::size_t index)
        {
            if (index > static_cast<std::size_t>(std::numeric_limits<index_t>::max()))
            {
                throw std::overflow_error("TransformationEstimationSVD: point index exceeds index_t range");
            }
            return static_cast<index_t>(index);
        }

        template <typename LeftMatrix, typename RightMatrix>
        static void multiplyRotation(const LeftMatrix& right,
                                     const RightMatrix& left,
                                     Scalar last_axis_sign,
                                     Scalar (&rotation)[3][3])
        {
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    rotation[row][column] = right(row, 0) * left(column, 0) + right(row, 1) * left(column, 1) +
                                            last_axis_sign * right(row, 2) * left(column, 2);
                }
            }
        }

        static Scalar determinant(const Scalar (&matrix)[3][3])
        {
            return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                   matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                   matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
        }

        static void validatePointIndex(index_t index, std::size_t size, const char* label)
        {
            if (index < 0 || static_cast<std::size_t>(index) >= size)
            {
                throw std::out_of_range(std::string("TransformationEstimationSVD: ") + label +
                                        " index is outside the cloud");
            }
        }

        void estimate(const PointCloud<PointSource>& source,
                      const Indices& source_indices,
                      const PointCloud<PointTarget>& target,
                      const Indices& target_indices,
                      Matrix4& transformation) const
        {
            if (source_indices.size() != target_indices.size() || source_indices.size() < 3)
            {
                throw std::invalid_argument("TransformationEstimationSVD: at least three paired indices are required");
            }

            long double source_centroid[3]{};
            long double target_centroid[3]{};
            for (std::size_t index = 0; index < source_indices.size(); ++index)
            {
                validatePointIndex(source_indices[index], source.size(), "source");
                validatePointIndex(target_indices[index], target.size(), "target");
                const auto& source_point = source.points[static_cast<std::size_t>(source_indices[index])];
                const auto& target_point = target.points[static_cast<std::size_t>(target_indices[index])];
                source_centroid[0] += source_point.x;
                source_centroid[1] += source_point.y;
                source_centroid[2] += source_point.z;
                target_centroid[0] += target_point.x;
                target_centroid[1] += target_point.y;
                target_centroid[2] += target_point.z;
            }
            const long double inverse_count = 1.0L / static_cast<long double>(source_indices.size());
            Eigen::Matrix<Scalar, 3, 1> source_center;
            Eigen::Matrix<Scalar, 3, 1> target_center;
            for (int axis = 0; axis < 3; ++axis)
            {
                source_centroid[axis] *= inverse_count;
                target_centroid[axis] *= inverse_count;
                source_center(axis) = static_cast<Scalar>(source_centroid[axis]);
                target_center(axis) = static_cast<Scalar>(target_centroid[axis]);
            }

            Eigen::Matrix<Scalar, 3, 3> correlation = Eigen::Matrix<Scalar, 3, 3>::Zero();
            for (std::size_t index = 0; index < source_indices.size(); ++index)
            {
                const auto& source_point = source.points[static_cast<std::size_t>(source_indices[index])];
                const auto& target_point = target.points[static_cast<std::size_t>(target_indices[index])];
                const long double source_values[3] = {static_cast<long double>(source_point.x) - source_centroid[0],
                                                      static_cast<long double>(source_point.y) - source_centroid[1],
                                                      static_cast<long double>(source_point.z) - source_centroid[2]};
                const long double target_values[3] = {static_cast<long double>(target_point.x) - target_centroid[0],
                                                      static_cast<long double>(target_point.y) - target_centroid[1],
                                                      static_cast<long double>(target_point.z) - target_centroid[2]};
                for (int row = 0; row < 3; ++row)
                {
                    for (int column = 0; column < 3; ++column)
                    {
                        correlation(row, column) += static_cast<Scalar>(source_values[row] * target_values[column]);
                    }
                }
            }
            getTransformationFromCorrelation(source_center, target_center, correlation, transformation);
        }

        bool _useUmeyama = true;
    };

} // namespace plapoint::registration
