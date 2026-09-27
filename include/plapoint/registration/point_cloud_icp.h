#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>

#include <plapoint/core/point_cloud_blob.h>
#include <plapoint/registration/correspondence_estimation.h>
#include <plapoint/registration/default_convergence_criteria.h>
#include <plapoint/registration/registration.h>
#include <plapoint/registration/transformation_estimation_svd.h>

namespace plapoint
{

    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class IterativeClosestPoint : public Registration<PointSource, PointTarget, Scalar>
    {
        static_assert(std::is_floating_point_v<Scalar>, "ICP scalar must be floating point");

    public:
        using Base = Registration<PointSource, PointTarget, Scalar>;
        using PointCloudSource = typename Base::PointCloudSource;
        using PointCloudSourcePtr = typename Base::PointCloudSourcePtr;
        using PointCloudSourceConstPtr = typename Base::PointCloudSourceConstPtr;
        using PointCloudTarget = typename Base::PointCloudTarget;
        using PointCloudTargetPtr = typename Base::PointCloudTargetPtr;
        using PointCloudTargetConstPtr = typename Base::PointCloudTargetConstPtr;
        using PointIndicesPtr = PointIndices::Ptr;
        using PointIndicesConstPtr = PointIndices::ConstPtr;
        using Ptr = std::shared_ptr<IterativeClosestPoint<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const IterativeClosestPoint<PointSource, PointTarget, Scalar>>;
        using Matrix4 = typename Base::Matrix4;
        using ConvergenceCriteria = registration::DefaultConvergenceCriteria<Scalar>;
        using ConvergenceCriteriaPtr = typename ConvergenceCriteria::Ptr;

        IterativeClosestPoint()
        {
            this->_registrationName = "IterativeClosestPoint";
            this->_transformationEstimation =
                std::make_shared<registration::TransformationEstimationSVD<PointSource, PointTarget, Scalar>>();
            this->_correspondenceEstimation =
                std::make_shared<registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>>();
            _convergenceCriteria = std::make_shared<ConvergenceCriteria>(
                this->_iterations, this->_transformation, *this->_correspondences);
        }

        IterativeClosestPoint(const IterativeClosestPoint&) = delete;
        IterativeClosestPoint(IterativeClosestPoint&&) = delete;
        IterativeClosestPoint& operator=(const IterativeClosestPoint&) = delete;
        IterativeClosestPoint& operator=(IterativeClosestPoint&&) = delete;
        ~IterativeClosestPoint() override = default;

        void setUseReciprocalCorrespondences(bool use_reciprocal)
        {
            _useReciprocalCorrespondences = use_reciprocal;
        }

        bool getUseReciprocalCorrespondences() const
        {
            return _useReciprocalCorrespondences;
        }

        ConvergenceCriteriaPtr getConvergeCriteria()
        {
            return _convergenceCriteria;
        }

        void setNumberOfThreads(unsigned int threads)
        {
            this->_correspondenceEstimation->setNumberOfThreads(threads);
        }

        Scalar getInlierFraction() const noexcept
        {
            return _inlierFraction;
        }
        Scalar getFinalRmse() const noexcept
        {
            return _finalRmse;
        }

    protected:
        void computeTransformation(PointCloudSource& output, const Matrix4& guess) override
        {
            if (this->_maximumIterations < 0 || this->_ransacIterations < 0 ||
                std::isnan(this->_correspondenceDistanceThreshold) || this->_correspondenceDistanceThreshold < 0.0)
            {
                throw std::invalid_argument("ICP: iteration counts and correspondence distance must be non-negative");
            }

            transformCloud(output, guess);
            this->_finalTransformation = guess;
            this->_transformation.setIdentity();
            this->_previousTransformation.setIdentity();
            this->_iterations = 0;
            this->_correspondences->clear();
            _inlierFraction = Scalar{};
            _finalRmse = std::numeric_limits<Scalar>::infinity();
            _convergenceCriteria->reset();
            _convergenceCriteria->setMaximumIterations(this->_maximumIterations);
            _convergenceCriteria->setRelativeMSE(this->_euclideanFitnessEpsilon);
            _convergenceCriteria->setTranslationThreshold(this->_transformationEpsilon);
            if (this->_transformationRotationEpsilon > 0.0)
            {
                _convergenceCriteria->setRotationThreshold(this->_transformationRotationEpsilon);
            }

            while (this->_iterations < this->_maximumIterations)
            {
                auto current_source = std::make_shared<PointCloudSource>(output);
                configureCorrespondenceEstimation(current_source);
                if (_useReciprocalCorrespondences)
                {
                    this->_correspondenceEstimation->determineReciprocalCorrespondences(
                        *this->_correspondences, this->_correspondenceDistanceThreshold);
                }
                else
                {
                    this->_correspondenceEstimation->determineCorrespondences(*this->_correspondences,
                                                                              this->_correspondenceDistanceThreshold);
                }

                applyRansac(output, *this->_target, *this->_correspondences);
                applyRejectors(output, *this->_target, *this->_correspondences);
                if (this->_correspondences->size() < static_cast<std::size_t>(this->_minimumCorrespondences))
                {
                    _convergenceCriteria->setConvergenceState(
                        ConvergenceCriteria::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES);
                    break;
                }

                this->_previousTransformation = this->_transformation;
                this->_transformationEstimation->estimateRigidTransformation(
                    output, *this->_target, *this->_correspondences, this->_transformation);
                transformCloud(output, this->_transformation);
                this->_finalTransformation = this->_transformation * this->_finalTransformation;

                const double mse = correspondenceMse(*this->_correspondences);
                _inlierFraction = static_cast<Scalar>(static_cast<double>(this->_correspondences->size()) /
                                                      static_cast<double>(output.size()));
                _finalRmse = static_cast<Scalar>(std::sqrt(mse));

                if (this->_updateVisualizer)
                {
                    Indices source_indices;
                    Indices target_indices;
                    source_indices.reserve(this->_correspondences->size());
                    target_indices.reserve(this->_correspondences->size());
                    for (const auto& correspondence : *this->_correspondences)
                    {
                        source_indices.push_back(correspondence.index_query);
                        target_indices.push_back(correspondence.index_match);
                    }
                    this->_updateVisualizer(output, source_indices, *this->_target, target_indices);
                }

                ++this->_iterations;
                if (_convergenceCriteria->hasConverged())
                {
                    this->_converged = true;
                    break;
                }
            }

            if (!this->_converged && this->_iterations >= this->_maximumIterations && this->_maximumIterations > 0 &&
                !_convergenceCriteria->getFailureAfterMaximumIterations())
            {
                this->_converged = true;
                _convergenceCriteria->setConvergenceState(ConvergenceCriteria::CONVERGENCE_CRITERIA_ITERATIONS);
            }
            recomputeFinalMetrics(output);
        }

    private:
        void configureCorrespondenceEstimation(const PointCloudSourceConstPtr& source)
        {
            this->_correspondenceEstimation->setInputSource(source);
            this->_correspondenceEstimation->setInputTarget(this->_target);
            this->_correspondenceEstimation->setSearchMethodTarget(this->_tree, this->_forceNoRecompute);
            this->_correspondenceEstimation->setSearchMethodSource(this->_treeReciprocal,
                                                                   this->_forceNoRecomputeReciprocal);
            if (this->_pointRepresentation)
            {
                this->_correspondenceEstimation->setPointRepresentation(this->_pointRepresentation);
            }
        }

        static double correspondenceMse(const Correspondences& correspondences)
        {
            if (correspondences.empty())
            {
                return std::numeric_limits<double>::infinity();
            }
            double sum = 0.0;
            for (const auto& correspondence : correspondences)
            {
                sum += correspondence.distance;
            }
            return sum / static_cast<double>(correspondences.size());
        }

        void
        applyRejectors(const PointCloudSource& source, const PointCloudTarget& target, Correspondences& correspondences)
        {
            if (this->_correspondenceRejectors.empty() || correspondences.empty())
            {
                return;
            }
            PCLPointCloud2::Ptr source_blob;
            PCLPointCloud2::Ptr target_blob;
            for (const auto& rejector : this->_correspondenceRejectors)
            {
                if (rejector->requiresSourcePoints() || rejector->requiresSourceNormals())
                {
                    if (!source_blob)
                    {
                        source_blob = std::make_shared<PCLPointCloud2>();
                        toPCLPointCloud2(source, *source_blob);
                    }
                    if (rejector->requiresSourcePoints())
                    {
                        rejector->setSourcePoints(source_blob);
                    }
                    if (rejector->requiresSourceNormals())
                    {
                        rejector->setSourceNormals(source_blob);
                    }
                }
                if (rejector->requiresTargetPoints() || rejector->requiresTargetNormals())
                {
                    if (!target_blob)
                    {
                        target_blob = std::make_shared<PCLPointCloud2>();
                        toPCLPointCloud2(target, *target_blob);
                    }
                    if (rejector->requiresTargetPoints())
                    {
                        rejector->setTargetPoints(target_blob);
                    }
                    if (rejector->requiresTargetNormals())
                    {
                        rejector->setTargetNormals(target_blob);
                    }
                }
                rejector->setInputCorrespondences(std::make_shared<const Correspondences>(correspondences));
                Correspondences filtered;
                rejector->getCorrespondences(filtered);
                correspondences.swap(filtered);
            }
        }

        void
        applyRansac(const PointCloudSource& source, const PointCloudTarget& target, Correspondences& correspondences)
        {
            if (this->_ransacIterations <= 0 || correspondences.size() < 3 || this->_inlierThreshold < 0.0)
            {
                return;
            }
            std::mt19937 generator(0x504c4150u);
            std::uniform_int_distribution<std::size_t> distribution(0, correspondences.size() - 1);
            Correspondences best;
            const double squared_threshold = this->_inlierThreshold * this->_inlierThreshold;
            for (int iteration = 0; iteration < this->_ransacIterations; ++iteration)
            {
                std::size_t first = distribution(generator);
                std::size_t second = distribution(generator);
                std::size_t third = distribution(generator);
                for (int attempt = 0; attempt < 12 && (first == second || first == third || second == third); ++attempt)
                {
                    second = distribution(generator);
                    third = distribution(generator);
                }
                if (first == second || first == third || second == third)
                {
                    continue;
                }
                Correspondences sample{correspondences[first], correspondences[second], correspondences[third]};
                Matrix4 candidate;
                try
                {
                    this->_transformationEstimation->estimateRigidTransformation(source, target, sample, candidate);
                }
                catch (const std::exception&)
                {
                    continue;
                }
                Correspondences inliers;
                inliers.reserve(correspondences.size());
                for (auto correspondence : correspondences)
                {
                    const auto& source_point = source.points.at(static_cast<std::size_t>(correspondence.index_query));
                    const auto& target_point = target.points.at(static_cast<std::size_t>(correspondence.index_match));
                    PointTarget transformed{};
                    Base::transformPoint(source_point, candidate, transformed);
                    const double dx = static_cast<double>(transformed.x) - target_point.x;
                    const double dy = static_cast<double>(transformed.y) - target_point.y;
                    const double dz = static_cast<double>(transformed.z) - target_point.z;
                    const double squared_distance = dx * dx + dy * dy + dz * dz;
                    if (squared_distance <= squared_threshold)
                    {
                        correspondence.distance = static_cast<float>(squared_distance);
                        inliers.push_back(correspondence);
                    }
                }
                if (inliers.size() > best.size())
                {
                    best.swap(inliers);
                }
            }
            if (best.size() >= static_cast<std::size_t>(this->_minimumCorrespondences))
            {
                correspondences.swap(best);
            }
        }

        void recomputeFinalMetrics(const PointCloudSource& source)
        {
            auto source_ptr = std::make_shared<PointCloudSource>(source);
            configureCorrespondenceEstimation(source_ptr);
            Correspondences final_correspondences;
            if (_useReciprocalCorrespondences)
            {
                this->_correspondenceEstimation->determineReciprocalCorrespondences(
                    final_correspondences, this->_correspondenceDistanceThreshold);
            }
            else
            {
                this->_correspondenceEstimation->determineCorrespondences(final_correspondences,
                                                                          this->_correspondenceDistanceThreshold);
            }
            applyRejectors(source, *this->_target, final_correspondences);
            this->_correspondences->swap(final_correspondences);
            if (!source.empty())
            {
                _inlierFraction = static_cast<Scalar>(static_cast<double>(this->_correspondences->size()) /
                                                      static_cast<double>(source.size()));
            }
            const double mse = correspondenceMse(*this->_correspondences);
            _finalRmse =
                std::isfinite(mse) ? static_cast<Scalar>(std::sqrt(mse)) : std::numeric_limits<Scalar>::infinity();
        }

        static void transformCloud(PointCloudSource& cloud, const Matrix4& transformation)
        {
            for (auto& point : cloud.points)
            {
                PointSource transformed = point;
                Base::transformPoint(point, transformation, transformed);
                if constexpr (traits::has_field_v<PointSource, fields::normal_x> &&
                              traits::has_field_v<PointSource, fields::normal_y> &&
                              traits::has_field_v<PointSource, fields::normal_z>)
                {
                    const Scalar nx = static_cast<Scalar>(point.normal_x);
                    const Scalar ny = static_cast<Scalar>(point.normal_y);
                    const Scalar nz = static_cast<Scalar>(point.normal_z);
                    transformed.normal_x = static_cast<decltype(transformed.normal_x)>(
                        transformation(0, 0) * nx + transformation(0, 1) * ny + transformation(0, 2) * nz);
                    transformed.normal_y = static_cast<decltype(transformed.normal_y)>(
                        transformation(1, 0) * nx + transformation(1, 1) * ny + transformation(1, 2) * nz);
                    transformed.normal_z = static_cast<decltype(transformed.normal_z)>(
                        transformation(2, 0) * nx + transformation(2, 1) * ny + transformation(2, 2) * nz);
                }
                point = transformed;
            }
        }

        bool _useReciprocalCorrespondences = false;
        ConvergenceCriteriaPtr _convergenceCriteria;
        Scalar _inlierFraction = Scalar{};
        Scalar _finalRmse = std::numeric_limits<Scalar>::infinity();
    };

} // namespace plapoint
