#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <plapoint/core/cloud_algorithm.h>
#include <plapoint/registration/correspondence_estimation.h>
#include <plapoint/registration/correspondence_rejection.h>
#include <plapoint/registration/transformation_estimation.h>

namespace plapoint
{

    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class Registration : public PCLBase<PointSource>
    {
    public:
        using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
        using Ptr = std::shared_ptr<Registration<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const Registration<PointSource, PointTarget, Scalar>>;
        using CorrespondenceRejectorPtr = registration::CorrespondenceRejector::Ptr;
        using KdTree = search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;
        using KdTreeReciprocal = search::KdTree<PointSource>;
        using KdTreeReciprocalPtr = typename KdTreeReciprocal::Ptr;
        using PointCloudSource = plapoint::PointCloud<PointSource>;
        using PointCloudSourcePtr = typename PointCloudSource::Ptr;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
        using PointCloudTarget = plapoint::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;
        using PointRepresentationConstPtr = typename KdTree::PointRepresentationConstPtr;
        using TransformationEstimation = registration::TransformationEstimation<PointSource, PointTarget, Scalar>;
        using TransformationEstimationPtr = typename TransformationEstimation::Ptr;
        using TransformationEstimationConstPtr = typename TransformationEstimation::ConstPtr;
        using CorrespondenceEstimation = registration::CorrespondenceEstimationBase<PointSource, PointTarget, Scalar>;
        using CorrespondenceEstimationPtr = typename CorrespondenceEstimation::Ptr;
        using CorrespondenceEstimationConstPtr = typename CorrespondenceEstimation::ConstPtr;
        using UpdateVisualizerCallbackSignature = void(const PointCloudSource&,
                                                       const Indices&,
                                                       const PointCloudTarget&,
                                                       const Indices&);

        Registration()
            : _tree(std::make_shared<KdTree>()), _treeReciprocal(std::make_shared<KdTreeReciprocal>()),
              _correspondences(std::make_shared<Correspondences>())
        {
        }

        ~Registration() override = default;

        void setTransformationEstimation(const TransformationEstimationPtr& estimation)
        {
            if (!estimation)
            {
                throw std::invalid_argument("Registration: transformation estimation must not be null");
            }
            _transformationEstimation = estimation;
        }

        void setCorrespondenceEstimation(const CorrespondenceEstimationPtr& estimation)
        {
            if (!estimation)
            {
                throw std::invalid_argument("Registration: correspondence estimation must not be null");
            }
            _correspondenceEstimation = estimation;
        }

        virtual void setInputSource(const PointCloudSourceConstPtr& cloud)
        {
            this->setInputCloud(cloud);
            _sourceCloudUpdated = true;
        }

        PointCloudSourceConstPtr getInputSource()
        {
            return this->input_;
        }

        virtual void setInputTarget(const PointCloudTargetConstPtr& cloud)
        {
            _target = cloud;
            _targetCloudUpdated = true;
        }

        PointCloudTargetConstPtr getInputTarget()
        {
            return _target;
        }

        void setSearchMethodTarget(const KdTreePtr& tree, bool force_no_recompute = false)
        {
            if (!tree)
            {
                throw std::invalid_argument("Registration: target search tree must not be null");
            }
            _tree = tree;
            _forceNoRecompute = force_no_recompute;
            _targetCloudUpdated = true;
        }

        KdTreePtr getSearchMethodTarget() const
        {
            return _tree;
        }

        void setSearchMethodSource(const KdTreeReciprocalPtr& tree, bool force_no_recompute = false)
        {
            if (!tree)
            {
                throw std::invalid_argument("Registration: source search tree must not be null");
            }
            _treeReciprocal = tree;
            _forceNoRecomputeReciprocal = force_no_recompute;
            _sourceCloudUpdated = true;
        }

        KdTreeReciprocalPtr getSearchMethodSource() const
        {
            return _treeReciprocal;
        }

        Matrix4 getFinalTransformation()
        {
            return _finalTransformation;
        }
        Matrix4 getLastIncrementalTransformation()
        {
            return _transformation;
        }

        void setMaximumIterations(int iterations)
        {
            _maximumIterations = iterations;
        }
        int getMaximumIterations()
        {
            return _maximumIterations;
        }
        int getMaximumIterations() const
        {
            return _maximumIterations;
        }

        void setRANSACIterations(int iterations)
        {
            _ransacIterations = iterations;
        }
        double getRANSACIterations()
        {
            return _ransacIterations;
        }
        double getRANSACIterations() const
        {
            return _ransacIterations;
        }

        void setRANSACOutlierRejectionThreshold(double threshold)
        {
            _inlierThreshold = threshold;
        }
        double getRANSACOutlierRejectionThreshold()
        {
            return _inlierThreshold;
        }
        double getRANSACOutlierRejectionThreshold() const
        {
            return _inlierThreshold;
        }

        void setMaxCorrespondenceDistance(double threshold)
        {
            _correspondenceDistanceThreshold = threshold;
        }
        double getMaxCorrespondenceDistance()
        {
            return _correspondenceDistanceThreshold;
        }
        double getMaxCorrespondenceDistance() const
        {
            return _correspondenceDistanceThreshold;
        }

        void setTransformationEpsilon(double epsilon)
        {
            _transformationEpsilon = epsilon;
        }
        double getTransformationEpsilon()
        {
            return _transformationEpsilon;
        }
        double getTransformationEpsilon() const
        {
            return _transformationEpsilon;
        }

        void setTransformationRotationEpsilon(double epsilon)
        {
            _transformationRotationEpsilon = epsilon;
        }
        double getTransformationRotationEpsilon()
        {
            return _transformationRotationEpsilon;
        }
        double getTransformationRotationEpsilon() const
        {
            return _transformationRotationEpsilon;
        }

        void setEuclideanFitnessEpsilon(double epsilon)
        {
            _euclideanFitnessEpsilon = epsilon;
        }
        double getEuclideanFitnessEpsilon()
        {
            return _euclideanFitnessEpsilon;
        }
        double getEuclideanFitnessEpsilon() const
        {
            return _euclideanFitnessEpsilon;
        }

        void setPointRepresentation(const PointRepresentationConstPtr& representation)
        {
            _pointRepresentation = representation;
            _targetCloudUpdated = true;
        }

        bool registerVisualizationCallback(std::function<UpdateVisualizerCallbackSignature>& callback)
        {
            if (!callback)
            {
                return false;
            }
            _updateVisualizer = callback;
            if (this->input_ && _target)
            {
                Indices indices;
                _updateVisualizer(*this->input_, indices, *_target, indices);
            }
            return true;
        }

        double getFitnessScore(double max_range = std::numeric_limits<double>::max(), bool use_indices = false)
        {
            if (!this->input_ || !_target)
            {
                return std::numeric_limits<double>::max();
            }
            if (_targetCloudUpdated && !_forceNoRecompute)
            {
                _tree->setInputCloud(_target);
                _targetCloudUpdated = false;
            }
            const bool use_subset = use_indices && static_cast<bool>(this->indices_);
            const std::size_t count = use_subset ? this->indices_->size() : this->input_->size();
            double score = 0.0;
            std::size_t accepted = 0;
            for (std::size_t position = 0; position < count; ++position)
            {
                const std::size_t source_index =
                    use_subset ? static_cast<std::size_t>(this->indices_->at(position)) : position;
                PointTarget transformed{};
                transformPoint(this->input_->points.at(source_index), _finalTransformation, transformed);
                Indices target_indices;
                std::vector<float> distances;
                if (_tree->nearestKSearch(transformed, 1, target_indices, distances) == 1 &&
                    static_cast<double>(distances[0]) <= max_range)
                {
                    score += distances[0];
                    ++accepted;
                }
            }
            return accepted == 0 ? std::numeric_limits<double>::max() : score / static_cast<double>(accepted);
        }

        double getFitnessScore(const std::vector<float>& distances_a, const std::vector<float>& distances_b)
        {
            const std::size_t count = std::min(distances_a.size(), distances_b.size());
            if (count == 0)
            {
                return std::numeric_limits<double>::max();
            }
            double score = 0.0;
            for (std::size_t index = 0; index < count; ++index)
            {
                score += static_cast<double>(distances_a[index] - distances_b[index]);
            }
            return score / static_cast<double>(count);
        }

        bool hasConverged() const
        {
            return _converged;
        }

        void align(PointCloudSource& output)
        {
            align(output, Matrix4::Identity());
        }

        void align(PointCloudSource& output, const Matrix4& guess)
        {
            if (!initCompute())
            {
                throw std::runtime_error("Registration: source and target clouds must be set and non-empty");
            }
            output.header = this->input_->header;
            output.sensor_origin_ = this->input_->sensor_origin_;
            output.sensor_orientation_ = this->input_->sensor_orientation_;
            output.is_dense = this->input_->is_dense;
            output.clear();
            output.reserve(this->indices_->size());
            for (const int index : *this->indices_)
            {
                if (index < 0 || static_cast<std::size_t>(index) >= this->input_->size())
                {
                    throw std::out_of_range("Registration: source index is outside the input cloud");
                }
                output.push_back(this->input_->points[static_cast<std::size_t>(index)]);
            }
            _converged = false;
            computeTransformation(output, guess);
            this->deinitCompute();
        }

        const std::string& getClassName() const
        {
            return _registrationName;
        }

        void addCorrespondenceRejector(const CorrespondenceRejectorPtr& rejector)
        {
            if (rejector)
            {
                _correspondenceRejectors.push_back(rejector);
            }
        }

        std::vector<CorrespondenceRejectorPtr> getCorrespondenceRejectors()
        {
            return _correspondenceRejectors;
        }

        bool removeCorrespondenceRejector(unsigned int index)
        {
            if (index >= _correspondenceRejectors.size())
            {
                return false;
            }
            _correspondenceRejectors.erase(_correspondenceRejectors.begin() + index);
            return true;
        }

        void clearCorrespondenceRejectors()
        {
            _correspondenceRejectors.clear();
        }

    protected:
        bool initCompute()
        {
            if (!this->PCLBase<PointSource>::initCompute() || !_target || this->input_->empty() || _target->empty())
            {
                return false;
            }
            if (_pointRepresentation)
            {
                _tree->setPointRepresentation(_pointRepresentation);
            }
            if (_targetCloudUpdated && !_forceNoRecompute)
            {
                _tree->setInputCloud(_target);
                _targetCloudUpdated = false;
            }
            return true;
        }

        bool initComputeReciprocal()
        {
            if (!initCompute())
            {
                return false;
            }
            if (_sourceCloudUpdated && !_forceNoRecomputeReciprocal)
            {
                _treeReciprocal->setInputCloud(this->input_, this->indices_);
                _sourceCloudUpdated = false;
            }
            return true;
        }

        virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) = 0;

        template <typename SourcePoint, typename DestinationPoint>
        static void
        transformPoint(const SourcePoint& source, const Matrix4& transformation, DestinationPoint& destination)
        {
            const Scalar x = static_cast<Scalar>(source.x);
            const Scalar y = static_cast<Scalar>(source.y);
            const Scalar z = static_cast<Scalar>(source.z);
            destination.x = static_cast<decltype(destination.x)>(transformation(0, 0) * x + transformation(0, 1) * y +
                                                                 transformation(0, 2) * z + transformation(0, 3));
            destination.y = static_cast<decltype(destination.y)>(transformation(1, 0) * x + transformation(1, 1) * y +
                                                                 transformation(1, 2) * z + transformation(1, 3));
            destination.z = static_cast<decltype(destination.z)>(transformation(2, 0) * x + transformation(2, 1) * y +
                                                                 transformation(2, 2) * z + transformation(2, 3));
        }

        std::string _registrationName = "Registration";
        KdTreePtr _tree;
        KdTreeReciprocalPtr _treeReciprocal;
        int _iterations = 0;
        int _maximumIterations = 10;
        int _ransacIterations = 0;
        PointCloudTargetConstPtr _target;
        Matrix4 _finalTransformation = Matrix4::Identity();
        Matrix4 _transformation = Matrix4::Identity();
        Matrix4 _previousTransformation = Matrix4::Identity();
        double _transformationEpsilon = 0.0;
        double _transformationRotationEpsilon = 0.0;
        double _euclideanFitnessEpsilon = -std::numeric_limits<double>::max();
        bool _converged = false;
        double _correspondenceDistanceThreshold = std::sqrt(std::numeric_limits<double>::max());
        double _inlierThreshold = 0.05;
        CorrespondencesPtr _correspondences;
        TransformationEstimationPtr _transformationEstimation;
        CorrespondenceEstimationPtr _correspondenceEstimation;
        std::vector<CorrespondenceRejectorPtr> _correspondenceRejectors;
        PointRepresentationConstPtr _pointRepresentation;
        std::function<UpdateVisualizerCallbackSignature> _updateVisualizer;
        bool _targetCloudUpdated = true;
        bool _sourceCloudUpdated = true;
        bool _forceNoRecompute = false;
        bool _forceNoRecomputeReciprocal = false;
        int _minimumCorrespondences = 3;
    };

} // namespace plapoint
