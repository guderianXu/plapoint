#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <plapoint/core/cloud_algorithm.h>
#include <plapoint/core/point_cloud_blob.h>
#include <plapoint/correspondence.h>
#include <plapoint/search/kdtree.h>

namespace plapoint::registration
{

    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class CorrespondenceEstimationBase : public PCLBase<PointSource>
    {
    public:
        using Ptr = std::shared_ptr<CorrespondenceEstimationBase<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const CorrespondenceEstimationBase<PointSource, PointTarget, Scalar>>;
        using PointCloudSource = plapoint::PointCloud<PointSource>;
        using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
        using PointCloudTarget = plapoint::PointCloud<PointTarget>;
        using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
        using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;
        using KdTree = search::KdTree<PointTarget>;
        using KdTreePtr = typename KdTree::Ptr;
        using KdTreeConstPtr = typename KdTree::ConstPtr;
        using KdTreeReciprocal = search::KdTree<PointSource>;
        using KdTreeReciprocalPtr = typename KdTreeReciprocal::Ptr;
        using KdTreeReciprocalConstPtr = typename KdTreeReciprocal::ConstPtr;
        using PointRepresentationConstPtr = typename KdTree::PointRepresentationConstPtr;
        using PointRepresentationReciprocalConstPtr = typename KdTreeReciprocal::PointRepresentationConstPtr;

        CorrespondenceEstimationBase()
            : _tree(std::make_shared<KdTree>()), _treeReciprocal(std::make_shared<KdTreeReciprocal>())
        {
        }

        ~CorrespondenceEstimationBase() override = default;

        void setInputSource(const PointCloudSourceConstPtr& cloud)
        {
            _sourceCloudUpdated = true;
            PCLBase<PointSource>::setInputCloud(cloud);
        }

        PointCloudSourceConstPtr getInputSource()
        {
            return this->input_;
        }

        void setInputTarget(const PointCloudTargetConstPtr& cloud)
        {
            _target = cloud;
            _targetCloudUpdated = true;
        }

        PointCloudTargetConstPtr getInputTarget()
        {
            return _target;
        }

        void setNumberOfThreads(unsigned int threads)
        {
            _threads = threads;
        }

        virtual bool requiresSourceNormals() const
        {
            return false;
        }
        virtual void setSourceNormals(PCLPointCloud2::ConstPtr)
        {
        }
        virtual bool requiresTargetNormals() const
        {
            return false;
        }
        virtual void setTargetNormals(PCLPointCloud2::ConstPtr)
        {
        }

        void setIndicesSource(const IndicesPtr& indices)
        {
            this->setIndices(indices);
        }
        IndicesPtr getIndicesSource()
        {
            return this->indices_;
        }

        void setIndicesTarget(const IndicesPtr& indices)
        {
            _targetIndices = indices;
            _targetCloudUpdated = true;
        }

        IndicesPtr getIndicesTarget()
        {
            return _targetIndices;
        }

        void setSearchMethodTarget(const KdTreePtr& tree, bool force_no_recompute = false)
        {
            if (!tree)
            {
                throw std::invalid_argument("CorrespondenceEstimation: target search tree must not be null");
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
                throw std::invalid_argument("CorrespondenceEstimation: source search tree must not be null");
            }
            _treeReciprocal = tree;
            _forceNoRecomputeReciprocal = force_no_recompute;
            _sourceCloudUpdated = true;
        }

        KdTreeReciprocalPtr getSearchMethodSource() const
        {
            return _treeReciprocal;
        }

        virtual void determineCorrespondences(Correspondences& correspondences,
                                              double max_distance = std::numeric_limits<double>::max()) = 0;
        virtual void determineReciprocalCorrespondences(Correspondences& correspondences,
                                                        double max_distance = std::numeric_limits<double>::max()) = 0;

        void setPointRepresentation(const PointRepresentationConstPtr& representation)
        {
            _pointRepresentation = representation;
            _targetCloudUpdated = true;
        }

        void setPointRepresentationReciprocal(const PointRepresentationReciprocalConstPtr& representation)
        {
            _pointRepresentationReciprocal = representation;
            _sourceCloudUpdated = true;
        }

        virtual Ptr clone() const = 0;

    protected:
        bool initCompute()
        {
            if (!this->PCLBase<PointSource>::initCompute() || !_target)
            {
                return false;
            }
            if (_pointRepresentation)
            {
                _tree->setPointRepresentation(_pointRepresentation);
            }
            if (_targetCloudUpdated && !_forceNoRecompute)
            {
                if (!_tree->setInputCloud(_target, _targetIndices))
                {
                    return false;
                }
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
            if (_pointRepresentationReciprocal)
            {
                _treeReciprocal->setPointRepresentation(_pointRepresentationReciprocal);
            }
            if (_sourceCloudUpdated && !_forceNoRecomputeReciprocal)
            {
                if (!_treeReciprocal->setInputCloud(this->input_, this->indices_))
                {
                    return false;
                }
                _sourceCloudUpdated = false;
            }
            return true;
        }

        std::string _correspondenceName = "CorrespondenceEstimationBase";
        KdTreePtr _tree;
        KdTreeReciprocalPtr _treeReciprocal;
        PointCloudTargetConstPtr _target;
        IndicesPtr _targetIndices;
        PointRepresentationConstPtr _pointRepresentation;
        PointRepresentationReciprocalConstPtr _pointRepresentationReciprocal;
        PointCloudTargetPtr _inputTransformed;
        bool _targetCloudUpdated = true;
        bool _sourceCloudUpdated = true;
        bool _forceNoRecompute = false;
        bool _forceNoRecomputeReciprocal = false;
        unsigned int _threads = 1;
    };

    template <typename PointSource, typename PointTarget, typename Scalar = float>
    class CorrespondenceEstimation : public CorrespondenceEstimationBase<PointSource, PointTarget, Scalar>
    {
    public:
        using Base = CorrespondenceEstimationBase<PointSource, PointTarget, Scalar>;
        using Ptr = std::shared_ptr<CorrespondenceEstimation<PointSource, PointTarget, Scalar>>;
        using ConstPtr = std::shared_ptr<const CorrespondenceEstimation<PointSource, PointTarget, Scalar>>;

        CorrespondenceEstimation()
        {
            this->_correspondenceName = "CorrespondenceEstimation";
        }

        void determineCorrespondences(Correspondences& correspondences,
                                      double max_distance = std::numeric_limits<double>::max()) override
        {
            correspondences.clear();
            if (!this->initCompute())
            {
                return;
            }
            const double squared_limit = squaredLimit(max_distance);
            correspondences.reserve(this->indices_->size());
            for (const int source_index : *this->indices_)
            {
                Indices matches;
                std::vector<float> distances;
                const auto& source_point = this->input_->points.at(static_cast<std::size_t>(source_index));
                if (this->_tree->nearestKSearchT(source_point, 1, matches, distances) == 1 &&
                    static_cast<double>(distances[0]) <= squared_limit)
                {
                    correspondences.emplace_back(source_index, matches[0], distances[0]);
                }
            }
            this->deinitCompute();
        }

        void determineReciprocalCorrespondences(Correspondences& correspondences,
                                                double max_distance = std::numeric_limits<double>::max()) override
        {
            correspondences.clear();
            if (!this->initComputeReciprocal())
            {
                return;
            }
            const double squared_limit = squaredLimit(max_distance);
            correspondences.reserve(this->indices_->size());
            for (const int source_index : *this->indices_)
            {
                Indices targets;
                std::vector<float> target_distances;
                const auto& source_point = this->input_->points.at(static_cast<std::size_t>(source_index));
                if (this->_tree->nearestKSearchT(source_point, 1, targets, target_distances) != 1 ||
                    static_cast<double>(target_distances[0]) > squared_limit)
                {
                    continue;
                }
                Indices sources;
                std::vector<float> source_distances;
                const auto& target_point = this->_target->points.at(static_cast<std::size_t>(targets[0]));
                if (this->_treeReciprocal->nearestKSearchT(target_point, 1, sources, source_distances) == 1 &&
                    sources[0] == source_index)
                {
                    correspondences.emplace_back(source_index, targets[0], target_distances[0]);
                }
            }
            this->deinitCompute();
        }

        typename Base::Ptr clone() const override
        {
            return std::make_shared<CorrespondenceEstimation>(*this);
        }

    private:
        static double squaredLimit(double distance)
        {
            if (std::isnan(distance) || distance < 0.0)
            {
                throw std::invalid_argument("CorrespondenceEstimation: maximum distance must be non-negative");
            }
            return distance >= std::sqrt(std::numeric_limits<double>::max()) ? std::numeric_limits<double>::max()
                                                                             : distance * distance;
        }
    };

} // namespace plapoint::registration
