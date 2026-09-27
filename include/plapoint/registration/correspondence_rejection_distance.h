#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

#include <plapoint/registration/correspondence_rejection.h>
#include <plapoint/search/kdtree.h>

namespace plapoint::registration
{

    class CorrespondenceRejectorDistance : public CorrespondenceRejector
    {
    public:
        using Ptr = std::shared_ptr<CorrespondenceRejectorDistance>;
        using ConstPtr = std::shared_ptr<const CorrespondenceRejectorDistance>;

        CorrespondenceRejectorDistance()
        {
            rejection_name_ = "CorrespondenceRejectorDistance";
        }

        void getRemainingCorrespondences(const Correspondences& original_correspondences,
                                         Correspondences& remaining_correspondences) override
        {
            remaining_correspondences.clear();
            remaining_correspondences.reserve(original_correspondences.size());
            for (auto correspondence : original_correspondences)
            {
                const float squared_distance =
                    hasPointClouds() ? pointDistance(correspondence) : correspondence.distance;
                if (std::isfinite(squared_distance) && squared_distance <= max_distance_)
                {
                    correspondence.distance = squared_distance;
                    remaining_correspondences.push_back(correspondence);
                }
            }
        }

        virtual void setMaximumDistance(float distance)
        {
            max_distance_ = distance * distance;
        }

        float getMaximumDistance() const
        {
            return std::sqrt(max_distance_);
        }

        template <typename PointT> void setInputSource(const typename PointCloud<PointT>::ConstPtr& cloud)
        {
            _source = std::make_shared<PCLPointCloud2>();
            if (cloud)
            {
                toPCLPointCloud2(*cloud, *_source);
            }
        }

        template <typename PointT> void setInputTarget(const typename PointCloud<PointT>::ConstPtr& cloud)
        {
            _target = std::make_shared<PCLPointCloud2>();
            if (cloud)
            {
                toPCLPointCloud2(*cloud, *_target);
            }
        }

        bool requiresSourcePoints() const override
        {
            return true;
        }
        bool requiresTargetPoints() const override
        {
            return true;
        }

        void setSourcePoints(PCLPointCloud2::ConstPtr cloud) override
        {
            _source = std::move(cloud);
        }
        void setTargetPoints(PCLPointCloud2::ConstPtr cloud) override
        {
            _target = std::move(cloud);
        }

        template <typename PointT> void setSearchMethodTarget(const typename search::KdTree<PointT>::Ptr&, bool = false)
        {
        }

    protected:
        void applyRejection(Correspondences& correspondences) override
        {
            getRemainingCorrespondences(*input_correspondences_, correspondences);
        }

    private:
        static const PCLPointField& field(const PCLPointCloud2& cloud, const char* name)
        {
            const auto iterator = std::find_if(cloud.fields.begin(),
                                               cloud.fields.end(),
                                               [name](const auto& candidate) { return candidate.name == name; });
            if (iterator == cloud.fields.end() || iterator->count == 0)
            {
                throw std::invalid_argument(std::string("CorrespondenceRejectorDistance: missing field ") + name);
            }
            return *iterator;
        }

        static double coordinate(const PCLPointCloud2& cloud, std::size_t index, const PCLPointField& point_field)
        {
            const std::size_t offset = index * cloud.point_step + point_field.offset;
            if (offset + getFieldSize(point_field.datatype) > cloud.data.size())
            {
                throw std::out_of_range("CorrespondenceRejectorDistance: point index is outside the cloud");
            }
            if (point_field.datatype == PCLPointField::FLOAT32)
            {
                float value{};
                std::memcpy(&value, cloud.data.data() + offset, sizeof(value));
                return value;
            }
            if (point_field.datatype == PCLPointField::FLOAT64)
            {
                double value{};
                std::memcpy(&value, cloud.data.data() + offset, sizeof(value));
                return value;
            }
            throw std::invalid_argument("CorrespondenceRejectorDistance: XYZ fields must be float or double");
        }

        bool hasPointClouds() const
        {
            return _source && _target && !_source->fields.empty() && !_target->fields.empty();
        }

        float pointDistance(const Correspondence& correspondence) const
        {
            if (correspondence.index_query < 0 || correspondence.index_match < 0)
            {
                return std::numeric_limits<float>::infinity();
            }
            double squared_distance = 0.0;
            for (const char* name : {"x", "y", "z"})
            {
                const double difference =
                    coordinate(*_source, static_cast<std::size_t>(correspondence.index_query), field(*_source, name)) -
                    coordinate(*_target, static_cast<std::size_t>(correspondence.index_match), field(*_target, name));
                squared_distance += difference * difference;
            }
            return static_cast<float>(squared_distance);
        }

        float max_distance_ = std::numeric_limits<float>::max();
        PCLPointCloud2::ConstPtr _source;
        PCLPointCloud2::ConstPtr _target;
    };

} // namespace plapoint::registration
