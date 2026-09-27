#pragma once

#include <memory>
#include <string>

#include <plapoint/core/point_cloud_blob.h>
#include <plapoint/correspondence.h>

namespace plapoint::registration
{

class CorrespondenceRejector
{
public:
    using Ptr = std::shared_ptr<CorrespondenceRejector>;
    using ConstPtr = std::shared_ptr<const CorrespondenceRejector>;

    virtual ~CorrespondenceRejector() = default;

    virtual void setInputCorrespondences(const CorrespondencesConstPtr& correspondences)
    {
        input_correspondences_ = correspondences;
    }

    CorrespondencesConstPtr getInputCorrespondences()
    {
        return input_correspondences_;
    }

    void getCorrespondences(Correspondences& correspondences)
    {
        if (input_correspondences_ && !input_correspondences_->empty())
        {
            applyRejection(correspondences);
        }
    }

    virtual void getRemainingCorrespondences(const Correspondences& original_correspondences,
                                             Correspondences& remaining_correspondences) = 0;

    void getRejectedQueryIndices(const Correspondences& correspondences, Indices& indices)
    {
        if (input_correspondences_)
        {
            plapoint::getRejectedQueryIndices(*input_correspondences_, correspondences, indices);
        }
    }

    const std::string& getClassName() const
    {
        return rejection_name_;
    }

    virtual bool requiresSourcePoints() const { return false; }
    virtual void setSourcePoints(PCLPointCloud2::ConstPtr) {}
    virtual bool requiresSourceNormals() const { return false; }
    virtual void setSourceNormals(PCLPointCloud2::ConstPtr) {}
    virtual bool requiresTargetPoints() const { return false; }
    virtual void setTargetPoints(PCLPointCloud2::ConstPtr) {}
    virtual bool requiresTargetNormals() const { return false; }
    virtual void setTargetNormals(PCLPointCloud2::ConstPtr) {}

protected:
    std::string rejection_name_;
    CorrespondencesConstPtr input_correspondences_;

    virtual void applyRejection(Correspondences& correspondences) = 0;
};

class DataContainerInterface
{
public:
    using Ptr = std::shared_ptr<DataContainerInterface>;
    using ConstPtr = std::shared_ptr<const DataContainerInterface>;

    virtual ~DataContainerInterface() = default;
    virtual double getCorrespondenceScore(int index) = 0;
    virtual double getCorrespondenceScore(const Correspondence& correspondence) = 0;
    virtual double getCorrespondenceScoreFromNormals(const Correspondence& correspondence) = 0;
};

} // namespace plapoint::registration
