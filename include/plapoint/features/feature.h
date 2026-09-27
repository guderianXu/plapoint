#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <plapoint/core/cloud_algorithm.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/search/search.h>

namespace plapoint
{

/// Common point-type feature estimation interface.
template <typename PointInT, typename PointOutT>
class Feature : public PCLBase<PointInT>
{
public:
    using BaseClass = PCLBase<PointInT>;
    using Ptr = std::shared_ptr<Feature<PointInT, PointOutT>>;
    using ConstPtr = std::shared_ptr<const Feature<PointInT, PointOutT>>;
    using KdTree = search::Search<PointInT>;
    using KdTreePtr = typename KdTree::Ptr;
    using PointCloudIn = plapoint::PointCloud<PointInT>;
    using PointCloudInPtr = typename PointCloudIn::Ptr;
    using PointCloudInConstPtr = typename PointCloudIn::ConstPtr;
    using PointCloudOut = plapoint::PointCloud<PointOutT>;
    using SearchMethod = std::function<int(std::size_t, double, Indices&, std::vector<float>&)>;
    using SearchMethodSurface = std::function<int(const PointCloudIn&, std::size_t, double,
                                                  Indices&, std::vector<float>&)>;

    void setSearchSurface(const PointCloudInConstPtr& cloud)
    {
        surface_ = cloud;
        fake_surface_ = false;
    }

    PointCloudInConstPtr getSearchSurface() const { return surface_; }
    void setSearchMethod(const KdTreePtr& tree) { tree_ = tree; }
    KdTreePtr getSearchMethod() const { return tree_; }
    double getSearchParameter() const { return search_radius_ != 0.0 ? search_radius_ : static_cast<double>(k_); }
    void setKSearch(int k) { k_ = k; }
    int getKSearch() const { return k_; }
    void setRadiusSearch(double radius) { search_radius_ = radius; }
    double getRadiusSearch() const { return search_radius_; }

    void compute(PointCloudOut& output)
    {
        if (!initCompute())
        {
            output.clear();
            return;
        }
        output.header = this->input_->header;
        output.sensor_origin_ = this->input_->sensor_origin_;
        output.sensor_orientation_ = this->input_->sensor_orientation_;
        output.resize(this->indices_->size());
        if (this->indices_->size() != this->input_->size() ||
            this->input_->width * this->input_->height == 0)
        {
            output.width = static_cast<std::uint32_t>(this->indices_->size());
            output.height = output.empty() ? 0u : 1u;
        }
        else
        {
            output.width = this->input_->width;
            output.height = this->input_->height;
        }
        output.is_dense = this->input_->is_dense;
        computeFeature(output);
        deinitCompute();
    }

protected:
    bool initCompute()
    {
        if (!BaseClass::initCompute() || this->input_->empty())
        {
            return false;
        }
        if (!surface_)
        {
            surface_ = this->input_;
            fake_surface_ = true;
        }
        if (!tree_)
        {
            tree_ = std::make_shared<search::KdTree<PointInT>>(false);
        }
        if (tree_->getInputCloud() != surface_ && !tree_->setInputCloud(surface_))
        {
            return false;
        }
        if (search_radius_ != 0.0)
        {
            if (search_radius_ < 0.0 || k_ != 0)
            {
                return false;
            }
            search_parameter_ = search_radius_;
            search_method_surface_ = [this](const PointCloudIn& cloud, std::size_t index, double radius,
                                            Indices& indices, std::vector<float>& distances)
            {
                return tree_->radiusSearch(cloud, static_cast<index_t>(index), radius, indices, distances);
            };
        }
        else if (k_ > 0)
        {
            search_parameter_ = k_;
            search_method_surface_ = [this](const PointCloudIn& cloud, std::size_t index, double k,
                                            Indices& indices, std::vector<float>& distances)
            {
                return tree_->nearestKSearch(cloud, static_cast<index_t>(index), static_cast<int>(k),
                                             indices, distances);
            };
        }
        else
        {
            return false;
        }
        return true;
    }

    bool deinitCompute()
    {
        if (fake_surface_)
        {
            surface_.reset();
            fake_surface_ = false;
        }
        return BaseClass::deinitCompute();
    }

    int searchForNeighbors(std::size_t index, double parameter, Indices& indices,
                           std::vector<float>& distances) const
    {
        return search_method_surface_(*this->input_, index, parameter, indices, distances);
    }

    int searchForNeighbors(const PointCloudIn& cloud, std::size_t index, double parameter,
                           Indices& indices, std::vector<float>& distances) const
    {
        return search_method_surface_(cloud, index, parameter, indices, distances);
    }

    const std::string& getClassName() const { return feature_name_; }

    virtual void computeFeature(PointCloudOut& output) = 0;

    std::string feature_name_;
    SearchMethodSurface search_method_surface_;
    PointCloudInConstPtr surface_;
    KdTreePtr tree_;
    double search_parameter_ = 0.0;
    double search_radius_ = 0.0;
    int k_ = 0;
    bool fake_surface_ = false;
};

} // namespace plapoint
