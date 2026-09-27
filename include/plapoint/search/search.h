#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <plapoint/core/point_cloud.h>

namespace plapoint::search
{

/// Common point-type search interface used by filters and feature estimators.
template <typename PointT>
class Search
{
public:
    using PointCloud = plapoint::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;
    using Indices = plapoint::Indices;
    using IndicesPtr = plapoint::IndicesPtr;
    using IndicesConstPtr = plapoint::IndicesConstPtr;
    using Ptr = std::shared_ptr<Search<PointT>>;
    using ConstPtr = std::shared_ptr<const Search<PointT>>;

    explicit Search(const std::string& name = "", bool sorted = false)
        : _name(name), _sortedResults(sorted)
    {
    }

    virtual ~Search() = default;

    virtual const std::string& getName() const { return _name; }
    virtual void setSortedResults(bool sorted) { _sortedResults = sorted; }
    virtual bool getSortedResults() { return _sortedResults; }
    bool getSortedResults() const { return _sortedResults; }

    virtual bool setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = {}) = 0;
    virtual PointCloudConstPtr getInputCloud() const noexcept = 0;
    virtual IndicesConstPtr getIndices() const noexcept = 0;

    virtual int nearestKSearch(const PointT& point, int k, Indices& indices,
                               std::vector<float>& squared_distances) const = 0;
    virtual int radiusSearch(const PointT& point, double radius, Indices& indices,
                             std::vector<float>& squared_distances, unsigned int max_nn = 0) const = 0;

    /// Search with a different XYZ-bearing query point type.
    template <typename PointTDiff>
    int nearestKSearchT(const PointTDiff& point, int k, Indices& indices,
                        std::vector<float>& squared_distances) const
    {
        PointT query{};
        query.x = point.x;
        query.y = point.y;
        query.z = point.z;
        return nearestKSearch(query, k, indices, squared_distances);
    }

    /// Radius search with a different XYZ-bearing query point type.
    template <typename PointTDiff>
    int radiusSearchT(const PointTDiff& point, double radius, Indices& indices,
                      std::vector<float>& squared_distances, unsigned int max_nn = 0) const
    {
        PointT query{};
        query.x = point.x;
        query.y = point.y;
        query.z = point.z;
        return radiusSearch(query, radius, indices, squared_distances, max_nn);
    }

    virtual int nearestKSearch(const PointCloud& cloud, index_t index, int k, Indices& indices,
                               std::vector<float>& squared_distances) const
    {
        return nearestKSearch(cloud.points.at(checkedIndex(index)), k, indices, squared_distances);
    }

    virtual int nearestKSearch(index_t index, int k, Indices& indices,
                               std::vector<float>& squared_distances) const
    {
        const auto cloud = getInputCloud();
        if (!cloud)
        {
            throw std::runtime_error("Search: input cloud not set");
        }
        const auto subset = getIndices();
        const index_t cloud_index = subset ? subset->at(checkedIndex(index)) : index;
        return nearestKSearch(*cloud, cloud_index, k, indices, squared_distances);
    }

    virtual void nearestKSearch(const PointCloud& cloud, const Indices& query_indices, int k,
                                std::vector<Indices>& indices,
                                std::vector<std::vector<float>>& squared_distances) const
    {
        const std::size_t count = query_indices.empty() ? cloud.size() : query_indices.size();
        indices.resize(count);
        squared_distances.resize(count);
        for (std::size_t row = 0; row < count; ++row)
        {
            const index_t query_index = query_indices.empty() ? static_cast<index_t>(row) : query_indices[row];
            nearestKSearch(cloud, query_index, k, indices[row], squared_distances[row]);
        }
    }

    template <typename PointTDiff>
    void nearestKSearchT(const plapoint::PointCloud<PointTDiff>& cloud, const Indices& query_indices, int k,
                         std::vector<Indices>& indices,
                         std::vector<std::vector<float>>& squared_distances) const
    {
        const std::size_t count = query_indices.empty() ? cloud.size() : query_indices.size();
        indices.resize(count);
        squared_distances.resize(count);
        for (std::size_t row = 0; row < count; ++row)
        {
            const index_t query_index = query_indices.empty() ? static_cast<index_t>(row) : query_indices[row];
            nearestKSearchT(cloud.points.at(checkedIndex(query_index)), k, indices[row], squared_distances[row]);
        }
    }

    virtual int radiusSearch(const PointCloud& cloud, index_t index, double radius, Indices& indices,
                             std::vector<float>& squared_distances, unsigned int max_nn = 0) const
    {
        return radiusSearch(cloud.points.at(checkedIndex(index)), radius, indices, squared_distances, max_nn);
    }

    virtual int radiusSearch(index_t index, double radius, Indices& indices,
                             std::vector<float>& squared_distances, unsigned int max_nn = 0) const
    {
        const auto cloud = getInputCloud();
        if (!cloud)
        {
            throw std::runtime_error("Search: input cloud not set");
        }
        const auto subset = getIndices();
        const index_t cloud_index = subset ? subset->at(checkedIndex(index)) : index;
        return radiusSearch(*cloud, cloud_index, radius, indices, squared_distances, max_nn);
    }

    virtual void radiusSearch(const PointCloud& cloud, const Indices& query_indices, double radius,
                              std::vector<Indices>& indices,
                              std::vector<std::vector<float>>& squared_distances,
                              unsigned int max_nn = 0) const
    {
        const std::size_t count = query_indices.empty() ? cloud.size() : query_indices.size();
        indices.resize(count);
        squared_distances.resize(count);
        for (std::size_t row = 0; row < count; ++row)
        {
            const index_t query_index = query_indices.empty() ? static_cast<index_t>(row) : query_indices[row];
            radiusSearch(cloud, query_index, radius, indices[row], squared_distances[row], max_nn);
        }
    }

    template <typename PointTDiff>
    void radiusSearchT(const plapoint::PointCloud<PointTDiff>& cloud, const Indices& query_indices, double radius,
                       std::vector<Indices>& indices, std::vector<std::vector<float>>& squared_distances,
                       unsigned int max_nn = 0) const
    {
        const std::size_t count = query_indices.empty() ? cloud.size() : query_indices.size();
        indices.resize(count);
        squared_distances.resize(count);
        for (std::size_t row = 0; row < count; ++row)
        {
            const index_t query_index = query_indices.empty() ? static_cast<index_t>(row) : query_indices[row];
            radiusSearchT(cloud.points.at(checkedIndex(query_index)), radius,
                          indices[row], squared_distances[row], max_nn);
        }
    }

private:
    std::string _name;
    bool _sortedResults = false;
    static std::size_t checkedIndex(index_t index)
    {
        if (index < 0)
        {
            throw std::out_of_range("Search: query index must be non-negative");
        }
        return static_cast<std::size_t>(index);
    }
};

} // namespace plapoint::search
