#pragma once

#include <plapoint/core/point_cloud.h>
#include <memory>
#include <vector>

namespace plapoint
{

template <typename Scalar, plamatrix::Device Dev>
class Filter
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;
    using PointCloudConstPtr = std::shared_ptr<const PointCloudType>;

    Filter() = default;
    virtual ~Filter() = default;

    void setInputCloud(const PointCloudConstPtr& cloud)
    {
        _input = cloud;
    }

    void filter(PointCloudType& output)
    {
        if (!_input)
        {
            throw std::runtime_error("Input cloud not set");
        }
        applyFilter(output);
    }

    virtual void filter(std::vector<int>& removed_indices)
    {
        (void)removed_indices;
        throw std::runtime_error("Not implemented");
    }

protected:
    virtual void applyFilter(PointCloudType& output) = 0;

    PointCloudConstPtr _input;
};

} // namespace plapoint
