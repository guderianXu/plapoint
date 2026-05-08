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

    /// Copy normals for selected indices from input to output cloud.
    void copyNormalsForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasNormals()) return;
        int n = static_cast<int>(indices.size());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(n, 3);
        for (int i = 0; i < n; ++i)
        {
            int src = indices[static_cast<std::size_t>(i)];
            nrm(i, 0) = normalCoord(src, 0);
            nrm(i, 1) = normalCoord(src, 1);
            nrm(i, 2) = normalCoord(src, 2);
        }
        output.setNormals(std::move(nrm));
    }

    Scalar normalCoord(int idx, int dim) const
    {
        auto* n = _input->normals();
        return n->getValue(idx, dim);
    }

    PointCloudConstPtr _input;
};

} // namespace plapoint
