#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <plapoint/core/point_cloud.h>

namespace plapoint
{

/// Base class for point-cloud filters with shared input validation and attribute-copy helpers.
template <typename Scalar, plamatrix::Device Dev>
class Filter
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;
    using PointCloudConstPtr = std::shared_ptr<const PointCloudType>;

    Filter() = default;
    virtual ~Filter() = default;

    /// Set the input cloud consumed by subsequent filter calls.
    void setInputCloud(const PointCloudConstPtr& cloud)
    {
        _input = cloud;
    }

    /// Run the filter and throw if no input cloud has been configured.
    void filter(PointCloudType& output)
    {
        if (!_input)
        {
            throw std::runtime_error("Filter: input cloud not set");
        }
        applyFilter(output);
    }

    /// Optional removed-index overload for filters that expose removal diagnostics.
    virtual void filter(std::vector<int>& removed_indices)
    {
        (void)removed_indices;
        throw std::runtime_error("Filter: removed-index overload not implemented");
    }

protected:
    virtual void applyFilter(PointCloudType& output) = 0;

    static PointCloudType makeOutputCloud(
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>&& points)
    {
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            return PointCloudType(std::move(points));
        }
        else
        {
            return PointCloudType(points.toGpu());
        }
    }

    /// Copy normals for selected indices from input to output cloud.
    void copyNormalsForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasNormals()) return;
        int n = static_cast<int>(indices.size());
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> nrm(n, 3);
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                nrm(i, 0) = normalCoord(src, 0);
                nrm(i, 1) = normalCoord(src, 1);
                nrm(i, 2) = normalCoord(src, 2);
            }
        }
        else
        {
            auto input_normals = _input->normals()->toCpu();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                nrm(i, 0) = input_normals(src, 0);
                nrm(i, 1) = input_normals(src, 1);
                nrm(i, 2) = input_normals(src, 2);
            }
        }

        if constexpr (Dev == plamatrix::Device::CPU)
        {
            output.setNormals(std::move(nrm));
        }
        else
        {
            output.setNormals(nrm.toGpu());
        }
    }

    /// Return one normal coordinate from the current input cloud.
    Scalar normalCoord(int idx, int dim) const
    {
        auto* n = _input->normals();
        return n->getValue(idx, dim);
    }

    PointCloudConstPtr _input;
};

} // namespace plapoint
