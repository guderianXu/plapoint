#pragma once

#include <cstdint>
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

    /// Optional combined output/removed-index overload for filters that expose removal diagnostics.
    virtual void filter(PointCloudType& output, std::vector<int>& removed_indices)
    {
        (void)output;
        (void)removed_indices;
        throw std::runtime_error("Filter: output/removed-index overload not implemented");
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

    /// Copy colors for selected indices from input to output cloud.
    void copyColorsForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasColors()) return;
        int n = static_cast<int>(indices.size());
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(n, 3);
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            auto* input_colors = _input->colors();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                colors(i, 0) = input_colors->getValue(src, 0);
                colors(i, 1) = input_colors->getValue(src, 1);
                colors(i, 2) = input_colors->getValue(src, 2);
            }
        }
        else
        {
            auto input_colors = _input->colors()->toCpu();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                colors(i, 0) = input_colors(src, 0);
                colors(i, 1) = input_colors(src, 1);
                colors(i, 2) = input_colors(src, 2);
            }
        }

        if constexpr (Dev == plamatrix::Device::CPU)
        {
            output.setColors(std::move(colors));
        }
        else
        {
            output.setColors(colors.toGpu());
        }
    }

    /// Copy intensities for selected indices from input to output cloud.
    void copyIntensitiesForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasIntensities()) return;
        int n = static_cast<int>(indices.size());
        plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU> intensities(n, 1);
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            auto* input_intensities = _input->intensities();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                intensities(i, 0) = input_intensities->getValue(src, 0);
            }
        }
        else
        {
            auto input_intensities = _input->intensities()->toCpu();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                intensities(i, 0) = input_intensities(src, 0);
            }
        }

        if constexpr (Dev == plamatrix::Device::CPU)
        {
            output.setIntensities(std::move(intensities));
        }
        else
        {
            output.setIntensities(intensities.toGpu());
        }
    }

    /// Copy all point-wise attributes for selected indices from input to output cloud.
    void copyAttributesForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        copyNormalsForIndices(indices, output);
        copyColorsForIndices(indices, output);
        copyIntensitiesForIndices(indices, output);
    }

    /// Build an output cloud from selected source point indices and copy point-wise attributes.
    void copyPointsAndAttributesForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        const auto& cpu_points = _input->pointsCpu();
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(indices.size()), 3);
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            const int src = indices[i];
            pts(static_cast<plamatrix::Index>(i), 0) = cpu_points(src, 0);
            pts(static_cast<plamatrix::Index>(i), 1) = cpu_points(src, 1);
            pts(static_cast<plamatrix::Index>(i), 2) = cpu_points(src, 2);
        }
        output = makeOutputCloud(std::move(pts));
        copyAttributesForIndices(indices, output);
    }

    /// Return input indices not present in a kept-index list.
    std::vector<int> removedIndicesFromKept(const std::vector<int>& kept_indices) const
    {
        const std::size_t n = _input ? _input->size() : 0;
        std::vector<std::uint8_t> kept(n, 0);
        for (int idx : kept_indices)
        {
            if (idx >= 0 && static_cast<std::size_t>(idx) < n)
            {
                kept[static_cast<std::size_t>(idx)] = 1;
            }
        }

        std::vector<int> removed;
        removed.reserve(n - kept_indices.size());
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!kept[i])
            {
                removed.push_back(static_cast<int>(i));
            }
        }
        return removed;
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
