#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <plapoint/core/point_cloud.h>
#include <plapoint/core/cloud_algorithm.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/filter_compaction.h>
#endif

namespace plapoint
{

/// Base class for point-cloud filters with shared input validation and attribute-copy helpers.
template <typename Scalar, plamatrix::internal::Device Dev = plamatrix::internal::Device::CPU, typename Enable = void>
class Filter
{
public:
    using PointCloudType = plapoint::internal::DeviceCloud<Scalar, Dev>;
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

    PointCloudType makeOutputCloud(plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>&& points) const
    {
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            return PointCloudType(std::move(points));
        }
        else
        {
            const auto context = _input->executionContext();
            auto resident = plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(points, *context);
            return PointCloudType(std::move(resident), context);
        }
    }

    template <typename Value>
    static plamatrix::internal::ResidentMatrix<Value>
    uploadMatrix(const plamatrix::Matrix<Value, plamatrix::Dynamic, plamatrix::Dynamic>& values,
                 const PointCloudType& output)
    {
        return plamatrix::internal::ResidentMatrix<Value>::copyFrom(values, *output.executionContext());
    }

    /// Copy normals for selected indices from input to output cloud.
    void copyNormalsForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasNormals())
            return;
        int n = static_cast<int>(indices.size());
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> nrm(n, 3);
        if constexpr (Dev == plamatrix::internal::Device::CPU)
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
            auto input_normals = _input->normals()->toHostMatrix();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                nrm(i, 0) = input_normals(src, 0);
                nrm(i, 1) = input_normals(src, 1);
                nrm(i, 2) = input_normals(src, 2);
            }
        }

        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            output.setNormals(std::move(nrm));
        }
        else
        {
            output.setNormals(uploadMatrix(nrm, output));
        }
    }

    /// Copy colors for selected indices from input to output cloud.
    void copyColorsForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasColors())
            return;
        int n = static_cast<int>(indices.size());
        plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(n, 3);
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            auto* input_colors = _input->colors();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                colors(i, 0) = input_colors->operator()(src, 0);
                colors(i, 1) = input_colors->operator()(src, 1);
                colors(i, 2) = input_colors->operator()(src, 2);
            }
        }
        else
        {
            auto input_colors = _input->colors()->toHostMatrix();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                colors(i, 0) = input_colors(src, 0);
                colors(i, 1) = input_colors(src, 1);
                colors(i, 2) = input_colors(src, 2);
            }
        }

        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            output.setColors(std::move(colors));
        }
        else
        {
            output.setColors(uploadMatrix(colors, output));
        }
    }

    /// Copy intensities for selected indices from input to output cloud.
    void copyIntensitiesForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasIntensities())
            return;
        int n = static_cast<int>(indices.size());
        plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(n, 1);
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            auto* input_intensities = _input->intensities();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                intensities(i, 0) = input_intensities->operator()(src, 0);
            }
        }
        else
        {
            auto input_intensities = _input->intensities()->toHostMatrix();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                intensities(i, 0) = input_intensities(src, 0);
            }
        }

        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            output.setIntensities(std::move(intensities));
        }
        else
        {
            output.setIntensities(uploadMatrix(intensities, output));
        }
    }

    /// Copy named scalar fields for selected indices from input to output cloud.
    void copyScalarFieldsForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if (!_input || !_input->hasScalarFields())
            return;
        int n = static_cast<int>(indices.size());
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> scalar_fields(
            n, static_cast<plamatrix::Index>(_input->scalarFieldNames().size()));
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            auto* input_scalar_fields = _input->scalarFields();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                for (plamatrix::Index c = 0; c < scalar_fields.cols(); ++c)
                {
                    scalar_fields(i, c) = input_scalar_fields->operator()(src, c);
                }
            }
        }
        else
        {
            auto input_scalar_fields = _input->scalarFields()->toHostMatrix();
            for (int i = 0; i < n; ++i)
            {
                int src = indices[static_cast<std::size_t>(i)];
                for (plamatrix::Index c = 0; c < scalar_fields.cols(); ++c)
                {
                    scalar_fields(i, c) = input_scalar_fields(src, c);
                }
            }
        }

        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            output.setScalarFields(_input->scalarFieldNames(), std::move(scalar_fields));
        }
        else
        {
            output.setScalarFields(_input->scalarFieldNames(), uploadMatrix(scalar_fields, output));
        }
    }

    /// Copy texture coordinates only when they are aligned one-to-one with points.
    void copyPointTextureCoordsForIndices(
        const std::vector<int>& indices,
        PointCloudType& output) const
    {
        if (!_input || !_input->hasPointAlignedTextureCoords())
        {
            return;
        }
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> texture_coords(
            static_cast<plamatrix::Index>(indices.size()), 2);
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            const auto* source = _input->textureCoords();
            for (std::size_t i = 0; i < indices.size(); ++i)
            {
                const int src = indices[i];
                texture_coords(static_cast<plamatrix::Index>(i), 0) = (*source)(src, 0);
                texture_coords(static_cast<plamatrix::Index>(i), 1) = (*source)(src, 1);
            }
        }
        else
        {
            const auto source = _input->textureCoords()->toHostMatrix();
            for (std::size_t i = 0; i < indices.size(); ++i)
            {
                const int src = indices[i];
                texture_coords(static_cast<plamatrix::Index>(i), 0) = source(src, 0);
                texture_coords(static_cast<plamatrix::Index>(i), 1) = source(src, 1);
            }
        }
        if constexpr (Dev == plamatrix::internal::Device::CPU)
        {
            output.setTextureCoords(std::move(texture_coords));
        }
        else
        {
            output.setTextureCoords(uploadMatrix(texture_coords, output));
        }
        output.setMaterialLibraryFile(_input->materialLibraryFile());
        output.setTextureImageFile(_input->textureImageFile());
    }

    /// Copy all point-wise attributes for selected indices from input to output cloud.
    void copyAttributesForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        copyNormalsForIndices(indices, output);
        copyColorsForIndices(indices, output);
        copyIntensitiesForIndices(indices, output);
        copyScalarFieldsForIndices(indices, output);
        copyPointTextureCoordsForIndices(indices, output);
    }

    /// Build an output cloud from selected source point indices and copy point-wise attributes.
    void copyPointsAndAttributesForIndices(const std::vector<int>& indices, PointCloudType& output) const
    {
        if constexpr (Dev == plamatrix::internal::Device::GPU)
        {
#ifdef PLAPOINT_WITH_CUDA
            output = gpu::gatherPointCloudByIndices(*_input, indices);
            return;
#else
            throw std::runtime_error("PlaPoint was built without CUDA support");
#endif
        }

        const auto& cpu_points = _input->pointsCpu();
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> pts(
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
        return n->operator()(idx, dim);
    }

    PointCloudConstPtr _input;
};

/// Polymorphic base for filters operating on point records.
template <typename PointT>
class Filter<PointT, plamatrix::internal::Device::CPU, std::enable_if_t<!std::is_arithmetic_v<PointT>>>
    : public PCLBase<PointT>
{
public:
    using PointCloud = plapoint::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;
    using Ptr = std::shared_ptr<Filter<PointT>>;
    using ConstPtr = std::shared_ptr<const Filter<PointT>>;

    explicit Filter(bool extract_removed_indices = false)
        : removed_indices_(std::make_shared<Indices>()), extract_removed_indices_(extract_removed_indices)
    {
    }

    virtual ~Filter() = default;

    IndicesConstPtr getRemovedIndices() const
    {
        return removed_indices_;
    }

    void getRemovedIndices(PointIndices& indices)
    {
        indices.indices = *removed_indices_;
    }

    void filter(PointCloud& output)
    {
        if (!this->initCompute())
        {
            return;
        }
        if (this->input_.get() == &output)
        {
            PointCloud temporary;
            applyFilter(temporary);
            output = std::move(temporary);
        }
        else
        {
            applyFilter(output);
        }
        output.header = this->input_->header;
        output.sensor_origin_ = this->input_->sensor_origin_;
        output.sensor_orientation_ = this->input_->sensor_orientation_;
        this->deinitCompute();
    }

protected:
    virtual void applyFilter(PointCloud& output) = 0;

    const std::string& getClassName() const
    {
        return filter_name_;
    }

    IndicesPtr removed_indices_;
    std::string filter_name_;
    bool extract_removed_indices_ = false;
};

template <typename PointT>
class FilterIndices : public Filter<PointT>
{
public:
    using PointCloud = plapoint::PointCloud<PointT>;
    using Ptr = std::shared_ptr<FilterIndices<PointT>>;
    using ConstPtr = std::shared_ptr<const FilterIndices<PointT>>;

    explicit FilterIndices(bool extract_removed_indices = false)
        : Filter<PointT>(extract_removed_indices)
    {
    }

    using Filter<PointT>::filter;

    void filter(Indices& indices)
    {
        if (!this->initCompute())
        {
            return;
        }
        applyFilter(indices);
        this->deinitCompute();
    }

    void setNegative(bool negative) { negative_ = negative; }
    bool getNegative() const { return negative_; }
    void setKeepOrganized(bool keep_organized) { keep_organized_ = keep_organized; }
    bool getKeepOrganized() const { return keep_organized_; }
    void setUserFilterValue(float value) { user_filter_value_ = value; }

protected:
    virtual void applyFilter(Indices& indices) = 0;

    void applyFilter(PointCloud& output) override
    {
        Indices kept;
        applyFilter(kept);
        if (keep_organized_)
        {
            output = *this->input_;
            std::vector<bool> keep_mask(output.size(), false);
            for (const index_t index : kept)
            {
                if (index >= 0 && static_cast<std::size_t>(index) < keep_mask.size())
                {
                    keep_mask[static_cast<std::size_t>(index)] = true;
                }
            }
            for (const index_t index : *this->indices_)
            {
                if (!keep_mask[static_cast<std::size_t>(index)])
                {
                    auto& point = output.points[static_cast<std::size_t>(index)];
                    point.x = static_cast<decltype(point.x)>(user_filter_value_);
                    point.y = static_cast<decltype(point.y)>(user_filter_value_);
                    point.z = static_cast<decltype(point.z)>(user_filter_value_);
                }
            }
            if (!std::isfinite(user_filter_value_) && kept.size() != this->indices_->size())
            {
                output.is_dense = false;
            }
            return;
        }

        output = PointCloud(*this->input_, kept);
        output.height = output.empty() ? 0u : 1u;
        output.width = static_cast<std::uint32_t>(output.size());
    }

    bool negative_ = false;
    bool keep_organized_ = false;
    float user_filter_value_ = std::numeric_limits<float>::quiet_NaN();
};

} // namespace plapoint
