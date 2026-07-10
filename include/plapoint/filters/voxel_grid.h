#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/core/point_cloud.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/voxel_grid.h>
#endif
#include <plamatrix/dense/dense_matrix.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class VoxelGrid : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setLeafSize(Scalar lx, Scalar ly, Scalar lz)
    {
        if (!std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz) ||
            lx <= 0 || ly <= 0 || lz <= 0)
        {
            throw std::invalid_argument("VoxelGrid: leaf size must be positive");
        }
        _leaf_x = lx;
        _leaf_y = ly;
        _leaf_z = lz;
    }

protected:
    void applyFilter(PointCloudType& output) override
    {
        if (!this->_input) return;
        if constexpr (Dev == plamatrix::Device::GPU)
        {
#ifndef PLAPOINT_WITH_CUDA
            throw std::runtime_error("PlaPoint was built without CUDA support");
#else
            applyFilterGpu(output);
            return;
#endif
        }

        const auto& cpu_points = this->_input->pointsCpu();

        struct Accum
        {
            long double mean_x = 0, mean_y = 0, mean_z = 0;
            long double mean_nx = 0, mean_ny = 0, mean_nz = 0;
            long double mean_r = 0, mean_g = 0, mean_b = 0;
            long double mean_intensity = 0;
            std::vector<long double> mean_scalar_fields;
            int count = 0;
        };
        std::unordered_map<VoxelKey, Accum, VoxelKeyHash> voxels;
        voxels.reserve(this->_input->size());
        const bool have_normals = this->_input->hasNormals();
        const bool have_colors = this->_input->hasColors();
        const bool have_intensities = this->_input->hasIntensities();
        const bool have_scalar_fields = this->_input->hasScalarFields();
        const auto* input_normals = this->_input->normals();
        const auto* input_colors = this->_input->colors();
        const auto* input_intensities = this->_input->intensities();
        const auto* input_scalar_fields = this->_input->scalarFields();
        const auto scalar_field_count = static_cast<plamatrix::Index>(
            this->_input->scalarFieldNames().size());

        for (std::size_t i = 0; i < this->_input->size(); ++i)
        {
            const auto row = static_cast<plamatrix::Index>(i);
            const Scalar x = cpu_points(row, 0);
            const Scalar y = cpu_points(row, 1);
            const Scalar z = cpu_points(row, 2);
            VoxelKey key{
                checkedVoxelIndex(x, _leaf_x),
                checkedVoxelIndex(y, _leaf_y),
                checkedVoxelIndex(z, _leaf_z)
            };
            auto& acc = voxels[key];
            acc.count += 1;
            updateMean(acc.mean_x, static_cast<long double>(x), acc.count);
            updateMean(acc.mean_y, static_cast<long double>(y), acc.count);
            updateMean(acc.mean_z, static_cast<long double>(z), acc.count);
            if (have_normals)
            {
                updateMean(acc.mean_nx, static_cast<long double>(input_normals->getValue(row, 0)), acc.count);
                updateMean(acc.mean_ny, static_cast<long double>(input_normals->getValue(row, 1)), acc.count);
                updateMean(acc.mean_nz, static_cast<long double>(input_normals->getValue(row, 2)), acc.count);
            }
            if (have_colors)
            {
                updateMean(acc.mean_r, static_cast<long double>(input_colors->getValue(row, 0)), acc.count);
                updateMean(acc.mean_g, static_cast<long double>(input_colors->getValue(row, 1)), acc.count);
                updateMean(acc.mean_b, static_cast<long double>(input_colors->getValue(row, 2)), acc.count);
            }
            if (have_intensities)
            {
                updateMean(acc.mean_intensity,
                           static_cast<long double>(input_intensities->getValue(row, 0)),
                           acc.count);
            }
            if (have_scalar_fields)
            {
                if (acc.mean_scalar_fields.empty())
                {
                    acc.mean_scalar_fields.assign(static_cast<std::size_t>(scalar_field_count), 0.0L);
                }
                for (plamatrix::Index c = 0; c < scalar_field_count; ++c)
                {
                    updateMean(
                        acc.mean_scalar_fields[static_cast<std::size_t>(c)],
                        static_cast<long double>(input_scalar_fields->getValue(row, c)),
                        acc.count);
                }
            }
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(voxels.size()), 3);
        std::unique_ptr<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>> normals;
        std::unique_ptr<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>> colors;
        std::unique_ptr<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>> intensities;
        std::unique_ptr<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>> scalar_fields;
        if (have_normals)
        {
            normals = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(
                static_cast<plamatrix::Index>(voxels.size()), 3);
        }
        if (have_colors)
        {
            colors = std::make_unique<plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>>(
                static_cast<plamatrix::Index>(voxels.size()), 3);
        }
        if (have_intensities)
        {
            intensities = std::make_unique<plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>>(
                static_cast<plamatrix::Index>(voxels.size()), 1);
        }
        if (have_scalar_fields)
        {
            scalar_fields = std::make_unique<plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>>(
                static_cast<plamatrix::Index>(voxels.size()), scalar_field_count);
        }
        std::vector<VoxelKey> ordered_keys;
        ordered_keys.reserve(voxels.size());
        for (const auto& kv : voxels)
        {
            ordered_keys.push_back(kv.first);
        }
        std::sort(ordered_keys.begin(), ordered_keys.end());

        int out_idx = 0;
        for (const auto& key : ordered_keys)
        {
            const auto& acc = voxels.at(key);
            pts(out_idx, 0) = checkedCentroid(acc.mean_x);
            pts(out_idx, 1) = checkedCentroid(acc.mean_y);
            pts(out_idx, 2) = checkedCentroid(acc.mean_z);
            if (normals)
            {
                (*normals)(out_idx, 0) = checkedCentroid(acc.mean_nx);
                (*normals)(out_idx, 1) = checkedCentroid(acc.mean_ny);
                (*normals)(out_idx, 2) = checkedCentroid(acc.mean_nz);
            }
            if (colors)
            {
                (*colors)(out_idx, 0) = roundedAttribute<std::uint8_t>(acc.mean_r);
                (*colors)(out_idx, 1) = roundedAttribute<std::uint8_t>(acc.mean_g);
                (*colors)(out_idx, 2) = roundedAttribute<std::uint8_t>(acc.mean_b);
            }
            if (intensities)
            {
                (*intensities)(out_idx, 0) = roundedAttribute<std::uint16_t>(acc.mean_intensity);
            }
            if (scalar_fields)
            {
                for (plamatrix::Index c = 0; c < scalar_field_count; ++c)
                {
                    (*scalar_fields)(out_idx, c) =
                        checkedCentroid(acc.mean_scalar_fields[static_cast<std::size_t>(c)]);
                }
            }
            ++out_idx;
        }
        output = this->makeOutputCloud(std::move(pts));
        setOutputAttributes(output, normals.get(), colors.get(), intensities.get(), scalar_fields.get());
    }

private:
#ifdef PLAPOINT_WITH_CUDA
    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::GPU, void>
    applyFilterGpu(PointCloudType& output)
    {
        if (this->_input->hasNormals() || this->_input->hasColors() ||
            this->_input->hasIntensities() || this->_input->hasScalarFields())
        {
            auto cpu_input = std::make_shared<PointCloud<Scalar, plamatrix::Device::CPU>>(this->_input->toCpu());
            VoxelGrid<Scalar, plamatrix::Device::CPU> cpu_filter;
            cpu_filter.setInputCloud(cpu_input);
            cpu_filter.setLeafSize(_leaf_x, _leaf_y, _leaf_z);
            PointCloud<Scalar, plamatrix::Device::CPU> cpu_output;
            cpu_filter.filter(cpu_output);
            output = cpu_output.toGpu();
            return;
        }

        const std::size_t n = this->_input->size();
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error("VoxelGrid GPU: point count exceeds int range");
        }
        const int point_count = static_cast<int>(n);
        if (point_count == 0)
        {
            output = PointCloudType(plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>(0, 3));
            return;
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> centroid_storage(
            static_cast<plamatrix::Index>(n), 3);
        const int centroid_count = gpu::voxelGridDownsampleColumnMajor(
            this->_input->points(), _leaf_x, _leaf_y, _leaf_z, centroid_storage);

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> points(
            static_cast<plamatrix::Index>(centroid_count), 3);
        if (centroid_count > 0)
        {
            PLAPOINT_CHECK_CUDA(cudaMemcpy(points.data(), centroid_storage.data(),
                                           static_cast<std::size_t>(centroid_count) * 3u * sizeof(Scalar),
                                           cudaMemcpyDeviceToDevice));
        }
        output = PointCloudType(std::move(points));
    }
#endif

    struct VoxelKey
    {
        int x;
        int y;
        int z;

        bool operator==(const VoxelKey& other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }

        bool operator<(const VoxelKey& other) const
        {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
    };

    struct VoxelKeyHash
    {
        std::size_t operator()(const VoxelKey& key) const
        {
            std::size_t seed = static_cast<std::size_t>(key.x) + 0x9e3779b97f4a7c15ULL;
            seed ^= static_cast<std::size_t>(key.y) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            seed ^= static_cast<std::size_t>(key.z) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    static int checkedVoxelIndex(Scalar coordinate, Scalar leaf)
    {
        if (!std::isfinite(coordinate))
        {
            throw std::invalid_argument("VoxelGrid: points must be finite");
        }
        const double scaled = std::floor(static_cast<double>(coordinate) / static_cast<double>(leaf));
        if (!std::isfinite(scaled) ||
            scaled < static_cast<double>(std::numeric_limits<int>::min()) ||
            scaled > static_cast<double>(std::numeric_limits<int>::max()))
        {
            throw std::out_of_range("VoxelGrid: voxel index is outside int range");
        }
        return static_cast<int>(scaled);
    }

    static Scalar checkedCentroid(long double centroid)
    {
        if (!std::isfinite(centroid) ||
            centroid < -static_cast<long double>(std::numeric_limits<Scalar>::max()) ||
            centroid > static_cast<long double>(std::numeric_limits<Scalar>::max()))
        {
            throw std::out_of_range("VoxelGrid: centroid is outside scalar range");
        }
        return static_cast<Scalar>(centroid);
    }

    static void updateMean(long double& mean, long double value, int count)
    {
        const long double weight = 1.0L / static_cast<long double>(count);
        mean += (value - mean) * weight;
    }

    template <typename Attribute>
    static Attribute roundedAttribute(long double value)
    {
        if (!std::isfinite(value))
        {
            throw std::out_of_range("VoxelGrid: attribute mean is not finite");
        }
        const long double rounded = std::round(value);
        const long double lo = static_cast<long double>(std::numeric_limits<Attribute>::min());
        const long double hi = static_cast<long double>(std::numeric_limits<Attribute>::max());
        return static_cast<Attribute>(std::clamp(rounded, lo, hi));
    }

    void setOutputAttributes(
        PointCloudType& output,
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>* normals,
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU>* colors,
        plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::CPU>* intensities,
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>* scalar_fields) const
    {
        if (normals)
        {
            if constexpr (Dev == plamatrix::Device::CPU)
            {
                output.setNormals(std::move(*normals));
            }
            else
            {
                output.setNormals(normals->toGpu());
            }
        }
        if (colors)
        {
            if constexpr (Dev == plamatrix::Device::CPU)
            {
                output.setColors(std::move(*colors));
            }
            else
            {
                output.setColors(colors->toGpu());
            }
        }
        if (intensities)
        {
            if constexpr (Dev == plamatrix::Device::CPU)
            {
                output.setIntensities(std::move(*intensities));
            }
            else
            {
                output.setIntensities(intensities->toGpu());
            }
        }
        if (scalar_fields)
        {
            if constexpr (Dev == plamatrix::Device::CPU)
            {
                output.setScalarFields(this->_input->scalarFieldNames(), std::move(*scalar_fields));
            }
            else
            {
                output.setScalarFields(this->_input->scalarFieldNames(), scalar_fields->toGpu());
            }
        }
    }

    Scalar _leaf_x = 1;
    Scalar _leaf_y = 1;
    Scalar _leaf_z = 1;
};

} // namespace plapoint
