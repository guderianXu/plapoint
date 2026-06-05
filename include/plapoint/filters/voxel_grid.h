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
#include <limits>
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

        struct Accum { Scalar sum_x = 0, sum_y = 0, sum_z = 0; int count = 0; };
        std::unordered_map<VoxelKey, Accum, VoxelKeyHash> voxels;
        voxels.reserve(this->_input->size());

        for (std::size_t i = 0; i < this->_input->size(); ++i)
        {
            VoxelKey key{
                static_cast<int>(std::floor(cpu_points(static_cast<plamatrix::Index>(i), 0) / _leaf_x)),
                static_cast<int>(std::floor(cpu_points(static_cast<plamatrix::Index>(i), 1) / _leaf_y)),
                static_cast<int>(std::floor(cpu_points(static_cast<plamatrix::Index>(i), 2) / _leaf_z))
            };
            auto& acc = voxels[key];
            acc.sum_x += cpu_points(static_cast<plamatrix::Index>(i), 0);
            acc.sum_y += cpu_points(static_cast<plamatrix::Index>(i), 1);
            acc.sum_z += cpu_points(static_cast<plamatrix::Index>(i), 2);
            acc.count += 1;
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(voxels.size()), 3);
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
            pts(out_idx, 0) = acc.sum_x / static_cast<Scalar>(acc.count);
            pts(out_idx, 1) = acc.sum_y / static_cast<Scalar>(acc.count);
            pts(out_idx, 2) = acc.sum_z / static_cast<Scalar>(acc.count);
            ++out_idx;
        }
        output = this->makeOutputCloud(std::move(pts));
    }

private:
#ifdef PLAPOINT_WITH_CUDA
    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::GPU, void>
    applyFilterGpu(PointCloudType& output)
    {
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

        gpu::DeviceBuffer<Scalar> d_centroids(n * 3u);
        const int centroid_count = gpu::voxelGridDownsampleColumnMajor(
            this->_input->points().data(), point_count, _leaf_x, _leaf_y, _leaf_z, d_centroids.get());

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> points(
            static_cast<plamatrix::Index>(centroid_count), 3);
        if (centroid_count > 0)
        {
            PLAPOINT_CHECK_CUDA(cudaMemcpy(points.data(), d_centroids.get(),
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

    Scalar _leaf_x = 1;
    Scalar _leaf_y = 1;
    Scalar _leaf_z = 1;
};

} // namespace plapoint
