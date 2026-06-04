#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

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
        const auto* cpu_points = static_cast<const plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>*>(nullptr);
        auto staged_points = stagePointsIfNeeded();
        if constexpr (Dev == plamatrix::Device::CPU)
        {
            cpu_points = &this->_input->points();
        }
        else
        {
            cpu_points = &staged_points;
        }

        using Key = std::tuple<int, int, int>;
        struct Accum { Scalar sum_x = 0, sum_y = 0, sum_z = 0; int count = 0; };
        std::map<Key, Accum> voxels;

        for (std::size_t i = 0; i < this->_input->size(); ++i)
        {
            Key key{
                static_cast<int>(std::floor((*cpu_points)(static_cast<plamatrix::Index>(i), 0) / _leaf_x)),
                static_cast<int>(std::floor((*cpu_points)(static_cast<plamatrix::Index>(i), 1) / _leaf_y)),
                static_cast<int>(std::floor((*cpu_points)(static_cast<plamatrix::Index>(i), 2) / _leaf_z))
            };
            auto& acc = voxels[key];
            acc.sum_x += (*cpu_points)(static_cast<plamatrix::Index>(i), 0);
            acc.sum_y += (*cpu_points)(static_cast<plamatrix::Index>(i), 1);
            acc.sum_z += (*cpu_points)(static_cast<plamatrix::Index>(i), 2);
            acc.count += 1;
        }

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(voxels.size()), 3);
        int out_idx = 0;
        for (const auto& kv : voxels)
        {
            const auto& acc = kv.second;
            pts(out_idx, 0) = acc.sum_x / static_cast<Scalar>(acc.count);
            pts(out_idx, 1) = acc.sum_y / static_cast<Scalar>(acc.count);
            pts(out_idx, 2) = acc.sum_z / static_cast<Scalar>(acc.count);
            ++out_idx;
        }
        output = this->makeOutputCloud(std::move(pts));
    }

private:
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> stagePointsIfNeeded() const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>(0, 3);
        else
            return this->_input->points().toCpu();
    }

    Scalar _leaf_x = 1;
    Scalar _leaf_y = 1;
    Scalar _leaf_z = 1;
};

} // namespace plapoint
