#pragma once

#include <algorithm>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/filter.h>

namespace plapoint
{

/// Keep every Nth point from the input cloud and preserve normals for retained points.
template <typename Scalar, plamatrix::Device Dev>
class UniformDownsample : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    /// Set the sampling step. Values less than one are clamped to one.
    void setStep(int s) { _step = std::max(1, s); }

protected:
    void applyFilter(PointCloudType& output) override
    {
        std::size_t n = this->_input->size();
        std::size_t out_n = (n + static_cast<std::size_t>(_step) - 1) / static_cast<std::size_t>(_step);
        const auto& cpu_points = this->_input->pointsCpu();

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(out_n), 3);
        std::size_t out_idx = 0;
        for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(_step))
        {
            pts(static_cast<plamatrix::Index>(out_idx), 0) = cpu_points(static_cast<plamatrix::Index>(i), 0);
            pts(static_cast<plamatrix::Index>(out_idx), 1) = cpu_points(static_cast<plamatrix::Index>(i), 1);
            pts(static_cast<plamatrix::Index>(out_idx), 2) = cpu_points(static_cast<plamatrix::Index>(i), 2);
            ++out_idx;
        }
        output = this->makeOutputCloud(std::move(pts));
        // Build index list for normal copying
        std::vector<int> kept_indices;
        kept_indices.reserve(out_n);
        for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(_step))
        {
            kept_indices.push_back(static_cast<int>(i));
        }
        this->copyAttributesForIndices(kept_indices, output);
    }

private:
    int _step = 2;
};

} // namespace plapoint
