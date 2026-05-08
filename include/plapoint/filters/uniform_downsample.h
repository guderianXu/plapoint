#pragma once

#include <plapoint/filters/filter.h>
#include <plapoint/core/point_cloud.h>
#include <plamatrix/dense/dense_matrix.h>
#include <algorithm>

namespace plapoint {

template <typename Scalar, plamatrix::Device Dev>
class UniformDownsample : public Filter<Scalar, Dev>
{
public:
    using PointCloudType = PointCloud<Scalar, Dev>;

    void setStep(int s) { _step = std::max(1, s); }

protected:
    void applyFilter(PointCloudType& output) override
    {
        std::size_t n = this->_input->size();
        std::size_t out_n = (n + static_cast<std::size_t>(_step) - 1) / static_cast<std::size_t>(_step);

        plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(
            static_cast<plamatrix::Index>(out_n), 3);
        std::size_t out_idx = 0;
        for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(_step))
        {
            pts(static_cast<plamatrix::Index>(out_idx), 0) = pointCoord(static_cast<int>(i), 0);
            pts(static_cast<plamatrix::Index>(out_idx), 1) = pointCoord(static_cast<int>(i), 1);
            pts(static_cast<plamatrix::Index>(out_idx), 2) = pointCoord(static_cast<int>(i), 2);
            ++out_idx;
        }
        output = PointCloudType(std::move(pts));
        // Build index list for normal copying
        std::vector<int> kept_indices;
        kept_indices.reserve(out_n);
        for (std::size_t i = 0; i < n; i += static_cast<std::size_t>(_step))
            kept_indices.push_back(static_cast<int>(i));
        this->copyNormalsForIndices(kept_indices, output);
    }

private:
    Scalar pointCoord(int idx, int dim) const
    {
        if constexpr (Dev == plamatrix::Device::CPU)
            return this->_input->points()(idx, dim);
        else
            return this->_input->points().getValue(idx, dim);
    }

    int _step = 2;
};

} // namespace plapoint
