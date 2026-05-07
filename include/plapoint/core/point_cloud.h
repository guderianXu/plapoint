#pragma once

#include <plamatrix/dense/dense_matrix.h>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace plapoint
{

template <typename Scalar, plamatrix::Device Dev>
class PointCloud
{
public:
    using MatrixType = plamatrix::DenseMatrix<Scalar, Dev>;

    PointCloud() : _points(0, 3) {}

    explicit PointCloud(size_t num_points) : _points(num_points, 3) {}

    explicit PointCloud(MatrixType&& pts)
    {
        if (pts.cols() != 3)
        {
            throw std::runtime_error("PointCloud requires Nx3 matrix");
        }
        _points = std::move(pts);
    }

    size_t size() const { return _points.rows(); }

    const MatrixType& points() const { return _points; }

    MatrixType& points() { return _points; }

    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::CPU, PointCloud<Scalar, plamatrix::Device::GPU>>
    toGpu() const
    {
        return PointCloud<Scalar, plamatrix::Device::GPU>(_points.toGpu());
    }

    template <plamatrix::Device D = Dev>
    std::enable_if_t<D == plamatrix::Device::GPU, PointCloud<Scalar, plamatrix::Device::CPU>>
    toCpu() const
    {
        return PointCloud<Scalar, plamatrix::Device::CPU>(_points.toCpu());
    }

private:
    MatrixType _points;
};

} // namespace plapoint
