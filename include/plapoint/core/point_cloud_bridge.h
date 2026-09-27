#pragma once

#include <plapoint/core/point_cloud.h>

#include <plamatrix/dense/matrix.h>
#include <plamatrix/internal/core/device.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace plapoint
{
    namespace detail
    {

        /// Materialize the public point structure cloud as a PlaMatrix column-major XYZ matrix.
        template <typename Scalar = float, typename PointT>
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, 3> pointCoordinates(const PointCloud<PointT>& cloud)
        {
            if (static_cast<std::size_t>(cloud.width) * cloud.height != cloud.size())
            {
                throw std::invalid_argument("PointCloud: width and height do not match the points vector");
            }

            plamatrix::Matrix<Scalar, plamatrix::Dynamic, 3> coordinates(static_cast<plamatrix::Index>(cloud.size()),
                                                                         3);
            for (plamatrix::Index row = 0; row < coordinates.rows(); ++row)
            {
                const auto& point = cloud.points[static_cast<std::size_t>(row)];
                coordinates(row, 0) = point.x;
                coordinates(row, 1) = point.y;
                coordinates(row, 2) = point.z;
            }
            return coordinates;
        }

        template <typename Scalar = float, typename PointT>
        std::shared_ptr<plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>>
        toMatrixCloud(const PointCloud<PointT>& cloud)
        {
            const auto coordinates = pointCoordinates<Scalar>(cloud);
            plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> legacy_points(coordinates.rows(),
                                                                                            coordinates.cols());
            std::copy_n(coordinates.data(), static_cast<std::size_t>(coordinates.size()), legacy_points.data());
            return std::make_shared<plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>>(
                std::move(legacy_points));
        }

        template <typename PointT> auto pointVector(const PointT& point)
        {
            using Scalar = std::decay_t<decltype(point.x)>;
            return plamatrix::Matrix<Scalar, 3, 1>(point.x, point.y, point.z);
        }

    } // namespace detail
} // namespace plapoint
