#pragma once

#include <plapoint/core/point_cloud.h>
#include <plamatrix/internal/core/device.h>

namespace plapoint
{
namespace opencl
{

    /// Estimate float normals using OpenCL uniform-grid KNN and CPU covariance eigensystems.
    plamatrix::MatrixXf
    estimateNormals(const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>& input, int k);

    /// Estimate double normals using OpenCL uniform-grid KNN and CPU covariance eigensystems.
    plamatrix::MatrixXd
    estimateNormals(const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>& input, int k);

} // namespace opencl
} // namespace plapoint
