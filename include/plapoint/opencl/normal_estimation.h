#pragma once

#include <plapoint/core/point_cloud.h>

namespace plapoint
{
namespace opencl
{

/// Estimate float normals using OpenCL uniform-grid KNN and CPU covariance eigensystems.
plamatrix::DenseMatrix<float, plamatrix::Device::CPU> estimateNormals(
    const PointCloud<float, plamatrix::Device::CPU>& input,
    int k);

/// Estimate double normals using OpenCL uniform-grid KNN and CPU covariance eigensystems.
plamatrix::DenseMatrix<double, plamatrix::Device::CPU> estimateNormals(
    const PointCloud<double, plamatrix::Device::CPU>& input,
    int k);

} // namespace opencl
} // namespace plapoint
