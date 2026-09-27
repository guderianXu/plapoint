// Explicit template instantiations for common types
// This reduces compilation time for downstream users

#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/filters/filter.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/features/normal_refinement.h>
#include <plapoint/registration/icp.h>
#include <plapoint/core/processing_policy.h>
#include <plapoint/opencl/opencl_runtime.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif
#include <plamatrix/internal/core/device.h>

namespace plapoint {

bool isProcessingDeviceAvailable(ProcessingDevice device) noexcept
{
    switch (device)
    {
    case ProcessingDevice::CPU:
    case ProcessingDevice::Auto:
        return true;
    case ProcessingDevice::CUDA:
#ifdef PLAPOINT_WITH_CUDA
        return gpu::hasUsableCudaDevice();
#else
        return false;
#endif
    case ProcessingDevice::OpenCL:
#ifdef PLAPOINT_WITH_OPENCL
        return opencl::hasUsableOpenClDevice();
#else
        return false;
#endif
    }
    return false;
}

// PointCloud
template class plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
template class plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>;

// KdTree
template class search::internal::DeviceKdTree<float, plamatrix::internal::Device::CPU>;
template class search::internal::DeviceKdTree<double, plamatrix::internal::Device::CPU>;

// Filters
template class Filter<float,  plamatrix::internal::Device::CPU>;
template class Filter<double, plamatrix::internal::Device::CPU>;

template class VoxelGrid<float,  plamatrix::internal::Device::CPU>;
template class VoxelGrid<double, plamatrix::internal::Device::CPU>;

template class StatisticalOutlierRemoval<float,  plamatrix::internal::Device::CPU>;
template class StatisticalOutlierRemoval<double, plamatrix::internal::Device::CPU>;

template class RadiusOutlierRemoval<float,  plamatrix::internal::Device::CPU>;
template class RadiusOutlierRemoval<double, plamatrix::internal::Device::CPU>;

template class UniformDownsample<float,  plamatrix::internal::Device::CPU>;
template class UniformDownsample<double, plamatrix::internal::Device::CPU>;

// Features
template class MatrixNormalEstimation<float,  plamatrix::internal::Device::CPU>;
template class MatrixNormalEstimation<double, plamatrix::internal::Device::CPU>;

template class NormalRefinement<float,  plamatrix::internal::Device::CPU>;
template class NormalRefinement<double, plamatrix::internal::Device::CPU>;

// Registration
template class MatrixIterativeClosestPoint<float,  plamatrix::internal::Device::CPU>;
template class MatrixIterativeClosestPoint<double, plamatrix::internal::Device::CPU>;

} // namespace plapoint
