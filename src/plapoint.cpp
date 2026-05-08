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

namespace plapoint {

// PointCloud
template class PointCloud<float,  plamatrix::Device::CPU>;
template class PointCloud<double, plamatrix::Device::CPU>;

// KdTree
template class search::KdTree<float,  plamatrix::Device::CPU>;
template class search::KdTree<double, plamatrix::Device::CPU>;

// Filters
template class Filter<float,  plamatrix::Device::CPU>;
template class Filter<double, plamatrix::Device::CPU>;

template class VoxelGrid<float,  plamatrix::Device::CPU>;
template class VoxelGrid<double, plamatrix::Device::CPU>;

template class StatisticalOutlierRemoval<float,  plamatrix::Device::CPU>;
template class StatisticalOutlierRemoval<double, plamatrix::Device::CPU>;

template class RadiusOutlierRemoval<float,  plamatrix::Device::CPU>;
template class RadiusOutlierRemoval<double, plamatrix::Device::CPU>;

template class UniformDownsample<float,  plamatrix::Device::CPU>;
template class UniformDownsample<double, plamatrix::Device::CPU>;

// Features
template class NormalEstimation<float,  plamatrix::Device::CPU>;
template class NormalEstimation<double, plamatrix::Device::CPU>;

template class NormalRefinement<float,  plamatrix::Device::CPU>;
template class NormalRefinement<double, plamatrix::Device::CPU>;

// Registration
template class IterativeClosestPoint<float,  plamatrix::Device::CPU>;
template class IterativeClosestPoint<double, plamatrix::Device::CPU>;

} // namespace plapoint
