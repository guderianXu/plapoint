#include <plapoint/opencl/height_grid.h>
#include <plapoint/opencl/normal_estimation.h>
#include <plapoint/opencl/preprocessing.h>

#include <stdexcept>

namespace plapoint
{
namespace opencl
{
namespace
{

[[noreturn]] void throwOpenClUnavailable()
{
    throw std::runtime_error("PlaPoint was built without OpenCL support");
}

} // namespace

OpenClKnnResult knnSearch(
    const PointCloud<float, plamatrix::Device::CPU>&,
    int)
{
    throwOpenClUnavailable();
}

OpenClKnnResult knnSearch(
    const PointCloud<double, plamatrix::Device::CPU>&,
    int)
{
    throwOpenClUnavailable();
}

PointCloud<float, plamatrix::Device::CPU> voxelDownsample(
    const PointCloud<float, plamatrix::Device::CPU>&,
    float,
    float,
    float)
{
    throwOpenClUnavailable();
}

PointCloud<double, plamatrix::Device::CPU> voxelDownsample(
    const PointCloud<double, plamatrix::Device::CPU>&,
    double,
    double,
    double)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const PointCloud<float, plamatrix::Device::CPU>&,
    int,
    float)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const PointCloud<double, plamatrix::Device::CPU>&,
    int,
    double)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> radiusOutlierKeepMask(
    const PointCloud<float, plamatrix::Device::CPU>&,
    float,
    int)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> radiusOutlierKeepMask(
    const PointCloud<double, plamatrix::Device::CPU>&,
    double,
    int)
{
    throwOpenClUnavailable();
}

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> estimateNormals(
    const PointCloud<float, plamatrix::Device::CPU>&,
    int)
{
    throwOpenClUnavailable();
}

plamatrix::DenseMatrix<double, plamatrix::Device::CPU> estimateNormals(
    const PointCloud<double, plamatrix::Device::CPU>&,
    int)
{
    throwOpenClUnavailable();
}

std::uint64_t heightGridOpenClExecutionCount() noexcept
{
    return 0;
}

mesh::HeightGrid<float> buildHeightGrid(
    const PointCloud<float, plamatrix::Device::CPU>&,
    const mesh::HeightGridOptions<float>&)
{
    throwOpenClUnavailable();
}

mesh::HeightGrid<double> buildHeightGrid(
    const PointCloud<double, plamatrix::Device::CPU>&,
    const mesh::HeightGridOptions<double>&)
{
    throwOpenClUnavailable();
}

} // namespace opencl
} // namespace plapoint
