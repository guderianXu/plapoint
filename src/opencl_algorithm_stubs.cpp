#include <plapoint/opencl/height_grid.h>
#include <plapoint/opencl/normal_estimation.h>
#include <plapoint/opencl/preprocessing.h>

#include <stdexcept>
#include <plamatrix/internal/core/device.h>

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
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>&,
    int)
{
    throwOpenClUnavailable();
}

OpenClKnnResult knnSearch(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>&,
    int)
{
    throwOpenClUnavailable();
}

plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> voxelDownsample(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>&,
    float,
    float,
    float)
{
    throwOpenClUnavailable();
}

plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU> voxelDownsample(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>&,
    double,
    double,
    double)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>&,
    int,
    float)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> statisticalOutlierKeepMask(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>&,
    int,
    double)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> radiusOutlierKeepMask(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>&,
    float,
    int)
{
    throwOpenClUnavailable();
}

std::vector<std::uint8_t> radiusOutlierKeepMask(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>&,
    double,
    int)
{
    throwOpenClUnavailable();
}

plamatrix::MatrixXf estimateNormals(const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>&,
                                    int)
{
    throwOpenClUnavailable();
}

plamatrix::MatrixXd estimateNormals(const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>&,
                                    int)
{
    throwOpenClUnavailable();
}

std::uint64_t heightGridOpenClExecutionCount() noexcept
{
    return 0;
}

mesh::HeightGrid<float> buildHeightGrid(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>&,
    const mesh::HeightGridOptions<float>&)
{
    throwOpenClUnavailable();
}

mesh::HeightGrid<double> buildHeightGrid(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>&,
    const mesh::HeightGridOptions<double>&)
{
    throwOpenClUnavailable();
}

} // namespace opencl
} // namespace plapoint
