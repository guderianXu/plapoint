#include <plapoint/opencl/opencl_runtime.h>

#include <plamatrix/internal/opencl/runtime.h>

namespace plapoint
{
namespace opencl
{

std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices()
{
    std::vector<OpenClDeviceInfo> result;
    for (const auto& device : plamatrix::internal::opencl::enumerateOpenClGpuDevices())
    {
        result.push_back({
            device.index,
            device.name,
            device.vendor,
            device.version,
            device.computeUnits,
            device.unifiedMemory,
            device.available,
            device.compilerAvailable});
    }
    return result;
}

bool hasUsableOpenClDevice() noexcept
{
    return plamatrix::internal::opencl::hasUsableOpenClDevice();
}

void requireUsableOpenClDevice()
{
    plamatrix::internal::opencl::requireUsableOpenClDevice();
}

std::string selectedOpenClDeviceName()
{
    return plamatrix::internal::opencl::selectedOpenClDeviceName();
}

int selectedOpenClDeviceIndex() noexcept
{
    return plamatrix::internal::opencl::selectedOpenClDeviceIndex();
}

} // namespace opencl
} // namespace plapoint
