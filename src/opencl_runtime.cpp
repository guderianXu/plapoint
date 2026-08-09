#include <plapoint/opencl/opencl_runtime.h>

#include <plamatrix/opencl/runtime.h>

namespace plapoint
{
namespace opencl
{

std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices()
{
    std::vector<OpenClDeviceInfo> result;
    for (const auto& device : plamatrix::opencl::enumerateOpenClGpuDevices())
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
    return plamatrix::opencl::hasUsableOpenClDevice();
}

void requireUsableOpenClDevice()
{
    plamatrix::opencl::requireUsableOpenClDevice();
}

std::string selectedOpenClDeviceName()
{
    return plamatrix::opencl::selectedOpenClDeviceName();
}

int selectedOpenClDeviceIndex() noexcept
{
    return plamatrix::opencl::selectedOpenClDeviceIndex();
}

} // namespace opencl
} // namespace plapoint
