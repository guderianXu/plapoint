#include <plapoint/opencl/opencl_runtime.h>

#include <stdexcept>

namespace plapoint
{
namespace opencl
{

std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices()
{
    return {};
}

bool hasUsableOpenClDevice() noexcept
{
    return false;
}

void requireUsableOpenClDevice()
{
    throw std::runtime_error("PlaPoint was built without OpenCL support");
}

std::string selectedOpenClDeviceName()
{
    return {};
}

int selectedOpenClDeviceIndex() noexcept
{
    return -1;
}

} // namespace opencl
} // namespace plapoint
