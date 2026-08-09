#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace plapoint
{
namespace opencl
{

/// Stable information for one GPU in OpenCL platform/device enumeration order.
struct OpenClDeviceInfo
{
    std::size_t index = 0;
    std::string name;
    std::string vendor;
    std::string version;
    std::uint32_t computeUnits = 0;
    bool unifiedMemory = false;
    bool available = false;
    bool compilerAvailable = false;
};

/// Enumerate OpenCL GPU devices in stable platform/device order. Returns empty when OpenCL is not built.
std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices();

/// Return true when a usable GPU OpenCL device and compiler can be initialized.
/// PLAMATRIX_OPENCL_DEVICE_INDEX is read on first runtime use; -1 or unset selects automatically.
/// PLAPOINT_OPENCL_DEVICE_INDEX remains available as a compatibility fallback.
bool hasUsableOpenClDevice() noexcept;

/// Initialize the selected OpenCL GPU or throw the full platform/device selection error.
void requireUsableOpenClDevice();

/// Return the selected OpenCL device name, or an empty string when none is usable/OpenCL is not built.
std::string selectedOpenClDeviceName();

/// Return the selected stable GPU index, or -1 when no device is usable.
int selectedOpenClDeviceIndex() noexcept;

} // namespace opencl
} // namespace plapoint
