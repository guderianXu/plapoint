#include "opencl_device_enumeration.h"

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace plapoint
{
namespace opencl
{
namespace detail
{
namespace
{

template <typename Value>
Value deviceValue(cl_device_id device, cl_device_info parameter)
{
    Value value{};
    checkOpenCl(
        clGetDeviceInfo(device, parameter, sizeof(Value), &value, nullptr),
        "clGetDeviceInfo");
    return value;
}

void appendEnumerationDiagnostic(
    std::vector<std::string>* diagnostics,
    std::string message)
{
    if (diagnostics)
    {
        diagnostics->push_back(std::move(message));
    }
}

} // namespace

std::string deviceString(cl_device_id device, cl_device_info parameter)
{
    std::size_t size = 0;
    checkOpenCl(clGetDeviceInfo(device, parameter, 0, nullptr, &size), "clGetDeviceInfo(size)");
    std::string value(size, '\0');
    checkOpenCl(
        clGetDeviceInfo(device, parameter, size, value.data(), nullptr),
        "clGetDeviceInfo(value)");
    while (!value.empty() && value.back() == '\0')
    {
        value.pop_back();
    }
    return value;
}

bool candidateBetter(const DeviceCandidate& lhs, const DeviceCandidate& rhs)
{
    if (lhs.unifiedMemory != rhs.unifiedMemory)
    {
        return lhs.unifiedMemory == CL_FALSE;
    }
    if (lhs.computeUnits != rhs.computeUnits)
    {
        return lhs.computeUnits > rhs.computeUnits;
    }
    if (lhs.vendor != rhs.vendor)
    {
        return lhs.vendor < rhs.vendor;
    }
    return lhs.name < rhs.name;
}

std::vector<DeviceCandidate> enumerateDeviceCandidates(
    std::vector<std::string>* diagnostics)
{
    cl_uint platform_count = 0;
    const cl_int platform_count_error = clGetPlatformIDs(0, nullptr, &platform_count);
    if (platform_count_error == -1001)
    {
        appendEnumerationDiagnostic(diagnostics, "OpenCL loader reported no platforms");
        return {};
    }
    if (platform_count_error != CL_SUCCESS)
    {
        appendEnumerationDiagnostic(
            diagnostics,
            "clGetPlatformIDs(count) failed with error "
                + std::to_string(platform_count_error));
        return {};
    }
    std::vector<cl_platform_id> platforms(platform_count);
    if (platform_count != 0)
    {
        const cl_int platform_error = clGetPlatformIDs(
            platform_count, platforms.data(), nullptr);
        if (platform_error != CL_SUCCESS)
        {
            appendEnumerationDiagnostic(
                diagnostics,
                "clGetPlatformIDs(platforms) failed with error "
                    + std::to_string(platform_error));
            return {};
        }
    }

    std::vector<DeviceCandidate> candidates;
    std::size_t global_index = 0;
    for (std::size_t platform_index = 0; platform_index < platforms.size(); ++platform_index)
    {
        const cl_platform_id platform = platforms[platform_index];
        cl_uint device_count = 0;
        const cl_int count_error = clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (count_error == CL_DEVICE_NOT_FOUND || device_count == 0)
        {
            continue;
        }
        if (count_error != CL_SUCCESS)
        {
            appendEnumerationDiagnostic(
                diagnostics,
                "OpenCL platform " + std::to_string(platform_index)
                    + " clGetDeviceIDs(count) failed with error "
                    + std::to_string(count_error));
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        const cl_int devices_error = clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
        if (devices_error != CL_SUCCESS)
        {
            appendEnumerationDiagnostic(
                diagnostics,
                "OpenCL platform " + std::to_string(platform_index)
                    + " clGetDeviceIDs(devices) failed with error "
                    + std::to_string(devices_error));
            global_index += device_count;
            continue;
        }
        for (cl_device_id device : devices)
        {
            DeviceCandidate candidate;
            candidate.index = global_index++;
            candidate.device = device;
            try
            {
                candidate.name = deviceString(device, CL_DEVICE_NAME);
                candidate.vendor = deviceString(device, CL_DEVICE_VENDOR);
                candidate.version = deviceString(device, CL_DEVICE_VERSION);
                candidate.computeUnits = deviceValue<cl_uint>(
                    device, CL_DEVICE_MAX_COMPUTE_UNITS);
                candidate.unifiedMemory = deviceValue<cl_bool>(
                    device, CL_DEVICE_HOST_UNIFIED_MEMORY);
                candidate.available = deviceValue<cl_bool>(device, CL_DEVICE_AVAILABLE);
                candidate.compilerAvailable = deviceValue<cl_bool>(
                    device, CL_DEVICE_COMPILER_AVAILABLE);
                candidates.push_back(std::move(candidate));
            }
            catch (const std::exception& error)
            {
                appendEnumerationDiagnostic(
                    diagnostics,
                    "OpenCL GPU " + std::to_string(candidate.index)
                        + " metadata query failed: " + error.what());
            }
        }
    }
    return candidates;
}

} // namespace detail
} // namespace opencl
} // namespace plapoint
