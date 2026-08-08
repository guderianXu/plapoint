#pragma once

#include "opencl_runtime_internal.h"

#include <cstddef>
#include <string>
#include <vector>

namespace plapoint
{
namespace opencl
{
namespace detail
{

struct DeviceCandidate
{
    std::size_t index = 0;
    cl_device_id device = nullptr;
    std::string name;
    std::string vendor;
    std::string version;
    cl_uint computeUnits = 0;
    cl_bool unifiedMemory = CL_TRUE;
    cl_bool available = CL_FALSE;
    cl_bool compilerAvailable = CL_FALSE;
};

std::string deviceString(cl_device_id device, cl_device_info parameter);

bool candidateBetter(const DeviceCandidate& lhs, const DeviceCandidate& rhs);

std::vector<DeviceCandidate> enumerateDeviceCandidates(
    std::vector<std::string>* diagnostics = nullptr);

} // namespace detail
} // namespace opencl
} // namespace plapoint
