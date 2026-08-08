#include <plapoint/opencl/opencl_runtime.h>

#include "opencl_device_enumeration.h"
#include "opencl_runtime_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace plapoint
{
namespace opencl
{
namespace detail
{
namespace
{

std::string lowerCase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
    {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

void checkOpenCl(cl_int error, const char* operation)
{
    if (error != CL_SUCCESS)
    {
        throw std::runtime_error(
            std::string("OpenCL ") + operation + " failed with error " + std::to_string(error));
    }
}

OpenClRuntime& OpenClRuntime::instance()
{
    static OpenClRuntime runtime;
    return runtime;
}

OpenClRuntime::OpenClRuntime()
{
    std::vector<std::string> enumeration_diagnostics;
    std::vector<DeviceCandidate> candidates = enumerateDeviceCandidates(&enumeration_diagnostics);
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(), [](const DeviceCandidate& candidate)
        {
            return candidate.available == CL_FALSE || candidate.compilerAvailable == CL_FALSE;
        }),
        candidates.end());
    if (candidates.empty())
    {
        std::string message = "OpenCL has no usable GPU device with an online compiler";
        if (!enumeration_diagnostics.empty())
        {
            message += "; enumeration diagnostics: ";
            for (std::size_t index = 0; index < enumeration_diagnostics.size(); ++index)
            {
                if (index != 0)
                {
                    message += "; ";
                }
                message += enumeration_diagnostics[index];
            }
        }
        throw std::runtime_error(message);
    }

    const char* requested_index = std::getenv("PLAPOINT_OPENCL_DEVICE_INDEX");
    if (requested_index && requested_index[0] != '\0' && std::string(requested_index) != "-1")
    {
        std::size_t parsed = 0;
        long long index = -1;
        try
        {
            index = std::stoll(requested_index, &parsed);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                "PLAPOINT_OPENCL_DEVICE_INDEX must be -1 or a non-negative integer");
        }
        if (parsed != std::string(requested_index).size() || index < 0)
        {
            throw std::runtime_error(
                "PLAPOINT_OPENCL_DEVICE_INDEX must be -1 or a non-negative integer");
        }
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(), [&](const DeviceCandidate& candidate)
            {
                return candidate.index != static_cast<std::size_t>(index);
            }),
            candidates.end());
        if (candidates.empty())
        {
            throw std::runtime_error(
                "PLAPOINT_OPENCL_DEVICE_INDEX does not identify a usable OpenCL GPU: "
                + std::to_string(index));
        }
    }
    else if (const char* requested_device = std::getenv("PLAPOINT_OPENCL_DEVICE");
             requested_device && requested_device[0] != '\0')
    {
        const std::string requested = lowerCase(requested_device);
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(), [&](const DeviceCandidate& candidate)
            {
                return lowerCase(candidate.vendor + " " + candidate.name).find(requested) == std::string::npos;
            }),
            candidates.end());
        if (candidates.empty())
        {
            throw std::runtime_error(
                "PLAPOINT_OPENCL_DEVICE did not match any usable GPU device: " + requested);
        }
    }
    std::sort(candidates.begin(), candidates.end(), candidateBetter);
    _device = candidates.front().device;
    _deviceName = candidates.front().name;
    _deviceIndex = static_cast<int>(candidates.front().index);

    const std::string extensions = deviceString(_device, CL_DEVICE_EXTENSIONS);
    _supportsFp64 = extensions.find("cl_khr_fp64") != std::string::npos
        || extensions.find("cl_amd_fp64") != std::string::npos;

    cl_int error = CL_SUCCESS;
    _context = clCreateContext(nullptr, 1, &_device, nullptr, nullptr, &error);
    checkOpenCl(error, "clCreateContext");
}

OpenClRuntime::~OpenClRuntime()
{
    for (const auto& entry : _programs)
    {
        clReleaseProgram(entry.second);
    }
    if (_context)
    {
        clReleaseContext(_context);
    }
}

cl_command_queue OpenClRuntime::createQueue() const
{
    cl_int error = CL_SUCCESS;
    cl_command_queue queue = clCreateCommandQueue(_context, _device, 0, &error);
    checkOpenCl(error, "clCreateCommandQueue");
    return queue;
}

cl_program OpenClRuntime::program(
    const std::string& key,
    const std::string& source,
    const std::string& options)
{
    std::lock_guard<std::mutex> lock(_programMutex);
    const auto found = _programs.find(key);
    if (found != _programs.end())
    {
        return found->second;
    }

    const char* source_pointer = source.c_str();
    const std::size_t source_size = source.size();
    cl_int error = CL_SUCCESS;
    cl_program result = clCreateProgramWithSource(
        _context, 1, &source_pointer, &source_size, &error);
    checkOpenCl(error, "clCreateProgramWithSource");

    const std::string build_options = "-cl-std=CL1.2 " + options;
    error = clBuildProgram(result, 1, &_device, build_options.c_str(), nullptr, nullptr);
    if (error != CL_SUCCESS)
    {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(result, _device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        if (log_size != 0)
        {
            clGetProgramBuildInfo(
                result, _device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        }
        clReleaseProgram(result);
        throw std::runtime_error(
            "OpenCL clBuildProgram failed with error " + std::to_string(error) + ": " + log);
    }
    _programs.emplace(key, result);
    return result;
}

} // namespace detail

std::vector<OpenClDeviceInfo> enumerateOpenClGpuDevices()
{
    std::vector<OpenClDeviceInfo> result;
    for (const auto& candidate : detail::enumerateDeviceCandidates())
    {
        result.push_back({
            candidate.index,
            candidate.name,
            candidate.vendor,
            candidate.version,
            candidate.computeUnits,
            candidate.unifiedMemory != CL_FALSE,
            candidate.available != CL_FALSE,
            candidate.compilerAvailable != CL_FALSE});
    }
    return result;
}

bool hasUsableOpenClDevice() noexcept
{
    try
    {
        (void)detail::OpenClRuntime::instance();
        return true;
    }
    catch (...)
    {
        return false;
    }
}

void requireUsableOpenClDevice()
{
    (void)detail::OpenClRuntime::instance();
}

std::string selectedOpenClDeviceName()
{
    try
    {
        return detail::OpenClRuntime::instance().deviceName();
    }
    catch (...)
    {
        return {};
    }
}

int selectedOpenClDeviceIndex() noexcept
{
    try
    {
        return detail::OpenClRuntime::instance().deviceIndex();
    }
    catch (...)
    {
        return -1;
    }
}

} // namespace opencl
} // namespace plapoint
