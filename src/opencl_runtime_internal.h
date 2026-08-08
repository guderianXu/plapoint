#pragma once

#include <CL/cl.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace plapoint
{
namespace opencl
{
namespace detail
{

void checkOpenCl(cl_int error, const char* operation);

class OpenClRuntime
{
public:
    static OpenClRuntime& instance();

    OpenClRuntime(const OpenClRuntime&) = delete;
    OpenClRuntime& operator=(const OpenClRuntime&) = delete;

    cl_context context() const noexcept { return _context; }
    cl_device_id device() const noexcept { return _device; }
    const std::string& deviceName() const noexcept { return _deviceName; }
    int deviceIndex() const noexcept { return _deviceIndex; }
    bool supportsFp64() const noexcept { return _supportsFp64; }

    cl_command_queue createQueue() const;
    cl_program program(const std::string& key,
                       const std::string& source,
                       const std::string& options = {});

private:
    OpenClRuntime();
    ~OpenClRuntime();

    cl_context _context = nullptr;
    cl_device_id _device = nullptr;
    std::string _deviceName;
    int _deviceIndex = -1;
    bool _supportsFp64 = false;
    std::mutex _programMutex;
    std::unordered_map<std::string, cl_program> _programs;
};

} // namespace detail
} // namespace opencl
} // namespace plapoint
