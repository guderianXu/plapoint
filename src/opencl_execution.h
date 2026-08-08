#pragma once

#include "opencl_runtime_internal.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace plapoint
{
namespace opencl
{
namespace detail
{

/// Bound serial work assigned to one OpenCL work-item to avoid driver watchdog resets.
inline constexpr int maximumSequentialReductionItems = 262144;

class CommandQueue
{
public:
    explicit CommandQueue(cl_command_queue queue) : _queue(queue) {}
    ~CommandQueue() { if (_queue) clReleaseCommandQueue(_queue); }
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;
    operator cl_command_queue() const noexcept { return _queue; }

private:
    cl_command_queue _queue = nullptr;
};

class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    DeviceBuffer(cl_context context, cl_mem_flags flags, std::size_t size, void* host = nullptr)
    {
        cl_int error = CL_SUCCESS;
        _memory = clCreateBuffer(context, flags, size, host, &error);
        checkOpenCl(error, "clCreateBuffer");
    }
    ~DeviceBuffer() { if (_memory) clReleaseMemObject(_memory); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : _memory(other._memory) { other._memory = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (_memory) clReleaseMemObject(_memory);
            _memory = other._memory;
            other._memory = nullptr;
        }
        return *this;
    }
    cl_mem get() const noexcept { return _memory; }

private:
    cl_mem _memory = nullptr;
};

class CompiledKernel
{
public:
    CompiledKernel(cl_program program, const char* name)
    {
        cl_int error = CL_SUCCESS;
        _kernel = clCreateKernel(program, name, &error);
        checkOpenCl(error, "clCreateKernel");
    }
    ~CompiledKernel() { if (_kernel) clReleaseKernel(_kernel); }
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;
    operator cl_kernel() const noexcept { return _kernel; }

private:
    cl_kernel _kernel = nullptr;
};

template <typename Value>
std::size_t byteSize(std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value))
    {
        throw std::overflow_error("OpenCL buffer byte size overflow");
    }
    return count * sizeof(Value);
}

template <typename Value>
DeviceBuffer inputVector(OpenClRuntime& runtime, const std::vector<Value>& values)
{
    return DeviceBuffer(
        runtime.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        byteSize<Value>(values.size()), const_cast<Value*>(values.data()));
}

template <typename Value>
DeviceBuffer inOutVector(OpenClRuntime& runtime, std::vector<Value>& values)
{
    return DeviceBuffer(
        runtime.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        byteSize<Value>(values.size()), values.data());
}

template <typename Value>
void kernelArg(cl_kernel kernel, cl_uint index, const Value& value)
{
    checkOpenCl(clSetKernelArg(kernel, index, sizeof(Value), &value), "clSetKernelArg");
}

inline void kernelBufferArg(cl_kernel kernel, cl_uint index, const DeviceBuffer& buffer)
{
    const cl_mem memory = buffer.get();
    checkOpenCl(
        clSetKernelArg(kernel, index, sizeof(memory), &memory),
        "clSetKernelArg(buffer)");
}

template <typename Value>
void readVector(cl_command_queue queue, const DeviceBuffer& buffer, std::vector<Value>& values)
{
    checkOpenCl(
        clEnqueueReadBuffer(
            queue, buffer.get(), CL_TRUE, 0, byteSize<Value>(values.size()),
            values.data(), 0, nullptr, nullptr),
        "clEnqueueReadBuffer");
}

template <typename Scalar>
void requireFp64(OpenClRuntime& runtime)
{
    if constexpr (std::is_same_v<Scalar, double>)
    {
        if (!runtime.supportsFp64())
        {
            throw std::runtime_error("Selected OpenCL device does not support double precision");
        }
    }
}

template <typename Scalar>
std::string realBuildOptions()
{
    return std::is_same_v<Scalar, double> ? "-DPLAPOINT_REAL_DOUBLE=1" : "";
}

} // namespace detail
} // namespace opencl
} // namespace plapoint
