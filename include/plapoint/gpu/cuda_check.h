#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>
#include <plamatrix/core/error_check.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#define PLAPOINT_CHECK_CUDA(call) PLAMATRIX_CHECK_CUDA(call)

namespace plapoint {
namespace gpu {

namespace detail {

template <typename T>
inline std::size_t checkedAllocationBytes(std::size_t count, const char* label)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        throw std::overflow_error(std::string(label) + " allocation size overflow");
    }
    return count * sizeof(T);
}

} // namespace detail

inline bool hasUsableCudaDevice()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0)
    {
        cudaGetLastError();
        return false;
    }

    err = cudaFree(nullptr);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }

    return true;
}

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count)
    {
        allocate(count);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : m_ptr(other.m_ptr)
        , m_count(other.m_count)
    {
        other.m_ptr = nullptr;
        other.m_count = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            m_ptr = other.m_ptr;
            m_count = other.m_count;
            other.m_ptr = nullptr;
            other.m_count = 0;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        reset();
    }

    void allocate(std::size_t count)
    {
        reset();
        if (count == 0)
        {
            return;
        }

        const std::size_t bytes = detail::checkedAllocationBytes<T>(count, "DeviceBuffer");
        PLAPOINT_CHECK_CUDA(cudaMalloc(&m_ptr, bytes));
        m_count = count;
    }

    void reset() noexcept
    {
        if (m_ptr)
        {
            static_cast<void>(cudaFree(m_ptr));
            m_ptr = nullptr;
            m_count = 0;
        }
    }

    T* get() { return m_ptr; }
    const T* get() const { return m_ptr; }
    std::size_t size() const { return m_count; }

private:
    T* m_ptr = nullptr;
    std::size_t m_count = 0;
};

template <typename T>
class HostPinnedBuffer
{
public:
    HostPinnedBuffer() = default;

    explicit HostPinnedBuffer(std::size_t count)
    {
        allocate(count);
    }

    HostPinnedBuffer(const HostPinnedBuffer&) = delete;
    HostPinnedBuffer& operator=(const HostPinnedBuffer&) = delete;

    HostPinnedBuffer(HostPinnedBuffer&& other) noexcept
        : m_ptr(other.m_ptr)
        , m_count(other.m_count)
    {
        other.m_ptr = nullptr;
        other.m_count = 0;
    }

    HostPinnedBuffer& operator=(HostPinnedBuffer&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            m_ptr = other.m_ptr;
            m_count = other.m_count;
            other.m_ptr = nullptr;
            other.m_count = 0;
        }
        return *this;
    }

    ~HostPinnedBuffer()
    {
        reset();
    }

    void allocate(std::size_t count)
    {
        reset();
        if (count == 0)
        {
            return;
        }

        const std::size_t bytes = detail::checkedAllocationBytes<T>(count, "HostPinnedBuffer");
        PLAPOINT_CHECK_CUDA(cudaHostAlloc(&m_ptr, bytes, cudaHostAllocDefault));
        m_count = count;
    }

    void reset() noexcept
    {
        if (m_ptr)
        {
            static_cast<void>(cudaFreeHost(m_ptr));
            m_ptr = nullptr;
            m_count = 0;
        }
    }

    T* get() { return m_ptr; }
    const T* get() const { return m_ptr; }
    std::size_t size() const { return m_count; }

private:
    T* m_ptr = nullptr;
    std::size_t m_count = 0;
};

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
