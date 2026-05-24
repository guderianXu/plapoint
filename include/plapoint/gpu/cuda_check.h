#pragma once

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>
#include <plamatrix/core/error_check.h>

#include <cstddef>
#include <utility>

#define PLAPOINT_CHECK_CUDA(call) PLAMATRIX_CHECK_CUDA(call)

namespace plapoint {
namespace gpu {

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

        PLAPOINT_CHECK_CUDA(cudaMalloc(&m_ptr, count * sizeof(T)));
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

} // namespace gpu
} // namespace plapoint

#endif // PLAPOINT_WITH_CUDA
