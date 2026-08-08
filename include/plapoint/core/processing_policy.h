#pragma once

#include <cstddef>
#include <string>

namespace plapoint
{

/// Execution device preference for CPU-owned convenience APIs.
enum class ProcessingDevice
{
    CPU = 0,
    CUDA = 1,
    /// Backward-compatible name for the CUDA backend.
    GPU = CUDA,
    /// Numeric value retained for compatibility with the original CPU/GPU/Auto enum.
    Auto = 2,
    OpenCL = 3
};

static_assert(static_cast<int>(ProcessingDevice::CPU) == 0);
static_assert(static_cast<int>(ProcessingDevice::CUDA) == 1);
static_assert(static_cast<int>(ProcessingDevice::GPU) == 1);
static_assert(static_cast<int>(ProcessingDevice::Auto) == 2);
static_assert(static_cast<int>(ProcessingDevice::OpenCL) == 3);

/// Neighbor-search implementation used by a high-level processing call.
enum class ProcessingNeighborBackend
{
    None = 0,
    CpuKdTree = 1,
    GpuBruteForce = 2,
    GpuUniformGrid = 3,
    CpuCompatibility = 4,
    OpenClUniformGrid = 5
};

static_assert(static_cast<int>(ProcessingNeighborBackend::None) == 0);
static_assert(static_cast<int>(ProcessingNeighborBackend::CpuKdTree) == 1);
static_assert(static_cast<int>(ProcessingNeighborBackend::GpuBruteForce) == 2);
static_assert(static_cast<int>(ProcessingNeighborBackend::GpuUniformGrid) == 3);
static_assert(static_cast<int>(ProcessingNeighborBackend::CpuCompatibility) == 4);
static_assert(static_cast<int>(ProcessingNeighborBackend::OpenClUniformGrid) == 5);

/// Reports the selected device/backend and any runtime fallback.
struct ProcessingReport
{
    ProcessingDevice requestedDevice = ProcessingDevice::CPU;
    ProcessingDevice actualDevice = ProcessingDevice::CPU;
    /// Backward-compatible alias for actualDevice.
    ProcessingDevice usedDevice = ProcessingDevice::CPU;
    ProcessingNeighborBackend neighborBackend = ProcessingNeighborBackend::None;
    bool usedFallback = false;
    std::string fallbackReason;
};

/// Legacy performance hints and indexed-neighbor-search thresholds.
struct ProcessingPolicy
{
    /// Legacy compatibility hint. Strict high-level Auto dispatch does not use this threshold.
    static constexpr std::size_t autoGpuPointThreshold = 4096;

    /// Minimum query-by-point work product before KNN builds the uniform-grid index.
    static constexpr std::size_t indexedKnnWorkThreshold = 4096;

    static constexpr bool autoPrefersGpu(std::size_t point_count) noexcept
    {
        return point_count >= autoGpuPointThreshold;
    }
};

} // namespace plapoint
