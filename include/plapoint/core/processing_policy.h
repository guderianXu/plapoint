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
    /// Explains a normal Auto policy choice. Unlike fallbackReason, this is not an error.
    std::string selectionReason;
};

/// Centralized, tunable Auto-dispatch and indexed-neighbor-search thresholds.
struct ProcessingPolicy
{
    /// Minimum point count for transfer-bound, approximately linear accelerator work.
    static constexpr std::size_t autoGpuPointThreshold = 4096;

    /// Do not accelerate tiny neighborhood problems even when k makes the product look large.
    static constexpr std::size_t autoNeighborhoodMinPointCount = 256;

    /// Minimum query-by-point work product before KNN builds the uniform-grid index.
    static constexpr std::size_t indexedKnnWorkThreshold = 4096;

    static constexpr bool autoPrefersGpu(std::size_t point_count) noexcept
    {
        return point_count >= autoGpuPointThreshold;
    }

    /// Decide whether CPU-owned neighborhood work is large enough to amortize transfers.
    static constexpr bool autoPrefersNeighborhoodGpu(std::size_t point_count,
                                                     std::size_t neighbors) noexcept
    {
        if (point_count < autoNeighborhoodMinPointCount || neighbors == 0)
        {
            return false;
        }
        if (point_count > static_cast<std::size_t>(-1) / neighbors)
        {
            return true;
        }
        return point_count * neighbors >= indexedKnnWorkThreshold;
    }
};

} // namespace plapoint
