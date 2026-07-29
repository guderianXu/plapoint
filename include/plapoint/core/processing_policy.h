#pragma once

#include <cstddef>
#include <string>

namespace plapoint
{

/// Execution device preference for CPU-owned convenience APIs.
enum class ProcessingDevice
{
    CPU,
    GPU,
    Auto
};

/// Neighbor-search implementation used by a high-level processing call.
enum class ProcessingNeighborBackend
{
    None,
    CpuKdTree,
    GpuBruteForce,
    GpuUniformGrid,
    CpuCompatibility
};

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

/// Benchmark-calibrated thresholds shared by high-level Auto APIs and GPU KNN.
struct ProcessingPolicy
{
    /// Minimum CPU-owned point count where transfer plus GPU setup is normally worthwhile.
    static constexpr std::size_t autoGpuPointThreshold = 4096;

    /// Minimum query-by-point work product before KNN builds the uniform-grid index.
    static constexpr std::size_t indexedKnnWorkThreshold = 4096;

    static constexpr bool autoPrefersGpu(std::size_t point_count) noexcept
    {
        return point_count >= autoGpuPointThreshold;
    }
};

} // namespace plapoint
