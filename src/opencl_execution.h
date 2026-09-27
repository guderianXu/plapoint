#pragma once

#include "opencl_runtime_internal.h"

#include <plamatrix/internal/opencl/execution.h>

#include <type_traits>
#include <string>

namespace plapoint
{
namespace opencl
{
namespace detail
{

/// Bound serial work assigned to one OpenCL work-item to avoid driver watchdog resets.
inline constexpr int maximumSequentialReductionItems = 262144;

using ::plamatrix::internal::opencl::byteSize;
using ::plamatrix::internal::opencl::CommandQueue;
using ::plamatrix::internal::opencl::CompiledKernel;
using ::plamatrix::internal::opencl::DeviceBuffer;
using ::plamatrix::internal::opencl::inOutVector;
using ::plamatrix::internal::opencl::inputVector;
using ::plamatrix::internal::opencl::kernelArg;
using ::plamatrix::internal::opencl::kernelBufferArg;
using ::plamatrix::internal::opencl::readVector;
using ::plamatrix::internal::opencl::requireFp64;

template <typename Scalar>
std::string realBuildOptions()
{
    return std::is_same_v<Scalar, double> ? "-DPLAPOINT_REAL_DOUBLE=1" : "";
}

} // namespace detail
} // namespace opencl
} // namespace plapoint
