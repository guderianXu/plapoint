#pragma once

#include "opencl_runtime_internal.h"

#include <plamatrix/opencl/execution.h>

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

using ::plamatrix::opencl::byteSize;
using ::plamatrix::opencl::CommandQueue;
using ::plamatrix::opencl::CompiledKernel;
using ::plamatrix::opencl::DeviceBuffer;
using ::plamatrix::opencl::inOutVector;
using ::plamatrix::opencl::inputVector;
using ::plamatrix::opencl::kernelArg;
using ::plamatrix::opencl::kernelBufferArg;
using ::plamatrix::opencl::readVector;
using ::plamatrix::opencl::requireFp64;

template <typename Scalar>
std::string realBuildOptions()
{
    return std::is_same_v<Scalar, double> ? "-DPLAPOINT_REAL_DOUBLE=1" : "";
}

} // namespace detail
} // namespace opencl
} // namespace plapoint
