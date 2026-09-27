#pragma once

namespace plapoint::detail
{

/// Downsample a CPU-owned point cloud using the requested device, returning CPU-owned output.
template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> voxelDownsample(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& input,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z,
    ProcessingDevice device,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    if (device == ProcessingDevice::CPU)
    {
        auto output = voxelDownsample(input, leaf_x, leaf_y, leaf_z);
        detail::setReport(report, device, ProcessingDevice::CPU, false);
        return output;
    }

    if (device == ProcessingDevice::Auto && !ProcessingPolicy::autoPrefersGpu(input.size()))
    {
        auto output = voxelDownsample(input, leaf_x, leaf_y, leaf_z);
        detail::setAutoCpuReport(
            report,
            ProcessingNeighborBackend::None,
            "Auto selected CPU because voxel work is below the accelerator transfer threshold");
        return output;
    }

    if (device == ProcessingDevice::CUDA || device == ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_CUDA
        if (detail::gpuIsAvailable())
        {
            try
            {
                const auto gpu_input = input.toGpu();
                const auto gpu_output = voxelDownsample(gpu_input, leaf_x, leaf_y, leaf_z);
                detail::setReport(
                    report, device, ProcessingDevice::CUDA, !fallback_reason.empty(), fallback_reason);
                return gpu_output.toCpu();
            }
            catch (const std::exception& ex)
            {
                if (device == ProcessingDevice::CUDA)
                {
                    throw;
                }
                detail::appendFallbackReason(fallback_reason, "CUDA", ex.what());
            }
        }
        else
        {
            const std::string reason = "device is not available";
            if (device == ProcessingDevice::CUDA)
            {
                throw std::runtime_error("CUDA " + reason);
            }
            detail::appendFallbackReason(fallback_reason, "CUDA", reason);
        }
#else
        const std::string reason = "PlaPoint was built without CUDA support";
        if (device == ProcessingDevice::CUDA)
        {
            throw std::runtime_error(reason);
        }
        detail::appendFallbackReason(fallback_reason, "CUDA", reason);
#endif
    }

#ifdef PLAPOINT_WITH_OPENCL
    const bool opencl_requires_cpu_attribute_aggregation =
        input.hasNormals() || input.hasColors() || input.hasIntensities() || input.hasScalarFields();
    if (device == ProcessingDevice::Auto && opencl_requires_cpu_attribute_aggregation)
    {
        auto output = voxelDownsample(input, leaf_x, leaf_y, leaf_z);
        detail::setReport(report,
                          device,
                          ProcessingDevice::CPU,
                          !fallback_reason.empty(),
                          fallback_reason);
        if (report)
        {
            report->selectionReason =
                "Auto skipped partial OpenCL voxel offload because attributes require CPU aggregation";
        }
        return output;
    }
#endif

    if (device == ProcessingDevice::OpenCL || device == ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_OPENCL
        try
        {
            opencl::requireUsableOpenClDevice();
            auto output = opencl::voxelDownsample(input, leaf_x, leaf_y, leaf_z);
            detail::setReport(
                report, device, ProcessingDevice::OpenCL,
                !fallback_reason.empty(), fallback_reason);
            return output;
        }
        catch (const std::exception& ex)
        {
            if (device == ProcessingDevice::OpenCL)
            {
                throw;
            }
            detail::appendFallbackReason(fallback_reason, "OpenCL", ex.what());
        }
#else
        const std::string reason = "PlaPoint was built without OpenCL support";
        if (device == ProcessingDevice::OpenCL)
        {
            throw std::runtime_error(reason);
        }
        detail::appendFallbackReason(fallback_reason, "OpenCL", reason);
#endif
    }

    auto output = voxelDownsample(input, leaf_x, leaf_y, leaf_z);
    detail::setReport(report, device, ProcessingDevice::CPU,
                      !fallback_reason.empty(), fallback_reason);
    return output;
}

/// Downsample a CPU-owned point cloud using cubic voxels and the requested device.
template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> voxelDownsample(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& input,
    Scalar leaf_size,
    ProcessingDevice device,
    ProcessingReport* report = nullptr)
{
    return voxelDownsample(input, leaf_size, leaf_size, leaf_size, device, report);
}

} // namespace plapoint::detail
