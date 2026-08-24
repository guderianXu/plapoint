#pragma once

namespace plapoint
{

/// Remove statistical outliers from a CPU-owned point cloud using the requested device.
template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> statisticalOutlierRemoval(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    int mean_k,
    Scalar stddev_mul,
    ProcessingDevice device,
    std::vector<int>* removed_indices = nullptr,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    if (device == ProcessingDevice::CPU)
    {
        auto output = statisticalOutlierRemoval(input, mean_k, stddev_mul, removed_indices);
        detail::setReport(
            report, device, ProcessingDevice::CPU, false, {},
            ProcessingNeighborBackend::CpuKdTree);
        return output;
    }

    const auto estimated_neighbors = mean_k > 0 ? static_cast<std::size_t>(mean_k) : std::size_t(0);
    if (device == ProcessingDevice::Auto &&
        !ProcessingPolicy::autoPrefersNeighborhoodGpu(input.size(), estimated_neighbors))
    {
        auto output = statisticalOutlierRemoval(input, mean_k, stddev_mul, removed_indices);
        detail::setAutoCpuReport(
            report,
            ProcessingNeighborBackend::CpuKdTree,
            "Auto selected CPU because statistical-neighbor work is below the accelerator threshold");
        return output;
    }

    const auto indexed_k = input.size() == 0
        ? std::size_t(0)
        : std::min<std::size_t>(static_cast<std::size_t>(mean_k) + 1, input.size());
    if ((device == ProcessingDevice::CUDA || device == ProcessingDevice::Auto) && indexed_k > 32)
    {
        const std::string reason =
            "indexed statistical outlier removal requires meanK + self <= 32";
        if (device == ProcessingDevice::CUDA)
        {
            throw std::invalid_argument("CUDA " + reason);
        }
        detail::appendFallbackReason(fallback_reason, "CUDA", reason);
    }
    else if (device == ProcessingDevice::CUDA || device == ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_CUDA
        if (detail::gpuIsAvailable())
        {
            try
            {
                std::vector<int> gpu_removed;
                const auto gpu_input = input.toGpu();
                const auto gpu_output = statisticalOutlierRemoval(
                    gpu_input, mean_k, stddev_mul, removed_indices ? &gpu_removed : nullptr);
                if (removed_indices)
                {
                    *removed_indices = std::move(gpu_removed);
                }
                detail::setReport(
                    report, device, ProcessingDevice::CUDA, !fallback_reason.empty(), fallback_reason,
                    ProcessingNeighborBackend::GpuUniformGrid);
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

    if (device == ProcessingDevice::OpenCL || device == ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_OPENCL
        try
        {
            opencl::requireUsableOpenClDevice();
            const auto keep_mask = opencl::statisticalOutlierKeepMask(
                input, mean_k, stddev_mul);
            auto output = detail::selectByKeepMask(input, keep_mask, removed_indices);
            detail::setReport(
                report, device, ProcessingDevice::OpenCL,
                !fallback_reason.empty(), fallback_reason,
                ProcessingNeighborBackend::OpenClUniformGrid);
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

    auto output = statisticalOutlierRemoval(input, mean_k, stddev_mul, removed_indices);
    detail::setReport(report, device, ProcessingDevice::CPU,
                      !fallback_reason.empty(), fallback_reason,
                      ProcessingNeighborBackend::CpuKdTree);
    return output;
}

/// Remove radius outliers from a CPU-owned point cloud using the requested device.
template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::CPU> radiusOutlierRemoval(
    const PointCloud<Scalar, plamatrix::Device::CPU>& input,
    Scalar radius,
    int min_neighbors,
    ProcessingDevice device,
    std::vector<int>* removed_indices = nullptr,
    ProcessingReport* report = nullptr)
{
    std::string fallback_reason;
    if (device == ProcessingDevice::CPU)
    {
        auto output = radiusOutlierRemoval(input, radius, min_neighbors, removed_indices);
        detail::setReport(
            report, device, ProcessingDevice::CPU, false, {},
            ProcessingNeighborBackend::CpuKdTree);
        return output;
    }

    if (device == ProcessingDevice::Auto && !ProcessingPolicy::autoPrefersGpu(input.size()))
    {
        auto output = radiusOutlierRemoval(input, radius, min_neighbors, removed_indices);
        detail::setAutoCpuReport(
            report,
            ProcessingNeighborBackend::CpuKdTree,
            "Auto selected CPU because radius-neighbor work is below the accelerator transfer threshold");
        return output;
    }

    if (device == ProcessingDevice::CUDA || device == ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_CUDA
        if (detail::gpuIsAvailable())
        {
            try
            {
                std::vector<int> gpu_removed;
                const auto gpu_input = input.toGpu();
                const auto gpu_output = radiusOutlierRemoval(
                    gpu_input, radius, min_neighbors, removed_indices ? &gpu_removed : nullptr);
                if (removed_indices)
                {
                    *removed_indices = std::move(gpu_removed);
                }
                detail::setReport(
                    report, device, ProcessingDevice::CUDA, !fallback_reason.empty(), fallback_reason,
                    ProcessingNeighborBackend::GpuUniformGrid);
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

    if (device == ProcessingDevice::OpenCL || device == ProcessingDevice::Auto)
    {
#ifdef PLAPOINT_WITH_OPENCL
        try
        {
            opencl::requireUsableOpenClDevice();
            const auto keep_mask = opencl::radiusOutlierKeepMask(
                input, radius, min_neighbors);
            auto output = detail::selectByKeepMask(input, keep_mask, removed_indices);
            detail::setReport(
                report, device, ProcessingDevice::OpenCL,
                !fallback_reason.empty(), fallback_reason,
                ProcessingNeighborBackend::OpenClUniformGrid);
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

    auto output = radiusOutlierRemoval(input, radius, min_neighbors, removed_indices);
    detail::setReport(report, device, ProcessingDevice::CPU,
                      !fallback_reason.empty(), fallback_reason,
                      ProcessingNeighborBackend::CpuKdTree);
    return output;
}

} // namespace plapoint
