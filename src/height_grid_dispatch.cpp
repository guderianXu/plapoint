#include <plapoint/mesh/height_grid.h>

#include <plapoint/opencl/height_grid.h>
#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/height_grid.h>
#endif

#include <stdexcept>
#include <string>

namespace plapoint::mesh
{
namespace
{

    template <typename Scalar>
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>
    toDeviceCloud(const GeometryCloud<Scalar>& source)
    {
        source.validate();
        plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> coordinates = source.points();
        plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> cloud(std::move(coordinates));
        if (source.hasNormals())
            cloud.setNormals(*source.normals());
        if (source.hasColors())
            cloud.setColors(*source.colors());
        if (source.hasIntensities())
            cloud.setIntensities(*source.intensities());
        if (source.hasScalarFields())
            cloud.setScalarFields(source.scalarFieldNames(), *source.scalarFields());
        if (source.hasTextureCoords())
            cloud.setTextureCoords(*source.textureCoords());
        if (source.hasFaces())
            cloud.setFaces(*source.faces());
        if (source.hasFaceTextureIndices())
            cloud.setFaceTextureIndices(*source.faceTextureIndices());
        cloud.setMaterialLibraryFile(source.materialLibraryFile());
        cloud.setTextureImageFile(source.textureImageFile());
        return cloud;
    }

template <typename Scalar>
HeightGrid<Scalar> buildOnDevice(const GeometryCloud<Scalar>& cloud,
                                const HeightGridOptions<Scalar>& options,
                                ProcessingDevice device)
{
    if (device == ProcessingDevice::CPU)
    {
        return buildHeightGridCpu(cloud, options);
    }
    auto device_cloud = toDeviceCloud(cloud);
    if (device == ProcessingDevice::CUDA)
    {
#ifdef PLAPOINT_WITH_CUDA
        return gpu::buildHeightGrid(device_cloud.toGpu(), options);
#else
        throw std::runtime_error("buildHeightGrid: CUDA support is not built");
#endif
    }
    if (device == ProcessingDevice::OpenCL)
    {
        return opencl::buildHeightGrid(device_cloud, options);
    }
    throw std::invalid_argument("buildHeightGrid: invalid processing device");
}

} // namespace

template <typename Scalar>
HeightGrid<Scalar> buildHeightGrid(const GeometryCloud<Scalar>& cloud,
                                   const HeightGridOptions<Scalar>& options,
                                   ProcessingDevice device,
                                   ProcessingReport* report)
{
    ProcessingDevice selected = device;
    if (selected == ProcessingDevice::Auto)
    {
        selected = ProcessingDevice::CPU;
        if (ProcessingPolicy::autoPrefersGpu(cloud.size()))
        {
            if (isProcessingDeviceAvailable(ProcessingDevice::CUDA))
            {
                selected = ProcessingDevice::CUDA;
            }
            else if (isProcessingDeviceAvailable(ProcessingDevice::OpenCL))
            {
                selected = ProcessingDevice::OpenCL;
            }
        }
    }
    else if (!isProcessingDeviceAvailable(selected))
    {
        throw std::runtime_error("buildHeightGrid: requested processing device is unavailable");
    }

    ProcessingReport local_report;
    local_report.requestedDevice = device;
    local_report.actualDevice = selected;
    local_report.usedDevice = selected;
    if (device == ProcessingDevice::Auto)
    {
        local_report.selectionReason = selected == ProcessingDevice::CPU
            ? "Auto selected CPU for this cloud or no accelerator is available"
            : "Auto selected an available accelerator for this cloud";
    }

    try
    {
        auto grid = buildOnDevice(cloud, options, selected);
        if (report) *report = std::move(local_report);
        return grid;
    }
    catch (const std::exception& error)
    {
        if (device != ProcessingDevice::Auto || selected == ProcessingDevice::CPU)
        {
            throw;
        }
        local_report.actualDevice = ProcessingDevice::CPU;
        local_report.usedDevice = ProcessingDevice::CPU;
        local_report.usedFallback = true;
        local_report.fallbackReason = error.what();
        auto grid = buildHeightGridCpu(cloud, options);
        if (report) *report = std::move(local_report);
        return grid;
    }
}

template HeightGrid<float> buildHeightGrid(const GeometryCloud<float>&,
                                          const HeightGridOptions<float>&,
                                          ProcessingDevice, ProcessingReport*);
template HeightGrid<double> buildHeightGrid(const GeometryCloud<double>&,
                                           const HeightGridOptions<double>&,
                                           ProcessingDevice, ProcessingReport*);

} // namespace plapoint::mesh
