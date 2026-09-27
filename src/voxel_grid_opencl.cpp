#include <plapoint/opencl/preprocessing.h>

#include "opencl_preprocessing_detail.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <plamatrix/internal/core/device.h>

namespace plapoint
{
namespace opencl
{
namespace
{

void updateMean(long double& mean, long double value, int count)
{
    const long double weight = 1.0L / static_cast<long double>(count);
    mean += (value - mean) * weight;
}

template <typename Scalar>
Scalar checkedCentroid(long double centroid)
{
    if (!std::isfinite(centroid)
        || centroid < -static_cast<long double>(std::numeric_limits<Scalar>::max())
        || centroid > static_cast<long double>(std::numeric_limits<Scalar>::max()))
    {
        throw std::out_of_range("VoxelGrid: centroid is outside scalar range");
    }
    return static_cast<Scalar>(centroid);
}

template <typename Attribute>
Attribute roundedAttribute(long double value)
{
    if (!std::isfinite(value))
    {
        throw std::out_of_range("VoxelGrid: attribute mean is not finite");
    }
    const long double rounded = std::round(value);
    const long double lower = static_cast<long double>(std::numeric_limits<Attribute>::min());
    const long double upper = static_cast<long double>(std::numeric_limits<Attribute>::max());
    return static_cast<Attribute>(std::clamp(rounded, lower, upper));
}

template <typename Scalar>
void setAveragedAttributes(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& input,
    const std::vector<int>& sorted_indices,
    const std::vector<int>& offsets,
    const std::vector<int>& counts,
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& output)
{
    const bool have_normals = input.hasNormals();
    const bool have_colors = input.hasColors();
    const bool have_intensities = input.hasIntensities();
    const bool have_scalar_fields = input.hasScalarFields();
    const auto* input_normals = input.normals();
    const auto* input_colors = input.colors();
    const auto* input_intensities = input.intensities();
    const auto* input_scalar_fields = input.scalarFields();
    const auto scalar_field_count = static_cast<plamatrix::Index>(input.scalarFieldNames().size());
    const auto voxel_count = static_cast<plamatrix::Index>(counts.size());

    std::unique_ptr<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>> normals;
    std::unique_ptr<plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>> colors;
    std::unique_ptr<plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>> intensities;
    std::unique_ptr<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>> scalar_fields;
    if (have_normals)
    {
        normals = std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(voxel_count, 3);
    }
    if (have_colors)
    {
        colors =
            std::make_unique<plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic>>(voxel_count, 3);
    }
    if (have_intensities)
    {
        intensities =
            std::make_unique<plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic>>(voxel_count, 1);
    }
    if (have_scalar_fields)
    {
        scalar_fields = std::make_unique<plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>>(
            voxel_count, scalar_field_count);
    }

    std::vector<long double> mean_scalar_fields(static_cast<std::size_t>(scalar_field_count));
    for (std::size_t voxel = 0; voxel < counts.size(); ++voxel)
    {
        long double mean_nx = 0;
        long double mean_ny = 0;
        long double mean_nz = 0;
        long double mean_r = 0;
        long double mean_g = 0;
        long double mean_b = 0;
        long double mean_intensity = 0;
        std::fill(mean_scalar_fields.begin(), mean_scalar_fields.end(), 0.0L);
        for (int item = 0; item < counts[voxel]; ++item)
        {
            const int point = sorted_indices[
                static_cast<std::size_t>(offsets[voxel] + item)];
            const int count = item + 1;
            if (have_normals)
            {
                updateMean(mean_nx, static_cast<long double>(input_normals->operator()(point, 0)), count);
                updateMean(mean_ny, static_cast<long double>(input_normals->operator()(point, 1)), count);
                updateMean(mean_nz, static_cast<long double>(input_normals->operator()(point, 2)), count);
            }
            if (have_colors)
            {
                updateMean(mean_r, static_cast<long double>(input_colors->operator()(point, 0)), count);
                updateMean(mean_g, static_cast<long double>(input_colors->operator()(point, 1)), count);
                updateMean(mean_b, static_cast<long double>(input_colors->operator()(point, 2)), count);
            }
            if (have_intensities)
            {
                updateMean(
                    mean_intensity,
                    static_cast<long double>(input_intensities->operator()(point, 0)),
                    count);
            }
            if (have_scalar_fields)
            {
                for (plamatrix::Index column = 0; column < scalar_field_count; ++column)
                {
                    updateMean(
                        mean_scalar_fields[static_cast<std::size_t>(column)],
                        static_cast<long double>(input_scalar_fields->operator()(point, column)),
                        count);
                }
            }
        }

        const auto row = static_cast<plamatrix::Index>(voxel);
        if (normals)
        {
            (*normals)(row, 0) = checkedCentroid<Scalar>(mean_nx);
            (*normals)(row, 1) = checkedCentroid<Scalar>(mean_ny);
            (*normals)(row, 2) = checkedCentroid<Scalar>(mean_nz);
        }
        if (colors)
        {
            (*colors)(row, 0) = roundedAttribute<std::uint8_t>(mean_r);
            (*colors)(row, 1) = roundedAttribute<std::uint8_t>(mean_g);
            (*colors)(row, 2) = roundedAttribute<std::uint8_t>(mean_b);
        }
        if (intensities)
        {
            (*intensities)(row, 0) = roundedAttribute<std::uint16_t>(mean_intensity);
        }
        if (scalar_fields)
        {
            for (plamatrix::Index column = 0; column < scalar_field_count; ++column)
            {
                (*scalar_fields)(row, column) = checkedCentroid<Scalar>(
                    mean_scalar_fields[static_cast<std::size_t>(column)]);
            }
        }
    }

    if (normals) output.setNormals(std::move(*normals));
    if (colors) output.setColors(std::move(*colors));
    if (intensities) output.setIntensities(std::move(*intensities));
    if (scalar_fields)
    {
        output.setScalarFields(input.scalarFieldNames(), std::move(*scalar_fields));
    }
}

template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> voxelDownsampleImpl(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& input,
    Scalar leaf_x,
    Scalar leaf_y,
    Scalar leaf_z)
{
    if (!std::isfinite(leaf_x) || !std::isfinite(leaf_y) || !std::isfinite(leaf_z)
        || leaf_x <= Scalar(0) || leaf_y <= Scalar(0) || leaf_z <= Scalar(0))
    {
        throw std::invalid_argument("OpenCL VoxelGrid: leaf size must be finite and positive");
    }
    if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("OpenCL VoxelGrid: point count exceeds int range");
    }
    if (input.size() == 0)
    {
        plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> output(0);
        const std::vector<int> empty;
        setAveragedAttributes(input, empty, empty, empty, output);
        return output;
    }

    const auto points = detail::rowMajorPoints(input);
    struct VoxelEntry
    {
        detail::CellCoord key;
        int point;
    };
    std::vector<VoxelEntry> entries;
    entries.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        const Scalar x = points[index * 3u];
        const Scalar y = points[index * 3u + 1u];
        const Scalar z = points[index * 3u + 2u];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            throw std::invalid_argument("OpenCL VoxelGrid: points must be finite");
        }
        entries.push_back({
            {detail::checkedVoxelCell(x, leaf_x), detail::checkedVoxelCell(y, leaf_y),
             detail::checkedVoxelCell(z, leaf_z)},
            static_cast<int>(index)});
    }
    std::sort(entries.begin(), entries.end(), [](const VoxelEntry& lhs, const VoxelEntry& rhs)
    {
        return lhs.key == rhs.key ? lhs.point < rhs.point : lhs.key < rhs.key;
    });
    std::vector<int> sorted_indices;
    std::vector<int> offsets;
    std::vector<int> counts;
    sorted_indices.reserve(entries.size());
    for (std::size_t begin = 0; begin < entries.size();)
    {
        std::size_t end = begin + 1u;
        while (end < entries.size() && entries[end].key == entries[begin].key) ++end;
        if (end - begin > static_cast<std::size_t>(detail::maximumSequentialReductionItems))
        {
            throw std::runtime_error(
                "OpenCL VoxelGrid rejected a voxel that exceeds the bounded serial reduction size");
        }
        offsets.push_back(static_cast<int>(begin));
        counts.push_back(static_cast<int>(end - begin));
        for (std::size_t cursor = begin; cursor < end; ++cursor)
        {
            sorted_indices.push_back(entries[cursor].point);
        }
        begin = end;
    }

    auto& runtime = detail::OpenClRuntime::instance();
    detail::requireFp64<Scalar>(runtime);
    detail::CommandQueue queue(runtime.createQueue());
    auto point_buffer = detail::inputVector(runtime, points);
    auto index_buffer = detail::inputVector(runtime, sorted_indices);
    auto offset_buffer = detail::inputVector(runtime, offsets);
    auto count_buffer = detail::inputVector(runtime, counts);
    std::vector<Scalar> centroids(counts.size() * 3u);
    detail::DeviceBuffer centroid_buffer(
        runtime.context(), CL_MEM_WRITE_ONLY, detail::byteSize<Scalar>(centroids.size()));
    const cl_program program = runtime.program(
        detail::programKey<Scalar>("preprocessing_voxel"),
        detail::voxelKernelSource,
        detail::realBuildOptions<Scalar>());
    detail::CompiledKernel kernel(program, "voxelCentroids");
    detail::kernelBufferArg(kernel, 0, point_buffer);
    detail::kernelBufferArg(kernel, 1, index_buffer);
    detail::kernelBufferArg(kernel, 2, offset_buffer);
    detail::kernelBufferArg(kernel, 3, count_buffer);
    const int voxel_count = static_cast<int>(counts.size());
    detail::kernelArg(kernel, 4, voxel_count);
    detail::kernelBufferArg(kernel, 5, centroid_buffer);
    const std::size_t global_size = counts.size();
    detail::checkOpenCl(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr),
                        "clEnqueueNDRangeKernel(voxelCentroids)");
    detail::readVector(queue, centroid_buffer, centroids);

    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> output_points(voxel_count, 3);
    for (int row = 0; row < voxel_count; ++row)
    {
        output_points(row, 0) = centroids[static_cast<std::size_t>(row) * 3u];
        output_points(row, 1) = centroids[static_cast<std::size_t>(row) * 3u + 1u];
        output_points(row, 2) = centroids[static_cast<std::size_t>(row) * 3u + 2u];
    }
    if (!input.hasNormals() && !input.hasColors() && !input.hasIntensities()
        && !input.hasScalarFields())
    {
        return plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>(std::move(output_points));
    }

    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> output(std::move(output_points));
    setAveragedAttributes(input, sorted_indices, offsets, counts, output);
    return output;
}

} // namespace

plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU> voxelDownsample(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>& input,
    float leaf_x,
    float leaf_y,
    float leaf_z)
{
    return voxelDownsampleImpl(input, leaf_x, leaf_y, leaf_z);
}

plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU> voxelDownsample(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::CPU>& input,
    double leaf_x,
    double leaf_y,
    double leaf_z)
{
    return voxelDownsampleImpl(input, leaf_x, leaf_y, leaf_z);
}

} // namespace opencl
} // namespace plapoint
