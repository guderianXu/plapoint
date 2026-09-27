#include <plapoint/gpu/filter_compaction.h>

#include <plapoint/gpu/cuda_check.h>

#include <cuda_runtime.h>

#include <plamatrix/internal/ops/indexing.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/dense/matrix_view.h>
#include <plamatrix/internal/device/device_matrix.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace plapoint {
namespace gpu {
namespace {

void validateIndices(const std::vector<int>& indices, std::size_t input_size)
{
    for (int idx : indices)
    {
        if (idx < 0 || static_cast<std::size_t>(idx) >= input_size)
        {
            throw std::out_of_range("gatherPointCloudByIndices: source index out of range");
        }
    }
}

plamatrix::internal::ResidentMatrix<plamatrix::Index> uploadIndices(
    const std::vector<int>& indices, plamatrix::internal::ExecutionContext& context)
{
    plamatrix::Matrix<plamatrix::Index, plamatrix::Dynamic, plamatrix::Dynamic> host_indices(static_cast<plamatrix::Index>(indices.size()), 1);
    for (std::size_t row = 0; row < indices.size(); ++row)
    {
        host_indices(static_cast<plamatrix::Index>(row), 0) =
            static_cast<plamatrix::Index>(indices[row]);
    }
    return plamatrix::internal::ResidentMatrix<plamatrix::Index>::copyFrom(host_indices, context);
}

template <typename T>
plamatrix::internal::ResidentMatrix<T> enqueueGatherRows(
    const plamatrix::internal::ResidentMatrix<T>& input,
    plamatrix::internal::ConstMatrixView<plamatrix::Index, plamatrix::internal::Device::GPU> indices,
    plamatrix::internal::IndexingWorkspace& workspace,
    cudaStream_t stream)
{
    plamatrix::internal::ResidentMatrix<T> output(indices.rows(), input.cols(), input.context());
    plamatrix::internal::gatherRowsAsync(input.template view<plamatrix::internal::Device::GPU>(), indices,
                              output.template view<plamatrix::internal::Device::GPU>(), workspace, stream);
    return output;
}

template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> gatherPointCloudByIndicesImpl(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& input,
    const std::vector<int>& indices)
{
    input.validate();
    validateIndices(indices, input.size());

    if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("gatherPointCloudByIndices: input size exceeds int range");
    }
    if (indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("gatherPointCloudByIndices: output size exceeds int range");
    }

    auto d_indices = uploadIndices(indices, *input.executionContext());
    plamatrix::internal::IndexingWorkspace indexing_workspace;

    auto points = enqueueGatherRows(
        input.points(), d_indices.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, nullptr);

    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> output(std::move(points), input.executionContext());
    if (input.hasNormals())
    {
        auto normals = enqueueGatherRows(
            *input.normals(), d_indices.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, nullptr);
        output.setNormals(std::move(normals));
    }
    if (input.hasColors())
    {
        auto colors = enqueueGatherRows(
            *input.colors(), d_indices.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, nullptr);
        output.setColors(std::move(colors));
    }
    if (input.hasIntensities())
    {
        auto intensities = enqueueGatherRows(
            *input.intensities(), d_indices.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, nullptr);
        output.setIntensities(std::move(intensities));
    }
    if (input.hasScalarFields())
    {
        auto scalar_fields = enqueueGatherRows(
            *input.scalarFields(), d_indices.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, nullptr);
        output.setScalarFields(input.scalarFieldNames(), std::move(scalar_fields));
    }
    if (input.hasPointAlignedTextureCoords())
    {
        auto texture_coords = enqueueGatherRows(
            *input.textureCoords(), d_indices.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, nullptr);
        output.setTextureCoords(std::move(texture_coords));
        output.setMaterialLibraryFile(input.materialLibraryFile());
        output.setTextureImageFile(input.textureImageFile());
    }

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    indexing_workspace.checkStatus("gatherPointCloudByIndices");
    indexing_workspace.closeAsyncAllocation();
    return output;
}

template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> compactPointCloudByKeepMaskImpl(
    const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& input,
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask,
    cudaStream_t stream)
{
    input.validate();
    if (keep_mask.rows() != input.points().rows() || keep_mask.cols() != 1)
    {
        throw std::invalid_argument(
            "compactPointCloudByKeepMask: keep mask must have shape point_count x 1");
    }
    keep_mask.validateContext(*input.executionContext());

    plamatrix::internal::IndexingWorkspace indexing_workspace;
    const auto point_count = input.points().rows();
    auto& context = *input.executionContext();
    plamatrix::internal::ResidentMatrix<Scalar> capacity_points(point_count, 3, context);
    plamatrix::internal::ResidentMatrix<plamatrix::Index> capacity_indices(point_count, 1, context);
    plamatrix::internal::ResidentMatrix<plamatrix::Index> selected_count(1, 1, context);
    plamatrix::internal::compactRowsAsync(
        input.points().template view<plamatrix::internal::Device::GPU>(), keep_mask.template view<plamatrix::internal::Device::GPU>(),
        capacity_points.template view<plamatrix::internal::Device::GPU>(), capacity_indices.template view<plamatrix::internal::Device::GPU>(),
        selected_count.template view<plamatrix::internal::Device::GPU>(), indexing_workspace, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    indexing_workspace.checkStatus("compactPointCloudByKeepMask");
    plamatrix::Index count = 0;
    selected_count.copyToHost(&count, 1);
    const auto selected_indices = plamatrix::internal::ConstMatrixView<plamatrix::Index, plamatrix::internal::Device::GPU>(
        capacity_indices.data(), count, 1, 1, count == 0 ? 1 : count);
    auto points = enqueueGatherRows(input.points(), selected_indices, indexing_workspace, stream);
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> output(std::move(points), input.executionContext());

    if (input.hasNormals())
    {
        auto normals = enqueueGatherRows(
            *input.normals(), selected_indices, indexing_workspace, stream);
        output.setNormals(std::move(normals));
    }
    if (input.hasColors())
    {
        auto colors = enqueueGatherRows(
            *input.colors(), selected_indices, indexing_workspace, stream);
        output.setColors(std::move(colors));
    }
    if (input.hasIntensities())
    {
        auto intensities = enqueueGatherRows(
            *input.intensities(), selected_indices, indexing_workspace, stream);
        output.setIntensities(std::move(intensities));
    }
    if (input.hasScalarFields())
    {
        auto scalar_fields = enqueueGatherRows(
            *input.scalarFields(), selected_indices, indexing_workspace, stream);
        output.setScalarFields(input.scalarFieldNames(), std::move(scalar_fields));
    }
    if (input.hasPointAlignedTextureCoords())
    {
        auto texture_coords = enqueueGatherRows(
            *input.textureCoords(), selected_indices, indexing_workspace, stream);
        output.setTextureCoords(std::move(texture_coords));
        output.setMaterialLibraryFile(input.materialLibraryFile());
        output.setTextureImageFile(input.textureImageFile());
    }

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    indexing_workspace.checkStatus("compactPointCloudByKeepMask");
    indexing_workspace.closeAsyncAllocation();
    return output;
}

} // namespace

plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> gatherPointCloudByIndices(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& input,
    const std::vector<int>& indices)
{
    return gatherPointCloudByIndicesImpl(input, indices);
}

plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> gatherPointCloudByIndices(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& input,
    const std::vector<int>& indices)
{
    return gatherPointCloudByIndicesImpl(input, indices);
}

plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> compactPointCloudByKeepMask(
    const plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU>& input,
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask,
    cudaStream_t stream)
{
    return compactPointCloudByKeepMaskImpl(input, keep_mask, stream);
}

plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> compactPointCloudByKeepMask(
    const plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU>& input,
    const plamatrix::internal::ResidentMatrix<std::uint8_t>& keep_mask,
    cudaStream_t stream)
{
    return compactPointCloudByKeepMaskImpl(input, keep_mask, stream);
}

} // namespace gpu
} // namespace plapoint
