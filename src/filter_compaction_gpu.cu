#include <plapoint/gpu/filter_compaction.h>

#include <plapoint/gpu/cuda_check.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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

DeviceBuffer<int> uploadIndices(const std::vector<int>& indices)
{
    DeviceBuffer<int> d_indices(indices.size());
    if (!indices.empty())
    {
        const std::size_t bytes = indices.size() * sizeof(int);
        PLAPOINT_CHECK_CUDA(cudaMemcpy(d_indices.get(), indices.data(), bytes, cudaMemcpyHostToDevice));
    }
    return d_indices;
}

template <typename T>
__global__ void gatherColumnMajorKernel(
    const T* input,
    const int* indices,
    int output_count,
    int input_rows,
    int cols,
    T* output)
{
    const int linear = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    const int total = output_count * cols;
    if (linear >= total)
    {
        return;
    }

    const int row = linear % output_count;
    const int col = linear / output_count;
    const int src = indices[row];
    output[row + col * output_count] = input[src + col * input_rows];
}

template <typename T>
void gatherColumnMajor(
    const plamatrix::DenseMatrix<T, plamatrix::Device::GPU>& input,
    const DeviceBuffer<int>& d_indices,
    int output_count,
    plamatrix::DenseMatrix<T, plamatrix::Device::GPU>& output)
{
    if (output_count == 0 || input.cols() == 0)
    {
        return;
    }

    constexpr int kBlockSize = 256;
    if (input.cols() > std::numeric_limits<int>::max() / output_count)
    {
        throw std::overflow_error("gatherPointCloudByIndices: kernel item count exceeds int range");
    }
    const int total = output_count * static_cast<int>(input.cols());
    const int grid_size = (total + kBlockSize - 1) / kBlockSize;
    gatherColumnMajorKernel<T><<<grid_size, kBlockSize>>>(
        input.data(),
        d_indices.get(),
        output_count,
        static_cast<int>(input.rows()),
        static_cast<int>(input.cols()),
        output.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::GPU> gatherPointCloudByIndicesImpl(
    const PointCloud<Scalar, plamatrix::Device::GPU>& input,
    const std::vector<int>& indices)
{
    validateIndices(indices, input.size());

    if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("gatherPointCloudByIndices: input size exceeds int range");
    }
    if (indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::overflow_error("gatherPointCloudByIndices: output size exceeds int range");
    }

    const int output_count = static_cast<int>(indices.size());
    auto d_indices = uploadIndices(indices);

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> points(
        static_cast<plamatrix::Index>(indices.size()), 3);
    gatherColumnMajor(input.points(), d_indices, output_count, points);

    PointCloud<Scalar, plamatrix::Device::GPU> output(std::move(points));
    if (input.hasNormals())
    {
        plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU> normals(
            static_cast<plamatrix::Index>(indices.size()), 3);
        gatherColumnMajor(*input.normals(), d_indices, output_count, normals);
        output.setNormals(std::move(normals));
    }
    if (input.hasColors())
    {
        plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::GPU> colors(
            static_cast<plamatrix::Index>(indices.size()), 3);
        gatherColumnMajor(*input.colors(), d_indices, output_count, colors);
        output.setColors(std::move(colors));
    }
    if (input.hasIntensities())
    {
        plamatrix::DenseMatrix<std::uint16_t, plamatrix::Device::GPU> intensities(
            static_cast<plamatrix::Index>(indices.size()), 1);
        gatherColumnMajor(*input.intensities(), d_indices, output_count, intensities);
        output.setIntensities(std::move(intensities));
    }

    PLAPOINT_CHECK_CUDA(cudaDeviceSynchronize());
    return output;
}

} // namespace

PointCloud<float, plamatrix::Device::GPU> gatherPointCloudByIndices(
    const PointCloud<float, plamatrix::Device::GPU>& input,
    const std::vector<int>& indices)
{
    return gatherPointCloudByIndicesImpl(input, indices);
}

PointCloud<double, plamatrix::Device::GPU> gatherPointCloudByIndices(
    const PointCloud<double, plamatrix::Device::GPU>& input,
    const std::vector<int>& indices)
{
    return gatherPointCloudByIndicesImpl(input, indices);
}

} // namespace gpu
} // namespace plapoint
