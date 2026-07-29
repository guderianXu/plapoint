#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include <plamatrix/ops/indexing.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/marching_cubes.h>
#include <plapoint/mesh/marching_cubes.h>

namespace plapoint
{
namespace gpu
{
namespace
{

#include "marching_cubes_gpu_kernels.cuh"

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right, const char* label)
{
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right)
    {
        throw std::invalid_argument(label);
    }
    return left * right;
}

template <typename Scalar>
plamatrix::Index validateArguments(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& field,
    int nx,
    int ny,
    int nz,
    const plamatrix::Vec3<Scalar>& minimum,
    const plamatrix::Vec3<Scalar>& maximum,
    Scalar iso,
    cudaStream_t stream)
{
    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        throw std::invalid_argument("marchingCubes GPU resolution must be positive");
    }
    if (!std::isfinite(static_cast<double>(minimum.x))
        || !std::isfinite(static_cast<double>(minimum.y))
        || !std::isfinite(static_cast<double>(minimum.z))
        || !std::isfinite(static_cast<double>(maximum.x))
        || !std::isfinite(static_cast<double>(maximum.y))
        || !std::isfinite(static_cast<double>(maximum.z))
        || !(minimum.x < maximum.x && minimum.y < maximum.y && minimum.z < maximum.z))
    {
        throw std::invalid_argument("marchingCubes GPU bounds must be finite and increasing");
    }
    if (!std::isfinite(static_cast<double>(iso)))
    {
        throw std::invalid_argument("marchingCubes GPU iso value must be finite");
    }
    const auto sample_count = checkedMultiply(
        checkedMultiply(static_cast<std::uint64_t>(nx) + 1,
                        static_cast<std::uint64_t>(ny) + 1,
                        "marchingCubes GPU sample count overflows"),
        static_cast<std::uint64_t>(nz) + 1,
        "marchingCubes GPU sample count overflows");
    const auto cube_count = checkedMultiply(
        checkedMultiply(static_cast<std::uint64_t>(nx), static_cast<std::uint64_t>(ny),
                        "marchingCubes GPU cube count overflows"),
        static_cast<std::uint64_t>(nz), "marchingCubes GPU cube count overflows");
    const auto max_index = static_cast<std::uint64_t>(
        std::numeric_limits<plamatrix::Index>::max());
    const auto max_cube_count = static_cast<std::uint64_t>(
        std::numeric_limits<int>::max() / 15);
    if (sample_count > max_index || cube_count > max_cube_count)
    {
        throw std::invalid_argument("marchingCubes GPU resolution is too large");
    }
    if (field.cols() != 1 || field.rows() != static_cast<plamatrix::Index>(sample_count))
    {
        throw std::invalid_argument(
            "marchingCubes GPU field must have shape ((nx+1)*(ny+1)*(nz+1)) x 1");
    }
    if (field.isAsyncAllocation() && field.asyncAllocationStream() != stream)
    {
        throw std::logic_error(
            "marchingCubes GPU field must use the stream that owns its async allocation");
    }
    return static_cast<plamatrix::Index>(cube_count);
}

template <typename Scalar>
void closeMatrix(plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& matrix)
{
    matrix.closeAsyncAllocation();
}

} // anonymous namespace

namespace marching_cubes_detail
{

struct PointCloudAccess
{
    template <typename Scalar>
    static void attachGeneratedFaces(
        PointCloud<Scalar, plamatrix::Device::GPU>& cloud,
        plamatrix::DenseMatrix<int, plamatrix::Device::GPU>&& faces)
    {
        if (faces.cols() != 3 || cloud._points.cols() != 3
            || cloud._points.rows() % 3 != 0
            || faces.rows() != cloud._points.rows() / 3)
        {
            throw std::logic_error("marchingCubes GPU generated inconsistent mesh storage");
        }
        cloud._faces = std::make_unique<
            plamatrix::DenseMatrix<int, plamatrix::Device::GPU>>(std::move(faces));
    }
};

struct WorkspaceAccess
{
    template <typename Scalar>
    static void bind(
        MarchingCubesGpuWorkspace<Scalar>& workspace,
        plamatrix::Index cube_count,
        cudaStream_t stream)
    {
        if (workspace._hasStream && workspace._stream != stream)
        {
            throw std::logic_error(
                "MarchingCubesGpuWorkspace cannot be reused on another stream; close it first");
        }
        if (workspace._capacityCubes < cube_count)
        {
            auto counts = plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
                ::uninitializedAsync(cube_count, 1, stream);
            auto offsets = plamatrix::DenseMatrix<plamatrix::Index, plamatrix::Device::GPU>
                ::uninitializedAsync(cube_count, 1, stream);
            auto status = plamatrix::DenseMatrix<int, plamatrix::Device::GPU>
                ::uninitializedAsync(1, 1, stream);
            if (workspace._hasStream)
            {
                workspace._triangleCounts.closeAsyncAllocation();
                workspace._triangleOffsets.closeAsyncAllocation();
                workspace._status.closeAsyncAllocation();
            }
            workspace._triangleCounts = std::move(counts);
            workspace._triangleOffsets = std::move(offsets);
            workspace._status = std::move(status);
            workspace._capacityCubes = cube_count;
        }
        workspace._stream = stream;
        workspace._hasStream = true;
    }

    template <typename Scalar>
    static auto& counts(MarchingCubesGpuWorkspace<Scalar>& workspace)
    {
        return workspace._triangleCounts;
    }

    template <typename Scalar>
    static auto& offsets(MarchingCubesGpuWorkspace<Scalar>& workspace)
    {
        return workspace._triangleOffsets;
    }

    template <typename Scalar>
    static auto& status(MarchingCubesGpuWorkspace<Scalar>& workspace)
    {
        return workspace._status;
    }

    template <typename Scalar>
    static auto& scan(MarchingCubesGpuWorkspace<Scalar>& workspace)
    {
        return workspace._scanWorkspace;
    }
};

} // namespace marching_cubes_detail

template <typename Scalar>
MarchingCubesGpuWorkspace<Scalar>::~MarchingCubesGpuWorkspace() noexcept
{
    if (_hasStream)
    {
        static_cast<void>(cudaStreamSynchronize(_stream));
    }
    try
    {
        closeAsyncAllocation();
    }
    catch (...)
    {
    }
}

template <typename Scalar>
MarchingCubesGpuWorkspace<Scalar>::MarchingCubesGpuWorkspace(
    MarchingCubesGpuWorkspace&& other) noexcept
    : _triangleCounts(std::move(other._triangleCounts))
    , _triangleOffsets(std::move(other._triangleOffsets))
    , _status(std::move(other._status))
    , _scanWorkspace(std::move(other._scanWorkspace))
    , _capacityCubes(other._capacityCubes)
    , _stream(other._stream)
    , _hasStream(other._hasStream)
{
    other._capacityCubes = 0;
    other._stream = nullptr;
    other._hasStream = false;
}

template <typename Scalar>
MarchingCubesGpuWorkspace<Scalar>& MarchingCubesGpuWorkspace<Scalar>::operator=(
    MarchingCubesGpuWorkspace&& other) noexcept
{
    if (this != &other)
    {
        if (_hasStream)
        {
            static_cast<void>(cudaStreamSynchronize(_stream));
        }
        try
        {
            closeAsyncAllocation();
        }
        catch (...)
        {
        }
        _triangleCounts = std::move(other._triangleCounts);
        _triangleOffsets = std::move(other._triangleOffsets);
        _status = std::move(other._status);
        _scanWorkspace = std::move(other._scanWorkspace);
        _capacityCubes = other._capacityCubes;
        _stream = other._stream;
        _hasStream = other._hasStream;
        other._capacityCubes = 0;
        other._stream = nullptr;
        other._hasStream = false;
    }
    return *this;
}

template <typename Scalar>
void MarchingCubesGpuWorkspace<Scalar>::closeAsyncAllocation()
{
    if (_hasStream)
    {
        const cudaError_t status = cudaStreamQuery(_stream);
        if (status == cudaErrorNotReady)
        {
            throw std::logic_error(
                "MarchingCubesGpuWorkspace::closeAsyncAllocation requires stream synchronization");
        }
        PLAPOINT_CHECK_CUDA(status);
    }
    _scanWorkspace.closeAsyncAllocation();
    closeMatrix(_triangleCounts);
    closeMatrix(_triangleOffsets);
    closeMatrix(_status);
    _capacityCubes = 0;
    _stream = nullptr;
    _hasStream = false;
}

template <typename Scalar>
PointCloud<Scalar, plamatrix::Device::GPU> marchingCubes(
    const plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>& field,
    int nx,
    int ny,
    int nz,
    const plamatrix::Vec3<Scalar>& min_corner,
    const plamatrix::Vec3<Scalar>& max_corner,
    Scalar iso,
    MarchingCubesGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    const plamatrix::Index cube_count = validateArguments(
        field, nx, ny, nz, min_corner, max_corner, iso, stream);
    initializeTriangleTable();
    marching_cubes_detail::WorkspaceAccess::bind(workspace, cube_count, stream);
    auto& counts = marching_cubes_detail::WorkspaceAccess::counts(workspace);
    auto& offsets = marching_cubes_detail::WorkspaceAccess::offsets(workspace);
    auto& status = marching_cubes_detail::WorkspaceAccess::status(workspace);

    PLAPOINT_CHECK_CUDA(cudaMemsetAsync(status.data(), 0, sizeof(int), stream));
    const int grid_size = static_cast<int>((cube_count + kBlockSize - 1) / kBlockSize);
    classifyCubesKernel<<<grid_size, kBlockSize, 0, stream>>>(
        field.data(), nx, ny, nz, iso, counts.data(), status.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    plamatrix::exclusiveScan(
        counts, offsets, marching_cubes_detail::WorkspaceAccess::scan(workspace), stream);

    int host_status = 0;
    plamatrix::Index last_count = 0;
    plamatrix::Index last_offset = 0;
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        &host_status, status.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        &last_count, counts.data() + cube_count - 1, sizeof(plamatrix::Index),
        cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        &last_offset, offsets.data() + cube_count - 1, sizeof(plamatrix::Index),
        cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (host_status != 0)
    {
        throw std::invalid_argument("marchingCubes GPU field values must be finite");
    }

    const plamatrix::Index face_count = last_offset + last_count;
    const plamatrix::Index vertex_count = face_count * 3;
    auto points = plamatrix::DenseMatrix<Scalar, plamatrix::Device::GPU>
        ::uninitialized(vertex_count, 3);
    auto faces = plamatrix::DenseMatrix<int, plamatrix::Device::GPU>
        ::uninitialized(face_count, 3);
    if (face_count > 0)
    {
        const Scalar dx = (max_corner.x - min_corner.x) / Scalar(nx);
        const Scalar dy = (max_corner.y - min_corner.y) / Scalar(ny);
        const Scalar dz = (max_corner.z - min_corner.z) / Scalar(nz);
        emitTrianglesKernel<<<grid_size, kBlockSize, 0, stream>>>(
            field.data(), nx, ny, nz,
            min_corner.x, min_corner.y, min_corner.z,
            dx, dy, dz, iso, offsets.data(), points.data(), faces.data(),
            vertex_count, face_count);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    PointCloud<Scalar, plamatrix::Device::GPU> result(std::move(points));
    marching_cubes_detail::PointCloudAccess::attachGeneratedFaces(result, std::move(faces));
    return result;
}

template class MarchingCubesGpuWorkspace<float>;
template class MarchingCubesGpuWorkspace<double>;
template PointCloud<float, plamatrix::Device::GPU> marchingCubes<float>(
    const plamatrix::DenseMatrix<float, plamatrix::Device::GPU>&,
    int, int, int, const plamatrix::Vec3<float>&, const plamatrix::Vec3<float>&,
    float, MarchingCubesGpuWorkspace<float>&, cudaStream_t);
template PointCloud<double, plamatrix::Device::GPU> marchingCubes<double>(
    const plamatrix::DenseMatrix<double, plamatrix::Device::GPU>&,
    int, int, int, const plamatrix::Vec3<double>&, const plamatrix::Vec3<double>&,
    double, MarchingCubesGpuWorkspace<double>&, cudaStream_t);

} // namespace gpu
} // namespace plapoint
