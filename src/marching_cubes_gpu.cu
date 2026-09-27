#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include <plamatrix/internal/ops/indexing.h>
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_matrix.h>

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
    const plamatrix::internal::ResidentMatrix<Scalar>& field,
    int nx,
    int ny,
    int nz,
    const std::array<Scalar, 3>& minimum,
    const std::array<Scalar, 3>& maximum,
    Scalar iso)
{
    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        throw std::invalid_argument("marchingCubes GPU resolution must be positive");
    }
    if (!std::isfinite(static_cast<double>(minimum[0]))
        || !std::isfinite(static_cast<double>(minimum[1]))
        || !std::isfinite(static_cast<double>(minimum[2]))
        || !std::isfinite(static_cast<double>(maximum[0]))
        || !std::isfinite(static_cast<double>(maximum[1]))
        || !std::isfinite(static_cast<double>(maximum[2]))
        || !(minimum[0] < maximum[0] && minimum[1] < maximum[1] && minimum[2] < maximum[2]))
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
    if (field.context().backend() != plamatrix::internal::Backend::Cuda)
    {
        throw std::invalid_argument("marchingCubes GPU field must reside in a CUDA context");
    }
    return static_cast<plamatrix::Index>(cube_count);
}

} // anonymous namespace

namespace marching_cubes_detail
{

struct PointCloudAccess
{
    template <typename Scalar>
    static void attachGeneratedFaces(
        plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>& cloud,
        plamatrix::internal::ResidentMatrix<int>&& faces)
    {
        if (faces.cols() != 3 || cloud._points.cols() != 3
            || cloud._points.rows() % 3 != 0
            || faces.rows() != cloud._points.rows() / 3)
        {
            throw std::logic_error("marchingCubes GPU generated inconsistent mesh storage");
        }
        cloud._faces = std::make_unique<plamatrix::internal::ResidentMatrix<int>>(std::move(faces));
    }
};

struct WorkspaceAccess
{
    template <typename Scalar>
    static void bind(
        MarchingCubesGpuWorkspace<Scalar>& workspace,
        plamatrix::Index cube_count,
        std::shared_ptr<plamatrix::internal::ExecutionContext> context,
        cudaStream_t stream)
    {
        if (workspace._hasStream && workspace._stream != stream)
        {
            throw std::logic_error(
                "MarchingCubesGpuWorkspace cannot be reused on another stream; close it first");
        }
        if (workspace._context != context || workspace._capacityCubes < cube_count)
        {
            if (workspace._hasStream)
            {
                PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
            }
            workspace._triangleCounts = std::make_unique<plamatrix::internal::ResidentMatrix<plamatrix::Index>>(
                cube_count, 1, context);
            workspace._triangleOffsets = std::make_unique<plamatrix::internal::ResidentMatrix<plamatrix::Index>>(
                cube_count, 1, context);
            workspace._status = std::make_unique<plamatrix::internal::ResidentMatrix<int>>(1, 1, context);
            workspace._context = std::move(context);
            workspace._capacityCubes = cube_count;
        }
        workspace._stream = stream;
        workspace._hasStream = true;
    }

    template <typename Scalar>
    static auto& counts(MarchingCubesGpuWorkspace<Scalar>& workspace)
    {
        return *workspace._triangleCounts;
    }

    template <typename Scalar>
    static auto& offsets(MarchingCubesGpuWorkspace<Scalar>& workspace)
    {
        return *workspace._triangleOffsets;
    }

    template <typename Scalar>
    static auto& status(MarchingCubesGpuWorkspace<Scalar>& workspace)
    {
        return *workspace._status;
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
    , _context(std::move(other._context))
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
        _context = std::move(other._context);
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
    _triangleCounts.reset();
    _triangleOffsets.reset();
    _status.reset();
    _context.reset();
    _capacityCubes = 0;
    _stream = nullptr;
    _hasStream = false;
}

template <typename Scalar>
plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> marchingCubes(
    const plamatrix::internal::ResidentMatrix<Scalar>& field,
    int nx,
    int ny,
    int nz,
    const std::array<Scalar, 3>& min_corner,
    const std::array<Scalar, 3>& max_corner,
    Scalar iso,
    MarchingCubesGpuWorkspace<Scalar>& workspace,
    cudaStream_t stream)
{
    const plamatrix::Index cube_count = validateArguments(
        field, nx, ny, nz, min_corner, max_corner, iso);
    auto context = field.contextOwner();
    if (!context)
    {
        context = plamatrix::internal::ExecutionContext::createShared(field.context().device());
    }
    initializeTriangleTable();
    marching_cubes_detail::WorkspaceAccess::bind(workspace, cube_count, context, stream);
    auto& counts = marching_cubes_detail::WorkspaceAccess::counts(workspace);
    auto& offsets = marching_cubes_detail::WorkspaceAccess::offsets(workspace);
    auto& status = marching_cubes_detail::WorkspaceAccess::status(workspace);

    PLAPOINT_CHECK_CUDA(cudaMemsetAsync(status.data(), 0, sizeof(int), stream));
    const int grid_size = static_cast<int>((cube_count + kBlockSize - 1) / kBlockSize);
    classifyCubesKernel<<<grid_size, kBlockSize, 0, stream>>>(
        field.data(), nx, ny, nz, iso, counts.data(), status.data());
    PLAPOINT_CHECK_CUDA(cudaGetLastError());
    plamatrix::internal::exclusiveScan(
        counts.template view<plamatrix::internal::Device::GPU>().asConst(),
        offsets.template view<plamatrix::internal::Device::GPU>(),
        marching_cubes_detail::WorkspaceAccess::scan(workspace), stream);

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
    plamatrix::internal::ResidentMatrix<Scalar> points(vertex_count, 3, context);
    plamatrix::internal::ResidentMatrix<int> faces(face_count, 3, context);
    if (face_count > 0)
    {
        const Scalar dx = (max_corner[0] - min_corner[0]) / Scalar(nx);
        const Scalar dy = (max_corner[1] - min_corner[1]) / Scalar(ny);
        const Scalar dz = (max_corner[2] - min_corner[2]) / Scalar(nz);
        emitTrianglesKernel<<<grid_size, kBlockSize, 0, stream>>>(
            field.data(), nx, ny, nz,
            min_corner[0], min_corner[1], min_corner[2],
            dx, dy, dz, iso, offsets.data(), points.data(), faces.data(),
            vertex_count, face_count);
        PLAPOINT_CHECK_CUDA(cudaGetLastError());
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU> result(std::move(points), context);
    marching_cubes_detail::PointCloudAccess::attachGeneratedFaces(result, std::move(faces));
    return result;
}

template class MarchingCubesGpuWorkspace<float>;
template class MarchingCubesGpuWorkspace<double>;
template plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::GPU> marchingCubes<float>(
    const plamatrix::internal::ResidentMatrix<float>&,
    int, int, int, const std::array<float, 3>&, const std::array<float, 3>&,
    float, MarchingCubesGpuWorkspace<float>&, cudaStream_t);
template plapoint::internal::DeviceCloud<double, plamatrix::internal::Device::GPU> marchingCubes<double>(
    const plamatrix::internal::ResidentMatrix<double>&,
    int, int, int, const std::array<double, 3>&, const std::array<double, 3>&,
    double, MarchingCubesGpuWorkspace<double>&, cudaStream_t);

} // namespace gpu
} // namespace plapoint
