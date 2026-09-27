#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/backend.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>
#include <plamatrix/internal/device/device_matrix.h>

#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/marching_cubes.h>
#include <plapoint/mesh/marching_cubes.h>

#ifdef PLAPOINT_WITH_CUDA

namespace
{

    template <typename Scalar> using CpuMatrix = plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>;

    template <typename Scalar> plamatrix::internal::ResidentMatrix<Scalar> toResident(const CpuMatrix<Scalar>& host)
    {
        static auto context =
            plamatrix::internal::ExecutionContext::createShared({plamatrix::internal::Backend::Cuda, 0});
        return plamatrix::internal::ResidentMatrix<Scalar>::copyFrom(host, context);
    }

    template <typename Scalar, typename Function>
    CpuMatrix<Scalar> sampleField(int nx,
                                  int ny,
                                  int nz,
                                  const std::array<Scalar, 3>& min_corner,
                                  const std::array<Scalar, 3>& max_corner,
                                  Function function)
    {
        const int sx = nx + 1;
        const int sy = ny + 1;
        CpuMatrix<Scalar> field(sx * sy * (nz + 1), 1);
        for (int iz = 0; iz <= nz; ++iz)
        {
            for (int iy = 0; iy <= ny; ++iy)
            {
                for (int ix = 0; ix <= nx; ++ix)
                {
                    const Scalar x = min_corner[0] + (max_corner[0] - min_corner[0]) * Scalar(ix) / Scalar(nx);
                    const Scalar y = min_corner[1] + (max_corner[1] - min_corner[1]) * Scalar(iy) / Scalar(ny);
                    const Scalar z = min_corner[2] + (max_corner[2] - min_corner[2]) * Scalar(iz) / Scalar(nz);
                    field(iz * sy * sx + iy * sx + ix, 0) = function(x, y, z);
                }
            }
        }
        return field;
    }

    template <typename Scalar>
    double meshArea(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& mesh)
    {
        const auto* faces = mesh.faces();
        if (!faces)
        {
            return 0.0;
        }

        double area = 0.0;
        for (plamatrix::Index face = 0; face < faces->rows(); ++face)
        {
            const auto ia = static_cast<plamatrix::Index>((*faces)(face, 0));
            const auto ib = static_cast<plamatrix::Index>((*faces)(face, 1));
            const auto ic = static_cast<plamatrix::Index>((*faces)(face, 2));
            const double ax = mesh.points()(ia, 0);
            const double ay = mesh.points()(ia, 1);
            const double az = mesh.points()(ia, 2);
            const double ux = static_cast<double>(mesh.points()(ib, 0)) - ax;
            const double uy = static_cast<double>(mesh.points()(ib, 1)) - ay;
            const double uz = static_cast<double>(mesh.points()(ib, 2)) - az;
            const double vx = static_cast<double>(mesh.points()(ic, 0)) - ax;
            const double vy = static_cast<double>(mesh.points()(ic, 1)) - ay;
            const double vz = static_cast<double>(mesh.points()(ic, 2)) - az;
            const double cx = uy * vz - uz * vy;
            const double cy = uz * vx - ux * vz;
            const double cz = ux * vy - uy * vx;
            area += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        }
        return area;
    }

    template <typename Scalar>
    void expectSameMesh(const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& lhs,
                        const plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>& rhs)
    {
        ASSERT_EQ(lhs.points().rows(), rhs.points().rows());
        ASSERT_EQ(lhs.hasFaces(), rhs.hasFaces());
        for (plamatrix::Index row = 0; row < lhs.points().rows(); ++row)
        {
            for (plamatrix::Index col = 0; col < 3; ++col)
            {
                EXPECT_EQ(lhs.points()(row, col), rhs.points()(row, col));
            }
        }
        ASSERT_TRUE(lhs.faces());
        ASSERT_TRUE(rhs.faces());
        ASSERT_EQ(lhs.faces()->rows(), rhs.faces()->rows());
        for (plamatrix::Index row = 0; row < lhs.faces()->rows(); ++row)
        {
            for (plamatrix::Index col = 0; col < 3; ++col)
            {
                EXPECT_EQ((*lhs.faces())(row, col), (*rhs.faces())(row, col));
            }
        }
    }

    template <typename Scalar> class MarchingCubesGpuTypedTest : public ::testing::Test
    {
    };

    using MarchingCubesScalarTypes = ::testing::Types<float, double>;
    TYPED_TEST_SUITE(MarchingCubesGpuTypedTest, MarchingCubesScalarTypes);

#define SKIP_IF_NO_GPU() \
    do \
    { \
        if (!plapoint::gpu::hasUsableCudaDevice()) \
        { \
            GTEST_SKIP() << "No CUDA-capable device detected"; \
        } \
    } while (0)

TYPED_TEST(MarchingCubesGpuTypedTest, EmptyAndSingleCellPlaneHaveExpectedTopology)
{
    SKIP_IF_NO_GPU();
    using Scalar = TypeParam;
    const std::array<Scalar, 3> minimum{Scalar(0), Scalar(0), Scalar(0)};
    const std::array<Scalar, 3> maximum{Scalar(1), Scalar(1), Scalar(1)};
    plapoint::gpu::MarchingCubesGpuWorkspace<Scalar> workspace;

    auto empty_field = toResident(sampleField<Scalar>(1, 1, 1, minimum, maximum,
        [](Scalar, Scalar, Scalar) { return Scalar(1); }));
    const auto empty = plapoint::gpu::marchingCubes(
        empty_field, 1, 1, 1, minimum, maximum, Scalar(0), workspace).toCpu();
    EXPECT_EQ(empty.size(), 0u);
    ASSERT_TRUE(empty.faces());
    EXPECT_EQ(empty.faces()->rows(), 0);

    auto plane_field = toResident(sampleField<Scalar>(1, 1, 1, minimum, maximum,
        [](Scalar x, Scalar, Scalar) { return x - Scalar(0.5); }));
    const auto plane = plapoint::gpu::marchingCubes(
        plane_field, 1, 1, 1, minimum, maximum, Scalar(0), workspace).toCpu();
    ASSERT_EQ(plane.size(), 6u);
    ASSERT_TRUE(plane.faces());
    ASSERT_EQ(plane.faces()->rows(), 2);
    for (plamatrix::Index row = 0; row < plane.points().rows(); ++row)
    {
        EXPECT_NEAR(plane.points()(row, 0), Scalar(0.5), Scalar(32) * std::numeric_limits<Scalar>::epsilon());
    }
}

TYPED_TEST(MarchingCubesGpuTypedTest, BoundaryIntersectionMatchesCpuAreaAndBounds)
{
    SKIP_IF_NO_GPU();
    using Scalar = TypeParam;
    const std::array<Scalar, 3> minimum{Scalar(0), Scalar(0), Scalar(0)};
    const std::array<Scalar, 3> maximum{Scalar(1), Scalar(1), Scalar(1)};
    auto field = toResident(sampleField<Scalar>(1, 1, 1, minimum, maximum,
        [](Scalar x, Scalar, Scalar) { return x - Scalar(1); }));
    plapoint::gpu::MarchingCubesGpuWorkspace<Scalar> workspace;
    const auto gpu_mesh = plapoint::gpu::marchingCubes(
        field, 1, 1, 1, minimum, maximum, Scalar(0), workspace).toCpu();

    plapoint::mesh::MarchingCubes<Scalar> cpu_marching_cubes;
    cpu_marching_cubes.setBounds(minimum, maximum);
    cpu_marching_cubes.setResolution(1, 1, 1);
    cpu_marching_cubes.setIsoLevel(Scalar(0));
    auto [cpu_points, cpu_faces_scalar] =
        cpu_marching_cubes.extract([](Scalar x, Scalar, Scalar) { return x - Scalar(1); });
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> cpu_faces(cpu_faces_scalar.rows(), 3);
    for (plamatrix::Index row = 0; row < cpu_faces.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < 3; ++col)
        {
            cpu_faces(row, col) = static_cast<int>(cpu_faces_scalar(row, col));
        }
    }
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> cpu_mesh(std::move(cpu_points));
    cpu_mesh.setFaces(std::move(cpu_faces));

    ASSERT_EQ(gpu_mesh.size(), cpu_mesh.size());
    EXPECT_NEAR(meshArea(gpu_mesh), meshArea(cpu_mesh), 1.0e-10);
    for (plamatrix::Index row = 0; row < gpu_mesh.points().rows(); ++row)
    {
        EXPECT_NEAR(gpu_mesh.points()(row, 0), Scalar(1), Scalar(32) * std::numeric_limits<Scalar>::epsilon());
    }
}

TYPED_TEST(MarchingCubesGpuTypedTest, SphereAgreesWithCpuSurfaceAreaAndBounds)
{
    SKIP_IF_NO_GPU();
    using Scalar = TypeParam;
    constexpr int resolution = 14;
    const std::array<Scalar, 3> minimum{Scalar(-1.5), Scalar(-1.5), Scalar(-1.5)};
    const std::array<Scalar, 3> maximum{Scalar(1.5), Scalar(1.5), Scalar(1.5)};
    const auto sphere = [](Scalar x, Scalar y, Scalar z)
    {
        return x * x + y * y + z * z - Scalar(1);
    };
    auto field = toResident(sampleField<Scalar>(resolution, resolution, resolution, minimum, maximum, sphere));
    plapoint::gpu::MarchingCubesGpuWorkspace<Scalar> workspace;
    const auto gpu_mesh = plapoint::gpu::marchingCubes(
        field, resolution, resolution, resolution, minimum, maximum, Scalar(0), workspace).toCpu();

    plapoint::mesh::MarchingCubes<Scalar> cpu_marching_cubes;
    cpu_marching_cubes.setBounds(minimum, maximum);
    cpu_marching_cubes.setResolution(resolution, resolution, resolution);
    cpu_marching_cubes.setIsoLevel(Scalar(0));
    auto [cpu_points, cpu_faces_scalar] = cpu_marching_cubes.extract(sphere);
    plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic> cpu_faces(cpu_faces_scalar.rows(), 3);
    for (plamatrix::Index row = 0; row < cpu_faces.rows(); ++row)
    {
        for (plamatrix::Index col = 0; col < 3; ++col)
        {
            cpu_faces(row, col) = static_cast<int>(cpu_faces_scalar(row, col));
        }
    }
    plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU> cpu_mesh(std::move(cpu_points));
    cpu_mesh.setFaces(std::move(cpu_faces));

    ASSERT_GT(gpu_mesh.size(), 0u);
    EXPECT_NEAR(meshArea(gpu_mesh), meshArea(cpu_mesh), meshArea(cpu_mesh) * 2.0e-5);
    for (plamatrix::Index axis = 0; axis < 3; ++axis)
    {
        Scalar gpu_min = gpu_mesh.points()(0, axis);
        Scalar gpu_max = gpu_min;
        Scalar cpu_min = cpu_mesh.points()(0, axis);
        Scalar cpu_max = cpu_min;
        for (plamatrix::Index row = 1; row < gpu_mesh.points().rows(); ++row)
        {
            gpu_min = std::min(gpu_min, gpu_mesh.points()(row, axis));
            gpu_max = std::max(gpu_max, gpu_mesh.points()(row, axis));
            cpu_min = std::min(cpu_min, cpu_mesh.points()(row, axis));
            cpu_max = std::max(cpu_max, cpu_mesh.points()(row, axis));
        }
        EXPECT_NEAR(gpu_min, cpu_min, Scalar(2.0e-5));
        EXPECT_NEAR(gpu_max, cpu_max, Scalar(2.0e-5));
    }
}

TYPED_TEST(MarchingCubesGpuTypedTest, RepeatedNonDefaultStreamOutputIsDeterministic)
{
    SKIP_IF_NO_GPU();
    using Scalar = TypeParam;
    const std::array<Scalar, 3> minimum{Scalar(-1), Scalar(-1), Scalar(-1)};
    const std::array<Scalar, 3> maximum{Scalar(1), Scalar(1), Scalar(1)};
    auto field = toResident(sampleField<Scalar>(5, 4, 3, minimum, maximum,
        [](Scalar x, Scalar y, Scalar z) { return x + Scalar(0.5) * y - Scalar(0.25) * z; }));

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        plapoint::gpu::MarchingCubesGpuWorkspace<Scalar> workspace;
        const auto first = plapoint::gpu::marchingCubes(
            field, 5, 4, 3, minimum, maximum, Scalar(0), workspace, stream).toCpu();
        const auto second = plapoint::gpu::marchingCubes(
            field, 5, 4, 3, minimum, maximum, Scalar(0), workspace, stream).toCpu();
        expectSameMesh(first, second);
        workspace.closeAsyncAllocation();
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TYPED_TEST(MarchingCubesGpuTypedTest, WorkspaceRetainsLargerCapacityAcrossSmallerCalls)
{
    SKIP_IF_NO_GPU();
    using Scalar = TypeParam;
    const std::array<Scalar, 3> minimum{Scalar(0), Scalar(0), Scalar(0)};
    const std::array<Scalar, 3> maximum{Scalar(1), Scalar(1), Scalar(1)};
    auto large_field = toResident(sampleField<Scalar>(5, 4, 3, minimum, maximum,
        [](Scalar x, Scalar y, Scalar z) { return x + y + z - Scalar(1); }));
    auto small_field = toResident(sampleField<Scalar>(1, 1, 1, minimum, maximum,
        [](Scalar x, Scalar, Scalar) { return x - Scalar(0.5); }));
    plapoint::gpu::MarchingCubesGpuWorkspace<Scalar> workspace;

    (void)plapoint::gpu::marchingCubes(
        large_field, 5, 4, 3, minimum, maximum, Scalar(0), workspace);
    ASSERT_EQ(workspace.capacityCubes(), 60);
    const auto small_mesh = plapoint::gpu::marchingCubes(
        small_field, 1, 1, 1, minimum, maximum, Scalar(0), workspace).toCpu();

    EXPECT_EQ(workspace.capacityCubes(), 60);
    EXPECT_EQ(small_mesh.points().rows(), 6);
}

TEST(MarchingCubesGpuTest, RejectsInvalidHostMetadataAndFieldShape)
{
    SKIP_IF_NO_GPU();
    using Scalar = float;
    const std::array<Scalar, 3> minimum{0, 0, 0};
    const std::array<Scalar, 3> maximum{1, 1, 1};
    plapoint::gpu::MarchingCubesGpuWorkspace<Scalar> workspace;
    auto field = toResident(CpuMatrix<Scalar>(8, 1));
    auto wrong_columns = toResident(CpuMatrix<Scalar>(4, 2));

    EXPECT_THROW((void)plapoint::gpu::marchingCubes(
        field, 0, 1, 1, minimum, maximum, 0.0f, workspace), std::invalid_argument);
    EXPECT_THROW((void)plapoint::gpu::marchingCubes(
        field, 1, 1, 1, maximum, minimum, 0.0f, workspace), std::invalid_argument);
    EXPECT_THROW((void)plapoint::gpu::marchingCubes(
        field, 1, 1, 1, minimum, maximum,
        std::numeric_limits<Scalar>::quiet_NaN(), workspace), std::invalid_argument);
    EXPECT_THROW((void)plapoint::gpu::marchingCubes(
        wrong_columns, 1, 1, 1, minimum, maximum, 0.0f, workspace), std::invalid_argument);
    try
    {
        (void)plapoint::gpu::marchingCubes(
            field, std::numeric_limits<int>::max(), 2, 2,
            minimum, maximum, 0.0f, workspace);
        FAIL() << "oversized resolution must be rejected before field-shape validation";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("resolution is too large"), std::string::npos);
    }
}

TEST(MarchingCubesGpuTest, RejectsNonFiniteDeviceFieldBeforePublishingOutput)
{
    SKIP_IF_NO_GPU();
    using Scalar = double;
    const std::array<Scalar, 3> minimum{0, 0, 0};
    const std::array<Scalar, 3> maximum{1, 1, 1};
    CpuMatrix<Scalar> host_field(8, 1);
    host_field.setConstant(Scalar(1));
    host_field(5, 0) = std::numeric_limits<Scalar>::infinity();
    auto field = toResident(host_field);
    plapoint::gpu::MarchingCubesGpuWorkspace<Scalar> workspace;

    EXPECT_THROW((void)plapoint::gpu::marchingCubes(
        field, 1, 1, 1, minimum, maximum, Scalar(0), workspace), std::invalid_argument);
}

static_assert(!std::is_copy_constructible_v<plapoint::gpu::MarchingCubesGpuWorkspace<float>>);
static_assert(!std::is_copy_assignable_v<plapoint::gpu::MarchingCubesGpuWorkspace<float>>);

} // namespace

#endif // PLAPOINT_WITH_CUDA
