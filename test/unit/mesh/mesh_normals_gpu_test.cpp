#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/core/execution_context.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/detail/geometry_bridge.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/mesh_normals.h>
#include <plapoint/mesh/mesh_processing.h>

namespace
{

    using CpuFloatCloud = plapoint::internal::DeviceCloud<float, plamatrix::internal::Device::CPU>;
    using FloatMatrix = plamatrix::MatrixXf;
    using IntMatrix = plamatrix::Matrix<int, plamatrix::Dynamic, plamatrix::Dynamic>;

    bool hasCudaDevice()
    {
        return plapoint::gpu::hasUsableCudaDevice();
    }

#define SKIP_IF_NO_GPU()                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!hasCudaDevice())                                                                                          \
        {                                                                                                              \
            GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU mesh test";                                 \
        }                                                                                                              \
    } while (0)

    CpuFloatCloud makeCloud(FloatMatrix&& points, IntMatrix&& faces)
    {
        CpuFloatCloud cloud(std::move(points));
        cloud.setFaces(std::move(faces));
        return cloud;
    }

    CpuFloatCloud makeRaisedFanMesh()
    {
        FloatMatrix points(5, 3);
        points.operator()(0, 0) = 0.0f;
        points.operator()(0, 1) = 0.0f;
        points.operator()(0, 2) = 1.0f;
        points.operator()(1, 0) = -1.0f;
        points.operator()(1, 1) = 0.0f;
        points.operator()(1, 2) = 0.0f;
        points.operator()(2, 0) = 0.0f;
        points.operator()(2, 1) = -1.0f;
        points.operator()(2, 2) = 0.0f;
        points.operator()(3, 0) = 1.0f;
        points.operator()(3, 1) = 0.0f;
        points.operator()(3, 2) = 0.0f;
        points.operator()(4, 0) = 0.0f;
        points.operator()(4, 1) = 1.0f;
        points.operator()(4, 2) = 0.0f;

        IntMatrix faces(4, 3);
        faces.operator()(0, 0) = 0;
        faces.operator()(0, 1) = 1;
        faces.operator()(0, 2) = 2;
        faces.operator()(1, 0) = 0;
        faces.operator()(1, 1) = 2;
        faces.operator()(1, 2) = 3;
        faces.operator()(2, 0) = 0;
        faces.operator()(2, 1) = 3;
        faces.operator()(2, 2) = 4;
        faces.operator()(3, 0) = 0;
        faces.operator()(3, 1) = 4;
        faces.operator()(3, 2) = 1;

        return makeCloud(std::move(points), std::move(faces));
    }

    CpuFloatCloud makeTetrahedronWithInwardNormals()
    {
        FloatMatrix points(4, 3);
        points.operator()(0, 0) = 1.0f;
        points.operator()(0, 1) = 1.0f;
        points.operator()(0, 2) = 1.0f;
        points.operator()(1, 0) = -1.0f;
        points.operator()(1, 1) = -1.0f;
        points.operator()(1, 2) = 1.0f;
        points.operator()(2, 0) = -1.0f;
        points.operator()(2, 1) = 1.0f;
        points.operator()(2, 2) = -1.0f;
        points.operator()(3, 0) = 1.0f;
        points.operator()(3, 1) = -1.0f;
        points.operator()(3, 2) = -1.0f;

        IntMatrix faces(4, 3);
        faces.operator()(0, 0) = 0;
        faces.operator()(0, 1) = 2;
        faces.operator()(0, 2) = 1;
        faces.operator()(1, 0) = 0;
        faces.operator()(1, 1) = 1;
        faces.operator()(1, 2) = 3;
        faces.operator()(2, 0) = 0;
        faces.operator()(2, 1) = 3;
        faces.operator()(2, 2) = 2;
        faces.operator()(3, 0) = 1;
        faces.operator()(3, 1) = 2;
        faces.operator()(3, 2) = 3;

        auto outward = plapoint::mesh::recomputeVertexNormals(
            plapoint::detail::fromDeviceCloud(makeCloud(std::move(points), std::move(faces))));

        FloatMatrix inward(static_cast<plamatrix::Index>(outward.size()), 3);
        for (plamatrix::Index row = 0; row < inward.rows(); ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                inward.operator()(row, col) = -outward.normals()->operator()(row, col);
            }
        }
        outward.setNormals(std::move(inward));
        return plapoint::detail::toDeviceCloud(outward);
    }

    template <typename ActualCloud, typename ExpectedCloud>
    void expectCloudPointsNear(const ActualCloud& actual, const ExpectedCloud& expected, float tolerance)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (plamatrix::Index row = 0; row < static_cast<plamatrix::Index>(actual.size()); ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                EXPECT_NEAR(actual.points().operator()(row, col), expected.points().operator()(row, col), tolerance)
                    << "row=" << row << " col=" << col;
            }
        }
    }

    template <typename ActualCloud, typename ExpectedCloud>
    void expectNormalsAligned(const ActualCloud& actual, const ExpectedCloud& expected)
    {
        ASSERT_TRUE(actual.hasNormals());
        ASSERT_TRUE(expected.hasNormals());
        ASSERT_EQ(actual.normals()->rows(), expected.normals()->rows());

        for (plamatrix::Index row = 0; row < actual.normals()->rows(); ++row)
        {
            float dot = 0.0f;
            for (int col = 0; col < 3; ++col)
            {
                const float actual_value = actual.normals()->operator()(row, col);
                const float expected_value = expected.normals()->operator()(row, col);
                dot += actual_value * expected_value;
                EXPECT_NEAR(actual_value, expected_value, 2.0e-5f) << "row=" << row << " col=" << col;
            }
            EXPECT_GT(dot, 0.999f) << "row=" << row;
        }
    }

} // namespace

TEST(MeshNormalsGpuTest, RecomputeVertexNormalsMatchesCpuDirections)
{
    SKIP_IF_NO_GPU();

    auto mesh = makeRaisedFanMesh();
    const auto expected = plapoint::mesh::recomputeVertexNormals(plapoint::detail::fromDeviceCloud(mesh));

    const auto actual_gpu = plapoint::mesh::recomputeVertexNormals(mesh.toGpu());
    const auto actual = actual_gpu.toCpu();

    expectNormalsAligned(actual, expected);
    ASSERT_TRUE(actual.hasFaces());
    ASSERT_TRUE(expected.hasFaces());
    EXPECT_EQ(actual.faces()->rows(), expected.faces()->rows());
}

TEST(MeshNormalsGpuTest, OrientNormalsOutwardFromCentroidFlipsSimpleClosedMesh)
{
    SKIP_IF_NO_GPU();

    auto mesh = makeTetrahedronWithInwardNormals();
    const auto expected = plapoint::mesh::orientNormalsOutwardFromCentroid(
        plapoint::detail::fromDeviceCloud(mesh));

    const auto actual_gpu = plapoint::mesh::orientNormalsOutwardFromCentroid(mesh.toGpu());
    const auto actual = actual_gpu.toCpu();

    expectNormalsAligned(actual, expected);
    ASSERT_TRUE(actual.hasFaces());
    ASSERT_TRUE(expected.hasFaces());
    ASSERT_EQ(actual.faces()->rows(), expected.faces()->rows());
    for (plamatrix::Index row = 0; row < actual.faces()->rows(); ++row)
    {
        EXPECT_EQ(actual.faces()->operator()(row, 0), expected.faces()->operator()(row, 0));
        EXPECT_EQ(actual.faces()->operator()(row, 1), expected.faces()->operator()(row, 1));
        EXPECT_EQ(actual.faces()->operator()(row, 2), expected.faces()->operator()(row, 2));
    }
}

TEST(MeshNormalsGpuTest, TaubinSmoothMatchesCpuForDefaultBoundaryBehavior)
{
    SKIP_IF_NO_GPU();

    auto mesh = makeRaisedFanMesh();
    const auto expected = plapoint::mesh::taubinSmooth(plapoint::detail::fromDeviceCloud(mesh), 2, 0.5f, -0.53f);

    const auto actual_gpu = plapoint::mesh::taubinSmooth(mesh.toGpu(), 2, 0.5f, -0.53f);
    const auto actual = actual_gpu.toCpu();

    expectCloudPointsNear(actual, expected, 2.0e-5f);
    ASSERT_TRUE(actual.hasFaces());
    ASSERT_TRUE(expected.hasFaces());
    EXPECT_EQ(actual.faces()->rows(), expected.faces()->rows());
}

TEST(MeshNormalsGpuTest, TaubinSmoothCanKeepBoundaryVerticesFixed)
{
    SKIP_IF_NO_GPU();

    auto mesh = makeRaisedFanMesh();

    const auto actual_gpu = plapoint::mesh::taubinSmooth(mesh.toGpu(), 1, 0.5f, 0.0f, true);
    const auto actual = actual_gpu.toCpu();

    EXPECT_LT(actual.points().operator()(0, 2), mesh.points().operator()(0, 2));
    for (plamatrix::Index row = 1; row < static_cast<plamatrix::Index>(mesh.size()); ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            EXPECT_NEAR(actual.points().operator()(row, col), mesh.points().operator()(row, col), 1.0e-6f)
                << "boundary row=" << row << " col=" << col;
        }
    }
}

TEST(MeshNormalsGpuTest, TaubinSmoothHandlesEmptyMeshLikeCpu)
{
    SKIP_IF_NO_GPU();

    FloatMatrix points(0, 3);
    IntMatrix faces(0, 3);
    auto mesh = makeCloud(std::move(points), std::move(faces));
    const auto expected = plapoint::mesh::taubinSmooth(plapoint::detail::fromDeviceCloud(mesh), 3, 0.5f, -0.53f);

    const auto actual_gpu = plapoint::mesh::taubinSmooth(mesh.toGpu(), 3, 0.5f, -0.53f);
    const auto actual = actual_gpu.toCpu();

    EXPECT_EQ(actual.size(), expected.size());
    ASSERT_TRUE(actual.hasFaces());
    ASSERT_TRUE(expected.hasFaces());
    EXPECT_EQ(actual.faces()->rows(), expected.faces()->rows());
}

TEST(MeshNormalsGpuTest, TaubinSmoothRejectsDegenerateParametersLikeCpu)
{
    SKIP_IF_NO_GPU();

    auto mesh = makeRaisedFanMesh();
    auto gpu_mesh = mesh.toGpu();

    EXPECT_THROW(plapoint::mesh::taubinSmooth(gpu_mesh, -1, 0.5f, -0.53f), std::invalid_argument);
    EXPECT_THROW(plapoint::mesh::taubinSmooth(gpu_mesh, 1, std::numeric_limits<float>::infinity(), -0.53f),
                 std::invalid_argument);
    EXPECT_THROW(plapoint::mesh::taubinSmooth(gpu_mesh, 1, 0.5f, std::numeric_limits<float>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(MeshNormalsGpuTest, ResidentResultRetainsContextAndWaitsForInputStream)
{
    SKIP_IF_NO_GPU();
    auto expected = makeRaisedFanMesh();
    FloatMatrix updated = expected.points();
    updated(0, 2) = 2.0f;
    expected.setPoints(updated);
    std::weak_ptr<plamatrix::internal::ExecutionContext> owner;
    const auto result = [&]()
    {
        auto input = makeRaisedFanMesh().toGpu();
        owner = input.executionContext();
        cudaStream_t stream = nullptr;
        PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try
        {
            PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(input.points().data(),
                                                updated.data(),
                                                static_cast<std::size_t>(updated.size()) * sizeof(float),
                                                cudaMemcpyHostToDevice,
                                                stream));
            auto output = plapoint::mesh::recomputeVertexNormals(input, stream);
            EXPECT_EQ(output.executionContext(), input.executionContext());
            PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
            return output;
        }
        catch (...)
        {
            cudaStreamDestroy(stream);
            throw;
        }
    }();
    EXPECT_FALSE(owner.expired());
    const auto host = result.toCpu();
    expectCloudPointsNear(host, expected, 0.0f);
    expectNormalsAligned(host, plapoint::mesh::recomputeVertexNormals(plapoint::detail::fromDeviceCloud(expected)));
}

#endif // PLAPOINT_WITH_CUDA
