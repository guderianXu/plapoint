#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_refinement.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/normal_refinement.h>
#include <plapoint/gpu/spatial_index.h>
#include <plapoint/search/kdtree.h>

namespace
{

template <typename Scalar>
using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

template <typename Scalar>
using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

void CUDART_CB brieflyHoldStream(void*)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

template <typename Scalar>
CpuCloud<Scalar> makeCloud(const std::vector<std::array<Scalar, 3>>& points,
                           const std::vector<std::array<Scalar, 3>>& normals)
{
    CpuCloud<Scalar> cloud(points.size());
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> normal_matrix(
        static_cast<plamatrix::Index>(normals.size()), 3);
    for (plamatrix::Index row = 0; row < static_cast<plamatrix::Index>(points.size()); ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            cloud.points().operator()(row, column) = points[static_cast<std::size_t>(row)][column];
            normal_matrix.operator()(row, column) = normals[static_cast<std::size_t>(row)][column];
        }
    }
    cloud.setNormals(std::move(normal_matrix));
    return cloud;
}

template <typename Scalar>
plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>
smoothDirect(const GpuCloud<Scalar>& cloud,
             const plapoint::gpu::GpuSpatialIndex<Scalar>& index,
             int k,
             plamatrix::internal::ResidentMatrix<Scalar>& output,
             plapoint::gpu::NormalRefinementGpuWorkspace<Scalar>& workspace,
             cudaStream_t stream)
{
    plapoint::gpu::smoothNormalsAsync(cloud, index, k, output, workspace, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    return output.toHostMatrix();
}

template <typename Scalar>
class NormalRefinementGpuTest : public ::testing::Test
{
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NormalRefinementGpuTest, ScalarTypes);

TYPED_TEST(NormalRefinementGpuTest, AveragesKNearestNormals)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cpu_cloud = makeCloud<Scalar>(
        {{Scalar(0), Scalar(0), Scalar(0)},
         {Scalar(1), Scalar(0), Scalar(0)},
         {Scalar(10), Scalar(0), Scalar(0)}},
        {{Scalar(1), Scalar(0), Scalar(0)},
         {Scalar(0), Scalar(1), Scalar(0)},
         {Scalar(0), Scalar(0), Scalar(1)}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<Scalar> workspace;
    plamatrix::internal::ResidentMatrix<Scalar> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());

    const auto actual = smoothDirect(gpu_cloud, index, 2, output, workspace, nullptr);
    const Scalar inverse_sqrt_two = Scalar(1) / std::sqrt(Scalar(2));
    EXPECT_NEAR(actual(0, 0), inverse_sqrt_two, Scalar(1e-6));
    EXPECT_NEAR(actual(0, 1), inverse_sqrt_two, Scalar(1e-6));
    EXPECT_NEAR(actual(0, 2), Scalar(0), Scalar(1e-6));
    EXPECT_NEAR(actual(1, 0), inverse_sqrt_two, Scalar(1e-6));
    EXPECT_NEAR(actual(1, 1), inverse_sqrt_two, Scalar(1e-6));
    EXPECT_NEAR(actual(1, 2), Scalar(0), Scalar(1e-6));
    EXPECT_NEAR(actual(2, 0), Scalar(0), Scalar(1e-6));
    EXPECT_NEAR(actual(2, 1), inverse_sqrt_two, Scalar(1e-6));
    EXPECT_NEAR(actual(2, 2), inverse_sqrt_two, Scalar(1e-6));
}

TYPED_TEST(NormalRefinementGpuTest, PreservesSourceNormalWhenNeighborSumIsZero)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cpu_cloud = makeCloud<Scalar>(
        {{Scalar(0), Scalar(0), Scalar(0)}, {Scalar(1), Scalar(0), Scalar(0)}},
        {{Scalar(1), Scalar(0), Scalar(0)}, {Scalar(-1), Scalar(0), Scalar(0)}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<Scalar> workspace;
    plamatrix::internal::ResidentMatrix<Scalar> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());

    const auto actual = smoothDirect(gpu_cloud, index, 2, output, workspace, nullptr);
    EXPECT_EQ(actual(0, 0), Scalar(1));
    EXPECT_EQ(actual(0, 1), Scalar(0));
    EXPECT_EQ(actual(0, 2), Scalar(0));
    EXPECT_EQ(actual(1, 0), Scalar(-1));
    EXPECT_EQ(actual(1, 1), Scalar(0));
    EXPECT_EQ(actual(1, 2), Scalar(0));
}

TYPED_TEST(NormalRefinementGpuTest, UsesAllAvailableNeighborsWhenKExceedsPointCount)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cpu_cloud = makeCloud<Scalar>(
        {{Scalar(0), Scalar(0), Scalar(0)}, {Scalar(1), Scalar(0), Scalar(0)}},
        {{Scalar(1), Scalar(0), Scalar(0)}, {Scalar(0), Scalar(1), Scalar(0)}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<Scalar> workspace;
    plamatrix::internal::ResidentMatrix<Scalar> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());

    const auto actual = smoothDirect(gpu_cloud, index, 8, output, workspace, nullptr);
    const Scalar inverse_sqrt_two = Scalar(1) / std::sqrt(Scalar(2));
    for (plamatrix::Index row = 0; row < actual.rows(); ++row)
    {
        EXPECT_NEAR(actual(row, 0), inverse_sqrt_two, Scalar(1e-6));
        EXPECT_NEAR(actual(row, 1), inverse_sqrt_two, Scalar(1e-6));
        EXPECT_NEAR(actual(row, 2), Scalar(0), Scalar(1e-6));
    }
}

TEST(NormalRefinementGpuDirectTest, RejectsCloudNormalStorageAsOutput)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_cloud = makeCloud<float>(
        {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
        {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<float> workspace;
    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    try
    {
        plapoint::gpu::smoothNormalsAsync(
            gpu_cloud, index, 2, *gpu_cloud.normals(), workspace, stream);
        ADD_FAILURE() << "Expected smoothNormalsAsync to reject aliased output";
    }
    catch (const std::invalid_argument& error)
    {
        EXPECT_NE(std::string(error.what()).find("independent output"), std::string::npos);
    }
    catch (...)
    {
        ADD_FAILURE() << "Expected std::invalid_argument for aliased output";
    }

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace.resetStream();
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST(NormalRefinementGpuDirectTest, PreservesSourceWhenSelectedNeighborNormalIsNonFinite)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const float maximum = std::numeric_limits<float>::max();
    auto cpu_cloud = makeCloud<float>(
        {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
         {100.0F, 0.0F, 0.0F}, {101.0F, 0.0F, 0.0F},
         {200.0F, 0.0F, 0.0F}, {201.0F, 0.0F, 0.0F}},
        {{2.0F, 0.0F, 0.0F}, {nan, 1.0F, 0.0F},
         {0.0F, 3.0F, 0.0F}, {0.0F, infinity, 0.0F},
         {maximum, 0.0F, 0.0F}, {maximum, 0.0F, 0.0F}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<float> workspace;
    plamatrix::internal::ResidentMatrix<float> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());

    const auto actual = smoothDirect(gpu_cloud, index, 2, output, workspace, nullptr);
    EXPECT_EQ(actual(0, 0), 2.0F);
    EXPECT_TRUE(std::isnan(actual(1, 0)));
    EXPECT_EQ(actual(1, 1), 1.0F);
    EXPECT_EQ(actual(2, 1), 3.0F);
    EXPECT_TRUE(std::isinf(actual(3, 1)));
    EXPECT_EQ(actual(4, 0), 1.0F);
    EXPECT_EQ(actual(5, 0), 1.0F);
}

TEST(NormalRefinementGpuDirectTest, OrientsInPlaceAndSkipsNonFiniteRows)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    const float infinity = std::numeric_limits<float>::infinity();
    auto cpu_cloud = makeCloud<float>(
        {{-1.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {infinity, 0.0F, 0.0F}},
        {{1.0F, 0.0F, 0.0F}, {-1.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}});
    auto gpu_cloud = cpu_cloud.toGpu();

    plapoint::gpu::orientNormalsTowardViewpointAsync(
        gpu_cloud, plamatrix::Matrix<float, 3, 1>(-10.0F, 0.0F, 0.0F), nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    const auto actual = gpu_cloud.normals()->toHostMatrix();
    EXPECT_EQ(actual(0, 0), -1.0F);
    EXPECT_EQ(actual(1, 0), -1.0F);
    EXPECT_EQ(actual(2, 0), 1.0F);
}

TEST(NormalRefinementGpuDirectTest, HighLevelUpdatesOnlyNormalsAndPreservesPointAttributes)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = float;
    auto cpu_cloud = makeCloud<Scalar>({{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}},
                                       {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}});
    plamatrix::Matrix<std::uint8_t, plamatrix::Dynamic, plamatrix::Dynamic> colors(3, 3);
    plamatrix::Matrix<std::uint16_t, plamatrix::Dynamic, plamatrix::Dynamic> intensities(3, 1);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> fields(3, 2);
    plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic> texture_coords(3, 2);
    for (plamatrix::Index row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            colors.operator()(row, column) = static_cast<std::uint8_t>(10 * row + column);
        }
        intensities.operator()(row, 0) = static_cast<std::uint16_t>(100 + row);
        fields.operator()(row, 0) = Scalar(row) + 0.25F;
        fields.operator()(row, 1) = Scalar(row) + 0.75F;
        texture_coords.operator()(row, 0) = Scalar(row) * 0.1F;
        texture_coords.operator()(row, 1) = Scalar(row) * 0.2F;
    }
    cpu_cloud.setColors(std::move(colors));
    cpu_cloud.setIntensities(std::move(intensities));
    cpu_cloud.setScalarFields({"confidence", "temperature"}, std::move(fields));
    cpu_cloud.setTextureCoords(std::move(texture_coords));
    auto gpu_cloud = std::make_shared<GpuCloud<Scalar>>(cpu_cloud.toGpu());
    auto tree = std::make_shared<
        plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::GPU>>();
    tree->setInputCloud(gpu_cloud);
    tree->build();
    plapoint::NormalRefinement<Scalar, plamatrix::internal::Device::GPU> refinement;
    refinement.setInputCloud(gpu_cloud);
    refinement.setSearchMethod(tree);

    refinement.smooth(3);
    refinement.orientConsistently(plamatrix::Matrix<float, 3, 1>(0.0F, 0.0F, 10.0F));

    const auto actual = gpu_cloud->toCpu();
    ASSERT_EQ(actual.scalarFieldNames(), cpu_cloud.scalarFieldNames());
    for (plamatrix::Index row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            EXPECT_EQ(actual.points()(row, column), cpu_cloud.points()(row, column));
            EXPECT_EQ((*actual.colors())(row, column), (*cpu_cloud.colors())(row, column));
        }
        EXPECT_EQ((*actual.intensities())(row, 0), (*cpu_cloud.intensities())(row, 0));
        for (int column = 0; column < 2; ++column)
        {
            EXPECT_EQ((*actual.scalarFields())(row, column), (*cpu_cloud.scalarFields())(row, column));
            EXPECT_EQ((*actual.textureCoords())(row, column), (*cpu_cloud.textureCoords())(row, column));
        }
    }
}

TEST(NormalRefinementGpuDirectTest, ReusesOutputAndWorkspaceOnNonDefaultStream)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_cloud = makeCloud<float>(
        {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {2.0F, 0.0F, 0.0F}},
        {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<float> workspace;
    plamatrix::internal::ResidentMatrix<float> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());
    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const auto first = smoothDirect(gpu_cloud, index, 3, output, workspace, stream);
    const float* output_address = output.data();
    const auto second = smoothDirect(gpu_cloud, index, 3, output, workspace, stream);
    EXPECT_EQ(output.data(), output_address);
    for (plamatrix::Index row = 0; row < first.rows(); ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            EXPECT_EQ(first(row, column), second(row, column));
        }
    }

    workspace.resetStream();
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST(NormalRefinementGpuDirectTest, RejectsCrossStreamReuseUntilReset)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_cloud = makeCloud<float>({}, {});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<float> workspace;
    plamatrix::internal::ResidentMatrix<float> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());
    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking));
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking));

    plapoint::gpu::smoothNormalsAsync(
        gpu_cloud, index, 3, output, workspace, first_stream);
    EXPECT_THROW(
        plapoint::gpu::smoothNormalsAsync(
            gpu_cloud, index, 3, output, workspace, second_stream),
        std::logic_error);

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(first_stream));
    workspace.resetStream();
    EXPECT_NO_THROW(plapoint::gpu::smoothNormalsAsync(
        gpu_cloud, index, 3, output, workspace, second_stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(second_stream));
    workspace.resetStream();
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(first_stream));
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(second_stream));
}

TEST(NormalRefinementGpuDirectTest, ResetStreamRejectsPendingWork)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_cloud = makeCloud<float>(
        {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
        {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<float> workspace;
    plamatrix::internal::ResidentMatrix<float> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());
    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    smoothDirect(gpu_cloud, index, 2, output, workspace, stream);
    workspace.resetStream();
    PLAPOINT_CHECK_CUDA(cudaLaunchHostFunc(stream, brieflyHoldStream, nullptr));
    plapoint::gpu::smoothNormalsAsync(
        gpu_cloud, index, 2, output, workspace, stream);
    EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);
    EXPECT_THROW(workspace.resetStream(), std::logic_error);

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    EXPECT_NO_THROW(workspace.resetStream());
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST(NormalRefinementGpuDirectTest, OrdinaryOutputSurvivesDestroyedStreamAndCanBeReused)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_cloud = makeCloud<float>(
        {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
        {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalRefinementGpuWorkspace<float> workspace;
    plamatrix::internal::ResidentMatrix<float> output(
        static_cast<plamatrix::Index>(gpu_cloud.size()), 3, gpu_cloud.executionContext());
    cudaStream_t first_stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking));
    smoothDirect(gpu_cloud, index, 2, output, workspace, first_stream);
    const float* output_address = output.data();
    workspace.resetStream();
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(first_stream));

    cudaStream_t second_stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking));
    const auto actual = smoothDirect(gpu_cloud, index, 2, output, workspace, second_stream);
    EXPECT_EQ(output.data(), output_address);
    EXPECT_NEAR(actual(0, 0), std::sqrt(0.5F), 1e-6F);
    workspace.resetStream();
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(second_stream));
}

TEST(NormalRefinementGpuDirectTest, HighLevelKGreaterThanThirtyTwoKeepsHostCompatibility)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_value = makeCloud<float>(
        {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
        {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}});
    auto cpu_cloud = std::make_shared<CpuCloud<float>>(std::move(cpu_value));
    auto gpu_cloud = std::make_shared<GpuCloud<float>>(cpu_cloud->toGpu());
    auto cpu_tree = std::make_shared<
        plapoint::search::internal::DeviceKdTree<float, plamatrix::internal::Device::CPU>>();
    auto gpu_tree = std::make_shared<
        plapoint::search::internal::DeviceKdTree<float, plamatrix::internal::Device::GPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    gpu_tree->setInputCloud(gpu_cloud);
    cpu_tree->build();
    gpu_tree->build();
    plapoint::NormalRefinement<float, plamatrix::internal::Device::CPU> cpu_refinement;
    plapoint::NormalRefinement<float, plamatrix::internal::Device::GPU> gpu_refinement;
    cpu_refinement.setInputCloud(cpu_cloud);
    cpu_refinement.setSearchMethod(cpu_tree);
    gpu_refinement.setInputCloud(gpu_cloud);
    gpu_refinement.setSearchMethod(gpu_tree);

    cpu_refinement.smooth(33);
    gpu_refinement.smooth(33);

    const auto actual = gpu_cloud->normals()->toHostMatrix();
    for (plamatrix::Index row = 0; row < actual.rows(); ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            EXPECT_EQ(actual(row, column), (*cpu_cloud->normals())(row, column));
        }
    }
}

} // namespace

#endif
