#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#ifdef PLAPOINT_WITH_CUDA

#include <cuda_runtime.h>

#include <plamatrix/plamatrix.h>
#include <plamatrix/internal/core/device.h>
#include <plamatrix/internal/device/device_matrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/normal_estimation.h>
#include <plapoint/gpu/spatial_index.h>
#include <plapoint/search/kdtree.h>

namespace
{

template <typename Scalar>
using CpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::CPU>;

template <typename Scalar>
using GpuCloud = plapoint::internal::DeviceCloud<Scalar, plamatrix::internal::Device::GPU>;

template <typename Scalar, typename Coordinate>
CpuCloud<Scalar> makeGridCloud(Coordinate coordinate)
{
    CpuCloud<Scalar> cloud(25);
    int row = 0;
    for (int y = -2; y <= 2; ++y)
    {
        for (int x = -2; x <= 2; ++x)
        {
            const auto point = coordinate(Scalar(x), Scalar(y));
            cloud.points().operator()(row, 0) = point[0];
            cloud.points().operator()(row, 1) = point[1];
            cloud.points().operator()(row, 2) = point[2];
            ++row;
        }
    }
    return cloud;
}

template <typename Scalar>
plamatrix::Matrix<Scalar, plamatrix::Dynamic, plamatrix::Dynamic>
estimateDirect(const GpuCloud<Scalar>& cloud,
               plapoint::gpu::GpuSpatialIndex<Scalar>& index,
               plapoint::gpu::NormalEstimationGpuWorkspace<Scalar>& workspace,
               plamatrix::internal::ResidentMatrix<Scalar>& normals,
               cudaStream_t stream)
{
    plapoint::gpu::estimateNormalsAsync(cloud, index, 9, normals, workspace, stream);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    workspace.checkStatus();
    return normals.toHostMatrix();
}

template <typename Scalar>
class NormalEstimationGpuTest : public ::testing::Test
{
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NormalEstimationGpuTest, ScalarTypes);

TYPED_TEST(NormalEstimationGpuTest, PlaneAndRotatedPlaneHaveDeterministicNormals)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto plane_cpu = makeGridCloud<Scalar>([](Scalar x, Scalar y)
    {
        return std::array<Scalar, 3>{x, y, Scalar(0)};
    });
    auto plane_gpu = plane_cpu.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(plane_gpu, Scalar(1));
    plapoint::gpu::NormalEstimationGpuWorkspace<Scalar> workspace;
    plamatrix::internal::ResidentMatrix<Scalar> normals(25, 3, plane_gpu.executionContext());

    const auto plane_normals = estimateDirect(
        plane_gpu, index, workspace, normals, nullptr);
    for (plamatrix::Index row = 0; row < plane_normals.rows(); ++row)
    {
        EXPECT_NEAR(plane_normals(row, 0), Scalar(0), Scalar(1e-5));
        EXPECT_NEAR(plane_normals(row, 1), Scalar(0), Scalar(1e-5));
        EXPECT_GT(plane_normals(row, 2), Scalar(0.999));
    }

    auto rotated_cpu = makeGridCloud<Scalar>([](Scalar x, Scalar y)
    {
        return std::array<Scalar, 3>{x, y, x};
    });
    auto rotated_gpu = rotated_cpu.toGpu();
    index.build(rotated_gpu, Scalar(1));
    plamatrix::internal::ResidentMatrix<Scalar> rotated_output(25, 3, rotated_gpu.executionContext());
    const auto rotated_normals = estimateDirect(
        rotated_gpu, index, workspace, rotated_output, nullptr);
    const Scalar inverse_sqrt_two = Scalar(1) / std::sqrt(Scalar(2));
    for (plamatrix::Index row = 0; row < rotated_normals.rows(); ++row)
    {
        EXPECT_NEAR(std::abs(rotated_normals(row, 0)), inverse_sqrt_two, Scalar(2e-4));
        EXPECT_NEAR(rotated_normals(row, 1), Scalar(0), Scalar(2e-4));
        EXPECT_NEAR(std::abs(rotated_normals(row, 2)), inverse_sqrt_two, Scalar(2e-4));
        int largest_component = 0;
        for (int column = 1; column < 3; ++column)
        {
            if (std::abs(rotated_normals(row, column))
                > std::abs(rotated_normals(row, largest_component)))
            {
                largest_component = column;
            }
        }
        EXPECT_GE(rotated_normals(row, largest_component), Scalar(0));
    }
}

TYPED_TEST(NormalEstimationGpuTest, InvalidAndDegenerateNeighborhoodsProduceZeroNormals)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    CpuCloud<Scalar> cpu_cloud(6);
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    const Scalar values[18] = {
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(1), Scalar(0), Scalar(0),
        Scalar(5), Scalar(0), Scalar(0),
        Scalar(6), Scalar(0), Scalar(0),
        infinity, Scalar(0), Scalar(0)};
    for (int row = 0; row < 6; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            cpu_cloud.points().operator()(row, column) = values[row * 3 + column];
        }
    }
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(gpu_cloud, Scalar(1));
    plapoint::gpu::NormalEstimationGpuWorkspace<Scalar> workspace;
    plamatrix::internal::ResidentMatrix<Scalar> normals(6, 3, gpu_cloud.executionContext());

    plapoint::gpu::estimateNormalsAsync(
        gpu_cloud, index, 5, normals, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    workspace.checkStatus();
    const auto host_normals = normals.toHostMatrix();
    for (plamatrix::Index row = 0; row < host_normals.rows(); ++row)
    {
        EXPECT_EQ(host_normals(row, 0), Scalar(0));
        EXPECT_EQ(host_normals(row, 1), Scalar(0));
        EXPECT_EQ(host_normals(row, 2), Scalar(0));
    }
}

TEST(NormalEstimationGpuTest, FloatCovarianceOverflowProducesZeroNormalsWithoutEigensolverError)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_cloud = makeGridCloud<float>([](float x, float y)
    {
        constexpr float scale = 1.0e30F;
        return std::array<float, 3>{x * scale, y * scale, 0.0F};
    });
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalEstimationGpuWorkspace<float> workspace;
    plamatrix::internal::ResidentMatrix<float> normals(25, 3, gpu_cloud.executionContext());

    plapoint::gpu::estimateNormalsAsync(
        gpu_cloud, index, 9, normals, workspace, nullptr);
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(nullptr));
    EXPECT_NO_THROW(workspace.checkStatus());
    const auto host_normals = normals.toHostMatrix();
    for (plamatrix::Index row = 0; row < host_normals.rows(); ++row)
    {
        EXPECT_EQ(host_normals(row, 0), 0.0F);
        EXPECT_EQ(host_normals(row, 1), 0.0F);
        EXPECT_EQ(host_normals(row, 2), 0.0F);
    }
}

TEST(NormalEstimationGpuTest, RepeatedHighLevelComputeHandlesDegenerateLineCloud)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    auto cpu_cloud = std::make_shared<CpuCloud<float>>(128);
    for (int row = 0; row < 128; ++row)
    {
        cpu_cloud->points().operator()(row, 0) = static_cast<float>(row) * 0.01f;
        cpu_cloud->points().operator()(row, 1) = 0.0f;
        cpu_cloud->points().operator()(row, 2) = 0.0f;
    }
    auto gpu_cloud = std::make_shared<GpuCloud<float>>(cpu_cloud->toGpu());
    auto tree = std::make_shared<
        plapoint::search::internal::DeviceKdTree<float, plamatrix::internal::Device::GPU>>();
    tree->setInputCloud(gpu_cloud);
    tree->build();
    plapoint::MatrixNormalEstimation<float, plamatrix::internal::Device::GPU> estimator;
    estimator.setInputCloud(gpu_cloud);
    estimator.setSearchMethod(tree);
    estimator.setKSearch(8);

    const auto first = estimator.compute().toHostMatrix();
    const auto second = estimator.compute().toHostMatrix();

    EXPECT_EQ(first.rows(), 128);
    EXPECT_EQ(second.rows(), 128);
}

TEST(NormalEstimationGpuTest, EmptyCloudBindsWorkspaceBeforeAnyLaunch)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    CpuCloud<float> cpu_cloud(0);
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<float> index;
    index.buildAdaptive(gpu_cloud);
    plapoint::gpu::NormalEstimationGpuWorkspace<float> workspace;
    plamatrix::internal::ResidentMatrix<float> normals(0, 3, gpu_cloud.executionContext());
    cudaStream_t first_stream = nullptr;
    cudaStream_t second_stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking));
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking));

    plapoint::gpu::estimateNormalsAsync(
        gpu_cloud, index, 3, normals, workspace, first_stream);
    EXPECT_THROW(
        plapoint::gpu::estimateNormalsAsync(
            gpu_cloud, index, 3, normals, workspace, second_stream),
        std::logic_error);

    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(first_stream));
    workspace.checkStatus();
    workspace.resetStream();
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(first_stream));
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(second_stream));
}

TYPED_TEST(NormalEstimationGpuTest, ReusesOutputAndWorkspaceOnNonDefaultStream)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cpu_cloud = makeGridCloud<Scalar>([](Scalar x, Scalar y)
    {
        return std::array<Scalar, 3>{x, y, Scalar(0)};
    });
    auto gpu_cloud = cpu_cloud.toGpu();
    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(gpu_cloud, Scalar(1));
    plapoint::gpu::NormalEstimationGpuWorkspace<Scalar> workspace;
    plamatrix::internal::ResidentMatrix<Scalar> normals(25, 3, gpu_cloud.executionContext());
    cudaStream_t stream = nullptr;
    PLAPOINT_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const auto first = estimateDirect(gpu_cloud, index, workspace, normals, stream);
    const Scalar* output_address = normals.data();
    const auto second = estimateDirect(gpu_cloud, index, workspace, normals, stream);
    EXPECT_EQ(normals.data(), output_address);
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

TYPED_TEST(NormalEstimationGpuTest, MatchesCpuNormalsByAngleOnCurvedSurface)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        GTEST_SKIP() << "No CUDA-capable device detected";
    }

    using Scalar = TypeParam;
    auto cpu_value = makeGridCloud<Scalar>([](Scalar x, Scalar y)
    {
        return std::array<Scalar, 3>{x, y, Scalar(0.05) * (x * x + y * y)};
    });
    auto cpu_cloud = std::make_shared<CpuCloud<Scalar>>(std::move(cpu_value));
    auto gpu_cloud = cpu_cloud->toGpu();
    auto cpu_tree = std::make_shared<
        plapoint::search::internal::DeviceKdTree<Scalar, plamatrix::internal::Device::CPU>>();
    cpu_tree->setInputCloud(cpu_cloud);
    cpu_tree->build();
    plapoint::MatrixNormalEstimation<Scalar, plamatrix::internal::Device::CPU> cpu_estimator;
    cpu_estimator.setInputCloud(cpu_cloud);
    cpu_estimator.setSearchMethod(cpu_tree);
    cpu_estimator.setKSearch(9);
    const auto cpu_normals = cpu_estimator.compute();

    plapoint::gpu::GpuSpatialIndex<Scalar> index;
    index.build(gpu_cloud, Scalar(1));
    plapoint::gpu::NormalEstimationGpuWorkspace<Scalar> workspace;
    plamatrix::internal::ResidentMatrix<Scalar> gpu_normals(25, 3, gpu_cloud.executionContext());
    const auto direct_normals = estimateDirect(
        gpu_cloud, index, workspace, gpu_normals, nullptr);
    for (plamatrix::Index row = 0; row < cpu_normals.rows(); ++row)
    {
        Scalar dot = Scalar(0);
        for (int column = 0; column < 3; ++column)
        {
            dot += cpu_normals(row, column) * direct_normals(row, column);
        }
        EXPECT_GT(std::abs(dot), Scalar(0.999));
    }
}

} // namespace

#endif
