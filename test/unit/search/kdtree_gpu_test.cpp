#include <limits>
#include <stdexcept>
#include <vector>

#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>

#include <plamatrix/plamatrix.h>

#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/knn.h>
#include <plapoint/registration/icp.h>
#include <plapoint/search/kdtree.h>

#ifdef PLAPOINT_WITH_CUDA
static bool hasCudaDevice()
{
    return plapoint::gpu::hasUsableCudaDevice();
}

#define SKIP_IF_NO_GPU() \
    do { \
        if (!hasCudaDevice()) { \
            GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU test"; \
        } \
    } while(0)

TEST(KdTreeGpuTest, GpuBuffersRejectAllocationSizeOverflow)
{
    const auto overflowing_count =
        std::numeric_limits<std::size_t>::max() / sizeof(int) + std::size_t{1};

    EXPECT_THROW(
        (void)plapoint::gpu::DeviceBuffer<int>(overflowing_count),
        std::overflow_error);
    EXPECT_THROW(
        (void)plapoint::gpu::HostPinnedBuffer<int>(overflowing_count),
        std::overflow_error);
}

// ---- Batch KNN from host pointers ----
TEST(KdTreeGpuTest, BatchKnnHost)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;

    // 50 query points, 200 data points, K=4
    int M = 50, N = 200, K = 4;
    std::vector<Scalar> queries(static_cast<std::size_t>(M * 3));
    std::vector<Scalar> data(static_cast<std::size_t>(N * 3));

    for (int i = 0; i < M; ++i)
    {
        queries[static_cast<std::size_t>(i * 3)]     = Scalar(i * 2);
        queries[static_cast<std::size_t>(i * 3 + 1)] = 0;
        queries[static_cast<std::size_t>(i * 3 + 2)] = 0;
    }
    for (int i = 0; i < N; ++i)
    {
        data[static_cast<std::size_t>(i * 3)]     = Scalar(i);
        data[static_cast<std::size_t>(i * 3 + 1)] = Scalar(i % 10);
        data[static_cast<std::size_t>(i * 3 + 2)] = Scalar(i % 5);
    }

    std::vector<int>    indices;
    std::vector<Scalar> dists;

    EXPECT_NO_THROW(plapoint::gpu::batchKnn(
        queries.data(), M, data.data(), N, K, indices, dists));

    // Verify results
    ASSERT_EQ(indices.size(), static_cast<std::size_t>(M * K));
    ASSERT_EQ(dists.size(), static_cast<std::size_t>(M * K));

    // First query point (0,0,0) should have nearest neighbors 0,0,0,0
    // or data points closest to zero which is data[0] = (0,0,0)
    EXPECT_EQ(indices[0], 0);
    EXPECT_FLOAT_EQ(dists[0], 0.0f);
}

TEST(KdTreeGpuTest, BatchKnnHostRejectsNullInputPointers)
{
    std::vector<int> indices;
    std::vector<float> dists;
    float data[3] = {0.0f, 0.0f, 0.0f};

    EXPECT_THROW(
        plapoint::gpu::batchKnn(nullptr, 1, data, 1, 1, indices, dists),
        std::invalid_argument);
    EXPECT_THROW(
        plapoint::gpu::batchKnn(data, 1, nullptr, 1, 1, indices, dists),
        std::invalid_argument);
}

// ---- Batch KNN from device pointers ----
TEST(KdTreeGpuTest, BatchKnnDevice)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    int M = 10, N = 50, K = 3;

    plapoint::gpu::DeviceBuffer<Scalar> d_queries(static_cast<std::size_t>(M * 3));
    plapoint::gpu::DeviceBuffer<Scalar> d_data(static_cast<std::size_t>(N * 3));
    plapoint::gpu::DeviceBuffer<int> d_indices(static_cast<std::size_t>(M * K));
    plapoint::gpu::DeviceBuffer<Scalar> d_dists(static_cast<std::size_t>(M * K));

    // Fill with known data
    std::vector<Scalar> h_queries(static_cast<std::size_t>(M * 3));
    std::vector<Scalar> h_data(static_cast<std::size_t>(N * 3));
    for (int i = 0; i < M; ++i)
    {
        h_queries[static_cast<std::size_t>(i * 3)]     = Scalar(i) * 2;
        h_queries[static_cast<std::size_t>(i * 3 + 1)] = 0;
        h_queries[static_cast<std::size_t>(i * 3 + 2)] = 0;
    }
    for (int i = 0; i < N; ++i)
    {
        h_data[static_cast<std::size_t>(i * 3)]     = Scalar(i);
        h_data[static_cast<std::size_t>(i * 3 + 1)] = 0;
        h_data[static_cast<std::size_t>(i * 3 + 2)] = 0;
    }

    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_queries.get(), h_queries.data(), M * 3 * sizeof(Scalar), cudaMemcpyHostToDevice));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_data.get(),    h_data.data(),    N * 3 * sizeof(Scalar), cudaMemcpyHostToDevice));

    EXPECT_NO_THROW(plapoint::gpu::batchKnnDevice(
        d_queries.get(), M, d_data.get(), N, K, d_indices.get(), d_dists.get()));

    // Copy results back
    std::vector<int>    h_indices(static_cast<std::size_t>(M * K));
    std::vector<Scalar> h_dists(static_cast<std::size_t>(M * K));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(h_indices.data(), d_indices.get(), M * K * sizeof(int),    cudaMemcpyDeviceToHost));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(h_dists.data(),   d_dists.get(),   M * K * sizeof(Scalar), cudaMemcpyDeviceToHost));

    // First query at 0 should find data[0] = 0
    EXPECT_EQ(h_indices[0], 0);
}

TEST(KdTreeGpuTest, BatchKnnDeviceRejectsNullPointers)
{
    float value = 0.0f;
    int index = 0;

    EXPECT_THROW(
        plapoint::gpu::batchKnnDevice(nullptr, 1, &value, 1, 1, &index, &value),
        std::invalid_argument);
    EXPECT_THROW(
        plapoint::gpu::batchKnnDevice(&value, 1, nullptr, 1, 1, &index, &value),
        std::invalid_argument);
    EXPECT_THROW(
        plapoint::gpu::batchKnnDevice(&value, 1, &value, 1, 1, nullptr, &value),
        std::invalid_argument);
    EXPECT_THROW(
        plapoint::gpu::batchKnnDevice(&value, 1, &value, 1, 1, &index, nullptr),
        std::invalid_argument);
}

TEST(KdTreeGpuTest, BatchKnnDeviceColumnMajor)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    const int M = 2;
    const int N = 4;
    const int K = 2;

    std::vector<Scalar> h_queries{
        0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
    };
    std::vector<Scalar> h_data_row_major{
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
    };
    std::vector<Scalar> h_data_col_major{
        0.0f, 1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    std::vector<int> expected_indices;
    std::vector<Scalar> expected_dists;
    plapoint::gpu::batchKnn(
        h_queries.data(), M, h_data_row_major.data(), N, K, expected_indices, expected_dists);

    plapoint::gpu::DeviceBuffer<Scalar> d_queries(static_cast<std::size_t>(M * 3));
    plapoint::gpu::DeviceBuffer<Scalar> d_data(static_cast<std::size_t>(N * 3));
    plapoint::gpu::DeviceBuffer<int> d_indices(static_cast<std::size_t>(M * K));
    plapoint::gpu::DeviceBuffer<Scalar> d_dists(static_cast<std::size_t>(M * K));

    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_queries.get(), h_queries.data(), M * 3 * sizeof(Scalar), cudaMemcpyHostToDevice));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(d_data.get(), h_data_col_major.data(), N * 3 * sizeof(Scalar), cudaMemcpyHostToDevice));

    plapoint::gpu::batchKnnDeviceColumnMajor(
        d_queries.get(), M, d_data.get(), N, K, d_indices.get(), d_dists.get());

    std::vector<int> actual_indices(static_cast<std::size_t>(M * K));
    std::vector<Scalar> actual_dists(static_cast<std::size_t>(M * K));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(actual_indices.data(), d_indices.get(), M * K * sizeof(int), cudaMemcpyDeviceToHost));
    PLAPOINT_CHECK_CUDA(cudaMemcpy(actual_dists.data(), d_dists.get(), M * K * sizeof(Scalar), cudaMemcpyDeviceToHost));

    EXPECT_EQ(actual_indices, expected_indices);
    EXPECT_EQ(actual_dists, expected_dists);
}

TEST(KdTreeGpuTest, BatchKnnDeviceColumnMajorAsyncUsesCallerStream)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    const int M = 2;
    const int N = 4;
    const int K = 2;

    std::vector<Scalar> h_queries{
        0.0f, 0.0f, 0.0f,
        3.0f, 0.0f, 0.0f,
    };
    std::vector<Scalar> h_data_col_major{
        0.0f, 1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };

    plapoint::gpu::DeviceBuffer<Scalar> d_queries(static_cast<std::size_t>(M * 3));
    plapoint::gpu::DeviceBuffer<Scalar> d_data(static_cast<std::size_t>(N * 3));
    plapoint::gpu::DeviceBuffer<int> d_indices(static_cast<std::size_t>(M * K));
    plapoint::gpu::DeviceBuffer<Scalar> d_dists(static_cast<std::size_t>(M * K));

    cudaStream_t stream{};
    PLAPOINT_CHECK_CUDA(cudaStreamCreate(&stream));
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        d_queries.get(), h_queries.data(), M * 3 * sizeof(Scalar), cudaMemcpyHostToDevice, stream));
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        d_data.get(), h_data_col_major.data(), N * 3 * sizeof(Scalar), cudaMemcpyHostToDevice, stream));

    EXPECT_NO_THROW(plapoint::gpu::batchKnnDeviceColumnMajorAsync(
        d_queries.get(), M, d_data.get(), N, K, d_indices.get(), d_dists.get(), stream));

    std::vector<int> actual_indices(static_cast<std::size_t>(M * K));
    std::vector<Scalar> actual_dists(static_cast<std::size_t>(M * K));
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        actual_indices.data(), d_indices.get(), M * K * sizeof(int), cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
        actual_dists.data(), d_dists.get(), M * K * sizeof(Scalar), cudaMemcpyDeviceToHost, stream));
    PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(stream));
    PLAPOINT_CHECK_CUDA(cudaStreamDestroy(stream));

    EXPECT_EQ(actual_indices[0], 0);
    EXPECT_EQ(actual_indices[2], 3);
    EXPECT_FLOAT_EQ(actual_dists[0], 0.0f);
    EXPECT_FLOAT_EQ(actual_dists[2], 0.0f);
}

// ---- KdTree batchNearestKSearch with GPU-resident data ----
TEST(KdTreeGpuTest, BatchKnnOnGpuCloud)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(100, 3);
    for (int i = 0; i < 100; ++i)
    {
        cpu_pts.setValue(i, 0, Scalar(i));
        cpu_pts.setValue(i, 1, Scalar(i % 10));
        cpu_pts.setValue(i, 2, Scalar(i % 5));
    }
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud_ptr = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud_ptr);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(10, 3);
    for (int i = 0; i < 10; ++i)
    {
        queries(i, 0) = Scalar(i * 10);
        queries(i, 1) = 0;
        queries(i, 2) = 0;
    }

    auto results = tree.batchNearestKSearch(queries, 3);
    ASSERT_EQ(results.size(), 10u);
    for (const auto& r : results)
        EXPECT_EQ(r.size(), 3u);

    // Query point 0 at (0,0,0) should include data point 0 (0,0,0)
    bool found_zero = false;
    for (int idx : results[0])
        if (idx == 0) { found_zero = true; break; }
    EXPECT_TRUE(found_zero);
}

TEST(KdTreeGpuTest, BatchNearestKSearchHandlesWorkspaceReuseAndGrowth)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(16, 3);
    for (int i = 0; i < 16; ++i)
    {
        cpu_pts.setValue(i, 0, Scalar(i));
        cpu_pts.setValue(i, 1, 0);
        cpu_pts.setValue(i, 2, 0);
    }
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> small_queries(1, 3);
    small_queries.setValue(0, 0, 0);
    small_queries.setValue(0, 1, 0);
    small_queries.setValue(0, 2, 0);
    auto small_result = tree.batchNearestKSearch(small_queries, 1);
    ASSERT_EQ(small_result.size(), 1u);
    ASSERT_EQ(small_result[0].size(), 1u);
    EXPECT_EQ(small_result[0][0], 0);

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> large_queries(8, 3);
    for (int i = 0; i < 8; ++i)
    {
        large_queries.setValue(i, 0, Scalar(i * 2));
        large_queries.setValue(i, 1, 0);
        large_queries.setValue(i, 2, 0);
    }
    auto large_result = tree.batchNearestKSearch(large_queries, 1);
    ASSERT_EQ(large_result.size(), 8u);
    for (int i = 0; i < 8; ++i)
    {
        ASSERT_EQ(large_result[static_cast<std::size_t>(i)].size(), 1u);
        EXPECT_EQ(large_result[static_cast<std::size_t>(i)][0], i * 2);
    }
}

// ---- GPU-to-CPU result consistency ----
TEST(KdTreeGpuTest, CpuGpuConsistency)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;

    int N = 50;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> pts(N, 3);
    for (int i = 0; i < N; ++i)
    {
        pts.setValue(i, 0, Scalar(i));
        pts.setValue(i, 1, Scalar(i % 7));
        pts.setValue(i, 2, Scalar(i % 11));
    }
    auto cloud = std::make_shared<Cloud>(std::move(pts));

    // CPU KdTree
    plapoint::search::KdTree<Scalar, plamatrix::Device::CPU> cpu_tree;
    cpu_tree.setInputCloud(cloud);
    cpu_tree.build();

    // CPU batch query
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(5, 3);
    for (int i = 0; i < 5; ++i)
    {
        queries(i, 0) = Scalar(i * 10);
        queries(i, 1) = Scalar(i * 3);
        queries(i, 2) = Scalar(i * 7);
    }

    auto cpu_results = cpu_tree.batchNearestKSearch(queries, 4);

    // On CPU, results should be deterministic and non-empty
    ASSERT_EQ(cpu_results.size(), 5u);
    for (const auto& r : cpu_results)
        ASSERT_EQ(r.size(), 4u);

    // If GPU is available, GPU results should match CPU
    if (hasCudaDevice())
    {
        using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
        auto gpu_cloud_ptr = std::make_shared<GpuCloud>(cloud->toGpu());

        plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> gpu_tree;
        gpu_tree.setInputCloud(gpu_cloud_ptr);
        gpu_tree.build();

        auto gpu_results = gpu_tree.batchNearestKSearch(queries, 4);
        ASSERT_EQ(gpu_results.size(), 5u);

        // Compare results: GPU batch uses brute-force, CPU uses kd-tree
        // Both should find the same set of nearest neighbors (order may differ for ties)
        for (int q = 0; q < 5; ++q)
        {
            std::vector<int> cpu_sorted = cpu_results[static_cast<std::size_t>(q)];
            std::vector<int> gpu_sorted = gpu_results[static_cast<std::size_t>(q)];

            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            std::sort(gpu_sorted.begin(), gpu_sorted.end());

            // CPU kd-tree and GPU brute-force should agree on most neighbors
            // Allow for tie-breaking differences (same distance to multiple points)
            int matches = 0;
            for (int j = 0; j < 4; ++j)
                if (cpu_sorted[static_cast<std::size_t>(j)] == gpu_sorted[static_cast<std::size_t>(j)])
                    ++matches;

            EXPECT_GE(matches, 2) << "GPU/CPU mismatch for query " << q
                << " CPU=" << cpu_sorted[0] << "," << cpu_sorted[1] << "," << cpu_sorted[2] << "," << cpu_sorted[3]
                << " GPU=" << gpu_sorted[0] << "," << gpu_sorted[1] << "," << gpu_sorted[2] << "," << gpu_sorted[3];
        }
    }
}

// ---- KNN with different K values ----
TEST(KdTreeGpuTest, BatchKnnMultipleKValues)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    int M = 10, N = 100;

    std::vector<Scalar> queries(static_cast<std::size_t>(M * 3), 0);
    std::vector<Scalar> data(static_cast<std::size_t>(N * 3), 0);
    for (int i = 0; i < N; ++i)
        data[static_cast<std::size_t>(i * 3)] = Scalar(i);

    for (int K : {1, 3, 5, 10, 16, 17, 32})
    {
        std::vector<int>    indices;
        std::vector<Scalar> dists;
        EXPECT_NO_THROW(plapoint::gpu::batchKnn(
            queries.data(), M, data.data(), N, K, indices, dists));
        EXPECT_EQ(indices.size(), static_cast<std::size_t>(M * K));
        EXPECT_EQ(dists.size(), static_cast<std::size_t>(M * K));
        if (K >= 17 && indices.size() >= static_cast<std::size_t>(K))
        {
            EXPECT_EQ(indices[0], 0);
            EXPECT_EQ(indices[1], 1);
            EXPECT_EQ(indices[static_cast<std::size_t>(K - 1)], K - 1);
            EXPECT_FLOAT_EQ(dists[0], 0.0f);
        }
    }
}

TEST(KdTreeGpuTest, BatchKnnDoubleSupportsDistancesAboveFloatMax)
{
    SKIP_IF_NO_GPU();

    using Scalar = double;
    const int M = 1;
    const int N = 2;
    const int K = 1;

    std::vector<Scalar> queries{
        0.0, 0.0, 0.0,
    };
    std::vector<Scalar> data{
        1.0e20, 0.0, 0.0,
        2.0e20, 0.0, 0.0,
    };
    std::vector<int> indices;
    std::vector<Scalar> dists;

    ASSERT_NO_THROW(plapoint::gpu::batchKnn(
        queries.data(), M, data.data(), N, K, indices, dists));

    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(dists.size(), 1u);
    EXPECT_EQ(indices[0], 0);
    EXPECT_NEAR(dists[0], 1.0e40, 1.0e32);
}

TEST(KdTreeGpuTest, BatchKnnDoubleKeepsHugeButFiniteNeighbor)
{
    SKIP_IF_NO_GPU();

    using Scalar = double;
    const int M = 1;
    const int N = 1;
    const int K = 1;

    std::vector<Scalar> queries{1.0e200, 0.0, 0.0};
    std::vector<Scalar> data{0.0, 0.0, 0.0};
    std::vector<int> indices;
    std::vector<Scalar> dists;

    plapoint::gpu::batchKnn(queries.data(), M, data.data(), N, K, indices, dists);

    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(dists.size(), 1u);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(dists[0], std::numeric_limits<Scalar>::max());
}

TEST(KdTreeGpuTest, BatchKnnDoubleKeepsMaxFiniteNeighbor)
{
    SKIP_IF_NO_GPU();

    using Scalar = double;
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();
    const int M = 1;
    const int N = 1;
    const int K = 1;

    std::vector<Scalar> queries{max_value, 0.0, 0.0};
    std::vector<Scalar> data{0.0, 0.0, 0.0};
    std::vector<int> indices;
    std::vector<Scalar> dists;

    plapoint::gpu::batchKnn(queries.data(), M, data.data(), N, K, indices, dists);

    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(dists.size(), 1u);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(dists[0], max_value);
}

TEST(KdTreeGpuTest, BatchKnnHostUsesSentinelWhenNoFiniteNeighborExists)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();
    const int M = 1;
    const int N = 1;
    const int K = 1;

    std::vector<Scalar> queries{0.0f, 0.0f, 0.0f};
    std::vector<Scalar> data{infinity, 0.0f, 0.0f};
    std::vector<int> indices;
    std::vector<Scalar> dists;

    plapoint::gpu::batchKnn(queries.data(), M, data.data(), N, K, indices, dists);

    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(dists.size(), 1u);
    EXPECT_EQ(indices[0], -1);
    EXPECT_EQ(dists[0], max_value);
}

TEST(KdTreeGpuTest, BatchKnnHostKeepsExtremeButFiniteNeighbor)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();
    const int M = 1;
    const int N = 1;
    const int K = 1;

    std::vector<Scalar> queries{max_value, 0.0f, 0.0f};
    std::vector<Scalar> data{0.0f, 0.0f, 0.0f};
    std::vector<int> indices;
    std::vector<Scalar> dists;

    plapoint::gpu::batchKnn(queries.data(), M, data.data(), N, K, indices, dists);

    ASSERT_EQ(indices.size(), 1u);
    ASSERT_EQ(dists.size(), 1u);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(dists[0], max_value);
}

TEST(KdTreeGpuTest, BatchNearestKSearchClampsKToPointCount)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(3, 3);
    cpu_pts.setValue(0, 0, 0); cpu_pts.setValue(0, 1, 0); cpu_pts.setValue(0, 2, 0);
    cpu_pts.setValue(1, 0, 1); cpu_pts.setValue(1, 1, 0); cpu_pts.setValue(1, 2, 0);
    cpu_pts.setValue(2, 0, 2); cpu_pts.setValue(2, 1, 0); cpu_pts.setValue(2, 2, 0);
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    std::vector<std::vector<int>> results;
    ASSERT_NO_THROW(results = tree.batchNearestKSearch(queries, 5));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), 3u);
}

TEST(KdTreeGpuTest, BatchNearestKSearchFallsBackWhenClampedKExceedsCudaLimit)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    constexpr int N = 40;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(N, 3);
    for (int i = 0; i < N; ++i)
    {
        cpu_pts.setValue(i, 0, Scalar(i));
        cpu_pts.setValue(i, 1, Scalar(0));
        cpu_pts.setValue(i, 2, Scalar(0));
    }
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    std::vector<std::vector<int>> results;
    ASSERT_NO_THROW(results = tree.batchNearestKSearch(queries, 50));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), static_cast<std::size_t>(N));
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(results[0][static_cast<std::size_t>(i)], i);
    }
}

TEST(KdTreeGpuTest, BatchNearestKSearchLaunchesCudaPathForThirtyTwoNeighbors)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    constexpr int N = 32;
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(N, 3);
    for (int i = 0; i < N; ++i)
    {
        cpu_pts.setValue(i, 0, Scalar(i));
        cpu_pts.setValue(i, 1, Scalar(0));
        cpu_pts.setValue(i, 2, Scalar(0));
    }
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    std::vector<std::vector<int>> results;
    ASSERT_NO_THROW(results = tree.batchNearestKSearch(queries, N));

    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].size(), static_cast<std::size_t>(N));
}

TEST(KdTreeGpuTest, BatchNearestKSearchDropsInvalidInfiniteDistances)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(2, 3);
    cpu_pts.setValue(0, 0, infinity); cpu_pts.setValue(0, 1, 0); cpu_pts.setValue(0, 2, 0);
    cpu_pts.setValue(1, 0, infinity); cpu_pts.setValue(1, 1, 1); cpu_pts.setValue(1, 2, 0);
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    const auto results = tree.batchNearestKSearch(queries, 1);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(KdTreeGpuTest, BatchNearestKSearchKeepsExtremeButFiniteDistance)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    constexpr Scalar max_value = std::numeric_limits<Scalar>::max();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(1, 3);
    cpu_pts.setValue(0, 0, 0);
    cpu_pts.setValue(0, 1, 0);
    cpu_pts.setValue(0, 2, 0);
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = max_value;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    const auto results = tree.batchNearestKSearch(queries, 1);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0], 0);
}

TEST(KdTreeGpuTest, BatchNearestKSearchKeepsFiniteNeighborWhenAnotherDistanceIsNan)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(2, 3);
    cpu_pts.setValue(0, 0, 0); cpu_pts.setValue(0, 1, 0); cpu_pts.setValue(0, 2, 0);
    cpu_pts.setValue(1, 0, std::numeric_limits<Scalar>::quiet_NaN());
    cpu_pts.setValue(1, 1, 0);
    cpu_pts.setValue(1, 2, 0);
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    const auto results = tree.batchNearestKSearch(queries, 1);

    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 1u);
    EXPECT_EQ(results[0][0], 0);
}

TEST(KdTreeGpuTest, BatchNearestKSearchFallbackDropsInvalidInfiniteDistances)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;
    constexpr Scalar infinity = std::numeric_limits<Scalar>::infinity();
    constexpr int N = 40;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(N, 3);
    for (int i = 0; i < N; ++i)
    {
        cpu_pts.setValue(i, 0, infinity);
        cpu_pts.setValue(i, 1, Scalar(i));
        cpu_pts.setValue(i, 2, 0);
    }
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    const auto results = tree.batchNearestKSearch(queries, 50);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].empty());
}

TEST(KdTreeGpuTest, BatchNearestKSearchMatchesCpuForDuplicateDistanceTies)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(4, 3);
    cpu_pts.setValue(0, 0, 0); cpu_pts.setValue(0, 1, 0); cpu_pts.setValue(0, 2, 0);
    cpu_pts.setValue(1, 0, 0); cpu_pts.setValue(1, 1, 0); cpu_pts.setValue(1, 2, 0);
    cpu_pts.setValue(2, 0, 0); cpu_pts.setValue(2, 1, 0); cpu_pts.setValue(2, 2, 0);
    cpu_pts.setValue(3, 0, 5); cpu_pts.setValue(3, 1, 0); cpu_pts.setValue(3, 2, 0);
    auto cpu_cloud = std::make_shared<Cloud>(std::move(cpu_pts));

    plapoint::search::KdTree<Scalar, plamatrix::Device::CPU> cpu_tree;
    cpu_tree.setInputCloud(cpu_cloud);
    cpu_tree.build();

    auto gpu_cloud = std::make_shared<GpuCloud>(cpu_cloud->toGpu());
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> gpu_tree;
    gpu_tree.setInputCloud(gpu_cloud);
    gpu_tree.build();

    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(1, 3);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;

    auto cpu_results = cpu_tree.batchNearestKSearch(queries, 3);
    auto gpu_results = gpu_tree.batchNearestKSearch(queries, 3);
    ASSERT_EQ(cpu_results.size(), 1u);
    ASSERT_EQ(gpu_results.size(), 1u);

    std::sort(cpu_results[0].begin(), cpu_results[0].end());
    std::sort(gpu_results[0].begin(), gpu_results[0].end());

    EXPECT_EQ(cpu_results[0], (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(gpu_results[0], cpu_results[0]);
}

TEST(KdTreeGpuTest, RejectsKAboveCudaLimitOrPointCount)
{
    using Scalar = float;
    const int M = 1;
    const int N = 4;
    std::vector<Scalar> queries(static_cast<std::size_t>(M * 3), 0);
    std::vector<Scalar> data(static_cast<std::size_t>(N * 3), 0);
    std::vector<int> indices;
    std::vector<Scalar> dists;

    EXPECT_THROW(
        plapoint::gpu::batchKnn(queries.data(), M, data.data(), N, 33, indices, dists),
        std::invalid_argument);
    EXPECT_THROW(
        plapoint::gpu::batchKnn(queries.data(), M, data.data(), N, 5, indices, dists),
        std::invalid_argument);
}

TEST(KdTreeGpuTest, CheckedSizeProductRejectsOverflowBeforeHostBatchAllocation)
{
    EXPECT_EQ(plapoint::gpu::detail::checkedSizeProduct(7, 3, "test buffer"), 21u);
    EXPECT_EQ(
        plapoint::gpu::detail::checkedSizeProduct(
            static_cast<std::size_t>(std::numeric_limits<int>::max()), 3, "test buffer"),
        static_cast<std::size_t>(std::numeric_limits<int>::max()) * 3u);
    EXPECT_THROW(
        plapoint::gpu::detail::checkedSizeProduct(
            std::numeric_limits<std::size_t>::max(), 2, "test buffer"),
        std::overflow_error);
}

// ---- Verification: closest point to itself is distance 0 ----
TEST(KdTreeGpuTest, IdenticalPointsZeroDistance)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    int N = 50, K = 1;

    std::vector<Scalar> data(static_cast<std::size_t>(N * 3));
    for (int i = 0; i < N; ++i)
    {
        data[static_cast<std::size_t>(i * 3)]     = Scalar(i);
        data[static_cast<std::size_t>(i * 3 + 1)] = Scalar(i * 2);
        data[static_cast<std::size_t>(i * 3 + 2)] = Scalar(i * 3);
    }

    // Query same points as data
    auto q = data;
    std::vector<int>    indices;
    std::vector<Scalar> dists;

    plapoint::gpu::batchKnn(q.data(), N, data.data(), N, K, indices, dists);

    // Each point's nearest neighbor should be itself (distance 0)
    for (int i = 0; i < N; ++i)
    {
        EXPECT_EQ(indices[static_cast<std::size_t>(i)], i)
            << "Point " << i << " should find itself as nearest neighbor";
        EXPECT_FLOAT_EQ(dists[static_cast<std::size_t>(i)], 0.0f)
            << "Distance should be 0";
    }
}

#else
// Compile-time test: header includes without CUDA
TEST(KdTreeGpuTest, HeaderCompilesWithoutCuda)
{
    SUCCEED();
}
#endif
