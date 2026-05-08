#include <gtest/gtest.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/knn.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/registration/icp.h>
#include <plamatrix/plamatrix.h>

#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>

static bool hasCudaDevice()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

#define SKIP_IF_NO_GPU() \
    do { \
        if (!hasCudaDevice()) { \
            GTEST_SKIP() << "No CUDA-capable device detected, skipping GPU test"; \
        } \
    } while(0)

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

// ---- Batch KNN from device pointers ----
TEST(KdTreeGpuTest, BatchKnnDevice)
{
    SKIP_IF_NO_GPU();

    using Scalar = float;
    int M = 10, N = 50, K = 3;

    // Allocate device memory
    Scalar* d_queries = nullptr;
    Scalar* d_data    = nullptr;
    int*    d_indices = nullptr;
    Scalar* d_dists   = nullptr;

    ASSERT_EQ(cudaMalloc(&d_queries, M * 3 * sizeof(Scalar)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_data,    N * 3 * sizeof(Scalar)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_indices, M * K * sizeof(int)),    cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dists,   M * K * sizeof(Scalar)), cudaSuccess);

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

    cudaMemcpy(d_queries, h_queries.data(), M * 3 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,    h_data.data(),    N * 3 * sizeof(Scalar), cudaMemcpyHostToDevice);

    EXPECT_NO_THROW(plapoint::gpu::batchKnnDevice(
        d_queries, M, d_data, N, K, d_indices, d_dists));

    // Copy results back
    std::vector<int>    h_indices(static_cast<std::size_t>(M * K));
    std::vector<Scalar> h_dists(static_cast<std::size_t>(M * K));
    cudaMemcpy(h_indices.data(), d_indices, M * K * sizeof(int),    cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dists.data(),   d_dists,   M * K * sizeof(Scalar), cudaMemcpyDeviceToHost);

    // First query at 0 should find data[0] = 0
    EXPECT_EQ(h_indices[0], 0);

    cudaFree(d_queries);
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_dists);
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

            int matches = 0;
            for (int j = 0; j < 4; ++j)
                if (cpu_sorted[static_cast<std::size_t>(j)] == gpu_sorted[static_cast<std::size_t>(j)])
                    ++matches;

            EXPECT_GE(matches, 3) << "GPU/CPU mismatch for query " << q;
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

    for (int K : {1, 3, 5, 10})
    {
        std::vector<int>    indices;
        std::vector<Scalar> dists;
        EXPECT_NO_THROW(plapoint::gpu::batchKnn(
            queries.data(), M, data.data(), N, K, indices, dists));
        EXPECT_EQ(indices.size(), static_cast<std::size_t>(M * K));
        EXPECT_EQ(dists.size(), static_cast<std::size_t>(M * K));
    }
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
