#include <gtest/gtest.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/gpu/knn.h>
#include <plamatrix/plamatrix.h>

#ifdef PLAPOINT_WITH_CUDA
#include <cuda_runtime.h>

TEST(KdTreeGpuTest, BatchKnnOnGpuCloud)
{
    using Scalar = float;
    using Cloud = plapoint::PointCloud<Scalar, plamatrix::Device::CPU>;
    using GpuCloud = plapoint::PointCloud<Scalar, plamatrix::Device::GPU>;

    // Create 100 points on CPU, transfer to GPU
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> cpu_pts(100, 3);
    for (int i = 0; i < 100; ++i)
    {
        cpu_pts.setValue(i, 0, Scalar(i));
        cpu_pts.setValue(i, 1, Scalar(i % 10));
        cpu_pts.setValue(i, 2, Scalar(i % 5));
    }
    Cloud cpu_cloud(std::move(cpu_pts));
    auto gpu_cloud_ptr = std::make_shared<GpuCloud>(cpu_cloud.toGpu());

    // Build KdTree on GPU-resident data
    plapoint::search::KdTree<Scalar, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud_ptr);
    tree.build();

    // Batch KNN from CPU queries
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
}
#endif
