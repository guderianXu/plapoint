#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/registration/icp.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/plamatrix.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

struct Options
{
    int points = 20000;
    int iterations = 3;
};

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--points" && i + 1 < argc)
        {
            options.points = std::max(1, std::atoi(argv[++i]));
        }
        else if (arg == "--iterations" && i + 1 < argc)
        {
            options.iterations = std::max(1, std::atoi(argv[++i]));
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: plapoint_benchmarks [--points N] [--iterations N]\n";
            std::exit(0);
        }
    }
    return options;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeGridPoints(int count)
{
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 257;
        const int y = (i / 257) % 251;
        const int z = (i / (257 * 251)) % 241;
        points(i, 0) = static_cast<Scalar>(x) * Scalar(0.01);
        points(i, 1) = static_cast<Scalar>(y) * Scalar(0.01);
        points(i, 2) = static_cast<Scalar>(z) * Scalar(0.01);
    }
    return points;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeTranslatedGridPoints(
    int count,
    Scalar tx,
    Scalar ty,
    Scalar tz)
{
    auto points = makeGridPoints<Scalar>(count);
    for (int i = 0; i < count; ++i)
    {
        points(i, 0) += tx;
        points(i, 1) += ty;
        points(i, 2) += tz;
    }
    return points;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeQueries(int count)
{
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> queries(count, 3);
    for (int i = 0; i < count; ++i)
    {
        queries(i, 0) = static_cast<Scalar>((i * 37) % 257) * Scalar(0.01);
        queries(i, 1) = static_cast<Scalar>((i * 17) % 251) * Scalar(0.01);
        queries(i, 2) = static_cast<Scalar>((i * 11) % 241) * Scalar(0.01);
    }
    return queries;
}

template <typename Fn>
double bestMilliseconds(int iterations, Fn&& fn)
{
    // Exclude one-time CUDA context, Thrust, and allocator startup from the measured loop.
    fn();

    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < iterations; ++i)
    {
        const auto start = Clock::now();
        fn();
        const auto end = Clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        best = std::min(best, elapsed);
    }
    return best;
}

void printResult(const std::string& name, int points, int iterations, double milliseconds)
{
    std::cout << name
              << ',' << points
              << ',' << iterations
              << ',' << milliseconds
              << '\n';
}

void printSkipped(const std::string& name, const std::string& reason)
{
    std::cout << name << ",skipped," << reason << ",\n";
}

template <plamatrix::Device Dev>
using Cloud = plapoint::PointCloud<float, Dev>;

void benchmarkCpuKnn(int points, int iterations)
{
    auto cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(points));
    plapoint::search::KdTree<float, plamatrix::Device::CPU> tree;
    tree.setInputCloud(cloud);
    tree.build();
    auto queries = makeQueries<float>(std::min(points, 512));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto result = tree.batchNearestKSearch(queries, 8);
        for (const auto& row : result)
        {
            sink += row.size();
        }
    });
    printResult("cpu_knn_batch_k8", points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "cpu_knn_batch_k8 produced no neighbors\n";
    }
}

void benchmarkCpuVoxelGrid(int points, int iterations)
{
    auto cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(points));
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::VoxelGrid<float, plamatrix::Device::CPU> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(0.1f, 0.1f, 0.1f);
        Cloud<plamatrix::Device::CPU> output;
        voxel.filter(output);
        sink += output.size();
    });
    printResult("cpu_voxel_grid", points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "cpu_voxel_grid produced no centroids\n";
    }
}

void benchmarkCpuNormalEstimation(int points, int iterations)
{
    const int normal_points = std::min(points, 3000);
    auto cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(normal_points));
    auto tree = std::make_shared<plapoint::search::KdTree<float, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::NormalEstimation<float, plamatrix::Device::CPU> normals;
        normals.setInputCloud(cloud);
        normals.setSearchMethod(tree);
        normals.setKSearch(8);
        auto result = normals.compute();
        sink += static_cast<std::size_t>(result.rows());
    });
    printResult("cpu_normal_estimation_k8", normal_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "cpu_normal_estimation_k8 produced no normals\n";
    }
}

void benchmarkCpuIcp(int iterations)
{
    constexpr int icp_points = 512;
    auto source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxIterations(3);
        Cloud<plamatrix::Device::CPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("cpu_icp_identity", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "cpu_icp_identity produced no aligned points\n";
    }
}

void benchmarkCpuIcpFiniteRadius(int iterations)
{
    constexpr int icp_points = 512;
    auto source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(3);
        Cloud<plamatrix::Device::CPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("cpu_icp_finite_radius", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "cpu_icp_finite_radius produced no aligned points\n";
    }
}

void benchmarkCpuIcpFiniteRadiusTranslation(int iterations)
{
    constexpr int icp_points = 512;
    auto source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(3);
        Cloud<plamatrix::Device::CPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("cpu_icp_finite_radius_translation", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "cpu_icp_finite_radius_translation produced no aligned points\n";
    }
}

#ifdef PLAPOINT_WITH_CUDA
void benchmarkGpuKnn(int points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_knn_batch_k8", "no_usable_cuda_device");
        return;
    }

    auto cpu_cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(points));
    auto gpu_cloud = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_cloud->toGpu());
    plapoint::search::KdTree<float, plamatrix::Device::GPU> tree;
    tree.setInputCloud(gpu_cloud);
    tree.build();
    auto queries = makeQueries<float>(std::min(points, 512));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto result = tree.batchNearestKSearch(queries, 8);
        for (const auto& row : result)
        {
            sink += row.size();
        }
    });
    printResult("gpu_knn_batch_k8", points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_knn_batch_k8 produced no neighbors\n";
    }
}

void benchmarkGpuVoxelGrid(int points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_voxel_grid", "no_usable_cuda_device");
        return;
    }

    auto cpu_cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(points));
    auto gpu_cloud = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_cloud->toGpu());
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::VoxelGrid<float, plamatrix::Device::GPU> voxel;
        voxel.setInputCloud(gpu_cloud);
        voxel.setLeafSize(0.1f, 0.1f, 0.1f);
        Cloud<plamatrix::Device::GPU> output;
        voxel.filter(output);
        sink += output.size();
    });
    printResult("gpu_voxel_grid", points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_voxel_grid produced no centroids\n";
    }
}

void benchmarkGpuIcp(int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_identity", "no_usable_cuda_device");
        return;
    }

    constexpr int icp_points = 512;
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxIterations(3);
        Cloud<plamatrix::Device::GPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("gpu_icp_identity", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_identity produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadius(int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius", "no_usable_cuda_device");
        return;
    }

    constexpr int icp_points = 512;
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(3);
        Cloud<plamatrix::Device::GPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("gpu_icp_finite_radius", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_finite_radius produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslation(int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius_translation", "no_usable_cuda_device");
        return;
    }

    constexpr int icp_points = 512;
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(3);
        Cloud<plamatrix::Device::GPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("gpu_icp_finite_radius_translation", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_finite_radius_translation produced no aligned points\n";
    }
}
#endif

} // namespace

int main(int argc, char** argv)
{
    const Options options = parseOptions(argc, argv);
    std::cout << "benchmark,points,iterations,best_ms\n";
    benchmarkCpuKnn(options.points, options.iterations);
    benchmarkCpuVoxelGrid(options.points, options.iterations);
    benchmarkCpuNormalEstimation(options.points, options.iterations);
    benchmarkCpuIcp(options.iterations);
    benchmarkCpuIcpFiniteRadius(options.iterations);
    benchmarkCpuIcpFiniteRadiusTranslation(options.iterations);
#ifdef PLAPOINT_WITH_CUDA
    benchmarkGpuKnn(options.points, options.iterations);
    benchmarkGpuVoxelGrid(options.points, options.iterations);
    benchmarkGpuIcp(options.iterations);
    benchmarkGpuIcpFiniteRadius(options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslation(options.iterations);
#endif
    return 0;
}
