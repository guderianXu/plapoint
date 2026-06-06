#include <plapoint/core/point_cloud.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/registration/icp.h>
#include <plapoint/search/kdtree.h>
#include <plamatrix/plamatrix.h>

#ifdef PLAPOINT_WITH_CUDA
#include <plapoint/gpu/cuda_check.h>
#include <plapoint/gpu/icp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

struct Options
{
    int points = 20000;
    int iterations = 3;
    int icp_points = 512;
    int icp_max_iterations = 3;
    bool skip_cpu_icp = false;
    bool skip_icp_identity = false;
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
        else if (arg == "--icp-points" && i + 1 < argc)
        {
            options.icp_points = std::max(3, std::atoi(argv[++i]));
        }
        else if (arg == "--icp-max-iterations" && i + 1 < argc)
        {
            options.icp_max_iterations = std::max(1, std::atoi(argv[++i]));
        }
        else if (arg == "--skip-cpu-icp")
        {
            options.skip_cpu_icp = true;
        }
        else if (arg == "--skip-icp-identity")
        {
            options.skip_icp_identity = true;
        }
        else if (arg == "--help")
        {
            std::cout
                << "Usage: plapoint_benchmarks [--points N] [--iterations N]\n"
                << "                           [--icp-points N] [--icp-max-iterations N]\n"
                << "                           [--skip-cpu-icp] [--skip-icp-identity]\n";
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
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeTranslatedPerturbedGridPoints(
    int count,
    Scalar tx,
    Scalar ty,
    Scalar tz)
{
    auto points = makeTranslatedGridPoints<Scalar>(count, tx, ty, tz);
    for (int i = 0; i < count; ++i)
    {
        points(i, 0) += static_cast<Scalar>((i % 7) - 3) * Scalar(0.0002);
        points(i, 1) += static_cast<Scalar>(((i / 7) % 5) - 2) * Scalar(0.00015);
        points(i, 2) += static_cast<Scalar>(((i / 35) % 3) - 1) * Scalar(0.0001);
    }
    return points;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeBinaryGridPoints(int count)
{
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        const int x = i % 257;
        const int y = (i / 257) % 251;
        const int z = (i / (257 * 251)) % 241;
        points(i, 0) = static_cast<Scalar>(x) * Scalar(0.125);
        points(i, 1) = static_cast<Scalar>(y) * Scalar(0.125);
        points(i, 2) = static_cast<Scalar>(z) * Scalar(0.125);
    }
    return points;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeTranslationTransform(
    Scalar tx,
    Scalar ty,
    Scalar tz)
{
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> transform(4, 4);
    transform.fill(Scalar(0));
    transform.setValue(0, 0, Scalar(1));
    transform.setValue(1, 1, Scalar(1));
    transform.setValue(2, 2, Scalar(1));
    transform.setValue(3, 3, Scalar(1));
    transform.setValue(0, 3, tx);
    transform.setValue(1, 3, ty);
    transform.setValue(2, 3, tz);
    return transform;
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

void benchmarkCpuIcp(int icp_points, int icp_max_iterations, int iterations)
{
    auto source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxIterations(icp_max_iterations);
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

void benchmarkCpuIcpFiniteRadius(int icp_points, int icp_max_iterations, int iterations)
{
    auto source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(icp_max_iterations);
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

void benchmarkCpuIcpFiniteRadiusTranslation(int icp_points, int icp_max_iterations, int iterations)
{
    auto source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(icp_max_iterations);
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

void benchmarkCpuIcpFiniteRadiusTranslationReuse(int icp_points, int icp_max_iterations, int iterations)
{
    auto source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    plapoint::IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        Cloud<plamatrix::Device::CPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("cpu_icp_finite_radius_translation_reuse", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "cpu_icp_finite_radius_translation_reuse produced no aligned points\n";
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

void benchmarkGpuIcp(int icp_points, int icp_max_iterations, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_identity", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxIterations(icp_max_iterations);
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

void benchmarkGpuIcpIdentitySameBufferReuseOutput(int icp_points, int icp_max_iterations, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_identity_same_buffer_reuse_output", "no_usable_cuda_device");
        return;
    }

    auto cpu_cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto cloud = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_cloud->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud);
    icp.setMaxIterations(icp_max_iterations);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(output);
        sink += output.size();
    });
    printResult("gpu_icp_identity_same_buffer_reuse_output", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_identity_same_buffer_reuse_output produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadius(int icp_points, int icp_max_iterations, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius", "no_usable_cuda_device");
        return;
    }

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
        icp.setMaxIterations(icp_max_iterations);
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

void benchmarkGpuIcpFiniteRadiusTranslation(int icp_points, int icp_max_iterations, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius_translation", "no_usable_cuda_device");
        return;
    }

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
        icp.setMaxIterations(icp_max_iterations);
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

void benchmarkGpuIcpFiniteRadiusTranslationReuse(int icp_points, int icp_max_iterations, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius_translation_reuse", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        Cloud<plamatrix::Device::GPU> output;
        icp.align(output);
        sink += output.size();
    });
    printResult("gpu_icp_finite_radius_translation_reuse", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_finite_radius_translation_reuse produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationReuseShrinking(int icp_points, int icp_max_iterations, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius_translation_reuse_shrinking", "no_usable_cuda_device");
        return;
    }

    const int large_points = std::max(4, icp_points);
    const int small_points = std::max(4, large_points / 2);
    auto cpu_large_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(large_points, 0.003f, -0.002f, 0.001f));
    auto cpu_large_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(large_points));
    auto cpu_small_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(small_points, 0.003f, -0.002f, 0.001f));
    auto cpu_small_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(small_points));
    auto large_source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_large_source->toGpu());
    auto large_target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_large_target->toGpu());
    auto small_source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_small_source->toGpu());
    auto small_target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_small_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);

    Cloud<plamatrix::Device::GPU> large_output;
    Cloud<plamatrix::Device::GPU> small_output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.setInputSource(large_source);
        icp.setInputTarget(large_target);
        icp.align(large_output);
        sink += large_output.size();

        icp.setInputSource(small_source);
        icp.setInputTarget(small_target);
        icp.align(small_output);
        sink += small_output.size();
    });
    printResult("gpu_icp_finite_radius_translation_reuse_shrinking", large_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_finite_radius_translation_reuse_shrinking produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationReuseOutput(int icp_points, int icp_max_iterations, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius_translation_reuse_output", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(output);
        sink += output.size();
    });
    printResult("gpu_icp_finite_radius_translation_reuse_output", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_finite_radius_translation_reuse_output produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationReuseOutputSkipFinalMetrics(
    int icp_points,
    int icp_max_iterations,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);
    icp.setComputeFinalMetrics(false);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(output);
        sink += output.size();
    });
    printResult(
        "gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationReuseOutputSkipFinalMetricsOneIteration(
    int icp_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics_one_iteration",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(1);
    icp.setComputeFinalMetrics(false);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(output);
        sink += output.size();
    });
    printResult(
        "gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics_one_iteration",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics_one_iteration"
            << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
    const char* benchmark_name,
    int icp_points,
    int icp_max_iterations,
    int iterations,
    bool compute_final_metrics)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);
    if (!compute_final_metrics)
    {
        icp.setComputeFinalMetrics(false);
    }
    icp.setGpuAssumeOrderedCorrespondences(true);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(output);
        sink += output.size();
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationTransformOnlySkipFinalMetrics(
    const char* benchmark_name,
    int icp_points,
    int icp_max_iterations,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);
    icp.setComputeFinalMetrics(false);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align();
        sink += static_cast<std::size_t>(icp.getFinalTransformationDevice().rows());
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no final transform\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationOrderedTransformOnly(
    const char* benchmark_name,
    int icp_points,
    int icp_max_iterations,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);
    icp.setGpuAssumeOrderedCorrespondences(true);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align();
        sink += static_cast<std::size_t>(icp.getFinalTransformationDevice().rows());
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no final transform\n";
    }
}

void benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
    const char* benchmark_name,
    int icp_points,
    int iterations,
    bool assume_ordered_correspondences)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    if (assume_ordered_correspondences)
    {
        icp.setGpuAssumeOrderedCorrespondences(true);
    }

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align();
        sink += static_cast<std::size_t>(icp.getFinalTransformationDevice().rows());
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no final transform\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutput(
    int icp_points,
    int icp_max_iterations,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_finite_radius_translation_reuse_target_output",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(*target);
        sink += target->size();
    });
    printResult(
        "gpu_icp_finite_radius_translation_reuse_target_output",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_finite_radius_translation_reuse_target_output produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutputSkipFinalMetrics(
    int icp_points,
    int icp_max_iterations,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_finite_radius_translation_reuse_target_output_skip_final_metrics",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);
    icp.setComputeFinalMetrics(false);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(*target);
        sink += target->size();
    });
    printResult(
        "gpu_icp_finite_radius_translation_reuse_target_output_skip_final_metrics",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_finite_radius_translation_reuse_target_output_skip_final_metrics produced no aligned points\n";
    }
}

void benchmarkGpuIcpStatsStepFiniteRadiusTranslationNewWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_step_finite_radius_translation_new_workspace", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        plapoint::gpu::IcpStepTransformWorkspace step_workspace;
        plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
        const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace,
            step_transform.data(),
            step_workspace);
        sink += static_cast<std::size_t>(std::max(0, result.stats.active_count));
    });
    printResult("gpu_icp_stats_step_finite_radius_translation_new_workspace", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_step_finite_radius_translation_new_workspace produced no correspondences\n";
    }
}

void benchmarkGpuIcpStatsStepFiniteRadiusTranslationCachedGrid(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_step_finite_radius_translation_cached_grid", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace,
            step_transform.data(),
            step_workspace);
        sink += static_cast<std::size_t>(std::max(0, result.stats.active_count));
    });
    printResult("gpu_icp_stats_step_finite_radius_translation_cached_grid", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_step_finite_radius_translation_cached_grid produced no correspondences\n";
    }
}

void benchmarkGpuIcpStatsStepFiniteRadiusTranslationOrdered(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_step_finite_radius_translation_ordered", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace,
            step_transform.data(),
            step_workspace,
            0,
            true);
        sink += static_cast<std::size_t>(std::max(0, result.stats.active_count));
    });
    printResult("gpu_icp_stats_step_finite_radius_translation_ordered", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_step_finite_radius_translation_ordered produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationNewWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_finite_radius_translation_new_workspace",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
        const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult("gpu_icp_alignment_step_finite_radius_translation_new_workspace", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_alignment_step_finite_radius_translation_new_workspace produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationCachedGrid(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_finite_radius_translation_cached_grid",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult("gpu_icp_alignment_step_finite_radius_translation_cached_grid", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_alignment_step_finite_radius_translation_cached_grid produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationCachedGridReservedWorkspace(
    int icp_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_finite_radius_translation_cached_grid_reserved_workspace",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    stats_workspace.reserveAlignmentStep(static_cast<int>(source->size()));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult(
        "gpu_icp_alignment_step_finite_radius_translation_cached_grid_reserved_workspace",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_finite_radius_translation_cached_grid_reserved_workspace"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepExactPointwiseSameBuffer(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_exact_pointwise_same_buffer",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_points = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto points = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_points->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
            points->points().data(),
            static_cast<int>(points->size()),
            points->points().data(),
            static_cast<int>(points->size()),
            std::numeric_limits<float>::infinity(),
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult("gpu_icp_alignment_step_exact_pointwise_same_buffer", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_alignment_step_exact_pointwise_same_buffer produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepExactPointwiseSameBufferReservedWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_exact_pointwise_same_buffer_reserved_workspace",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_points = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto points = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_points->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    stats_workspace.reserveAlignmentStep(static_cast<int>(points->size()));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
            points->points().data(),
            static_cast<int>(points->size()),
            points->points().data(),
            static_cast<int>(points->size()),
            std::numeric_limits<float>::infinity(),
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult(
        "gpu_icp_alignment_step_exact_pointwise_same_buffer_reserved_workspace",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_exact_pointwise_same_buffer_reserved_workspace"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepOrderedSameBufferFiniteRadius(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_ordered_same_buffer_finite_radius",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_points = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto points = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_points->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    stats_workspace.reserveAlignmentStep(static_cast<int>(points->size()));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
            points->points().data(),
            static_cast<int>(points->size()),
            points->points().data(),
            static_cast<int>(points->size()),
            0.02f,
            stats_workspace,
            step_transform.data(),
            0,
            true);
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult(
        "gpu_icp_alignment_step_ordered_same_buffer_finite_radius",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_ordered_same_buffer_finite_radius"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCachedGrid(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_transformed_exact_pointwise_cached_grid",
            "no_usable_cuda_device");
        return;
    }

    constexpr float source_tx = 0.5f;
    constexpr float source_ty = -0.25f;
    constexpr float source_tz = 0.125f;
    auto target_points_cpu = makeBinaryGridPoints<float>(icp_points);
    auto source_to_target_cpu = makeTranslationTransform<float>(-source_tx, -source_ty, -source_tz);
    auto target_to_source_cpu = makeTranslationTransform<float>(source_tx, source_ty, source_tz);
    auto source_points_cpu = plamatrix::transformPoints(target_to_source_cpu, target_points_cpu);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points_cpu));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points_cpu));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    auto source_to_target_gpu = source_to_target_cpu.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    stats_workspace.reserveFloatAlignmentStep(static_cast<int>(source->size()));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result =
            plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
                source_to_target_gpu.data(),
                source->points().data(),
                static_cast<int>(source->size()),
                target->points().data(),
                static_cast<int>(target->size()),
                0.02f,
                stats_workspace,
                step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult(
        "gpu_icp_alignment_step_transformed_exact_pointwise_cached_grid",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_transformed_exact_pointwise_cached_grid"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCacheHit(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit",
            "no_usable_cuda_device");
        return;
    }

    constexpr float source_tx = 0.5f;
    constexpr float source_ty = -0.25f;
    constexpr float source_tz = 0.125f;
    auto target_points_cpu = makeBinaryGridPoints<float>(icp_points);
    auto source_to_target_cpu = makeTranslationTransform<float>(-source_tx, -source_ty, -source_tz);
    auto target_to_source_cpu = makeTranslationTransform<float>(source_tx, source_ty, source_tz);
    auto source_points_cpu = plamatrix::transformPoints(target_to_source_cpu, target_points_cpu);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points_cpu));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points_cpu));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    auto source_to_target_gpu = source_to_target_cpu.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    stats_workspace.reserveFloatAlignmentStep(static_cast<int>(source->size()));

    (void)plapoint::gpu::detail::computeIcpAlignmentStepColumnMajorWithReservedWorkspace(
        source->points().data(),
        static_cast<int>(source->size()),
        target->points().data(),
        static_cast<int>(target->size()),
        0.02f,
        stats_workspace,
        step_transform.data());

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result =
            plapoint::gpu::detail::computeTransformedIcpAlignmentStepColumnMajorWithReservedWorkspace(
                source_to_target_gpu.data(),
                source->points().data(),
                static_cast<int>(source->size()),
                target->points().data(),
                static_cast<int>(target->size()),
                0.02f,
                stats_workspace,
                step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult(
        "gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpStatsFiniteRadiusTranslationCachedGrid(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_finite_radius_translation_cached_grid", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            nullptr,
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult("gpu_icp_stats_finite_radius_translation_cached_grid", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_finite_radius_translation_cached_grid produced no correspondences\n";
    }
}

void benchmarkGpuIcpResidualStatsFiniteRadiusTranslationNewWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_residual_stats_finite_radius_translation_new_workspace", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult("gpu_icp_residual_stats_finite_radius_translation_new_workspace", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_residual_stats_finite_radius_translation_new_workspace produced no correspondences\n";
    }
}

void benchmarkGpuIcpResidualStatsFiniteRadiusTranslationCachedGrid(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_residual_stats_finite_radius_translation_cached_grid", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto stats = plapoint::gpu::computeIcpResidualStatsColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult("gpu_icp_residual_stats_finite_radius_translation_cached_grid", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_residual_stats_finite_radius_translation_cached_grid produced no correspondences\n";
    }
}

void benchmarkGpuIcpResidualStatsFiniteRadiusTranslationCachedGridReservedWorkspace(
    int icp_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_residual_stats_finite_radius_translation_cached_grid_reserved_workspace",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    stats_workspace.reserveResidualStats(static_cast<int>(source->size()));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto stats = plapoint::gpu::detail::computeIcpResidualStatsColumnMajorWithReservedWorkspace(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult(
        "gpu_icp_residual_stats_finite_radius_translation_cached_grid_reserved_workspace",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_residual_stats_finite_radius_translation_cached_grid_reserved_workspace"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationNewWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_transform_residual_stats_finite_radius_translation_new_workspace",
                     "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> transform(4, 4);
    plapoint::gpu::setIdentityTransform4x4(transform.data());
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output(icp_points, 3);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
            transform.data(),
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            output.data(),
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult(
        "gpu_icp_transform_residual_stats_finite_radius_translation_new_workspace",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_transform_residual_stats_finite_radius_translation_new_workspace produced no correspondences\n";
    }
}

void benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationCachedGrid(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_transform_residual_stats_finite_radius_translation_cached_grid",
                     "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> transform(4, 4);
    plapoint::gpu::setIdentityTransform4x4(transform.data());
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output(icp_points, 3);
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
            transform.data(),
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.02f,
            output.data(),
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult(
        "gpu_icp_transform_residual_stats_finite_radius_translation_cached_grid",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_transform_residual_stats_finite_radius_translation_cached_grid produced no correspondences\n";
    }
}

void benchmarkGpuIcpTransformResidualStatsTransformedExactPointwiseNewWorkspace(
    int icp_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_transform_residual_stats_transformed_exact_pointwise_new_workspace",
            "no_usable_cuda_device");
        return;
    }

    auto source_points = makeBinaryGridPoints<float>(icp_points);
    auto transform_cpu = makeTranslationTransform<float>(0.125f, -0.25f, 0.375f);
    auto target_points = plamatrix::transformPoints(transform_cpu, source_points);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    auto transform = transform_cpu.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output(icp_points, 3);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
            transform.data(),
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.5f,
            output.data(),
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult(
        "gpu_icp_transform_residual_stats_transformed_exact_pointwise_new_workspace",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_transform_residual_stats_transformed_exact_pointwise_new_workspace"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpStatsFallbackTileBoundsNewWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_fallback_tile_bounds_new_workspace", "no_usable_cuda_device");
        return;
    }

    const int fallback_points = std::min(icp_points, 4096);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.0f,
            nullptr,
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult("gpu_icp_stats_fallback_tile_bounds_new_workspace", fallback_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_fallback_tile_bounds_new_workspace produced no correspondences\n";
    }
}

void benchmarkGpuIcpStatsFallbackTileBoundsCachedBounds(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_fallback_tile_bounds_cached_bounds", "no_usable_cuda_device");
        return;
    }

    const int fallback_points = std::min(icp_points, 4096);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto stats = plapoint::gpu::computeIcpCorrespondenceStatsColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.0f,
            nullptr,
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult("gpu_icp_stats_fallback_tile_bounds_cached_bounds", fallback_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_fallback_tile_bounds_cached_bounds produced no correspondences\n";
    }
}

void benchmarkGpuIcpStatsStepFallbackTileBoundsNewWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_step_fallback_tile_bounds_new_workspace", "no_usable_cuda_device");
        return;
    }

    const int fallback_points = std::min(icp_points, 4096);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        plapoint::gpu::IcpStepTransformWorkspace step_workspace;
        plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
        const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.0f,
            stats_workspace,
            step_transform.data(),
            step_workspace);
        sink += static_cast<std::size_t>(std::max(0, result.stats.active_count));
    });
    printResult("gpu_icp_stats_step_fallback_tile_bounds_new_workspace", fallback_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_step_fallback_tile_bounds_new_workspace produced no correspondences\n";
    }
}

void benchmarkGpuIcpStatsStepFallbackTileBoundsCachedBounds(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_stats_step_fallback_tile_bounds_cached_bounds", "no_usable_cuda_device");
        return;
    }

    const int fallback_points = std::min(icp_points, 4096);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plapoint::gpu::IcpStepTransformWorkspace step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::computeIcpStatsAndStepTransformColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.0f,
            stats_workspace,
            step_transform.data(),
            step_workspace);
        sink += static_cast<std::size_t>(std::max(0, result.stats.active_count));
    });
    printResult("gpu_icp_stats_step_fallback_tile_bounds_cached_bounds", fallback_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_stats_step_fallback_tile_bounds_cached_bounds produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepFallbackTileBoundsNewWorkspace(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_alignment_step_fallback_tile_bounds_new_workspace", "no_usable_cuda_device");
        return;
    }

    const int fallback_points = std::min(icp_points, 4096);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
        const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.0f,
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult("gpu_icp_alignment_step_fallback_tile_bounds_new_workspace", fallback_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_alignment_step_fallback_tile_bounds_new_workspace produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepFallbackTileBoundsCachedBounds(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_alignment_step_fallback_tile_bounds_cached_bounds", "no_usable_cuda_device");
        return;
    }

    const int fallback_points = std::min(icp_points, 4096);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(fallback_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const auto result = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.0f,
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult("gpu_icp_alignment_step_fallback_tile_bounds_cached_bounds", fallback_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_alignment_step_fallback_tile_bounds_cached_bounds produced no correspondences\n";
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
    if (options.skip_cpu_icp)
    {
        printSkipped("cpu_icp_identity", "disabled");
        printSkipped("cpu_icp_finite_radius", "disabled");
        printSkipped("cpu_icp_finite_radius_translation", "disabled");
        printSkipped("cpu_icp_finite_radius_translation_reuse", "disabled");
    }
    else
    {
        if (options.skip_icp_identity)
        {
            printSkipped("cpu_icp_identity", "disabled");
        }
        else
        {
            benchmarkCpuIcp(options.icp_points, options.icp_max_iterations, options.iterations);
        }
        benchmarkCpuIcpFiniteRadius(options.icp_points, options.icp_max_iterations, options.iterations);
        benchmarkCpuIcpFiniteRadiusTranslation(options.icp_points, options.icp_max_iterations, options.iterations);
        benchmarkCpuIcpFiniteRadiusTranslationReuse(
            options.icp_points,
            options.icp_max_iterations,
            options.iterations);
    }
#ifdef PLAPOINT_WITH_CUDA
    benchmarkGpuKnn(options.points, options.iterations);
    benchmarkGpuVoxelGrid(options.points, options.iterations);
    if (options.skip_icp_identity)
    {
        printSkipped("gpu_icp_identity", "disabled");
        printSkipped("gpu_icp_identity_same_buffer_reuse_output", "disabled");
    }
    else
    {
        benchmarkGpuIcp(options.icp_points, options.icp_max_iterations, options.iterations);
        benchmarkGpuIcpIdentitySameBufferReuseOutput(
            options.icp_points,
            options.icp_max_iterations,
            options.iterations);
    }
    benchmarkGpuIcpFiniteRadius(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslation(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuse(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseShrinking(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutput(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutputSkipFinalMetrics(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutputSkipFinalMetricsOneIteration(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output_one_iteration",
        options.icp_points,
        1,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedTransformOnly(
        "gpu_icp_finite_radius_translation_ordered_transform_only",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedTransformOnly(
        "gpu_icp_finite_radius_translation_ordered_transform_only_one_iteration",
        options.icp_points,
        1,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output_skip_final_metrics",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusTranslationOrderedOutput(
        "gpu_icp_finite_radius_translation_ordered_output_skip_final_metrics_one_iteration",
        options.icp_points,
        1,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_transform_only_two_iterations",
        options.icp_points,
        options.iterations,
        false);
    benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
        "gpu_icp_finite_radius_nonrigid_ordered_transform_only_two_iterations",
        options.icp_points,
        options.iterations,
        true);
    benchmarkGpuIcpFiniteRadiusTranslationTransformOnlySkipFinalMetrics(
        "gpu_icp_finite_radius_translation_transform_only_skip_final_metrics",
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationTransformOnlySkipFinalMetrics(
        "gpu_icp_finite_radius_translation_transform_only_skip_final_metrics_one_iteration",
        options.icp_points,
        1,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutput(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutputSkipFinalMetrics(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationOrdered(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationCachedGridReservedWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepExactPointwiseSameBuffer(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepExactPointwiseSameBufferReservedWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepOrderedSameBufferFiniteRadius(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCachedGrid(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCacheHit(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationCachedGridReservedWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpTransformResidualStatsTransformedExactPointwiseNewWorkspace(
        options.icp_points,
        options.iterations);
    benchmarkGpuIcpStatsFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
#endif
    return 0;
}
