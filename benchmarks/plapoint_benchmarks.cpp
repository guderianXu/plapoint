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
    }
    else
    {
        benchmarkGpuIcp(options.icp_points, options.icp_max_iterations, options.iterations);
    }
    benchmarkGpuIcpFiniteRadius(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslation(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuse(options.icp_points, options.icp_max_iterations, options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutput(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpFiniteRadiusTranslationReuseOutputSkipFinalMetrics(
        options.icp_points,
        options.icp_max_iterations,
        options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpResidualStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpTransformResidualStatsFiniteRadiusTranslationCachedGrid(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpStatsStepFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFallbackTileBoundsNewWorkspace(options.icp_points, options.iterations);
    benchmarkGpuIcpAlignmentStepFallbackTileBoundsCachedBounds(options.icp_points, options.iterations);
#endif
    return 0;
}
