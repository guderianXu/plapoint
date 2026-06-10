// Included once by plapoint_benchmarks.cpp inside its anonymous namespace.
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

void benchmarkGpuIcpFiniteRadiusIdentityReuseOutput(
    const char* benchmark_name,
    int icp_points,
    int icp_max_iterations,
    int iterations,
    bool probe_exact_pointwise_on_finite_radius)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(icp_max_iterations);
    if (probe_exact_pointwise_on_finite_radius)
    {
        icp.setGpuProbeExactPointwiseOnFiniteRadius(true);
    }

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

    // The shrunken grid must span more than one x-row; otherwise ICP sees collinear geometry.
    const int large_points = std::max(600, icp_points);
    const int small_points = large_points / 2;
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

void benchmarkGpuIcpFiniteRadiusTranslationReuseOutput(
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

void benchmarkGpuIcpFiniteRadiusTranslationNoOutputOneIteration(int icp_points, int iterations)
{
    const char* benchmark_name = "gpu_icp_finite_radius_translation_no_output_one_iteration";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = icp_points + std::max(1, icp_points / 4);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    double sink = 0.0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(1);
        icp.align();
        sink += static_cast<double>(icp.getFitnessScore());
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (!std::isfinite(sink))
    {
        std::cerr << benchmark_name << " produced non-finite metrics\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationTargetOutputOneIteration(int icp_points, int iterations)
{
    const char* benchmark_name = "gpu_icp_finite_radius_translation_target_output_one_iteration";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = icp_points + std::max(1, icp_points / 4);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    std::vector<std::shared_ptr<Cloud<plamatrix::Device::GPU>>> targets;
    targets.reserve(static_cast<std::size_t>(iterations) + 1u);
    for (int i = 0; i < iterations + 1; ++i)
    {
        targets.push_back(std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu()));
    }

    std::size_t target_index = 0;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto& target = targets[target_index++];
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(1);
        icp.align(*target);
        sink += target->size();
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationSourceOutputOneIteration(int icp_points, int iterations)
{
    const char* benchmark_name = "gpu_icp_finite_radius_translation_source_output_one_iteration";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = icp_points + std::max(1, icp_points / 4);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    std::vector<std::shared_ptr<Cloud<plamatrix::Device::GPU>>> sources;
    sources.reserve(static_cast<std::size_t>(iterations) + 1u);
    for (int i = 0; i < iterations + 1; ++i)
    {
        sources.push_back(std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu()));
    }

    std::size_t source_index = 0;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto& source = sources[source_index++];
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.02f);
        icp.setMaxIterations(1);
        icp.align(*source);
        sink += source->size();
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
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

void benchmarkGpuIcpFiniteRadiusOrderedLowResidualOutputOneIteration(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_finite_radius_ordered_low_residual_output_one_iteration",
            "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.2f);
    icp.setMaxIterations(1);
    icp.setGpuAssumeOrderedCorrespondences(true);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(output);
        sink += output.size();
    });
    printResult(
        "gpu_icp_finite_radius_ordered_low_residual_output_one_iteration",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_finite_radius_ordered_low_residual_output_one_iteration"
            << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpOrderedInfiniteRadiusOutputOneIteration(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_ordered_infinite_radius_output_one_iteration", "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxIterations(1);
    icp.setGpuAssumeOrderedCorrespondences(true);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(output);
        sink += output.size();
    });
    printResult("gpu_icp_ordered_infinite_radius_output_one_iteration", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_ordered_infinite_radius_output_one_iteration produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationOrderedReuseTargetOutput(
    int icp_points,
    int icp_max_iterations,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_finite_radius_translation_ordered_reuse_target_output",
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
    icp.setGpuAssumeOrderedCorrespondences(true);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        icp.align(*target);
        sink += target->size();
    });
    printResult(
        "gpu_icp_finite_radius_translation_ordered_reuse_target_output",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_finite_radius_translation_ordered_reuse_target_output"
            << " produced no aligned points\n";
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

void benchmarkGpuIcpFiniteRadiusBinaryTranslationTransformOnly(
    const char* benchmark_name,
    int icp_points,
    int iterations,
    bool probe_transformed_exact_pointwise_on_cache_hit)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    constexpr float source_tx = 0.03125f;
    constexpr float source_ty = -0.015625f;
    constexpr float source_tz = 0.0078125f;
    auto target_points_cpu = makeBinaryGridPoints<float>(icp_points);
    auto target_to_source_cpu = makeTranslationTransform<float>(source_tx, source_ty, source_tz);
    auto source_points_cpu = plamatrix::transformPoints(target_to_source_cpu, target_points_cpu);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points_cpu));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points_cpu));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);
    if (probe_transformed_exact_pointwise_on_cache_hit)
    {
        icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);
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

void benchmarkGpuIcpFiniteRadiusBinaryTranslationReuseOutput(
    const char* benchmark_name,
    int icp_points,
    int icp_max_iterations,
    int iterations,
    bool probe_transformed_exact_pointwise_on_cache_hit)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    constexpr float source_tx = 0.03125f;
    constexpr float source_ty = -0.015625f;
    constexpr float source_tz = 0.0078125f;
    auto target_points_cpu = makeBinaryGridPoints<float>(icp_points);
    auto target_to_source_cpu = makeTranslationTransform<float>(source_tx, source_ty, source_tz);
    auto source_points_cpu = plamatrix::transformPoints(target_to_source_cpu, target_points_cpu);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points_cpu));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points_cpu));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.0625f);
    icp.setMaxIterations(icp_max_iterations);
    icp.setTransformationEpsilon(1.0e-12f);
    if (probe_transformed_exact_pointwise_on_cache_hit)
    {
        icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);
    }

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

void benchmarkGpuIcpFiniteRadiusNonRigidTransformOnly(
    const char* benchmark_name,
    int icp_points,
    int iterations,
    bool assume_ordered_correspondences,
    bool probe_transformed_exact_pointwise_on_cache_hit,
    bool cache_full_coverage_residual_results,
    bool assume_ordered_after_same_index_step,
    bool compute_final_metrics = false,
    bool prewarm_workspace = false)
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
    if (!compute_final_metrics)
    {
        icp.setComputeFinalMetrics(false);
    }
    if (assume_ordered_correspondences)
    {
        icp.setGpuAssumeOrderedCorrespondences(true);
    }
    if (probe_transformed_exact_pointwise_on_cache_hit)
    {
        icp.setGpuProbeTransformedExactPointwiseOnCacheHit(true);
    }
    if (cache_full_coverage_residual_results)
    {
        icp.setGpuCacheFullCoverageResidualResults(true);
    }
    if (assume_ordered_after_same_index_step)
    {
        icp.setGpuAssumeOrderedCorrespondencesAfterSameIndexStep(true);
    }
    if (prewarm_workspace)
    {
        icp.reserveGpuWorkspace();
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

void benchmarkGpuIcpFiniteRadiusNonRigidFinalMetricsOutput(
    const char* benchmark_name,
    int icp_points,
    int iterations,
    bool target_alias_output,
    bool same_size_output = false)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = target_alias_output
        ? icp_points
        : (same_size_output ? icp_points : icp_points + std::max(1, icp_points / 4));
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        if (target_alias_output)
        {
            icp.align(*target);
            sink += target->size();
        }
        else
        {
            icp.align(output);
            sink += output.size();
        }
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusNonRigidTargetAliasTransformOnlyTwoIterations(int icp_points, int iterations)
{
    constexpr const char* benchmark_name =
        "gpu_icp_finite_radius_nonrigid_target_alias_transform_only_two_iterations";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    std::vector<std::shared_ptr<Cloud<plamatrix::Device::GPU>>> targets;
    targets.reserve(static_cast<std::size_t>(iterations) + 1u);
    for (int i = 0; i < iterations + 1; ++i)
    {
        targets.push_back(std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu()));
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    std::size_t target_index = 0;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto& target = targets[target_index++];
        icp.setInputTarget(target);
        icp.align(*target);
        sink += target->size();
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusNonRigidOutputTransformOnlyTwoIterations(
    const char* benchmark_name,
    int icp_points,
    int iterations,
    bool same_size_output = false)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = same_size_output ? icp_points : icp_points + std::max(1, icp_points / 4);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

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

void benchmarkGpuIcpFiniteRadiusNonRigidOutputTransformOnlyFreshTargetTwoIterations(int icp_points, int iterations)
{
    constexpr const char* benchmark_name =
        "gpu_icp_finite_radius_nonrigid_output_transform_only_fresh_target_two_iterations";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = icp_points + std::max(1, icp_points / 4);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    std::vector<std::shared_ptr<Cloud<plamatrix::Device::GPU>>> targets;
    targets.reserve(static_cast<std::size_t>(iterations) + 1u);
    for (int i = 0; i < iterations + 1; ++i)
    {
        targets.push_back(std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu()));
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t target_index = 0;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto& target = targets[target_index++];
        icp.setInputTarget(target);
        icp.align(output);
        sink += output.size();
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationSourceAliasFinalMetricsTwoIterations(int icp_points, int iterations)
{
    constexpr const char* benchmark_name =
        "gpu_icp_finite_radius_translation_source_alias_final_metrics_two_iterations";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = icp_points + std::max(1, icp_points / 4);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    std::vector<std::shared_ptr<Cloud<plamatrix::Device::GPU>>> sources;
    sources.reserve(static_cast<std::size_t>(iterations) + 1u);
    for (int i = 0; i < iterations + 1; ++i)
    {
        sources.push_back(std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu()));
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    std::size_t source_index = 0;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto& source = sources[source_index++];
        icp.setInputSource(source);
        icp.align(*source);
        sink += source->size();
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpFiniteRadiusTranslationSourceAliasTransformOnlyTwoIterations(int icp_points, int iterations)
{
    constexpr const char* benchmark_name =
        "gpu_icp_finite_radius_translation_source_alias_transform_only_two_iterations";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    const int target_points = icp_points + std::max(1, icp_points / 4);
    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(target_points));
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    std::vector<std::shared_ptr<Cloud<plamatrix::Device::GPU>>> sources;
    sources.reserve(static_cast<std::size_t>(iterations) + 1u);
    for (int i = 0; i < iterations + 1; ++i)
    {
        sources.push_back(std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu()));
    }

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.02f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    std::size_t source_index = 0;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto& source = sources[source_index++];
        icp.setInputSource(source);
        icp.align(*source);
        sink += source->size();
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
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

void benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationNewWorkspaceOneSource(int icp_points, int iterations)
{
    constexpr const char* benchmark_name =
        "gpu_icp_alignment_step_finite_radius_translation_new_workspace_one_source";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(1, 0.003f, -0.002f, 0.001f));
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
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationRebuildReservedGrid(int icp_points, int iterations)
{
    constexpr const char* benchmark_name =
        "gpu_icp_alignment_step_finite_radius_translation_rebuild_reserved_grid";
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
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    stats_workspace.reserveAlignmentStep(static_cast<int>(source->size()));
    stats_workspace.reserveTargetSpatialGridForScalar<float>(static_cast<int>(target->size()));

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        stats_workspace.invalidateTargetSpatialGridCache();
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
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no correspondences\n";
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

void benchmarkGpuIcpAlignmentStepFiniteRadiusTranslationAsyncLaunchCachedGrid(
    int icp_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_finite_radius_translation_async_launch_cached_grid",
            "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

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
        const bool launched =
            plapoint::gpu::detail::launchIcpAlignmentStepColumnMajorWithReservedWorkspace(
                source->points().data(),
                static_cast<int>(source->size()),
                target->points().data(),
                static_cast<int>(target->size()),
                0.02f,
                stats_workspace,
                step_transform.data(),
                0);
        if (launched)
        {
            const auto result =
                plapoint::gpu::detail::copyAlignmentStepResultFromReservedWorkspace<float>(
                    stats_workspace,
                    0);
            sink += static_cast<std::size_t>(std::max(0, result.active_count));
        }
    });
    printResult(
        "gpu_icp_alignment_step_finite_radius_translation_async_launch_cached_grid",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_finite_radius_translation_async_launch_cached_grid"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepTransformedAccumulatedAsyncLaunchCachedGrid(
    int icp_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_transformed_accumulated_async_launch_cached_grid",
            "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_transform(4, 4);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source->points().data(),
        static_cast<int>(source->size()),
        target->points().data(),
        static_cast<int>(target->size()),
        0.02f,
        stats_workspace,
        first_step_transform.data());
    if (!first_step.step_valid)
    {
        printSkipped(
            "gpu_icp_alignment_step_transformed_accumulated_async_launch_cached_grid",
            "invalid_first_step");
        return;
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_transform(4, 4);
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const bool launched =
            plapoint::gpu::detail::
                launchTransformedIcpAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                    first_step_transform.data(),
                    source->points().data(),
                    static_cast<int>(source->size()),
                    target->points().data(),
                    static_cast<int>(target->size()),
                    0.02f,
                    stats_workspace,
                    step_transform.data(),
                    first_step_transform.data(),
                    accumulated_transform.data(),
                    0);
        if (launched)
        {
            const auto result =
                plapoint::gpu::detail::copyAlignmentStepResultFromReservedWorkspace<float>(
                    stats_workspace,
                    0);
            sink += static_cast<std::size_t>(std::max(0, result.active_count));
        }
    });
    printResult(
        "gpu_icp_alignment_step_transformed_accumulated_async_launch_cached_grid",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_transformed_accumulated_async_launch_cached_grid"
            << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepTwoStepAsyncLaunchSeparateWorkspaces(
    int icp_points,
    int iterations)
{
    constexpr const char* benchmark_name =
        "gpu_icp_alignment_step_two_step_async_launch_separate_workspaces";
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedGridPoints<float>(icp_points, 0.003f, -0.002f, 0.001f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(icp_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_step_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace second_step_workspace;
    first_step_workspace.reserveAlignmentStep(static_cast<int>(source->size()));
    second_step_workspace.reserveAlignmentStep(static_cast<int>(source->size()));
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_transform(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> second_step_transform(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const bool launched =
            plapoint::gpu::detail::launchIcpTwoStepAlignmentColumnMajorWithReservedWorkspaces(
                source->points().data(),
                static_cast<int>(source->size()),
                target->points().data(),
                static_cast<int>(target->size()),
                0.02f,
                first_step_workspace,
                second_step_workspace,
                first_step_transform.data(),
                second_step_transform.data(),
                accumulated_transform.data(),
                0);
        if (launched)
        {
            const auto result =
                plapoint::gpu::detail::copyIcpTwoStepAlignmentResultFromReservedWorkspaces<float>(
                    first_step_workspace,
                    second_step_workspace,
                    0);
            sink += static_cast<std::size_t>(std::max(0, result.first_alignment_step.active_count));
            sink += static_cast<std::size_t>(std::max(0, result.second_alignment_step.active_count));
        }
    });
    printResult(benchmark_name, icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no correspondences\n";
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

void benchmarkGpuIcpAlignmentStepTransformedExactPointwiseCacheHitPreflight(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(
            "gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit_preflight",
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
                step_transform.data(),
                0,
                false,
                true);
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult(
        "gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit_preflight",
        icp_points,
        iterations,
        elapsed);
    if (sink == 0)
    {
        std::cerr
            << "gpu_icp_alignment_step_transformed_exact_pointwise_cache_hit_preflight"
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

void benchmarkGpuIcpResidualStatsFiniteRadiusTranslationOrdered(int icp_points, int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped("gpu_icp_residual_stats_finite_radius_translation_ordered", "no_usable_cuda_device");
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
            stats_workspace,
            0,
            true);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult("gpu_icp_residual_stats_finite_radius_translation_ordered", icp_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << "gpu_icp_residual_stats_finite_radius_translation_ordered produced no correspondences\n";
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

void benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTarget(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
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
            0.08f,
            stats_workspace,
            step_transform.data());
        sink += static_cast<std::size_t>(std::max(0, result.active_count));
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpAlignmentStepSmallFiniteRadiusTargetAsyncLaunch(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_transform(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const bool launched =
            plapoint::gpu::detail::launchSmallTargetAlignmentStepColumnMajorWithReservedWorkspace(
                source->points().data(),
                static_cast<int>(source->size()),
                target->points().data(),
                static_cast<int>(target->size()),
                0.08f,
                stats_workspace,
                step_transform.data(),
                0);
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(0));
        sink += launched ? 1u : 0u;
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " did not launch\n";
    }
}

void benchmarkGpuIcpStatsStepSmallFiniteRadiusTarget(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
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
            0.08f,
            stats_workspace,
            step_transform.data(),
            step_workspace);
        sink += static_cast<std::size_t>(std::max(0, result.stats.active_count));
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpResidualStatsSmallFiniteRadiusTarget(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
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
            0.08f,
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpTransformResidualStatsSmallFiniteRadiusTarget(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> transform(4, 4);
    plapoint::gpu::setIdentityTransform4x4(transform.data());
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output(target_points, 3);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
        const auto stats = plapoint::gpu::transformPointsAndComputeIcpResidualStatsColumnMajor(
            transform.data(),
            source->points().data(),
            static_cast<int>(source->size()),
            target->points().data(),
            static_cast<int>(target->size()),
            0.08f,
            output.data(),
            stats_workspace);
        sink += static_cast<std::size_t>(std::max(0, stats.active_count));
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no correspondences\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusTransformOnlyTwoIterations(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        (void)source->points();
        icp.align();
        sink += static_cast<std::size_t>(icp.getFinalTransformationDevice().rows());
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no final transform\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusOutputTransformOnlyTwoIterations(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    Cloud<plamatrix::Device::GPU> output;
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        (void)source->points();
        icp.align(output);
        sink += output.size();
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusTargetAliasTransformOnlyTwoIterations(
    const std::string& benchmark_name,
    int target_points,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeTranslatedPerturbedCompactNonCollinearGridPoints<float>(target_points, 0.01f, -0.005f, 0.0025f));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(
        makeCompactNonCollinearGridPoints<float>(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());
    const auto target_snapshot = cpu_target->points().toGpu();

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);
    icp.setComputeFinalMetrics(false);

    std::size_t sink = 0;
    const auto reset_bytes = static_cast<std::size_t>(target_points) * 3u * sizeof(float);
    const double elapsed = bestMilliseconds(iterations, [&] {
        (void)source->points();
        PLAPOINT_CHECK_CUDA(cudaMemcpyAsync(
            target->points().data(),
            target_snapshot.data(),
            reset_bytes,
            cudaMemcpyDeviceToDevice,
            0));
        icp.align(*target);
        sink += target->size();
    });
    printResult(benchmark_name, target_points, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " produced no aligned points\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusFinalMetricsTwoIterations(
    const std::string& benchmark_name,
    int target_count,
    int iterations,
    int max_iterations = 2)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    double rmse_sink = 0.0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.08f);
        icp.setMaxIterations(max_iterations);
        icp.setTransformationEpsilon(1.0e-12f);
        icp.align();
        rmse_sink += static_cast<double>(icp.getFinalRmse());
        sink += std::isfinite(icp.getFinalRmse()) ? 1u : 0u;
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0 || !std::isfinite(rmse_sink))
    {
        std::cerr << benchmark_name << " produced no finite final metrics\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusReuseFinalMetricsTwoIterations(
    const std::string& benchmark_name,
    int target_count,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.08f);
    icp.setMaxIterations(2);
    icp.setTransformationEpsilon(1.0e-12f);

    std::size_t sink = 0;
    double rmse_sink = 0.0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        (void)source->points();
        icp.align();
        rmse_sink += static_cast<double>(icp.getFinalRmse());
        sink += std::isfinite(icp.getFinalRmse()) ? 1u : 0u;
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0 || !std::isfinite(rmse_sink))
    {
        std::cerr << benchmark_name << " produced no finite final metrics\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusOutputFinalMetricsTwoIterations(
    const std::string& benchmark_name,
    int target_count,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    auto target = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu());

    std::size_t sink = 0;
    double rmse_sink = 0.0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.08f);
        icp.setMaxIterations(2);
        icp.setTransformationEpsilon(1.0e-12f);
        Cloud<plamatrix::Device::GPU> output;
        icp.align(output);
        rmse_sink += static_cast<double>(icp.getFinalRmse());
        sink += output.size();
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0 || !std::isfinite(rmse_sink))
    {
        std::cerr << benchmark_name << " produced no finite output metrics\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusTargetAliasFinalMetricsTwoIterations(
    const std::string& benchmark_name,
    int target_count,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto cpu_source = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(source_points));
    auto cpu_target = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(target_points));
    auto source = std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_source->toGpu());
    std::vector<std::shared_ptr<Cloud<plamatrix::Device::GPU>>> targets;
    targets.reserve(static_cast<std::size_t>(iterations) + 1u);
    for (int i = 0; i <= iterations; ++i)
    {
        targets.push_back(std::make_shared<Cloud<plamatrix::Device::GPU>>(cpu_target->toGpu()));
    }

    std::size_t target_index = 0;
    std::size_t sink = 0;
    double rmse_sink = 0.0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        auto& target = targets[target_index++ % targets.size()];
        plapoint::IterativeClosestPoint<float, plamatrix::Device::GPU> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);
        icp.setMaxCorrespondenceDistance(0.08f);
        icp.setMaxIterations(2);
        icp.setTransformationEpsilon(1.0e-12f);
        icp.align(*target);
        rmse_sink += static_cast<double>(icp.getFinalRmse());
        sink += target->size();
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0 || !std::isfinite(rmse_sink))
    {
        std::cerr << benchmark_name << " produced no finite target-alias output metrics\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusTransformedAccumulatedAsyncLaunch(
    const std::string& benchmark_name,
    int target_count,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        first_step_gpu.data());
    if (!first_step.step_valid)
    {
        printSkipped(benchmark_name, "invalid_first_step");
        return;
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_gpu(4, 4);
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const bool launched =
            plapoint::gpu::detail::
                launchTransformedSmallTargetAlignmentStepAndAccumulateTransformColumnMajorWithReservedWorkspace(
                    first_step_gpu.data(),
                    source_gpu.data(),
                    static_cast<int>(source_gpu.rows()),
                    target_gpu.data(),
                    static_cast<int>(target_gpu.rows()),
                    0.08f,
                    stats_workspace,
                    step_gpu.data(),
                    first_step_gpu.data(),
                    accumulated_gpu.data(),
                    0);
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(0));
        sink += launched ? 1u : 0u;
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " did not launch\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusTerminalAsyncLaunch(
    const std::string& benchmark_name,
    int target_count,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace stats_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    const auto first_step = plapoint::gpu::computeIcpAlignmentStepColumnMajor(
        source_gpu.data(),
        static_cast<int>(source_gpu.rows()),
        target_gpu.data(),
        static_cast<int>(target_gpu.rows()),
        0.08f,
        stats_workspace,
        first_step_gpu.data());
    if (!first_step.step_valid)
    {
        printSkipped(benchmark_name, "invalid_first_step");
        return;
    }

    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> terminal_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source_gpu.rows(), 3);
    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const bool launched =
            plapoint::gpu::detail::
                launchTransformedSmallTargetTerminalAlignmentAndResidualColumnMajorWithReservedWorkspace(
                first_step_gpu.data(),
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                stats_workspace,
                terminal_step_gpu.data(),
                first_step_gpu.data(),
                accumulated_gpu.data(),
                0,
                output_gpu.data());
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(0));
        sink += launched ? 1u : 0u;
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " did not launch\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusTwoStepTerminalAsyncLaunch(
    const std::string& benchmark_name,
    int target_count,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_step_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace terminal_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> terminal_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> output_gpu(source_gpu.rows(), 3);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const bool launched =
            plapoint::gpu::detail::
                launchSmallTargetTwoStepTerminalAlignmentAndResidualColumnMajorWithReservedWorkspaces(
                    source_gpu.data(),
                    static_cast<int>(source_gpu.rows()),
                    target_gpu.data(),
                    static_cast<int>(target_gpu.rows()),
                    0.08f,
                    first_step_workspace,
                    terminal_workspace,
                    first_step_gpu.data(),
                    terminal_step_gpu.data(),
                    accumulated_gpu.data(),
                    0,
                    output_gpu.data());
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(0));
        sink += launched ? 1u : 0u;
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " did not launch\n";
    }
}

void benchmarkGpuIcpSmallFiniteRadiusTwoStepTransformOnlyAsyncLaunch(
    const std::string& benchmark_name,
    int target_count,
    int iterations)
{
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        printSkipped(benchmark_name, "no_usable_cuda_device");
        return;
    }
    const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);

    auto target_points = makeCompactNonCollinearGridPoints<float>(target_count);
    auto source_points =
        makeTranslatedCompactNonCollinearGridPoints<float>(target_count, 0.01f, -0.005f, 0.0025f);
    if (target_count > 3)
    {
        target_points(1, 0) += 0.03f;
        target_points(2, 1) -= 0.02f;
        target_points(3, 2) += 0.015f;
    }

    auto source_gpu = source_points.toGpu();
    auto target_gpu = target_points.toGpu();
    plapoint::gpu::IcpCorrespondenceStatsWorkspace first_step_workspace;
    plapoint::gpu::IcpCorrespondenceStatsWorkspace second_step_workspace;
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> first_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> second_step_gpu(4, 4);
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> accumulated_gpu(4, 4);

    std::size_t sink = 0;
    const double elapsed = bestMilliseconds(iterations, [&] {
        const bool launched =
            plapoint::gpu::detail::launchSmallTargetTwoStepAlignmentColumnMajorWithReservedWorkspaces(
                source_gpu.data(),
                static_cast<int>(source_gpu.rows()),
                target_gpu.data(),
                static_cast<int>(target_gpu.rows()),
                0.08f,
                first_step_workspace,
                second_step_workspace,
                first_step_gpu.data(),
                second_step_gpu.data(),
                accumulated_gpu.data(),
                0);
        PLAPOINT_CHECK_CUDA(cudaStreamSynchronize(0));
        sink += launched ? 1u : 0u;
    });
    printResult(benchmark_name, target_count, iterations, elapsed);
    if (sink == 0)
    {
        std::cerr << benchmark_name << " did not launch\n";
    }
}
#endif

