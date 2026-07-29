// Included once by plapoint_benchmarks.cpp inside its anonymous namespace.
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

void benchmarkCpuNormalSmoothing(int points, int iterations)
{
    const int normal_points = std::min(points, 3000);
    auto cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(normal_points));
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(normal_points, 3);
    normals.fill(0.0f);
    for (int row = 0; row < normal_points; ++row)
    {
        normals(row, 2) = 1.0f;
    }
    cloud->setNormals(std::move(normals));
    auto tree = std::make_shared<plapoint::search::KdTree<float, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();

    const double elapsed = bestMilliseconds(iterations, [&] {
        plapoint::NormalRefinement<float, plamatrix::Device::CPU> refinement;
        refinement.setInputCloud(cloud);
        refinement.setSearchMethod(tree);
        refinement.smooth(8);
    });
    printResult("cpu_normal_smoothing_k8", normal_points, iterations, elapsed);
}

void benchmarkCpuStatisticalOutlierRemoval(int points, int iterations)
{
    auto cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(points));
    auto tree = std::make_shared<plapoint::search::KdTree<float, plamatrix::Device::CPU>>();
    tree->setInputCloud(cloud);
    tree->build();
    plapoint::StatisticalOutlierRemoval<float, plamatrix::Device::CPU> filter;
    filter.setInputCloud(cloud);
    filter.setSearchMethod(tree);
    filter.setMeanK(8);
    Cloud<plamatrix::Device::CPU> output;

    const double elapsed = bestMilliseconds(iterations, [&] {
        filter.filter(output);
    });
    printResult("cpu_statistical_outlier_removal_k8", points, iterations, elapsed);
}

void benchmarkCpuRadiusOutlierRemoval(int points, int iterations)
{
    auto cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(makeGridPoints<float>(points));
    plapoint::RadiusOutlierRemoval<float, plamatrix::Device::CPU> filter;
    filter.setInputCloud(cloud);
    filter.setRadius(0.011f);
    filter.setMinNeighbors(1);
    Cloud<plamatrix::Device::CPU> output;

    const double elapsed = bestMilliseconds(iterations, [&] {
        filter.filter(output);
    });
    printResult("cpu_radius_outlier_removal", points, iterations, elapsed);
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

