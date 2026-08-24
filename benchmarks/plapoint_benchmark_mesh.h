// Included once by plapoint_benchmarks.cpp inside its anonymous namespace.
#ifdef PLAPOINT_WITH_CUDA

std::shared_ptr<Cloud<plamatrix::Device::CPU>> makePoissonBenchmarkCloud(int count)
{
    count = std::max(count, 48);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(count, 3);
    constexpr float pi = 3.14159265358979323846f;
    constexpr float golden_angle = pi * (3.0f - 2.2360679774997896964f);
    for (int index = 0; index < count; ++index)
    {
        const float z = 1.0f - 2.0f * (static_cast<float>(index) + 0.5f) / static_cast<float>(count);
        const float radius = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float theta = golden_angle * static_cast<float>(index);
        const float x = radius * std::cos(theta);
        const float y = radius * std::sin(theta);
        points.setValue(index, 0, x);
        points.setValue(index, 1, y);
        points.setValue(index, 2, z);
        normals.setValue(index, 0, x);
        normals.setValue(index, 1, y);
        normals.setValue(index, 2, z);
    }
    auto cloud = std::make_shared<Cloud<plamatrix::Device::CPU>>(std::move(points));
    cloud->setNormals(std::move(normals));
    return cloud;
}

Cloud<plamatrix::Device::CPU> makeHeightGridBenchmarkCloud(int count, int& side)
{
    side = std::max(4, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count)))));
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(count, 3);
    for (int index = 0; index < count; ++index)
    {
        const int x = index % side;
        const int y = index / side;
        points.setValue(index, 0, static_cast<float>(x));
        points.setValue(index, 1, static_cast<float>(y));
        points.setValue(index, 2, std::sin(static_cast<float>(x) * 0.2f) + std::cos(static_cast<float>(y) * 0.15f));
    }
    return Cloud<plamatrix::Device::CPU>(std::move(points));
}

void benchmarkMarchingCubesField(int points, int iterations)
{
    const int cubes = std::clamp(static_cast<int>(std::cbrt(static_cast<double>(points))) * 2, 8, 24);
    const int samples = cubes + 1;
    const int sample_count = samples * samples * samples;
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> field_cpu(sample_count, 1);
    for (int z = 0; z < samples; ++z)
    {
        for (int y = 0; y < samples; ++y)
        {
            for (int x = 0; x < samples; ++x)
            {
                const float fx = static_cast<float>(x) / static_cast<float>(cubes) * 2.0f - 1.0f;
                const float fy = static_cast<float>(y) / static_cast<float>(cubes) * 2.0f - 1.0f;
                const float fz = static_cast<float>(z) / static_cast<float>(cubes) * 2.0f - 1.0f;
                const int offset = x + samples * (y + samples * z);
                field_cpu.setValue(offset, 0, fx * fx + fy * fy + fz * fz - 0.55f);
            }
        }
    }
    const auto field_gpu = field_cpu.toGpu();
    plapoint::gpu::MarchingCubesGpuWorkspace<float> workspace;
    Cloud<plamatrix::Device::GPU> output;
    const double elapsed =
        bestMilliseconds(iterations,
                         [&]()
                         {
                             output = plapoint::gpu::marchingCubes(field_gpu,
                                                                   cubes,
                                                                   cubes,
                                                                   cubes,
                                                                   plamatrix::Vec3<float>{-1.0f, -1.0f, -1.0f},
                                                                   plamatrix::Vec3<float>{1.0f, 1.0f, 1.0f},
                                                                   0.0f,
                                                                   workspace);
                         });
    printResult("marching_cubes_field", sample_count, iterations, elapsed);
    if (output.size() == 0)
    {
        throw std::runtime_error("marching_cubes_field produced no vertices");
    }
}

void benchmarkHeightGridFill(int points, int iterations)
{
    int side = 0;
    auto cloud_cpu = makeHeightGridBenchmarkCloud(points, side);
    const auto cloud_gpu = cloud_cpu.toGpu();
    plapoint::mesh::HeightGridOptions<float> options;
    options.width = side;
    options.height = side;
    options.useExplicitBounds = true;
    options.minX = 0.0f;
    options.maxX = static_cast<float>(side - 1);
    options.minY = 0.0f;
    options.maxY = static_cast<float>(side - 1);
    options.useBilinearSplat = false;
    plapoint::gpu::HeightGridGpuWorkspace<float> workspace;
    std::size_t sink = 0;
    const double elapsed =
        bestMilliseconds(iterations,
                         [&]()
                         {
                             auto grid = plapoint::gpu::buildHeightGridDeviceAsync(cloud_gpu, options, workspace);
                             plapoint::gpu::fillHolesAsync(grid, 4, 1, 1, workspace);
                             grid.synchronize();
                             sink += grid.cellCount();
                         });
    printResult("height_grid_fill", points, iterations, elapsed);
    if (sink == 0)
    {
        throw std::runtime_error("height_grid_fill produced an empty grid");
    }
}

struct PoissonBenchmarkFixture
{
    std::shared_ptr<Cloud<plamatrix::Device::CPU>> cloud;
    plapoint::mesh::PoissonReconstruction<float> assembled;
    int depth = 0;
};

PoissonBenchmarkFixture preparePoissonBenchmark(int points, int depth)
{
    PoissonBenchmarkFixture fixture;
    fixture.cloud = makePoissonBenchmarkCloud(points);
    fixture.depth = depth;
    fixture.assembled.setInputCloud(fixture.cloud);
    fixture.assembled.setDepth(depth);
    fixture.assembled.setSolverIterations(500);
    fixture.assembled.setSolverTolerance(1.0e-3);
    plamatrix::Index preflight_vertex_count = 0;
    {
        auto preflight_mesh = fixture.assembled.reconstruct();
        preflight_vertex_count = std::get<0>(preflight_mesh).rows();
    }

    const auto& preflight_report = fixture.assembled.lastReport();
    if (preflight_vertex_count == 0 || !(preflight_report.fieldMinimum < preflight_report.fieldMaximum))
    {
        throw std::runtime_error("Poisson benchmark preflight produced a constant field or empty mesh; "
                                 "increase --poisson-points or reduce --poisson-depth");
    }
    return fixture;
}

void benchmarkPoisson(PoissonBenchmarkFixture& fixture, int iterations)
{
    const auto& system = fixture.assembled.lastSystem();
    const auto matrix_gpu = system.matrix.toGpu();
    const auto rhs_gpu = system.rhs.toGpu();
    plamatrix::DenseMatrix<float, plamatrix::Device::GPU> solution(system.matrix.rows(), 1);
    plamatrix::IterativeSolverWorkspace<float> workspace;
    plamatrix::IterativeSolverOptions solver_options;
    solver_options.maxIterations = 500;
    solver_options.relativeTolerance = 1.0e-3;
    plamatrix::IterativeSolverReport solver_report;
    double elapsed = bestMilliseconds(iterations,
                                      [&]()
                                      {
                                          solution.fill(0.0f);
                                          solver_report =
                                              plamatrix::pcg(matrix_gpu, rhs_gpu, solution, workspace, solver_options);
                                      });
    printResult("poisson_solve", static_cast<int>(system.leafNodes.size()), iterations, elapsed);
    if (!solver_report.converged)
    {
        throw std::runtime_error(std::string("poisson_solve did not converge after ") +
                                 std::to_string(solver_report.iterations) +
                                 " iterations; residual=" + std::to_string(solver_report.finalResidual));
    }

    plapoint::mesh::PoissonReconstruction<float> reconstruction;
    reconstruction.setInputCloud(fixture.cloud);
    reconstruction.setDepth(fixture.depth);
    reconstruction.setSolverIterations(500);
    reconstruction.setSolverTolerance(1.0e-3);
    reconstruction.setProcessingDevice(plapoint::ProcessingDevice::GPU);
    plamatrix::Index vertex_sink = 0;
    elapsed = bestMilliseconds(iterations,
                               [&]()
                               {
                                   auto mesh = reconstruction.reconstruct();
                                   vertex_sink += std::get<0>(mesh).rows();
                               });
    printResult("poisson_end_to_end", static_cast<int>(fixture.cloud->size()), iterations, elapsed);
    if (vertex_sink == 0)
    {
        throw std::runtime_error("poisson_end_to_end produced no vertices");
    }
}

void benchmarkGpuMesh(int points, int iterations, int poisson_points, int poisson_depth)
{
    const std::vector<std::string> rows = {
        "marching_cubes_field", "height_grid_fill", "poisson_solve", "poisson_end_to_end"};
    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        for (const auto& row : rows)
        {
            printSkipped(row, "no_usable_cuda_device");
        }
        return;
    }
    auto poisson_fixture = preparePoissonBenchmark(poisson_points, poisson_depth);
    benchmarkMarchingCubesField(points, iterations);
    benchmarkHeightGridFill(points, iterations);
    benchmarkPoisson(poisson_fixture, iterations);
}

#endif // PLAPOINT_WITH_CUDA
