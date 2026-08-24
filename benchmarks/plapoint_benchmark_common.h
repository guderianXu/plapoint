// Included once by plapoint_benchmarks.cpp inside its anonymous namespace.
using Clock = std::chrono::steady_clock;

#ifdef PLAPOINT_WITH_CUDA
bool g_synchronize_cuda_after_benchmark_iteration = false;

void synchronizeCudaBenchmarkDevice();

class ScopedCudaBenchmarkSynchronization
{
public:
    explicit ScopedCudaBenchmarkSynchronization(bool enabled) : _previous(g_synchronize_cuda_after_benchmark_iteration)
    {
        g_synchronize_cuda_after_benchmark_iteration = enabled;
    }

    ~ScopedCudaBenchmarkSynchronization()
    {
        g_synchronize_cuda_after_benchmark_iteration = _previous;
    }

private:
    bool _previous;
};
#endif

struct Options
{
    int points = 20000;
    int iterations = 7;
    int icp_points = 512;
    int icp_max_iterations = 3;
    int poisson_points = 8192;
    int poisson_depth = 6;
    bool skip_cpu_icp = false;
    bool skip_icp_identity = false;
    bool search_features_only = false;
    bool mesh_only = false;
    bool self_test_benchmark_gpu_sync = false;
    bool self_test_benchmark_statistics = false;
};

int parseIntegerOption(const std::string& option, const std::string& value, int minimum)
{
    int parsed = 0;
    const auto* begin = value.data();
    const auto* end = value.data() + value.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc() || result.ptr != end)
    {
        throw std::invalid_argument("Invalid value for " + option + ": " + value);
    }
    if (parsed < minimum)
    {
        throw std::invalid_argument("Invalid value for " + option + ": " + value);
    }
    return parsed;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--points")
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for --points");
            }
            options.points = parseIntegerOption(arg, argv[++i], 1);
        }
        else if (arg == "--iterations")
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for --iterations");
            }
            options.iterations = parseIntegerOption(arg, argv[++i], 1);
        }
        else if (arg == "--icp-points")
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for --icp-points");
            }
            options.icp_points = parseIntegerOption(arg, argv[++i], 3);
        }
        else if (arg == "--icp-max-iterations")
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for --icp-max-iterations");
            }
            options.icp_max_iterations = parseIntegerOption(arg, argv[++i], 1);
        }
        else if (arg == "--poisson-points")
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for --poisson-points");
            }
            options.poisson_points = parseIntegerOption(arg, argv[++i], 48);
        }
        else if (arg == "--poisson-depth")
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for --poisson-depth");
            }
            options.poisson_depth = parseIntegerOption(arg, argv[++i], 1);
            if (options.poisson_depth > 8)
            {
                throw std::invalid_argument("Invalid value for --poisson-depth: " +
                                            std::to_string(options.poisson_depth));
            }
        }
        else if (arg == "--skip-cpu-icp")
        {
            options.skip_cpu_icp = true;
        }
        else if (arg == "--skip-icp-identity")
        {
            options.skip_icp_identity = true;
        }
        else if (arg == "--search-features-only")
        {
            options.search_features_only = true;
        }
        else if (arg == "--mesh-only")
        {
            options.mesh_only = true;
        }
        else if (arg == "--self-test-benchmark-gpu-sync")
        {
            options.self_test_benchmark_gpu_sync = true;
        }
        else if (arg == "--self-test-benchmark-statistics")
        {
            options.self_test_benchmark_statistics = true;
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: plapoint_benchmarks [--points N] [--iterations N]\n"
                      << "                           [--icp-points N] [--icp-max-iterations N]\n"
                      << "                           [--poisson-points N] [--poisson-depth 1..8]\n"
                      << "                           [--skip-cpu-icp] [--skip-icp-identity]\n"
                      << "                           [--search-features-only] [--mesh-only]\n"
                      << "                           [--self-test-benchmark-gpu-sync]\n"
                      << "                           [--self-test-benchmark-statistics]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    const bool runs_poisson_benchmark =
        options.mesh_only && !options.self_test_benchmark_gpu_sync && !options.self_test_benchmark_statistics;
    int minimum_poisson_points = 1;
    for (int level = 0; level < options.poisson_depth; ++level)
    {
        minimum_poisson_points *= 4;
    }
    if (runs_poisson_benchmark && options.poisson_points < minimum_poisson_points)
    {
        throw std::invalid_argument("Poisson benchmark sampling is too sparse: --poisson-points " +
                                    std::to_string(options.poisson_points) + " with --poisson-depth " +
                                    std::to_string(options.poisson_depth) + "; use at least " +
                                    std::to_string(minimum_poisson_points) + " points or reduce the depth");
    }
    return options;
}

template <typename Scalar> plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeGridPoints(int count)
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
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>
makeTranslatedGridPoints(int count, Scalar tx, Scalar ty, Scalar tz)
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
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>
makeTranslatedPerturbedGridPoints(int count, Scalar tx, Scalar ty, Scalar tz)
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

template <typename Scalar> plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeBinaryGridPoints(int count)
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
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeCompactNonCollinearGridPoints(int count)
{
    plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> points(count, 3);
    for (int i = 0; i < count; ++i)
    {
        if (i < 4)
        {
            points(i, 0) = (i == 1) ? Scalar(0.25) : Scalar(0);
            points(i, 1) = (i == 2) ? Scalar(0.25) : Scalar(0);
            points(i, 2) = (i == 3) ? Scalar(0.25) : Scalar(0);
        }
        else
        {
            const int j = i - 4;
            const int x = j % 8;
            const int y = (j / 8) % 8;
            const int z = j / 64;
            points(i, 0) = static_cast<Scalar>(x) * Scalar(0.25) + Scalar(0.125);
            points(i, 1) = static_cast<Scalar>(y) * Scalar(0.25) + Scalar(0.125);
            points(i, 2) = static_cast<Scalar>(z) * Scalar(0.25) + Scalar(0.125);
        }
    }
    return points;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>
makeTranslatedCompactNonCollinearGridPoints(int count, Scalar tx, Scalar ty, Scalar tz)
{
    auto points = makeCompactNonCollinearGridPoints<Scalar>(count);
    for (int i = 0; i < count; ++i)
    {
        points(i, 0) += tx;
        points(i, 1) += ty;
        points(i, 2) += tz;
    }
    return points;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU>
makeTranslatedPerturbedCompactNonCollinearGridPoints(int count, Scalar tx, Scalar ty, Scalar tz)
{
    auto points = makeTranslatedCompactNonCollinearGridPoints<Scalar>(count, tx, ty, tz);
    for (int i = 0; i < count; ++i)
    {
        points(i, 0) += static_cast<Scalar>((i % 7) - 3) * Scalar(0.0002);
        points(i, 1) += static_cast<Scalar>(((i / 7) % 5) - 2) * Scalar(0.00015);
        points(i, 2) += static_cast<Scalar>(((i / 35) % 3) - 1) * Scalar(0.0001);
    }
    return points;
}

template <typename Scalar>
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeTranslationTransform(Scalar tx, Scalar ty, Scalar tz)
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

template <typename Scalar> plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeQueries(int count)
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

struct BenchmarkTimingStatistics
{
    double best_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double stddev_ms = 0.0;
    double cv = 0.0;
};

BenchmarkTimingStatistics g_last_benchmark_timing;
bool g_has_last_benchmark_timing = false;

BenchmarkTimingStatistics summarizeMilliseconds(std::vector<double> samples)
{
    if (samples.empty())
    {
        throw std::invalid_argument("Cannot summarize an empty benchmark sample set");
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t sample_count = samples.size();
    const std::size_t middle = sample_count / 2;
    const double median = sample_count % 2 == 0 ? (samples[middle - 1] + samples[middle]) * 0.5 : samples[middle];
    const std::size_t p95_rank = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(sample_count)));

    double mean = 0.0;
    for (const double sample : samples)
    {
        mean += sample;
    }
    mean /= static_cast<double>(sample_count);

    double squared_deviation_sum = 0.0;
    for (const double sample : samples)
    {
        const double deviation = sample - mean;
        squared_deviation_sum += deviation * deviation;
    }
    const double stddev = std::sqrt(squared_deviation_sum / static_cast<double>(sample_count));

    BenchmarkTimingStatistics result;
    result.best_ms = samples.front();
    result.median_ms = median;
    result.p95_ms = samples[std::max<std::size_t>(1, p95_rank) - 1];
    result.stddev_ms = stddev;
    result.cv = mean > 0.0 ? stddev / mean : 0.0;
    return result;
}

int runBenchmarkStatisticsSelfTest()
{
    const auto timing = summarizeMilliseconds({4.0, 1.0, 3.0, 2.0});
    const double expected_stddev = std::sqrt(1.25);
    const double expected_cv = expected_stddev / 2.5;
    constexpr double tolerance = 1.0e-12;
    if (std::abs(timing.best_ms - 1.0) > tolerance || std::abs(timing.median_ms - 2.5) > tolerance ||
        std::abs(timing.p95_ms - 4.0) > tolerance || std::abs(timing.stddev_ms - expected_stddev) > tolerance ||
        std::abs(timing.cv - expected_cv) > tolerance)
    {
        std::cerr << "benchmark_statistics_self_test failed\n";
        return 1;
    }

    std::cout << "benchmark_statistics_self_test,passed\n";
    return 0;
}

template <typename Fn, typename SyncFn> double bestMilliseconds(int iterations, Fn&& fn, SyncFn&& sync)
{
    // Exclude one-time CUDA context, Thrust, and allocator startup from the measured loop.
    fn();
    sync();

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i)
    {
        const auto start = Clock::now();
        fn();
        sync();
        const auto end = Clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        samples.push_back(elapsed);
    }
    g_last_benchmark_timing = summarizeMilliseconds(std::move(samples));
    g_has_last_benchmark_timing = true;
    return g_last_benchmark_timing.best_ms;
}

template <typename Fn> double bestMilliseconds(int iterations, Fn&& fn)
{
    return bestMilliseconds(iterations,
                            std::forward<Fn>(fn),
                            []
                            {
#ifdef PLAPOINT_WITH_CUDA
                                if (g_synchronize_cuda_after_benchmark_iteration)
                                {
                                    synchronizeCudaBenchmarkDevice();
                                }
#endif
                            });
}

void printResult(const std::string& name, int points, int iterations, double milliseconds)
{
    BenchmarkTimingStatistics timing;
    if (g_has_last_benchmark_timing && g_last_benchmark_timing.best_ms == milliseconds)
    {
        timing = g_last_benchmark_timing;
    }
    else
    {
        timing.best_ms = milliseconds;
        timing.median_ms = milliseconds;
        timing.p95_ms = milliseconds;
    }

    std::cout << name << ',' << points << ',' << iterations << ',' << timing.best_ms << ',' << timing.median_ms << ','
              << timing.p95_ms << ',' << timing.stddev_ms << ',' << timing.cv << '\n';
}

void printSkipped(const std::string& name, const std::string& reason)
{
    std::cout << name << ",skipped," << reason << ",,,,,\n";
}

#ifdef PLAPOINT_WITH_CUDA
void synchronizeCudaBenchmarkDevice()
{
    PLAPOINT_CHECK_CUDA(cudaDeviceSynchronize());
}

void CUDART_CB benchmarkGpuSyncSelfTestCallback(void* user_data)
{
    auto* callback_count = static_cast<int*>(user_data);
    ++(*callback_count);
}

int runBenchmarkGpuSyncSelfTest()
{
    constexpr int iterations = 3;
    int callback_count = 0;

    if (!plapoint::gpu::hasUsableCudaDevice())
    {
        std::cout << "benchmark_gpu_sync_self_test,skipped,no_usable_cuda_device\n";
        return 0;
    }

    {
        const ScopedCudaBenchmarkSynchronization scoped_sync(true);
        bestMilliseconds(
            iterations,
            [&] { PLAPOINT_CHECK_CUDA(cudaLaunchHostFunc(0, benchmarkGpuSyncSelfTestCallback, &callback_count)); });
    }

    const int expected_callbacks = iterations + 1;
    if (callback_count != expected_callbacks)
    {
        std::cerr << "benchmark_gpu_sync_self_test expected " << expected_callbacks << " callbacks, got "
                  << callback_count << '\n';
        return 1;
    }

    std::cout << "benchmark_gpu_sync_self_test,passed\n";
    if (g_synchronize_cuda_after_benchmark_iteration)
    {
        std::cerr << "benchmark_gpu_sync_scope_self_test expected sync to start disabled\n";
        return 1;
    }
    {
        const ScopedCudaBenchmarkSynchronization scoped_sync(true);
        if (!g_synchronize_cuda_after_benchmark_iteration)
        {
            std::cerr << "benchmark_gpu_sync_scope_self_test failed to enable sync\n";
            return 1;
        }
        {
            const ScopedCudaBenchmarkSynchronization scoped_launch_only(false);
            if (g_synchronize_cuda_after_benchmark_iteration)
            {
                std::cerr << "benchmark_gpu_sync_scope_self_test failed to disable nested sync\n";
                return 1;
            }
        }
        if (!g_synchronize_cuda_after_benchmark_iteration)
        {
            std::cerr << "benchmark_gpu_sync_scope_self_test failed to restore enabled sync\n";
            return 1;
        }
    }
    if (g_synchronize_cuda_after_benchmark_iteration)
    {
        std::cerr << "benchmark_gpu_sync_scope_self_test failed to restore disabled sync\n";
        return 1;
    }
    std::cout << "benchmark_gpu_sync_scope_self_test,passed\n";
    return 0;
}
#endif

template <plamatrix::Device Dev> using Cloud = plapoint::PointCloud<float, Dev>;
