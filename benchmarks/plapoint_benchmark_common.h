// Included once by plapoint_benchmarks.cpp inside its anonymous namespace.
using Clock = std::chrono::steady_clock;

#ifdef PLAPOINT_WITH_CUDA
bool g_synchronize_cuda_after_benchmark_iteration = false;

void synchronizeCudaBenchmarkDevice();

class ScopedCudaBenchmarkSynchronization
{
public:
    explicit ScopedCudaBenchmarkSynchronization(bool enabled)
        : _previous(g_synchronize_cuda_after_benchmark_iteration)
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
    int iterations = 3;
    int icp_points = 512;
    int icp_max_iterations = 3;
    bool skip_cpu_icp = false;
    bool skip_icp_identity = false;
    bool self_test_benchmark_gpu_sync = false;
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
        else if (arg == "--skip-cpu-icp")
        {
            options.skip_cpu_icp = true;
        }
        else if (arg == "--skip-icp-identity")
        {
            options.skip_icp_identity = true;
        }
        else if (arg == "--self-test-benchmark-gpu-sync")
        {
            options.self_test_benchmark_gpu_sync = true;
        }
        else if (arg == "--help")
        {
            std::cout
                << "Usage: plapoint_benchmarks [--points N] [--iterations N]\n"
                << "                           [--icp-points N] [--icp-max-iterations N]\n"
                << "                           [--skip-cpu-icp] [--skip-icp-identity]\n"
                << "                           [--self-test-benchmark-gpu-sync]\n";
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("Unknown option: " + arg);
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
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeTranslatedCompactNonCollinearGridPoints(
    int count,
    Scalar tx,
    Scalar ty,
    Scalar tz)
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
plamatrix::DenseMatrix<Scalar, plamatrix::Device::CPU> makeTranslatedPerturbedCompactNonCollinearGridPoints(
    int count,
    Scalar tx,
    Scalar ty,
    Scalar tz)
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

template <typename Fn, typename SyncFn>
double bestMilliseconds(int iterations, Fn&& fn, SyncFn&& sync)
{
    // Exclude one-time CUDA context, Thrust, and allocator startup from the measured loop.
    fn();
    sync();

    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < iterations; ++i)
    {
        const auto start = Clock::now();
        fn();
        sync();
        const auto end = Clock::now();
        const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        best = std::min(best, elapsed);
    }
    return best;
}

template <typename Fn>
double bestMilliseconds(int iterations, Fn&& fn)
{
    return bestMilliseconds(iterations, std::forward<Fn>(fn), [] {
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
        bestMilliseconds(iterations, [&] {
            PLAPOINT_CHECK_CUDA(cudaLaunchHostFunc(0, benchmarkGpuSyncSelfTestCallback, &callback_count));
        });
    }

    const int expected_callbacks = iterations + 1;
    if (callback_count != expected_callbacks)
    {
        std::cerr << "benchmark_gpu_sync_self_test expected "
                  << expected_callbacks << " callbacks, got " << callback_count << '\n';
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

template <plamatrix::Device Dev>
using Cloud = plapoint::PointCloud<float, Dev>;
