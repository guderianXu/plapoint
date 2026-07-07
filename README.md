# PlaPoint

GPU-accelerated point cloud processing library built on [PlaMatrix](https://github.com/guderianXu/plamatrix).

## Features

### Core
- **PointCloud\<Scalar, Dev\>** — Nx3 point cloud with optional normals, colors, intensities, texture coordinates, mesh/material metadata, and named scalar fields. CPU/GPU transfer uses `toGpu()`/`toCpu()`, and `pointsCpu()` provides a cached CPU point view for GPU clouds.

### Spatial Indexing
- **KdTree\<Scalar, Dev\>** — 3D kd-tree with KNN search (priority queue) and radius search. Call `build()` after `setInputCloud()`; searches throw a clear exception if the tree has not been built.

### Filters
- **VoxelGrid** — centroid-based voxel downsampling with mean aggregation for normals, colors, intensities, and named scalar fields
- **StatisticalOutlierRemoval** — KNN distance statistics outlier filtering
- **RadiusOutlierRemoval** — radius-based neighbor count filtering
- **UniformDownsample** — keep every Nth point
- **Filter\<Scalar, Dev\>** — abstract base class for all filters

Point-index preserving filters and GPU gather helpers keep point-aligned attributes
(`normals`, `colors`, `intensities`, and named scalar fields) for the selected
points. Mesh compaction uses the same rule for surviving vertices. VoxelGrid is
the main aggregation filter: it averages per-voxel scalar fields instead of
dropping them.

### Features
- **NormalEstimation** — PCA-based surface normal estimation via covariance + SVD
- **NormalRefinement** — normal smoothing (KNN averaging) and viewpoint-based orientation

### Registration
- **IterativeClosestPoint** (ICP) — point-to-point ICP with SVD-based rigid transform, initial guess support, and PCL-style correspondence/convergence controls

### Mesh
- **MarchingCubes** — isosurface extraction from implicit scalar fields
- **PoissonReconstruction** — Poisson surface reconstruction (Gauss-Seidel solver + MC extraction)

### I/O
- **PLY** — ASCII read/write with positions and optional normals
- **XYZ** — strict-by-default text XYZ reader. Strict rows must be exactly `x y z` or `x y z r g b`; malformed rows throw with file path and line number. Pass `io::XyzReadMode::Permissive` to skip bad legacy rows and keep the older trailing-column tolerance.

## GPU Acceleration

When `PLAPOINT_WITH_CUDA=ON`, CUDA Toolkit is available, and `plamatrix::plamatrix`
was built with CUDA support:

- **Brute-force KNN** (`src/knn_gpu.cu`) — batched K-nearest neighbor search with one CUDA block per query point, shared memory top-K reduction. `KdTree<Scalar, GPU>::batchNearestKSearch()` reads PlaMatrix column-major device buffers directly.
- **Stream-aware device KNN** — `gpu::batchKnnDeviceAsync()` and `gpu::batchKnnDeviceColumnMajorAsync()` launch on a caller-provided `cudaStream_t`; existing non-async overloads preserve synchronous behavior.
- **VoxelGrid CUDA downsampling** (`src/voxel_grid_gpu.cu`) — GPU path computes voxel keys, sorts them, reduces centroids, and preserves deterministic sorted voxel-key output.
- **ICP GPU path** (`src/icp_gpu.cu`) — `IterativeClosestPoint<Scalar, GPU>` keeps source/target point buffers on GPU, reads the initial source buffer directly without a startup device-to-device copy, computes correspondences with a cached finite-radius target spatial grid or the shared-memory target-tiling fallback, uses precomputed finite-radius tile bounding-box skips and per-candidate axis pruning on the fallback path, accumulates centroid/covariance/residual stats with block-level reductions, derives degeneracy flags from covariance invariants, fuses stats reduction with device-side step-transform solving through a CUDA quaternion/Jacobi solver, applies point transforms through persistent GPU scratch buffers, initializes and asynchronously accumulates the final 4x4 transform on GPU, and writes terminal-iteration transforms directly into a plain non-input caller output cloud when possible. Reduced stats, step deltas, and metric checks still synchronize to CPU, while `getFinalTransformationDevice()` exposes the final transform without forcing callers through the CPU copy and the legacy CPU `getFinalTransformation()` materializes that copy lazily. The stats helper can skip per-source correspondence index output when callers only need aggregate ICP moments, persistent workspaces and GPU buffers avoid repeated reduction, target spatial-grid, target-tile bound, step-solver, transform-buffer, point-scratch, output-allocation, and final output-copy overhead across repeated `align()` calls on the same ICP object and plain same-shaped output cloud, and `alignGpu()` skips transformed final-stats scans on non-terminal iterations. `setComputeFinalMetrics(false)` is an opt-in throughput mode that skips the terminal post-transform fitness/RMSE scan when callers only need the transform or aligned output. Input-aliased, attributed, or metadata-bearing output clouds still use safe scratch/copy or replacement paths so stale normals, colors, intensities, named scalar fields, mesh, material, or texture data cannot leak into aligned-point results. PCL-style robust ICP options that need correspondence rejectors currently preserve GPU input/output types through a CPU-staged semantic fallback; the default CUDA fast path remains unchanged for the base ICP configuration.
- **CPU-staged GPU fallbacks** — remaining filters and normal estimation/refinement preserve GPU input/output types but stage data through CPU for algorithms that do not yet have production CUDA kernels. GPU point staging is cached by `PointCloud::pointsCpu()` and invalidated when mutable `points()` is requested.
- **VoxelGrid CPU hot path** — CPU path uses hash aggregation and sorted voxel keys to keep deterministic centroid order.
- Explicit template instantiations in `src/plapoint.cpp` reduce downstream compile times

## Requirements

- C++17
- CMake ≥ 3.18
- [PlaMatrix](https://github.com/guderianXu/plamatrix) (math backend)
- CUDA Toolkit (optional, for GPU kernels)
- Google Test (for tests)

## Build

```bash
# Build and install plamatrix first
cd plamatrix && mkdir build && cd build
cmake .. -DPLAMATRIX_WITH_CUDA=ON
cmake --build . -j$(nproc)
cmake --install . --prefix ../install

# Build plapoint
cd plapoint && mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=/path/to/plamatrix/install
cmake --build . -j$(nproc)
./test/plapoint_tests
```

## Mesh Quality Reports

When tests are enabled, PlaPoint also builds a mesh quality report tool. The helper
script builds the tool, generates Marching Cubes and Poisson sphere meshes, and writes
both machine-readable metrics and inspectable PLY files:

```bash
./scripts/mesh_quality_report.py --build-dir build
```

The output directory contains `mesh_quality_report.json`,
`marching_cubes_sphere.ply`, and `poisson_sphere.ply`.

## Benchmarks

PlaPoint includes a dependency-free benchmark executable for local performance baselines:

```bash
cmake -S . -B build-bench \
  -DPLAPOINT_BUILD_BENCHMARKS=ON \
  -DPLAPOINT_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/plamatrix/install
cmake --build build-bench -j$(nproc)
./build-bench/benchmarks/plapoint_benchmarks --points 20000 --iterations 3
```

ICP benchmark rows use 512 points and 3 ICP iterations by default. Use `--icp-points`,
`--icp-max-iterations`, `--skip-cpu-icp`, and `--skip-icp-identity` to stress larger
finite-radius GPU ICP cases without waiting on CPU ICP or the infinite-radius identity
baseline:

```bash
./build-bench/benchmarks/plapoint_benchmarks \
  --points 1000 \
  --iterations 3 \
  --icp-points 10000 \
  --icp-max-iterations 3 \
  --skip-cpu-icp \
  --skip-icp-identity
```

The benchmark prints CSV columns:

```text
benchmark,points,iterations,best_ms
```

Each benchmark case runs one unmeasured warm-up before reporting the best timed iteration.
CUDA benchmark rows are emitted only when PlaPoint is built with `PLAPOINT_WITH_CUDA=ON` and a usable CUDA device is available.

For repeatable local baseline artifacts, use the wrapper script. It writes CSV,
JSON, and Markdown into the selected build output directory:

```bash
./scripts/run_benchmark_baseline.py \
  --benchmark-exe build-bench/benchmarks/plapoint_benchmarks \
  --output-dir build-bench/benchmark_baseline
```

Compare two baseline JSON files with:

```bash
./scripts/compare_benchmark_baseline.py \
  build-bench/benchmark_baseline_old/plapoint_benchmark_baseline.json \
  build-bench/benchmark_baseline/plapoint_benchmark_baseline.json \
  --config scripts/benchmark_gate_config.json \
  --json-output build-bench/benchmark_baseline/comparison.json \
  --markdown-output build-bench/benchmark_baseline/comparison.md
```

The comparison script reports regressions, improvements, added rows, and missing
rows. `scripts/benchmark_gate_config.json` stores the default regression and
improvement thresholds plus any known noisy benchmark names that should be
reported as `ignored`. Add `--fail-on-regression` when using it as a CI gate.

Generate a CUDA hotspot report from a benchmark JSON file with:

```bash
./scripts/report_cuda_hotspots.py \
  build-bench/benchmark_baseline/plapoint_benchmark_baseline.json \
  --markdown-output build-bench/benchmark_baseline/cuda_hotspots.md \
  --json-output build-bench/benchmark_baseline/cuda_hotspots.json
```

The report lists the slowest `gpu_*` rows and, when a matching `cpu_*` row
exists, the CPU/GPU timing ratio. Use this as the first pass for choosing which
CUDA row to profile in Nsight or optimize next.

## Validation Helpers

Run a CPU-only configure, build, and CTest pass with:

```bash
./scripts/run_cpu_only_validation.py --parallel $(nproc)
```

The script configures `PLAPOINT_WITH_CUDA=OFF`, enables tests and benchmarks, and
auto-detects a sibling PlaMatrix install prefix when available.

The real reconstruction regression helper validates the source image/camera set
and compares generated PLY files against `testData/real_reconstruction`.
It also evaluates basic real-output quality metrics from PLY fields, including
finite coordinate ratio, `error` mean/max, and grayscale intensity range:

```bash
./scripts/run_real_reconstruction_regression.py \
  --generated-root /path/to/generated/testData_dense \
  --actual-layout plascan-legacy \
  --json-output build/real_reconstruction_regression/comparison.json \
  --quality-json-output build/real_reconstruction_regression/quality.json \
  --max-error 0.01 \
  --max-mean-error 0.005 \
  --min-finite-ratio 1.0
```

Without `--generated-root` or `--pipeline-command`, the script compares the
checked-in reference tree to itself as a fast smoke test. Use `--pipeline-command`
with `{img_dir}`, `{tsai_dir}`, `{output_dir}`, and `{plapoint_root}` placeholders
to regenerate outputs before comparison.

For external PlaScan-style reconstruction commands, the pipeline wrapper keeps
the generation step and PlaPoint regression check in one command:

```bash
./scripts/run_real_reconstruction_pipeline.py \
  --command-template "python3 /path/to/reconstruct.py --img {img_dir} --tsai {tsai_dir} --out {output_dir}" \
  --output-dir build/real_reconstruction_pipeline \
  --actual-layout plascan-legacy \
  --json-output build/real_reconstruction_pipeline/comparison.json \
  --quality-json-output build/real_reconstruction_pipeline/quality.json
```

The wrapper validates the default source inputs at `../../testData/img` and
`../../testData/tsai`, expands the placeholders, runs the external command, and
then invokes the same PLY comparison and quality gates.

## API Overview

```cpp
#include <plapoint/core/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/filters/radius_outlier_removal.h>
#include <plapoint/filters/uniform_downsample.h>
#include <plapoint/features/normal_estimation.h>
#include <plapoint/features/normal_refinement.h>
#include <plapoint/registration/icp.h>
#include <plapoint/mesh/marching_cubes.h>
#include <plapoint/mesh/poisson_reconstruction.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/xyz_io.h>

using namespace plapoint;

// Create a point cloud on CPU
PointCloud<float, plamatrix::Device::CPU> cloud(1000);
cloud.points().fill(1.0f);
plamatrix::DenseMatrix<float, plamatrix::Device::CPU> scalar_fields(1000, 1);
scalar_fields.fill(0.0f);
cloud.setScalarFields({"error"}, std::move(scalar_fields));

// Transfer to GPU
auto gpu_cloud = cloud.toGpu();

// Build kd-tree and search
auto tree = std::make_shared<search::KdTree<float, plamatrix::Device::CPU>>();
tree->setInputCloud(std::make_shared<PointCloud<...>>(std::move(cloud)));
tree->build();
auto neighbors = tree->nearestKSearch({0, 0, 0}, 10);

// Filter
VoxelGrid<float, plamatrix::Device::CPU> vg;
vg.setInputCloud(...);
vg.setLeafSize(0.1, 0.1, 0.1);
PointCloud<float, plamatrix::Device::CPU> output;
vg.filter(output);

// Estimate normals
NormalEstimation<float, plamatrix::Device::CPU> ne;
ne.setInputCloud(...);
ne.setSearchMethod(tree);
auto normals = ne.compute();

// ICP registration
IterativeClosestPoint<float, plamatrix::Device::CPU> icp;
icp.setInputSource(source);
icp.setInputTarget(target);
icp.setMaximumIterations(50);
icp.setMaxCorrespondenceDistance(0.05f);
icp.setUseReciprocalCorrespondences(true);
icp.setUseOneToOneCorrespondences(true);
icp.setTrimmedOverlapRatio(0.8f);
icp.setEuclideanFitnessEpsilon(1e-6f);
icp.setTransformationRotationEpsilon(1e-6f);
icp.align(aligned);

plamatrix::DenseMatrix<float, plamatrix::Device::CPU> initial_guess(4, 4);
// Fill initial_guess as a rigid 4x4 transform before passing it to align().
icp.align(aligned, initial_guess);

// Reconstruct surface
PoissonReconstruction<float> pr;
pr.setInputCloud(point_cloud_with_normals);
pr.setDepth(6);
auto [verts, faces] = pr.reconstruct();

// PLY I/O
auto ply_cloud = io::readPly<float>("input.ply");
io::writePly("output.ply", *ply_cloud);

// XYZ I/O: strict by default, permissive for legacy files with bad rows.
auto xyz_cloud = io::readXyz<float>("input.xyz");
auto legacy_xyz_cloud = io::readXyz<float>("legacy.xyz", io::XyzReadMode::Permissive);
```

## License

MIT
