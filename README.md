# PlaPoint

CUDA/OpenCL-accelerated point cloud processing library built on [PlaMatrix](https://github.com/guderianXu/plamatrix).

## Features

### Core
- **`PointXYZ`, `PointXYZI`, `PointXYZRGB`, `PointXYZRGBA`, `Normal`, `PointNormal`, `PointXYZd`** — common point records use the same field names, aligned storage, color packing, constructors, and Eigen map accessors as the corresponding PCL 1.15.1 records. `PointXYZd` is a PlaPoint extension for georeferenced double-precision positions.
- **`PointCloud<PointT>`** — point-record cloud with aligned `points`, shape, header, sensor pose, `Ptr`/`ConstPtr`, subset construction, container mutation, advanced Eigen float views, organized indexing, and iteration. Its search, filter, normal-estimation, and ICP APIs currently run on CPU.
- **Point reflection and blobs** — `POINT_CLOUD_REGISTER_POINT_STRUCT`, `POINT_CLOUD_REGISTER_POINT_WRAPPER`, field traits, `PCLPointCloud2`, `PCLPointField`, `PointIndices`, `Vertices`, and `PolygonMesh` provide the data contracts used by the supported algorithms and I/O overloads.
- **`GeometryCloud<Scalar>`** — independently owned CPU Nx3 coordinates with optional normals,
  colors, intensities, texture coordinates, faces, material metadata, and named scalar fields.
  PLY/OBJ/XYZ/LAS matrix I/O, mesh processing, and device-selecting preprocessing use this type.
  The former scalar/device cloud is `internal::DeviceCloud<Scalar, Dev>` for backend execution.
  `editPoints()` bounds mutable access so point-derived caches can become reusable again.

The numerical implementation uses Eigen-style `plamatrix::Matrix`, fixed vectors, and
`SparseMatrix` on the host. CUDA storage uses `ResidentMatrix` with a shared execution
context; returned device results retain that context. PlaPoint owns voxel quantization,
centroids, spatial search, normals, and registration. PlaMatrix supplies general numerical
primitives such as sort, run-length encoding, scans, reductions, and sparse solvers.
BA observations, residuals, and problem assembly belong to PlaBundle.

### Spatial Indexing
- **`search::Search<PointT>` / `search::KdTree<PointT>` / `KdTreeFLANN<PointT, Dist>`** — point-type queries through a polymorphic search interface by point, cloud/index, subset position, or batch; results use original-cloud indices and squared `float` distances. The public contracts include point representations, epsilon, minimum-points, sorted results, and PCL-compatible `setInputCloud()` return types.
- **`search::internal::DeviceKdTree<Scalar, Dev>`** — backend matrix kd-tree used by CPU and CUDA algorithms. Point-type `search::KdTree<PointT>` builds its index during `setInputCloud()` so independent read-only queries can run concurrently.
- **GpuSpatialIndex\<Scalar\>** — reusable prepared uniform-grid snapshot for finite GPU points,
  with bounded radius count/search and KNN (`k <= 32`). Queries use caller-owned
  `GpuSpatialQueryWorkspace` storage; separate workspaces allow independent streams to share the
  immutable index. Cached users invalidate the index when point storage identity, point count, or
  `pointsRevision()` changes.

### Filters
- **VoxelGrid** — centroid-based voxel downsampling with mean aggregation for normals, colors, intensities, and named scalar fields
- **StatisticalOutlierRemoval** — KNN distance statistics outlier filtering
- **RadiusOutlierRemoval** — radius-based neighbor count filtering
- **UniformDownsample** — keep every Nth point
- **`Filter<PointT>`** — public point-record filter base; scalar/device filters remain backend implementations

Point-index preserving filters and GPU gather helpers keep point-aligned attributes
(`normals`, `colors`, `intensities`, and named scalar fields) for the selected
points. Mesh compaction uses the same rule for surviving vertices. VoxelGrid is
the main aggregation filter: it averages per-voxel scalar fields instead of
dropping them.

`VoxelGrid<PointT>`, `StatisticalOutlierRemoval<PointT>`, and
`RadiusOutlierRemoval<PointT>` expose the common PCL 1.15.1 CPU point-cloud calls.
Voxel filtering supports minimum points per voxel, full-field averaging, field limits,
negative limits, saved leaf layouts, and centroid lookup. The outlier filters use the
`PCLBase`/`Filter`/`FilterIndices` contracts, and radius filtering exposes the thread-count
setting.
The outlier filters preserve the original point record and metadata when
compacting, and support search injection, input-index subsets, negative selection,
and organized output. The subset limits query points; neighbor searches still use
the full input cloud.

### Features
- **`Feature<PointInT, PointOutT>` / `NormalEstimation<PointInT, PointOutT>`** — polymorphic K/radius neighborhood, search surface, input indices, viewpoint API, `computePointNormal()` overloads, and normal-flip helpers writing `PointCloud<Normal>` or `PointCloud<PointNormal>` with curvature. Covariance and eigendecomposition use PlaMatrix fixed matrices.
- **`estimateNormals(GeometryCloud<Scalar>, ...)`** — public CPU-owned geometry entry point selecting CPU, CUDA, or OpenCL. The scalar/device PCA estimator handles backend execution.
- **NormalRefinement** — normal smoothing and viewpoint orientation, with device-resident CUDA kernels for GPU clouds

### Registration
- **`Registration<PointSource, PointTarget, Scalar>` / `IterativeClosestPoint<PointSource, PointTarget, Scalar>`** — polymorphic point-type ICP with public Eigen 4x4 transforms and a PlaMatrix computation backend. The supported contract includes correspondence and SVD transformation estimators, distance rejection, deterministic RANSAC, reciprocal correspondences, convergence criteria, incremental/final transforms, and visualization callbacks. `getFitnessScore()` reports mean squared nearest-neighbor distance; `getInlierFraction()` reports PlaPoint's accepted-correspondence ratio.
- **`MatrixIterativeClosestPoint<Scalar, Dev>`** — backend scalar/device ICP implementation, including the existing CUDA path and correspondence controls.

For the types and algorithms listed above, public class names, templates, method names,
parameter order, return types, defaults, and metric meanings follow PCL 1.15.1. Existing
code confined to this supported surface can normally migrate by changing the include
prefix and namespace from `pcl` to `plapoint`. PlaPoint is not an implementation of all
PCL modules: point types and algorithms not listed here, including segmentation,
descriptors, keypoints, visualization, NDT, and generalized ICP, still require source
changes or another library. PlaPoint does not link against PCL and does not use
PCL-specific implementation filenames; compatibility names such as `PCLHeader` and
`PCLPointCloud2` remain public because applications use them directly.

### Mesh
- **MarchingCubes** — CPU callback extraction plus deterministic CUDA extraction from a device scalar field
- **HeightGrid** — CPU/CUDA/OpenCL terrain aggregation with mean/min/max elevation and multi-pass hole fill
- **PoissonReconstruction** — deterministic 2:1-balanced adaptive octree, symmetric CSR assembly, PlaMatrix Jacobi-PCG, single-pass field sampling, and MC extraction

### I/O
- **PCD** — typed and `PCLPointCloud2` ASCII, binary, and binary-compressed read/write APIs, including header/body stream overloads, sensor pose, organized dimensions, selected indices, and header generation.
- **PLY** — typed and `PCLPointCloud2` ASCII/binary I/O, selected indices, `PolygonMesh`, sensor pose, stream output, and public header generation. Existing matrix I/O and chunked binary traversal remain available.
- **OBJ** — geometry/material read/write plus bounded-memory line-record streaming with non-owning token views
- **XYZ** — strict-by-default text XYZ reader. Strict rows must be exactly `x y z` or `x y z r g b`; malformed rows throw with file path and line number. Pass `io::XyzReadMode::Permissive` to skip bad legacy rows and keep the older trailing-column tolerance.

## Accelerator Backends

When `PLAPOINT_WITH_CUDA=ON`, CUDA Toolkit is available, and `plamatrix::plamatrix`
was built with CUDA support:

- **Adaptive KNN** — `search::internal::DeviceKdTree<Scalar, GPU>` selects between the shared-memory brute-force kernel and `GpuSpatialIndex`. Grid KNN is limited to `k <= 32`; wide empty-shell queries, pathological occupancy, unsafe quantization, small work, and larger `k` use a compatibility backend. Results are ordered deterministically by distance then source index.
- **Stream-aware device KNN** — `gpu::batchKnnDeviceAsync()` and `gpu::batchKnnDeviceColumnMajorAsync()` launch on a caller-provided `cudaStream_t`; existing non-async overloads preserve synchronous behavior.
- **Indexed radius and outlier filters** — RadiusOR uses saturated uniform-grid counts. SOR uses indexed KNN plus PlaMatrix reductions and device compaction. Normals, colors, intensities, named scalar fields, and point-aligned UV data stay aligned during stable GPU compaction.
- **Device normal processing** — normal estimation uses batched symmetric eigensolves; smoothing and viewpoint orientation remain on device for supported GPU inputs.
- **VoxelGrid CUDA downsampling** (`src/voxel_grid_gpu.cu`) — PlaPoint quantizes points and computes overflow-safe centroids, while PlaMatrix stably sorts and groups three-component signed voxel keys and scans group offsets; output remains deterministic in voxel-key order.
- **GPU mesh primitives** — `gpu::marchingCubes()` returns a GPU triangle soup directly from a device field. `buildHeightGridDeviceAsync()` aggregates into a device grid using explicit bounds, and `fillHolesAsync()` keeps multi-pass fill on the producing stream until one final download.
- **Poisson PCG** — CPU, CUDA, and OpenCL solver selections share the same deterministic symmetric CSR. A scale-aware anchor per connected component makes the system SPD. Explicit accelerator non-convergence throws; `Auto` tries CUDA, OpenCL, then CPU and continues from the latest available iterate. The extraction grid is evaluated once and reused for range selection and Marching Cubes; face orientation builds one CPU kd-tree instead of scanning every input point per face. Field evaluation and MC extraction remain CPU, so `PoissonProcessingReport::solverDevice` reports the PCG device while `actualDevice` remains CPU for the complete chain.
- **ICP GPU path** (`src/icp_gpu.cu`) — `MatrixIterativeClosestPoint<Scalar, GPU>` keeps source/target point buffers on GPU, reads the initial source buffer directly without a startup device-to-device copy, computes correspondences with a cached finite-radius target spatial grid or the shared-memory target-tiling fallback, and builds the target grid through PlaMatrix's stable three-component key sort, run-length encoding, and offset scan. It uses precomputed finite-radius tile bounding-box skips and per-candidate axis pruning on the fallback path, accumulates centroid/covariance/residual stats with block-level reductions, derives degeneracy flags from covariance invariants, fuses stats reduction with device-side step-transform solving through a CUDA quaternion/Jacobi solver, applies point transforms through persistent GPU scratch buffers, initializes and asynchronously accumulates the final 4x4 transform on GPU, and writes terminal-iteration transforms directly into a plain non-input caller output cloud when possible. Reduced stats, step deltas, and metric checks still synchronize to CPU, while `getFinalTransformationDevice()` exposes the final transform without forcing callers through the CPU copy and the legacy CPU `getFinalTransformation()` materializes that copy lazily. The stats helper can skip per-source correspondence index output when callers only need aggregate ICP moments, persistent workspaces and GPU buffers avoid repeated reduction, target spatial-grid, target-tile bound, step-solver, transform-buffer, point-scratch, output-allocation, and final output-copy overhead across repeated `align()` calls on the same ICP object and plain same-shaped output cloud, and `alignGpu()` skips transformed final-stats scans on non-terminal iterations. `setComputeFinalMetrics(false)` is an opt-in throughput mode that skips the terminal post-transform fitness/RMSE scan when callers only need the transform or aligned output. Input-aliased, attributed, or metadata-bearing output clouds still use safe scratch/copy or replacement paths so stale normals, colors, intensities, named scalar fields, mesh, material, or texture data cannot leak into aligned-point results. PCL-style robust ICP options that need correspondence rejectors currently preserve GPU input/output types through a CPU-staged semantic fallback; the default CUDA fast path remains unchanged for the base ICP configuration.

For latency-sensitive GPU ICP, call `reserveGpuWorkspace()` to preallocate scratch and
`prepareGpuTargetSpatialIndex()` to synchronously build the finite-radius target grid before
the first `align()`. The prepare call returns `false` when the current radius or target size
selects a different search path. A prepared grid is reused until the target positions or
correspondence radius change.
- **Compatibility fallbacks** — KNN/SOR requests above the indexed `k` limit and pathological grid queries use documented fallback paths.
- **VoxelGrid CPU hot path** — CPU path uses hash aggregation and sorted voxel keys to keep deterministic centroid order.
- Explicit template instantiations in `src/plapoint.cpp` reduce downstream compile times

When `PLAPOINT_WITH_OPENCL=ON`, PlaPoint uses PlaMatrix's OpenCL runtime and buffer
resources to run OpenCL C 1.2 kernels for CPU-owned voxel downsampling,
statistical/radius outlier removal, normal-estimation KNN, height-grid aggregation,
and Poisson's CSR Jacobi-PCG solve.
The current high-level geometry APIs accept and return `GeometryCloud<Scalar>` or
PlaMatrix host matrices, while PlaMatrix owns OpenCL device discovery, program caching,
queues, and device storage.
Point-cloud-specific kernels and their dispatch policy remain in PlaPoint.
Enumerate devices with `opencl::enumerateOpenClGpuDevices()`. Set
`PLAMATRIX_OPENCL_DEVICE_INDEX` to a stable enumerated GPU index (`-1` or unset means
automatic selection); `PLAPOINT_OPENCL_DEVICE_INDEX` remains a compatibility fallback.
For attributed voxel clouds, OpenCL computes geometric centroids while the existing CPU
VoxelGrid performs exact attribute aggregation, so that path is a partial offload.

CPU-owned convenience APIs expose `ProcessingReport`, including `requestedDevice`,
`actualDevice` (`usedDevice` remains a compatibility alias), `neighborBackend`,
`usedFallback`, `fallbackReason`, and `selectionReason`. `ProcessingDevice::Auto`
keeps small transfer-bound work on CPU, uses a point-count threshold for linear work,
and a point-count-by-neighbor threshold for KNN-style work. Larger calls try CUDA,
then OpenCL, then CPU; attributed voxel clouds skip the partial OpenCL path in Auto
mode because exact attribute aggregation is still CPU-side. An explicit `CUDA` or
`OpenCL` request never silently falls back. `ProcessingDevice::GPU` remains a
compatibility alias for `CUDA`.
See [GPU search and selection](docs/gpu-search.md) and [GPU mesh processing](docs/gpu-mesh.md)
for limits and lifetime rules.

## Requirements

- C++17
- CMake ≥ 3.21
- [PlaMatrix](https://github.com/guderianXu/plamatrix) (math backend)
- Eigen3 ≥ 3.3 (public PCL-compatible types)
- CUDA Toolkit (optional, for CUDA kernels)
- OpenCL SDK/loader and a GPU OpenCL driver (optional, for OpenCL C 1.2 kernels)
- Google Test (for tests)

## Build

Set `PLAPOINT_DEPS_PREFIX` to a CMake prefix list containing PlaMatrix and Eigen3,
and, for the test-enabled `*-release` presets, Google Test. Use the checked-in
presets for reproducible CPU, OpenCL, CUDA, and
instrumentation-free benchmark builds:

```bash
export PLAPOINT_DEPS_PREFIX="/path/to/plamatrix/install;/path/to/gtest/install"
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release
```

PowerShell uses
`$env:PLAPOINT_DEPS_PREFIX = 'C:\path\to\plamatrix\install;C:\path\to\gtest\install'`.
The equivalent manual build remains available:

```bash
# Build and install plamatrix first
cd plamatrix && mkdir build && cd build
cmake .. -DPLAMATRIX_WITH_CUDA=ON
cmake --build . -j$(nproc)
cmake --install . --prefix ../install

# Build plapoint
cd plapoint && mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DPLAPOINT_WITH_OPENCL=ON \
  -DCMAKE_PREFIX_PATH=/path/to/plamatrix/install
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
  -DPLAPOINT_BUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH=/path/to/plamatrix/install
cmake --build build-bench -j$(nproc)
./build-bench/benchmarks/plapoint_benchmarks --points 20000 --iterations 7
```

Performance baselines must use a benchmark-only build. Test-enabled libraries contain
deliberate ICP instrumentation for white-box assertions; CMake warns when tests and
benchmarks share a build tree. The `cpu-benchmark` and `cuda-benchmark` presets configure
the clean baseline mode.

ICP benchmark rows use 512 points and 3 ICP iterations by default. Use `--icp-points`,
`--icp-max-iterations`, `--skip-cpu-icp`, and `--skip-icp-identity` to stress larger
finite-radius GPU ICP cases without waiting on CPU ICP or the infinite-radius identity
baseline:

```bash
./build-bench/benchmarks/plapoint_benchmarks \
  --points 1000 \
  --iterations 7 \
  --icp-points 10000 \
  --icp-max-iterations 3 \
  --skip-cpu-icp \
  --skip-icp-identity
```

The benchmark prints CSV columns:

```text
benchmark,points,iterations,best_ms,median_ms,p95_ms,stddev_ms,cv
```

Each benchmark case runs one unmeasured warm-up, retains the best value for backward
compatibility, and uses median latency as the primary regression metric. It also reports
nearest-rank p95, population standard deviation, and coefficient of variation.
CUDA benchmark rows are emitted only when PlaPoint is built with `PLAPOINT_WITH_CUDA=ON` and a usable CUDA device is available.
Use `--search-features-only` to run KNN brute-force/indexed comparisons, radius
counts, normal estimation/smoothing, SOR, and RadiusOR without the ICP suite.
Use `--mesh-only` to emit `marching_cubes_field`, `height_grid_fill`,
`poisson_solve`, and `poisson_end_to_end` without the search or ICP rows:

```bash
./build-bench/benchmarks/plapoint_benchmarks \
  --points 4096 --poisson-points 8192 --poisson-depth 6 \
  --iterations 7 --mesh-only
```

`--poisson-points` and `--poisson-depth` are independent from the general point count so
the PCG row can exercise a production-sized adaptive octree. Their defaults are 8192 and 6.
For the uniform sphere fixture, keep at least `4^depth` points; undersampled combinations are
rejected before timing with guidance to increase the point count or reduce the depth.

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
rows. New baselines compare `median_ms`; comparisons involving a legacy four-column
baseline consistently fall back to `best_ms`. `scripts/benchmark_gate_config.json` stores the default regression and
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

To prove that CUDA kernels execute on real hardware, configure the test build with
`PLAPOINT_REQUIRE_CUDA_TEST_DEVICE=ON`. Configuration rejects this option unless both CUDA
and tests are enabled, and the test suite fails if device discovery or CUDA context creation
fails:

```bash
cmake -S . -B build-cuda-runtime \
  -DPLAPOINT_WITH_CUDA=ON \
  -DPLAPOINT_BUILD_TESTS=ON \
  -DPLAPOINT_REQUIRE_CUDA_TEST_DEVICE=ON \
  -DCMAKE_PREFIX_PATH=/path/to/cuda-plamatrix/install
cmake --build build-cuda-runtime
ctest --test-dir build-cuda-runtime --output-on-failure --no-tests=error
```

The ordinary hosted CUDA CI job checks compilation and any host-visible tests. The manual
`gpu-runtime-validation` workflow targets a stable self-hosted runner labeled `gpu`, enables
the required-device test, runs the CUDA suite, and compares an instrumentation-free benchmark
against that runner's latest cached baseline with `--fail-on-regression` and
`--fail-on-missing`. Its first run, or a run with `refresh_baseline`, seeds the baseline.

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
#include <plapoint/point_cloud.h>
#include <plapoint/search/kdtree.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/features/normal_3d.h>
#include <plapoint/registration/icp.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/io/pcd_io.h>

using Cloud = plapoint::PointCloud<plapoint::PointXYZ>;
Cloud::Ptr source = std::make_shared<Cloud>();
source->push_back(plapoint::PointXYZ(0.0f, 0.0f, 0.0f));
source->push_back(plapoint::PointXYZ(1.0f, 0.0f, 0.0f));
source->push_back(plapoint::PointXYZ(0.0f, 1.0f, 0.0f));
source->push_back(plapoint::PointXYZ(0.0f, 0.0f, 1.0f));

plapoint::search::KdTree<plapoint::PointXYZ> tree;
tree.setInputCloud(source);
std::vector<int> indices;
std::vector<float> squared_distances;
tree.nearestKSearch(source->points[0], 3, indices, squared_distances);
tree.radiusSearch(source->points[0], 1.0, indices, squared_distances);

plapoint::VoxelGrid<plapoint::PointXYZ> voxel;
voxel.setInputCloud(source);
voxel.setLeafSize(0.5f, 0.5f, 0.5f);
Cloud filtered;
voxel.filter(filtered);

plapoint::NormalEstimation<plapoint::PointXYZ, plapoint::Normal> estimator;
estimator.setInputCloud(source);
estimator.setKSearch(4);
plapoint::PointCloud<plapoint::Normal> normals;
estimator.compute(normals);

plapoint::IterativeClosestPoint<plapoint::PointXYZ, plapoint::PointXYZ> icp;
icp.setInputSource(source);
icp.setInputTarget(source);
icp.setMaximumIterations(50);
icp.setTransformationEpsilon(1.0e-8); // squared translation threshold
Cloud aligned;
icp.align(aligned, Eigen::Matrix4f::Identity());
const Eigen::Matrix4f transform = icp.getFinalTransformation();
const double mean_squared_error = icp.getFitnessScore();

plapoint::io::savePLYFileBinary("aligned.ply", aligned);
Cloud loaded;
plapoint::io::loadPLYFile("aligned.ply", loaded);
```

The point-type API above currently materializes CPU matrices for search, filters, and ICP.
The backend normal-estimation and ICP implementations retain existing CUDA/OpenCL
processing routes. Typed PCD/PLY conversion uses registered fields, so the built-in point
records and custom registered point records share the same I/O path. Matrix I/O and
streaming APIs remain available through `GeometryCloud` for mesh faces, extra scalar fields,
and large files.

## License

MIT
