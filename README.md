# PlaPoint

GPU-accelerated point cloud processing library built on [PlaMatrix](https://github.com/guderianXu/plamatrix).

## Features

### Core
- **PointCloud\<Scalar, Dev\>** — Nx3 point cloud with optional normals, GPU/CPU transfer via `toGpu()`/`toCpu()`, and cached CPU point views via `pointsCpu()`

### Spatial Indexing
- **KdTree\<Scalar, Dev\>** — 3D kd-tree with KNN search (priority queue) and radius search

### Filters
- **VoxelGrid** — centroid-based voxel downsampling
- **StatisticalOutlierRemoval** — KNN distance statistics outlier filtering
- **RadiusOutlierRemoval** — radius-based neighbor count filtering
- **UniformDownsample** — keep every Nth point
- **Filter\<Scalar, Dev\>** — abstract base class for all filters

### Features
- **NormalEstimation** — PCA-based surface normal estimation via covariance + SVD
- **NormalRefinement** — normal smoothing (KNN averaging) and viewpoint-based orientation

### Registration
- **IterativeClosestPoint** (ICP) — point-to-point ICP with SVD-based rigid transform

### Mesh
- **MarchingCubes** — isosurface extraction from implicit scalar fields
- **PoissonReconstruction** — Poisson surface reconstruction (Gauss-Seidel solver + MC extraction)

### I/O
- **PLY** — ASCII read/write with positions and optional normals

## GPU Acceleration

When `PLAPOINT_WITH_CUDA=ON`, CUDA Toolkit is available, and `plamatrix::plamatrix`
was built with CUDA support:

- **Brute-force KNN** (`src/knn_gpu.cu`) — batched K-nearest neighbor search with one CUDA block per query point, shared memory top-K reduction. `KdTree<Scalar, GPU>::batchNearestKSearch()` reads PlaMatrix column-major device buffers directly.
- **Stream-aware device KNN** — `gpu::batchKnnDeviceAsync()` and `gpu::batchKnnDeviceColumnMajorAsync()` launch on a caller-provided `cudaStream_t`; existing non-async overloads preserve synchronous behavior.
- **VoxelGrid CUDA downsampling** (`src/voxel_grid_gpu.cu`) — GPU path computes voxel keys, sorts them, reduces centroids, and preserves deterministic sorted voxel-key output.
- **ICP GPU path** (`src/icp_gpu.cu`) — `IterativeClosestPoint<Scalar, GPU>` keeps source/target point buffers on GPU, reads the initial source buffer directly without a startup device-to-device copy, computes correspondences with a cached finite-radius target spatial grid or the shared-memory target-tiling fallback, uses precomputed finite-radius tile bounding-box skips and per-candidate axis pruning on the fallback path, accumulates centroid/covariance/residual stats with block-level reductions, derives degeneracy flags from covariance invariants, fuses stats reduction with device-side step-transform solving through a CUDA quaternion/Jacobi solver, applies point transforms through persistent GPU scratch buffers, initializes and asynchronously accumulates the final 4x4 transform on GPU, and writes terminal-iteration transforms directly into a plain non-input caller output cloud when possible. Reduced stats, step deltas, and metric checks still synchronize to CPU, while `getFinalTransformationDevice()` exposes the final transform without forcing callers through the CPU copy and the legacy CPU `getFinalTransformation()` materializes that copy lazily. The stats helper can skip per-source correspondence index output when callers only need aggregate ICP moments, persistent workspaces and GPU buffers avoid repeated reduction, target spatial-grid, target-tile bound, step-solver, transform-buffer, point-scratch, output-allocation, and final output-copy overhead across repeated `align()` calls on the same ICP object and plain same-shaped output cloud, and `alignGpu()` skips transformed final-stats scans on non-terminal iterations. Input-aliased, attributed, or metadata-bearing output clouds still use safe scratch/copy or replacement paths so stale normals, colors, mesh, material, or texture data cannot leak into aligned-point results.
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

using namespace plapoint;

// Create a point cloud on CPU
PointCloud<float, plamatrix::Device::CPU> cloud(1000);
cloud.points().fill(1.0f);

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
icp.align(aligned);

// Reconstruct surface
PoissonReconstruction<float> pr;
pr.setInputCloud(point_cloud_with_normals);
pr.setDepth(6);
auto [verts, faces] = pr.reconstruct();

// PLY I/O
auto cloud = io::readPly<float>("input.ply");
io::writePly("output.ply", *cloud);
```

## License

MIT
