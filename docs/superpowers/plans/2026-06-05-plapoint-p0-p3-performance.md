# PlaPoint P0-P3 Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the next PlaPoint optimization pass covering benchmark baselines, high-impact GPU paths, CPU hot spots, and engineering quality gates.

**Architecture:** Keep the public API source-compatible and add narrow optional performance APIs. Use deterministic CPU implementations as the correctness reference, and require every GPU optimization to preserve existing output ordering or document a new ordering explicitly.

**Tech Stack:** C++17, CUDA runtime, PlaMatrix, CMake, Google Test, standard-library chrono benchmarks.

---

### Scope Mapping

- **P0 Benchmark:** Add a self-contained `plapoint_benchmarks` executable behind `PLAPOINT_BUILD_BENCHMARKS`.
- **P1 GPU KNN:** Add stream-aware asynchronous device KNN overloads while keeping existing synchronous APIs.
- **P1 VoxelGrid:** Replace the CPU `std::map` hot path with an ordered `unordered_map` implementation and add a production CUDA sort/reduce voxel pipeline.
- **P2 Staging/ICP/CPU:** Add a cached `PointCloud::pointsCpu()` mirror for GPU fallback paths and reduce CPU hot-path overhead where it is local and measurable; keep ICP algorithmic behavior unchanged in this pass.
- **P3 Engineering:** Add CMake targets, README support matrix notes, and tests that cover new APIs.

### Task 1: Benchmark Harness

**Files:**
- Create: `benchmarks/CMakeLists.txt`
- Create: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `CMakeLists.txt`
- Modify: `README.md`

- [x] Add `PLAPOINT_BUILD_BENCHMARKS` option.
- [x] Add deterministic benchmark data generation.
- [x] Benchmark CPU KNN, CPU VoxelGrid, NormalEstimation, and ICP.
- [x] Add CUDA benchmark cases when `PLAPOINT_WITH_CUDA` and a usable CUDA device are available.
- [x] Verify `cmake -S . -B build-codex-cpu-bench -DPLAPOINT_BUILD_BENCHMARKS=ON -DPLAPOINT_BUILD_TESTS=ON -DPLAPOINT_WITH_CUDA=OFF -DCMAKE_PREFIX_PATH=/tmp/plamatrix-install-smoke`.

### Task 2: Stream-Aware GPU KNN API

**Files:**
- Modify: `include/plapoint/gpu/knn.h`
- Modify: `src/batch_knn_gpu.cu`
- Test: `test/unit/search/kdtree_gpu_test.cpp`

- [x] Add CUDA-only `batchKnnDeviceAsync()` and `batchKnnDeviceColumnMajorAsync()` overloads accepting `cudaStream_t`.
- [x] Preserve existing synchronous functions by calling async overloads and synchronizing the default stream.
- [x] Add a CUDA test that launches KNN on a non-default stream and copies results back with `cudaMemcpyAsync`.
- [x] Reuse KdTree GPU batch query/index/distance device workspace across repeated batch calls.
- [x] Verify targeted KNN tests in CUDA build.

### Task 3: Cached CPU Point Mirror For GPU Fallbacks

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`
- Modify: staged fallback callers under `include/plapoint/search/`, `include/plapoint/filters/`, `include/plapoint/features/`, and `include/plapoint/registration/`
- Test: `test/unit/core/point_cloud_test.cpp`

- [x] Add `PointCloud::pointsCpu()` that returns CPU point storage directly for CPU clouds and caches a GPU-to-CPU copy for GPU clouds.
- [x] Invalidate the GPU CPU mirror when mutable `points()` is requested.
- [x] Route KdTree, filters, normal estimation/refinement, and ICP through `pointsCpu()` where they need a CPU point view.
- [x] Add CPU and CUDA tests for `pointsCpu()` values, cache reuse, and invalidation.

### Task 4: VoxelGrid CPU Hot Path

**Files:**
- Modify: `include/plapoint/filters/voxel_grid.h`
- Modify: `include/plapoint/gpu/voxel_grid.h`
- Modify: `src/voxel_grid_gpu.cu`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/filters/voxel_grid_test.cpp`

- [x] Replace `std::map<std::tuple<int,int,int>, Accum>` with `std::unordered_map<VoxelKey, Accum>`.
- [x] Sort final voxel keys before output to preserve deterministic centroid order.
- [x] Implement CUDA VoxelGrid with device key computation, thrust sort-by-key, reduce-by-key centroid sums, and sorted output.
- [x] Add tests covering negative voxel coordinates and deterministic output order.
- [x] Verify voxel tests in CPU and CUDA builds.

### Task 5: Engineering Quality And Documentation

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/*` if present, otherwise document local verification.

- [x] Update GPU support matrix and benchmark commands.
- [x] Run `git diff --check`.
- [x] Run full CPU and CUDA test suites.
- [x] Commit with configured GitHub identity and push to `origin master`.

Current benchmark smoke evidence:

- CPU: `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`
  ran CPU KNN, VoxelGrid, NormalEstimation, and ICP; sample rows included
  `cpu_knn_batch_k8,1000,1,33.2153` and `cpu_voxel_grid,1000,1,0.050443`.
- CUDA: `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`
  ran CPU cases plus GPU KNN and GPU VoxelGrid; sample rows included
  `gpu_knn_batch_k8,1000,1,1.33904` and `gpu_voxel_grid,1000,1,0.182839`.

Current verification evidence:

- `git diff --check`: clean.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  179 tests, 0 failed.
