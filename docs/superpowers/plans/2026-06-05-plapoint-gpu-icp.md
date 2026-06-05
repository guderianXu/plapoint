# PlaPoint GPU ICP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `IterativeClosestPoint<Scalar, GPU>` away from full CPU-staged point processing and keep the per-point ICP workload on GPU.

**Architecture:** Keep the existing CPU ICP path unchanged. Add CUDA helpers that accept PlaMatrix column-major GPU buffers, compute nearest-neighbor correspondences with shared-memory target tiling, accumulate centroid/covariance/residual stats on device with block-level reductions, update the 4x4 accumulated transform on device, and expose the GPU final transform to callers. The small Kabsch SVD remains CPU-side while current points and transform accumulation stay GPU-resident.

**Tech Stack:** C++17, CUDA runtime, PlaMatrix GPU `DenseMatrix`, Google Test, PlaPoint benchmark executable.

---

### Task 1: Regression Test For CPU Staging Removal

**Files:**
- Create: `test/unit/registration/icp_gpu_path_test.cpp`

- [x] Add a CUDA-only test that exposes `PointCloud::_points_cpu_cache` for test inspection.
- [x] Verify the test fails on the previous CPU-staged GPU ICP path because source/target caches are populated.
- [x] Keep the test focused on GPU ICP alignment and cache state, not performance timing.

### Task 2: CUDA Correspondence And Stats Helper

**Files:**
- Create: `include/plapoint/gpu/icp.h`
- Create: `src/icp_gpu.cu`
- Modify: `src/CMakeLists.txt`

- [x] Add `gpu::IcpCorrespondenceStats<Scalar>` with active count, invalid source count, centroids, covariance sums, and residual sum.
- [x] Implement a CUDA kernel that scans target points for each source point, filters non-finite distances and max correspondence distance, and accumulates stats with device-side reductions.
- [x] Copy only the small stats struct back to host after stream synchronization.

### Task 3: GPU ICP Align Path

**Files:**
- Modify: `include/plapoint/registration/icp.h`

- [x] Branch `align()` to `alignGpu()` when `Dev == plamatrix::Device::GPU`.
- [x] Copy the source points device-to-device into the current-iteration buffer without calling `pointsCpu()`.
- [x] Use GPU stats for correspondence count, residual metrics, centroids, and cross-covariance.
- [x] Keep CPU Kabsch SVD for the 3x3 step transform and apply the step transform with PlaMatrix GPU `transformPoints()`.
- [x] Preserve CPU path behavior and error messages for missing input, empty clouds, too few correspondences, and non-finite source points.

### Task 4: Device-Side Final Transform Accumulation

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`

- [x] Add a CUDA-only failing test for `getFinalTransformationDevice()` after GPU ICP alignment.
- [x] Add a direct CUDA test for non-trivial column-major 4x4 transform multiplication.
- [x] Implement `gpu::multiplyTransform4x4()` for float and double device matrices.
- [x] Keep `alignGpu()` transform accumulation in a GPU `DenseMatrix` and copy it to CPU only for the legacy `getFinalTransformation()` API.
- [x] Expose `getFinalTransformationDevice()` for GPU ICP callers.

### Task 5: Correspondence Stats Hot-Path Reduction

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`

- [x] Add a CUDA-only failing test showing aggregate stats can be computed with `d_correspondence_indices == nullptr`.
- [x] Keep explicit correspondence index output covered for callers that still request it.
- [x] Remove the unused GPU ICP `DeviceBuffer<int>` allocation and pass `nullptr` from `alignGpu()`.
- [x] Replace per-point global double atomics with per-block shared-memory stats reduction and a second-stage partial reducer.

### Task 6: Shared-Memory Target Tiling

**Files:**
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`

- [x] Add a CUDA stats test where the nearest target lies past the first 128-target tile.
- [x] Load target coordinates and finite flags into shared memory per source block.
- [x] Reuse each target tile across source threads in the block before loading the next tile.

### Task 7: Reusable Stats Workspace

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`

- [x] Add a CUDA stats test requiring `IcpCorrespondenceStatsWorkspace` to reuse device storage across repeated calls.
- [x] Implement reusable partial-reduction and final-stats device storage.
- [x] Keep the original no-workspace stats overload for compatibility.
- [x] Reserve workspace once in `alignGpu()` and reuse it for both stats calls in each iteration.

### Task 8: Documentation, Benchmark, And Verification

**Files:**
- Modify: `README.md`
- Modify: `benchmarks/plapoint_benchmarks.cpp`

- [x] Update README to state ICP is no longer a CPU-staged fallback for GPU point buffers.
- [x] Add `gpu_icp_identity` to the benchmark executable.
- [x] Run full CPU and CUDA tests.
- [x] Run benchmark smoke for CPU/CUDA builds.
- [x] Commit and push to `origin master`.

Verification evidence:

- `git diff --check`: clean.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `FinalTransformationDeviceIsAvailableAfterGpuAlign`:
  failed as expected because `IterativeClosestPoint<float, GPU>` had no `getFinalTransformationDevice()` member.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*`:
  5 targeted GPU ICP path tests passed after adding optional correspondence output coverage.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput`:
  failed as expected before the null-output optimization with `ICP GPU: device pointers must not be null`.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment`:
  8 targeted ICP GPU/stats tests passed after block-reduction, target-tiling, and reusable workspace implementation.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `CorrespondenceStatsWorkspaceReusesDeviceStorage`:
  failed as expected because `plapoint::gpu::IcpCorrespondenceStatsWorkspace` did not exist yet.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  186 tests, 0 failed.
- `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`:
  CPU benchmark rows emitted through `cpu_icp_identity,512,1,32.7031`.
- `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`:
  CUDA benchmark rows included `gpu_icp_identity,512,1,1.05141`.
