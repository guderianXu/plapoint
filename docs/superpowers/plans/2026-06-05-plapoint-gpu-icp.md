# PlaPoint GPU ICP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `IterativeClosestPoint<Scalar, GPU>` away from full CPU-staged point processing and keep the per-point ICP workload on GPU.

**Architecture:** Keep the existing CPU ICP path unchanged. Add CUDA helpers that accept PlaMatrix column-major GPU buffers, compute nearest-neighbor correspondences with shared-memory target tiling, finite-radius target tile bounding-box skips, and per-candidate axis pruning, accumulate centroid/covariance/residual stats on device with block-level reductions, compute the rigid step transform from device-side stats reductions with a GPU quaternion/Jacobi solver, initialize and asynchronously update the 4x4 accumulated transform on device, and expose the GPU final transform to callers. Reduced stats, step deltas, and small metric checks still synchronize to CPU while current points, step transforms, degeneracy flags, and transform accumulation stay GPU-resident. The GPU align loop reads the initial source buffer directly, skips transformed final-stats scans on non-terminal iterations, and computes final metrics only when convergence or the final iteration requires them. The legacy CPU final transform is copied from the GPU lazily only when `getFinalTransformation()` is called.

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
- [x] Move the 3x3 step transform off the CPU Kabsch/SVD path and apply the step transform with GPU point kernels.
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

### Task 9: Preallocated Transform Buffers

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test requiring `transformPointsColumnMajor()` to write into caller-owned GPU output storage.
- [x] Implement the float/double caller-owned column-major point transform helper, with synchronous and async entry points.
- [x] Preallocate `alignGpu()` step-transform, accumulated-transform, and next-point buffers once before the ICP loop.
- [x] Swap preallocated GPU buffers inside the loop instead of allocating new point and transform matrices per iteration.

### Task 10: GPU Step Transform Solver

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test requiring `computeIcpStepTransformFromStats()` to write a known rigid step transform to GPU memory.
- [x] Implement reusable step-solver workspace and float/double API overloads.
- [x] Compute the point-to-point ICP step rotation in CUDA with a Davenport quaternion matrix and 4x4 Jacobi eigen solve.
- [x] Make `alignGpu()` use the GPU step solver and remove the private CPU Kabsch/SVD step helper.

### Task 11: Stats Degeneracy Flags

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add CUDA stats tests requiring non-collinear geometry flags for valid and collinear correspondence sets.
- [x] Derive source and target rank-2 geometry flags from covariance invariants while building the reduced stats summary.
- [x] Make `alignGpu()` use stats geometry flags instead of running CPU SVD on 3x3 covariance matrices.
- [x] Remove the GPU path's private CPU SVD degeneracy helper.

### Task 12: Conditional Final Stats

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test counter showing a two-iteration GPU ICP run should call correspondence stats three times instead of four.
- [x] Keep the counter compiled only for test builds with `PLAPOINT_ENABLE_TESTING`.
- [x] Skip transformed final-stats scans on non-terminal iterations.
- [x] Still compute final metrics for convergence and max-iteration exit cases.

### Task 13: Lazy CPU Final Transform

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test requiring GPU align to leave the CPU final transform cache invalid.
- [x] Keep `getFinalTransformationDevice()` available immediately after GPU align.
- [x] Materialize the CPU final transform lazily when `getFinalTransformation()` is called.
- [x] Preserve the CPU ICP path's eager CPU final transform behavior.

### Task 14: Device-Side Identity Transform Initialization

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test requiring a GPU helper to write a column-major 4x4 identity transform.
- [x] Implement synchronous and async float/double identity-transform write helpers.
- [x] Initialize `alignGpu()` accumulated transform on GPU instead of using `identity4x4().toGpu()`.

### Task 15: Async Transform Accumulation

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test requiring async 4x4 transform multiplication on a caller-provided stream.
- [x] Keep the existing synchronous `multiplyTransform4x4()` API behavior unchanged.
- [x] Make `alignGpu()` queue transform accumulation asynchronously on stream 0.

### Task 16: Direct Initial Source Buffer Read

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test hook requiring the first GPU ICP stats scan to read the input source buffer directly.
- [x] Remove the startup device-to-device copy from source points into a temporary current-points buffer.
- [x] Keep transformed points in caller-owned GPU double buffers after the first iteration.

### Task 17: Finite-Radius Correspondence Pruning

**Files:**
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test counter requiring far targets to be rejected before full squared-distance evaluation.
- [x] Prune candidates by axis deltas when `max_correspondence_distance` is finite.
- [x] Preserve exact nearest-neighbor and final max-distance acceptance semantics.

### Task 18: Finite-Radius Target Tile Skipping

**Files:**
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test counter requiring far target tiles to be skipped before the per-candidate loop.
- [x] Compute per-tile target bounding boxes when finite-radius pruning is enabled.
- [x] Skip per-source scans of target tiles outside the expanded correspondence radius.

### Task 19: Device-Side Stats Step Transform Input

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test hook requiring `alignGpu()` to avoid host-to-device step input copies.
- [x] Add a device-stats step-transform helper that reads the latest stats workspace reduction.
- [x] Route `alignGpu()` through the device-stats helper while preserving host-side validation and metrics.

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
- `cmake --build build-codex-cuda -j$(nproc)` after adding `TransformPointsColumnMajorWritesCallerOwnedOutput`:
  failed as expected because `plapoint::gpu::transformPointsColumnMajor` did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.TransformPointsColumnMajorWritesCallerOwnedOutput:ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment`:
  9 targeted ICP GPU/stats tests passed after adding caller-owned point transform output and `alignGpu()` double buffering.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `StepTransformFromStatsWritesDeviceTransform`:
  failed as expected because `plapoint::gpu::computeIcpStepTransformFromStats` did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPValidation.RecoversKnownTransform`:
  11 targeted ICP GPU/stats/validation tests passed after switching `alignGpu()` to the GPU step solver.
- `cmake --build build-codex-cuda -j$(nproc)` after adding stats geometry flag assertions:
  failed as expected because `IcpCorrespondenceStats` did not expose source/target non-collinear geometry flags.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  13 targeted ICP GPU/stats/validation tests passed after moving GPU-path degeneracy checks off CPU SVD.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `AlignSkipsFinalStatsForNonTerminalGpuIterations`:
  failed as expected because the test-only stats call counter hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignSkipsFinalStatsForNonTerminalGpuIterations` before the conditional final-stats change:
  failed as expected with 4 correspondence stats calls instead of the expected 3.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignSkipsFinalStatsForNonTerminalGpuIterations`:
  1 targeted test passed after skipping transformed final-stats scans on non-terminal iterations.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  14 targeted ICP GPU/stats/validation tests passed after the conditional final-stats change.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `GpuAlignMaterializesCpuFinalTransformLazily`:
  failed as expected because `_final_T_cpu_valid` did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.GpuAlignMaterializesCpuFinalTransformLazily`:
  1 targeted test passed after making the legacy CPU final transform copy lazy for GPU align.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  15 targeted ICP GPU/stats/validation tests passed after lazy CPU final transform materialization.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `SetIdentityTransform4x4WritesColumnMajorIdentity`:
  failed as expected because `plapoint::gpu::setIdentityTransform4x4` did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.SetIdentityTransform4x4WritesColumnMajorIdentity`:
  1 targeted test passed after adding the GPU identity-transform helper.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  16 targeted ICP GPU/stats/validation tests passed after using GPU identity initialization in `alignGpu()`.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `MultiplyTransform4x4AsyncUsesCallerStream`:
  failed as expected because `plapoint::gpu::multiplyTransform4x4Async` did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.MultiplyTransform4x4AsyncUsesCallerStream`:
  1 targeted test passed after adding async transform multiplication.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  17 targeted ICP GPU/stats/validation tests passed after queueing transform accumulation asynchronously in `alignGpu()`.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `AlignReadsInitialGpuSourceBufferDirectly`:
  failed as expected because the test-only first-stats source pointer hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReadsInitialGpuSourceBufferDirectly`:
  1 targeted test passed after making the first stats scan read the input source GPU buffer.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  18 targeted ICP GPU/stats/validation tests passed after removing the startup source copy.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `CorrespondenceStatsPrunesFarTargetsBeforeFullDistanceEvaluation`:
  failed as expected because the test-only full-distance evaluation counter hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesFarTargetsBeforeFullDistanceEvaluation` before pruning:
  failed as expected with 3 full distance evaluations instead of the expected 1.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesFarTargetsBeforeFullDistanceEvaluation`:
  1 targeted test passed after adding finite-radius axis pruning.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  19 targeted ICP GPU/stats/validation tests passed after pruning far target candidates before full distance evaluation.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop`:
  failed as expected because the test-only target candidate visit counter hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop` before tile skipping:
  failed as expected with 257 target candidate visits instead of the expected at most 128.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop`:
  1 targeted test passed after skipping far target tiles.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  20 targeted ICP GPU/stats/validation tests passed after finite-radius target tile skipping.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `AlignComputesStepFromDeviceStatsWithoutHostInputCopy`:
  failed as expected because the test-only step-transform input copy counter hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy` before device-stats step input:
  failed as expected with 1 host-to-device step-transform input copy instead of the expected 0.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy`:
  1 targeted test passed after computing the GPU step transform from device-side stats.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  21 targeted ICP GPU/stats/validation tests passed after avoiding host-to-device step input copies in `alignGpu()`.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  197 tests, 0 failed.
- `cmake --build build-codex-cpu-bench -j$(nproc) && ./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`:
  CPU benchmark rows emitted through `cpu_icp_identity,512,1,33.1726`.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) && ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`:
  CUDA benchmark rows included `gpu_icp_identity,512,1,0.376097`.
