# PlaPoint GPU ICP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `IterativeClosestPoint<Scalar, GPU>` away from full CPU-staged point processing and keep the per-point ICP workload on GPU.

**Architecture:** Keep the existing CPU ICP path unchanged. Add CUDA helpers that accept PlaMatrix column-major GPU buffers, compute nearest-neighbor correspondences with a cached finite-radius target spatial grid or the shared-memory target-tiling fallback, use precomputed finite-radius target tile bounding-box skips and per-candidate axis pruning on the fallback path, accumulate centroid/covariance/residual stats on device with block-level reductions, fuse stats reduction with rigid step-transform solving through a GPU quaternion/Jacobi solver, initialize and asynchronously update the 4x4 accumulated transform on device, and expose the GPU final transform to callers. Reduced stats, step deltas, and small metric checks still synchronize to CPU while current points, step transforms, degeneracy flags, and transform accumulation stay GPU-resident. The GPU align loop reads the initial source buffer directly, skips transformed final-stats scans on non-terminal iterations, computes final metrics only when convergence or the final iteration requires them, and can skip terminal final metrics in the opt-in throughput mode. The legacy CPU final transform is copied from the GPU lazily only when `getFinalTransformation()` is called.

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

### Task 20: Fused Stats And Step Host Synchronization

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test counter requiring one iteration of `alignGpu()` to avoid a separate step-result host synchronization.
- [x] Add a fused stats-and-step helper that launches stats reduction and step solving before one host synchronization.
- [x] Keep `alignGpu()` error priority by validating reduced stats before consuming the fused step result.

### Task 21: Precomputed Finite-Radius Target Tile Bounds

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only test counter requiring finite-radius target tile bounds to be computed once per target tile.
- [x] Store reusable target tile bounds in `IcpCorrespondenceStatsWorkspace`.
- [x] Feed precomputed bounds into stats kernels so source blocks do not repeat tile bound reductions.

### Task 22: Finite-Radius Target Spatial Grid

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `README.md`

- [x] Add a CUDA-only failing test showing tile bounding boxes still visit too many far candidates when one tile mixes near and far targets.
- [x] Store reusable sorted target cell keys, sorted indices, cell starts, and cell counts in `IcpCorrespondenceStatsWorkspace`.
- [x] Build a finite-radius target spatial grid with Thrust sort/reduce and scan only the source cell's 27 neighboring cells.
- [x] Keep the existing target-tiling fallback for infinite radius and zero-radius direct helper calls.
- [x] Add finite-radius CPU/GPU ICP benchmark rows.

### Task 23: Cached Finite-Radius Target Spatial Grid

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`

- [x] Add CUDA-only tests proving repeated same-target finite-radius stats and a finite-radius `alignGpu()` reuse one target spatial grid.
- [x] Cache target spatial-grid metadata in `IcpCorrespondenceStatsWorkspace` by target pointer, target count, and cell size.
- [x] Reuse cached cell keys, sorted indices, cell starts, and counts when the target grid identity is unchanged.
- [x] Invalidate the cache on spatial-grid workspace growth or explicit `invalidateTargetSpatialGridCache()`.

### Task 24: Finite-Radius Translation ICP Benchmark

**Files:**
- Modify: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add deterministic translated-source grid data for non-identity ICP benchmark coverage.
- [x] Add CPU and CUDA finite-radius translation ICP benchmark rows.
- [x] Run CPU/CUDA benchmark smoke and record the new rows.

### Task 25: Persistent GPU ICP Workspaces Across Align Calls

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add CUDA-only repeated-`align()` coverage requiring GPU stats, target spatial-grid, and step workspaces to persist on the ICP object.
- [x] Reuse `IcpCorrespondenceStatsWorkspace` and `IcpStepTransformWorkspace` across repeated GPU `align()` calls.
- [x] Invalidate the persistent finite-radius target spatial-grid cache when `setInputTarget()` is called.
- [x] Add repeated-object CPU/CUDA finite-radius translation benchmark rows.
- [x] Run targeted GPU ICP tests and benchmark smoke.
- [x] Run full CPU/CUDA tests.

### Task 26: Persistent GPU ICP Transform Buffers

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Extend repeated-`align()` CUDA-only coverage to require reusable 4x4 step and accumulated transform buffers.
- [x] Persist GPU ICP step, accumulated-transform, and next accumulated-transform matrices on the ICP object.
- [x] Return `getFinalTransformationDevice()` from the persistent accumulated-transform buffer instead of moving a per-call matrix.
- [x] Run targeted GPU ICP tests and benchmark smoke.
- [x] Run full CPU/CUDA tests.

### Task 27: Persistent GPU ICP Point Scratch Buffers

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Extend repeated-`align()` CUDA-only coverage to require reusable transformed-source point scratch buffers.
- [x] Persist the two GPU transformed-source scratch matrices on the ICP object.
- [x] Copy final aligned GPU points into an independent output cloud so output ownership remains stable while scratch buffers are reused.
- [x] Run targeted GPU ICP tests and benchmark smoke.
- [x] Run full CPU/CUDA tests.

### Task 28: Reuse Caller-Owned GPU ICP Output Buffers

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add CUDA-only coverage requiring repeated `align()` calls into the same plain same-shaped output cloud
  to reuse caller-owned GPU point storage.
- [x] Preserve old output semantics for attributed or metadata-bearing output clouds by replacing them instead of
  reusing storage that would retain stale normals, colors, mesh, material, or texture data.
- [x] Reuse caller-owned GPU output point storage at the final copy step when the output cloud is plain and
  shape-compatible.
- [x] Add a CUDA benchmark row for repeated finite-radius translation ICP with both the ICP object and caller
  output object reused.
- [x] Run targeted GPU ICP tests and benchmark smoke.
- [x] Run full CPU/CUDA tests.

### Task 29: Direct Terminal GPU ICP Transform Into Caller Output

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add a CUDA-only test hook for the most recent point-transform output pointer.
- [x] Add CUDA-only coverage requiring terminal GPU ICP transforms to write directly into a reusable caller
  output cloud instead of internal scratch.
- [x] Keep source/target-aliased output clouds on the scratch-and-copy fallback path so final stats do not observe
  mutated inputs.
- [x] Direct terminal transforms into plain non-input caller output storage and skip the final device-to-device copy.
- [x] Run targeted GPU ICP tests and benchmark smoke.
- [x] Run full CPU/CUDA tests.

### Task 30: Scalable GPU ICP Benchmark Controls

**Files:**
- Modify: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add `--icp-points` so ICP benchmark rows can scale independently from the general KNN/voxel/normal rows.
- [x] Add `--icp-max-iterations` so GPU ICP iteration count is explicit in scaling experiments.
- [x] Add `--skip-cpu-icp` so large GPU ICP runs are not dominated by CPU ICP baselines.
- [x] Add `--skip-icp-identity` so large finite-radius GPU ICP runs can avoid the infinite-radius identity baseline.
- [x] Preserve the existing default 512-point, 3-ICP-iteration benchmark behavior.
- [x] Run CPU/CUDA benchmark smoke with default settings.
- [x] Run a larger GPU ICP benchmark smoke with CPU ICP skipped.
- [x] Run full CPU/CUDA tests.

### Task 31: Optional Final Metrics And ICP Phase Benchmarks

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_test.cpp`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `benchmarks/plapoint_benchmarks.cpp`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add an opt-in `setComputeFinalMetrics(false)` throughput mode that skips the terminal post-transform
  fitness/RMSE correspondence scan while preserving the default metric behavior.
- [x] Cover the CPU path to prove the final aligned output and transform remain correct when final metrics are
  disabled.
- [x] Cover the GPU path with the test-only stats-call counter to prove the terminal final-stats scan is skipped
  only when final metrics are disabled.
- [x] Add large-point benchmark rows for skip-final-metrics ICP, one-shot stats+step with a new workspace,
  cached-grid stats+step, and cached-grid stats-only.
- [x] Run targeted ICP tests, full CPU/CUDA tests, and benchmark smoke after the final documentation update.

### Task 32: Batched Spatial-Grid Neighbor Cell Lookups

**Files:**
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add a CUDA-only test counter proving the old finite-radius spatial-grid path performs 27 cell lookups per
  source point.
- [x] Replace per-neighbor exact-cell binary searches with one lower-bound lookup per `(x,y)` neighbor pair and
  a short sequential scan across adjacent `z` cells.
- [x] Preserve candidate visit and correspondence semantics for the cached finite-radius spatial-grid path.
- [x] Run targeted spatial-grid tests, full CPU/CUDA tests, and the 100k finite-radius GPU ICP benchmark.

### Task 33: Spatial-Grid Cell Distance Pruning

**Files:**
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add a CUDA-only candidate-visit counter test showing the cached spatial-grid path still scans all 27
  neighboring cell candidates when a same-cell exact match already proves the best distance.
- [x] Visit the same `(x,y)` cell group first, then prune cells whose cell AABB lower-bound distance cannot beat
  the current best correspondence distance or fit inside the max correspondence radius.
- [x] Keep equal-distance cells conservative and make finite-grid ties choose the lower target index.
- [x] Run targeted spatial-grid tests, full CPU/CUDA tests, and the 100k finite-radius GPU ICP benchmark.

### Task 34: Lightweight Terminal Residual Metrics

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add a lightweight GPU residual stats helper that computes only active count, invalid source count, and
  residual sum for final fitness/RMSE metrics.
- [x] Keep the main iteration path on full correspondence moment/covariance stats for step solving and degeneracy
  checks.
- [x] Route default terminal final metrics through the lightweight residual stats helper.
- [x] Keep `setComputeFinalMetrics(false)` behavior unchanged.
- [x] Run targeted final-metrics tests, full CPU/CUDA tests, and the 100k finite-radius GPU ICP benchmark.

### Task 35: Fused Terminal Residual Output And Exact-Match Grid Early Exit

**Files:**
- Modify: `include/plapoint/gpu/icp.h`
- Modify: `include/plapoint/registration/icp.h`
- Modify: `src/icp_gpu.cu`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Add a fused GPU helper that applies the terminal ICP step transform into caller-owned output storage and
  computes final residual metrics in the same source scan.
- [x] Route terminal default final metrics through the fused helper while keeping the opt-in skip-final-metrics
  path on the transform-only fast path.
- [x] Add test-only transform-kernel call counting to prove terminal default final metrics no longer launch the
  standalone point-transform helper.
- [x] Stop finite-radius residual spatial-grid lookup after an exact zero-distance match, avoiding the remaining
  neighboring `(x, y)` cell lookups for already aligned points.
- [x] Run targeted final-metrics/residual tests, full CPU/CUDA tests, and the 100k finite-radius GPU ICP benchmark.

### Task 36: Exact-Identity Terminal Fast Path

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Detect terminal GPU ICP steps whose solved delta is exactly zero.
- [x] Reuse the iteration stats as final metrics for exact-identity terminal steps instead of launching terminal
  transform, transform-accumulation, or residual-stats kernels.
- [x] Skip the final device-to-device output copy when the output buffer is already the current point buffer.
- [x] Keep non-identity terminal metrics on the fused residual-output path and keep skip-final-metrics coverage on
  a non-identity case.
- [x] Run targeted identity/non-identity final-metrics tests, full CPU/CUDA tests, and the 100k finite-radius GPU
  ICP benchmark.

### Task 37: Lazy GPU Point Scratch Allocation

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Modify: `test/unit/registration/icp_gpu_path_test.cpp`
- Modify: `docs/superpowers/plans/2026-06-05-plapoint-gpu-icp.md`

- [x] Allocate GPU transformed-point scratch buffers only when a transform cannot write directly to the caller
  output or must avoid mutating aliased input.
- [x] Keep terminal non-identity direct-output and exact-identity terminal paths free of internal point scratch
  allocations.
- [x] Preserve scratch allocation and reuse when output aliases the source or target cloud.
- [x] Update GPU ICP path tests to distinguish direct-output, exact-identity, and alias-scratch cases.
- [x] Run targeted scratch/output tests, full CPU/CUDA tests, and the 100k finite-radius GPU ICP benchmark.

Verification evidence:

- `git diff --check && cmake --build build-codex-cuda -j$(nproc)`:
  CUDA test build passed after making GPU ICP point scratch allocation lazy.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls:ICPGpuPathTest.AlignReusesCallerGpuOutputStorageWhenShapeMatches:ICPGpuPathTest.AlignWritesTerminalGpuTransformDirectlyToReusableOutput:ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesSource:ICPGpuPathTest.AlignReusesIterationStatsForExactIdentityTerminalMetrics:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled`:
  7 targeted scratch/output tests passed. Direct-output and exact-identity cases leave internal point scratch null,
  while aliased output still allocates and reuses scratch.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) &&
  ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius,100000,5,1.68297`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,3.09038`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.60597`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.3107`.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  39 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  216 tests, 0 failed.
- `git diff --check && cmake --build build-codex-cuda -j$(nproc)`:
  CUDA test build passed after adding the exact-identity terminal fast path.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesIterationStatsForExactIdentityTerminalMetrics:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled:ICPGpuPathTest.AlignWritesTerminalGpuTransformDirectlyToReusableOutput:ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesSource:ICPGpuPathTest.AlignDoesNotPopulateGpuPointCpuCaches:ICPGpuPathTest.AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls`:
  7 targeted identity/non-identity terminal tests passed. The new identity test confirms exact-identity terminal
  metrics use one stats call, zero residual-stats calls, and zero standalone point-transform calls.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) &&
  ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius,100000,5,1.85335`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,3.10362`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.61417`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.31886`.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  39 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  216 tests, 0 failed.
- `git diff --check && cmake --build build-codex-cuda -j$(nproc)`:
  CUDA test build passed after adding fused terminal residual output and exact-match residual grid early exit.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled:ICPGpuPathTest.AlignWritesTerminalGpuTransformDirectlyToReusableOutput:ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesSource`:
  5 targeted residual/final-output tests passed. The new residual test proves exact-match final residual scans stop
  after one spatial-grid lookup per source point.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) &&
  ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 2 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius_translation_reuse_output,100000,2,3.09521`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,2,2.62227`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,2,1.35625`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,2,1.31944`.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  38 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  215 tests, 0 failed.
- `git diff --check && cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignSkipsFinalStatsForNonTerminalGpuIterations:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled`:
  3 targeted final-metrics tests passed. The new test confirms terminal default metrics use one residual-stats scan,
  while the opt-in skip-final-metrics mode still skips it.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  37 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) &&
  ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 2 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius_translation_reuse_output,100000,2,3.19518`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,2,2.61584`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,2,1.34983`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,2,1.31153`.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  214 tests, 0 failed.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance`
  before cell-distance pruning:
  failed as expected with 27 target candidate visits instead of the expected at most 2.
- `git diff --check && cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPGpuPathTest.CorrespondenceStatsBatchesSpatialGridNeighborLookupsByXY:ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget:ICPGpuPathTest.AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls`:
  6 targeted spatial-grid tests passed after adding cell AABB pruning and deterministic finite-grid tie handling.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  36 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) &&
  ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 2 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius_translation_reuse_output,100000,2,3.81614`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,2,2.60521`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,2,1.34839`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,2,1.31845`.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  213 tests, 0 failed.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsBatchesSpatialGridNeighborLookupsByXY`
  before batching spatial-grid lookups:
  failed as expected with 27 grid cell lookups instead of the expected at most 9.
- `git diff --check && cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsBatchesSpatialGridNeighborLookupsByXY:ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget:ICPGpuPathTest.AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization`:
  5 targeted spatial-grid/fused-step tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  34 targeted ICP GPU/stats/validation tests passed after batching spatial-grid neighbor lookups.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) &&
  ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 2 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius_translation_reuse_output,100000,2,5.85697`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,2,3.91724`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,2,1.93988`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,2,1.89291`.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  211 tests, 0 failed.
- `cmake --build build-codex-cuda -j$(nproc)` after adding the CPU/GPU final-metrics opt-out tests:
  failed as expected because `IterativeClosestPoint<..., CPU/GPU>` had no `setComputeFinalMetrics()` member.
- `git diff --check && cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPTest.CanDisableFinalMetricComputationForThroughput:ICPTest.FinalRmseReflectsResidualAfterLastStep:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled:ICPGpuPathTest.AlignSkipsFinalStatsForNonTerminalGpuIterations`:
  4 targeted final-metrics tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  33 targeted ICP GPU/stats/validation tests passed after adding the opt-in final-metrics skip.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  210 tests, 0 failed.
- `cmake --build build-codex-cpu-bench -j$(nproc) &&
  ./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`:
  CPU benchmark rows included `cpu_icp_identity,512,1,32.5152` and
  `cpu_icp_finite_radius_translation_reuse,512,1,30.6308`.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) &&
  ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 2 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius_translation_reuse_output,100000,2,8.78242`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,2,5.87974`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,2,3.3311`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,2,2.91339`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,2,2.85862`.
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
- `cmake --build build-codex-cuda -j$(nproc)` after adding `AlignFusesStatsAndStepToAvoidExtraHostSynchronization`:
  failed as expected because the test-only host synchronization counter hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization` before fusing stats and step:
  failed as expected with 3 host synchronizations instead of the expected 2.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization`:
  1 targeted test passed after fusing stats reduction and step-transform solving.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  22 targeted ICP GPU/stats/validation tests passed after reducing the per-iteration host synchronization count.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `CorrespondenceStatsPrecomputesFiniteRadiusTargetTileBoundsOnce`:
  failed as expected because the test-only target tile bound computation counter hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrecomputesFiniteRadiusTargetTileBoundsOnce` before precomputing target tile bounds:
  failed as expected with 9 target tile bound computations instead of the expected at most 3.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrecomputesFiniteRadiusTargetTileBoundsOnce`:
  1 targeted test passed after precomputing finite-radius target tile bounds.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  23 targeted ICP GPU/stats/validation tests passed after reusing target tile bounds from the stats workspace.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates` before the spatial grid:
  failed as expected with 128 target candidate visits instead of the expected at most 4.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates`:
  1 targeted test passed after scanning finite-radius target spatial-grid candidates.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPGpuPathTest.CorrespondenceStatsPrecomputesFiniteRadiusTargetTileBoundsOnce:ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates`:
  3 targeted finite-radius tile/grid pruning tests passed after separating zero-radius tile fallback coverage from finite-radius spatial-grid coverage.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  24 targeted ICP GPU/stats/validation tests passed after adding the finite-radius target spatial grid.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget` after adding the reuse test:
  failed as expected because the test-only target spatial-grid build counter hooks did not exist yet.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget` before caching:
  failed as expected with 4 target spatial-grid builds instead of the expected 3.
- `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget:ICPGpuPathTest.AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls`:
  2 targeted finite-radius spatial-grid cache tests passed after caching by target pointer, target count, and cell size.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  26 targeted ICP GPU/stats/validation tests passed after adding cached finite-radius target spatial-grid reuse.
- `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure`:
  202 tests, 0 failed.
- `cmake --build build-codex-cpu-bench -j$(nproc) && ./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`:
  CPU benchmark rows emitted through `cpu_icp_identity,512,1,58.2848` and `cpu_icp_finite_radius,512,1,56.3166`.
- `cmake --build build-codex-cuda-bench-only -j$(nproc) && ./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1`:
  CUDA benchmark rows included `gpu_icp_identity,512,1,0.378443` and `gpu_icp_finite_radius,512,1,0.265103`.
- `cmake --build build-codex-cpu-bench -j$(nproc)` and
  `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after adding finite-radius translation benchmark coverage:
  CPU benchmark rows included `cpu_icp_finite_radius_translation,512,1,30.5572`.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after adding finite-radius translation benchmark coverage:
  CUDA benchmark rows included `gpu_icp_finite_radius_translation,512,1,0.360243`.
- `git diff --check` after adding finite-radius translation benchmark coverage:
  clean.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
  202 tests, 0 failed.
- `cmake --build build-codex-cuda -j$(nproc)` after adding `AlignReusesGpuWorkspacesAcrossRepeatedCalls`:
  failed as expected because `IterativeClosestPoint<float, GPU>` had no persistent `_gpu_stats_workspace`
  or `_gpu_step_workspace` members.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls:ICPGpuPathTest.SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCache`:
  2 targeted persistent workspace/cache tests passed after moving GPU ICP workspaces onto the ICP object.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  28 targeted ICP GPU/stats/validation tests passed after persistent GPU ICP workspaces.
- `cmake --build build-codex-cpu-bench -j$(nproc)` and
  `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after adding repeated-object finite-radius translation benchmark coverage:
  CPU benchmark rows included `cpu_icp_finite_radius_translation_reuse,512,1,28.8398`.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after adding repeated-object finite-radius translation benchmark coverage:
  CUDA benchmark rows included `gpu_icp_finite_radius_translation,512,1,0.364015` and
  `gpu_icp_finite_radius_translation_reuse,512,1,0.282149`.
- `git diff --check` after persistent GPU ICP workspaces:
  clean.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
  204 tests, 0 failed.

- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesCallerGpuOutputStorageWhenShapeMatches:ICPGpuPathTest.AlignReplacesAttributedGpuOutputInsteadOfKeepingStaleMetadata`
  after adding output-storage reuse tests:
  `AlignReusesCallerGpuOutputStorageWhenShapeMatches` failed as expected because `alignGpu()` still allocated a
  fresh output point matrix on each call, while the attributed-output replacement test passed under the old behavior.
- `cmake --build build-codex-cuda -j$(nproc)` after first implementing the output-buffer helper:
  failed because the non-SFINAE helper was explicitly instantiated for CPU ICP types; constraining the helper to
  `Device::GPU` fixed the build.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesCallerGpuOutputStorageWhenShapeMatches:ICPGpuPathTest.AlignReplacesAttributedGpuOutputInsteadOfKeepingStaleMetadata`:
  2 targeted GPU output-storage tests passed after adding safe caller-owned output reuse.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  30 targeted ICP GPU/stats/validation tests passed after caller-owned output reuse.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after caller-owned
  output reuse:
  CUDA benchmark rows included `gpu_icp_finite_radius_translation,512,1,0.367608`,
  `gpu_icp_finite_radius_translation_reuse,512,1,0.26877`, and
  `gpu_icp_finite_radius_translation_reuse_output,512,1,0.264002`.
- `cmake --build build-codex-cpu-bench -j$(nproc)` and
  `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after caller-owned
  output reuse:
  CPU benchmark rows included `cpu_icp_finite_radius_translation,512,1,31.0323` and
  `cpu_icp_finite_radius_translation_reuse,512,1,31.5042`.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
  206 tests, 0 failed.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignWritesTerminalGpuTransformDirectlyToReusableOutput`
  after adding the terminal direct-output test hook and test:
  failed as expected because the final transform still wrote `_gpu_points_a` and then copied into output.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesCallerGpuOutputStorageWhenShapeMatches:ICPGpuPathTest.AlignWritesTerminalGpuTransformDirectlyToReusableOutput:ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesSource:ICPGpuPathTest.AlignReplacesAttributedGpuOutputInsteadOfKeepingStaleMetadata`:
  4 targeted GPU output-path tests passed after direct terminal output writes and input-alias fallback.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  32 targeted ICP GPU/stats/validation tests passed after direct terminal output writes.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after direct
  terminal output writes:
  CUDA benchmark rows included `gpu_icp_finite_radius_translation,512,1,0.361795`,
  `gpu_icp_finite_radius_translation_reuse,512,1,0.263994`, and
  `gpu_icp_finite_radius_translation_reuse_output,512,1,0.26528`.
- `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5` after direct
  terminal output writes:
  CUDA benchmark rows included `gpu_icp_finite_radius_translation,512,5,0.357869`,
  `gpu_icp_finite_radius_translation_reuse,512,5,0.260998`, and
  `gpu_icp_finite_radius_translation_reuse_output,512,5,0.260873`.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
  208 tests, 0 failed.
- `cmake --build build-codex-cpu-bench -j$(nproc)` and
  `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after scalable
  ICP benchmark controls:
  CPU benchmark rows still used the default ICP size, including
  `cpu_icp_identity,512,1,28.9231` and `cpu_icp_finite_radius_translation_reuse,512,1,27.4772`.
- `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --help` after scalable ICP benchmark controls:
  help output included `--icp-points`, `--icp-max-iterations`, `--skip-cpu-icp`, and `--skip-icp-identity`.
- `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after scalable
  ICP benchmark controls:
  CUDA benchmark rows still used the default ICP size, including
  `gpu_icp_finite_radius_translation_reuse_output,512,1,0.262835`.
- `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 2 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius,100000,2,6.59437`,
  `gpu_icp_finite_radius_translation,100000,2,9.4794`,
  `gpu_icp_finite_radius_translation_reuse,100000,2,8.8691`, and
  `gpu_icp_finite_radius_translation_reuse_output,100000,2,8.79715`.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
  208 tests, 0 failed.
- `cmake --build build-codex-cuda -j$(nproc)` after extending `AlignReusesGpuWorkspacesAcrossRepeatedCalls`
  to check 4x4 transform buffers:
  failed as expected because `IterativeClosestPoint<float, GPU>` had no persistent `_gpu_T_step`, `_gpu_T_acc`,
  or `_gpu_next_T_acc` members.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls:ICPGpuPathTest.FinalTransformationDeviceIsAvailableAfterGpuAlign:ICPGpuPathTest.GpuAlignMaterializesCpuFinalTransformLazily`:
  3 targeted transform-buffer/final-transform tests passed after persisting GPU ICP transform buffers.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  28 targeted ICP GPU/stats/validation tests passed after persistent GPU ICP transform buffers.
- `cmake --build build-codex-cpu-bench -j$(nproc)` and
  `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after persistent GPU ICP transform buffers:
  CPU benchmark rows included `cpu_icp_finite_radius_translation_reuse,512,1,29.6661`.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after persistent GPU ICP transform buffers:
  CUDA benchmark rows included `gpu_icp_finite_radius_translation,512,1,0.363721` and
  `gpu_icp_finite_radius_translation_reuse,512,1,0.27001`.
- `git diff --check` after persistent GPU ICP transform buffers:
  clean.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
  204 tests, 0 failed.
- `cmake --build build-codex-cuda -j$(nproc)` after extending `AlignReusesGpuWorkspacesAcrossRepeatedCalls`
  to check transformed-source point scratch buffers:
  failed as expected because `IterativeClosestPoint<float, GPU>` had no persistent `_gpu_points_a`
  or `_gpu_points_b` members.
- `cmake --build build-codex-cuda -j$(nproc) &&
  ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls:ICPGpuPathTest.FinalTransformationDeviceIsAvailableAfterGpuAlign:ICPGpuPathTest.GpuAlignMaterializesCpuFinalTransformLazily:ICPGpuPathTest.AlignDoesNotPopulateGpuPointCpuCaches`:
  4 targeted persistent point-scratch/final-transform tests passed after persisting GPU ICP point scratch buffers.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  28 targeted ICP GPU/stats/validation tests passed after persistent GPU ICP point scratch buffers.
- `cmake --build build-codex-cpu-bench -j$(nproc)` and
  `./build-codex-cpu-bench/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after persistent GPU ICP point scratch buffers:
  CPU benchmark rows included `cpu_icp_finite_radius_translation_reuse,512,1,31.0539`.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1` after persistent GPU ICP point scratch buffers:
  CUDA benchmark rows included `gpu_icp_finite_radius_translation,512,1,0.365436` and
  `gpu_icp_finite_radius_translation_reuse,512,1,0.266467`.
- `git diff --check` after persistent GPU ICP point scratch buffers:
  clean.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  142 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
  204 tests, 0 failed.

## Task 38: First-Iteration Transform Accumulation Fast Path

- Goal: skip the initial GPU ICP accumulated-transform 4x4 multiply. The accumulated transform starts as identity, so
  the first non-identity step can become the accumulated transform by swapping the persistent step and accumulated
  transform buffers instead of launching `multiplyTransform4x4Kernel`.
- Implementation:
  - Captured `_gpu_T_step->data()` as `step_transform` before any buffer-role swap so point transformation and fused
    final residual stats still use the current iteration step.
  - Swapped `_gpu_T_acc` and `_gpu_T_step` only for the first non-identity iteration; later iterations still multiply
    `step * accumulated` into `_gpu_next_T_acc`.
  - Added a testing-only transform-multiply launch counter and updated GPU ICP path tests to assert that one-iteration
    non-identity and skip-final-metrics alignments avoid the 4x4 multiply launch.
  - Updated the repeated-align workspace test to verify the three transform buffers are reused as a set, because buffer
    roles may rotate after the first-iteration swap.
- `git diff --check` after the fast path:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls:ICPGpuPathTest.FinalTransformationDeviceIsAvailableAfterGpuAlign:ICPGpuPathTest.GpuAlignMaterializesCpuFinalTransformLazily`:
  4 targeted GPU ICP transform/workspace tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  39 targeted ICP GPU/stats/validation tests passed.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  216 tests, 0 failed.
- `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius,100000,5,1.68775`,
  `gpu_icp_finite_radius_translation,100000,5,3.78043`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,3.14122`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,3.10376`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.61468`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.34966`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.31061`.
  The 4x4 multiply skip removes one small launch in one-iteration paths; measured end-to-end speed is within run-to-run
  noise because nearest-neighbor/grid work remains dominant at this scale.

## Task 39: Omitted-Index Correspondence Exact-Match Grid Early Exit

- Goal: reduce finite-radius spatial-grid correspondence work for already-aligned or exact-match points. When
  `correspondence_indices` is null, the stats path does not need deterministic target-index tie resolution; once an
  exact zero-distance correspondence is found, further cell/candidate scans cannot change the accumulated ICP stats.
- Implementation:
  - Added exact-match early exit to `collectCorrespondenceStatsSpatialGridKernel` only when
    `correspondence_indices == nullptr`.
  - Left index-writing calls on the full scan path so equal-distance ties still choose the lower target index.
  - Tightened existing residual exact-match early exits by making the active z-cell `while` loops honor
    `stop_cell_scan`, not only the outer xy loops.
  - Added CUDA test coverage proving omitted-index correspondence visits one exact candidate and performs one grid
    lookup, while the indexed call still scans additional candidates and writes the expected target index.
- `git diff --check` after the fast path:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsStopsSpatialGridAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.CorrespondenceStatsBatchesSpatialGridNeighborLookupsByXY`:
  5 targeted finite-radius grid/exact-match tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  40 targeted ICP GPU/stats/validation tests passed.
- `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  217 tests, 0 failed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp --skip-icp-identity`:
  large finite-radius GPU ICP benchmark rows included
  `gpu_icp_finite_radius,100000,5,1.02394`,
  `gpu_icp_finite_radius_translation,100000,5,3.58469`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.96396`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.90484`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.42765`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.37249`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.33395`.
- `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  with `gpu_icp_identity` enabled, rows included
  `gpu_icp_identity,100000,5,691.875`,
  `gpu_icp_finite_radius,100000,5,0.93941`,
  `gpu_icp_finite_radius_translation,100000,5,3.26018`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.17734`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.19467`.
  The finite-radius exact-match path improved substantially; the infinite-radius identity benchmark remains a separate
  full-scan bottleneck.

## Task 40: Exact Pointwise Stats Fast Path For Infinite-Radius Identity

- Goal: remove the infinite-radius O(NxM) nearest-neighbor scan for identity inputs where the source and target GPU
  point buffers contain the same points in the same order. This was the dominant remaining benchmark outlier:
  `gpu_icp_identity,100000,5` was around 691 ms after the finite-radius grid work.
- Implementation:
  - Added `collectExactPointwiseCorrespondenceStatsKernel`, which checks source/target rows pointwise and accumulates
    the same raw ICP stats as exact zero-distance correspondences.
  - Enabled the fast path only when correspondence indices are omitted and source/target counts match. It runs
    unconditionally for identical device pointers, and probes separate buffers only for non-finite correspondence
    radius where the normal path would otherwise be a full scan.
  - Falls back to the existing nearest-neighbor path if the pointwise equality probe finds any mismatch.
  - Kept index-writing calls on the existing full-scan path so requested correspondence indices and tie semantics are
    unchanged.
  - Adjusted the host-sync regression test to cover finite-radius fused stats+step, keeping the ordinary GPU ICP path
    at two host synchronizations while the infinite-radius equality probe remains an explicit tradeoff.
- `git diff --check` after the fast path:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization`:
  5 targeted identity fast-path/index/full-scan/host-sync tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  41 targeted ICP GPU/stats/validation tests passed.
- `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  218 tests, 0 failed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.322281`,
  `gpu_icp_finite_radius,100000,5,1.01601`,
  `gpu_icp_finite_radius_translation,100000,5,3.57202`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.94931`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.89765`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.42457`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.36599`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.33416`.
  The identity benchmark dropped from the previous 691 ms scale to sub-millisecond on this machine.

## Task 41: Single-Sync Exact Pointwise Stats+Step

- Goal: remove the extra host synchronization introduced by the exact pointwise identity probe in the fused
  stats+step path. After Task 40, exact infinite-radius identity avoided the O(NxM) scan but still synchronized once
  to validate the equality probe and again to read the step result.
- Implementation:
  - Split exact pointwise stats into a launch-only helper and the existing stats-only sync helper.
  - In `computeIcpStatsAndStepTransformColumnMajorImpl`, launch exact pointwise stats, reduce raw stats, compute the
    step transform from that raw stats buffer, then copy raw stats and step result back with one stream
    synchronization.
  - Preserve fallback behavior: if the pointwise probe reports a mismatch, run the existing nearest-neighbor
    correspondence path and compute the step normally.
  - Added a host-sync assertion to the exact infinite-radius identity align test, requiring one stats synchronization.
- `git diff --check` after the single-sync exact path:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.AlignSkipsFinalStatsForNonTerminalGpuIterations:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics`:
  6 targeted exact/fallback/index/host-sync tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  41 targeted ICP GPU/stats/validation tests passed.
- `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  218 tests, 0 failed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.305878`,
  `gpu_icp_finite_radius,100000,5,1.01823`,
  `gpu_icp_finite_radius_translation,100000,5,3.57709`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.96364`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.90675`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.42251`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.36473`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.33488`.

## Task 42: Exact Pointwise Identity Step Kernel

- Goal: remove the generic raw-stats-to-step solver from the exact pointwise identity path. The previous Task 41
  path already avoided the nearest-neighbor scan and used one host synchronization, but it still launched the
  full Jacobi/quaternion step solver even though exact pointwise correspondences imply identity transform and
  zero delta.
- Implementation:
  - Added `setExactPointwiseIdentityStepKernel`, which writes a column-major identity transform and reports
    `delta = 0` directly from the exact raw stats validity.
  - Added a launch helper so testing builds can count this fast-path step launch.
  - Switched only `computeIcpStatsAndStepTransformColumnMajorImpl`'s exact pointwise branch to the identity-step
    helper. The stats-only path and the normal nearest-neighbor/fallback step solver are unchanged.
  - Extended `AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs` to assert the identity-step fast path is
    used once, while full distance evaluations, target visits, and grid lookups remain zero.
- `git diff --check` after the identity-step specialization:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.AlignSkipsFinalStatsForNonTerminalGpuIterations:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics`:
  6 targeted exact/fallback/index/host-sync tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  41 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  218 tests, 0 failed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.280924`,
  `gpu_icp_finite_radius,100000,5,1.01802`,
  `gpu_icp_finite_radius_translation,100000,5,3.57185`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.95562`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.90308`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.4269`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.37322`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.3357`.
  The identity path improved from the previous 0.305878 ms record to 0.280924 ms on this run; the remaining cost is
  now mostly exact stats scan/reduction, kernel launch overhead, and the required final host copy/synchronization.

## Task 43: Fuse Exact Pointwise Stats Reduce And Identity Step

- Goal: remove one kernel launch from the exact pointwise identity stats+step path. After Task 42, the path still
  launched a raw stats reduction kernel and then a separate identity-step kernel.
- Implementation:
  - Added `reduceRawIcpStatsAndSetExactPointwiseIdentityStepKernel`, which reduces exact pointwise partial stats,
    writes the reduced raw stats buffer, writes the column-major identity step transform, and sets `delta = 0`
    plus the step-valid flag in one kernel.
  - Added `launchExactPointwiseStatsAndIdentityStep` for the fused stats+step entry point.
  - Removed the standalone exact pointwise identity-step kernel from the fused path.
  - Kept `launchExactPointwiseStats` unchanged for the stats-only API, and kept the normal nearest-neighbor fallback
    and generic step solver unchanged.
- `git diff --check` after the fused exact reduce+step path:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.AlignSkipsFinalStatsForNonTerminalGpuIterations:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics`:
  6 targeted exact/fallback/index/host-sync tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  41 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  218 tests, 0 failed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.283319`,
  `gpu_icp_finite_radius,100000,5,1.02232`,
  `gpu_icp_finite_radius_translation,100000,5,3.56708`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.94672`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.89868`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,2.43083`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.37289`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.33411`.
  The identity benchmark stayed within run-to-run noise compared with the previous 0.280924 ms record; reducing this
  path further likely requires attacking exact stats scan/reduction or the final host copy/synchronization.

## Task 44: Prune Spatial Grid XY Lookups Before Search

- Goal: reduce the finite-radius spatial-grid hot path by avoiding neighbor-cell binary searches whose x/y cell
  bounds already prove that no point in that x/y column can beat the correspondence radius or the current best
  distance.
- Implementation:
  - Added `minDistanceSqToIcpGridCellXY` to compute a source point's lower-bound squared distance to a candidate
    grid cell in x/y only.
  - In `collectCorrespondenceStatsSpatialGridKernel`, skip the `lowerBoundIcpGridCell` call when the x/y lower
    bound is outside `max_correspondence_distance` or strictly worse than the current best. The strict current-best
    comparison preserves lower-target-index tie behavior for correspondence stats.
  - Applied the same pre-lookup prune to `collectResidualStatsSpatialGridKernel` and
    `transformAndCollectResidualStatsSpatialGridKernel`, using `>= best` because residual-only stats do not need
    target-index tie preservation.
  - Added `CorrespondenceStatsPrunesSpatialGridXYLookupsBeforeSearch`, which constructs nine neighboring x/y cell
    candidates and requires lookup count to drop to at most five.
- `git diff --check` after the x/y lookup prune:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridXYLookupsBeforeSearch:ICPGpuPathTest.CorrespondenceStatsBatchesSpatialGridNeighborLookupsByXY:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled`:
  6 targeted spatial-grid/residual/align tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  42 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  219 tests, 0 failed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.27995`,
  `gpu_icp_finite_radius,100000,5,1.02104`,
  `gpu_icp_finite_radius_translation,100000,5,3.1017`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.50335`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.4625`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.96362`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.10259`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.07306`.
  Compared with Task 43's benchmark, the cached-grid finite-radius stats row improved from 1.33411 ms to 1.07306 ms,
  and the reusable skip-final-metrics align row improved from 2.43083 ms to 1.96362 ms on this run.

## Task 45: Fuse Normal Stats Reduction And Step Solve

- Goal: remove the standalone raw-stats-to-step kernel launch from the normal fused stats+step path. Before this
  task, `computeIcpStatsAndStepTransformColumnMajorImpl` launched correspondence stats, reduced raw stats, launched
  a separate step solver kernel, then copied stats and step result back.
- Implementation:
  - Added `reduceRawIcpStatsAndComputeStepTransformKernel`, which reduces partial raw ICP stats, writes the reduced
    stats buffer, builds the Kabsch/quaternion step input, and computes the step transform from thread 0 in the same
    kernel.
  - Switched only the normal stats+step path to this fused reduce+step kernel.
  - Kept the stats-only API on `reduceRawIcpStatsKernel` and kept `computeIcpStepTransformFromDeviceStats` on the
    existing standalone raw-stats step kernel.
  - Added a testing counter for standalone raw-stats step kernel launches and extended
    `AlignFusesStatsAndStepToAvoidExtraHostSynchronization` to require zero such launches during `align()`.
- `git diff --check` after the normal reduce+step fusion:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridXYLookupsBeforeSearch:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance`:
  5 targeted stats+step/spatial-grid/align tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  42 targeted ICP GPU/stats/validation tests passed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  219 tests, 0 failed.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.281052`,
  `gpu_icp_finite_radius,100000,5,1.02139`,
  `gpu_icp_finite_radius_translation,100000,5,3.11762`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.5013`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.4525`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.95691`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,5,1.53215`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.11012`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.07211`.
  The launch reduction is structurally useful, but this benchmark run stayed within noise compared with Task 44's
  1.10259 ms stats+step row; the hot cost is now dominated by correspondence search/reduction and the required host
  copy/synchronization rather than the removed small launch.

## Task 46: Use Compact Alignment Step Summary

- Goal: reduce per-iteration CPU/GPU transfer payload and avoid allocating the old step-transform workspace in
  `IterativeClosestPoint::alignGpu()`. The alignment loop only needs active/invalid counts, residual sum,
  non-collinear geometry flags, step validity, and step delta; it does not need the full host covariance summary.
- Implementation:
  - Added `IcpAlignmentStepResult` and `computeIcpAlignmentStepColumnMajor`, a compact stats+step path for the
    alignment loop.
  - Added compact device-side raw result generation that computes source/target non-collinear geometry flags on GPU
    and copies back a small summary instead of full `RawIcpStats` plus a separate step result.
  - Kept the public full `computeIcpStatsAndStepTransformColumnMajor` API unchanged for callers that need full
    covariance details.
  - Switched `alignGpu()` to the compact path and removed its persistent `IcpStepTransformWorkspace` member/reserve.
  - Added tests comparing compact and full stats+step outputs and verifying `align()` uses the compact path.
- `git diff --check` after the compact alignment step summary:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignmentStepCompactResultMatchesFullStatsStepResult:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled:ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls`:
  6 targeted compact-step/align/workspace tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  43 targeted ICP GPU/stats/validation tests passed.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  220 tests, 0 failed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.271869`,
  `gpu_icp_finite_radius,100000,5,1.01293`,
  `gpu_icp_finite_radius_translation,100000,5,3.10399`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.48447`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.45308`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.95356`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,5,1.53502`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.1019`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.07391`.
  The align rows improved slightly on this run; the standalone public stats+step rows are intentionally unchanged
  because the compact path is used by `alignGpu()` rather than by the full-stats benchmark rows.

## Task 47: Skip Redundant Target Finite Checks In Spatial-Grid Candidate Loops

- Goal: reduce per-candidate work in finite-radius spatial-grid ICP searches. Target-grid construction already maps
  non-finite target points to a sentinel cell key, and candidate acceptance still requires finite squared distance, so
  the spatial-grid candidate loops do not need to call `loadFiniteColumnMajorPoint` for every visited target.
- Implementation:
  - Added a lightweight `loadColumnMajorPoint` helper for hot candidate loops that have already gone through spatial
    grid pruning.
  - Switched the correspondence, residual, and transform+residual spatial-grid candidate loops to load coordinates
    directly and rely on existing axis radius checks plus `isfinite(dist_sq)` before accepting candidates.
  - Left the non-spatial-grid tile-scan fallback paths on `loadFiniteColumnMajorPoint`, because those paths scan all
    target tiles and need explicit target validity filtering.
  - Added `CorrespondenceStatsSpatialGridSkipsNonFiniteTargetInSaturatedCell`, which puts a non-finite target and a
    finite target in the same saturated grid cell and verifies the finite target is selected.
- `git diff --check` after the spatial-grid target finite-check prune:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSpatialGridSkipsNonFiniteTargetInSaturatedCell:ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
  5 targeted spatial-grid tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  44 targeted ICP GPU/spatial-grid/validation tests passed.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  221 tests, 0 failed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.270089`,
  `gpu_icp_finite_radius,100000,5,1.01112`,
  `gpu_icp_finite_radius_translation,100000,5,3.04044`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.4269`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.38473`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.90428`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,5,1.48977`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.06889`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,1.0321`.
  Compared with Task 46 on this machine, the cached-grid stats+step row improved from 1.1019 ms to 1.06889 ms,
  and the stats-only cached-grid row improved from 1.07391 ms to 1.0321 ms on this benchmark run.

## Task 48: Cache Sorted Target Coordinates For Spatial Grid

- Goal: reduce indirect/random target coordinate loads in finite-radius spatial-grid ICP candidate loops. After grid
  construction sorts target indices by cell, the hot correspondence and residual kernels can read sorted target
  coordinates directly from contiguous arrays instead of loading each point through the original column-major target
  buffer by index.
- Implementation:
  - Added reusable sorted target x/y/z storage to `IcpCorrespondenceStatsWorkspace`, with test accessors mirroring the
    existing spatial-grid workspace buffers.
  - Added `gatherSortedIcpTargetPointsKernel`, launched after `thrust::sort_by_key`, to materialize sorted target
    coordinates beside the sorted target index array.
  - Extended `IcpTargetSpatialGrid` with sorted x/y/z pointers.
  - Switched the correspondence, residual, and transform+residual spatial-grid candidate loops to read coordinates
    from the sorted arrays. The correspondence path still reads the sorted target index for output and tie-breaking;
    residual-only paths no longer read target indices.
  - Extended spatial-grid/workspace reuse tests to require the sorted coordinate buffers to be allocated and reused.
- Tradeoff: target-grid rebuilds now pay one extra gather kernel and store three extra `double[target_count]` arrays.
  This is worthwhile for cached-grid ICP iterations, but the new-workspace microbenchmark can move slightly depending
  on the gather cost.
- `git diff --check` after the sorted target coordinate cache:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsSpatialGridSkipsNonFiniteTargetInSaturatedCell:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls`:
  5 targeted spatial-grid/workspace tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  44 targeted ICP GPU/spatial-grid/validation tests passed.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  221 tests, 0 failed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.270874`,
  `gpu_icp_finite_radius,100000,5,1.00409`,
  `gpu_icp_finite_radius_translation,100000,5,2.88503`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.25452`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.214`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.80534`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,5,1.50741`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,1.00432`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,0.97373`.
  Compared with Task 47 on this machine, cached-grid stats+step improved from 1.06889 ms to 1.00432 ms, stats-only
  cached-grid improved from 1.0321 ms to 0.97373 ms, and the full alignment reuse rows also improved. The
  new-workspace row moved from 1.48977 ms to 1.50741 ms, consistent with the added grid-build gather kernel.

## Task 49: Delay Spatial-Grid Target Index Loads Until A Candidate Can Win

- Goal: reduce per-candidate memory traffic in the finite-radius spatial-grid correspondence kernel. After Task 48,
  residual-only kernels no longer read target indices, but the correspondence stats kernel still loaded
  `sorted_target_indices` before distance evaluation for every visited candidate.
- Implementation:
  - Moved the `sorted_target_indices[sorted_offset]` read until after coordinate axis pruning, squared-distance
    computation, and the `dist_sq <= best_dist_sq` check.
  - Preserved lower-index tie-breaking by loading the target index before accepting equal-distance candidates.
  - Added a testing-only `g_icp_target_index_load_count` counter and
    `CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates`, which visits two candidates but
    loads the target index only once.
- Rejected experiment:
  - I tried storing sorted target coordinates as `Scalar` instead of `double` to reduce float-grid memory traffic.
    The benchmark regressed (`gpu_icp_stats_step_finite_radius_translation_cached_grid` around 1.078 ms), likely
    because it reintroduced float-to-double conversion into the hot candidate loop. That change was reverted before
    this task was committed.
- `git diff --check` after the delayed target-index load:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.CorrespondenceStatsStopsSpatialGridAfterExactMatchWhenIndicesOmitted`:
  4 targeted spatial-grid/tie tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  45 targeted ICP GPU/spatial-grid/validation tests passed.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  222 tests, 0 failed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.272014`,
  `gpu_icp_finite_radius,100000,5,1.00019`,
  `gpu_icp_finite_radius_translation,100000,5,2.83384`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.20899`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.16538`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.74765`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,5,1.47031`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,0.970505`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,0.933812`.
  Compared with Task 48 on this machine, cached-grid stats+step improved from 1.00432 ms to 0.970505 ms, stats-only
  cached-grid improved from 0.97373 ms to 0.933812 ms, and the alignment reuse rows improved again.

## Task 50: Use Fast Finite Cell-Bound Distance In Spatial Grid

- Goal: reduce fixed per-cell pruning cost in the finite-radius spatial-grid ICP hot path. The previous
  `distanceOutsideIcpGridCellAxis` guarded every cell-bound computation with `isfinite(cell_min/cell_max)`, which is
  necessary only for extreme correspondence radii.
- Implementation:
  - Added `icpGridCellBoundsAreFinite(cell_size)` on the host side and stored the result in `IcpTargetSpatialGrid`.
  - Added finite-bound distance helpers that skip repeated `isfinite` checks when all possible int cell bounds are
    representable for the current radius.
  - Routed correspondence, residual, and transform+residual spatial-grid pruning through the fast helpers for normal
    finite radii, while retaining the old guarded path for extreme radii.
  - Kept target-grid cache semantics unchanged; the fast/safe flag is derived from the current cell size when the
    grid wrapper is prepared.
- `git diff --check` after the finite cell-bound fast path:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridXYLookupsBeforeSearch:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPGpuPathTest.CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates:ICPTest.CollectCorrespondencesKeepsFloatDistancesThatOverflowScalarDifference:ICPTest.AlignReportsUnrepresentableFloatStateInsteadOfProducingNonFiniteTransform:ICPTest.NonCollinearGeometryHandlesNearDoubleMaxFiniteScale`:
  7 targeted spatial-grid/extreme-value tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  45 targeted ICP GPU/spatial-grid/validation tests passed.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  222 tests, 0 failed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.272964`,
  `gpu_icp_finite_radius,100000,5,1.0101`,
  `gpu_icp_finite_radius_translation,100000,5,2.78807`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.16523`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.12461`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.72318`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,5,1.4626`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,0.94603`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,0.91683`.
  Compared with Task 49 on this machine, cached-grid stats+step improved from 0.970505 ms to 0.94603 ms, stats-only
  cached-grid improved from 0.933812 ms to 0.91683 ms, and alignment reuse rows improved from 2.20899 ms to
  2.16523 ms.

## Task 51: Defer Spatial-Grid Best Target Index Materialization

- Goal: reduce target-index memory traffic in the no-output finite-radius spatial-grid correspondence stats path.
  After Task 49, candidate target indices were loaded only for candidates that could match the current best distance,
  but the first/better candidate still loaded an index even when `correspondence_indices == nullptr`.
- Implementation:
  - Changed the spatial-grid correspondence kernel to track the best sorted-grid offset separately from the materialized
    target index.
  - Deferred `sorted_target_indices` reads until equal-distance tie-breaking or explicit correspondence-index output
    requires the index.
  - Preserved deterministic lower-target-index tie-breaking by lazily loading both the challenger index and the current
    best index when an equal-distance tie occurs.
  - Updated the target-index load counter test to expect zero index loads for an ordinary no-output nearest-neighbor
    stats call, and extended the tie test to cover the no-output path.
- Rejected experiment:
  - I tried dynamic X/Y neighbor-cell ordering based on the source position inside its grid cell. It reduced some align
    rows but regressed cached stats rows on repeat benchmark runs, for example
    `gpu_icp_stats_step_finite_radius_translation_cached_grid` around 0.954 ms versus Task 50's 0.94603 ms and
    `gpu_icp_stats_finite_radius_translation_cached_grid` around 0.917 ms versus Task 50's 0.91683 ms. That change was
    reverted before this task was kept.
- `git diff --check` after the deferred best-index materialization:
  clean.
- `cmake --build build-codex-cuda -j$(nproc)` and
  `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPGpuPathTest.CorrespondenceStatsStopsSpatialGridAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance`:
  4 targeted spatial-grid/tie tests passed.
- `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.*:ICPTest.GpuRejectsNonFiniteSourcePointsBeforeAlignment:ICPTest.RejectsCollinearCorrespondenceGeometry:ICPValidation.RecoversKnownTransform`:
  45 targeted ICP GPU/spatial-grid/validation tests passed.
- `ctest --test-dir build-codex-cuda --output-on-failure`:
  222 tests, 0 failed.
- `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
  143 tests, 0 failed, 1 skipped CUDA-only transfer case.
- `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
  `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
  rows included
  `gpu_icp_identity,100000,5,0.273054`,
  `gpu_icp_finite_radius,100000,5,1.00459`,
  `gpu_icp_finite_radius_translation,100000,5,2.7777`,
  `gpu_icp_finite_radius_translation_reuse,100000,5,2.14185`,
  `gpu_icp_finite_radius_translation_reuse_output,100000,5,2.09804`,
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics,100000,5,1.69878`,
  `gpu_icp_stats_step_finite_radius_translation_new_workspace,100000,5,1.44623`,
  `gpu_icp_stats_step_finite_radius_translation_cached_grid,100000,5,0.937737`, and
  `gpu_icp_stats_finite_radius_translation_cached_grid,100000,5,0.901087`.
  Compared with Task 50 on this machine, cached-grid stats+step improved from 0.94603 ms to 0.937737 ms,
  stats-only cached-grid improved from 0.91683 ms to 0.901087 ms, alignment reuse improved from 2.16523 ms to
  2.14185 ms, and the skip-final-metrics reuse row improved from 1.72318 ms to 1.69878 ms.

## Task 52: Delay Spatial-Grid Y/Z Coordinate Loads After X-Axis Pruning

- Goal: reduce global-memory traffic in finite-radius spatial-grid candidate loops. The hot correspondence, residual,
  and transform+residual spatial-grid kernels loaded sorted target x/y/z coordinates before applying the x-axis radius
  prune, so candidates rejected by x still paid for y/z loads.
- Implementation:
  - Added inline sorted-target coordinate load helpers with a testing-only coordinate-load counter.
  - Changed all three spatial-grid candidate loops to load sorted x first, prune by x, then load y and z only if each
    earlier axis passes.
  - Added `SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning`, which constructs a visited candidate outside the
    x radius and expects one candidate visit, zero full-distance evaluations, and one sorted-coordinate load for both
    correspondence stats and residual stats paths.
- Verification performed in this session:
  - Recreated the temporary PlaMatrix install prefixes used by PlaPoint builds:
    `/tmp/plamatrix-cuda-install-plapoint` and `/tmp/plamatrix-install-smoke`.
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning:ICPGpuPathTest.CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates:ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    5 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    223 test entries, 0 failed; GPU-dependent tests skipped because `nvidia-smi` cannot communicate with the driver in
    the current session.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark before using Task 52 as confirmed
  performance evidence.

## Task 53: Prune Spatial-Grid Candidate Coordinate Loads By Partial Distance

- Goal: continue reducing sorted-coordinate memory traffic in finite-radius spatial-grid candidate loops. Once a best
  distance exists, a candidate whose partial lower bound already exceeds that best distance cannot win, so later
  coordinate loads can be skipped.
- Implementation:
  - In the correspondence spatial-grid kernel, after the x coordinate is loaded and radius-pruned, skip y/z loads when
    `dx * dx > best_dist_sq`; after y is loaded, skip z when `dx * dx + dy * dy > best_dist_sq`.
  - In the residual and transform+residual spatial-grid kernels, use the stricter `>= best_dist_sq` form because those
    kernels only need strict residual improvement and do not perform target-index tie-breaking.
  - Added `SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest`, which expects two candidate visits but only five
    sorted-coordinate loads: x/y/z for the first best candidate and x/y only for the second candidate rejected by the
    partial XY lower bound.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    224 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 70: Add Exact Pointwise Residual Stats Fast Path

- Goal: avoid finite-radius spatial-grid construction and candidate scans when GPU ICP residual metrics are computed for
  a source and target that are already known to be pointwise identical. This targets terminal/final metric paths where
  the residual-only computation can prove zero residuals with one linear pass.
- Implementation:
  - Added `collectExactPointwiseResidualStatsKernel()` for residual-only exact pointwise statistics.
  - Added `launchExactPointwiseResidualStats()` and `tryComputeExactPointwiseResidualStats()` using the existing exact
    pointwise predicate: same device buffer always qualifies, and separate equal-sized buffers qualify only for
    non-finite correspondence radius and fall back on the first raw value mismatch.
  - Routed `computeIcpResidualStatsColumnMajorImpl()` through the new fast path before preparing the target spatial
    grid.
  - Added `ResidualStatsUsesExactPointwiseFastPathForSameBuffer`, which checks that identical-buffer residual stats
    produce all active finite points, zero residual sum, and no full-distance/candidate/grid lookup work.
  - Kept fallback behavior for mismatched values and NaNs by marking the fast-path reduction with infinite residual
    sum, then running the existing residual search path.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
    226 test entries, 0 failed; GPU-dependent tests, including the new residual fast-path test, were discovered but
    skipped because the current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    benchmark binary built.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`, while CPU smoke rows still emitted timing
    data.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU test
  `ICPGpuPathTest.ResidualStatsUsesExactPointwiseFastPathForSameBuffer`, full CUDA `ctest`, and the 100k-point ICP
  benchmark to confirm runtime behavior and performance impact.

## Task 71: Share GPU ICP Transform Coefficient Loads Per Block

- Goal: reduce repeated per-thread transform coefficient loads and duplicated arithmetic expressions in GPU ICP point
  transform paths. These paths are used both by the standalone point transform API and by terminal transform+residual
  metric kernels.
- Implementation:
  - Added `transformColumnMajorPoint3x4()` as a `__forceinline__` device helper.
  - Added a block-level 12-value shared-memory cache for the used 3x4 transform coefficients, populated with
    `loadReadOnlyIcpValue()`.
  - Reused the helper in `transformPointsColumnMajorKernel()`, `transformAndCollectResidualStatsKernel()`, and
    `transformAndCollectResidualStatsSpatialGridKernel()`.
  - Kept source-point reads and output writes unchanged so existing caller-owned/in-place output behavior is not
    tightened by new aliasing assumptions.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.TransformPointsColumnMajorWritesCallerOwnedOutput:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.ResidualStatsUsesExactPointwiseFastPathForSameBuffer:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
    226 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU transform/residual tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm
  runtime behavior and performance impact.

## Task 72: Stop Non-Spatial Residual Scans After Exact Matches

- Goal: reduce residual-only nearest-neighbor work when the non-spatial-grid residual kernels find an exact target match.
  Residual metrics do not need correspondence index tie-breaking, so once a zero squared distance is found for a source
  point no later target can improve that source's residual.
- Implementation:
  - Added a per-source `stop_target_scan` flag to `collectResidualStatsKernel()`.
  - Added the same early-scan skip to `transformAndCollectResidualStatsKernel()`.
  - Kept all threads participating in the target-tile load loop and `__syncthreads()` calls to avoid divergent block
    synchronization; only later per-source candidate evaluation is skipped.
  - Counted non-spatial residual full-distance evaluations under `PLAPOINT_ENABLE_TESTING`.
  - Added `ResidualStatsStopsNonSpatialScanAfterExactMatch`, which forces the non-spatial residual path with infinite
    correspondence radius and unequal source/target counts, then expects only one full-distance evaluation after the
    first exact match.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPGpuPathTest.ResidualStatsUsesExactPointwiseFastPathForSameBuffer:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
    227 test entries, 0 failed; GPU-dependent tests, including the new non-spatial residual fast-path test, were
    skipped because the current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch`, full CUDA `ctest`, and the 100k-point ICP
  benchmark to confirm runtime behavior and performance impact.

## Task 75: Reuse Squared Radius In GPU ICP Fallback Kernels

- Goal: remove repeated per-thread radius squaring from the shared-memory target-tiling fallback path. This fallback
  remains active for infinite-radius ICP and for finite-radius cases where the target spatial grid is not usable.
- Implementation:
  - Added a local `max_dist_sq` in `collectCorrespondenceStatsKernel()`.
  - Added a local `max_dist_sq` in `collectResidualStatsKernel()`.
  - Added a local `max_dist_sq` in `transformAndCollectResidualStatsKernel()`.
  - Reused that value for final acceptance checks instead of recomputing `max_dist * max_dist`.
  - Kept axis pruning, finite-radius checks, exact-match early stop behavior, and max-distance semantics unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `rg -n "max_dist \* max_dist|const double max_dist_sq" src/icp_gpu.cu`:
    remaining occurrences are the six local `max_dist_sq` initializations in the GPU ICP correspondence/residual kernels.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsFindsNearestTargetsPastFirstTile:ICPGpuPathTest.CorrespondenceStatsPrunesFarTargetsBeforeFullDistanceEvaluation:ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    227 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected fallback GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior
  and performance impact.

## Task 76: Specialize GPU ICP Fallback Correspondence Index Output

- Goal: remove runtime correspondence-index output branches from the non-spatial shared-memory target-tiling stats
  kernel, and let callers that omit index output stop scanning a source point once an exact target match has been found.
  This mirrors the existing spatial-grid correspondence specialization while preserving requested-index behavior.
- Implementation:
  - Templated `collectCorrespondenceStatsKernel()` on `WriteCorrespondenceIndices`.
  - Routed invalid-source, rejected-correspondence, and accepted-correspondence index writes through `if constexpr`.
  - Added `launchCollectCorrespondenceStatsKernel()` to select indexed or unindexed kernel instances at launch time.
  - Added exact-distance early stop to the unindexed fallback path only; indexed callers still scan all candidates needed
    to preserve requested output and tie behavior.
  - Updated `CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop` to request index output so it continues to verify
    tile-level pruning rather than the new omitted-index exact-stop shortcut.
  - Added `CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted` to cover the new fallback behavior
    and to confirm indexed callers still visit later candidates.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `rg -n "collectCorrespondenceStatsKernel<Scalar>" src/icp_gpu.cu`:
    no old unspecialized fallback launches remain.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 5 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    228 test entries, 0 failed; GPU-dependent tests, including the new exact-stop test, were skipped because the
    current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `ICPGpuPathTest.CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted`, full CUDA `ctest`,
  and the 100k-point ICP benchmark to confirm runtime behavior and performance impact.

## Task 77: Specialize GPU ICP Fallback Tile-Bounds Branches

- Goal: remove the remaining runtime tile-bounds and radius-pruning branches from the shared-memory target-tiling
  fallback kernels. The fallback path is now mainly used for zero-radius tile-bounds pruning and infinite-radius
  unbounded scans, so the branch choice is known at launch time.
- TDD red:
  - Added `FallbackStatsLaunchesTileBoundSpecializationsByRadius`, which references new test-only fallback launch
    counters and requires zero-radius correspondence and transform+residual calls to use the tile-bounds variant while
    infinite-radius residual calls use the unbounded variant.
  - `cmake --build build-codex-cuda -j$(nproc)` failed as expected at link time because the new test hooks were not yet
    defined.
- Implementation:
  - Templated `collectCorrespondenceStatsKernel()` on `UseTargetTileBounds` in addition to index-output specialization.
  - Templated `collectResidualStatsKernel()` and `transformAndCollectResidualStatsKernel()` on `UseTargetTileBounds`.
  - Replaced runtime `target_tile_bounds` / `can_prune_by_radius` checks in those fallback kernels with `if constexpr`.
  - Added `launchCollectResidualStatsKernel()` and `launchTransformAndCollectResidualStatsKernel()` helpers matching the
    existing correspondence fallback launch helper.
  - Added test-only fallback launch counters to verify bounded vs unbounded fallback selection.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `rg -n "can_prune_by_radius|collectResidualStatsKernel<Scalar>|transformAndCollectResidualStatsKernel<Scalar>|collectCorrespondenceStatsKernel<Scalar>" src/icp_gpu.cu`:
    no remaining runtime prune flag or old unspecialized fallback launches.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPGpuPathTest.CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    229 test entries, 0 failed; GPU-dependent tests, including the new fallback-specialization test, were skipped
    because the current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius`, full CUDA `ctest`, and the 100k-point
  ICP benchmark to confirm runtime behavior and performance impact.

## Task 78: Stop Fallback Target-Tile Loads After Block Exact Matches

- Goal: avoid loading remaining target tiles in the shared-memory fallback kernels once every source thread in a block
  has either become inactive or found an exact zero-distance match. Previous per-thread exact-stop logic avoided later
  candidate evaluation but still forced all threads through the remaining target-tile loads to preserve synchronization.
- TDD red:
  - Added `FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched`, which expects correspondence, residual-only, and
    transform+residual fallback calls to load one target tile instead of all three when a block exact-matches in the
    first tile.
  - `cmake --build build-codex-cuda -j$(nproc)` failed as expected at link time because the new target-tile load
    counter hooks were not yet defined.
- Implementation:
  - Added a CUDA-only target-tile load counter for the shared-memory fallback kernels.
  - Counted one target-tile load per block per fallback tile in `collectCorrespondenceStatsKernel()`,
    `collectResidualStatsKernel()`, and `transformAndCollectResidualStatsKernel()`.
  - Replaced the end-of-tile `__syncthreads()` with `__syncthreads_count(source_valid && !stop_target_scan)`.
  - Broke out of the tile loop only when the synchronized unfinished-source count is zero, so all threads leave the loop
    together and the shared-memory tile synchronization remains uniform.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched:ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    230 test entries, 0 failed; GPU-dependent tests, including the new target-tile-load early-exit test, were skipped
    because the current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `ICPGpuPathTest.FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched`, full CUDA `ctest`, and the
  100k-point ICP benchmark to confirm runtime behavior and performance impact.

## Task 79: Skip Bounded Fallback Target-Tile Loads Before Shared-Memory Fill

- Goal: avoid loading target tiles into shared memory when precomputed target tile bounds prove that no unfinished
  source thread in the block can accept any point from that tile. Previous bounded fallback pruning skipped the
  per-source candidate loop for irrelevant far tiles, but it still loaded each target tile before making that decision.
- TDD coverage:
  - Extended `CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop` to require the indexed zero-radius fallback
    case to load only the first relevant target tile instead of all three target tiles.
  - Added `FallbackStatsSkipTargetTileLoadsWhenBoundsRejectWholeBlock`, which covers correspondence with index output,
    residual-only stats, and transform+residual stats when every target tile is outside the zero-radius bounds for the
    source block.
  - Runtime RED could not be observed in this session because the current machine has no usable CUDA device. The tests
    were added before production changes; on the previous implementation the target-tile load counter would increment
    once per fallback tile after the shared-memory fill.
- Implementation:
  - Moved the `UseTargetTileBounds` relevance check before target coordinate loads in
    `collectCorrespondenceStatsKernel()`, `collectResidualStatsKernel()`, and
    `transformAndCollectResidualStatsKernel()`.
  - Used a block-uniform `__syncthreads_count(tile_relevant)` vote so all threads either skip the tile together or
    proceed to shared-memory target loading together.
  - Kept the existing end-of-loaded-tile unfinished-source vote, exact-match early exit, indexed correspondence output,
    and unbounded fallback path unchanged.
- Verification performed in this session:
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `git diff --check`:
    clean.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPGpuPathTest.FallbackStatsSkipTargetTileLoadsWhenBoundsRejectWholeBlock:ICPGpuPathTest.FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched:ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 6 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    231 test entries, 0 failed; GPU-dependent tests, including the new bounded fallback target-tile load test, were
    skipped because the current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `ICPGpuPathTest.FallbackStatsSkipTargetTileLoadsWhenBoundsRejectWholeBlock`, full CUDA `ctest`, and the
  100k-point ICP benchmark to confirm runtime behavior and performance impact.

## Task 80: Skip Fallback Target-Tile Loop For Blocks With No Valid Sources

- Goal: avoid entering the shared-memory fallback target-tile loop when every source thread in a block is invalid.
  Before this change, unbounded fallback kernels could still load the first target tile before the synchronized
  unfinished-source vote discovered that no thread needed target scanning.
- TDD coverage:
  - Added `FallbackStatsSkipUnboundedTargetTileLoadsWhenBlockHasNoValidSources`, which feeds a NaN source point through
    correspondence with index output, residual-only stats, and transform+residual stats on the unbounded fallback path.
  - The test expects zero active correspondences, one invalid source, correspondence index `-1`, and zero target-tile
    loads for all three fallback kernels.
  - Runtime RED could not be observed in this session because the current machine has no usable CUDA device. The test
    was added before production changes; on the previous unbounded fallback behavior the target-tile load counter would
    increment once before the end-of-tile unfinished-source vote broke out of the loop.
- Implementation:
  - Added a block-uniform `__syncthreads_count(source_valid)` precheck in `collectCorrespondenceStatsKernel()`,
    `collectResidualStatsKernel()`, and `transformAndCollectResidualStatsKernel()`.
  - Used the precheck result to set `scan_target_count` to either the original target count or zero, keeping the existing
    target-tile loop body and synchronization structure unchanged.
  - Preserved invalid-source accounting, requested correspondence index output, bounded fallback tile pruning, and the
    unbounded normal-source scan path.
- Verification performed in this session:
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `git diff --check`:
    clean.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.FallbackStatsSkipUnboundedTargetTileLoadsWhenBlockHasNoValidSources:ICPGpuPathTest.FallbackStatsSkipTargetTileLoadsWhenBoundsRejectWholeBlock:ICPGpuPathTest.FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched:ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 6 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    232 test entries, 0 failed; GPU-dependent tests, including the new no-valid-source fallback test, were skipped
    because the current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `ICPGpuPathTest.FallbackStatsSkipUnboundedTargetTileLoadsWhenBlockHasNoValidSources`, full CUDA `ctest`, and the
  100k-point ICP benchmark to confirm runtime behavior and performance impact.

## Task 81: Skip Exact Pointwise Target Loads For Same Buffers

- Goal: reduce global-memory traffic in the exact pointwise identity path when source and target are the same GPU
  buffer. The previous exact pointwise kernels still read source and target columns separately even when the pointers
  were identical, duplicating three target-coordinate loads per point before recording zero-residual correspondences.
- TDD red:
  - Added CUDA-only target-load counter declarations and same-buffer assertions to
    `ResidualStatsUsesExactPointwiseFastPathForSameBuffer`.
  - Added `CorrespondenceStatsSameBufferExactPointwiseAvoidsTargetCoordinateLoads`, which first proves separate but
    equal buffers still count three target loads per point, then requires the same-buffer exact correspondence path to
    count zero target loads.
  - `cmake --build build-codex-cuda -j$(nproc)` failed as expected at link time because
    `resetIcpExactPointwiseTargetLoadCountForTesting()` and `icpExactPointwiseTargetLoadCountForTesting()` did not
    exist yet.
- Implementation:
  - Added a CUDA-only exact pointwise target-load counter.
  - Templated `collectExactPointwiseCorrespondenceStatsKernel()` and `collectExactPointwiseResidualStatsKernel()` on
    `SameBuffer`.
  - The `SameBuffer=true` kernel instances reuse source values for target values and skip target coordinate loads and
    equality comparison loads; the `SameBuffer=false` instances keep the previous separate-buffer comparison behavior.
  - Updated exact pointwise stats, stats+identity-step, alignment-step, and residual launch helpers to select the
    same-buffer kernel instance when `d_source_points == d_target_points`.
- Verification performed in this session:
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed after implementing the hooks and same-buffer kernel instances.
  - `git diff --check`:
    clean.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSameBufferExactPointwiseAvoidsTargetCoordinateLoads:ICPGpuPathTest.ResidualStatsUsesExactPointwiseFastPathForSameBuffer:ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 6 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests, including the new exact pointwise target-load test, were skipped
    because the current session cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `ICPGpuPathTest.CorrespondenceStatsSameBufferExactPointwiseAvoidsTargetCoordinateLoads`, full CUDA `ctest`, and
  the 100k-point ICP benchmark to confirm runtime behavior and performance impact.

## Task 82: Share Exact Pointwise Same-Buffer Launch Selection

- Goal: reduce maintenance risk in the GPU ICP exact-pointwise fast path after Task 81. The same `SameBuffer` kernel
  selection was duplicated across the stats-only, stats+identity-step, alignment-step, and residual-stats launch
  helpers, so later exact-pointwise kernel changes would require touching multiple equivalent branches.
- Refactor baseline:
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSameBufferExactPointwiseAvoidsTargetCoordinateLoads:ICPGpuPathTest.ResidualStatsUsesExactPointwiseFastPathForSameBuffer:ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
- Implementation:
  - Added `launchExactPointwiseCorrespondencePartials()` to centralize correspondence partial-stat kernel launches and
    choose `SameBuffer=true` only when `d_source_points == d_target_points`.
  - Added `launchExactPointwiseResidualPartials()` to centralize residual partial-stat kernel launches with the same
    same-buffer selection rule.
  - Updated `launchExactPointwiseStats()`, `launchExactPointwiseStatsAndIdentityStep()`,
    `launchExactPointwiseAlignmentStep()`, and `launchExactPointwiseResidualStats()` to call the shared helpers.
  - Kept all exact-pointwise gating, CUDA error checks, testing counters, reduction kernels, and public behavior
    unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsSameBufferExactPointwiseAvoidsTargetCoordinateLoads:ICPGpuPathTest.ResidualStatsUsesExactPointwiseFastPathForSameBuffer:ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `rg -n "collectExactPointwiseCorrespondenceStatsKernel<Scalar|collectExactPointwiseResidualStatsKernel<Scalar|launchExactPointwiseCorrespondencePartials|launchExactPointwiseResidualPartials" src/icp_gpu.cu`:
    direct exact-pointwise kernel launches are centralized in the two partial-launch helpers, with call sites using the
    shared helpers.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.

## Task 83: Use Read-Only Source Loads In GPU ICP Transform Residual Paths

- Goal: reduce global-memory pressure in GPU ICP transform and transform+residual paths. These kernels read source
  point columns as immutable input, but still used normal global loads while nearby ICP kernels already route immutable
  point loads through `loadReadOnlyIcpValue()`.
- Test and instrumentation coverage:
  - Extended the spatial-grid candidate pruning tests so residual-only stats also assert target-candidate visits and
    full-distance evaluation counts, matching the correspondence-path pruning expectations.
  - The current machine has no usable CUDA device, so these GPU assertions are compiled and discovered but skipped at
    runtime until a CUDA-capable runner is available.
- Implementation:
  - Changed `transformAndCollectResidualStatsKernel()`,
    `transformAndCollectResidualStatsSpatialGridKernel()`, and `transformPointsColumnMajorKernel()` to read source point
    coordinates through `loadReadOnlyIcpValue()`.
  - Added testing-only target-candidate visit counters to residual fallback, transform+residual fallback, residual
    spatial-grid, and transform+residual spatial-grid candidate loops.
  - Added testing-only full-distance evaluation counters to residual spatial-grid and transform+residual spatial-grid
    loops after radius pruning, so future GPU runs can catch pruning regressions on residual paths.
  - Kept output writes, alias-handling assumptions, residual acceptance, tie-breaking, and public APIs unchanged.
- Verification performed in this session:
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.TransformPointsColumnMajorWritesCallerOwnedOutput:ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched:ICPGpuPathTest.SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 7 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `git diff --check`:
    clean.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected spatial-grid/fallback GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to verify
  runtime behavior and measure the read-only load impact.

## Task 84: Use Read-Only Loads For GPU ICP Target Tile Bounds

- Goal: reduce global-memory pressure in the bounded shared-memory fallback path. The target tile bounds are immutable
  metadata during fallback scans, but the correspondence, residual, and transform+residual fallback kernels loaded the
  whole `IcpTargetTileBounds` struct directly from global memory.
- Static RED:
  - `test "$(rg -n "const IcpTargetTileBounds bounds = target_tile_bounds\\[tile_idx\\]" src/icp_gpu.cu | wc -l)" -eq 0`:
    failed before the change because the three bounded fallback kernels still used direct struct reads.
- Implementation:
  - Added `loadIcpTargetTileBounds()` to read each tile-bound field through `loadReadOnlyIcpValue()`.
  - Replaced direct `target_tile_bounds[tile_idx]` reads in `collectCorrespondenceStatsKernel()`,
    `collectResidualStatsKernel()`, and `transformAndCollectResidualStatsKernel()`.
  - Kept bounded/unbounded specialization selection, tile relevance tests, tile-load skipping, and output behavior
    unchanged.
- Verification performed in this session:
  - `test "$(rg -n "const IcpTargetTileBounds bounds = target_tile_bounds\\[tile_idx\\]" src/icp_gpu.cu | wc -l)" -eq 0`:
    passed after the helper replacement.
  - `rg -n "loadIcpTargetTileBounds|target_tile_bounds\\[tile_idx\\]|IcpTargetTileBounds bounds" src/icp_gpu.cu`:
    showed the helper and the three helper call sites, with no remaining direct `target_tile_bounds[tile_idx]` reads.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched:ICPGpuPathTest.FallbackStatsSkipTargetTileLoadsWhenBoundsRejectWholeBlock:ICPGpuPathTest.FallbackStatsSkipUnboundedTargetTileLoadsWhenBlockHasNoValidSources:ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 5 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `git diff --check`:
    clean.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected fallback GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior
  and measure the metadata load impact.

## Task 85: Reuse Squared Distances In GPU ICP Fallback Bounds Checks

- Goal: reduce arithmetic in the bounded shared-memory fallback candidate loops. The fallback kernels used three
  `fabs(axis_delta) > max_dist` checks and then recomputed `dx * dx + dy * dy + dz * dz` for the same candidate.
- Static RED:
  - `test "$(rg -n "fabs\\(d[xyz]\\) > max_dist" src/icp_gpu.cu | wc -l)" -eq 0`:
    failed before the change because the correspondence, residual, and transform+residual fallback kernels still used
    nine per-axis `fabs` checks.
- Implementation:
  - Changed the correspondence fallback loop to compute `dx_sq`, `dy_sq`, `xy_dist_sq`, `dz_sq`, and `dist_sq` once and
    use those values for bounded pruning and final distance comparison.
  - Applied the same squared-distance reuse to residual-only and transform+residual fallback loops.
  - Kept candidate visit counters, full-distance counters, zero-distance early stop, accepted correspondence output, and
    bounded/unbounded kernel specialization behavior unchanged.
- Verification performed in this session:
  - `test "$(rg -n "fabs\\(d[xyz]\\) > max_dist" src/icp_gpu.cu | wc -l)" -eq 0`:
    passed after replacing the fallback checks.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesFarTargetsBeforeFullDistanceEvaluation:ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.FallbackStatsStopsLoadingTargetTilesWhenBlockExactMatched:ICPGpuPathTest.FallbackStatsSkipTargetTileLoadsWhenBoundsRejectWholeBlock:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 6 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected fallback GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior
  and measure the bounded fallback arithmetic impact.

## Task 86: Reuse Fallback Best Target Coordinates

- Goal: reduce shared-memory reloads in the GPU ICP correspondence fallback scan. The candidate loop already reads
  `target_tile_x/y/z[tile_offset]` to compute `dx/dy/dz`, then reloaded the same three coordinates when a candidate
  became the current best target.
- Static RED:
  - `test "$(rg -n "best_t[xyz] = target_tile_[xyz]\\[tile_offset\\]" src/icp_gpu.cu | wc -l)" -eq 0`:
    failed before the change because the correspondence fallback best update still read `target_tile_x/y/z` directly.
- Implementation:
  - Loaded fallback candidate coordinates into local `tx`, `ty`, and `tz` before computing `dx`, `dy`, and `dz`.
  - Reused those locals for `best_tx`, `best_ty`, and `best_tz` on best-candidate updates.
  - Kept best-distance comparison, best index output, zero-distance early stop, and bounded/unbounded specialization
    behavior unchanged.
- Verification performed in this session:
  - `test "$(rg -n "best_t[xyz] = target_tile_[xyz]\\[tile_offset\\]" src/icp_gpu.cu | wc -l)" -eq 0`:
    passed after replacing direct best-coordinate reloads.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsFindsNearestTargetsPastFirstTile:ICPGpuPathTest.CorrespondenceStatsPrunesFarTargetsBeforeFullDistanceEvaluation:ICPGpuPathTest.CorrespondenceStatsSkipsFarTargetTilesBeforeCandidateLoop:ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStopsNonSpatialScanAfterExactMatchWhenIndicesOmitted:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 6 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected correspondence fallback GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm
  runtime behavior and measure the shared-memory reload impact.

## Task 87: Parallelize Exact Identity Alignment-Step Transform Write

- Goal: reduce work done by thread 0 in the exact pointwise GPU ICP alignment-step path. The normal exact pointwise step
  kernel already writes the 4x4 identity transform with 16 threads, but the compact alignment-step variant still called
  a helper that wrote all 16 identity values serially from thread 0.
- Static RED:
  - `test "$(rg -n "writeAlignmentStepRawResultFromRawStats<Scalar>\\(shared_stats\\[0\\], step_transform, result, true\\)|bool exact_identity_step|if \\(exact_identity_step\\)" src/icp_gpu.cu | wc -l)" -eq 0`:
    failed before the change because the exact identity alignment-step kernel still used the helper's
    `exact_identity_step` branch.
- Implementation:
  - Changed `reduceRawIcpStatsAndSetExactPointwiseIdentityAlignmentStepKernel()` to write identity transform values with
    `local_idx < 16`, matching the existing exact pointwise step kernel.
  - Added `writeAlignmentStepRawResultFields()` so exact identity and computed-step alignment paths share result-field
    writing.
  - Removed the `exact_identity_step` branch from `writeAlignmentStepRawResultFromRawStats()`; that helper now only
    computes the non-identity step transform and writes shared result fields.
  - Kept active-count, invalid-source count, non-collinearity flags, step-valid semantics, residual sum, and delta
    values unchanged.
- Verification performed in this session:
  - `test "$(rg -n "writeAlignmentStepRawResultFromRawStats<Scalar>\\(shared_stats\\[0\\], step_transform, result, true\\)|bool exact_identity_step|if \\(exact_identity_step\\)" src/icp_gpu.cu | wc -l)" -eq 0`:
    passed after removing the exact-identity helper branch.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.AlignReusesIterationStatsForExactIdentityTerminalMetrics:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.AlignmentStepCompactResultMatchesFullStatsStepResult:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 5 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected exact pointwise alignment GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm
  runtime behavior and measure the identity-step impact.

## Task 88: Unroll GPU ICP 4x4 Eigen Solver Small Loops

- Goal: reduce fixed-loop overhead inside the device-side 4x4 Jacobi eigenvector helper used when GPU ICP computes a
  step transform from reduced correspondence statistics.
- Static RED:
  - `test "$(sed -n '/__device__ void jacobiRotate4x4/,/__device__ void largestEigenvectorSymmetric4x4/p' src/icp_gpu.cu | rg -n '#pragma unroll' | wc -l)" -ge 2`:
    failed before the change because the two fixed 4-entry loops in `jacobiRotate4x4()` were not explicitly unrolled.
  - `test "$(sed -n '/__device__ void largestEigenvectorSymmetric4x4/,/template <typename Scalar>/p' src/icp_gpu.cu | rg -n '#pragma unroll' | wc -l)" -ge 4`:
    failed before the change because the fixed initialization, identity, best-index, and output-copy loops in
    `largestEigenvectorSymmetric4x4()` were not explicitly unrolled.
- Implementation:
  - Added `#pragma unroll` to the two fixed 4-entry loops in `jacobiRotate4x4()`.
  - Added `#pragma unroll` to the fixed 16-entry copy/zero loop, the 4-entry identity loop, the 3-comparison best-index
    loop, and the 4-entry output-copy loop in `largestEigenvectorSymmetric4x4()`.
  - Kept the 32-sweep Jacobi iteration loop unchanged to avoid unnecessary code-size growth.
- Verification performed in this session:
  - Both static RED commands passed after adding the unroll hints.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.StepTransformFromStatsWritesDeviceTransform:ICPGpuPathTest.AlignmentStepCompactResultMatchesFullStatsStepResult:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPTest.Multiply4x4RejectsUnrepresentableAccumulatedTransform:ICPValidation.RecoversKnownTransform`:
    2 CPU tests passed; 4 selected GPU tests were discovered but skipped because the current session has no usable CUDA
    device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected step-transform GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime
  behavior and measure the transform-solve impact.

## Task 89: Force-Inline GPU ICP Step-Transform Helpers

- Goal: reduce device helper call overhead in the GPU ICP stats-to-step-transform path after the small fixed loops were
  unrolled. This path is used by the stats+step kernels and the compact alignment-step result writer.
- Static RED:
  - `test "$(rg -n "__device__ (void jacobiRotate4x4|void largestEigenvectorSymmetric4x4|bool rawStatsCovarianceHasNonCollinearGeometry)" src/icp_gpu.cu | wc -l)" -eq 0`:
    failed before the change because those helpers were still declared or defined as plain `__device__` functions.
  - `test "$(rg -n "__device__ (bool scalarRepresentable|Scalar checkedDeviceScalar)" src/icp_gpu.cu | wc -l)" -eq 0`:
    failed before the change because the scalar conversion helpers were still plain `__device__` helpers.
  - `test "$(rg -n "__device__ void (computeStepTransformFromInput|computeStepTransformFromRawStatsValue|writeAlignmentStepRawResultFromRawStats)" src/icp_gpu.cu | wc -l)" -eq 0`:
    failed before the change because the step-transform helper declarations and definitions were still plain
    `__device__` functions.
- Implementation:
  - Added `__forceinline__` to the forward declarations and definitions for
    `computeStepTransformFromInput()`, `computeStepTransformFromRawStatsValue()`, and
    `writeAlignmentStepRawResultFromRawStats()`.
  - Added `__forceinline__` to the definitions for `jacobiRotate4x4()`, `largestEigenvectorSymmetric4x4()`,
    `scalarRepresentable()`, `checkedDeviceScalar()`, and `rawStatsCovarianceHasNonCollinearGeometry()`.
  - Kept all transform math, covariance checks, scalar validity handling, and result fields unchanged.
- Verification performed in this session:
  - The three static RED commands passed after adding the force-inline qualifiers.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.StepTransformFromStatsWritesDeviceTransform:ICPGpuPathTest.AlignmentStepCompactResultMatchesFullStatsStepResult:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPTest.Multiply4x4RejectsUnrepresentableAccumulatedTransform:ICPValidation.RecoversKnownTransform`:
    2 CPU tests passed; 4 selected GPU tests were discovered but skipped because the current session has no usable CUDA
    device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected step-transform GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime
  behavior and measure any call-overhead/code-size tradeoff on real hardware.

## Task 90: Cache GPU ICP Transform Multiplication Inputs In Shared Memory

- Goal: reduce repeated global-memory transform loads inside `multiplyTransform4x4Kernel()`. The kernel has one thread
  per output matrix element, so the previous 4-term dot products repeatedly loaded the same A/B transform values across
  the 16 threads.
- Static RED:
  - `test "$(sed -n '/__global__ void multiplyTransform4x4Kernel/,/template <typename Scalar>/p' src/icp_gpu.cu | rg -n "__shared__ Scalar shared_[AB]\\[16\\]" | wc -l)" -eq 2`:
    failed before the change because the multiply kernel had no shared A/B transform cache.
  - `test "$(sed -n '/__global__ void multiplyTransform4x4Kernel/,/template <typename Scalar>/p' src/icp_gpu.cu | rg -n "loadReadOnlyIcpValue\\((A|B) \\+ idx\\)" | wc -l)" -eq 2`:
    failed before the change because the multiply kernel loaded A/B directly inside the dot-product loop.
- Implementation:
  - Added `shared_A[16]` and `shared_B[16]` to `multiplyTransform4x4Kernel()`.
  - Populated the shared arrays once per block with `loadReadOnlyIcpValue()` and synchronized before the existing
    per-element dot product.
  - Kept column-major indexing, double accumulation, output layout, and launch shape unchanged.
- Verification performed in this session:
  - Both static RED commands passed after adding the shared transform cache.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.MultiplyTransform4x4UsesColumnMajorTransformComposition:ICPGpuPathTest.MultiplyTransform4x4AsyncUsesCallerStream:ICPGpuPathTest.TransformPointsColumnMajorWritesCallerOwnedOutput:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPTest.Multiply4x4RejectsUnrepresentableAccumulatedTransform:ICPValidation.RecoversKnownTransform`:
    2 CPU tests passed; 4 selected GPU tests were discovered but skipped because the current session has no usable CUDA
    device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    233 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected transform-multiply GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm
  runtime behavior and measure the transform-update impact.

## Task 91: Reuse GPU ICP Target Tile Bounds Across Fallback Stats Calls

- Goal: avoid recomputing finite-radius fallback target tile bounds when repeated stats/residual calls use the same
  target point buffer and the same reusable `IcpCorrespondenceStatsWorkspace`. The tile bounds depend on target points
  and tile partitioning, not on the correspondence radius.
- Static RED:
  - `test "$(rg -n "targetTileBoundsCacheMatches|markTargetTileBoundsCache|invalidateTargetTileBoundsCache" include/plapoint/gpu/icp.h src/icp_gpu.cu | wc -l)" -ge 6`:
    failed before the change because workspace only cached target spatial grids, not target tile bounds.
- Behavioral RED:
  - Added `ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusTargetTileBoundsForSameTarget`, which calls
    `computeIcpCorrespondenceStatsColumnMajor()` twice with the same target, radius, and workspace, and expects the
    tile-bound computation counter to remain at one 257-point, 3-tile precompute.
  - On this machine the test is registered but skips at runtime because there is no usable CUDA device; on a CUDA
    machine it would have failed before the cache.
- Implementation:
  - Added target tile-bound cache metadata to `IcpCorrespondenceStatsWorkspace`, keyed by target device pointer and
    target point count.
  - Added `targetTileBoundsCacheMatches()`, `markTargetTileBoundsCache()`, and
    `invalidateTargetTileBoundsCache()`.
  - `prepareTargetTileBounds()` now returns cached bounds when they match and only launches
    `computeTargetTileBoundsKernel()` on cache miss.
  - Target tile-bound cache is invalidated when tile-bound storage grows and when ICP target metadata is invalidated.
- Verification performed in this session:
  - The static RED command passed after adding the tile-bound cache API and implementation.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusTargetTileBoundsForSameTarget:ICPGpuPathTest.CorrespondenceStatsPrecomputesFiniteRadiusTargetTileBoundsOnce:ICPGpuPathTest.FallbackStatsLaunchesTileBoundSpecializationsByRadius:ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    234 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the new tile-bound reuse GPU test, full CUDA `ctest`, and fallback finite-radius ICP benchmarks to measure the
  precompute-kernel reduction on real hardware.

## Task 92: Add GPU ICP Fallback Tile-Bounds Cache Benchmarks

- Goal: make the Task 91 target tile-bound cache measurable. The existing finite-radius GPU ICP benchmark rows use a
  positive radius and therefore exercise the spatial-grid path rather than the fallback tile-bound path.
- Static RED:
  - `test "$(rg -n "gpu_icp_stats_fallback_tile_bounds_(new_workspace|cached_bounds)" benchmarks/plapoint_benchmarks.cpp | wc -l)" -ge 4`:
    failed before the change because no benchmark rows existed for fallback tile-bound stats calls.
  - `test "$(rg -n "benchmarkGpuIcpStatsFallbackTileBounds" benchmarks/plapoint_benchmarks.cpp | wc -l)" -ge 2`:
    failed before the change because the benchmark functions did not exist.
- Implementation:
  - Added `gpu_icp_stats_fallback_tile_bounds_new_workspace`, which creates a fresh stats workspace for each measured
    call and therefore includes target tile-bound precompute cost.
  - Added `gpu_icp_stats_fallback_tile_bounds_cached_bounds`, which reuses one stats workspace so the benchmark warm-up
    builds the tile bounds and measured calls can reuse them.
  - Both rows use distinct source/target device buffers and `max_correspondence_distance = 0.0f` to force the fallback
    tile-bound path instead of same-buffer exact pointwise or positive-radius spatial-grid search.
  - The fallback benchmark point count is capped at 4096 to avoid making the fallback O(N^2) path too expensive when
    the standard smoke command passes `--icp-points 100000`.
- Verification performed in this session:
  - Both static RED commands passed after adding the benchmark rows.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; the two new rows were printed and reported `skipped,no_usable_cuda_device` on this
    machine.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    234 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the benchmark with enough iterations to compare the new-workspace and cached-bounds rows and quantify target
  tile-bound cache savings.

## Task 93: Add GPU ICP Fallback Tile-Bounds Step Benchmarks

- Goal: extend the Task 92 fallback tile-bound cache measurement from standalone correspondence stats into the fused
  stats+step helper and the compact alignment-step helper used by the GPU ICP alignment loop.
- Static RED:
  - `test "$(rg -n "gpu_icp_stats_step_fallback_tile_bounds_(new_workspace|cached_bounds)" benchmarks/plapoint_benchmarks.cpp | wc -l)" -ge 4`:
    failed before the change because no fallback tile-bound stats+step benchmark rows existed.
  - `test "$(rg -n "gpu_icp_alignment_step_fallback_tile_bounds_(new_workspace|cached_bounds)" benchmarks/plapoint_benchmarks.cpp | wc -l)" -ge 4`:
    failed before the change because no fallback tile-bound compact alignment-step benchmark rows existed.
- Implementation:
  - Added `gpu_icp_stats_step_fallback_tile_bounds_new_workspace`, which creates fresh stats and step workspaces for
    every measured fused stats+step call and therefore includes target tile-bound precompute cost.
  - Added `gpu_icp_stats_step_fallback_tile_bounds_cached_bounds`, which reuses the fused stats+step workspaces so the
    benchmark warm-up can populate the target tile-bound cache before measured calls.
  - Added `gpu_icp_alignment_step_fallback_tile_bounds_new_workspace`, which creates a fresh stats workspace for every
    compact alignment-step call.
  - Added `gpu_icp_alignment_step_fallback_tile_bounds_cached_bounds`, which reuses one stats workspace across compact
    alignment-step calls so the target tile-bound cache can be measured on the path closest to `alignGpu()`.
  - All four rows use distinct source/target device buffers with identical grid data and
    `max_correspondence_distance = 0.0f`, forcing fallback tile-bound search instead of same-buffer exact pointwise
    stats or the positive-radius spatial-grid path.
  - The fallback benchmark point count remains capped at 4096 to keep the fallback O(N^2) smoke benchmark bounded when
    the standard command passes `--icp-points 100000`.
- Verification performed in this session:
  - Both static RED commands passed after adding the benchmark rows.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    benchmark-only CUDA build succeeded.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran and printed all four new rows; each reported `skipped,no_usable_cuda_device` on this machine.
  - `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
    CUDA test build succeeded; 234 test entries, 0 failed. GPU-dependent tests skipped because the current session
    cannot communicate with the NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    CPU test build succeeded; 143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `git diff --check`:
    reported no whitespace errors.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the benchmark with enough iterations to compare the four new rows against the standalone stats fallback rows and
  decide whether the next optimization should target tile-bound precompute reuse, fallback scan occupancy, or remaining
  host synchronization in the compact alignment-step path.

## Task 94: Reserve Only GPU ICP Step Result Storage On Device-Stats Paths

- Goal: avoid allocating the host-stats step input buffer on GPU step-transform paths that read device-reduced stats
  directly and only need the small step-result buffer.
- RED:
  - Added `ICPGpuPathTest.StepTransformWorkspaceCanReserveOnlyResultStorage`.
  - `cmake --build build-codex-cuda -j$(nproc)` failed before the implementation with
    `IcpStepTransformWorkspace has no member named 'reserveResult'`.
- Implementation:
  - Added `IcpStepTransformWorkspace::reserveResult()`, which allocates only the reusable
    `IcpStepTransformRawResult` device buffer.
  - Kept `IcpStepTransformWorkspace::reserve()` as the full input+result reservation for
    `computeIcpStepTransformFromStats()`, which still copies host-side step input.
  - Changed `computeIcpStepTransformFromDeviceStatsImpl()` to call `reserveResult()` because it reads reduced
    `RawIcpStats` from `IcpCorrespondenceStatsWorkspace` and never uses `inputStorage()`.
  - Changed `computeIcpStatsAndStepTransformColumnMajorImpl()` to call `reserveResult()` because the fused stats+step
    reducer writes directly to `resultStorage()` and does not need the host-stats input buffer.
- Verification performed in this session:
  - Static check:
    `test "$(rg -n "step_workspace\\.reserveResult\\(\\)" src/icp_gpu.cu | wc -l)" -ge 2 &&
    test "$(rg -n "step_workspace\\.reserve\\(\\)" src/icp_gpu.cu | wc -l)" -eq 0` passed after the change.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    CUDA test build succeeded after adding `reserveResult()`.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.StepTransformWorkspaceCanReserveOnlyResultStorage:ICPGpuPathTest.StepTransformFromStatsWritesDeviceTransform:ICPGpuPathTest.AlignmentStepCompactResultMatchesFullStatsStepResult`:
    all 3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests and the fused stats+step fallback benchmark rows from Task 93 to confirm allocation
  reduction and measure any effect on repeated new-workspace calls.

## Task 95: Reserve Compact GPU ICP Alignment-Step Result Storage

- Goal: avoid allocating the full `RawIcpStats` result buffer on the compact alignment-step path used by
  `alignGpu()`. The compact path still needs full `RawIcpStats` partials for the reduction, but its final device result
  is only `IcpAlignmentStepRawResult`.
- RED:
  - Added `ICPGpuPathTest.CorrespondenceStatsWorkspaceCanReserveCompactAlignmentStepResult`, which requires a fresh
    `reserveAlignmentStep(4)` workspace to allocate partial storage and a stats/result storage buffer smaller than the
    full `reserve(4)` buffer.
  - `cmake --build build-codex-cuda -j$(nproc)` failed before implementation with
    `IcpCorrespondenceStatsWorkspace has no member named 'reserveAlignmentStep'`.
  - Static check failed before implementation because `alignGpu()` still called
    `_gpu_stats_workspace.reserve(source_count)`.
- Implementation:
  - Added `IcpCorrespondenceStatsWorkspace::reserveAlignmentStep(int)`, which reserves full correspondence partials
    and only the compact `IcpAlignmentStepRawResult` final result buffer.
  - Split `IcpCorrespondenceStatsWorkspace::reserve()` internally into `reservePartialStats()` and
    `reserveStatsStorage()` so the full stats and compact alignment-step paths can share partial reservation without
    forcing the same final-result buffer size.
  - Changed `computeIcpAlignmentStepColumnMajorImpl()` to call `reserveAlignmentStep(source_count)`.
  - Changed `alignGpu()`'s pre-loop workspace warm-up to call `reserveAlignmentStep(source_count)`.
  - Kept `computeIcpStatsAndStepTransformColumnMajorImpl()` on full `reserve(source_count)` because that API still
    writes and copies a full `RawIcpStats` result to host.
- Verification performed in this session:
  - Static check:
    `test "$(rg -n "_gpu_stats_workspace\\.reserveAlignmentStep\\(source_count\\)" include/plapoint/registration/icp.h | wc -l)" -eq 1 &&
    test "$(rg -n "_gpu_stats_workspace\\.reserve\\(source_count\\)" include/plapoint/registration/icp.h | wc -l)" -eq 0 &&
    test "$(rg -n "stats_workspace\\.reserveAlignmentStep\\(source_count\\)" src/icp_gpu.cu | wc -l)" -eq 1` passed after the change.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    CUDA test build succeeded after adding `reserveAlignmentStep()`.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsWorkspaceCanReserveCompactAlignmentStepResult:ICPGpuPathTest.CorrespondenceStatsWorkspaceReusesDeviceStorage:ICPGpuPathTest.AlignmentStepCompactResultMatchesFullStatsStepResult:ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls`:
    all 4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
- Follow-up required when a CUDA device is available:
  rerun the selected tests and compare the Task 93 `gpu_icp_alignment_step_fallback_tile_bounds_*` benchmark rows,
  especially the new-workspace row where smaller result storage can reduce per-call allocation overhead.

## Task 96: Reserve Compact GPU ICP Residual Stats Storage

- Goal: avoid allocating full correspondence-stats partial and result buffers for residual-only GPU ICP stats paths.
  Residual stats kernels write `RawIcpResidualStats` partials and a `RawIcpResidualStats` final result, so reserving
  `RawIcpStats` storage is unnecessary for standalone residual calls and the `alignGpu()` terminal final-metrics path.
- RED:
  - Added `ICPGpuPathTest.CorrespondenceStatsWorkspaceCanReserveCompactResidualStats`, which requires
    `reserveResidualStats(4)` to allocate partial and final result storage smaller than full `reserve(4)`.
  - `cmake --build build-codex-cuda -j$(nproc)` failed before implementation with
    `IcpCorrespondenceStatsWorkspace has no member named 'reserveResidualStats'`.
  - Static check failed before implementation because both residual implementations still called
    `workspace.reserve(source_count)`.
- Implementation:
  - Added `IcpCorrespondenceStatsWorkspace::reserveResidualStats(int)`, which reserves `RawIcpResidualStats` partials
    and final result storage.
  - Replaced the old full-stats partial reservation helper with byte-sized `reservePartialStorage(int, size_t)`, so a
    workspace first used for residual stats will grow correctly if a later full correspondence-stats call needs larger
    `RawIcpStats` partials.
  - Changed both `computeIcpResidualStatsColumnMajorImpl()` and
    `transformPointsAndComputeIcpResidualStatsColumnMajorImpl()` to call `reserveResidualStats(source_count)`.
  - Kept full correspondence stats and compact alignment-step reserve paths unchanged.
- Verification performed in this session:
  - Static check:
    `test "$(rg -n "workspace\\.reserveResidualStats\\(source_count\\)" src/icp_gpu.cu | wc -l)" -eq 2` passed after
    the change.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    CUDA test build succeeded after adding `reserveResidualStats()`.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsWorkspaceCanReserveCompactResidualStats:ICPGpuPathTest.CorrespondenceStatsWorkspaceCanReserveCompactAlignmentStepResult:ICPGpuPathTest.ResidualStatsUsesExactPointwiseFastPathForSameBuffer:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics`:
    all 4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
- Follow-up required when a CUDA device is available:
  rerun the selected tests and benchmark ICP terminal final-metrics cases, especially non-identity terminal iterations
  that call `transformPointsAndComputeIcpResidualStatsColumnMajor()`.

## Task 97: Add GPU ICP Residual Stats Benchmarks

- Goal: make the Task 96 residual-stats workspace optimization measurable. Existing benchmark rows cover full
  correspondence stats, fused stats+step, and compact alignment-step paths, but they do not isolate the residual-only
  stats path or the transform+residual final-metrics path used by `alignGpu()` terminal iterations.
- Static RED:
  - `test "$(rg -n "gpu_icp_residual_stats_finite_radius_translation_(new_workspace|cached_grid)" benchmarks/plapoint_benchmarks.cpp | wc -l)" -ge 4`:
    failed before the change because no residual-stats benchmark rows existed.
  - `test "$(rg -n "gpu_icp_transform_residual_stats_finite_radius_translation_(new_workspace|cached_grid)" benchmarks/plapoint_benchmarks.cpp | wc -l)" -ge 4`:
    failed before the change because no transform+residual benchmark rows existed.
- Implementation:
  - Added `gpu_icp_residual_stats_finite_radius_translation_new_workspace`, which creates a fresh stats workspace for
    each measured residual-only stats call.
  - Added `gpu_icp_residual_stats_finite_radius_translation_cached_grid`, which reuses one stats workspace so the
    finite-radius target spatial grid can be cached across measured residual stats calls.
  - Added `gpu_icp_transform_residual_stats_finite_radius_translation_new_workspace`, which reuses caller-owned output
    and transform buffers but creates a fresh stats workspace per measured transform+residual call.
  - Added `gpu_icp_transform_residual_stats_finite_radius_translation_cached_grid`, which reuses caller-owned output,
    transform, and stats workspace buffers across transform+residual calls.
  - All four rows use the existing translated finite-radius grid dataset and `max_correspondence_distance = 0.02f` so
    they exercise the positive-radius spatial-grid path rather than the fallback tile-bound path.
- Verification performed in this session:
  - Both static RED commands passed after adding the benchmark rows.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    benchmark-only CUDA build succeeded.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran and printed all four new rows; each reported `skipped,no_usable_cuda_device` on this machine.
- Follow-up required when a CUDA device is available:
  rerun the benchmark with enough iterations to compare the new residual rows and quantify whether Task 96's compact
  residual workspace reservation reduces new-workspace overhead or final-metrics path runtime.

## Task 74: Use Read-Only Loads For GPU ICP Spatial-Grid Cell Metadata

- Goal: finish the read-only load cleanup inside the finite-radius spatial-grid search kernels. The per-cell
  `cell_starts` and `cell_counts` arrays are immutable metadata during correspondence and residual scans, and they sit
  on the hot path immediately before sorted target coordinate visits.
- Implementation:
  - Added `loadIcpGridCellStart()` and `loadIcpGridCellCount()` helpers that route metadata reads through
    `loadReadOnlyIcpValue()`.
  - Replaced direct `target_grid.cell_starts[cell_idx]` and `target_grid.cell_counts[cell_idx]` reads in
    `collectCorrespondenceStatsSpatialGridKernel()`, `collectResidualStatsSpatialGridKernel()`, and
    `transformAndCollectResidualStatsSpatialGridKernel()`.
  - Kept target candidate order, tie-breaking, exact-match early stop behavior, and output writes unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `rg -n "target_grid\.(cell_starts|cell_counts)\[[^\]]+\]" src/icp_gpu.cu`:
    no direct spatial-grid start/count array reads remain in `src/icp_gpu.cu`.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates:ICPGpuPathTest.SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 5 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    227 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected spatial-grid GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime
  behavior and performance impact.

## Task 73: Use Read-Only Loads In Remaining GPU ICP Read Paths

- Goal: make the remaining read-only GPU ICP paths consistent with the existing `loadReadOnlyIcpValue()` convention so
  nvcc can use the read-only cache where available. This targets exact pointwise correspondence stats and target
  spatial-grid build/gather loads.
- Implementation:
  - Made `loadReadOnlyIcpValue()` callable from host/device contexts so Thrust device functors can use it.
  - Changed `ComputeIcpTargetGridCellKey` to load target point coordinates through `loadReadOnlyIcpValue()`.
  - Changed `gatherSortedIcpTargetPointsKernel()` to load gathered target coordinates through `loadReadOnlyIcpValue()`.
  - Changed `collectExactPointwiseCorrespondenceStatsKernel()` to match the residual exact fast path's read-only source
    and target loads.
  - Kept output writes, source/output alias assumptions, exact mismatch fallback behavior, and target grid ordering
    unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget:ICPGpuPathTest.SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCache:ICPGpuPathTest.ResidualStatsStopsNonSpatialScanAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 5 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cuda -j$(nproc)` and `ctest --test-dir build-codex-cuda --output-on-failure`:
    227 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected exact-pointwise/grid-build GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to
  confirm runtime behavior and performance impact.

## Task 69: Add Safe Alias Hints To GPU ICP Workspace Paths

- Goal: give nvcc stronger aliasing and read-only load information in GPU ICP kernels without changing public
  aliasing semantics. ICP source and target point buffers may legitimately be the same buffer, and public point
  transforms allow in-place output, so those paths must not be marked `__restrict__`.
- Implementation:
  - Added `__restrict__` only to internal workspace, grid metadata, transform, reduction, and output/result pointers
    where the implementation owns separate storage.
  - Kept source/target point pointer pairs unrestricted where callers may pass the same buffer.
  - Kept public point-transform source/output pointers unrestricted because `transformPointsColumnMajor()` documents
    in-place output support.
  - Moved `loadReadOnlyIcpValue()` before the point loader and routed `loadFiniteColumnMajorPoint()` through it so
    read-only source/target point loads in stats and grid-preparation paths can use the CUDA read-only load path.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.TransformPointsColumnMajorWritesCallerOwnedOutput:ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesSource:ICPGpuPathTest.AlignUsesExactPointwiseStatsForEqualInfiniteRadiusInputs:ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 5 selected GPU alias/grid tests were discovered but skipped because the current
    session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU alias/grid tests and benchmark/profiling rows to confirm runtime impact.

## Task 68: Use Read-Only Loads For GPU ICP Spatial-Grid Metadata

- Goal: reduce read-only global-memory pressure in the finite-radius GPU ICP spatial-grid search path. The hot
  correspondence and residual kernels repeatedly read sorted target coordinates, target indices, and cell keys while
  scanning nearby cells; these arrays are immutable during the search kernels.
- Implementation:
  - Added a `loadReadOnlyIcpValue()` device helper that uses `__ldg` in CUDA device code and falls back to a plain
    dereference for non-device compilation.
  - Added a read-only `loadIcpGridCellKey()` helper and used it in the grid-cell binary search and active z-cell scan.
  - Routed sorted target x/y/z coordinate loads through the read-only helper while preserving the testing coordinate
    load counter.
  - Added `loadSortedIcpTargetIndex()` so target-index tie-breaking and final index writes use read-only loads while
    preserving the testing index-load counter.
  - Kept spatial-grid ordering, candidate pruning, tie-breaking, correspondence stats, and residual stats semantics
    unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsLoadsSpatialGridTargetIndexOnlyForCompetitiveCandidates:ICPGpuPathTest.SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.CorrespondenceStatsBatchesSpatialGridNeighborLookupsByXY:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 5 selected GPU spatial-grid tests were discovered but skipped because the current
    session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests and compare finite-radius ICP benchmark rows with Nsight memory metrics to confirm
  runtime impact.

## Task 57: Specialize Spatial-Grid Kernels By Finite Cell Bounds

- Goal: remove a runtime `finite_cell_bounds` branch from the finite-radius spatial-grid ICP hot path. The branch value
  is fixed when the target grid is prepared, but the candidate-loop cell-bound distance helpers previously received it
  as a per-call boolean inside the GPU kernel.
- Implementation:
  - Added `FiniteCellBounds` template parameters to the correspondence, residual, and transform+residual spatial-grid
    kernels.
  - Changed XY/Z cell-bound distance helper dispatch to `if constexpr`, so each kernel instance keeps only the finite
    or guarded distance path.
  - Added host-side launch helpers that choose the `true` or `false` kernel instance from
    `target_grid.finite_cell_bounds`.
  - Replaced all five spatial-grid correspondence launch sites plus the residual and transform+residual launch sites
    with the host launch helpers.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 58: Unroll Spatial-Grid XY Neighbor Offset Loops

- Goal: reduce small per-thread overhead in the finite-radius spatial-grid candidate search. The correspondence,
  residual, and transform+residual spatial-grid kernels each used a local `offset_order[3]` array and short dynamic
  loop condition while scanning the fixed `{0, -1, 1}` XY neighbor cells.
- Implementation:
  - Added `icpNeighborCellOffset()` as a small `__forceinline__` device helper for the fixed neighbor offset order.
  - Removed the three per-kernel `offset_order` local arrays.
  - Changed the three XY neighbor loops to canonical fixed-trip-count loops with `#pragma unroll`, preserving the
    existing early stop after exact matches via explicit `break` checks.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 59: Specialize Spatial-Grid Correspondence Index Output

- Goal: remove per-thread `correspondence_indices` null checks from the spatial-grid correspondence stats kernel. The
  stats-only and fused step paths launch the same kernel with a null index output, while explicit correspondence
  collection launches it with an output buffer.
- Implementation:
  - Added `WriteCorrespondenceIndices` as a template parameter to `collectCorrespondenceStatsSpatialGridKernel`.
  - Replaced the kernel's index-output branches with `if constexpr`.
  - Made the exact-match early-stop flag compile-time for each kernel instance.
  - Extended the host launch helper to choose among finite/non-finite cell-bound variants and write/no-write index
    variants.
  - Preserved existing tie-break behavior; target indices are still loaded on equal-distance candidates when needed to
    keep deterministic lower-index selection.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStopsSpatialGridAfterExactMatchWhenIndicesOmitted:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance`:
    5 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 60: Remove Redundant Spatial-Grid Active Checks From Kernels

- Goal: remove a per-thread `target_grid.active` branch from the three finite-radius spatial-grid kernels. The host
  code already guards every spatial-grid kernel launch with `if (target_grid.active)`, so the device-side check was
  redundant on all supported call paths.
- Implementation:
  - Removed `&& target_grid.active` from the source-valid gate in the correspondence, residual, and transform+residual
    spatial-grid kernels.
  - Kept the host-side active checks unchanged in all five launch sites, so inactive target grids still fall back to
    the non-spatial-grid kernels.
  - No public API, search order, tie-break behavior, or statistics semantics changed.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 56: Force-Inline GPU ICP Spatial-Grid Helpers

- Goal: reduce device-side call overhead in the finite-radius spatial-grid ICP hot path without changing algorithmic
  behavior. The candidate loops call small helpers for neighbor-cell offsetting, cell lower-bound lookup, cell-bound
  distance computation, and accepted-correspondence accumulation.
- Implementation:
  - Marked frequently used `__device__` helpers in `src/icp_gpu.cu` with `__forceinline__`, including
    `offsetGridCellCoordinate`, `lowerBoundIcpGridCell`, the XY/Z cell-bound distance helpers, and
    `recordAcceptedCorrespondence`.
  - No public API, numeric branch, or test-only statistic semantics changed.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 61: Unroll Accepted Correspondence Statistics Recording

- Goal: reduce fixed-loop overhead in the GPU ICP accepted-correspondence accumulation helper. This helper runs for
  every accepted source/target pair and writes a fixed 3-vector plus 3x3 cross-covariance block into `RawIcpStats`.
- Implementation:
  - Added `#pragma unroll` to the fixed 3-row and 3-column loops inside `recordAcceptedCorrespondence()`.
  - Kept all source sums, target sums, cross sums, residual, correspondence count, and instrumentation assignments
    unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.CorrespondenceStatsSpatialGridTieKeepsLowerTargetIndex:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 62: Force-Inline And Unroll Raw ICP Stats Reduction

- Goal: reduce per-reduction call and loop overhead in the GPU ICP stats accumulation path. `addRawIcpStats()` is used
  by the per-block reductions and final raw-stats reduction kernels, so it sits on the hot path for all GPU ICP stats
  variants.
- Implementation:
  - Marked `addRawIcpStats()` as `__forceinline__`.
  - Added `#pragma unroll` to its fixed 3-entry centroid-sum loop and fixed 9-entry covariance/outer-product loop.
  - Kept every `RawIcpStats` field update and accumulation order unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsAllowOmittedIndexOutput:ICPGpuPathTest.CorrespondenceStatsStillWriteRequestedIndexOutput:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 63: Force-Inline Raw ICP Residual Stats Reduction

- Goal: reduce device call overhead in the GPU ICP residual-stats reduction path. `addRawIcpResidualStats()` is used by
  residual-only per-block reductions and final residual-stats reduction kernels.
- Implementation:
  - Marked `addRawIcpResidualStats()` as `__forceinline__`.
  - Kept all residual stats field updates and accumulation order unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 3 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 64: Force-Inline GPU ICP Point Load And Grid Coordinate Helpers

- Goal: reduce device call overhead in helpers shared by the GPU ICP pointwise and spatial-grid paths. The finite point
  loader is used by correspondence, residual, transform+residual, and target-tile-bound kernels; the grid coordinate
  helper is used during target-grid construction and per-source spatial-grid lookup.
- Implementation:
  - Marked `loadFiniteColumnMajorPoint()` as `__forceinline__`.
  - Marked `icpGridCellCoordinate()` as `__forceinline__` while keeping it `__host__ __device__` for Thrust and host
    callable code paths.
  - Kept all column-major loads, finite checks, floor behavior, and INT_MIN/INT_MAX clamping unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsSpatialGridSkipsNonFiniteTargetInSaturatedCell:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 65: Force-Inline GPU ICP Target Grid Thrust Functors

- Goal: reduce functor call overhead in the finite-radius target spatial-grid build path. Grid construction uses Thrust
  transform, sort-by-key, and reduce-by-key with the ICP grid key transform, comparator, and equality functors.
- Implementation:
  - Marked `IcpGridCellKeyLess::operator()` as `__forceinline__`.
  - Marked `IcpGridCellKeyEqual::operator()` as `__forceinline__`.
  - Marked `ComputeIcpTargetGridCellKey::operator()` as `__forceinline__`.
  - Kept all grid-key ordering, equality, non-finite sentinel, and coordinate conversion semantics unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsUsesFiniteRadiusSpatialGridCandidates:ICPGpuPathTest.CorrespondenceStatsReusesFiniteRadiusSpatialGridForSameTarget:ICPGpuPathTest.AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls:ICPGpuPathTest.SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCache:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 66: Unroll GPU ICP 4x4 Transform Multiplication

- Goal: reduce fixed-loop overhead in `multiplyTransform4x4Kernel()`, which is used to compose column-major 4x4
  transforms in the GPU ICP transform update path.
- Implementation:
  - Added `#pragma unroll` to the fixed 4-term dot-product loop inside `multiplyTransform4x4Kernel()`.
  - Kept the column-major indexing, double accumulation, and scalar output conversion unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.MultiplyTransform4x4UsesColumnMajorTransformComposition:ICPGpuPathTest.MultiplyTransform4x4AsyncUsesCallerStream:ICPTest.Multiply4x4RejectsUnrepresentableAccumulatedTransform:ICPValidation.RecoversKnownTransform`:
    2 CPU tests passed; 2 selected GPU tests were discovered but skipped because the current session has no usable CUDA
    device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 67: Unroll GPU ICP Step-Transform Stats Loops

- Goal: reduce fixed-loop overhead in the GPU ICP raw-stats-to-step-transform path without changing the Jacobi solve
  or transform math. This path is used by the device-side stats+step kernels and compact alignment-step result path.
- Implementation:
  - Added `#pragma unroll` to the fixed 3-entry centroid loop in `computeStepTransformFromRawStatsValue()`.
  - Added `#pragma unroll` to the fixed 3x3 cross-covariance loop in `computeStepTransformFromRawStatsValue()`.
  - Added `#pragma unroll` to the fixed 16-entry identity transform write in
    `writeAlignmentStepRawResultFromRawStats()`.
  - Kept active-count handling, covariance arithmetic, column-major identity layout, and all step validity semantics
    unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.StepTransformFromStatsWritesDeviceTransform:ICPGpuPathTest.AlignmentStepCompactResultMatchesFullStatsStepResult:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPValidation.RecoversKnownTransform`:
    1 CPU validation test passed; 4 selected GPU tests were discovered but skipped because the current session has no
    usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi || true`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 54: Reuse Spatial-Grid Candidate Squared Distances

- Goal: reduce per-candidate arithmetic in the finite-radius spatial-grid kernels. After Task 53, the hot loops already
  compute `dx * dx` and `dx * dx + dy * dy` as partial lower bounds, but the final distance path still recomputed the
  same squared terms and used separate `fabs` axis checks.
- Implementation:
  - Reused `dx_sq`, `dy_sq`, `xy_dist_sq`, and `dz_sq` to compute the final candidate `dist_sq`.
  - Replaced per-axis `fabs(axis_delta) > max_dist` checks in the spatial-grid candidate loops with squared partial
    radius checks against `max_dist_sq`.
  - This also prunes candidates where each individual x/y component is within radius but the 2D partial distance is
    already outside the 3D correspondence radius.
  - Added `SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius`, which expects one visited candidate to load only x/y,
    skip z, and perform zero full-distance evaluations when `dx^2 + dy^2 > radius^2`.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.SpatialGridCandidateLoadsYzCoordinatesOnlyAfterXPruning:ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    5 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 55: Reuse Spatial-Grid XY Cell Distance Across Z Cells

- Goal: reduce repeated cell-bound distance arithmetic in finite-radius spatial-grid pruning. Each source/query XY
  neighbor pair already computes `min_xy_dist_sq`, but the per-Z-cell loop recomputed x/y cell-bound distances for
  every matching Z cell.
- Implementation:
  - Added Z-only cell-bound distance helpers for the normal finite-bound path and the guarded extreme-value path.
  - Changed the correspondence, residual, and transform+residual spatial-grid kernels to compute per-cell distance as
    `min_xy_dist_sq + min_z_dist_sq`.
  - Removed the now-unused full 3D cell-distance helpers.
  - No public behavior changed; existing spatial-grid cell-pruning tests remain the behavioral coverage for this path.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.CorrespondenceStatsPrunesSpatialGridCellsByCurrentBestDistance:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYExceedsRadius:ICPGpuPathTest.SpatialGridCandidateSkipsZLoadWhenXYCannotImproveBest:ICPGpuPathTest.ResidualStatsStopsSpatialGridLookupsAfterExactMatch`:
    4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `cmake --build build-codex-cpu -j$(nproc)` and `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    225 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)` and
    `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran, but all GPU rows reported `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests, full CUDA `ctest`, and the 100k-point ICP benchmark to confirm runtime behavior and
  performance impact.

## Task 98: Skip Redundant Initial GPU Identity Transform Write

- Goal: remove a small unconditional kernel launch from `alignGpu()`. The accumulated transform buffer was initialized
  to identity before every GPU ICP alignment, but non-identity first iterations immediately replaced that buffer with
  the solved step transform.
- RED check:
  - Added `AlignSkipsInitialIdentityTransformWriteForNonIdentityFirstStep` and new test-only identity-transform write
    counters.
  - Before implementing the hooks and optimization, `cmake --build build-codex-cuda -j$(nproc)` failed at link time
    with undefined references to `resetIcpIdentityTransformWriteCountForTesting()` and
    `icpIdentityTransformWriteCountForTesting()`.
- Implementation:
  - Added a test-only counter in `setIdentityTransform4x4Impl()` to count identity-transform write launches.
  - Removed the unconditional `setIdentityTransform4x4Async()` call from the start of `alignGpu()`.
  - For first-iteration exact-identity terminal alignments, reused the step solver's identity transform by swapping
    `_gpu_T_acc` with `_gpu_T_step`.
  - Non-identity first iterations continue to swap the solved step transform into `_gpu_T_acc`, so final transform
    semantics are unchanged.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignSkipsInitialIdentityTransformWriteForNonIdentityFirstStep:ICPGpuPathTest.AlignReusesIterationStatsForExactIdentityTerminalMetrics:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - Static checks confirmed `include/plapoint/registration/icp.h` no longer calls `setIdentityTransform4x4Async()` and
    the identity write counter/test path is present.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    238 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests and 100k-point GPU ICP benchmark to quantify the launch-count and runtime impact.

## Task 99: Avoid Duplicate GPU ICP Alignment-Step Workspace Reserve

- Goal: remove a redundant host-side workspace reserve from `alignGpu()`. The alignment loop called
  `reserveAlignmentStep()` before entering the loop, and `computeIcpAlignmentStepColumnMajor()` reserved the same
  workspace again before launching kernels.
- RED check:
  - Added `AlignReservesAlignmentStepWorkspaceOncePerCall` and test-only alignment-step reserve counters.
  - Before implementing the hooks and optimization, `cmake --build build-codex-cuda -j$(nproc)` failed at link time
    with undefined references to `resetIcpAlignmentStepReserveCountForTesting()` and
    `icpAlignmentStepReserveCountForTesting()`.
- Implementation:
  - Added a test-only counter inside `IcpCorrespondenceStatsWorkspace::reserveAlignmentStep()`.
  - Removed the explicit `_gpu_stats_workspace.reserveAlignmentStep(source_count)` call from the start of `alignGpu()`.
  - The stats helper remains responsible for reserving the workspace, preserving storage reuse and growth behavior.
- Verification performed in this session:
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed after adding the counter and deleting the duplicate reserve.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReservesAlignmentStepWorkspaceOncePerCall:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.AlignComputesStepFromDeviceStatsWithoutHostInputCopy`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - Static checks confirmed `alignGpu()` no longer calls `reserveAlignmentStep()` directly and the reserve counter is
    wired through `IcpCorrespondenceStatsWorkspace::reserveAlignmentStep()`.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    239 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary built and ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests and 100k-point GPU ICP benchmark to confirm the counter and measure any host overhead
  reduction.

## Task 100: Lazily Allocate GPU ICP Next Transform Buffer

- Goal: avoid allocating the extra 4x4 accumulated-transform scratch buffer for one-iteration or first-iteration
  terminal GPU ICP paths. `_gpu_next_T_acc` is only needed when a second or later non-identity step must multiply the
  new step transform into an existing accumulated transform.
- RED check:
  - Added `AlignSkipsNextTransformBufferAllocationForSingleIteration`, which requires a one-iteration non-identity GPU
    align to leave `_gpu_next_T_acc` null.
  - A static RED check against `reserveGpuTransformBuffers()` failed before the implementation because the helper still
    preallocated `_gpu_next_T_acc`.
- Implementation:
  - Removed `_gpu_next_T_acc` allocation from `reserveGpuTransformBuffers()`.
  - Added `reserveGpuNextTransformBuffer()` and call it only immediately before the transform-accumulation multiply on
    iterations after the first.
  - Updated the repeated-workspace test so one-iteration alignments no longer assume the next accumulated-transform
    buffer exists.
- Verification performed in this session:
  - Static checks confirmed `reserveGpuTransformBuffers()` no longer allocates `_gpu_next_T_acc` and
    `reserveGpuNextTransformBuffer()` is present on the transform-multiply path.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignSkipsNextTransformBufferAllocationForSingleIteration:ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the final verification pass.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    240 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests and large finite-radius ICP benchmark to confirm runtime behavior and allocation impact.

## Task 101: Lazily Allocate GPU ICP Step Transform Buffer

- Goal: avoid allocating the accumulated-transform buffer before the first GPU ICP step. The first non-identity or
  exact-identity step already writes `_gpu_T_step`, and `_gpu_T_acc` can take ownership of that buffer instead of
  requiring a separate preallocated 4x4 matrix.
- RED check:
  - Updated `AlignSkipsNextTransformBufferAllocationForSingleIteration` to expect one-iteration GPU ICP to leave
    `_gpu_T_step` null after `_gpu_T_acc` takes ownership of the first step buffer.
  - Updated `AlignReusesGpuWorkspacesAcrossRepeatedCalls` so the first call may allocate only one transform buffer,
    while the second and third calls verify the steady-state transform buffer set is reused.
  - A static RED check against `reserveGpuTransformBuffers()` failed before the implementation because the helper still
    preallocated `_gpu_T_acc`.
- Implementation:
  - Removed the upfront transform-buffer reserve before the GPU ICP loop.
  - Replaced `reserveGpuTransformBuffers()` with `reserveGpuStepTransformBuffer()`, which only ensures the next step
    output buffer exists.
  - Calls `reserveGpuStepTransformBuffer()` at the start of each iteration, allowing `_gpu_T_step` to be reallocated
    only after the first iteration has transferred its buffer into `_gpu_T_acc`.
- Verification performed in this session:
  - Static checks confirmed `_gpu_T_acc` is no longer allocated by the step-buffer reserve path and
    `reserveGpuStepTransformBuffer()` is present on the per-iteration path.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignSkipsNextTransformBufferAllocationForSingleIteration:ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls:ICPGpuPathTest.AlignUsesResidualStatsForNonIdentityTerminalFinalMetrics:ICPGpuPathTest.AlignReservesAlignmentStepWorkspaceOncePerCall`:
    4 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    240 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU tests and large finite-radius ICP benchmark to confirm runtime behavior and quantify the
  first-call allocation reduction.

## Task 102: Preserve GPU ICP Target Cache When Reusing the Same Target

- Goal: avoid discarding the persistent GPU target spatial-grid and tile-bound caches when callers repeat
  `setInputTarget()` with the same target cloud. Reconfiguring the same target should not force the next finite-radius
  GPU ICP align to rebuild target-side acceleration data.
- RED check:
  - Changed the existing target-cache invalidation test into
    `SetInputTargetKeepsPersistentGpuTargetSpatialGridCacheForSameTarget`, expecting the build count to remain 1 after
    resetting the same target pointer.
  - Added `SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCacheForNewTarget`, preserving the requirement that
    switching to a distinct target still invalidates and rebuilds the cache.
  - A static RED check against `setInputTarget()` failed before the implementation because the GPU path invalidated
    caches unconditionally.
- Implementation:
  - `setInputTarget()` now compares the new target shared pointer with the current one for GPU ICP.
  - The GPU target caches are invalidated only when the target pointer changes; CPU behavior remains a direct assignment.
- Verification performed in this session:
  - Static checks confirmed `setInputTarget()` now has the same-target guard.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.SetInputTargetKeepsPersistentGpuTargetSpatialGridCacheForSameTarget:ICPGpuPathTest.SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCacheForNewTarget:ICPGpuPathTest.AlignReusesFiniteRadiusSpatialGridAcrossStatsCalls`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    241 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU cache tests and a repeated-target finite-radius ICP benchmark to quantify avoided target-cache
  rebuilds.

## Task 103: Synchronize Terminal GPU ICP Transform When Final Metrics Are Disabled

- Goal: keep the `setComputeFinalMetrics(false)` GPU ICP path from returning before the terminal output-transform
  kernel has completed. When the final non-identity iteration writes directly to the caller-owned output and no final
  residual metrics are requested, there is no later stats or copy synchronization to naturally wait for that transform.
- RED check:
  - Updated `AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled` to reset the ICP host-synchronization counter
    and expect two synchronizations: one for the fused alignment step result and one for the terminal output transform.
  - A static RED check against the terminal transform branch failed before the implementation because it only called
    `transformPointsColumnMajorAsync()`.
- Implementation:
  - The transform-only branch in `alignGpu()` now uses synchronous `transformPointsColumnMajor()` for the terminal
    iteration and keeps the async transform path for non-terminal iterations.
  - Added `synchronizeIcpStreamWithHost()` in `icp_gpu.cu` so synchronous transform wrappers update the testing-only
    host synchronization counter consistently before calling `cudaStreamSynchronize()`.
- Verification performed in this session:
  - Static checks confirmed the terminal transform branch now contains a synchronous `transformPointsColumnMajor()`
    call and that ICP stream synchronization uses the shared helper.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignCanSkipTerminalFinalStatsWhenFinalMetricsAreDisabled:ICPGpuPathTest.AlignFusesStatsAndStepToAvoidExtraHostSynchronization:ICPGpuPathTest.TransformPointsColumnMajorWritesCallerOwnedOutput`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the full verification pass.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    241 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected GPU synchronization tests and a large finite-radius ICP benchmark with
  `setComputeFinalMetrics(false)` to confirm the runtime synchronization behavior and measure the cost of the terminal
  host wait.

## Task 104: Lazily Allocate GPU ICP Point Scratch Buffers

- Goal: avoid allocating both Nx3 GPU scratch point buffers when a GPU ICP call only needs one. The common
  one-iteration terminal path with output aliasing the source only needs a single temporary output buffer before the
  final device-to-device copy.
- RED check:
  - Updated `AlignUsesScratchForTerminalTransformWhenOutputAliasesSource` to expect only `_gpu_points_a` to be
    allocated and the terminal transform output pointer to match that buffer.
  - Updated `AlignReusesGpuWorkspacesAcrossRepeatedCalls` so repeated one-iteration aliased-output alignments reuse
    `_gpu_points_a` without forcing `_gpu_points_b` allocation.
  - A static RED check against `gpuPointScratchBuffer()` failed before the implementation because it still called
    `reserveGpuPointBuffers()`, which allocated both scratch buffers together.
- Implementation:
  - Replaced the paired `reserveGpuPointBuffers()` helper with `reserveGpuPointBuffer()`, which validates and allocates
    only the requested scratch matrix.
  - `gpuPointScratchBuffer()` now reserves `_gpu_points_a` or `_gpu_points_b` lazily based on the current ping-pong
    side, preserving multi-iteration behavior while avoiding the second allocation in single-scratch paths.
- Verification performed in this session:
  - Static RED check failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesSource:ICPGpuPathTest.AlignReusesGpuWorkspacesAcrossRepeatedCalls`:
    2 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    241 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected scratch-buffer tests and a one-iteration aliased-output GPU ICP benchmark to quantify the avoided
  allocation and memory footprint.

## Task 105: Write Terminal GPU ICP Output Directly When Output Aliases Source

- Goal: avoid the scratch-buffer transform plus final device-to-device copy when callers pass the GPU source cloud as
  the output. The terminal transform can safely write back into the source buffer because each transform kernel reads
  and writes only the same point row, while output aliasing the target must still use scratch to avoid mutating the
  target search data during residual/stat collection.
- RED check:
  - Renamed and updated `AlignWritesTerminalTransformDirectlyWhenOutputAliasesSource` to expect the terminal transform
    output pointer to be the source point buffer and no point scratch buffers to be allocated.
  - Added `AlignUsesScratchForTerminalTransformWhenOutputAliasesTarget`, preserving the safety requirement that target
    aliasing still transforms into scratch before copying into the target output buffer.
  - A static RED check failed before the implementation because `alignGpu()` still used a single
    `output_aliases_input` flag instead of distinguishing target aliasing.
- Implementation:
  - Replaced `output_aliases_input` with `output_aliases_target` in the GPU ICP terminal output decision.
  - Renamed the helper to `outputAliasesGpuTarget()` and only forces scratch when the output object is the target.
- Verification performed in this session:
  - Static RED check failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignWritesTerminalTransformDirectlyWhenOutputAliasesSource:ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesTarget:ICPGpuPathTest.AlignWritesTerminalGpuTransformDirectlyToReusableOutput`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    242 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected output-alias tests and compare one-iteration source-output GPU ICP with and without final metrics
  to quantify the avoided scratch allocation and final copy.

## Task 106: Write Target-Aliased GPU ICP Output Directly Without Final Metrics

- Goal: extend the terminal direct-output fast path to callers that pass the GPU target cloud as the output when final
  residual metrics are disabled. With `setComputeFinalMetrics(false)`, the terminal transform no longer needs to read
  target points after writing the aligned output, so it can skip the scratch transform buffer and final device-to-device
  copy. The default final-metrics path still uses scratch to preserve target points for residual/stat collection.
- RED check:
  - Added `AlignWritesTerminalTransformDirectlyWhenOutputAliasesTargetAndFinalMetricsDisabled`, expecting the terminal
    transform output pointer to match the target point buffer and both point scratch buffers to remain unallocated.
  - A static RED check against the terminal output decision failed before the implementation because the target-alias
    path did not consult `_compute_final_metrics`.
- Implementation:
  - Added a `terminal_output_needs_target_points` guard in `alignGpu()` so target-alias output only forces scratch when
    final metrics are requested.
  - Kept source-alias and caller-owned-output direct terminal paths unchanged.
- Verification performed in this session:
  - Static RED check failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesTarget:ICPGpuPathTest.AlignWritesTerminalTransformDirectlyWhenOutputAliasesTargetAndFinalMetricsDisabled:ICPGpuPathTest.AlignWritesTerminalTransformDirectlyWhenOutputAliasesSource`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the plan update.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    243 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the selected target-alias test and benchmark target-output GPU ICP with `setComputeFinalMetrics(false)` to
  quantify the avoided scratch allocation and final copy.

## Task 107: Invalidate Persistent GPU Target Cache After Target-Aliased Output

- Goal: keep the persistent GPU target spatial-grid and tile-bound caches correct when callers pass the target cloud as
  the align output. Target-aliased output mutates the target device point buffer in place, so a later `align()` with the
  same target pointer must not reuse target-search structures built from the old point contents.
- RED check:
  - Added `AlignInvalidatesPersistentGpuTargetSpatialGridCacheAfterTargetAliasedOutput`, expecting two spatial-grid
    builds across two align calls: the first writes into the target, and the second must rebuild the target grid even
    though the target pointer is unchanged.
  - A static RED check against the end of `alignGpu()` failed before the implementation because the successful
    target-aliased-output path did not invalidate target-cache metadata.
- Implementation:
  - `alignGpu()` now calls `_gpu_stats_workspace.invalidateTargetSpatialGridCache()` after successful alignment when
    `output` aliases the current GPU target. The workspace method also clears tile-bound cache metadata.
  - Existing same-target cache reuse remains intact for ordinary caller-owned output; the new invalidation is limited
    to target-aliased output where target contents can change.
- Verification performed in this session:
  - Static RED check failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignInvalidatesPersistentGpuTargetSpatialGridCacheAfterTargetAliasedOutput:ICPGpuPathTest.SetInputTargetKeepsPersistentGpuTargetSpatialGridCacheForSameTarget:ICPGpuPathTest.SetInputTargetInvalidatesPersistentGpuTargetSpatialGridCacheForNewTarget`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the plan update.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    244 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    143 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the target-cache invalidation test on real GPU hardware, then benchmark repeated target-output align calls to
  confirm the rebuild cost is only paid when target contents were actually mutated.

## Task 108: Add GPU ICP Target-Output Skip-Final-Metrics Benchmark

- Goal: make the target-aliased output fast path measurable in the benchmark binary. The existing reusable-output
  skip-final-metrics row measured caller-owned output, but did not exercise `align(*target)` with
  `setComputeFinalMetrics(false)`, which is the path optimized in Task 106.
- RED check:
  - Ran the benchmark smoke command and grepped for
    `gpu_icp_finite_radius_translation_reuse_target_output_skip_final_metrics`.
  - The check failed before the implementation because no such benchmark row existed.
- Implementation:
  - Added `benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutputSkipFinalMetrics()`.
  - The benchmark reuses one GPU ICP object, passes the target cloud as the output, disables final metrics, and reports
    a separate CSV row next to the existing caller-owned-output skip-final-metrics benchmark.
- Verification performed in this session:
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 1 --icp-points 1000 --icp-max-iterations 1 --skip-cpu-icp | rg "gpu_icp_finite_radius_translation_reuse_target_output_skip_final_metrics"`:
    matched the new row; in this environment it reports `skipped,no_usable_cuda_device`.
- Follow-up required when a CUDA device is available:
  compare the new target-output row against
  `gpu_icp_finite_radius_translation_reuse_output_skip_final_metrics` to quantify the direct target-output fast path.

## Task 109: Version GPU Target Points for Persistent ICP Cache Validation

- Goal: keep the persistent GPU ICP target spatial-grid and tile-bound caches valid when user code mutates a target
  cloud in place between two `align()` calls. The previous cache key only used the target device pointer, so a mutable
  `target->points()` access could change target contents without changing the pointer and allow stale target-search
  structures to be reused.
- RED check:
  - Added `PointCloudTest.MutablePointAccessIncrementsPointsVersion`.
  - Added `AlignInvalidatesPersistentGpuTargetSpatialGridCacheAfterMutableTargetPointsAccess`, expecting a second
    spatial-grid build after external mutable target point access with the same target object.
  - `cmake --build build-codex-cpu -j$(nproc)` failed before implementation because `PointCloud` did not expose
    `pointsVersion()`.
- Implementation:
  - Added a `PointCloud::pointsVersion()` counter that increments on mutable `points()` access and remains stable for
    const `points()` and `pointsCpu()` reads.
  - GPU ICP now records both target point device pointer and `pointsVersion()` before using the persistent workspace
    cache; a mismatch invalidates the cached spatial grid and tile bounds while preserving reuse for unchanged target
    objects.
  - Target-aliased output invalidation now resets the ICP-level cache key as well as the workspace cache metadata.
- Verification performed in this session:
  - `cmake --build build-codex-cpu -j$(nproc) && ./build-codex-cpu/test/plapoint_tests --gtest_filter=PointCloudTest.MutablePointAccessIncrementsPointsVersion`:
    passed after implementation.
  - `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignInvalidatesPersistentGpuTargetSpatialGridCacheAfterMutableTargetPointsAccess:ICPGpuPathTest.SetInputTargetKeepsPersistentGpuTargetSpatialGridCacheForSameTarget`:
    2 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the plan update.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    246 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    144 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the mutable-target cache invalidation test on GPU hardware and verify unchanged-target cache reuse still builds
  the finite-radius spatial grid only once.

## Task 110: Invalidate GPU Target Cache Before Target-Output Writes

- Goal: make target-aliased GPU ICP output cache invalidation exception-safe. When `align(*target)` writes into the
  target point buffer, the persistent target spatial-grid and tile-bound caches must be invalidated before the write is
  attempted so a later exception cannot leave mutated target contents paired with stale cache metadata. Avoid
  invalidating the cache when the output buffer already aliases the current points and no target write occurs.
- RED check:
  - A static check against the terminal direct-output branch failed before the implementation because
    `invalidateGpuTargetWorkspaceCache()` did not appear before `gpu::transformPointsColumnMajor(...)`.
  - A static check against the final copy branch failed before the implementation because
    `invalidateGpuTargetWorkspaceCache()` did not appear before the final `cudaMemcpy(...)` into output storage.
- Implementation:
  - Moved target-cache invalidation into the terminal direct-output branch immediately after preparing the target
    output buffer and before launching the transform write.
  - Moved target-cache invalidation into the final device-to-device copy branch immediately before copying into target
    output storage.
  - Removed the unconditional end-of-align target-alias invalidation so equal-buffer no-write cases keep their valid
    target cache.
- Verification performed in this session:
  - Both static RED checks failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignInvalidatesPersistentGpuTargetSpatialGridCacheAfterTargetAliasedOutput:ICPGpuPathTest.AlignWritesTerminalTransformDirectlyWhenOutputAliasesTargetAndFinalMetricsDisabled:ICPGpuPathTest.SetInputTargetKeepsPersistentGpuTargetSpatialGridCacheForSameTarget`:
    3 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the plan update.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    246 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    144 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the target-output cache invalidation cases on real GPU hardware, including the skip-final-metrics direct target
  output path and the same-buffer no-write path.

## Task 111: Skip Repeated Alignment-Step Workspace Reserve Checks

- Goal: reduce CPU-side overhead in multi-iteration GPU ICP by avoiding repeated `reserveAlignmentStep()` work once the
  reusable partial-reduction and compact alignment-step result buffers already have enough capacity for the current
  source point count.
- RED check:
  - Tightened `AlignReservesAlignmentStepWorkspaceOncePerCall` so a two-iteration GPU align must make two compact
    alignment-step calls but only one alignment-step workspace reserve.
  - Added a static RED check against `IcpCorrespondenceStatsWorkspace::reserveAlignmentStep()` requiring a capacity
    guard and early return before the testing reserve counter increments. The check failed before the implementation.
- Implementation:
  - `reserveAlignmentStep()` now returns immediately for zero source count before touching the test counter.
  - For non-empty sources, it computes the required partial count and byte count, then returns early when the existing
    partial storage, partial capacity, and result storage are already sufficient.
  - The public GPU ICP helper API is unchanged; direct callers still get automatic first-use allocation.
- Verification performed in this session:
  - The static RED check failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignReservesAlignmentStepWorkspaceOncePerCall`:
    the selected GPU test was discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the plan update.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    246 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    144 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun `AlignReservesAlignmentStepWorkspaceOncePerCall` on GPU hardware and compare repeated-iteration ICP benchmark
  rows before/after this change to quantify host-side overhead reduction.

## Task 112: Avoid Mutable Output Access For Same-Buffer No-Write GPU Output

- Goal: avoid invalidating the target cache version in GPU ICP paths that do not write output points. The final output
  copy path used `prepareGpuOutputPointBuffer()` before checking whether the output points already matched
  `cur_points`; that helper calls mutable `PointCloud::points()`, which increments `pointsVersion()` even when no
  device write or copy follows.
- RED check:
  - Added `AlignDoesNotIncrementTargetPointsVersionForSameBufferNoWriteOutput`, which runs identity GPU ICP with the
    same cloud as source, target, and output, then expects `pointsVersion()` to remain unchanged.
  - Added a static RED check requiring the final copy branch to call a same-buffer guard before
    `prepareGpuOutputPointBuffer()`. The check failed before the implementation.
- Implementation:
  - Added GPU-only `gpuOutputAlreadyContainsCurrentPoints()` using const point access and the existing reusable-output
    metadata checks.
  - The final copy branch now skips mutable output preparation when the output already owns the current point buffer.
  - Kept the existing copy and target-cache invalidation behavior for all paths that actually write or replace output
    storage.
- Verification performed in this session:
  - The static RED check failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignDoesNotIncrementTargetPointsVersionForSameBufferNoWriteOutput`:
    the selected GPU test was discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the plan update.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    247 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    144 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun the new same-buffer no-write version test on GPU hardware and verify repeated identity `align(*target)` calls
  keep persistent target cache reuse intact.

## Task 113: Direct Target Output For Spatial-Grid Final Metrics

- Goal: let target-aliased GPU ICP output write the terminal transform directly into the target point buffer when final
  metrics can use an already cached finite-radius target spatial-grid snapshot. This avoids the previous scratch
  transform buffer and final device-to-device copy while preserving the conservative scratch path for unbounded or
  fallback final metrics that still read the target point buffer directly.
- RED check:
  - Renamed the existing target-alias final-metrics scratch test to
    `AlignUsesScratchForTerminalTransformWhenOutputAliasesTargetWithoutSpatialGridSnapshot` and made it use the
    unbounded/fallback path.
  - Added `AlignWritesTerminalTransformDirectlyWhenOutputAliasesTargetAndFinalMetricsUseSpatialGrid`, expecting
    finite-radius final metrics to write to the target point buffer without allocating point scratch buffers.
  - Added static RED checks requiring the high-level terminal-output decision to consult
    `gpuFinalMetricsCanUseCachedTargetSpatialGridSnapshot()` and requiring a GPU snapshot residual helper. Both checks
    failed before the implementation.
- Implementation:
  - Added `transformPointsAndComputeIcpResidualStatsWithTargetSpatialGridSnapshotColumnMajor()` for float and double.
    It launches the existing transform+spatial-grid residual kernel using workspace snapshot storage and a captured
    cell count, without reading `d_target_points`.
  - `alignGpu()` now records whether the terminal final metrics can use the cached target spatial-grid snapshot before
    invalidating target cache metadata for target output writes.
  - Target-aliased direct output remains disabled for final metrics paths without a cached spatial-grid snapshot, so
    fallback kernels that read target points still use scratch output.
- Verification performed in this session:
  - Static RED checks failed before the implementation and passed after the implementation.
  - `cmake --build build-codex-cuda -j$(nproc) && ./build-codex-cuda/test/plapoint_tests --gtest_filter=ICPGpuPathTest.AlignUsesScratchForTerminalTransformWhenOutputAliasesTargetWithoutSpatialGridSnapshot:ICPGpuPathTest.AlignWritesTerminalTransformDirectlyWhenOutputAliasesTargetAndFinalMetricsUseSpatialGrid`:
    2 selected GPU tests were discovered but skipped because the current session has no usable CUDA device.
  - `git diff --check`:
    clean before the plan update.
  - `cmake --build build-codex-cuda -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cpu -j$(nproc)`:
    passed.
  - `cmake --build build-codex-cuda-bench-only -j$(nproc)`:
    passed.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    248 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    144 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran; GPU rows reported `skipped,no_usable_cuda_device`.
  - `nvidia-smi`:
    reported that it could not communicate with the NVIDIA driver.
- Follow-up required when a CUDA device is available:
  rerun both target-alias final-metrics tests on GPU hardware and add a benchmark row comparing target-output final
  metrics against caller-owned output and skip-final-metrics target output.

## Task 114: Benchmark Target Output With Final Metrics

- Goal: add a benchmark row for finite-radius GPU ICP target-output alignment with terminal final metrics enabled. This
  measures the Task 113 direct target-output path against the existing caller-owned output row and the target-output
  skip-final-metrics row.
- RED check:
  - Ran the benchmark smoke command and filtered for
    `^gpu_icp_finite_radius_translation_reuse_target_output,`; the grep failed before the implementation because the
    row did not exist.
  - Searched `benchmarks/plapoint_benchmarks.cpp` for
    `gpu_icp_finite_radius_translation_reuse_target_output` and
    `benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutput`; both were absent before the implementation.
- Implementation:
  - Added `benchmarkGpuIcpFiniteRadiusTranslationReuseTargetOutput()`, which builds the same translated finite-radius
    GPU ICP fixture as the caller-owned output benchmarks and times `icp.align(*target)`.
  - Kept the no-device behavior consistent with the existing GPU benchmark rows by printing
    `gpu_icp_finite_radius_translation_reuse_target_output,skipped,no_usable_cuda_device,`.
  - Inserted the new row between the caller-owned skip-final-metrics row and the existing target-output
    skip-final-metrics row so the output groups the comparable paths together.
- Verification performed in this session:
  - The targeted RED check failed before the implementation and passed after rebuilding the benchmark binary:
    `gpu_icp_finite_radius_translation_reuse_target_output,skipped,no_usable_cuda_device,`.
  - `git diff --check`:
    clean before the plan update.
  - `ctest --test-dir build-codex-cuda --output-on-failure`:
    248 test entries, 0 failed; GPU-dependent tests skipped because the current session cannot communicate with the
    NVIDIA driver.
  - `ctest --test-dir build-codex-cpu --output-on-failure`:
    144 tests, 0 failed, 1 skipped CUDA-only transfer case.
  - `./build-codex-cuda-bench-only/benchmarks/plapoint_benchmarks --points 1000 --iterations 5 --icp-points 100000 --icp-max-iterations 3 --skip-cpu-icp`:
    benchmark binary ran and printed the new
    `gpu_icp_finite_radius_translation_reuse_target_output,skipped,no_usable_cuda_device,` row.
- Follow-up required when a CUDA device is available:
  rerun the benchmark on real GPU hardware and compare `gpu_icp_finite_radius_translation_reuse_output`,
  `gpu_icp_finite_radius_translation_reuse_target_output`, and
  `gpu_icp_finite_radius_translation_reuse_target_output_skip_final_metrics`.
