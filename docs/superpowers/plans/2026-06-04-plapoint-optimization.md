# PlaPoint Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five reviewed optimization areas: GPU KNN data movement, GPU-path clarity, voxel CUDA correctness, parameter/bounds validation, and ICP robustness/metrics.

**Architecture:** Keep the public API compatible where practical. Add narrow helper APIs for cached GPU KNN and safer CPU fallback behavior, then improve validation and ICP configuration without rewriting unrelated modules. Prefer small targeted tests that fail against the current code before each implementation step.

**Tech Stack:** C++17, CUDA runtime, PlaMatrix, Google Test, CMake.

---

### Task 1: GPU KNN Layout And Cache

**Files:**
- Modify: `include/plapoint/search/kdtree.h`
- Modify: `include/plapoint/gpu/knn.h`
- Modify: `src/knn_gpu.cu`
- Modify: `src/batch_knn_gpu.cu`
- Test: `test/unit/search/kdtree_gpu_test.cpp`

- [x] Add a GPU KNN test that reads column-major PlaMatrix device data directly, matching CPU KNN results without a device-to-host point cloud roundtrip per query.
- [x] Update the KNN kernel to read PlaMatrix column-major buffers directly via `batchKnnDeviceColumnMajor()`.
- [x] Stage GPU kd-tree CPU fallback points once during `build()` instead of per-coordinate during recursive search.
- [x] Re-run `ctest --test-dir build-codex-cuda --output-on-failure`.

### Task 2: GPU Path Semantics And Performance Guards

**Files:**
- Modify: `README.md`
- Modify: `include/plapoint/features/normal_estimation.h`
- Modify: `include/plapoint/features/normal_refinement.h`
- Modify: `include/plapoint/filters/filter.h`
- Modify: `include/plapoint/filters/uniform_downsample.h`
- Modify: `include/plapoint/filters/voxel_grid.h`
- Modify: `include/plapoint/filters/statistical_outlier_removal.h`
- Modify: `include/plapoint/filters/radius_outlier_removal.h`
- Test: module tests under `test/unit/features/` and `test/unit/filters/`

- [x] Add tests that GPU filters/features either use one bulk host staging step or document and expose CPU fallback behavior.
- [x] Replace repeated GPU single-element reads/writes in common output assembly paths with bulk CPU staging followed by one `toGpu()`.
- [x] Update README to state which operations are real GPU kernels and which are CPU-staged fallback paths.
- [x] Re-run CPU and CUDA module tests.

### Task 3: Voxel CUDA Correctness

**Files:**
- Modify: `src/voxel_grid_gpu.cu`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/filters/voxel_grid_test.cpp`

- [x] Remove the unused incomplete CUDA voxel source from the production build until the sort/reduce pipeline is implemented.
- [x] Mark `src/voxel_grid_gpu.cu` as experimental and not part of the production target.
- [x] Keep the public `VoxelGrid` behavior deterministic and covered by existing CPU/GPU output tests.
- [x] Re-run CUDA tests.

### Task 4: Parameter And Bounds Validation

**Files:**
- Modify: `include/plapoint/core/point_cloud.h`
- Modify: `include/plapoint/search/kdtree.h`
- Modify: `include/plapoint/features/normal_estimation.h`
- Modify: `include/plapoint/features/normal_refinement.h`
- Modify: `include/plapoint/filters/statistical_outlier_removal.h`
- Modify: `include/plapoint/filters/radius_outlier_removal.h`
- Modify: `include/plapoint/mesh/poisson_reconstruction.h`
- Modify: `include/plapoint/io/las_io.h`
- Test: matching unit tests

- [x] Add tests for invalid `k`, radius, neighbor count, Poisson depth/iteration count, point view bounds, optional attribute access, and ICP edge cases.
- [x] Add clear `std::invalid_argument`, `std::out_of_range`, or `std::overflow_error` exceptions.
- [x] Add int-range guard helpers where APIs or kernels require `int`.
- [x] Re-run full CPU and CUDA tests.

### Task 5: ICP Robustness And Metrics

**Files:**
- Modify: `include/plapoint/registration/icp.h`
- Test: `test/unit/registration/icp_test.cpp`
- Test: `test/validation/validation_test.cpp`

- [x] Add tests for max correspondence distance rejecting outliers, final fitness score, final RMSE, invalid correspondence parameters, low fitness, and degenerate geometry.
- [x] Add `setMaxCorrespondenceDistance()`, `setMinFitnessScore()`, `getFitnessScore()`, `getFinalRmse()`, and robust correspondence filtering.
- [x] Preserve current identity and known-transform validation behavior.
- [x] Re-run registration tests and full test suites.

### Final Verification

- [x] Configure/build CPU and CUDA test trees if needed.
- [x] Run `ctest --test-dir build-codex-cpu --output-on-failure`.
- [x] Run `ctest --test-dir build-codex-cuda --output-on-failure`.
- [x] Review `git diff`.
- [ ] Commit with the configured GitHub identity.
- [ ] Push to `origin master` and report whether GitHub authentication succeeded.

Verification evidence:

- CPU: `cmake --build build-codex-cpu -j$(nproc) && ctest --test-dir build-codex-cpu --output-on-failure` passed 103/103 tests, with `plapoint.PointCloudTest.GpuTransfer` skipped in the CPU-only build.
- CUDA: `cmake --build build-codex-cuda -j$(nproc) && ctest --test-dir build-codex-cuda --output-on-failure` passed 118/118 tests.
- Diff hygiene: `git diff --check` returned no issues.
