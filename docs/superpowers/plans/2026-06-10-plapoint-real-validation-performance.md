# PlaPoint Real Validation And Performance Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add repeatable real reconstruction quality checks, a structured img/tsai pipeline entry, fixed benchmark baseline gating, and a CUDA hotspot report entry.

**Architecture:** Keep the work in PlaPoint maintenance scripts and tests so it does not disturb the C++ library API. Reuse existing PLY parsing and benchmark JSON formats, add focused Python helpers, and expose the workflows through README commands.

**Tech Stack:** Python 3 standard library, CMake/CTest, existing PlaPoint benchmark executable, existing `testData/real_reconstruction` PLY and summary files.

---

### Task 1: Real Reconstruction Quality Gate

**Files:**
- Modify: `scripts/run_real_reconstruction_regression.py`
- Modify: `scripts/test_python_tools.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Add tests that call `real_regression.evaluate_quality_metrics()` on a tiny PLY with `error` and `intensity`, assert that it reports point count, finite coordinate ratio, error mean/max, and intensity range, and assert that `real_regression.check_quality_thresholds()` fails when `max_error` exceeds a configured threshold.

- [ ] **Step 2: Run RED test**

Run:

```bash
python3 scripts/test_python_tools.py
```

Expected before implementation: failure because `evaluate_quality_metrics` and `check_quality_thresholds` do not exist.

- [ ] **Step 3: Implement quality metrics**

Extend `scripts/run_real_reconstruction_regression.py` with reusable PLY metric extraction for `error` and `intensity`, quality threshold checks, CLI flags `--quality-json-output`, `--max-error`, `--max-mean-error`, `--min-finite-ratio`, and include quality failures in the exit code.

- [ ] **Step 4: Run GREEN test**

Run:

```bash
python3 scripts/test_python_tools.py
```

Expected after implementation: all Python script unit tests pass.

### Task 2: Real Pipeline Command Wrapper

**Files:**
- Create: `scripts/run_real_reconstruction_pipeline.py`
- Modify: `scripts/test_python_tools.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Add tests for `pipeline.build_pipeline_command()` verifying placeholder expansion for `{img_dir}`, `{tsai_dir}`, `{output_dir}`, and `{plapoint_root}`, and `pipeline.default_output_dir()` producing a PlaPoint-local build directory.

- [ ] **Step 2: Run RED test**

Run:

```bash
python3 scripts/test_python_tools.py
```

Expected before implementation: import failure or missing function failure for `run_real_reconstruction_pipeline`.

- [ ] **Step 3: Implement wrapper**

Create a Python CLI that validates source img/tsai directories using `run_real_reconstruction_regression.validate_source_inputs`, runs a user-supplied command template, and optionally invokes `run_real_reconstruction_regression.main()` to compare generated outputs.

- [ ] **Step 4: Run GREEN test**

Run:

```bash
python3 scripts/test_python_tools.py
```

Expected after implementation: all Python script unit tests pass.

### Task 3: Benchmark Baseline Gate

**Files:**
- Create: `scripts/benchmark_gate_config.json`
- Modify: `scripts/compare_benchmark_baseline.py`
- Modify: `scripts/test_python_tools.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Add tests showing `compare_benchmark.load_gate_config()` reads thresholds and ignored benchmark names from JSON, and `compare_benchmark.compare_files()` ignores configured names while still reporting other regressions.

- [ ] **Step 2: Run RED test**

Run:

```bash
python3 scripts/test_python_tools.py
```

Expected before implementation: failure because the config loader and ignore handling do not exist.

- [ ] **Step 3: Implement gate config**

Add default threshold config, support `--config`, `--ignore-benchmark`, and include ignored row counts in JSON/Markdown output.

- [ ] **Step 4: Run GREEN test**

Run:

```bash
python3 scripts/test_python_tools.py
```

Expected after implementation: all Python script unit tests pass.

### Task 4: CUDA Hotspot Report

**Files:**
- Create: `scripts/report_cuda_hotspots.py`
- Modify: `scripts/test_python_tools.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests**

Add tests for `report_cuda_hotspots.select_hotspots()` using a synthetic benchmark JSON. Verify it keeps `gpu_` rows, sorts by current time descending, computes CPU/GPU ratio when a matching `cpu_` row exists, and emits Markdown.

- [ ] **Step 2: Run RED test**

Run:

```bash
python3 scripts/test_python_tools.py
```

Expected before implementation: import failure or missing function failure for `report_cuda_hotspots`.

- [ ] **Step 3: Implement report script**

Create a Python CLI that reads a benchmark baseline JSON or comparison JSON, produces top-N CUDA hotspot rows, writes JSON/Markdown artifacts, and exits non-zero only on malformed input.

- [ ] **Step 4: Run GREEN and integration tests**

Run:

```bash
python3 scripts/test_python_tools.py
ctest --test-dir build-abc-cuda -R plapoint.scripts.python_tools --output-on-failure
```

Expected after implementation: both commands pass.

### Task 5: Final Verification And Commit

**Files:**
- Modify as listed above.

- [ ] **Step 1: Run full verification**

Run:

```bash
git diff --check
cmake --build build-abc-cuda --target plapoint_tests plapoint_benchmarks -j$(nproc)
ctest --test-dir build-abc-cuda --output-on-failure
python3 scripts/run_real_reconstruction_regression.py --generated-root /home/xjw/code/mygithub/plascan/build/real_data/testData_dense --actual-layout plascan-legacy --json-output build-abc-cuda/real_reconstruction_regression/comparison_after_1_4.json --quality-json-output build-abc-cuda/real_reconstruction_regression/quality_after_1_4.json
python3 scripts/run_benchmark_baseline.py --benchmark-exe build-abc-cuda/benchmarks/plapoint_benchmarks --output-dir build-abc-cuda/benchmark_1_4_smoke --timeout-seconds 180
python3 scripts/report_cuda_hotspots.py build-abc-cuda/benchmark_1_4_smoke/plapoint_benchmark_baseline.json --markdown-output build-abc-cuda/benchmark_1_4_smoke/cuda_hotspots.md
```

- [ ] **Step 2: Commit PlaPoint changes**

Run:

```bash
git status --short
git add docs/superpowers/plans/2026-06-10-plapoint-real-validation-performance.md scripts README.md
git commit -m "chore: add real validation and performance gates"
```

Do not update the parent `plascan` submodule pointer unless the user explicitly asks.
