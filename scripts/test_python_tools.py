#!/usr/bin/env python3
"""Unit tests for PlaPoint maintenance scripts."""

from __future__ import annotations

import json
import importlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import compare_benchmark_baseline as compare_benchmark
import run_cpu_only_validation as cpu_validation
import run_real_reconstruction_regression as real_regression


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")


def write_ply(path: Path, rows: list[tuple[float, float, float, float, int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(rows)}",
        "property float x",
        "property float y",
        "property float z",
        "property float error",
        "property uchar intensity",
        "end_header",
    ]
    for x, y, z, error, intensity in rows:
        lines.append(f"{x} {y} {z} {error} {intensity}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class BenchmarkBaselineCompareTest(unittest.TestCase):
    def test_detects_regression_and_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.json"
            current = root / "current.json"
            write_json(
                baseline,
                [
                    {"benchmark": "stable", "status": "measured", "best_ms": 10.0},
                    {"benchmark": "slow", "status": "measured", "best_ms": 10.0},
                    {"benchmark": "fast", "status": "measured", "best_ms": 10.0},
                ],
            )
            write_json(
                current,
                [
                    {"benchmark": "stable", "status": "measured", "best_ms": 10.5},
                    {"benchmark": "slow", "status": "measured", "best_ms": 13.0},
                    {"benchmark": "fast", "status": "measured", "best_ms": 7.0},
                ],
            )

            result = compare_benchmark.compare_files(
                baseline,
                current,
                regression_threshold=0.20,
                improvement_threshold=0.20,
                min_ms=0.001,
            )

        by_name = {row.name: row for row in result.rows}
        self.assertEqual(by_name["stable"].status, "unchanged")
        self.assertEqual(by_name["slow"].status, "regressed")
        self.assertEqual(by_name["fast"].status, "improved")
        self.assertEqual(result.regression_count, 1)
        self.assertEqual(result.improvement_count, 1)
        self.assertEqual(result.exit_code(fail_on_regression=True), 1)

    def test_marks_missing_and_added_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.json"
            current = root / "current.json"
            write_json(
                baseline,
                [{"benchmark": "old", "status": "measured", "best_ms": 1.0}],
            )
            write_json(
                current,
                [{"benchmark": "new", "status": "measured", "best_ms": 2.0}],
            )

            result = compare_benchmark.compare_files(baseline, current)

        by_name = {row.name: row for row in result.rows}
        self.assertEqual(by_name["old"].status, "missing")
        self.assertEqual(by_name["new"].status, "added")
        self.assertEqual(result.exit_code(fail_on_regression=False, fail_on_missing=True), 1)

    def test_gate_config_ignores_noisy_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline.json"
            current = root / "current.json"
            config_path = root / "gate.json"
            write_json(
                baseline,
                [
                    {"benchmark": "gpu_noisy", "status": "measured", "best_ms": 10.0},
                    {"benchmark": "gpu_real", "status": "measured", "best_ms": 10.0},
                ],
            )
            write_json(
                current,
                [
                    {"benchmark": "gpu_noisy", "status": "measured", "best_ms": 20.0},
                    {"benchmark": "gpu_real", "status": "measured", "best_ms": 13.0},
                ],
            )
            config_path.write_text(
                json.dumps(
                    {
                        "regression_threshold": 0.20,
                        "improvement_threshold": 0.20,
                        "min_ms": 0.001,
                        "ignore_benchmarks": ["gpu_noisy"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = compare_benchmark.load_gate_config(config_path)
            result = compare_benchmark.compare_files(
                baseline,
                current,
                regression_threshold=config.regression_threshold,
                improvement_threshold=config.improvement_threshold,
                min_ms=config.min_ms,
                ignored_benchmarks=config.ignore_benchmarks,
            )

        by_name = {row.name: row for row in result.rows}
        self.assertEqual(by_name["gpu_noisy"].status, "ignored")
        self.assertEqual(by_name["gpu_real"].status, "regressed")
        self.assertEqual(result.ignored_count, 1)
        self.assertEqual(result.regression_count, 1)


class RealReconstructionRegressionTest(unittest.TestCase):
    def test_compares_matching_ply_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = root / "expected"
            actual = root / "actual"
            for base in (expected, actual):
                (base / "pairs" / "002_001").mkdir(parents=True)
                write_ply(
                    base / "pairs" / "002_001" / "cloud_gray.ply",
                    [(1.0, 2.0, 3.0, 0.1, 10), (2.0, 3.0, 4.0, 0.2, 20)],
                )

            result = real_regression.compare_reference_tree(
                expected,
                actual,
                ["pairs/002_001/cloud_gray.ply"],
                point_tolerance=0,
                mean_tolerance=0.0,
                bbox_tolerance=0.0,
                intensity_mean_tolerance=0.0,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.compared_files, 1)

    def test_detects_intensity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = root / "expected"
            actual = root / "actual"
            for base, intensity in ((expected, 10), (actual, 30)):
                (base / "pairs" / "002_001").mkdir(parents=True)
                write_ply(
                    base / "pairs" / "002_001" / "cloud_gray.ply",
                    [(1.0, 2.0, 3.0, 0.1, intensity)],
                )

            result = real_regression.compare_reference_tree(
                expected,
                actual,
                ["pairs/002_001/cloud_gray.ply"],
                point_tolerance=0,
                mean_tolerance=0.0,
                bbox_tolerance=0.0,
                intensity_mean_tolerance=0.0,
            )

        self.assertFalse(result.ok)
        self.assertIn("intensity mean", result.failures[0])

    def test_compares_plascan_legacy_merged_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = root / "expected"
            actual = root / "actual"
            (expected / "merged").mkdir(parents=True)
            actual.mkdir(parents=True)
            write_ply(
                expected / "merged" / "merged_dense_gray.ply",
                [(1.0, 2.0, 3.0, 0.1, 10)],
            )
            write_ply(
                actual / "merged_dense_gray.ply",
                [(1.0, 2.0, 3.0, 0.1, 10)],
            )

            result = real_regression.compare_reference_tree(
                expected,
                actual,
                ["merged/merged_dense_gray.ply"],
                point_tolerance=0,
                mean_tolerance=0.0,
                bbox_tolerance=0.0,
                intensity_mean_tolerance=0.0,
                actual_layout="plascan-legacy",
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.compared_files, 1)

    def test_evaluates_quality_metrics_from_error_and_intensity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cloud = root / "cloud_gray.ply"
            write_ply(
                cloud,
                [(1.0, 2.0, 3.0, 0.01, 10), (math.nan, 2.0, 3.0, 0.03, 40)],
            )

            metrics = real_regression.evaluate_quality_metrics(root, ["cloud_gray.ply"])
            failures = real_regression.check_quality_thresholds(
                metrics,
                max_error=0.02,
                max_mean_error=0.015,
                min_finite_ratio=0.9,
            )

        metric = metrics["cloud_gray.ply"]
        self.assertEqual(metric["point_count"], 2)
        self.assertAlmostEqual(metric["finite_coordinate_ratio"], 0.5)
        self.assertAlmostEqual(metric["error_mean"], 0.02)
        self.assertAlmostEqual(metric["error_max"], 0.03)
        self.assertEqual(metric["intensity_min"], 10.0)
        self.assertEqual(metric["intensity_max"], 40.0)
        self.assertEqual(len(failures), 3)
        self.assertTrue(any("max error" in failure for failure in failures))
        self.assertTrue(any("mean error" in failure for failure in failures))
        self.assertTrue(any("finite coordinate ratio" in failure for failure in failures))

    def test_main_rejects_missing_reference_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_dir = root / "img"
            tsai_dir = root / "tsai"
            img_dir.mkdir()
            tsai_dir.mkdir()
            for index in range(1, 6):
                (img_dir / f"{index}.png").write_bytes(b"")
                (tsai_dir / f"{index}.tsai").write_text("# camera\n", encoding="utf-8")

            exit_code = real_regression.main(
                [
                    "--source-img-dir",
                    str(img_dir),
                    "--source-tsai-dir",
                    str(tsai_dir),
                    "--reference-root",
                    str(root / "missing-reference"),
                    "--output-dir",
                    str(root / "out"),
                    "--json-output",
                    str(root / "comparison.json"),
                ]
            )

        self.assertEqual(exit_code, 2)


class RealReconstructionPipelineTest(unittest.TestCase):
    def test_builds_pipeline_command_from_placeholders(self) -> None:
        pipeline = importlib.import_module("run_real_reconstruction_pipeline")

        command = pipeline.build_pipeline_command(
            "python3 make.py --img {img_dir} --tsai {tsai_dir} --out {output_dir} --root {plapoint_root}",
            image_dir=Path("/data/img"),
            camera_dir=Path("/data/tsai"),
            output_dir=Path("/tmp/out"),
            root=Path("/repo/plapoint"),
        )

        self.assertEqual(
            command,
            [
                "python3",
                "make.py",
                "--img",
                "/data/img",
                "--tsai",
                "/data/tsai",
                "--out",
                "/tmp/out",
                "--root",
                "/repo/plapoint",
            ],
        )
        self.assertEqual(
            pipeline.default_output_dir(Path("/repo/plapoint")),
            Path("/repo/plapoint/build/real_reconstruction_pipeline"),
        )


class CudaHotspotReportTest(unittest.TestCase):
    def test_selects_gpu_hotspots_and_cpu_ratio(self) -> None:
        hotspots = importlib.import_module("report_cuda_hotspots")

        rows = [
            {"benchmark": "cpu_knn", "status": "measured", "best_ms": 10.0},
            {"benchmark": "gpu_knn", "status": "measured", "best_ms": 2.0},
            {"benchmark": "gpu_icp", "status": "measured", "best_ms": 5.0},
            {"benchmark": "cpu_voxel_grid", "status": "measured", "best_ms": 3.0},
        ]

        selected = hotspots.select_hotspots(rows, limit=2)
        markdown = hotspots.markdown_report(selected)

        self.assertEqual([row.name for row in selected], ["gpu_icp", "gpu_knn"])
        self.assertIsNone(selected[0].cpu_to_gpu_ratio)
        self.assertAlmostEqual(selected[1].cpu_to_gpu_ratio, 5.0)
        self.assertIn("gpu_icp", markdown)
        self.assertIn("gpu_knn", markdown)


class CpuOnlyValidationTest(unittest.TestCase):
    def test_cmake_configure_command_disables_cuda(self) -> None:
        command = cpu_validation.cmake_configure_command(
            source_dir=Path("/repo"),
            build_dir=Path("/repo/build-cpu"),
            generator="Ninja",
            extra_cmake_args=["-DCMAKE_BUILD_TYPE=Release"],
            prefix_paths=[Path("/deps/plamatrix")],
        )

        self.assertIn("-DPLAPOINT_WITH_CUDA=OFF", command)
        self.assertIn("-DPLAPOINT_BUILD_TESTS=ON", command)
        self.assertIn("-DPLAPOINT_BUILD_BENCHMARKS=ON", command)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", command)
        self.assertIn("-DCMAKE_PREFIX_PATH=/deps/plamatrix", command)
        self.assertIn("-Dplamatrix_DIR=/deps/plamatrix/lib/cmake/plamatrix", command)

    def test_default_prefix_path_finds_sibling_plamatrix_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plapoint = root / "plapoint"
            plamatrix_prefix = root / "plamatrix" / "build-task-c" / "install"
            plamatrix_config = plamatrix_prefix / "lib" / "cmake" / "plamatrix" / "plamatrixConfig.cmake"
            plamatrix_targets = plamatrix_config.parent / "plamatrixTargets.cmake"
            plapoint.mkdir()
            plamatrix_config.parent.mkdir(parents=True)
            plamatrix_config.write_text("# config\n", encoding="utf-8")
            plamatrix_targets.write_text("# targets\n", encoding="utf-8")

            prefixes = cpu_validation.default_cmake_prefix_paths(plapoint)

        self.assertEqual(prefixes, [plamatrix_prefix])


if __name__ == "__main__":
    unittest.main(verbosity=2)
