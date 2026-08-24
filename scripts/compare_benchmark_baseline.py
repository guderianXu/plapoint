#!/usr/bin/env python3
"""Compare two PlaPoint benchmark baseline JSON files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REGRESSION_THRESHOLD = 0.20
DEFAULT_IMPROVEMENT_THRESHOLD = 0.20
DEFAULT_MIN_MS = 0.001
DEFAULT_PRIMARY_METRIC = "median_ms"
SUPPORTED_PRIMARY_METRICS = {"median_ms", "best_ms"}


@dataclass(frozen=True)
class GateConfig:
    regression_threshold: float
    improvement_threshold: float
    min_ms: float
    ignore_benchmarks: set[str]
    primary_metric: str = DEFAULT_PRIMARY_METRIC


@dataclass(frozen=True)
class BenchmarkMeasurement:
    best_ms: float
    median_ms: float | None


@dataclass(frozen=True)
class ComparisonRow:
    name: str
    status: str
    baseline_ms: float | None
    current_ms: float | None
    ratio: float | None
    delta_ms: float | None
    metric: str


@dataclass(frozen=True)
class ComparisonResult:
    rows: list[ComparisonRow]
    regression_threshold: float
    improvement_threshold: float
    min_ms: float
    primary_metric: str

    @property
    def regression_count(self) -> int:
        return sum(1 for row in self.rows if row.status == "regressed")

    @property
    def improvement_count(self) -> int:
        return sum(1 for row in self.rows if row.status == "improved")

    @property
    def missing_count(self) -> int:
        return sum(1 for row in self.rows if row.status == "missing")

    @property
    def added_count(self) -> int:
        return sum(1 for row in self.rows if row.status == "added")

    @property
    def ignored_count(self) -> int:
        return sum(1 for row in self.rows if row.status == "ignored")

    def exit_code(self, *, fail_on_regression: bool, fail_on_missing: bool = False) -> int:
        if fail_on_regression and self.regression_count:
            return 1
        if fail_on_missing and self.missing_count:
            return 1
        return 0


def load_gate_config(path: Path) -> GateConfig:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: gate config must be a JSON object")

    regression_threshold = document.get("regression_threshold", DEFAULT_REGRESSION_THRESHOLD)
    improvement_threshold = document.get("improvement_threshold", DEFAULT_IMPROVEMENT_THRESHOLD)
    min_ms = document.get("min_ms", DEFAULT_MIN_MS)
    ignore_benchmarks = document.get("ignore_benchmarks", [])
    primary_metric = document.get("primary_metric", DEFAULT_PRIMARY_METRIC)
    if (
        isinstance(regression_threshold, bool)
        or not isinstance(regression_threshold, (int, float))
        or not math.isfinite(regression_threshold)
        or regression_threshold < 0
    ):
        raise ValueError(f"{path}: regression_threshold must be a non-negative number")
    if (
        isinstance(improvement_threshold, bool)
        or not isinstance(improvement_threshold, (int, float))
        or not math.isfinite(improvement_threshold)
        or improvement_threshold < 0
    ):
        raise ValueError(f"{path}: improvement_threshold must be a non-negative number")
    if (
        isinstance(min_ms, bool)
        or not isinstance(min_ms, (int, float))
        or not math.isfinite(min_ms)
        or min_ms < 0
    ):
        raise ValueError(f"{path}: min_ms must be a non-negative number")
    if not isinstance(ignore_benchmarks, list) or not all(isinstance(name, str) for name in ignore_benchmarks):
        raise ValueError(f"{path}: ignore_benchmarks must be a list of strings")
    if not isinstance(primary_metric, str) or primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(
            f"{path}: primary_metric must be one of {sorted(SUPPORTED_PRIMARY_METRICS)}"
        )

    return GateConfig(
        regression_threshold=float(regression_threshold),
        improvement_threshold=float(improvement_threshold),
        min_ms=float(min_ms),
        ignore_benchmarks=set(ignore_benchmarks),
        primary_metric=primary_metric,
    )


def measured_rows(path: Path) -> dict[str, BenchmarkMeasurement]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing JSON list field 'rows'")

    measured: dict[str, BenchmarkMeasurement] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row is not an object")
        if row.get("status") != "measured":
            continue
        name = row.get("benchmark")
        best_ms = row.get("best_ms")
        median_ms = row.get("median_ms")
        if not isinstance(name, str):
            raise ValueError(f"{path}: measured row is missing benchmark name")
        if isinstance(best_ms, bool) or not isinstance(best_ms, (int, float)):
            raise ValueError(f"{path}: measured row '{name}' is missing numeric best_ms")
        if not math.isfinite(best_ms) or best_ms < 0:
            raise ValueError(f"{path}: measured row '{name}' has invalid best_ms")
        if median_ms is not None:
            if isinstance(median_ms, bool) or not isinstance(median_ms, (int, float)):
                raise ValueError(f"{path}: measured row '{name}' has non-numeric median_ms")
            if not math.isfinite(median_ms) or median_ms < 0:
                raise ValueError(f"{path}: measured row '{name}' has invalid median_ms")
        measured[name] = BenchmarkMeasurement(
            best_ms=float(best_ms),
            median_ms=None if median_ms is None else float(median_ms),
        )
    return measured


def select_common_metric(
    baseline: BenchmarkMeasurement,
    current: BenchmarkMeasurement,
    primary_metric: str,
) -> tuple[str, float, float]:
    if primary_metric == "median_ms" and baseline.median_ms is not None and current.median_ms is not None:
        return primary_metric, baseline.median_ms, current.median_ms
    return "best_ms", baseline.best_ms, current.best_ms


def select_available_metric(
    measurement: BenchmarkMeasurement,
    primary_metric: str,
) -> tuple[str, float]:
    if primary_metric == "median_ms" and measurement.median_ms is not None:
        return primary_metric, measurement.median_ms
    return "best_ms", measurement.best_ms


def classify_ratio(
    baseline_ms: float,
    current_ms: float,
    *,
    regression_threshold: float,
    improvement_threshold: float,
    min_ms: float,
) -> str:
    if baseline_ms < min_ms and current_ms < min_ms:
        return "unchanged"
    ratio = current_ms / baseline_ms if baseline_ms else float("inf")
    if ratio >= 1.0 + regression_threshold:
        return "regressed"
    if ratio <= 1.0 - improvement_threshold:
        return "improved"
    return "unchanged"


def compare_files(
    baseline_path: Path,
    current_path: Path,
    *,
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
    min_ms: float = DEFAULT_MIN_MS,
    ignored_benchmarks: set[str] | None = None,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> ComparisonResult:
    if not isinstance(primary_metric, str) or primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ValueError(f"unsupported primary metric: {primary_metric}")
    baseline = measured_rows(baseline_path)
    current = measured_rows(current_path)
    if ignored_benchmarks is None:
        ignored_benchmarks = set()

    rows: list[ComparisonRow] = []
    for name in sorted(set(baseline) | set(current)):
        baseline_measurement = baseline.get(name)
        current_measurement = current.get(name)
        if name in ignored_benchmarks:
            if baseline_measurement is not None and current_measurement is not None:
                metric, baseline_ms, current_ms = select_common_metric(
                    baseline_measurement,
                    current_measurement,
                    primary_metric,
                )
                ratio = current_ms / baseline_ms if baseline_ms else None
                delta_ms = current_ms - baseline_ms
            elif baseline_measurement is not None:
                metric, baseline_ms = select_available_metric(baseline_measurement, primary_metric)
                current_ms = None
                ratio = None
                delta_ms = None
            else:
                assert current_measurement is not None
                metric, current_ms = select_available_metric(current_measurement, primary_metric)
                baseline_ms = None
                ratio = None
                delta_ms = None
            rows.append(
                ComparisonRow(name, "ignored", baseline_ms, current_ms, ratio, delta_ms, metric)
            )
            continue

        if baseline_measurement is None:
            assert current_measurement is not None
            metric, current_ms = select_available_metric(current_measurement, primary_metric)
            rows.append(ComparisonRow(name, "added", None, current_ms, None, None, metric))
            continue
        if current_measurement is None:
            metric, baseline_ms = select_available_metric(baseline_measurement, primary_metric)
            rows.append(ComparisonRow(name, "missing", baseline_ms, None, None, None, metric))
            continue

        metric, baseline_ms, current_ms = select_common_metric(
            baseline_measurement,
            current_measurement,
            primary_metric,
        )
        ratio = current_ms / baseline_ms if baseline_ms else None
        delta_ms = current_ms - baseline_ms
        rows.append(
            ComparisonRow(
                name,
                classify_ratio(
                    baseline_ms,
                    current_ms,
                    regression_threshold=regression_threshold,
                    improvement_threshold=improvement_threshold,
                    min_ms=min_ms,
                ),
                baseline_ms,
                current_ms,
                ratio,
                delta_ms,
                metric,
            )
        )

    return ComparisonResult(
        rows=rows,
        regression_threshold=regression_threshold,
        improvement_threshold=improvement_threshold,
        min_ms=min_ms,
        primary_metric=primary_metric,
    )


def write_json_report(path: Path, result: ComparisonResult) -> None:
    payload = {
        "schema_version": 2,
        "regression_threshold": result.regression_threshold,
        "improvement_threshold": result.improvement_threshold,
        "min_ms": result.min_ms,
        "primary_metric": result.primary_metric,
        "regression_count": result.regression_count,
        "improvement_count": result.improvement_count,
        "missing_count": result.missing_count,
        "added_count": result.added_count,
        "ignored_count": result.ignored_count,
        "rows": [
            {
                "benchmark": row.name,
                "status": row.status,
                "baseline_ms": row.baseline_ms,
                "current_ms": row.current_ms,
                "ratio": row.ratio,
                "delta_ms": row.delta_ms,
                "metric": row.metric,
            }
            for row in result.rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(result: ComparisonResult) -> str:
    lines = [
        "# PlaPoint Benchmark Comparison",
        "",
        f"- Regression threshold: `{result.regression_threshold:.3f}`",
        f"- Improvement threshold: `{result.improvement_threshold:.3f}`",
        f"- Preferred metric: `{result.primary_metric}` (falls back to `best_ms` for legacy pairs)",
        f"- Regressions: `{result.regression_count}`",
        f"- Improvements: `{result.improvement_count}`",
        f"- Missing rows: `{result.missing_count}`",
        f"- Added rows: `{result.added_count}`",
        f"- Ignored rows: `{result.ignored_count}`",
        "",
        "| Benchmark | Status | Metric | Baseline ms | Current ms | Ratio | Delta ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        baseline = "" if row.baseline_ms is None else f"{row.baseline_ms:.6f}"
        current = "" if row.current_ms is None else f"{row.current_ms:.6f}"
        ratio = "" if row.ratio is None else f"{row.ratio:.3f}"
        delta = "" if row.delta_ms is None else f"{row.delta_ms:.6f}"
        lines.append(
            f"| `{row.name}` | {row.status} | `{row.metric}` | "
            f"{baseline} | {current} | {ratio} | {delta} |"
        )
    return "\n".join(lines) + "\n"


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two PlaPoint benchmark baseline JSON files.")
    parser.add_argument("baseline_json", type=Path)
    parser.add_argument("current_json", type=Path)
    parser.add_argument("--config", type=Path, help="JSON gate config with thresholds and ignored benchmarks.")
    parser.add_argument("--regression-threshold", type=non_negative_float)
    parser.add_argument("--improvement-threshold", type=non_negative_float)
    parser.add_argument("--min-ms", type=non_negative_float)
    parser.add_argument("--primary-metric", choices=sorted(SUPPORTED_PRIMARY_METRICS))
    parser.add_argument("--ignore-benchmark", action="append", default=[])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = (
            load_gate_config(args.config)
            if args.config is not None
            else GateConfig(
                DEFAULT_REGRESSION_THRESHOLD,
                DEFAULT_IMPROVEMENT_THRESHOLD,
                DEFAULT_MIN_MS,
                set(),
                DEFAULT_PRIMARY_METRIC,
            )
        )
        regression_threshold = (
            args.regression_threshold if args.regression_threshold is not None else config.regression_threshold
        )
        improvement_threshold = (
            args.improvement_threshold if args.improvement_threshold is not None else config.improvement_threshold
        )
        min_ms = args.min_ms if args.min_ms is not None else config.min_ms
        primary_metric = args.primary_metric if args.primary_metric is not None else config.primary_metric
        ignored_benchmarks = set(config.ignore_benchmarks)
        ignored_benchmarks.update(args.ignore_benchmark)
        result = compare_files(
            args.baseline_json,
            args.current_json,
            regression_threshold=regression_threshold,
            improvement_threshold=improvement_threshold,
            min_ms=min_ms,
            ignored_benchmarks=ignored_benchmarks,
            primary_metric=primary_metric,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark comparison failed: {exc}", file=sys.stderr)
        return 2

    report = markdown_table(result)
    print(report, end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_report(args.json_output, result)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")

    return result.exit_code(
        fail_on_regression=args.fail_on_regression,
        fail_on_missing=args.fail_on_missing,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
