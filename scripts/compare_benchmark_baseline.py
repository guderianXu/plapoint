#!/usr/bin/env python3
"""Compare two PlaPoint benchmark baseline JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonRow:
    name: str
    status: str
    baseline_ms: float | None
    current_ms: float | None
    ratio: float | None
    delta_ms: float | None


@dataclass(frozen=True)
class ComparisonResult:
    rows: list[ComparisonRow]
    regression_threshold: float
    improvement_threshold: float
    min_ms: float

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

    def exit_code(self, *, fail_on_regression: bool) -> int:
        if fail_on_regression and self.regression_count:
            return 1
        return 0


def measured_rows(path: Path) -> dict[str, float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing JSON list field 'rows'")

    measured: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row is not an object")
        if row.get("status") != "measured":
            continue
        name = row.get("benchmark")
        best_ms = row.get("best_ms")
        if not isinstance(name, str):
            raise ValueError(f"{path}: measured row is missing benchmark name")
        if not isinstance(best_ms, (int, float)):
            raise ValueError(f"{path}: measured row '{name}' is missing numeric best_ms")
        measured[name] = float(best_ms)
    return measured


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
    regression_threshold: float = 0.20,
    improvement_threshold: float = 0.20,
    min_ms: float = 0.001,
) -> ComparisonResult:
    baseline = measured_rows(baseline_path)
    current = measured_rows(current_path)

    rows: list[ComparisonRow] = []
    for name in sorted(set(baseline) | set(current)):
        baseline_ms = baseline.get(name)
        current_ms = current.get(name)
        if baseline_ms is None:
            rows.append(ComparisonRow(name, "added", None, current_ms, None, None))
            continue
        if current_ms is None:
            rows.append(ComparisonRow(name, "missing", baseline_ms, None, None, None))
            continue
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
            )
        )

    return ComparisonResult(
        rows=rows,
        regression_threshold=regression_threshold,
        improvement_threshold=improvement_threshold,
        min_ms=min_ms,
    )


def write_json_report(path: Path, result: ComparisonResult) -> None:
    payload = {
        "regression_threshold": result.regression_threshold,
        "improvement_threshold": result.improvement_threshold,
        "min_ms": result.min_ms,
        "regression_count": result.regression_count,
        "improvement_count": result.improvement_count,
        "missing_count": result.missing_count,
        "added_count": result.added_count,
        "rows": [
            {
                "benchmark": row.name,
                "status": row.status,
                "baseline_ms": row.baseline_ms,
                "current_ms": row.current_ms,
                "ratio": row.ratio,
                "delta_ms": row.delta_ms,
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
        f"- Regressions: `{result.regression_count}`",
        f"- Improvements: `{result.improvement_count}`",
        f"- Missing rows: `{result.missing_count}`",
        f"- Added rows: `{result.added_count}`",
        "",
        "| Benchmark | Status | Baseline ms | Current ms | Ratio | Delta ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result.rows:
        baseline = "" if row.baseline_ms is None else f"{row.baseline_ms:.6f}"
        current = "" if row.current_ms is None else f"{row.current_ms:.6f}"
        ratio = "" if row.ratio is None else f"{row.ratio:.3f}"
        delta = "" if row.delta_ms is None else f"{row.delta_ms:.6f}"
        lines.append(f"| `{row.name}` | {row.status} | {baseline} | {current} | {ratio} | {delta} |")
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
    parser.add_argument("--regression-threshold", type=non_negative_float, default=0.20)
    parser.add_argument("--improvement-threshold", type=non_negative_float, default=0.20)
    parser.add_argument("--min-ms", type=non_negative_float, default=0.001)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = compare_files(
            args.baseline_json,
            args.current_json,
            regression_threshold=args.regression_threshold,
            improvement_threshold=args.improvement_threshold,
            min_ms=args.min_ms,
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

    return result.exit_code(fail_on_regression=args.fail_on_regression)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
