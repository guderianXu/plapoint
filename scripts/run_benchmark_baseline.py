#!/usr/bin/env python3
"""Run PlaPoint benchmarks and write reproducible baseline artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def resolve_user_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path.resolve()
    return (root / path).resolve()


def find_benchmark_exe(root: Path) -> Path:
    preferred = [
        root / "build-abc-cuda" / "benchmarks" / "plapoint_benchmarks",
        root / "build-abc-cpu" / "benchmarks" / "plapoint_benchmarks",
        root / "build-cuda" / "benchmarks" / "plapoint_benchmarks",
        root / "build-cpu" / "benchmarks" / "plapoint_benchmarks",
        root / "build" / "benchmarks" / "plapoint_benchmarks",
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    for candidate in sorted(root.glob("build*/benchmarks/plapoint_benchmarks")):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "could not find plapoint_benchmarks; build it first or pass --benchmark-exe"
    )


def parse_benchmark_csv(stdout: str) -> list[dict[str, object]]:
    reader = csv.reader(StringIO(stdout))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("benchmark produced no CSV output") from exc

    expected_header = ["benchmark", "points", "iterations", "best_ms"]
    if header != expected_header:
        raise ValueError(f"unexpected CSV header {header!r}, expected {expected_header!r}")

    rows: list[dict[str, object]] = []
    for line_number, row in enumerate(reader, start=2):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError(f"line {line_number}: expected 4 CSV columns, got {len(row)}")

        benchmark, points, iterations, best_ms = row
        if points == "skipped":
            rows.append(
                {
                    "benchmark": benchmark,
                    "status": "skipped",
                    "reason": iterations,
                }
            )
            continue

        rows.append(
            {
                "benchmark": benchmark,
                "status": "measured",
                "points": int(points),
                "iterations": int(iterations),
                "best_ms": float(best_ms),
            }
        )
    return rows


def write_markdown(path: Path, metadata: dict[str, object], rows: list[dict[str, object]]) -> None:
    command = metadata["command"]
    if not isinstance(command, list):
        raise TypeError("metadata command must be a list")

    lines = [
        "# PlaPoint Benchmark Baseline",
        "",
        f"- Generated UTC: `{metadata['generated_at_utc']}`",
        f"- Benchmark executable: `{metadata['benchmark_exe']}`",
        f"- Command: `{shlex.join(str(part) for part in command)}`",
        "",
        "| Benchmark | Status | Points | Iterations | Best ms | Reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        if row["status"] == "skipped":
            lines.append(
                f"| `{row['benchmark']}` | skipped |  |  |  | {row.get('reason', '')} |"
            )
        else:
            lines.append(
                f"| `{row['benchmark']}` | measured | {row['points']} | "
                f"{row['iterations']} | {row['best_ms']:.6f} |  |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PlaPoint benchmark executable and store baseline artifacts."
    )
    parser.add_argument(
        "--benchmark-exe",
        type=Path,
        help="Path to plapoint_benchmarks; auto-detected from build*/ when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for CSV/JSON/Markdown artifacts. Defaults to build/benchmark_baseline.",
    )
    parser.add_argument("--points", type=positive_int, default=1000)
    parser.add_argument("--iterations", type=positive_int, default=1)
    parser.add_argument("--icp-points", type=positive_int, default=1000)
    parser.add_argument("--icp-max-iterations", type=positive_int, default=1)
    parser.add_argument("--skip-cpu-icp", action="store_true")
    parser.add_argument("--skip-icp-identity", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=non_negative_int,
        default=0,
        help="Abort the benchmark after this many seconds; 0 disables the timeout.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed directly to plapoint_benchmarks.",
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()

    if args.benchmark_exe is None:
        benchmark_exe = find_benchmark_exe(root)
    else:
        benchmark_exe = resolve_user_path(args.benchmark_exe, root)
        if not benchmark_exe.is_file():
            parser.error(f"benchmark executable not found: {benchmark_exe}")

    if args.output_dir is None:
        output_dir = root / "build" / "benchmark_baseline"
    else:
        output_dir = resolve_user_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = [
        str(benchmark_exe),
        "--points",
        str(args.points),
        "--iterations",
        str(args.iterations),
        "--icp-points",
        str(args.icp_points),
        "--icp-max-iterations",
        str(args.icp_max_iterations),
    ]
    if args.skip_cpu_icp:
        command.append("--skip-cpu-icp")
    if args.skip_icp_identity:
        command.append("--skip-icp-identity")
    command.extend(extra_args)

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds if args.timeout_seconds else None,
        )
    except subprocess.TimeoutExpired as exc:
        stderr_path = output_dir / "plapoint_benchmark_baseline.stderr.txt"
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        print(
            f"Benchmark timed out after {args.timeout_seconds}s; stderr written to {stderr_path}",
            file=sys.stderr,
        )
        return 124

    csv_path = output_dir / "plapoint_benchmark_baseline.csv"
    json_path = output_dir / "plapoint_benchmark_baseline.json"
    markdown_path = output_dir / "plapoint_benchmark_baseline.md"
    stderr_path = output_dir / "plapoint_benchmark_baseline.stderr.txt"

    csv_path.write_text(completed.stdout, encoding="utf-8")
    if completed.stderr:
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    elif stderr_path.exists():
        stderr_path.unlink()

    parse_error: str | None = None
    rows: list[dict[str, object]] = []
    try:
        rows = parse_benchmark_csv(completed.stdout)
    except (TypeError, ValueError) as exc:
        parse_error = str(exc)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "benchmark_exe": str(benchmark_exe),
        "command": command,
        "returncode": completed.returncode,
        "parameters": {
            "points": args.points,
            "iterations": args.iterations,
            "icp_points": args.icp_points,
            "icp_max_iterations": args.icp_max_iterations,
            "skip_cpu_icp": args.skip_cpu_icp,
            "skip_icp_identity": args.skip_icp_identity,
            "extra_args": extra_args,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(markdown_path),
            "stderr": str(stderr_path) if completed.stderr else None,
        },
        "rows": rows,
        "stderr": completed.stderr,
    }
    if parse_error is not None:
        metadata["parse_error"] = parse_error

    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if parse_error is None:
        write_markdown(markdown_path, metadata, rows)

    measured = sum(1 for row in rows if row.get("status") == "measured")
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    print(f"Wrote benchmark CSV: {csv_path}")
    print(f"Wrote benchmark JSON: {json_path}")
    if parse_error is None:
        print(f"Wrote benchmark Markdown: {markdown_path}")
        print(f"Rows: measured={measured}, skipped={skipped}")
    else:
        print(f"Could not parse benchmark CSV: {parse_error}", file=sys.stderr)

    if completed.returncode != 0:
        return completed.returncode
    if parse_error is not None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
