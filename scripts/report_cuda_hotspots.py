#!/usr/bin/env python3
"""Summarize the slowest CUDA benchmark rows from PlaPoint benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HotspotRow:
    name: str
    current_ms: float
    cpu_to_gpu_ratio: float | None


def measured_time(row: dict[str, object]) -> float | None:
    if row.get("status") != "measured":
        return None
    best_ms = row.get("best_ms", row.get("current_ms"))
    if isinstance(best_ms, (int, float)):
        return float(best_ms)
    return None


def select_hotspots(rows: list[dict[str, object]], *, limit: int = 10) -> list[HotspotRow]:
    measured: dict[str, float] = {}
    for row in rows:
        name = row.get("benchmark")
        if not isinstance(name, str):
            continue
        time_ms = measured_time(row)
        if time_ms is not None:
            measured[name] = time_ms

    hotspots: list[HotspotRow] = []
    for name, current_ms in measured.items():
        if not name.startswith("gpu_"):
            continue
        cpu_name = "cpu_" + name[len("gpu_") :]
        cpu_ms = measured.get(cpu_name)
        ratio = cpu_ms / current_ms if cpu_ms is not None and current_ms > 0 else None
        hotspots.append(HotspotRow(name, current_ms, ratio))

    hotspots.sort(key=lambda row: row.current_ms, reverse=True)
    return hotspots[:limit]


def load_rows(path: Path) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: missing JSON list field 'rows'")
    return rows


def rows_to_json(rows: list[HotspotRow]) -> dict[str, object]:
    return {
        "rows": [
            {
                "benchmark": row.name,
                "current_ms": row.current_ms,
                "cpu_to_gpu_ratio": row.cpu_to_gpu_ratio,
            }
            for row in rows
        ]
    }


def markdown_report(rows: list[HotspotRow]) -> str:
    lines = [
        "# PlaPoint CUDA Hotspots",
        "",
        "| Benchmark | Current ms | CPU/GPU ratio |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        ratio = "" if row.cpu_to_gpu_ratio is None else f"{row.cpu_to_gpu_ratio:.3f}"
        lines.append(f"| `{row.name}` | {row.current_ms:.6f} | {ratio} |")
    return "\n".join(lines) + "\n"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report slow CUDA benchmark rows from a PlaPoint benchmark JSON.")
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("--limit", type=positive_int, default=10)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        rows = select_hotspots(load_rows(args.benchmark_json), limit=args.limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CUDA hotspot report failed: {exc}", file=sys.stderr)
        return 2

    markdown = markdown_report(rows)
    print(markdown, end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows_to_json(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
