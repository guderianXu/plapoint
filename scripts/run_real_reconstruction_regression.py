#!/usr/bin/env python3
"""Validate real reconstruction inputs and compare generated PLYs against PlaPoint references."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlyStats:
    point_count: int
    mean: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    intensity_mean: float | None


@dataclass(frozen=True)
class TreeComparison:
    ok: bool
    compared_files: int
    failures: list[str]
    stats: dict[str, dict[str, object]]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_source_data_root(root: Path) -> Path:
    return root.parent.parent / "testData"


def default_reference_paths(reference_root: Path) -> list[str]:
    return sorted(path.relative_to(reference_root).as_posix() for path in reference_root.rglob("*.ply"))


def resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def parse_ascii_ply_stats(path: Path) -> PlyStats:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"{path}: invalid PLY magic")

        vertex_count: int | None = None
        properties: list[str] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: missing end_header")
            stripped = line.strip()
            if stripped == "end_header":
                break
            parts = stripped.split()
            if parts[:2] == ["format", "ascii"]:
                continue
            if parts[:2] == ["element", "vertex"] and len(parts) == 3:
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if parts[:1] == ["element"]:
                in_vertex = False
                continue
            if in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[-1])

        if vertex_count is None:
            raise ValueError(f"{path}: missing vertex element")
        for required in ("x", "y", "z"):
            if required not in properties:
                raise ValueError(f"{path}: missing property {required}")

        x_index = properties.index("x")
        y_index = properties.index("y")
        z_index = properties.index("z")
        intensity_index = properties.index("intensity") if "intensity" in properties else None

        sums = [0.0, 0.0, 0.0]
        mins = [math.inf, math.inf, math.inf]
        maxs = [-math.inf, -math.inf, -math.inf]
        intensity_sum = 0.0
        for row_index in range(vertex_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: truncated at vertex row {row_index}")
            values = line.split()
            if len(values) < len(properties):
                raise ValueError(f"{path}: malformed vertex row {row_index}")
            coords = [float(values[x_index]), float(values[y_index]), float(values[z_index])]
            for axis, value in enumerate(coords):
                sums[axis] += value
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
            if intensity_index is not None:
                intensity_sum += float(values[intensity_index])

    if vertex_count == 0:
        mean = (0.0, 0.0, 0.0)
        bbox_min = (0.0, 0.0, 0.0)
        bbox_max = (0.0, 0.0, 0.0)
    else:
        mean = tuple(value / vertex_count for value in sums)
        bbox_min = tuple(mins)
        bbox_max = tuple(maxs)
    intensity_mean = intensity_sum / vertex_count if intensity_index is not None and vertex_count else None
    return PlyStats(vertex_count, mean, bbox_min, bbox_max, intensity_mean)


def parse_ascii_ply_quality_metrics(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"{path}: invalid PLY magic")

        vertex_count: int | None = None
        properties: list[str] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: missing end_header")
            stripped = line.strip()
            if stripped == "end_header":
                break
            parts = stripped.split()
            if parts[:2] == ["format", "ascii"]:
                continue
            if parts[:2] == ["element", "vertex"] and len(parts) == 3:
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if parts[:1] == ["element"]:
                in_vertex = False
                continue
            if in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[-1])

        if vertex_count is None:
            raise ValueError(f"{path}: missing vertex element")
        for required in ("x", "y", "z"):
            if required not in properties:
                raise ValueError(f"{path}: missing property {required}")

        x_index = properties.index("x")
        y_index = properties.index("y")
        z_index = properties.index("z")
        error_index = properties.index("error") if "error" in properties else None
        intensity_index = properties.index("intensity") if "intensity" in properties else None

        finite_point_count = 0
        error_count = 0
        error_sum = 0.0
        error_max: float | None = None
        intensity_count = 0
        intensity_sum = 0.0
        intensity_min: float | None = None
        intensity_max: float | None = None

        for row_index in range(vertex_count):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: truncated at vertex row {row_index}")
            values = line.split()
            if len(values) < len(properties):
                raise ValueError(f"{path}: malformed vertex row {row_index}")

            coords = [float(values[x_index]), float(values[y_index]), float(values[z_index])]
            if all(math.isfinite(value) for value in coords):
                finite_point_count += 1

            if error_index is not None:
                error = float(values[error_index])
                error_count += 1
                error_sum += error
                error_max = error if error_max is None else max(error_max, error)

            if intensity_index is not None:
                intensity = float(values[intensity_index])
                intensity_count += 1
                intensity_sum += intensity
                intensity_min = intensity if intensity_min is None else min(intensity_min, intensity)
                intensity_max = intensity if intensity_max is None else max(intensity_max, intensity)

    finite_ratio = 1.0 if vertex_count == 0 else finite_point_count / vertex_count
    return {
        "point_count": vertex_count,
        "finite_coordinate_ratio": finite_ratio,
        "error_mean": error_sum / error_count if error_count else None,
        "error_max": error_max,
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "intensity_mean": intensity_sum / intensity_count if intensity_count else None,
    }


def evaluate_quality_metrics(
    actual_root: Path,
    relative_paths: list[str] | None = None,
    *,
    actual_layout: str = "reference",
) -> dict[str, dict[str, object]]:
    if relative_paths is None:
        relative_paths = default_reference_paths(actual_root)

    metrics: dict[str, dict[str, object]] = {}
    for relative_path in relative_paths:
        path = actual_ply_path(actual_root, relative_path, actual_layout)
        if path.exists():
            metrics[relative_path] = parse_ascii_ply_quality_metrics(path)
    return metrics


def check_quality_thresholds(
    metrics: dict[str, dict[str, object]],
    *,
    max_error: float | None = None,
    max_mean_error: float | None = None,
    min_finite_ratio: float = 1.0,
) -> list[str]:
    failures: list[str] = []
    for relative_path, metric in sorted(metrics.items()):
        finite_ratio = float(metric["finite_coordinate_ratio"])
        if finite_ratio < min_finite_ratio:
            failures.append(
                f"{relative_path}: finite coordinate ratio {finite_ratio:.6f} "
                f"is below {min_finite_ratio:.6f}"
            )

        error_max = metric.get("error_max")
        if max_error is not None:
            if error_max is None:
                failures.append(f"{relative_path}: missing error property for max error threshold")
            elif float(error_max) > max_error:
                failures.append(f"{relative_path}: max error {float(error_max):.6f} exceeds {max_error:.6f}")

        error_mean = metric.get("error_mean")
        if max_mean_error is not None:
            if error_mean is None:
                failures.append(f"{relative_path}: missing error property for mean error threshold")
            elif float(error_mean) > max_mean_error:
                failures.append(
                    f"{relative_path}: mean error {float(error_mean):.6f} exceeds {max_mean_error:.6f}"
                )
    return failures


def max_tuple_delta(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def actual_ply_path(actual_root: Path, relative_path: str, actual_layout: str) -> Path:
    direct = actual_root / relative_path
    if actual_layout == "reference" or direct.exists():
        return direct
    if actual_layout == "plascan-legacy" and relative_path.startswith("merged/"):
        return actual_root / Path(relative_path).name
    return direct


def compare_reference_tree(
    reference_root: Path,
    actual_root: Path,
    relative_paths: list[str],
    *,
    point_tolerance: int,
    mean_tolerance: float,
    bbox_tolerance: float,
    intensity_mean_tolerance: float,
    actual_layout: str = "reference",
) -> TreeComparison:
    failures: list[str] = []
    stats: dict[str, dict[str, object]] = {}
    compared_files = 0
    for relative_path in relative_paths:
        expected_path = reference_root / relative_path
        actual_path = actual_ply_path(actual_root, relative_path, actual_layout)
        if not expected_path.exists():
            failures.append(f"missing reference PLY: {relative_path}")
            continue
        if not actual_path.exists():
            failures.append(f"missing actual PLY: {relative_path}")
            continue

        expected = parse_ascii_ply_stats(expected_path)
        actual = parse_ascii_ply_stats(actual_path)
        compared_files += 1
        stats[relative_path] = {
            "expected_points": expected.point_count,
            "actual_points": actual.point_count,
            "mean_delta": max_tuple_delta(expected.mean, actual.mean),
            "bbox_min_delta": max_tuple_delta(expected.bbox_min, actual.bbox_min),
            "bbox_max_delta": max_tuple_delta(expected.bbox_max, actual.bbox_max),
            "expected_intensity_mean": expected.intensity_mean,
            "actual_intensity_mean": actual.intensity_mean,
        }

        if abs(expected.point_count - actual.point_count) > point_tolerance:
            failures.append(
                f"{relative_path}: point count expected {expected.point_count}, got {actual.point_count}"
            )
        if max_tuple_delta(expected.mean, actual.mean) > mean_tolerance:
            failures.append(f"{relative_path}: mean drift exceeds tolerance")
        if max_tuple_delta(expected.bbox_min, actual.bbox_min) > bbox_tolerance:
            failures.append(f"{relative_path}: bbox min drift exceeds tolerance")
        if max_tuple_delta(expected.bbox_max, actual.bbox_max) > bbox_tolerance:
            failures.append(f"{relative_path}: bbox max drift exceeds tolerance")
        if expected.intensity_mean is not None or actual.intensity_mean is not None:
            if expected.intensity_mean is None or actual.intensity_mean is None:
                failures.append(f"{relative_path}: intensity property presence changed")
            elif abs(expected.intensity_mean - actual.intensity_mean) > intensity_mean_tolerance:
                failures.append(f"{relative_path}: intensity mean drift exceeds tolerance")

    return TreeComparison(
        ok=not failures,
        compared_files=compared_files,
        failures=failures,
        stats=stats,
    )


def validate_source_inputs(image_dir: Path, camera_dir: Path) -> list[str]:
    failures: list[str] = []
    for directory in (image_dir, camera_dir):
        if not directory.is_dir():
            failures.append(f"missing input directory: {directory}")

    expected_images = [f"{index}.png" for index in range(1, 6)]
    expected_cameras = [f"{index}.tsai" for index in range(1, 6)]
    for name in expected_images:
        if not (image_dir / name).is_file():
            failures.append(f"missing source image: {image_dir / name}")
    for name in expected_cameras:
        if not (camera_dir / name).is_file():
            failures.append(f"missing source camera: {camera_dir / name}")
    return failures


def run_pipeline_command(command_template: str, *, image_dir: Path, camera_dir: Path, output_dir: Path, root: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_template.format(
        img_dir=str(image_dir),
        tsai_dir=str(camera_dir),
        output_dir=str(output_dir),
        plapoint_root=str(root),
    )
    print(command)
    return subprocess.run(shlex.split(command), check=False).returncode


def write_quality_report(
    path: Path,
    metrics: dict[str, dict[str, object]],
    quality_failures: list[str],
) -> None:
    payload = {
        "ok": not quality_failures,
        "quality_failures": quality_failures,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    result: TreeComparison,
    source_failures: list[str],
    quality_metrics: dict[str, dict[str, object]] | None = None,
    quality_failures: list[str] | None = None,
) -> None:
    if quality_failures is None:
        quality_failures = []
    payload = {
        "ok": result.ok and not source_failures and not quality_failures,
        "compared_files": result.compared_files,
        "source_failures": source_failures,
        "comparison_failures": result.failures,
        "quality_failures": quality_failures,
        "stats": result.stats,
    }
    if quality_metrics is not None:
        payload["quality_metrics"] = quality_metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify the PlaPoint real reconstruction regression data.")
    parser.add_argument("--source-img-dir", type=Path)
    parser.add_argument("--source-tsai-dir", type=Path)
    parser.add_argument("--reference-root", type=Path, default=Path("testData/real_reconstruction"))
    parser.add_argument("--generated-root", type=Path)
    parser.add_argument("--actual-layout", choices=["reference", "plascan-legacy"], default="reference")
    parser.add_argument("--output-dir", type=Path, default=Path("build/real_reconstruction_regression"))
    parser.add_argument(
        "--pipeline-command",
        default="",
        help="Optional command template. Placeholders: {img_dir}, {tsai_dir}, {output_dir}, {plapoint_root}.",
    )
    parser.add_argument("--point-tolerance", type=int, default=0)
    parser.add_argument("--mean-tolerance", type=float, default=0.0)
    parser.add_argument("--bbox-tolerance", type=float, default=0.0)
    parser.add_argument("--intensity-mean-tolerance", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--quality-json-output", type=Path)
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--max-mean-error", type=float)
    parser.add_argument("--min-finite-ratio", type=float, default=1.0)
    parser.add_argument("--require-generated", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()
    source_root = default_source_data_root(root)
    image_dir = resolve_path(args.source_img_dir, root) if args.source_img_dir else source_root / "img"
    camera_dir = resolve_path(args.source_tsai_dir, root) if args.source_tsai_dir else source_root / "tsai"
    reference_root = resolve_path(args.reference_root, root)
    output_dir = resolve_path(args.output_dir, root)

    source_failures = validate_source_inputs(image_dir, camera_dir)
    if source_failures:
        for failure in source_failures:
            print(failure, file=sys.stderr)
        return 2

    if args.pipeline_command:
        result = run_pipeline_command(
            args.pipeline_command,
            image_dir=image_dir,
            camera_dir=camera_dir,
            output_dir=output_dir,
            root=root,
        )
        if result != 0:
            return result

    if args.generated_root is not None:
        actual_root = resolve_path(args.generated_root, root)
    elif args.pipeline_command:
        actual_root = output_dir
    elif args.require_generated:
        print("--generated-root or --pipeline-command is required with --require-generated", file=sys.stderr)
        return 2
    else:
        actual_root = reference_root

    relative_paths = default_reference_paths(reference_root)
    comparison = compare_reference_tree(
        reference_root,
        actual_root,
        relative_paths,
        point_tolerance=args.point_tolerance,
        mean_tolerance=args.mean_tolerance,
        bbox_tolerance=args.bbox_tolerance,
        intensity_mean_tolerance=args.intensity_mean_tolerance,
        actual_layout=args.actual_layout,
    )
    quality_metrics = evaluate_quality_metrics(actual_root, relative_paths, actual_layout=args.actual_layout)
    quality_failures = check_quality_thresholds(
        quality_metrics,
        max_error=args.max_error,
        max_mean_error=args.max_mean_error,
        min_finite_ratio=args.min_finite_ratio,
    )

    report_path = resolve_path(args.json_output, root) if args.json_output else output_dir / "comparison.json"
    write_report(report_path, comparison, source_failures, quality_metrics, quality_failures)
    if args.quality_json_output is not None:
        write_quality_report(resolve_path(args.quality_json_output, root), quality_metrics, quality_failures)
    print(f"Compared {comparison.compared_files} PLY files")
    print(f"Wrote report: {report_path}")
    for failure in comparison.failures:
        print(failure, file=sys.stderr)
    for failure in quality_failures:
        print(failure, file=sys.stderr)
    return 0 if comparison.ok and not quality_failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
