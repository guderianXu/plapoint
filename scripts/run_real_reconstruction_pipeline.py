#!/usr/bin/env python3
"""Run a real img/tsai reconstruction command and optionally verify PlaPoint references."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import run_real_reconstruction_regression as regression


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir(root: Path) -> Path:
    return root / "build" / "real_reconstruction_pipeline"


def resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def command_path(path: Path) -> str:
    return path.as_posix()


def build_pipeline_command(
    command_template: str,
    *,
    image_dir: Path,
    camera_dir: Path,
    output_dir: Path,
    root: Path,
) -> list[str]:
    command = command_template.format(
        img_dir=command_path(image_dir),
        tsai_dir=command_path(camera_dir),
        output_dir=command_path(output_dir),
        plapoint_root=command_path(root),
    )
    return shlex.split(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real reconstruction command from img/tsai inputs and verify generated PLYs."
    )
    parser.add_argument(
        "--command-template",
        required=True,
        help="Command with {img_dir}, {tsai_dir}, {output_dir}, and {plapoint_root} placeholders.",
    )
    parser.add_argument("--source-img-dir", type=Path)
    parser.add_argument("--source-tsai-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--actual-layout", choices=["reference", "plascan-legacy"], default="reference")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--quality-json-output", type=Path)
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--max-mean-error", type=float)
    parser.add_argument("--min-finite-ratio", type=float, default=1.0)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()
    source_root = regression.default_source_data_root(root)
    image_dir = resolve_path(args.source_img_dir, root) if args.source_img_dir else source_root / "img"
    camera_dir = resolve_path(args.source_tsai_dir, root) if args.source_tsai_dir else source_root / "tsai"
    output_dir = resolve_path(args.output_dir, root) if args.output_dir else default_output_dir(root)

    source_failures = regression.validate_source_inputs(image_dir, camera_dir)
    if source_failures:
        for failure in source_failures:
            print(failure, file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_pipeline_command(
        args.command_template,
        image_dir=image_dir,
        camera_dir=camera_dir,
        output_dir=output_dir,
        root=root,
    )
    print(shlex.join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if args.skip_regression:
        return 0

    regression_args = [
        "--source-img-dir",
        str(image_dir),
        "--source-tsai-dir",
        str(camera_dir),
        "--generated-root",
        str(output_dir),
        "--actual-layout",
        args.actual_layout,
        "--min-finite-ratio",
        str(args.min_finite_ratio),
    ]
    if args.json_output is not None:
        regression_args.extend(["--json-output", str(args.json_output)])
    if args.quality_json_output is not None:
        regression_args.extend(["--quality-json-output", str(args.quality_json_output)])
    if args.max_error is not None:
        regression_args.extend(["--max-error", str(args.max_error)])
    if args.max_mean_error is not None:
        regression_args.extend(["--max-mean-error", str(args.max_mean_error)])

    return regression.main(regression_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
