#!/usr/bin/env python3
"""Build and run the PlaPoint mesh quality report tool."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_build_dir(root: Path) -> Path:
    preferred = root / "build-abc-cpu"
    if preferred.exists():
        return preferred
    return root / "build"


def tool_path(build_dir: Path) -> Path:
    return build_dir / "tools" / "plapoint_mesh_quality_report"


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Generate mesh quality metrics and PLY artifacts for PlaPoint."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build_dir(root),
        help="Configured CMake build directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PLY files and mesh_quality_report.json.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Run the existing report executable without invoking cmake --build.",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if not build_dir.exists():
        raise SystemExit(
            f"Build directory does not exist: {build_dir}\n"
            "Configure PlaPoint first, then rerun this script."
        )

    if not args.skip_build:
        build_command = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "plapoint_mesh_quality_report",
        ]
        subprocess.run(
            build_command,
            cwd=root,
            check=True,
        )

    executable = tool_path(build_dir)
    if not executable.exists():
        raise SystemExit(
            f"Report executable does not exist: {executable}\n"
            "Build with PLAPOINT_BUILD_TESTS=ON so the quality tool target is available."
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = build_dir / "mesh_quality_report"
    output_dir = output_dir.resolve()

    subprocess.run(
        [str(executable), "--output-dir", str(output_dir)],
        cwd=root,
        check=True,
    )
    print(f"Report JSON: {output_dir / 'mesh_quality_report.json'}")
    print("PLY artifacts:")
    print(f"  {output_dir / 'marching_cubes_sphere.ply'}")
    print(f"  {output_dir / 'poisson_sphere.ply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
