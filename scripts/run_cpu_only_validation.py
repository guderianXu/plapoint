#!/usr/bin/env python3
"""Configure, build, and test PlaPoint without CUDA."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cmake_configure_command(
    *,
    source_dir: Path,
    build_dir: Path,
    generator: str | None,
    extra_cmake_args: list[str],
    prefix_paths: list[Path] | None = None,
) -> list[str]:
    command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DPLAPOINT_WITH_CUDA=OFF",
        "-DPLAPOINT_BUILD_TESTS=ON",
        "-DPLAPOINT_BUILD_BENCHMARKS=ON",
    ]
    if generator:
        command.extend(["-G", generator])
    if prefix_paths:
        command.append(f"-Dplamatrix_DIR={plamatrix_dir_for_prefix(prefix_paths[0])}")
        joined = ";".join(str(path) for path in prefix_paths)
        command.append(f"-DCMAKE_PREFIX_PATH={joined}")
    command.extend(extra_cmake_args)
    return command


def has_explicit_package_hint(extra_cmake_args: list[str]) -> bool:
    return any(
        arg.startswith("-DCMAKE_PREFIX_PATH=") or arg.startswith("-Dplamatrix_DIR=")
        for arg in extra_cmake_args
    )


def has_complete_plamatrix_package(prefix: Path) -> bool:
    config_dirs = [
        prefix,
        prefix / "lib" / "cmake" / "plamatrix",
    ]
    return any(
        (config_dir / "plamatrixConfig.cmake").is_file()
        and (config_dir / "plamatrixTargets.cmake").is_file()
        for config_dir in config_dirs
    )


def plamatrix_dir_for_prefix(prefix: Path) -> Path:
    if (prefix / "plamatrixConfig.cmake").is_file():
        return prefix
    return prefix / "lib" / "cmake" / "plamatrix"


def default_cmake_prefix_paths(source_dir: Path) -> list[Path]:
    candidates = [
        source_dir.parent / "plamatrix" / "build-task-c" / "install",
        source_dir.parent / "plamatrix" / "build-cpu" / "install",
        source_dir.parent / "plamatrix" / "build-task-c-cuda" / "install",
        source_dir.parent / "plamatrix" / "build",
        source_dir.parent / "plamatrix" / "build-cpu",
    ]
    return [candidate for candidate in candidates if has_complete_plamatrix_package(candidate)]


def run_command(command: list[str], *, dry_run: bool) -> int:
    print(shlex.join(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


def resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    return (root / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PlaPoint CPU-only validation build.")
    parser.add_argument("--build-dir", type=Path, default=Path("build-abc-cpu"))
    parser.add_argument("--generator", default="")
    parser.add_argument("--parallel", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-configure", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "extra_cmake_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are appended to the CMake configure command.",
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()
    build_dir = resolve_path(args.build_dir, root)
    generator = args.generator or None
    extra_cmake_args = list(args.extra_cmake_args)
    if extra_cmake_args and extra_cmake_args[0] == "--":
        extra_cmake_args = extra_cmake_args[1:]

    if not args.skip_configure:
        prefix_paths = [] if has_explicit_package_hint(extra_cmake_args) else default_cmake_prefix_paths(root)
        configure = cmake_configure_command(
            source_dir=root,
            build_dir=build_dir,
            generator=generator,
            extra_cmake_args=extra_cmake_args,
            prefix_paths=prefix_paths,
        )
        result = run_command(configure, dry_run=args.dry_run)
        if result != 0:
            return result

    if not args.skip_build:
        build = ["cmake", "--build", str(build_dir)]
        if args.parallel:
            build.extend(["-j", args.parallel])
        result = run_command(build, dry_run=args.dry_run)
        if result != 0:
            return result

    if not args.skip_test:
        test = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
        result = run_command(test, dry_run=args.dry_run)
        if result != 0:
            return result

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
