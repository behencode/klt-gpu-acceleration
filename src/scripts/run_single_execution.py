#!/usr/bin/env python3
"""Build and run a single KLT benchmark configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run one KLT version on a specific dataset."
    )
    parser.add_argument(
        "version",
        help='Version identifier (e.g. "v1", "V2").',
    )
    parser.add_argument(
        "dataset",
        help='Dataset name under dataset/ (e.g. "small", "medium", "large").',
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=default_root,
        help="Project src/ directory; defaults to the parent of this script.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Path to datasets relative to --src-root (default: dataset/).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Select which binary to run. 'auto' prefers GPU when available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the inferred actions without running make/binaries.",
    )
    return parser.parse_args()


def normalize_version(version: str) -> str:
    version = version.strip()
    if not version:
        raise ValueError("Empty version provided.")
    if version.lower().startswith("v"):
        suffix = version[1:]
    else:
        suffix = version
    return f"V{suffix}"


def resolve_version_dir(src_root: Path, version_name: str) -> Path:
    version_dir = (src_root / version_name).resolve()
    if not version_dir.is_dir():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")
    return version_dir


def resolve_dataset(src_root: Path, dataset_root: Path, dataset_name: str) -> Path:
    dataset_path = (src_root / dataset_root / dataset_name.lower()).resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    return dataset_path


def pick_target(version_dir: Path, device: str) -> Tuple[str, Path]:
    cpu_target = ("example3", version_dir / "example3")
    gpu_target = ("example3_gpu", version_dir / "example3_gpu")
    gpu_sources_exist = (version_dir / "convolve_gpu.cu").exists()

    if device == "cpu":
        return cpu_target
    if device == "gpu":
        if not gpu_sources_exist:
            raise RuntimeError(f"{version_dir.name} has no GPU implementation.")
        return gpu_target
    # auto
    if gpu_sources_exist:
        return gpu_target
    return cpu_target


def run_command(cmd, cwd: Path, dry_run: bool = False) -> int:
    cmd_display = " ".join(map(str, cmd))
    print(f"[run-single] ({cwd}) $ {cmd_display}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def main() -> None:
    args = parse_args()
    try:
        version_name = normalize_version(args.version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    src_root = args.src_root.resolve()
    dataset_root = args.dataset_root
    try:
        version_dir = resolve_version_dir(src_root, version_name)
        dataset_path = resolve_dataset(src_root, dataset_root, args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        target_name, binary_path = pick_target(version_dir, args.device)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[run-single] Using version {version_name} ({target_name}) on dataset {dataset_path.name}"
    )

    try:
        run_command(["make", target_name], version_dir, args.dry_run)
        if args.dry_run:
            run_command([str(binary_path)], dataset_path, True)
        else:
            start_time = time.perf_counter()
            run_command([str(binary_path)], dataset_path, False)
            elapsed = time.perf_counter() - start_time
            print(f"[run-single] Binary execution time: {elapsed:.3f}s")
    except subprocess.CalledProcessError as exc:
        print(
            f"error: command {' '.join(map(str, exc.cmd))} failed with exit code {exc.returncode}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
