#!/usr/bin/env python3
"""
Runs example3 benchmarks for V1–V4 across multiple datasets.

For every version and dataset, the script executes the CPU binary and, when
available, the GPU binary. Results are printed as a table of wall-clock times.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VERSIONS = ["V1", "V2", "V3", "V4"]
DATASET_NAMES = ["small", "medium", "large"]
INTERMEDIATE_PATTERNS = ["feat*.ppm", "features.*", "gmon.out", "profile.txt"]


@dataclass
class RunResult:
    dataset: str
    version: str
    device: str
    elapsed: Optional[float]
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the src/ directory (defaults to the repository src).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_NAMES,
        choices=DATASET_NAMES,
        help="Datasets to run (default: small medium large).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to run each binary per dataset (default: 1).",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[run-matrix] {msg}")


def run_make(version_dir: Path, target: Sequence[str]) -> Tuple[bool, str]:
    proc = subprocess.run(
        ["make", *target],
        cwd=str(version_dir),
        capture_output=True,
        text=True,
    )
    success = proc.returncode == 0
    if not success:
        log(f"make {' '.join(target)} in {version_dir.name} failed:\n{proc.stderr.strip()}")
    return success, proc.stdout + proc.stderr


def cleanup_dataset_outputs(dataset_dir: Path) -> None:
    for pattern in INTERMEDIATE_PATTERNS:
        for path in dataset_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def run_binary(binary: Path, dataset_dir: Path, iterations: int) -> Tuple[Optional[float], str]:
    cleanup_dataset_outputs(dataset_dir)
    timings: List[float] = []
    for i in range(iterations):
        log(f"Running {binary.name} ({binary.parent.name}) on {dataset_dir.name} [iter {i + 1}]")
        start = time.perf_counter()
        proc = subprocess.run(
            [str(binary)],
            cwd=str(dataset_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            return None, proc.stderr.strip() or "execution failed"
        timings.append(elapsed)
    avg = sum(timings) / len(timings) if timings else None
    return avg, "ok"


def ensure_cpu_binary(version_dir: Path) -> Optional[Path]:
    binary = version_dir / "example3"
    if binary.exists():
        return binary
    ok, _ = run_make(version_dir, ["example3"])
    return binary if ok and binary.exists() else None


def ensure_gpu_binary(version_dir: Path) -> Optional[Path]:
    binary = version_dir / "example3_gpu"
    if binary.exists():
        return binary
    ok, _ = run_make(version_dir, ["gpu"])
    return binary if ok and binary.exists() else None


def format_table(rows: Sequence[RunResult]) -> str:
    headers = ["Dataset", "Version", "Device", "Time (s)", "Status"]
    table = [headers]
    for row in rows:
        time_str = f"{row.elapsed:.3f}" if row.elapsed is not None else "-"
        table.append([row.dataset, row.version, row.device, time_str, row.status])
    widths = [max(len(str(cell)) for cell in column) for column in zip(*table)]
    lines = []
    for idx, row in enumerate(table):
        formatted = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(formatted)
        if idx == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    src_root = args.src_root
    dataset_root = src_root / "dataset"
    if not dataset_root.is_dir():
        log(f"Dataset root {dataset_root} not found.")
        return 1

    results: List[RunResult] = []

    for version in VERSIONS:
        version_dir = src_root / version
        if not version_dir.is_dir():
            log(f"Skipping {version}: directory not found.")
            continue

        cpu_binary = ensure_cpu_binary(version_dir)
        gpu_binary = ensure_gpu_binary(version_dir) if version != "V1" else None

        for dataset in args.datasets:
            dataset_dir = dataset_root / dataset
            if not dataset_dir.is_dir():
                log(f"Dataset {dataset_dir} missing; skipping.")
                continue

            if cpu_binary:
                elapsed, status = run_binary(cpu_binary, dataset_dir, args.iterations)
                results.append(RunResult(dataset, version, "cpu", elapsed, status))
            else:
                results.append(RunResult(dataset, version, "cpu", None, "build failed"))

            if gpu_binary:
                elapsed, status = run_binary(gpu_binary, dataset_dir, args.iterations)
                results.append(RunResult(dataset, version, "gpu", elapsed, status))
            elif version != "V1":
                results.append(RunResult(dataset, version, "gpu", None, "unavailable"))

    print()
    print(format_table(results))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
