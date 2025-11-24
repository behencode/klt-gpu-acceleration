"""Build every version of example3 and collect timing data for CPU and GPU.

This script generates ``performance_data.json`` which ``performance_graphs.py``
can consume to power the plots.  Usage:

    python3 collect_performance_data.py --iterations 3

GPU builds are attempted for versions that ship CUDA targets.  If CUDA or a GPU
is not available the script records the failure and continues so you still
capture CPU measurements.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "performance_data.json"
ARTIFACT_PATTERNS = ("feat*.ppm", "features.ft", "features.txt")


@dataclass(frozen=True)
class VersionConfig:
    name: str
    path: Path
    cpu_target: str = "example3"
    cpu_binary: str = "example3"
    gpu_target: Optional[str] = "example3_gpu"
    gpu_binary: Optional[str] = "example3_gpu"

    def supports_gpu(self) -> bool:
        return self.gpu_target is not None and self.gpu_binary is not None


VERSIONS: List[VersionConfig] = [
    VersionConfig(name="V1", path=ROOT / "V1", gpu_target=None, gpu_binary=None),
    VersionConfig(name="V2", path=ROOT / "V2"),
    VersionConfig(name="V3", path=ROOT / "V3"),
    VersionConfig(name="V4", path=ROOT / "V4"),
]


def run_command(cmd: List[str], *, cwd: Path) -> None:
    """Run command and stream output, raising on failure."""
    subprocess.run(cmd, cwd=cwd, check=True)


def clean_directory(config: VersionConfig) -> None:
    run_command(["make", "clean"], cwd=config.path)


def build_cpu_executable(config: VersionConfig) -> None:
    run_command(["make", "lib"], cwd=config.path)
    run_command(["make", config.cpu_target], cwd=config.path)


def build_gpu_executable(config: VersionConfig) -> None:
    run_command(["make", "libklt_gpu.a"], cwd=config.path)
    run_command(["make", config.gpu_target], cwd=config.path)


def cleanup_artifacts(path: Path) -> None:
    for pattern in ARTIFACT_PATTERNS:
        for candidate in path.glob(pattern):
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue


def measure_run(exec_path: Path, *, cwd: Path, iterations: int) -> Dict[str, float]:
    timings: List[float] = []
    for idx in range(iterations):
        start = time.perf_counter()
        run_command([str(exec_path)], cwd=cwd)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        cleanup_artifacts(cwd)
        print(f"{cwd.name}: iteration {idx + 1}/{iterations} finished in {elapsed:.3f}s")
    stats = {
        "runs": timings,
        "mean": statistics.fmean(timings),
        "median": statistics.median(timings),
        "stdev": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
    }
    return stats


def collect_for_version(config: VersionConfig, iterations: int) -> Dict[str, object]:
    print(f"=== {config.name}: preparing CPU build ===")
    clean_directory(config)
    build_cpu_executable(config)
    cpu_stats = measure_run(config.path / config.cpu_binary, cwd=config.path, iterations=iterations)

    gpu_stats: Optional[Dict[str, float]] = None
    gpu_error: Optional[str] = None
    if config.supports_gpu():
        print(f"=== {config.name}: preparing GPU build ===")
        try:
            build_gpu_executable(config)
            gpu_stats = measure_run(config.path / config.gpu_binary, cwd=config.path, iterations=iterations)
        except subprocess.CalledProcessError as exc:
            gpu_error = f"GPU run failed: {exc}"
    else:
        gpu_error = "GPU target not available"

    return {"cpu": cpu_stats, "gpu": gpu_stats, "gpu_error": gpu_error}


def compute_speedup(results: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    baseline = results["V1"]["cpu"]["mean"]  # type: ignore[index]
    speedups: Dict[str, float] = {}
    for version, metrics in results.items():
        cpu_mean = metrics["cpu"]["mean"]  # type: ignore[index]
        best = cpu_mean
        gpu_stats = metrics.get("gpu")
        if isinstance(gpu_stats, dict):
            best = min(best, gpu_stats["mean"])
        speedups[version] = baseline / best
    return speedups


def write_dataset(results: Dict[str, Dict[str, object]], output: Path, iterations: int) -> None:
    payload: Dict[str, object] = {}
    if output.exists():
        with output.open() as existing:
            payload = json.load(existing)
    payload.update(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iterations": iterations,
            "results": results,
            "derived": {"speedup": compute_speedup(results)},
        }
    )
    output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote dataset to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CPU/GPU timings for example3 across versions.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of times to run each executable (default: 3)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Destination JSON file.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    results: Dict[str, Dict[str, object]] = {}
    for config in VERSIONS:
        results[config.name] = collect_for_version(config, args.iterations)
    write_dataset(results, args.output, args.iterations)


if __name__ == "__main__":
    args = parse_args()
    main(args)
