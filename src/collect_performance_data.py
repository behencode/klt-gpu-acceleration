"""Build example3 binaries and collect timings for each dataset/variant.

This script sweeps all versions (V1-V4) across every dataset under ``dataset/``
(``small``, ``medium``, ``large``) and records CPU/GPU runtimes into
``performance_data.json``.  Usage:

    python3 collect_performance_data.py --iterations 3

GPU builds are attempted when CUDA is available; failures are logged per dataset
so CPU measurements still complete.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "performance_data.json"
ARTIFACT_PATTERNS = ("feat*.ppm", "features.ft", "features.txt")
DATASET_ROOT = ROOT / "dataset"
DATASETS = ("small", "medium", "large")


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
        cleanup_artifacts(cwd)
        start = time.perf_counter()
        run_command([str(exec_path)], cwd=cwd)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        print(f"{cwd.name}: iteration {idx + 1}/{iterations} finished in {elapsed:.3f}s")
    cleanup_artifacts(cwd)
    stats = {
        "runs": timings,
        "mean": statistics.fmean(timings),
        "median": statistics.median(timings),
        "stdev": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
    }
    return stats


def prepare_builds(config: VersionConfig) -> Dict[str, Optional[str]]:
    print(f"=== Preparing {config.name} builds ===")
    clean_directory(config)
    build_cpu_executable(config)
    gpu_error: Optional[str] = None
    if config.supports_gpu():
        try:
            build_gpu_executable(config)
        except subprocess.CalledProcessError as exc:
            gpu_error = f"GPU build failed: {exc}"
    else:
        gpu_error = "GPU target not available"
    return {"gpu_error": gpu_error}


def collect_for_dataset(
    dataset_name: str,
    dataset_path: Path,
    iterations: int,
    build_status: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, Dict[str, object]]:
    dataset_results: Dict[str, Dict[str, object]] = {}
    for config in VERSIONS:
        print(f"--- Dataset {dataset_name}: running {config.name} CPU ---")
        cpu_stats = measure_run(config.path / config.cpu_binary, cwd=dataset_path, iterations=iterations)

        gpu_stats: Optional[Dict[str, float]] = None
        gpu_error = build_status[config.name]["gpu_error"]
        if gpu_error is None and config.supports_gpu():
            try:
                print(f"--- Dataset {dataset_name}: running {config.name} GPU ---")
                gpu_stats = measure_run(config.path / config.gpu_binary, cwd=dataset_path, iterations=iterations)
            except subprocess.CalledProcessError as exc:
                gpu_error = f"GPU run failed: {exc}"

        dataset_results[config.name] = {"cpu": cpu_stats, "gpu": gpu_stats, "gpu_error": gpu_error}
    return dataset_results


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


def write_dataset(dataset_payload: Dict[str, Any], output: Path, iterations: int) -> None:
    payload: Dict[str, Any] = {}
    if output.exists():
        with output.open() as existing:
            payload = json.load(existing)
    payload.pop("results", None)
    payload.pop("derived", None)
    payload["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["iterations"] = iterations
    payload["datasets"] = dataset_payload
    output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote dataset to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CPU/GPU timings for example3 across datasets.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of times to run each executable (default: 3)",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Destination JSON file.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    datasets_payload: Dict[str, Any] = {}
    build_status: Dict[str, Dict[str, Optional[str]]] = {}
    for config in VERSIONS:
        build_status[config.name] = prepare_builds(config)

    for dataset_name in DATASETS:
        dataset_path = DATASET_ROOT / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path missing: {dataset_path}")
        dataset_results = collect_for_dataset(dataset_name, dataset_path, args.iterations, build_status)
        datasets_payload[dataset_name] = {
            "results": dataset_results,
            "derived": {"speedup": compute_speedup(dataset_results)},
        }

    write_dataset(datasets_payload, args.output, args.iterations)


if __name__ == "__main__":
    args = parse_args()
    main(args)
