#!/usr/bin/env python3
"""
Graph generation utility for the KLT GPU acceleration project.

Reads the JSON artifact produced by collect_performance_data.py and renders all
required plots (speedup, breakdown, scaling, profiler insights, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate performance graphs from JSON.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("performance_data.json"),
        help="Path to the JSON file produced by collect_performance_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("graphs"),
        help="Directory to store the generated figures.",
    )
    parser.add_argument(
        "--dataset",
        default="large",
        help="Dataset key to use for per-dataset graphs (speedup, breakdown).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Image DPI for the generated figures.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, dpi: int) -> Path:
    ensure_output_dir(output_dir)
    target = output_dir / filename
    fig.tight_layout()
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_speedup(dataset: Dict, dataset_name: str, output_dir: Path, dpi: int) -> Path:
    speedup = dataset.get("derived", {}).get("speedup")
    if not speedup:
        raise RuntimeError(f"No speedup data found for dataset '{dataset_name}'.")
    versions = list(speedup.keys())
    values = [speedup[v] or 0.0 for v in versions]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(versions, values, color="#4c72b0")
    ax.set_xlabel("Versions")
    ax.set_ylabel("Speedup vs V1")
    ax.set_title(f"Speedup progression ({dataset_name})")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}", ha="center")
    return save_figure(fig, output_dir, f"speedup_{dataset_name}.png", dpi)


def plot_execution_breakdown(dataset: Dict, dataset_name: str, output_dir: Path, dpi: int) -> Path:
    breakdown = dataset.get("derived", {}).get("breakdown")
    if not breakdown:
        raise RuntimeError(f"No execution breakdown data available for {dataset_name}")
    versions = list(breakdown.keys())
    comp = [safe_value(breakdown[v], "computation") for v in versions]
    mem = [safe_value(breakdown[v], "memory_transfer") for v in versions]
    other = [safe_value(breakdown[v], "other") for v in versions]
    fig, ax = plt.subplots(figsize=(9, 4))
    bar_width = 0.6
    indices = np.arange(len(versions))
    ax.bar(indices, comp, bar_width, label="Computation")
    ax.bar(indices, mem, bar_width, bottom=comp, label="Memory transfer")
    ax.bar(indices, other, bar_width, bottom=[c + m for c, m in zip(comp, mem)], label="Other")
    ax.set_xticks(indices, versions)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Execution time breakdown ({dataset_name})")
    ax.legend()
    return save_figure(fig, output_dir, f"breakdown_{dataset_name}.png", dpi)


def safe_value(entry: Dict, key: str) -> float:
    if not entry:
        return 0.0
    value = entry.get(key)
    return float(value) if value is not None else 0.0


def plot_strong_scaling(data: Dict, output_dir: Path, dpi: int) -> Path:
    scaling = data.get("strong_scaling") or {}
    block_sizes = scaling.get("block_sizes")
    results = scaling.get("results")
    if not block_sizes or not results:
        raise RuntimeError("Strong scaling data is unavailable.")
    fig, ax = plt.subplots(figsize=(8, 4))
    for version, entries in results.items():
        entries_sorted = sorted(entries, key=lambda item: item.get("block", 0))
        x = []
        y = []
        for entry in entries_sorted:
            if entry.get("time") is None:
                continue
            x.append(entry["block"])
            y.append(entry["time"])
        if not x:
            continue
        ax.plot(x, y, marker="o", label=version)
    ax.set_xlabel("Threads per block")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Strong scaling (GPU block sweep)")
    ax.set_xscale("log", base=2)
    ax.legend()
    return save_figure(fig, output_dir, "strong_scaling.png", dpi)


def plot_hotspots(data: Dict, output_dir: Path, dpi: int) -> Path:
    hotspot = data.get("hotspot_analysis") or {}
    functions: List[str] = hotspot.get("functions", [])
    version_keys = sorted(k for k in hotspot.keys() if k != "functions")
    if not functions or not version_keys:
        raise RuntimeError("Hotspot analysis data missing.")
    y_pos = np.arange(len(functions))
    height = 0.8 / len(version_keys)
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, version in enumerate(version_keys):
        values = hotspot.get(version) or [0.0] * len(functions)
        offset = (-0.4 + height / 2) + idx * height
        ax.barh(y_pos + offset, values, height=height, label=version)
    ax.set_yticks(y_pos, functions)
    ax.set_xlabel("% of total time")
    ax.set_title("Hotspot comparison")
    ax.legend()
    return save_figure(fig, output_dir, "hotspot_analysis.png", dpi)


def plot_weak_scaling(data: Dict, output_dir: Path, dpi: int) -> Path:
    weak = data.get("weak_scaling") or {}
    problem_sizes = weak.get("problem_sizes")
    times = weak.get("execution_time")
    speedup = weak.get("speedup")
    if not problem_sizes or not times:
        raise RuntimeError("Weak scaling data missing.")
    fig, ax1 = plt.subplots(figsize=(9, 4))
    indices = np.arange(len(problem_sizes))
    ax1.bar(indices, times, color="#55a868", label="Execution time (s)")
    ax1.set_ylabel("Execution time (s)")
    ax1.set_xticks(indices, problem_sizes, rotation=20, ha="right")
    ax1.set_xlabel("Problem size (WxHxFrames)")
    ax2 = ax1.twinx()
    if speedup:
        ax2.plot(indices, speedup, color="#c44e52", marker="o", label="Speedup vs smallest")
        ax2.set_ylabel("Speedup")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    ax1.set_title("Weak scaling")
    return save_figure(fig, output_dir, "weak_scaling.png", dpi)


def plot_gpu_utilization(data: Dict, output_dir: Path, dpi: int) -> Path:
    util = data.get("gpu_utilization") or {}
    versions = util.get("versions")
    metrics = util.get("metrics")
    if not versions or not metrics:
        raise RuntimeError("GPU utilization data missing.")
    categories = ["occupancy", "memory_bw", "sm_efficiency"]
    fig, ax = plt.subplots(figsize=(9, 4))
    indices = np.arange(len(versions))
    width = 0.2
    for i, cat in enumerate(categories):
        values = metrics.get(cat) or [0.0] * len(versions)
        ax.bar(indices + (i - 1) * width, values, width, label=cat.replace("_", " ").title())
    ax.set_xticks(indices, versions)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 120)
    ax.set_title("GPU utilization metrics")
    ax.legend()
    return save_figure(fig, output_dir, "gpu_utilization.png", dpi)


def plot_roofline(data: Dict, output_dir: Path, dpi: int) -> Path:
    roof = data.get("roofline") or {}
    points = roof.get("points")
    if not points:
        raise RuntimeError("Roofline data missing.")
    machine_peak = roof.get("machine_peak")
    bandwidth_bound = roof.get("bandwidth_bound")
    intensities = [p["intensity"] for p in points]
    performances = [p["performance"] for p in points]
    labels = [p["label"] for p in points]
    min_intensity = min(intensities)
    max_intensity = max(intensities)
    lower = max(min_intensity * 0.5, 1e-6)
    upper = max(max_intensity * 1.5, lower * 1.01)
    x_vals = np.linspace(lower, upper, 100)
    y_peak = [machine_peak] * len(x_vals) if machine_peak else None
    y_bw = (
        bandwidth_bound * x_vals if bandwidth_bound is not None else None
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(intensities, performances, color="#dd8452")
    for x, y, label in zip(intensities, performances, labels):
        ax.text(x * 1.02, y * 1.02, label)
    if y_peak is not None:
        ax.plot(x_vals, y_peak, linestyle="--", color="gray", label="Peak compute")
    if y_bw is not None:
        ax.plot(x_vals, y_bw, linestyle=":", color="steelblue", label="Bandwidth bound")
    ax.set_xlabel("Arithmetic intensity (ops/byte)")
    ax.set_ylabel("Performance (pixels/s)")
    ax.set_title("Roofline model")
    ax.set_ylim(bottom=0)
    ax.legend()
    return save_figure(fig, output_dir, "roofline.png", dpi)


def plot_efficiency(data: Dict, output_dir: Path, dpi: int) -> Path:
    eff = data.get("efficiency") or {}
    versions = eff.get("versions")
    pct = eff.get("percent_peak")
    if not versions or not pct:
        raise RuntimeError("Efficiency data missing.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(versions, pct, color="#8172b2")
    ax.set_ylabel("% of peak")
    ax.set_ylim(0, 110)
    ax.set_title("Efficiency vs theoretical peak")
    return save_figure(fig, output_dir, "efficiency.png", dpi)


def plot_memory_access(data: Dict, output_dir: Path, dpi: int) -> Path:
    mem = data.get("memory_access") or {}
    strategies = mem.get("strategies")
    latency = mem.get("latency")
    throughput = mem.get("throughput")
    if not strategies or not latency or not throughput:
        raise RuntimeError("Memory access data missing.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    axes[0].bar(strategies, latency, color="#937860")
    axes[0].set_ylabel("Latency (s / pixel)")
    axes[0].set_title("Per-pixel latency")
    axes[1].bar(strategies, throughput, color="#da8bc3")
    axes[1].set_ylabel("Throughput (bytes / s)")
    axes[1].set_title("Memory throughput")
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
            tick.set_horizontalalignment("right")
    fig.suptitle("Memory access pattern impact")
    return save_figure(fig, output_dir, "memory_access.png", dpi)


def main() -> None:
    args = parse_args()
    with args.data.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    dataset = data.get("datasets", {}).get(args.dataset)
    if not dataset:
        raise RuntimeError(f"Dataset '{args.dataset}' not found in data file.")
    outputs = {}
    outputs["speedup"] = plot_speedup(dataset, args.dataset, args.output_dir, args.dpi)
    outputs["breakdown"] = plot_execution_breakdown(dataset, args.dataset, args.output_dir, args.dpi)
    try:
        outputs["strong_scaling"] = plot_strong_scaling(data, args.output_dir, args.dpi)
    except RuntimeError as exc:
        print(f"[warn] {exc}")
    try:
        outputs["hotspot"] = plot_hotspots(data, args.output_dir, args.dpi)
    except RuntimeError as exc:
        print(f"[warn] {exc}")
    try:
        outputs["weak_scaling"] = plot_weak_scaling(data, args.output_dir, args.dpi)
    except RuntimeError as exc:
        print(f"[warn] {exc}")
    try:
        outputs["gpu_utilization"] = plot_gpu_utilization(data, args.output_dir, args.dpi)
    except RuntimeError as exc:
        print(f"[warn] {exc}")
    try:
        outputs["roofline"] = plot_roofline(data, args.output_dir, args.dpi)
    except RuntimeError as exc:
        print(f"[warn] {exc}")
    try:
        outputs["efficiency"] = plot_efficiency(data, args.output_dir, args.dpi)
    except RuntimeError as exc:
        print(f"[warn] {exc}")
    try:
        outputs["memory_access"] = plot_memory_access(data, args.output_dir, args.dpi)
    except RuntimeError as exc:
        print(f"[warn] {exc}")
    for name, path in outputs.items():
        if path:
            print(f"[graphs] {name}: {path}")


if __name__ == "__main__":
    main()
