"""Generate dataset-specific performance graphs from performance_data.json.

Run `python3 performance_graphs.py` after refreshing metrics with
`collect_performance_data.py`.  The script emits one set of PNGs per dataset
under `figures/<dataset>/`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

PLOT_ROOT = Path(__file__).with_name("figures")
DATA_PATH = Path(__file__).with_name("performance_data.json")
PLOT_ROOT.mkdir(exist_ok=True)
CURRENT_PLOT_DIR = PLOT_ROOT


def set_output_dir(directory: Path) -> None:
    global CURRENT_PLOT_DIR
    CURRENT_PLOT_DIR = directory
    CURRENT_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str) -> None:
    """Save figure to the active figures directory and close it."""
    fig.savefig(CURRENT_PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_dataset() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    with DATA_PATH.open() as fh:
        return json.load(fh)


def require_section(dataset: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in dataset:
        raise KeyError(f"`{name}` missing from performance_data.json")
    return dataset[name]


def build_speedup_data(dataset_info: Dict[str, Any]) -> Dict[str, List[float]]:
    derived = dataset_info.get("derived", {})
    speedup = derived.get("speedup")
    if not speedup:
        raise KeyError("`derived.speedup` missing for dataset")
    versions = list(speedup.keys())
    return {"versions": versions, "speedup": [speedup[v] for v in versions]}


def build_hotspot_series(data: Dict[str, Any]) -> Dict[str, List[float]]:
    return {k: v for k, v in data.items() if k != "functions"}


def plot_speedup_chart(data: Dict[str, List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(data["versions"], data["speedup"], color="#377eb8")
    ax.set_ylabel("Speedup vs V1")
    ax.set_title("Speedup Progression")
    for idx, value in enumerate(data["speedup"]):
        ax.text(idx, value + 0.05, f"{value:.2f}x", ha="center", va="bottom")
    save_fig(fig, "speedup_bar_chart.png")


def plot_execution_breakdown(data: Dict[str, List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_width = 0.6
    ind = np.arange(len(data["versions"]))
    comp = np.array(data["computation"])
    mem = np.array(data["memory_transfer"])
    other = np.array(data["other"])
    ax.bar(ind, comp, bar_width, label="Computation", color="#4daf4a")
    ax.bar(ind, mem, bar_width, bottom=comp, label="Memory Transfer", color="#ff7f00")
    ax.bar(ind, other, bar_width, bottom=comp + mem, label="Other", color="#984ea3")
    ax.set_xticks(ind)
    ax.set_xticklabels(data["versions"])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Execution Time Breakdown")
    ax.legend(loc="upper right")
    save_fig(fig, "execution_time_breakdown.png")


def plot_strong_scaling(data: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    threads = data["threads"]
    for label, values in data["series"].items():
        ax.plot(threads, values, marker="o", label=label)
    ax.set_xlabel("Threads per block")
    ax.set_ylabel(data.get("ylabel", "Execution time (ms)"))
    ax.set_title("Strong Scaling")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    save_fig(fig, "strong_scaling.png")


def plot_weak_scaling(data: Dict[str, Any]) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    x = np.arange(len(data["problem_sizes"]))
    ax1.plot(x, data["execution_time"], color="#e41a1c", marker="s", label="Execution time")
    ax1.set_ylabel("Time (ms)", color="#e41a1c")
    ax1.tick_params(axis="y", labelcolor="#e41a1c")
    ax2.plot(x, data["speedup"], color="#377eb8", marker="o", label="Speedup")
    ax2.set_ylabel("Speedup", color="#377eb8")
    ax2.tick_params(axis="y", labelcolor="#377eb8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(data["problem_sizes"])
    ax1.set_title("Weak Scaling")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    save_fig(fig, "weak_scaling.png")


def plot_hotspot_analysis(data: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y_pos = np.arange(len(data["functions"]))
    height = 0.35
    series = build_hotspot_series(data)
    offsets = np.linspace(-height, height, num=len(series))
    for offset, (label, values) in zip(offsets, series.items()):
        ax.barh(y_pos + offset, values, height, label=label)
    ax.set_xlabel("% of total time")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["functions"])
    ax.set_title("Hotspot Analysis")
    ax.legend()
    save_fig(fig, "hotspot_analysis.png")


def plot_gpu_utilization(data: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    versions = data["versions"]
    x = np.arange(len(versions))
    width = 0.25
    metrics = data["metrics"]
    metric_names = list(metrics.keys())
    offsets = np.linspace(-width, width, num=len(metric_names)) if metric_names else [0]
    for offset, name in zip(offsets, metric_names):
        ax.bar(x + offset, metrics[name], width, label=name.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("Utilization (%)")
    ax.set_title("GPU Utilization Metrics")
    ax.set_ylim(0, 100)
    ax.legend()
    save_fig(fig, "gpu_utilization.png")


def plot_roofline(data: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    points = data["points"]
    intensity = np.array([pt["intensity"] for pt in points])
    performance = np.array([pt["performance"] for pt in points])
    ax.scatter(intensity, performance, color="#7570b3")
    for pt in points:
        ax.annotate(pt["label"], (pt["intensity"], pt["performance"]), textcoords="offset points", xytext=(5, 5))
    peak = data["machine_peak"]
    bw = data["bandwidth_bound"]
    ai = np.linspace(min(intensity) / 2, max(intensity) * 1.1, 100)
    ax.plot(ai, np.minimum(peak, bw * ai), linestyle="--", color="#444444", label="Hardware roofline")
    ax.axhline(peak, color="#999999", linewidth=0.75)
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (GFLOPs/s)")
    ax.set_title("Roofline Model")
    ax.legend()
    save_fig(fig, "roofline.png")


def plot_efficiency(data: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(data["versions"], data["percent_peak"], marker="o", color="#d95f02")
    ax.set_ylabel("% of peak performance")
    ax.set_ylim(0, 100)
    ax.set_title("Efficiency vs Version")
    for x, y in zip(data["versions"], data["percent_peak"]):
        ax.text(x, y + 2, f"{y}%", ha="center")
    save_fig(fig, "efficiency.png")


def plot_memory_access(data: Dict[str, Any]) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    strategies = np.arange(len(data["strategies"]))
    ax1.bar(strategies - 0.15, data["latency"], width=0.3, color="#80b1d3", label="Normalized latency")
    ax2.bar(strategies + 0.15, data["throughput"], width=0.3, color="#fb8072", label="Throughput (GB/s)")
    ax1.set_ylabel("Normalized latency")
    ax2.set_ylabel("Throughput (GB/s)")
    ax1.set_xticks(strategies)
    ax1.set_xticklabels(data["strategies"])
    ax1.set_title("Memory Access Strategy Impact")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    save_fig(fig, "memory_access.png")


def dataset_section(dataset_info: Dict[str, Any], global_data: Dict[str, Any], section: str) -> Dict[str, Any]:
    if section in dataset_info:
        return dataset_info[section]
    return require_section(global_data, section)


def generate_for_dataset(name: str, dataset_info: Dict[str, Any], global_data: Dict[str, Any]) -> None:
    print(f"Generating figures for dataset '{name}'")
    set_output_dir(PLOT_ROOT / name)
    speedup = build_speedup_data(dataset_info)
    execution_breakdown = dataset_section(dataset_info, global_data, "execution_breakdown")
    strong_scaling = dataset_section(dataset_info, global_data, "strong_scaling")
    weak_scaling = dataset_section(dataset_info, global_data, "weak_scaling")
    hotspot = dataset_section(dataset_info, global_data, "hotspot_analysis")
    gpu_util = dataset_section(dataset_info, global_data, "gpu_utilization")
    roof = dataset_section(dataset_info, global_data, "roofline")
    eff = dataset_section(dataset_info, global_data, "efficiency")
    memory = dataset_section(dataset_info, global_data, "memory_access")

    plot_speedup_chart(speedup)
    plot_execution_breakdown(execution_breakdown)
    plot_strong_scaling(strong_scaling)
    plot_weak_scaling(weak_scaling)
    plot_hotspot_analysis(hotspot)
    plot_gpu_utilization(gpu_util)
    plot_roofline(roof)
    plot_efficiency(eff)
    plot_memory_access(memory)


def main() -> None:
    dataset = load_dataset()
    datasets_section = dataset.get("datasets")
    if not datasets_section:
        raise KeyError("`datasets` section missing from performance_data.json")

    for name, dataset_info in datasets_section.items():
        generate_for_dataset(name, dataset_info, dataset)
    print(f"Saved figures to {PLOT_ROOT.resolve()}/*")


if __name__ == "__main__":
    main()
