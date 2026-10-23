from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "common"))
from paths import cpu_dir24_8_results_dir, cpu_multibit_trie_results_dir, cuda_dir24_8_results_dir, cuda_multibit_trie_results_dir, figure_results_dir, hybrid_dir24_8_results_dir
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# VIZUÁLNY ŠTÝL GRAFOV PRE VKLADANIE DO BP
# ============================================================
# Väčšie fonty sú zvolené kvôli čitateľnosti po vložení grafu
# do dokumentu vo formáte A4. Pri potrebe jemného doladenia stačí
# upraviť hodnoty nižšie.
PLOT_TITLE_FONTSIZE = 23
PLOT_LABEL_FONTSIZE = 20
PLOT_TICK_FONTSIZE = 16
PLOT_LEGEND_FONTSIZE = 16
PLOT_LEGEND_TITLE_FONTSIZE = 18
PLOT_MARKER_SIZE = 9

plt.rcParams.update({
    "font.size": PLOT_TICK_FONTSIZE,
    "axes.titlesize": PLOT_TITLE_FONTSIZE,
    "axes.labelsize": PLOT_LABEL_FONTSIZE,
    "xtick.labelsize": PLOT_TICK_FONTSIZE,
    "ytick.labelsize": PLOT_TICK_FONTSIZE,
    "legend.fontsize": PLOT_LEGEND_FONTSIZE,
    "legend.title_fontsize": PLOT_LEGEND_TITLE_FONTSIZE,
    "lines.markersize": PLOT_MARKER_SIZE,
})


# ============================================================
# Metodická voľba pre finálne výsledky bakalárskej práce
# ============================================================
# - CPU: medián času modelu dávkového CPU spracovania.
# - GPU: medián ČJ času a medián času ČJP.
# - Priepustnosť sa vždy odvodzuje z použitého času:
#       Mpps = ip_count / (time_ms * 1000)
#   Nečíta sa z uložených throughput stĺpcov, aby čas a priepustnosť
#   zostali navzájom konzistentné aj pri nestabilných behoch.
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: chýbajú stĺpce {missing}. Dostupné: {list(df.columns)}")

def read_csv_flexible(primary_path: Path, label: str) -> pd.DataFrame:
    local_path = SCRIPT_DIR / primary_path.name
    path = primary_path if primary_path.exists() else local_path
    if not path.exists():
        raise FileNotFoundError(f"{label}: nenašiel sa vstupný CSV súbor:\n - {primary_path}\n - {local_path}")
    df = pd.read_csv(path, encoding="utf-8-sig").copy()
    if df.empty:
        raise ValueError(f"{label}: CSV je prázdne: {path}")
    return df

def numeric(df: pd.DataFrame, columns: Iterable[str], label: str) -> pd.DataFrame:
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"{label}: stĺpec {c} obsahuje nečíselné hodnoty.")
    return df

def throughput_from_time(ip_count: pd.Series, time_ms: pd.Series) -> pd.Series:
    if (time_ms <= 0).any():
        raise ValueError("Čas spracovania musí byť kladný.")
    return ip_count / (time_ms * 1000.0)

def config_short_label(row: pd.Series) -> str:
    fib = int(row["fib_size"])
    ips = int(row["ip_count"])
    fib_base = fib - 1 if fib in (10001, 20001, 40001, 50001) else fib
    fib_txt = f"{fib_base // 1000}K" if fib_base >= 1000 else str(fib_base)
    if ips == 1_500_000:
        ip_txt = "1,5M"
    elif ips % 1_000_000 == 0 and ips >= 1_000_000:
        ip_txt = f"{ips // 1_000_000}M"
    elif ips >= 1000:
        ip_txt = f"{ips // 1000}K"
    else:
        ip_txt = str(ips)
    return f"{fib_txt} FIB / {ip_txt} IP"

def sort_and_validate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    duplicated = df.duplicated(subset=["fib_size", "ip_count"], keep=False)
    if duplicated.any():
        raise ValueError(f"{label}: duplicitné konfigurácie fib_size/ip_count.")
    return df.sort_values(["fib_size", "ip_count"]).reset_index(drop=True)

def save_plot(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Uložený graf: {path}")

def style_plot(title: str, ylabel: str, labels: list[str]) -> None:
    plt.title(title, fontsize=PLOT_TITLE_FONTSIZE, pad=14, wrap=True)
    plt.xlabel("Konfigurácia datasetu", fontsize=PLOT_LABEL_FONTSIZE, labelpad=10)
    plt.ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE, labelpad=10)
    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=30,
        ha="right",
        fontsize=PLOT_TICK_FONTSIZE,
    )
    plt.yticks(fontsize=PLOT_TICK_FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(
        fontsize=PLOT_LEGEND_FONTSIZE,
        title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
        frameon=True,
    )

TRAFFIC = "hotspot"
TRAFFIC_LABEL = "hotspot prevádzka"

BASE_DIR = figure_results_dir()
WORK_DIR = BASE_DIR
OUTPUT_DIR = WORK_DIR / "graphs_cpu_vs_gpu_hotspot_median"

CPU_DIR24_FILE = cpu_dir24_8_results_dir() / "cpu_dir24_8_lpm_hotspot_ids1_2_3_4_5_6_7_threads12_summary.csv"
CPU_MULTIBIT_FILE = cpu_multibit_trie_results_dir() / "cpu_multibit_trie_lpm_hotspot_ids1_2_3_4_5_6_7_threads12_summary.csv"
GPU_DIR24_FILE = cuda_dir24_8_results_dir() / "gpu_dir24_8_lpm_hotspot_ids1_2_3_4_5_6_7_summary.csv"
GPU_MULTIBIT_FILE = cuda_multibit_trie_results_dir() / "gpu_multibit_trie_lpm_hotspot_ids1_2_3_4_5_6_7_summary.csv"

def load_cpu(path: Path, label: str) -> pd.DataFrame:
    df = read_csv_flexible(path, label)
    required = ["fib_size", "ip_count", "threads", "router_batch_median_time_ms"]
    require_columns(df, required, label)
    numeric(df, required, label)
    df = df[df["threads"] == 12].copy()
    if df.empty:
        raise ValueError(f"{label}: chýba konfigurácia pre 12 vlákien.")
    df["time_ms"] = df["router_batch_median_time_ms"]
    df["throughput_mpps"] = throughput_from_time(df["ip_count"], df["time_ms"])
    df["config_short"] = df.apply(config_short_label, axis=1)
    return sort_and_validate(df, label)

def load_gpu(path: Path, label: str) -> pd.DataFrame:
    df = read_csv_flexible(path, label)
    required = ["fib_size", "ip_count", "kernel_lookup_median_ms", "transfer_plus_kernel_median_ms"]
    require_columns(df, required, label)
    numeric(df, required, label)
    df["kernel_time_ms"] = df["kernel_lookup_median_ms"]
    df["kernel_throughput_mpps"] = throughput_from_time(df["ip_count"], df["kernel_time_ms"])
    df["transfer_time_ms"] = df["transfer_plus_kernel_median_ms"]
    df["transfer_throughput_mpps"] = throughput_from_time(df["ip_count"], df["transfer_time_ms"])
    df["config_short"] = df.apply(config_short_label, axis=1)
    return sort_and_validate(df, label)

def build_table() -> pd.DataFrame:
    keys = ["fib_size", "ip_count", "config_short"]
    c1 = load_cpu(CPU_DIR24_FILE, "CPU DIR24/8")
    c2 = load_cpu(CPU_MULTIBIT_FILE, "CPU MultiBit Trie")
    g1 = load_gpu(GPU_DIR24_FILE, "CUDA DIR24/8")
    g2 = load_gpu(GPU_MULTIBIT_FILE, "CUDA MultiBit Trie")
    parts = [
        c1[keys + ["time_ms", "throughput_mpps"]].rename(columns={"time_ms": "cpu_dir24_router_batch_median_time_ms", "throughput_mpps": "cpu_dir24_router_batch_median_throughput_derived_mpps"}),
        c2[keys + ["time_ms", "throughput_mpps"]].rename(columns={"time_ms": "cpu_multibit_router_batch_median_time_ms", "throughput_mpps": "cpu_multibit_router_batch_median_throughput_derived_mpps"}),
        g1[keys + ["kernel_time_ms", "kernel_throughput_mpps", "transfer_time_ms", "transfer_throughput_mpps"]].rename(columns={"kernel_time_ms": "gpu_dir24_kernel_median_time_ms", "kernel_throughput_mpps": "gpu_dir24_kernel_median_throughput_derived_mpps", "transfer_time_ms": "gpu_dir24_transfer_plus_kernel_median_time_ms", "transfer_throughput_mpps": "gpu_dir24_transfer_plus_kernel_median_throughput_derived_mpps"}),
        g2[keys + ["kernel_time_ms", "kernel_throughput_mpps", "transfer_time_ms", "transfer_throughput_mpps"]].rename(columns={"kernel_time_ms": "gpu_multibit_kernel_median_time_ms", "kernel_throughput_mpps": "gpu_multibit_kernel_median_throughput_derived_mpps", "transfer_time_ms": "gpu_multibit_transfer_plus_kernel_median_time_ms", "transfer_throughput_mpps": "gpu_multibit_transfer_plus_kernel_median_throughput_derived_mpps"}),
    ]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.merge(part, on=keys, how="outer", validate="one_to_one")
    return combined.sort_values(["fib_size", "ip_count"]).reset_index(drop=True)

def get_series(metric: str, detailed: bool) -> list[tuple[str, str]]:
    suffix = "throughput_derived_mpps" if metric == "throughput" else "time_ms"
    series = [
        (f"cpu_dir24_router_batch_median_{suffix}", "CPU DIR24/8"),
        (f"cpu_multibit_router_batch_median_{suffix}", "CPU MultiBit Trie"),
        (f"gpu_dir24_transfer_plus_kernel_median_{suffix}", "CUDA DIR24/8 (ČJP)"),
        (f"gpu_multibit_transfer_plus_kernel_median_{suffix}", "CUDA MultiBit Trie (ČJP)"),
    ]
    if detailed:
        series.extend([
            (f"gpu_dir24_kernel_median_{suffix}", "CUDA DIR24/8 (ČJ)"),
            (f"gpu_multibit_kernel_median_{suffix}", "CUDA MultiBit Trie (ČJ)"),
        ])
    return series

def plot(df: pd.DataFrame, metric: str, detailed: bool, bars: bool) -> None:
    labels = df["config_short"].tolist()
    x = np.arange(len(df))
    ylabel = "Priepustnosť [Mpps]" if metric == "throughput" else "Čas spracovania dávky [ms]"
    scope = "všetky metriky" if detailed else "realistické porovnanie"
    title = f"CPU vs GPU LPM – {TRAFFIC_LABEL}, {scope} – medián"
    series = get_series(metric, detailed)
    plt.figure(figsize=(16, 8))
    if bars:
        width = 0.8 / len(series)
        for i, (column, label) in enumerate(series):
            plt.bar(x - 0.4 + width / 2 + i * width, df[column], width=width, label=label)
    else:
        for column, label in series:
            plt.plot(x, df[column], marker="o", linewidth=2, label=label)
    style_plot(title, ylabel, labels)
    suffix = "priepustnost" if metric == "throughput" else "cas"
    mode = "detail" if detailed else "realistic"
    kind = "bar" if bars else "line"
    save_plot(OUTPUT_DIR, f"cpu_gpu_hotspot_{mode}_{suffix}_median_{kind}.png")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.to_csv(OUTPUT_DIR / "cpu_gpu_hotspot_combined_median_derived.csv", index=False, encoding="utf-8-sig")
    for metric in ["throughput", "time"]:
        for detailed in [False, True]:
            for bars in [False, True]:
                plot(table, metric, detailed, bars)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
