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

# ============================================================
# CUDA DIR24/8 vs CUDA MultiBit Trie – vplyv typu prevádzky
# ============================================================
# Finálne grafy pre kapitolu 9 používajú MEDIÁN časov.
# Priemerné grafy sa generujú iba ako doplnková kontrola stability.
# Priepustnosť je odvodená z príslušného času, nie preberaná zo CSV.
# ============================================================

BASE_DIR_MULTIBIT = cuda_multibit_trie_results_dir()
BASE_DIR_DIR24 = cuda_dir24_8_results_dir()
OUTPUT_DIR = figure_results_dir() / "graphs_gpu_traffic_types_median_fixed"

FILES = {
    "MultiBit Trie": {
        "synthetic": BASE_DIR_MULTIBIT / "gpu_multibit_trie_lpm_synthetic_ids1_2_3_4_5_6_7_summary.csv",
        "enterprise": BASE_DIR_MULTIBIT / "gpu_multibit_trie_lpm_enterprise_ids1_2_3_4_5_6_7_summary.csv",
        "hotspot": BASE_DIR_MULTIBIT / "gpu_multibit_trie_lpm_hotspot_ids1_2_3_4_5_6_7_summary.csv",
    },
    "DIR24/8": {
        "synthetic": BASE_DIR_DIR24 / "gpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_summary.csv",
        "enterprise": BASE_DIR_DIR24 / "gpu_dir24_8_lpm_enterprise_ids1_2_3_4_5_6_7_summary.csv",
        "hotspot": BASE_DIR_DIR24 / "gpu_dir24_8_lpm_hotspot_ids1_2_3_4_5_6_7_summary.csv",
    },
}
TRAFFIC_NAMES = {
    "synthetic": "Syntetická prevádzka",
    "enterprise": "Enterprise prevádzka",
    "hotspot": "Hotspot prevádzka",
}

def load_gpu(path: Path, algorithm: str, traffic: str) -> pd.DataFrame:
    label = f"CUDA {algorithm} / {traffic}"
    df = read_csv_flexible(path, label)
    required = [
        "fib_size", "ip_count",
        "kernel_lookup_avg_ms", "kernel_lookup_median_ms",
        "transfer_plus_kernel_avg_ms", "transfer_plus_kernel_median_ms",
    ]
    require_columns(df, required, label)
    numeric(df, required, label)
    df["kernel_avg_time_ms"] = df["kernel_lookup_avg_ms"]
    df["kernel_median_time_ms"] = df["kernel_lookup_median_ms"]
    df["transfer_avg_time_ms"] = df["transfer_plus_kernel_avg_ms"]
    df["transfer_median_time_ms"] = df["transfer_plus_kernel_median_ms"]
    for mode in ["kernel", "transfer"]:
        for stat in ["avg", "median"]:
            df[f"{mode}_{stat}_throughput_mpps_derived"] = throughput_from_time(
                df["ip_count"], df[f"{mode}_{stat}_time_ms"]
            )
    df["config_short"] = df.apply(config_short_label, axis=1)
    return sort_and_validate(df, label)

def columns(metric: str, statistic: str, mode: str) -> tuple[str, str]:
    if metric == "throughput":
        return (f"{mode}_{statistic}_throughput_mpps_derived", "Priepustnosť [Mpps]")
    return (f"{mode}_{statistic}_time_ms", "Čas spracovania dávky [ms]")

def plot_algorithm(algorithm: str, metric: str, statistic: str) -> None:
    plt.figure(figsize=(17, 9))
    labels = None
    for traffic, path in FILES[algorithm].items():
        df = load_gpu(path, algorithm, traffic)
        labels = df["config_short"].tolist()
        for mode, mode_name, marker in [
            ("kernel", "ČJ", "o"),
            ("transfer", "ČJP", "s"),
        ]:
            column, ylabel = columns(metric, statistic, mode)
            plt.plot(
                np.arange(len(df)), df[column], marker=marker, linewidth=2,
                label=f"{TRAFFIC_NAMES[traffic]} – {mode_name}",
            )
    stat_label = "medián" if statistic == "median" else "priemer – doplnková kontrola"
    style_plot(f"CUDA {algorithm} – vplyv typu prevádzky, {stat_label}", ylabel, labels or [])
    suffix = "priepustnost" if metric == "throughput" else "cas"
    save_plot(OUTPUT_DIR, f"gpu_{algorithm.replace('/', '_').replace(' ', '_')}_{suffix}_{statistic}_derived.png")

def plot_combined(metric: str, statistic: str, mode: str) -> None:
    plt.figure(figsize=(17, 9))
    labels = None
    column, ylabel = columns(metric, statistic, mode)
    for algorithm, traffic_map in FILES.items():
        for traffic, path in traffic_map.items():
            df = load_gpu(path, algorithm, traffic)
            labels = df["config_short"].tolist()
            plt.plot(
                np.arange(len(df)), df[column], marker="o", linewidth=2,
                label=f"CUDA {algorithm} – {TRAFFIC_NAMES[traffic]}",
            )
    stat_label = "medián" if statistic == "median" else "priemer – doplnková kontrola"
    mode_label = "ČJ" if mode == "kernel" else "ČJP"
    style_plot(f"CUDA LPM – {mode_label}, porovnanie algoritmov a prevádzky, {stat_label}", ylabel, labels or [])
    suffix = "priepustnost" if metric == "throughput" else "cas"
    save_plot(OUTPUT_DIR, f"gpu_combined_{mode}_{suffix}_{statistic}_derived.png")

def save_derived_tables() -> None:
    frames = []
    selected = [
        "fib_size", "ip_count", "config_short",
        "kernel_avg_time_ms", "kernel_median_time_ms",
        "kernel_avg_throughput_mpps_derived", "kernel_median_throughput_mpps_derived",
        "transfer_avg_time_ms", "transfer_median_time_ms",
        "transfer_avg_throughput_mpps_derived", "transfer_median_throughput_mpps_derived",
    ]
    for algorithm, traffic_map in FILES.items():
        for traffic, path in traffic_map.items():
            df = load_gpu(path, algorithm, traffic)[selected].copy()
            df.insert(0, "traffic_type", traffic)
            df.insert(0, "algorithm", algorithm)
            frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "gpu_traffic_types_times_and_derived_throughput.csv",
        index=False, encoding="utf-8-sig",
    )

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_derived_tables()
    for metric in ["throughput", "time"]:
        for statistic in ["median", "avg"]:
            for algorithm in FILES:
                plot_algorithm(algorithm, metric, statistic)
            for mode in ["kernel", "transfer"]:
                plot_combined(metric, statistic, mode)

if __name__ == "__main__":
    main()

