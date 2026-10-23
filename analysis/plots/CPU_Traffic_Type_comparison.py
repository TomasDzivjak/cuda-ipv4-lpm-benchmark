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
# CPU DIR24/8 vs MultiBit Trie – vplyv typu prevádzky
# ============================================================
# Finálne grafy pre kapitolu 9 používajú MEDIÁN času dávkového CPU spracovania.
# Priemerné grafy sa generujú iba ako doplnková kontrola stability.
# ============================================================

BASE_DIR_DIR = cpu_dir24_8_results_dir()
BASE_DIR_MULTIBIT = cpu_multibit_trie_results_dir()
WORK_DIR = SCRIPT_DIR
OUTPUT_DIR = figure_results_dir() / "graphs_cpu_traffic_types_median_fixed"

FILES = {
    "DIR24/8": {
        "synthetic": BASE_DIR_DIR / "cpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv",
        "enterprise": BASE_DIR_DIR / "cpu_dir24_8_lpm_enterprise_ids1_2_3_4_5_6_7_threads12_summary.csv",
        "hotspot": BASE_DIR_DIR / "cpu_dir24_8_lpm_hotspot_ids1_2_3_4_5_6_7_threads12_summary.csv",
    },
    "MultiBit Trie": {
        "synthetic": BASE_DIR_MULTIBIT / "cpu_multibit_trie_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv",
        "enterprise": BASE_DIR_MULTIBIT / "cpu_multibit_trie_lpm_enterprise_ids1_2_3_4_5_6_7_threads12_summary.csv",
        "hotspot": BASE_DIR_MULTIBIT / "cpu_multibit_trie_lpm_hotspot_ids1_2_3_4_5_6_7_threads12_summary.csv",
    },
}
TRAFFIC_NAMES = {
    "synthetic": "Syntetická prevádzka",
    "enterprise": "Enterprise prevádzka",
    "hotspot": "Hotspot prevádzka",
}

def load_cpu(path: Path, algorithm: str, traffic: str) -> pd.DataFrame:
    label = f"{algorithm} / {traffic}"
    df = read_csv_flexible(path, label)
    required = [
        "fib_size", "ip_count", "threads",
        "router_batch_time_ms", "router_batch_median_time_ms",
    ]
    require_columns(df, required, label)
    numeric(df, required, label)
    df = df[df["threads"] == 12].copy()
    if df.empty:
        raise ValueError(f"{label}: chýbajú 12-vláknové výsledky.")
    if "traffic_type" in df.columns and (df["traffic_type"].astype(str).str.lower() != traffic).any():
        raise ValueError(f"{label}: stĺpec traffic_type nesúhlasí.")
    df["avg_time_ms"] = df["router_batch_time_ms"]
    df["median_time_ms"] = df["router_batch_median_time_ms"]
    df["avg_throughput_mpps_derived"] = throughput_from_time(df["ip_count"], df["avg_time_ms"])
    df["median_throughput_mpps_derived"] = throughput_from_time(df["ip_count"], df["median_time_ms"])
    df["config_short"] = df.apply(config_short_label, axis=1)
    return sort_and_validate(df, label)

def metric_column(metric: str, statistic: str) -> tuple[str, str]:
    if metric == "throughput":
        return (f"{statistic}_throughput_mpps_derived", "Priepustnosť [Mpps]")
    return (f"{statistic}_time_ms", "Čas spracovania dávky [ms]")

def plot_algorithm(algorithm: str, metric: str, statistic: str) -> None:
    column, ylabel = metric_column(metric, statistic)
    plt.figure(figsize=(16, 8))
    labels = None
    for traffic, path in FILES[algorithm].items():
        df = load_cpu(path, algorithm, traffic)
        labels = df["config_short"].tolist()
        plt.plot(np.arange(len(df)), df[column], marker="o", linewidth=2, label=TRAFFIC_NAMES[traffic])
    stat_label = "medián" if statistic == "median" else "priemer – doplnková kontrola"
    style_plot(f"CPU {algorithm} – vplyv typu prevádzky, {stat_label}", ylabel, labels or [])
    suffix = "priepustnost" if metric == "throughput" else "cas"
    save_plot(OUTPUT_DIR, f"cpu_{algorithm.replace('/', '_').replace(' ', '_')}_{suffix}_{statistic}_derived.png")

def plot_combined(metric: str, statistic: str) -> None:
    column, ylabel = metric_column(metric, statistic)
    plt.figure(figsize=(17, 9))
    labels = None
    for algorithm, traffic_map in FILES.items():
        for traffic, path in traffic_map.items():
            df = load_cpu(path, algorithm, traffic)
            labels = df["config_short"].tolist()
            plt.plot(
                np.arange(len(df)), df[column], marker="o", linewidth=2,
                label=f"{algorithm} – {TRAFFIC_NAMES[traffic]}",
            )
    stat_label = "medián" if statistic == "median" else "priemer – doplnková kontrola"
    style_plot(f"CPU LPM – porovnanie algoritmov a typov prevádzky, {stat_label}", ylabel, labels or [])
    suffix = "priepustnost" if metric == "throughput" else "cas"
    save_plot(OUTPUT_DIR, f"cpu_combined_{suffix}_{statistic}_derived.png")

def save_derived_tables() -> None:
    frames = []
    for algorithm, traffic_map in FILES.items():
        for traffic, path in traffic_map.items():
            df = load_cpu(path, algorithm, traffic)[[
                "fib_size", "ip_count", "config_short", "avg_time_ms", "median_time_ms",
                "avg_throughput_mpps_derived", "median_throughput_mpps_derived",
            ]].copy()
            df.insert(0, "traffic_type", traffic)
            df.insert(0, "algorithm", algorithm)
            frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "cpu_traffic_types_router_batch_times_and_derived_throughput.csv",
        index=False, encoding="utf-8-sig",
    )

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_derived_tables()
    for metric in ["throughput", "time"]:
        for statistic in ["median", "avg"]:
            for algorithm in FILES:
                plot_algorithm(algorithm, metric, statistic)
            plot_combined(metric, statistic)

if __name__ == "__main__":
    main()

