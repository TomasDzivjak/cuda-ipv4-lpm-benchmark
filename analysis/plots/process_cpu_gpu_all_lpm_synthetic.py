from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "common"))
from paths import cpu_dir24_8_results_dir, cpu_multibit_trie_results_dir, cuda_dir24_8_results_dir, cuda_multibit_trie_results_dir, figure_results_dir, hybrid_dir24_8_results_dir
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# VIZUÁLNY ŠTÝL GRAFOV PRE VKLADANIE DO BP
# ============================================================
# Väčšie fonty sú zvolené kvôli čitateľnosti po vložení grafu
# do dokumentu vo formáte A4. Pri potrebe jemného doladenia stačí
# upraviť hodnoty nižšie.
PLOT_TITLE_FONTSIZE = 18
PLOT_LABEL_FONTSIZE = 16
PLOT_TICK_FONTSIZE = 13
PLOT_LEGEND_FONTSIZE = 13
PLOT_LEGEND_TITLE_FONTSIZE = 14
PLOT_MARKER_SIZE = 7

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

import numpy as np


# ============================================================
# METODICKÁ VOĽBA PRE FINÁLNE GRAFY
# ============================================================
# CPU: medián času modelu dávkového CPU spracovania.
# GPU: medián času ČJ a medián času ČJP.
# Priepustnosť sa jednotne odvodzuje z použitého mediánového času:
#     Mpps = ip_count / (time_ms * 1000)
# Uložené throughput stĺpce sa vo finálnych grafoch nepoužívajú.
# ============================================================

# ============================================================
# KONFIGURÁCIA
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

BASE_DIR = figure_results_dir()
WORK_DIR = BASE_DIR

# Výstupný priečinok pre syntetickú prevádzku
OUTPUT_DIR = WORK_DIR / "graphs_cpu_vs_gpu_synthetic_median_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CPU_DIR24_DIR = cpu_dir24_8_results_dir()
CPU_MULTIBIT_DIR = cpu_multibit_trie_results_dir()

# Áno, priečinok sa volá CUDA DUR_24_8
GPU_DIR24_DIR = cuda_dir24_8_results_dir()
GPU_MULTIBIT_DIR = cuda_multibit_trie_results_dir()

CPU_DIR24_FILE = CPU_DIR24_DIR / "cpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv"
CPU_MULTIBIT_FILE = CPU_MULTIBIT_DIR / "cpu_multibit_trie_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv"

GPU_DIR24_FILE = GPU_DIR24_DIR / (
    "gpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_summary.csv")

GPU_MULTIBIT_FILE = GPU_MULTIBIT_DIR / "gpu_multibit_trie_lpm_synthetic_ids1_2_3_4_5_6_7_summary.csv"


# ============================================================
# POMOCNÉ FUNKCIE
# ============================================================

def config_short_label(row: pd.Series) -> str:
    """
    Vytvorí krátky popis konfigurácie vo formáte napr.:
    10K / 200K
    50K / 1M
    50K / 1,5M
    """
    fib = int(row["fib_size"])
    ip = int(row["ip_count"])

    # Niektoré generátory môžu ukladať o 1 prefix navyše.
    fib_base = fib - 1 if fib in (10001, 20001, 40001, 50001) else fib

    if fib_base >= 1000:
        fib_txt = f"{fib_base // 1000}K"
    else:
        fib_txt = str(fib_base)

    if ip == 1_000_000:
        ip_txt = "1M"
    elif ip == 1_500_000:
        ip_txt = "1,5M"
    elif ip % 1_000_000 == 0:
        ip_txt = f"{ip // 1_000_000}M"
    elif ip >= 1000:
        ip_txt = f"{ip // 1000}K"
    else:
        ip_txt = str(ip)

    return f"{fib_txt} FIB / {ip_txt} IP"


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: chýbajú stĺpce {missing}\n"
            f"Dostupné stĺpce: {list(df.columns)}"
        )


def load_csv(path: Path, label: str) -> pd.DataFrame:
    """
    Primárne načíta CSV z benchmarkového projektu. Pri prenose skriptu
    spolu so vstupmi použije rovnomenný CSV súbor vedľa skriptu.
    """
    resolved_path = path if path.exists() else SCRIPT_DIR / path.name

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{label} CSV neexistuje ani v projektovom priečinku, ani vedľa skriptu:\n"
            f" - {path}\n"
            f" - {SCRIPT_DIR / path.name}"
        )

    df = pd.read_csv(resolved_path).copy()

    if df.empty:
        raise ValueError(f"{label} CSV je prázdne:\n{resolved_path}")

    return df


def sort_configurations(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["fib_size", "ip_count"]).reset_index(drop=True)


# ============================================================
# NAČÍTANIE CPU DÁT
# ============================================================

def load_cpu_dir24_data() -> pd.DataFrame:
    df = load_csv(CPU_DIR24_FILE, "CPU DIR24/8")

    required = ["fib_size", "ip_count", "router_batch_median_time_ms"]
    require_columns(df, required, "CPU DIR24/8 CSV")

    df["lookup_time_ms_norm"] = pd.to_numeric(df["router_batch_median_time_ms"], errors="coerce")
    df["throughput_mpps_norm"] = pd.to_numeric(df["ip_count"], errors="coerce") / (df["lookup_time_ms_norm"] * 1000.0)
    df["config_short"] = df.apply(config_short_label, axis=1)

    # Ak by CSV obsahovalo viacero vlákien, necháme iba 12T.
    if "threads" in df.columns:
        df = df[pd.to_numeric(df["threads"], errors="coerce") == 12].copy()

    df["platform"] = "CPU DIR24/8 (12 vlákien)"
    return sort_configurations(df)


def load_cpu_multibit_data() -> pd.DataFrame:
    df = load_csv(CPU_MULTIBIT_FILE, "CPU Multibit Trie")

    required = ["fib_size", "ip_count", "router_batch_median_time_ms"]
    require_columns(df, required, "CPU Multibit Trie CSV")

    df["lookup_time_ms_norm"] = pd.to_numeric(df["router_batch_median_time_ms"], errors="coerce")
    df["throughput_mpps_norm"] = pd.to_numeric(df["ip_count"], errors="coerce") / (df["lookup_time_ms_norm"] * 1000.0)
    df["config_short"] = df.apply(config_short_label, axis=1)

    # Ak by CSV obsahovalo viacero vlákien, necháme iba 12T.
    if "threads" in df.columns:
        df = df[pd.to_numeric(df["threads"], errors="coerce") == 12].copy()

    df["platform"] = "CPU Multibit Trie stride 4 (12 vlákien)"
    return sort_configurations(df)


# ============================================================
# NAČÍTANIE GPU DÁT
# ============================================================

def load_gpu_common(path: Path, label: str) -> pd.DataFrame:
    df = load_csv(path, label)

    required = [
        "fib_size",
        "ip_count",
        "kernel_lookup_median_ms",
        "transfer_plus_kernel_median_ms",
    ]
    require_columns(df, required, f"{label} CSV")

    df["kernel_lookup_avg_ms_norm"] = pd.to_numeric(df["kernel_lookup_median_ms"], errors="coerce")
    df["kernel_lookup_throughput_mpps_norm"] = (
        pd.to_numeric(df["ip_count"], errors="coerce") /
        (df["kernel_lookup_avg_ms_norm"] * 1000.0)
    )

    df["transfer_plus_kernel_avg_ms_norm"] = pd.to_numeric(
        df["transfer_plus_kernel_median_ms"],
        errors="coerce"
    )
    df["transfer_plus_kernel_throughput_mpps_norm"] = (
        pd.to_numeric(df["ip_count"], errors="coerce") /
        (df["transfer_plus_kernel_avg_ms_norm"] * 1000.0)
    )

    df["config_short"] = df.apply(config_short_label, axis=1)
    df["platform"] = label

    return sort_configurations(df)


def load_gpu_dir24_data() -> pd.DataFrame:
    return load_gpu_common(GPU_DIR24_FILE, "CUDA DIR24/8")


def load_gpu_multibit_data() -> pd.DataFrame:
    return load_gpu_common(GPU_MULTIBIT_FILE, "CUDA Multibit Trie stride 4")


# ============================================================
# ZLÚČENÁ TABUĽKA
# ============================================================

def build_combined_table(
    cpu_dir24_df: pd.DataFrame,
    cpu_multibit_df: pd.DataFrame,
    gpu_dir24_df: pd.DataFrame,
    gpu_multibit_df: pd.DataFrame,
) -> pd.DataFrame:

    cpu_dir24_part = cpu_dir24_df[
        ["fib_size", "ip_count", "config_short", "lookup_time_ms_norm", "throughput_mpps_norm"]
    ].rename(
        columns={
            "lookup_time_ms_norm": "cpu_dir24_router_batch_median_time_ms",
            "throughput_mpps_norm": "cpu_dir24_router_batch_median_throughput_derived_mpps",
        }
    )

    cpu_multibit_part = cpu_multibit_df[
        ["fib_size", "ip_count", "config_short", "lookup_time_ms_norm", "throughput_mpps_norm"]
    ].rename(
        columns={
            "lookup_time_ms_norm": "cpu_multibit_router_batch_median_time_ms",
            "throughput_mpps_norm": "cpu_multibit_router_batch_median_throughput_derived_mpps",
        }
    )

    gpu_dir24_part = gpu_dir24_df[
        [
            "fib_size",
            "ip_count",
            "config_short",
            "kernel_lookup_avg_ms_norm",
            "kernel_lookup_throughput_mpps_norm",
            "transfer_plus_kernel_avg_ms_norm",
            "transfer_plus_kernel_throughput_mpps_norm",
        ]
    ].rename(
        columns={
            "kernel_lookup_avg_ms_norm": "gpu_dir24_kernel_median_time_ms",
            "kernel_lookup_throughput_mpps_norm": "gpu_dir24_kernel_median_throughput_derived_mpps",
            "transfer_plus_kernel_avg_ms_norm": "gpu_dir24_transfer_plus_kernel_median_time_ms",
            "transfer_plus_kernel_throughput_mpps_norm": "gpu_dir24_transfer_plus_kernel_median_throughput_derived_mpps",
        }
    )

    gpu_multibit_part = gpu_multibit_df[
        [
            "fib_size",
            "ip_count",
            "config_short",
            "kernel_lookup_avg_ms_norm",
            "kernel_lookup_throughput_mpps_norm",
            "transfer_plus_kernel_avg_ms_norm",
            "transfer_plus_kernel_throughput_mpps_norm",
        ]
    ].rename(
        columns={
            "kernel_lookup_avg_ms_norm": "gpu_multibit_kernel_median_time_ms",
            "kernel_lookup_throughput_mpps_norm": "gpu_multibit_kernel_median_throughput_derived_mpps",
            "transfer_plus_kernel_avg_ms_norm": "gpu_multibit_transfer_plus_kernel_median_time_ms",
            "transfer_plus_kernel_throughput_mpps_norm": "gpu_multibit_transfer_plus_kernel_median_throughput_derived_mpps",
        }
    )

    combined = pd.merge(
        cpu_dir24_part,
        cpu_multibit_part,
        on=["fib_size", "ip_count", "config_short"],
        how="outer",
    )

    combined = pd.merge(
        combined,
        gpu_dir24_part,
        on=["fib_size", "ip_count", "config_short"],
        how="outer",
    )

    combined = pd.merge(
        combined,
        gpu_multibit_part,
        on=["fib_size", "ip_count", "config_short"],
        how="outer",
    )

    combined = combined.sort_values(["fib_size", "ip_count"]).reset_index(drop=True)
    return combined


def save_combined_table(df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / "cpu_gpu_all_synthetic_combined_median_derived.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Uložená tabuľka: {out_path}")


# ============================================================
# VIZUALIZÁCIA
# ============================================================

def apply_common_style(ylabel: str, title: str) -> None:
    plt.xlabel("Konfigurácia datasetu", fontsize=PLOT_LABEL_FONTSIZE, labelpad=10)
    plt.ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE, labelpad=10)
    plt.title(title, fontsize=PLOT_TITLE_FONTSIZE, pad=14, wrap=True)
    plt.xticks(fontsize=PLOT_TICK_FONTSIZE)
    plt.yticks(fontsize=PLOT_TICK_FONTSIZE)
    plt.tick_params(axis="both", which="major", labelsize=PLOT_TICK_FONTSIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(
        title="Implementácia",
        fontsize=PLOT_LEGEND_FONTSIZE,
        title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
        frameon=True,
    )
    plt.tight_layout()


def get_x_axis(combined: pd.DataFrame):
    x = np.arange(len(combined))
    labels = combined["config_short"].tolist()
    return x, labels


def save_current_plot(filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Uložený graf: {out_path}")


def plot_throughput_line(combined: pd.DataFrame) -> None:
    x, labels = get_x_axis(combined)

    plt.figure(figsize=(14, 7))

    plt.plot(x, combined["cpu_dir24_router_batch_median_throughput_derived_mpps"], marker="o", linewidth=2, label="CPU DIR24/8 (12 vlákien)")
    plt.plot(x, combined["cpu_multibit_router_batch_median_throughput_derived_mpps"], marker="o", linewidth=2, label="CPU Multibit Trie stride 4 (12 vlákien)")

    plt.plot(x, combined["gpu_dir24_kernel_median_throughput_derived_mpps"], marker="o", linewidth=2, label="CUDA DIR24/8 - ČJ")
    plt.plot(x, combined["gpu_dir24_transfer_plus_kernel_median_throughput_derived_mpps"], marker="o", linewidth=2, label="CUDA DIR24/8 - ČJP")

    plt.plot(x, combined["gpu_multibit_kernel_median_throughput_derived_mpps"], marker="o", linewidth=2, label="CUDA Multibit Trie stride 4 - ČJ")
    plt.plot(x, combined["gpu_multibit_transfer_plus_kernel_median_throughput_derived_mpps"], marker="o", linewidth=2, label="CUDA Multibit Trie stride 4 - ČJP")

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Priepustnosť [Mpps]",
        "CPU vs GPU LPM - syntetická prevádzka - porovnanie priepustnosti odvodenej z mediánového času pre všetky konfigurácie"
    )

    save_current_plot("cpu_gpu_all_synthetic_priepustnost_vsetky_konfiguracie.png")


def plot_lookup_time_line(combined: pd.DataFrame) -> None:
    x, labels = get_x_axis(combined)

    plt.figure(figsize=(14, 7))

    plt.plot(x, combined["cpu_dir24_router_batch_median_time_ms"], marker="o", linewidth=2, label="CPU DIR24/8 (12 vlákien)")
    plt.plot(x, combined["cpu_multibit_router_batch_median_time_ms"], marker="o", linewidth=2, label="CPU Multibit Trie stride 4 (12 vlákien)")

    plt.plot(x, combined["gpu_dir24_kernel_median_time_ms"], marker="o", linewidth=2, label="CUDA DIR24/8 - ČJ")
    plt.plot(x, combined["gpu_dir24_transfer_plus_kernel_median_time_ms"], marker="o", linewidth=2, label="CUDA DIR24/8 - ČJP")

    plt.plot(x, combined["gpu_multibit_kernel_median_time_ms"], marker="o", linewidth=2, label="CUDA Multibit Trie stride 4 - ČJ")
    plt.plot(x, combined["gpu_multibit_transfer_plus_kernel_median_time_ms"], marker="o", linewidth=2, label="CUDA Multibit Trie stride 4 - ČJP")

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Čas spracovania dávky [ms]",
        "CPU vs GPU LPM - syntetická prevádzka - porovnanie mediánového času spracovania dávky pre všetky konfigurácie"
    )

    save_current_plot("cpu_gpu_all_synthetic_cas_lookupu_vsetky_konfiguracie.png")


def plot_throughput_bar(combined: pd.DataFrame) -> None:
    x, labels = get_x_axis(combined)
    width = 0.13

    plt.figure(figsize=(16, 8))

    plt.bar(x - 2.5 * width, combined["cpu_dir24_router_batch_median_throughput_derived_mpps"], width=width, label="CPU DIR24/8 (12 vlákien)")
    plt.bar(x - 1.5 * width, combined["cpu_multibit_router_batch_median_throughput_derived_mpps"], width=width, label="CPU Multibit Trie stride 4 (12 vlákien)")

    plt.bar(x - 0.5 * width, combined["gpu_dir24_kernel_median_throughput_derived_mpps"], width=width, label="CUDA DIR24/8 - ČJ")
    plt.bar(x + 0.5 * width, combined["gpu_dir24_transfer_plus_kernel_median_throughput_derived_mpps"], width=width, label="CUDA DIR24/8 - ČJP")

    plt.bar(x + 1.5 * width, combined["gpu_multibit_kernel_median_throughput_derived_mpps"], width=width, label="CUDA Multibit Trie stride 4 - ČJ")
    plt.bar(x + 2.5 * width, combined["gpu_multibit_transfer_plus_kernel_median_throughput_derived_mpps"], width=width, label="CUDA Multibit Trie stride 4 - ČJP")

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Priepustnosť [Mpps]",
        "CPU vs GPU LPM - syntetická prevádzka - stĺpcové porovnanie priepustnosti odvodenej z mediánového času"
    )

    save_current_plot("cpu_gpu_all_synthetic_priepustnost_stlpcovy_graf.png")


def plot_lookup_time_bar(combined: pd.DataFrame) -> None:
    x, labels = get_x_axis(combined)
    width = 0.13

    plt.figure(figsize=(16, 8))

    plt.bar(x - 2.5 * width, combined["cpu_dir24_router_batch_median_time_ms"], width=width, label="CPU DIR24/8 (12 vlákien)")
    plt.bar(x - 1.5 * width, combined["cpu_multibit_router_batch_median_time_ms"], width=width, label="CPU Multibit Trie stride 4 (12 vlákien)")

    plt.bar(x - 0.5 * width, combined["gpu_dir24_kernel_median_time_ms"], width=width, label="CUDA DIR24/8 - ČJ")
    plt.bar(x + 0.5 * width, combined["gpu_dir24_transfer_plus_kernel_median_time_ms"], width=width, label="CUDA DIR24/8 - ČJP")

    plt.bar(x + 1.5 * width, combined["gpu_multibit_kernel_median_time_ms"], width=width, label="CUDA Multibit Trie stride 4 - ČJ")
    plt.bar(x + 2.5 * width, combined["gpu_multibit_transfer_plus_kernel_median_time_ms"], width=width, label="CUDA Multibit Trie stride 4 - ČJP")

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Čas spracovania dávky [ms]",
        "CPU vs GPU LPM - syntetická prevádzka - stĺpcové porovnanie mediánového času spracovania dávky"
    )

    save_current_plot("cpu_gpu_all_synthetic_cas_lookupu_stlpcovy_graf.png")

def plot_throughput_bar_without_kernel_only(combined: pd.DataFrame) -> None:
    """
    Stĺpcový graf priepustnosti bez GPU ČJ meraní.
    Zostáva iba CPU vs GPU ČJP.
    Tento graf je vhodnejší na vizuálne porovnanie reálnejšieho GPU scenára s CPU.
    """
    x, labels = get_x_axis(combined)
    width = 0.20

    plt.figure(figsize=(16, 8))

    plt.bar(
        x - 1.5 * width,
        combined["cpu_dir24_router_batch_median_throughput_derived_mpps"],
        width=width,
        label="CPU DIR24/8 (12 vlákien)"
    )

    plt.bar(
        x - 0.5 * width,
        combined["cpu_multibit_router_batch_median_throughput_derived_mpps"],
        width=width,
        label="CPU Multibit Trie stride 4 (12 vlákien)"
    )

    plt.bar(
        x + 0.5 * width,
        combined["gpu_dir24_transfer_plus_kernel_median_throughput_derived_mpps"],
        width=width,
        label="CUDA DIR24/8 - ČJP"
    )

    plt.bar(
        x + 1.5 * width,
        combined["gpu_multibit_transfer_plus_kernel_median_throughput_derived_mpps"],
        width=width,
        label="CUDA Multibit Trie stride 4 - ČJP"
    )

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Priepustnosť [Mpps]",
        "CPU vs GPU LPM - syntetická prevádzka - porovnanie priepustnosti odvodenej z mediánového času bez GPU ČJ meraní"
    )

    save_current_plot("cpu_gpu_all_synthetic_priepustnost_bez_kernel_only_stlpcovy_graf.png")


def plot_lookup_time_bar_without_kernel_only(combined: pd.DataFrame) -> None:
    """
    Stĺpcový graf času lookupu bez GPU ČJ meraní.
    Zostáva iba CPU vs GPU ČJP.
    Tento graf lepšie ukazuje rozdiely medzi CPU a celkovým GPU časom vrátane prenosov.
    """
    x, labels = get_x_axis(combined)
    width = 0.20

    plt.figure(figsize=(16, 8))

    plt.bar(
        x - 1.5 * width,
        combined["cpu_dir24_router_batch_median_time_ms"],
        width=width,
        label="CPU DIR24/8 (12 vlákien)"
    )

    plt.bar(
        x - 0.5 * width,
        combined["cpu_multibit_router_batch_median_time_ms"],
        width=width,
        label="CPU Multibit Trie stride 4 (12 vlákien)"
    )

    plt.bar(
        x + 0.5 * width,
        combined["gpu_dir24_transfer_plus_kernel_median_time_ms"],
        width=width,
        label="CUDA DIR24/8 - ČJP"
    )

    plt.bar(
        x + 1.5 * width,
        combined["gpu_multibit_transfer_plus_kernel_median_time_ms"],
        width=width,
        label="CUDA Multibit Trie stride 4 - ČJP"
    )

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Čas spracovania dávky [ms]",
        "CPU vs GPU LPM - syntetická prevádzka - porovnanie mediánového času spracovania dávky bez GPU ČJ meraní"
    )

    save_current_plot("cpu_gpu_all_synthetic_cas_lookupu_bez_kernel_only_stlpcovy_graf.png")

def plot_throughput_line_without_kernel_only(combined: pd.DataFrame) -> None:
    """
    Čiarový graf priepustnosti bez GPU ČJ meraní.
    Zostáva iba CPU vs GPU ČJP.
    """
    x, labels = get_x_axis(combined)

    plt.figure(figsize=(14, 7))

    plt.plot(
        x,
        combined["cpu_dir24_router_batch_median_throughput_derived_mpps"],
        marker="o",
        linewidth=2,
        label="CPU DIR24/8 (12 vlákien)"
    )

    plt.plot(
        x,
        combined["cpu_multibit_router_batch_median_throughput_derived_mpps"],
        marker="o",
        linewidth=2,
        label="CPU Multibit Trie stride 4 (12 vlákien)"
    )

    plt.plot(
        x,
        combined["gpu_dir24_transfer_plus_kernel_median_throughput_derived_mpps"],
        marker="o",
        linewidth=2,
        label="CUDA DIR24/8 - ČJP"
    )

    plt.plot(
        x,
        combined["gpu_multibit_transfer_plus_kernel_median_throughput_derived_mpps"],
        marker="o",
        linewidth=2,
        label="CUDA Multibit Trie stride 4 - ČJP"
    )

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Priepustnosť [Mpps]",
        "CPU vs GPU LPM - syntetická prevádzka - porovnanie priepustnosti odvodenej z mediánového času bez GPU ČJ meraní"
    )

    save_current_plot("cpu_gpu_all_synthetic_priepustnost_bez_kernel_only_ciarovy_graf.png")


def plot_lookup_time_line_without_kernel_only(combined: pd.DataFrame) -> None:
    """
    Čiarový graf času lookupu bez GPU ČJ meraní.
    Zostáva iba CPU vs GPU ČJP.
    """
    x, labels = get_x_axis(combined)

    plt.figure(figsize=(14, 7))

    plt.plot(
        x,
        combined["cpu_dir24_router_batch_median_time_ms"],
        marker="o",
        linewidth=2,
        label="CPU DIR24/8 (12 vlákien)"
    )

    plt.plot(
        x,
        combined["cpu_multibit_router_batch_median_time_ms"],
        marker="o",
        linewidth=2,
        label="CPU Multibit Trie stride 4 (12 vlákien)"
    )

    plt.plot(
        x,
        combined["gpu_dir24_transfer_plus_kernel_median_time_ms"],
        marker="o",
        linewidth=2,
        label="CUDA DIR24/8 - ČJP"
    )

    plt.plot(
        x,
        combined["gpu_multibit_transfer_plus_kernel_median_time_ms"],
        marker="o",
        linewidth=2,
        label="CUDA Multibit Trie stride 4 - ČJP"
    )

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)

    apply_common_style(
        "Čas spracovania dávky [ms]",
        "CPU vs GPU LPM - syntetická prevádzka - porovnanie mediánového času spracovania dávky bez GPU ČJ meraní"
    )

    save_current_plot("cpu_gpu_all_synthetic_cas_lookupu_bez_kernel_only_ciarovy_graf.png")

def print_summary(combined: pd.DataFrame) -> None:
    print("\nPrehľad konfigurácií:")
    for _, row in combined.iterrows():
        print(
            f" - {row['config_short']}: "
            f"CPU DIR24/8 = {row['cpu_dir24_router_batch_median_throughput_derived_mpps']:.3f} Mpps, "
            f"CPU Multibit = {row['cpu_multibit_router_batch_median_throughput_derived_mpps']:.3f} Mpps, "
            f"CUDA DIR24/8 ČJ = {row['gpu_dir24_kernel_median_throughput_derived_mpps']:.3f} Mpps, "
            f"CUDA DIR24/8 ČJP = {row['gpu_dir24_transfer_plus_kernel_median_throughput_derived_mpps']:.3f} Mpps, "
            f"CUDA Multibit ČJ = {row['gpu_multibit_kernel_median_throughput_derived_mpps']:.3f} Mpps, "
            f"CUDA Multibit ČJP = {row['gpu_multibit_transfer_plus_kernel_median_throughput_derived_mpps']:.3f} Mpps"
        )


# ============================================================
# HLAVNÝ PROGRAM
# ============================================================

def main() -> None:
    print("Typ prevádzky: SYNTHETIC / syntetická prevádzka")
    print("Použitá metrika: MEDIÁN času; priepustnosť je odvodená z rovnakého mediánového času.")
    print(f"Pracovný priečinok: {WORK_DIR}")
    print(f"Výstupný priečinok: {OUTPUT_DIR}")

    print("\nPoužité vstupné súbory:")
    print(f"CPU DIR24/8:        {CPU_DIR24_FILE}")
    print(f"CPU Multibit Trie:  {CPU_MULTIBIT_FILE}")
    print(f"GPU DIR24/8:        {GPU_DIR24_FILE}")
    print(f"GPU Multibit Trie:  {GPU_MULTIBIT_FILE}")

    cpu_dir24_df = load_cpu_dir24_data()
    cpu_multibit_df = load_cpu_multibit_data()
    gpu_dir24_df = load_gpu_dir24_data()
    gpu_multibit_df = load_gpu_multibit_data()

    combined = build_combined_table(
        cpu_dir24_df,
        cpu_multibit_df,
        gpu_dir24_df,
        gpu_multibit_df
    )

    print("\nPočet načítaných riadkov:")
    print(f"CPU DIR24/8:        {len(cpu_dir24_df)}")
    print(f"CPU Multibit Trie:  {len(cpu_multibit_df)}")
    print(f"GPU DIR24/8:        {len(gpu_dir24_df)}")
    print(f"GPU Multibit Trie:  {len(gpu_multibit_df)}")
    print(f"Zlúčená tabuľka:    {len(combined)}")

    save_combined_table(combined)

    print("\nVytváram grafy...")

    # Všetky veľkosti tabuliek sú v jednom grafe.
    # 50K-only grafy sú zámerne odstránené.
    plot_throughput_line(combined)
    plot_lookup_time_line(combined)
    plot_throughput_bar(combined)
    plot_lookup_time_bar(combined)
    plot_throughput_bar_without_kernel_only(combined)
    plot_lookup_time_bar_without_kernel_only(combined)
    plot_throughput_line_without_kernel_only(combined)
    plot_lookup_time_line_without_kernel_only(combined)
    print_summary(combined)

    print("\nHotovo.")
    print(f"Výstupy sú uložené v priečinku:\n{OUTPUT_DIR}")

    print("\nVygenerované súbory:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
