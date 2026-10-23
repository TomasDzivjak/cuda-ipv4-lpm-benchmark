from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "common"))
from paths import cpu_dir24_8_results_dir, cpu_multibit_trie_results_dir, cuda_dir24_8_results_dir, cuda_multibit_trie_results_dir, figure_results_dir, hybrid_dir24_8_results_dir
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# DIR24/8: CPU vs GPU CUDA vs HYBRID
# Porovnanie benchmarkov pre zrýchlenie smerovania
#
# Grafy:
#   - CPU DIR24/8 (12 vlákien)
#   - GPU DIR24/8 CUDA (ČJP)
#   - HYBRID DIR24/8 bez rozhodovania
#   - HYBRID DIR24/8 s rozhodovaním
#
# Typy prevádzky:
#   - Syntetická prevádzka
#   - Enterprise prevádzka
#   - Hotspot prevádzka
#
# Metodika:
#   - GPU používa výlučne transfer + kernel, nie ČJ a nie total end-to-end.
#   - HYBRID sa kreslí samostatne bez rozhodovania aj s rozhodovaním.
#   - Priepustnosť sa pre finálne grafy počíta z vybraného času:
#       Mpps = ip_count / (čas_ms * 1000)
#     Tým sa zabraňuje miešaniu average(time) s average(per-run throughput).
#   - METRIC_MODE="median" používa mediánový čas, METRIC_MODE="avg" priemerný čas.
# ============================================================


# ============================================================
# KONFIGURÁCIA
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Priečinok, v ktorom je tento porovnávací skript. Ak skript presunieš do pôvodného
# Windows priečinka, výstupy sa budú ukladať vedľa neho.
BASE_DIR = figure_results_dir()
WORK_DIR = BASE_DIR

# Povolené hodnoty:
#   "median" - odporúčané pre finálne grafy, používa medián času
#   "avg"    - používa priemer času
METRIC_MODE = "median"

if METRIC_MODE not in {"median", "avg"}:
    raise ValueError('METRIC_MODE musí byť "median" alebo "avg".')

OUTPUT_DIR = WORK_DIR / f"graphs_dir24_8_cpu_gpu_cuda_hybrid_all_traffic_v8_{METRIC_MODE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ak niektorá implementácia nemá pre danú prevádzku konkrétnu konfiguráciu,
# táto voľba ju z osi grafu odstráni. Tým sa zabráni vizuálnym dieram/zalomeniam
# v čiarových grafoch a porovnávajú sa iba konfigurácie dostupné pre všetky 4 série.
# Pri syntetickej prevádzke CPU súbor ids1_2_3_4_6_7 neobsahuje 1,5M konfiguráciu,
# preto sa 1,5M bod z porovnávacích grafov automaticky vynechá.
PLOT_ONLY_COMMON_CONFIGS = True


CPU_DIR24_DIR = hybrid_dir24_8_results_dir()
GPU_DIR24_DIR = cuda_dir24_8_results_dir()
HYBRID_DIR24_DIR = hybrid_dir24_8_results_dir()

PLATFORM_CPU = "CPU DIR24/8 (12 vlákien)"
PLATFORM_GPU = "GPU DIR24/8 CUDA (ČJP)"
PLATFORM_GPU_KERNEL_ONLY = "GPU DIR24/8 CUDA (ČJ)"
PLATFORM_HYBRID_WITHOUT_DECISION = "HYBRID DIR24/8 bez rozhodovania"
PLATFORM_HYBRID_WITH_DECISION = "HYBRID DIR24/8 s rozhodovaním"

# Pôvodná sada platforiem - zachováva pôvodné grafy a pôvodné názvy súborov.
PLOT_PLATFORMS = [
    PLATFORM_CPU,
    PLATFORM_GPU,
    PLATFORM_HYBRID_WITHOUT_DECISION,
    PLATFORM_HYBRID_WITH_DECISION,
]

# Rozšírená sada platforiem - pridáva hypotetické GPU ČJ meranie.
PLOT_PLATFORMS_WITH_KERNEL_ONLY = [
    PLATFORM_CPU,
    PLATFORM_GPU,
    PLATFORM_GPU_KERNEL_ONLY,
    PLATFORM_HYBRID_WITHOUT_DECISION,
    PLATFORM_HYBRID_WITH_DECISION,
]

PLATFORM_ORDER = {
    PLATFORM_CPU: 1,
    PLATFORM_GPU: 2,
    PLATFORM_GPU_KERNEL_ONLY: 3,
    PLATFORM_HYBRID_WITHOUT_DECISION: 4,
    PLATFORM_HYBRID_WITH_DECISION: 5,
}

DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
PLATFORM_COLORS = {
    PLATFORM_CPU: DEFAULT_COLORS[0],
    PLATFORM_GPU: DEFAULT_COLORS[1],
    PLATFORM_GPU_KERNEL_ONLY: DEFAULT_COLORS[4] if len(DEFAULT_COLORS) > 4 else DEFAULT_COLORS[0],
    PLATFORM_HYBRID_WITHOUT_DECISION: DEFAULT_COLORS[2],
    PLATFORM_HYBRID_WITH_DECISION: DEFAULT_COLORS[3],
}

PLATFORM_FILENAME_SLUGS = {
    PLATFORM_CPU: "cpu",
    PLATFORM_GPU: "gpu_transfer",
    PLATFORM_GPU_KERNEL_ONLY: "gpu_kernel_only",
    PLATFORM_HYBRID_WITHOUT_DECISION: "hybrid_bez_rozh",
    PLATFORM_HYBRID_WITH_DECISION: "hybrid_s_rozh",
}


TRAFFIC_CONFIGS = {
    "synthetic": {
        "label": "Syntetická prevádzka",
        "filename_slug": "synteticka_prevadzka",
        "traffic_order": 1,
        "cpu_file": CPU_DIR24_DIR / "cpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv",
        "gpu_file": GPU_DIR24_DIR / "gpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_summary.csv",
        "hybrid_file": HYBRID_DIR24_DIR / "hybrid_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv",
    },
    "enterprise": {
        "label": "Enterprise prevádzka",
        "filename_slug": "enterprise_prevadzka",
        "traffic_order": 2,
        "cpu_file": CPU_DIR24_DIR / "cpu_dir24_8_lpm_enterprise_ids1_2_3_4_5_6_7_threads12_summary.csv",
        "gpu_file": GPU_DIR24_DIR / "gpu_dir24_8_lpm_enterprise_ids1_2_3_4_5_6_7_summary.csv",
        "hybrid_file": HYBRID_DIR24_DIR / "hybrid_dir24_8_lpm_enterprise_ids1_2_3_4_5_6_7_threads12_summary.csv",
    },
    "hotspot": {
        "label": "Hotspot prevádzka",
        "filename_slug": "hotspot_prevadzka",
        "traffic_order": 3,
        "cpu_file": CPU_DIR24_DIR / "cpu_dir24_8_lpm_hotspot_ids1_2_3_4_5_6_7_threads12_summary.csv",
        "gpu_file": GPU_DIR24_DIR / "gpu_dir24_8_lpm_hotspot_ids1_2_3_4_5_6_7_summary.csv",
        "hybrid_file": HYBRID_DIR24_DIR / "hybrid_dir24_8_lpm_hotspot_ids1_2_3_4_5_6_7_threads12_summary.csv",
    },
}


# Očakávané konfigurácie pre nový syntetický summary súbor:
# cpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv
# Nový súbor obsahuje aj dataset 1,5M IP, teda konfiguráciu id5.
EXPECTED_CPU_SYNTHETIC_CONFIGS = {
    (10_000, 200_000),
    (20_000, 400_000),
    (40_000, 800_000),
    (50_000, 1_000_000),
    (50_000, 1_500_000),
    (50_000, 2_000_000),
    (50_000, 3_000_000),
}


# ============================================================
# ČÍTANIE A NORMALIZÁCIA CSV
# ============================================================

def resolve_existing_path(path: Path) -> Path:
    """
    Primárne používa pôvodnú Windows cestu. Ak sú CSV súbory uložené vedľa skriptu,
    použije lokálny súbor s rovnakým názvom. To zjednodušuje testovanie a presúvanie skriptu.
    """
    if path.exists():
        return path

    local = SCRIPT_DIR / path.name
    if local.exists():
        return local

    work_local = WORK_DIR / path.name
    if work_local.exists():
        return work_local

    return path


def detect_separator(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                return ";" if line.count(";") > line.count(",") else ","
    return ","


def load_csv(path: Path, label: str) -> pd.DataFrame:
    path = resolve_existing_path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV neexistuje:\n{path}")

    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", engine="python").copy()
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise ValueError(f"{label} CSV je prázdne:\n{path}")

    return df


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: chýbajú stĺpce {missing}\n"
            f"Dostupné stĺpce: {list(df.columns)}"
        )


def first_existing_column(df: pd.DataFrame, candidates: list[str], label: str, purpose: str,
                          required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"{label}: nenašiel sa stĺpec pre {purpose}.\n"
            f"Hľadané možnosti: {candidates}\n"
            f"Dostupné stĺpce: {list(df.columns)}"
        )

    return None


def select_time_column(df: pd.DataFrame, label: str, avg_candidates: list[str], median_candidates: list[str]) -> str:
    """
    Vyberá časový stĺpec podľa METRIC_MODE. Priepustnosť sa následne počíta z tohto času,
    aby sa nemiešal medián času s priemerom priepustnosti.
    """
    if METRIC_MODE == "median":
        col = first_existing_column(df, median_candidates, label, "mediánový čas", required=False)
        if col is not None:
            return col

        fallback = first_existing_column(df, avg_candidates, label, "čas", required=True)
        warnings.warn(
            f"{label}: mediánový časový stĺpec neexistuje. Použije sa fallback '{fallback}'. "
            f"Ak chceš čisté mediánové grafy, musí byť v CSV uložený mediánový čas.",
            RuntimeWarning,
        )
        return fallback

    return first_existing_column(df, avg_candidates, label, "priemerný čas", required=True)


def to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_fib_size(value) -> int:
    fib = int(float(value))
    if fib in (10001, 20001, 40001, 50001):
        return fib - 1
    return fib


def normalize_ip_count(value) -> int:
    return int(float(value))


def throughput_from_time_mpps(ip_count: pd.Series, time_ms: pd.Series) -> pd.Series:
    time_ms = pd.to_numeric(time_ms, errors="coerce")
    ip_count = pd.to_numeric(ip_count, errors="coerce")
    out = ip_count / (time_ms * 1000.0)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def format_fib_label(fib_size: int) -> str:
    fib_size = normalize_fib_size(fib_size)
    if fib_size >= 1000:
        return f"{fib_size // 1000}K"
    return str(fib_size)


def format_ip_label(ip_count: int) -> str:
    ip_count = normalize_ip_count(ip_count)
    if ip_count == 1_500_000:
        return "1,5M"
    if ip_count >= 1_000_000:
        if ip_count % 1_000_000 == 0:
            return f"{ip_count // 1_000_000}M"
        return f"{ip_count / 1_000_000:.1f}M".replace(".", ",")
    if ip_count >= 1000:
        return f"{ip_count // 1000}K"
    return str(ip_count)


def make_config_short(fib_size: int, ip_count: int) -> str:
    return f"{format_fib_label(fib_size)} FIB / {format_ip_label(ip_count)} IP"


# ============================================================
# NAČÍTANIE CPU / GPU / HYBRID DÁT
# ============================================================

def build_common_frame(
    df: pd.DataFrame,
    traffic_key: str,
    traffic_label: str,
    platform: str,
    fib_col: str,
    ip_col: str,
    time_col: Optional[str],
    throughput_col: Optional[str] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    fib_raw = to_number(df[fib_col])
    ip_raw = to_number(df[ip_col])

    out["traffic_key"] = traffic_key
    out["traffic_label"] = traffic_label
    out["traffic_order"] = TRAFFIC_CONFIGS[traffic_key]["traffic_order"]
    out["platform"] = platform
    out["platform_order"] = PLATFORM_ORDER[platform]
    out["fib_size"] = fib_raw.apply(lambda x: normalize_fib_size(x) if pd.notna(x) else np.nan)
    out["ip_count"] = ip_raw.apply(lambda x: normalize_ip_count(x) if pd.notna(x) else np.nan)

    if time_col is not None:
        out["lookup_time_ms"] = to_number(df[time_col])
    else:
        out["lookup_time_ms"] = np.nan

    if throughput_col is not None:
        out["throughput_mpps"] = to_number(df[throughput_col])

        # Ak časový stĺpec pre danú metodiku nie je dostupný, dopočíta sa z priepustnosti.
        # Platí: Mpps = ip_count / (time_ms * 1000), teda time_ms = ip_count / (Mpps * 1000).
        missing_time = out["lookup_time_ms"].isna()
        out.loc[missing_time, "lookup_time_ms"] = (
            out.loc[missing_time, "ip_count"] / (out.loc[missing_time, "throughput_mpps"] * 1000.0)
        )
    else:
        out["throughput_mpps"] = throughput_from_time_mpps(out["ip_count"], out["lookup_time_ms"])

    out["source_time_column"] = time_col if time_col is not None else "derived_from_throughput"
    out["source_throughput_column"] = throughput_col if throughput_col is not None else "derived_from_time"
    out["metric_mode"] = METRIC_MODE

    before = len(out)
    out = out.dropna(
        subset=[
            "traffic_key",
            "traffic_label",
            "traffic_order",
            "platform",
            "platform_order",
            "fib_size",
            "ip_count",
            "lookup_time_ms",
            "throughput_mpps",
        ]
    ).copy()

    if len(out) == 0:
        print(f"\nVAROVANIE: {platform} / {traffic_label} nemá po prevode žiadne platné riadky.")
        print(f"Pôvodný počet riadkov: {before}")
        print("Ukážka vstupných hodnôt:")
        preview_cols = [fib_col, ip_col]
        if time_col is not None:
            preview_cols.append(time_col)
        if throughput_col is not None:
            preview_cols.append(throughput_col)
        print(df[preview_cols].head().to_string(index=False))

    out["fib_size"] = out["fib_size"].astype(int)
    out["ip_count"] = out["ip_count"].astype(int)
    out["traffic_order"] = out["traffic_order"].astype(int)
    out["platform_order"] = out["platform_order"].astype(int)

    return out


def load_cpu_dir24(path: Path, traffic_key: str, traffic_label: str) -> pd.DataFrame:
    label = f"CPU DIR24/8 - {traffic_label}"
    df = load_csv(path, label)
    require_columns(df, ["fib_size", "ip_count"], label)

    if "threads" in df.columns:
        df = df[to_number(df["threads"]) == 12].copy()

    # CPU sa má porovnávať rovnakou štatistikou ako GPU a HYBRID.
    # Preto sa v režime METRIC_MODE="median" nepoužíva avg throughput stĺpec
    # router_batch_throughput_mpps, ale mediánový čas dávkového CPU spracovania.
    # Priepustnosť sa následne dopočíta jednotne: Mpps = ip_count / (čas_ms * 1000).
    time_col = select_time_column(
        df=df,
        label=label + " CPU spracovanie",
        avg_candidates=[
            "router_batch_time_ms",
            "router_batch_avg_time_ms",
            "router_batch_avg_ms",
            "lookup_time_ms",
            "avg_lookup_time_ms",
            "processing_avg_ms",
        ],
        median_candidates=[
            "router_batch_median_time_ms",
            "router_batch_median_ms",
            "median_time_ms",
        ],
    )

    if METRIC_MODE == "median" and time_col not in {
        "router_batch_median_time_ms",
        "router_batch_median_ms",
        "median_time_ms",
    }:
        warnings.warn(
            f"{label}: pre mediánový graf sa nenašiel mediánový čas dávkového CPU spracovania. "
            f"Používa sa fallback '{time_col}'. Spusti novú CPU verziu s mediánovými stĺpcami.",
            RuntimeWarning,
        )

    return build_common_frame(
        df=df,
        traffic_key=traffic_key,
        traffic_label=traffic_label,
        platform=PLATFORM_CPU,
        fib_col="fib_size",
        ip_col="ip_count",
        time_col=time_col,
        throughput_col=None,
    )


def load_gpu_dir24(path: Path, traffic_key: str, traffic_label: str) -> pd.DataFrame:
    label = f"GPU DIR24/8 CUDA - {traffic_label}"
    df = load_csv(path, label)
    require_columns(df, ["fib_size", "ip_count"], label)

    # Reálny GPU scenár: kernel + transfer.
    transfer_time_col = select_time_column(
        df=df,
        label=label + " ČJP",
        avg_candidates=["transfer_plus_kernel_avg_ms"],
        median_candidates=["transfer_plus_kernel_median_ms"],
    )

    transfer_plus_kernel = build_common_frame(
        df=df,
        traffic_key=traffic_key,
        traffic_label=traffic_label,
        platform=PLATFORM_GPU,
        fib_col="fib_size",
        ip_col="ip_count",
        time_col=transfer_time_col,
    )

    # Hypotetický najrýchlejší GPU scenár: iba čas kernelu bez H2D/D2H prenosov.
    kernel_only_time_col = select_time_column(
        df=df,
        label=label + " ČJ",
        avg_candidates=["kernel_lookup_avg_ms"],
        median_candidates=["kernel_lookup_median_ms"],
    )

    kernel_only = build_common_frame(
        df=df,
        traffic_key=traffic_key,
        traffic_label=traffic_label,
        platform=PLATFORM_GPU_KERNEL_ONLY,
        fib_col="fib_size",
        ip_col="ip_count",
        time_col=kernel_only_time_col,
    )

    return pd.concat([transfer_plus_kernel, kernel_only], ignore_index=True)



def load_hybrid_dir24(path: Path, traffic_key: str, traffic_label: str) -> pd.DataFrame:
    label = f"HYBRID DIR24/8 - {traffic_label}"
    df = load_csv(path, label)
    require_columns(df, ["fib_size", "ip_count"], label)

    without_time_col = select_time_column(
        df=df,
        label=label + " bez rozhodovania",
        avg_candidates=["processing_avg_ms"],
        median_candidates=["processing_median_ms"],
    )

    with_time_col = select_time_column(
        df=df,
        label=label + " s rozhodovaním",
        avg_candidates=["with_decision_avg_ms"],
        median_candidates=["with_decision_median_ms"],
    )

    without_decision = build_common_frame(
        df=df,
        traffic_key=traffic_key,
        traffic_label=traffic_label,
        platform=PLATFORM_HYBRID_WITHOUT_DECISION,
        fib_col="fib_size",
        ip_col="ip_count",
        time_col=without_time_col,
    )

    with_decision = build_common_frame(
        df=df,
        traffic_key=traffic_key,
        traffic_label=traffic_label,
        platform=PLATFORM_HYBRID_WITH_DECISION,
        fib_col="fib_size",
        ip_col="ip_count",
        time_col=with_time_col,
    )

    return pd.concat([without_decision, with_decision], ignore_index=True)


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    group_cols = [
        "traffic_key",
        "traffic_label",
        "traffic_order",
        "platform",
        "platform_order",
        "fib_size",
        "ip_count",
    ]

    # Ak CSV obsahuje viac riadkov pre rovnakú konfiguráciu, agregujeme primárne čas:
    # - pri METRIC_MODE="median" medián z vybraných časov,
    # - pri METRIC_MODE="avg" priemer z vybraných časov.
    # Priepustnosť sa potom vždy dopočíta z agregovaného času.
    # Tým sa nemieša median(time) s average(throughput) ani average(time) s median(throughput).
    agg_func = "median" if METRIC_MODE == "median" else "mean"

    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(lookup_time_ms=("lookup_time_ms", agg_func))
    )
    agg["throughput_mpps"] = throughput_from_time_mpps(agg["ip_count"], agg["lookup_time_ms"])

    agg["config_short"] = [
        make_config_short(fib, ip)
        for fib, ip in zip(agg["fib_size"], agg["ip_count"])
    ]

    agg = agg.sort_values(
        ["traffic_order", "fib_size", "ip_count", "platform_order"]
    ).reset_index(drop=True)

    return agg


def load_all_results() -> pd.DataFrame:
    parts = []

    for traffic_key, cfg in TRAFFIC_CONFIGS.items():
        label = cfg["label"]
        print(f"\nNačítavam: {label}")

        cpu = load_cpu_dir24(cfg["cpu_file"], traffic_key, label)
        gpu = load_gpu_dir24(cfg["gpu_file"], traffic_key, label)
        hybrid = load_hybrid_dir24(cfg["hybrid_file"], traffic_key, label)

        print(f"  CPU riadky:    {len(cpu)}")
        print(f"  GPU riadky:    {len(gpu)}")
        print(f"  HYBRID riadky: {len(hybrid)}")

        parts.extend([cpu, gpu, hybrid])

    raw = pd.concat(parts, ignore_index=True)
    if raw.empty:
        raise ValueError("Po načítaní nevznikli žiadne dáta. Skontroluj vstupné CSV.")

    print("\nKontrola pred agregáciou:")
    print(raw[["traffic_key", "traffic_label", "platform", "fib_size", "ip_count", "lookup_time_ms", "throughput_mpps", "source_time_column", "source_throughput_column"]].head(30).to_string(index=False))

    print("\nPočet raw riadkov podľa prevádzky a platformy:")
    print(
        raw.groupby(["traffic_label", "platform"], dropna=False)
        .size()
        .reset_index(name="pocet_riadkov")
        .to_string(index=False)
    )

    all_data = aggregate_by_config(raw)
    if all_data.empty:
        raise ValueError("Po agregácii nevznikli žiadne dáta. Skontroluj vstupné CSV.")

    return all_data


# ============================================================
# TABUĽKY
# ============================================================

def save_tables(all_data: pd.DataFrame) -> None:
    long_path = OUTPUT_DIR / "dir24_8_cpu_gpu_cuda_hybrid_bez_s_rozhodovanim_all_traffic_long.csv"
    wide_path = OUTPUT_DIR / "dir24_8_cpu_gpu_cuda_hybrid_bez_s_rozhodovanim_all_traffic_wide.csv"

    all_data.to_csv(long_path, index=False, encoding="utf-8-sig")

    wide = all_data.pivot_table(
        index=["traffic_key", "traffic_label", "traffic_order", "fib_size", "ip_count", "config_short"],
        columns="platform",
        values=["lookup_time_ms", "throughput_mpps"],
        aggfunc="mean",
    )

    wide.columns = [
        f"{metric}_{platform}"
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("ľ", "l")
        .replace("á", "a")
        .replace("í", "i")
        .replace("ý", "y")
        .replace("é", "e")
        .replace("ť", "t")
        .replace("š", "s")
        .replace("č", "c")
        .replace("ž", "z")
        .replace("ô", "o")
        .replace("_+_", "_plus_")
        .replace("__", "_")
        for metric, platform in wide.columns
    ]

    wide = wide.reset_index().sort_values(["traffic_order", "fib_size", "ip_count"])
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    print(f"\nUložená tabuľka: {long_path}")
    print(f"Uložená tabuľka: {wide_path}")


# ============================================================
# VIZUALIZÁCIA
# ============================================================

def metric_mode_label() -> str:
    return "medián" if METRIC_MODE == "median" else "priemer"


def apply_common_style(ylabel: str, title: str) -> None:
    plt.xlabel("Konfigurácia datasetu", fontsize=PLOT_LABEL_FONTSIZE, labelpad=10)
    plt.ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE, labelpad=10)
    plt.title(
        f"{title} - použitá metrika: {metric_mode_label()}",
        fontsize=PLOT_TITLE_FONTSIZE,
        pad=14,
        wrap=True,
    )
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


def save_current_plot(filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Uložený graf: {out_path}")


def get_axis_for_traffic(all_data: pd.DataFrame, traffic_key: str, platforms: Optional[list[str]] = None) -> pd.DataFrame:
    if platforms is None:
        platforms = PLOT_PLATFORMS

    traffic_data = all_data[
        (all_data["traffic_key"] == traffic_key)
        & (all_data["platform"].isin(platforms))
    ].copy()

    if PLOT_ONLY_COMMON_CONFIGS:
        counts = (
            traffic_data.groupby(["fib_size", "ip_count"], as_index=False)["platform"]
            .nunique()
            .rename(columns={"platform": "pocet_implementacii"})
        )
        common = counts[counts["pocet_implementacii"] == len(platforms)][["fib_size", "ip_count"]]
        traffic_data = traffic_data.merge(common, on=["fib_size", "ip_count"], how="inner")

    axis = (
        traffic_data[["fib_size", "ip_count", "config_short"]]
        .drop_duplicates()
        .sort_values(["fib_size", "ip_count"])
        .reset_index(drop=True)
    )
    if axis.empty:
        raise ValueError(f"Pre {traffic_key} nie sú dostupné spoločné konfigurácie pre zvolenú sadu implementácií.")
    return axis

def get_values(axis: pd.DataFrame, data: pd.DataFrame, platform: str, metric_col: str) -> pd.Series:
    platform_df = data[data["platform"] == platform][["fib_size", "ip_count", metric_col]].copy()
    merged = axis.merge(platform_df, on=["fib_size", "ip_count"], how="left")
    return merged[metric_col]


def plot_metric_line_for_traffic(
    all_data: pd.DataFrame,
    traffic_key: str,
    metric_col: str,
    ylabel: str,
    metric_slug: str,
    title_metric: str,
    platforms: Optional[list[str]] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
) -> None:
    if platforms is None:
        platforms = PLOT_PLATFORMS

    cfg = TRAFFIC_CONFIGS[traffic_key]
    traffic_label = cfg["label"]
    filename_slug = cfg["filename_slug"]

    sub = all_data[
        (all_data["traffic_key"] == traffic_key)
        & (all_data["platform"].isin(platforms))
    ].copy()
    axis = get_axis_for_traffic(all_data, traffic_key, platforms)

    x = np.arange(len(axis))
    labels = axis["config_short"].tolist()

    plt.figure(figsize=(14, 7))
    for platform in platforms:
        values = get_values(axis, sub, platform, metric_col)
        plt.plot(x, values, marker="o", linewidth=2, label=platform, color=PLATFORM_COLORS[platform])

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)
    apply_common_style(
        ylabel,
        f"DIR24/8 - CPU vs GPU CUDA vs HYBRID bez/s rozhodovaním - {traffic_label} - {title_metric}{title_suffix}",
    )
    save_current_plot(f"dir24_8_cpu_gpu_cuda_hybrid_{filename_slug}_{metric_slug}_ciarovy_graf{filename_suffix}.png")


def plot_metric_bar_for_traffic(
    all_data: pd.DataFrame,
    traffic_key: str,
    metric_col: str,
    ylabel: str,
    metric_slug: str,
    title_metric: str,
    platforms: Optional[list[str]] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
) -> None:
    if platforms is None:
        platforms = PLOT_PLATFORMS

    cfg = TRAFFIC_CONFIGS[traffic_key]
    traffic_label = cfg["label"]
    filename_slug = cfg["filename_slug"]

    sub = all_data[
        (all_data["traffic_key"] == traffic_key)
        & (all_data["platform"].isin(platforms))
    ].copy()
    axis = get_axis_for_traffic(all_data, traffic_key, platforms)

    x = np.arange(len(axis))
    labels = axis["config_short"].tolist()
    width = min(0.18, 0.80 / max(len(platforms), 1))
    offsets = np.linspace(-((len(platforms) - 1) / 2) * width, ((len(platforms) - 1) / 2) * width, len(platforms))

    plt.figure(figsize=(15, 8))
    for platform, offset in zip(platforms, offsets):
        values = get_values(axis, sub, platform, metric_col)
        plt.bar(x + offset, values, width=width, label=platform, color=PLATFORM_COLORS[platform])

    plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)
    apply_common_style(
        ylabel,
        f"DIR24/8 - CPU vs GPU CUDA vs HYBRID bez/s rozhodovaním - {traffic_label} - {title_metric}{title_suffix}",
    )
    save_current_plot(f"dir24_8_cpu_gpu_cuda_hybrid_{filename_slug}_{metric_slug}_stlpcovy_graf{filename_suffix}.png")


def build_global_axis(all_data: pd.DataFrame, platforms: Optional[list[str]] = None) -> pd.DataFrame:
    if platforms is None:
        platforms = PLOT_PLATFORMS

    parts = []
    for traffic_key in TRAFFIC_CONFIGS.keys():
        traffic_data = all_data[
            (all_data["traffic_key"] == traffic_key)
            & (all_data["platform"].isin(platforms))
        ].copy()

        if PLOT_ONLY_COMMON_CONFIGS:
            counts = (
                traffic_data.groupby(["fib_size", "ip_count"], as_index=False)["platform"]
                .nunique()
                .rename(columns={"platform": "pocet_implementacii"})
            )
            common = counts[counts["pocet_implementacii"] == len(platforms)][["fib_size", "ip_count"]]
            traffic_data = traffic_data.merge(common, on=["fib_size", "ip_count"], how="inner")

        parts.append(traffic_data)

    filtered = pd.concat(parts, ignore_index=True) if parts else all_data.iloc[0:0].copy()

    axis = (
        filtered[["traffic_order", "traffic_key", "traffic_label", "fib_size", "ip_count", "config_short"]]
        .drop_duplicates()
        .sort_values(["traffic_order", "fib_size", "ip_count"])
        .reset_index(drop=True)
    )
    axis["global_label"] = axis["traffic_label"] + "\n" + axis["config_short"]
    return axis


def plot_metric_line_all_traffic(
    all_data: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    metric_slug: str,
    title_metric: str,
    platforms: Optional[list[str]] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
) -> None:
    if platforms is None:
        platforms = PLOT_PLATFORMS

    axis = build_global_axis(all_data, platforms)
    x = np.arange(len(axis))

    plt.figure(figsize=(21, 8))
    for platform in platforms:
        platform_df = all_data[
            all_data["platform"] == platform
        ][["traffic_key", "fib_size", "ip_count", metric_col]].copy()
        merged = axis.merge(platform_df, on=["traffic_key", "fib_size", "ip_count"], how="left")
        plt.plot(x, merged[metric_col], marker="o", linewidth=2, label=platform, color=PLATFORM_COLORS[platform])

    plt.xticks(x, axis["global_label"], rotation=45, ha="right", fontsize=PLOT_TICK_FONTSIZE)
    apply_common_style(
        ylabel,
        f"DIR24/8 - CPU vs GPU CUDA vs HYBRID bez/s rozhodovaním - všetky typy prevádzky - {title_metric}{title_suffix}",
    )
    save_current_plot(f"dir24_8_cpu_gpu_cuda_hybrid_vsetky_typy_prevadzky_{metric_slug}_ciarovy_graf{filename_suffix}.png")


def plot_metric_bar_all_traffic(
    all_data: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    metric_slug: str,
    title_metric: str,
    platforms: Optional[list[str]] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
) -> None:
    if platforms is None:
        platforms = PLOT_PLATFORMS

    axis = build_global_axis(all_data, platforms)
    x = np.arange(len(axis))
    width = min(0.18, 0.80 / max(len(platforms), 1))
    offsets = np.linspace(-((len(platforms) - 1) / 2) * width, ((len(platforms) - 1) / 2) * width, len(platforms))

    plt.figure(figsize=(21, 8))
    for platform, offset in zip(platforms, offsets):
        platform_df = all_data[
            all_data["platform"] == platform
        ][["traffic_key", "fib_size", "ip_count", metric_col]].copy()
        merged = axis.merge(platform_df, on=["traffic_key", "fib_size", "ip_count"], how="left")
        plt.bar(x + offset, merged[metric_col], width=width, label=platform, color=PLATFORM_COLORS[platform])

    plt.xticks(x, axis["global_label"], rotation=45, ha="right", fontsize=PLOT_TICK_FONTSIZE)
    apply_common_style(
        ylabel,
        f"DIR24/8 - CPU vs GPU CUDA vs HYBRID bez/s rozhodovaním - všetky typy prevádzky - {title_metric}{title_suffix}",
    )
    save_current_plot(f"dir24_8_cpu_gpu_cuda_hybrid_vsetky_typy_prevadzky_{metric_slug}_stlpcovy_graf{filename_suffix}.png")


def build_overlay_axis_by_config(all_data: pd.DataFrame, platforms: list[str]) -> pd.DataFrame:
    """
    Os pre prekryté grafy: konfigurácie sú na osi X iba raz.
    Jednotlivé typy prevádzky sa potom kreslia ako samostatné krivky v tom istom grafe.
    """
    data = all_data[all_data["platform"].isin(platforms)].copy()

    if PLOT_ONLY_COMMON_CONFIGS:
        valid_parts = []
        for traffic_key in TRAFFIC_CONFIGS.keys():
            traffic_data = data[data["traffic_key"] == traffic_key].copy()
            counts = (
                traffic_data.groupby(["fib_size", "ip_count"], as_index=False)["platform"]
                .nunique()
                .rename(columns={"platform": "pocet_implementacii"})
            )
            common = counts[counts["pocet_implementacii"] == len(platforms)][["fib_size", "ip_count"]]
            valid_parts.append(traffic_data.merge(common, on=["fib_size", "ip_count"], how="inner"))
        data = pd.concat(valid_parts, ignore_index=True) if valid_parts else data.iloc[0:0].copy()

    axis = (
        data[["fib_size", "ip_count", "config_short"]]
        .drop_duplicates()
        .sort_values(["fib_size", "ip_count"])
        .reset_index(drop=True)
    )
    if axis.empty:
        raise ValueError("Pre prekryté porovnanie prevádzok nie sú dostupné spoločné konfigurácie.")
    return axis


def plot_metric_line_traffic_overlay(
    all_data: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    metric_slug: str,
    title_metric: str,
    platforms: Optional[list[str]] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
) -> None:
    """
    Pre každú implementáciu vytvorí samostatný čiarový graf.
    V jednom grafe sú prekryté všetky tri typy prevádzky nad rovnakou osou konfigurácií.
    """
    if platforms is None:
        platforms = PLOT_PLATFORMS

    axis = build_overlay_axis_by_config(all_data, platforms)
    x = np.arange(len(axis))
    labels = axis["config_short"].tolist()

    traffic_linestyles = {
        "synthetic": "-",
        "enterprise": "--",
        "hotspot": ":",
    }

    for platform in platforms:
        plt.figure(figsize=(14, 7))

        for traffic_key, cfg in TRAFFIC_CONFIGS.items():
            platform_traffic = all_data[
                (all_data["platform"] == platform)
                & (all_data["traffic_key"] == traffic_key)
            ][["fib_size", "ip_count", metric_col]].copy()

            merged = axis.merge(platform_traffic, on=["fib_size", "ip_count"], how="left")

            plt.plot(
                x,
                merged[metric_col],
                marker="o",
                linewidth=2,
                linestyle=traffic_linestyles.get(traffic_key, "-"),
                label=cfg["label"],
            )

        plt.xticks(x, labels, rotation=30, ha="right", fontsize=PLOT_TICK_FONTSIZE)
        apply_common_style(
            ylabel,
            f"DIR24/8 - {platform} - prekryté porovnanie typov prevádzky - {title_metric}{title_suffix}",
        )

        platform_slug = PLATFORM_FILENAME_SLUGS.get(platform, "platform")

        save_current_plot(
            f"dir24_8_overlay_{platform_slug}_{metric_slug}{filename_suffix}.png"
        )


def plot_all_graphs(all_data: pd.DataFrame) -> None:
    # ------------------------------------------------------------
    # 1) Pôvodné grafy - bez GPU ČJ.
    #    Názvy výstupných súborov ostávajú zachované.
    # ------------------------------------------------------------
    for traffic_key in TRAFFIC_CONFIGS.keys():
        plot_metric_line_for_traffic(
            all_data, traffic_key, "throughput_mpps",
            "Priepustnosť [Mpps]", "priepustnost", "porovnanie priepustnosti",
            platforms=PLOT_PLATFORMS,
        )
        plot_metric_bar_for_traffic(
            all_data, traffic_key, "throughput_mpps",
            "Priepustnosť [Mpps]", "priepustnost", "stĺpcové porovnanie priepustnosti",
            platforms=PLOT_PLATFORMS,
        )
        plot_metric_line_for_traffic(
            all_data, traffic_key, "lookup_time_ms",
            "Čas spracovania dávky [ms]", "cas_lookupu", "porovnanie času lookupu",
            platforms=PLOT_PLATFORMS,
        )
        plot_metric_bar_for_traffic(
            all_data, traffic_key, "lookup_time_ms",
            "Čas spracovania dávky [ms]", "cas_lookupu", "stĺpcové porovnanie času lookupu",
            platforms=PLOT_PLATFORMS,
        )

    plot_metric_line_all_traffic(
        all_data, "throughput_mpps", "Priepustnosť [Mpps]",
        "priepustnost", "porovnanie priepustnosti",
        platforms=PLOT_PLATFORMS,
    )
    plot_metric_bar_all_traffic(
        all_data, "throughput_mpps", "Priepustnosť [Mpps]",
        "priepustnost", "stĺpcové porovnanie priepustnosti",
        platforms=PLOT_PLATFORMS,
    )
    plot_metric_line_all_traffic(
        all_data, "lookup_time_ms", "Čas spracovania dávky [ms]",
        "cas_lookupu", "porovnanie času lookupu",
        platforms=PLOT_PLATFORMS,
    )
    plot_metric_bar_all_traffic(
        all_data, "lookup_time_ms", "Čas spracovania dávky [ms]",
        "cas_lookupu", "stĺpcové porovnanie času lookupu",
        platforms=PLOT_PLATFORMS,
    )

    # ------------------------------------------------------------
    # 2) Duplicitné grafy s pridanou GPU ČJ sériou.
    #    Slúžia na hypotetické porovnanie GPU bez obmedzení transferom.
    # ------------------------------------------------------------
    for traffic_key in TRAFFIC_CONFIGS.keys():
        plot_metric_line_for_traffic(
            all_data, traffic_key, "throughput_mpps",
            "Priepustnosť [Mpps]", "priepustnost", "porovnanie priepustnosti",
            platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
            filename_suffix="_s_kernel_only",
            title_suffix=" - vrátane GPU ČJ",
        )
        plot_metric_bar_for_traffic(
            all_data, traffic_key, "throughput_mpps",
            "Priepustnosť [Mpps]", "priepustnost", "stĺpcové porovnanie priepustnosti",
            platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
            filename_suffix="_s_kernel_only",
            title_suffix=" - vrátane GPU ČJ",
        )
        plot_metric_line_for_traffic(
            all_data, traffic_key, "lookup_time_ms",
            "Čas spracovania dávky [ms]", "cas_lookupu", "porovnanie času lookupu",
            platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
            filename_suffix="_s_kernel_only",
            title_suffix=" - vrátane GPU ČJ",
        )
        plot_metric_bar_for_traffic(
            all_data, traffic_key, "lookup_time_ms",
            "Čas spracovania dávky [ms]", "cas_lookupu", "stĺpcové porovnanie času lookupu",
            platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
            filename_suffix="_s_kernel_only",
            title_suffix=" - vrátane GPU ČJ",
        )

    plot_metric_line_all_traffic(
        all_data, "throughput_mpps", "Priepustnosť [Mpps]",
        "priepustnost", "porovnanie priepustnosti",
        platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
        filename_suffix="_s_kernel_only",
        title_suffix=" - vrátane GPU ČJ",
    )
    plot_metric_bar_all_traffic(
        all_data, "throughput_mpps", "Priepustnosť [Mpps]",
        "priepustnost", "stĺpcové porovnanie priepustnosti",
        platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
        filename_suffix="_s_kernel_only",
        title_suffix=" - vrátane GPU ČJ",
    )
    plot_metric_line_all_traffic(
        all_data, "lookup_time_ms", "Čas spracovania dávky [ms]",
        "cas_lookupu", "porovnanie času lookupu",
        platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
        filename_suffix="_s_kernel_only",
        title_suffix=" - vrátane GPU ČJ",
    )
    plot_metric_bar_all_traffic(
        all_data, "lookup_time_ms", "Čas spracovania dávky [ms]",
        "cas_lookupu", "stĺpcové porovnanie času lookupu",
        platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
        filename_suffix="_s_kernel_only",
        title_suffix=" - vrátane GPU ČJ",
    )

    # ------------------------------------------------------------
    # 3) Nové prekryté čiarové grafy.
    #    Prevádzky nie sú za sebou na osi X; všetky tri sú v jednom grafe
    #    nad rovnakými konfiguráciami. Vygeneruje sa samostatný graf pre
    #    každú implementáciu.
    # ------------------------------------------------------------
    plot_metric_line_traffic_overlay(
        all_data, "throughput_mpps", "Priepustnosť [Mpps]",
        "priepustnost", "porovnanie priepustnosti",
        platforms=PLOT_PLATFORMS,
    )
    plot_metric_line_traffic_overlay(
        all_data, "lookup_time_ms", "Čas spracovania dávky [ms]",
        "cas_lookupu", "porovnanie času lookupu",
        platforms=PLOT_PLATFORMS,
    )
    plot_metric_line_traffic_overlay(
        all_data, "throughput_mpps", "Priepustnosť [Mpps]",
        "priepustnost", "porovnanie priepustnosti",
        platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
        filename_suffix="_s_kernel_only",
        title_suffix=" - vrátane GPU ČJ",
    )
    plot_metric_line_traffic_overlay(
        all_data, "lookup_time_ms", "Čas spracovania dávky [ms]",
        "cas_lookupu", "porovnanie času lookupu",
        platforms=PLOT_PLATFORMS_WITH_KERNEL_ONLY,
        filename_suffix="_s_kernel_only",
        title_suffix=" - vrátane GPU ČJ",
    )


# ============================================================
# KONTROLA
# ============================================================

def print_input_files() -> None:
    print("\nPoužité vstupné súbory:")
    for _, cfg in TRAFFIC_CONFIGS.items():
        print(f"\n{cfg['label']}:")
        print(f"CPU:    {resolve_existing_path(cfg['cpu_file'])}")
        print(f"GPU:    {resolve_existing_path(cfg['gpu_file'])}")
        print(f"HYBRID: {resolve_existing_path(cfg['hybrid_file'])}")


def print_loaded_summary(all_data: pd.DataFrame) -> None:
    print("\nKontrola načítaných dát:")
    print(f"Počet riadkov po agregácii: {len(all_data)}")

    counts = (
        all_data.groupby(["traffic_label", "platform"], as_index=False)
        .size()
        .rename(columns={"size": "pocet_konfiguracii"})
        .sort_values(["traffic_label", "platform"])
    )
    print("\nPočet konfigurácií podľa prevádzky a platformy:")
    print(counts.to_string(index=False))

    print("\nAgregované dáta:")
    print(all_data.to_string(index=False))


def validate_required_configs(all_data: pd.DataFrame) -> None:
    cpu_synthetic = all_data[
        (all_data["traffic_key"] == "synthetic")
        & (all_data["platform"] == PLATFORM_CPU)
    ].copy()

    present = set(zip(cpu_synthetic["fib_size"].astype(int), cpu_synthetic["ip_count"].astype(int)))
    missing = sorted(EXPECTED_CPU_SYNTHETIC_CONFIGS - present)

    if missing:
        missing_labels = ", ".join(make_config_short(fib, ip) for fib, ip in missing)
        raise ValueError(
            "CPU syntetická prevádzka nemá očakávané konfigurácie pre súbor "
            "cpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv.\n"
            f"Chýba: {missing_labels}\n"
            "Skontroluj, že sa načítava nový CSV z priečinka:\n"
            f"{CPU_DIR24_DIR}"
        )

    extra = sorted(present - EXPECTED_CPU_SYNTHETIC_CONFIGS)
    if extra:
        extra_labels = ", ".join(make_config_short(fib, ip) for fib, ip in extra)
        print(
            "\nUPOZORNENIE: CPU syntetický CSV obsahuje aj neočakávané konfigurácie: "
            f"{extra_labels}"
        )

    print("\nKontrola CPU syntetického CSV: OK")
    print("Načítaný súbor musí byť: cpu_dir24_8_lpm_synthetic_ids1_2_3_4_5_6_7_threads12_summary.csv")
    print("Očakávané konfigurácie CPU synthetic: 200K, 400K, 800K, 1M, 1,5M, 2M, 3M IP")


def main() -> None:
    print("Porovnanie algoritmu DIR24/8: CPU vs GPU CUDA vs HYBRID")
    print("Typy prevádzky: Syntetická, Enterprise, Hotspot")
    print("GPU výsledky: používa sa iba ČJP")
    print("HYBRID výsledky: zobrazujú sa samostatne bez rozhodovania aj s rozhodovaním")
    print(f"Režim metrík pre grafy: {METRIC_MODE} ({metric_mode_label()})")
    print("CPU/GPU/HYBRID priepustnosť sa počíta jednotne z vybraného času podľa METRIC_MODE; pri CPU sa používa čas dávkového CPU spracovania.")
    print(f"Pracovný priečinok: {WORK_DIR}")
    print(f"Výstupný priečinok: {OUTPUT_DIR}")
    print(f"Grafy používajú iba spoločné konfigurácie pre všetky implementácie: {PLOT_ONLY_COMMON_CONFIGS}")

    print_input_files()
    all_data = load_all_results()
    validate_required_configs(all_data)
    print_loaded_summary(all_data)
    save_tables(all_data)

    print("\nVytváram grafy...")
    plot_all_graphs(all_data)

    print("\nHotovo.")
    print(f"Výstupy sú uložené v priečinku:\n{OUTPUT_DIR}")

    print("\nVygenerované súbory:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
