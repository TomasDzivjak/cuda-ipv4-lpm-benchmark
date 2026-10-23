#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate enterprise and hotspot LPM query datasets as BIN files from synthetic CSV files.

Input CSV format expected:
    index,ip_uint32,ip_dotted
Only ip_uint32 is required.

Output BIN format:
    raw little-endian uint32_t values, one IP after another.
    Compatible with C/C++ loading using uint32_t and little-endian x86.

Default input:
    data/traffic/synthetic under LPM_PROJECT_ROOT

Default outputs:
    data/traffic/enterprise and data/traffic/hotspot under LPM_PROJECT_ROOT
"""

import argparse
import csv
import random
import struct
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

import sys

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from paths import enterprise_query_dir, hotspot_query_dir, synthetic_ip_dir

TARGET_SIZES = [200_000, 400_000, 800_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000]


# -----------------------------
# I/O
# -----------------------------

def load_ip_uint32_csv(path: Path) -> List[int]:
    ips: List[int] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "ip_uint32" not in reader.fieldnames:
            raise ValueError(f"CSV must contain column 'ip_uint32': {path}")

        for row_idx, row in enumerate(reader, start=2):
            raw = row.get("ip_uint32", "").strip()

            if not raw:
                continue

            ip = int(raw)

            if ip < 0 or ip > 0xFFFFFFFF:
                raise ValueError(f"Invalid uint32 IP at {path}:{row_idx}: {ip}")

            ips.append(ip)

    return ips


def save_uint32_bin(path: Path, ips: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        # write in chunks to avoid huge temporary bytes object for multi-million files
        chunk_size = 250_000
        for i in range(0, len(ips), chunk_size):
            chunk = ips[i:i + chunk_size]
            f.write(struct.pack(f"<{len(chunk)}I", *chunk))


def find_csv_for_size(input_dir: Path, size: int) -> Path:
    candidates = sorted(input_dir.glob(f"*{size}*.csv"))

    exact_name = input_dir / f"generated_ips_{size}_seed_12345.csv"
    if exact_name.exists():
        return exact_name

    if not candidates:
        raise FileNotFoundError(f"No CSV found for size {size} in {input_dir}")

    if len(candidates) > 1:
        names = "\n  ".join(str(c.name) for c in candidates)
        raise RuntimeError(
            f"Multiple CSV candidates found for size {size}. "
            f"Rename files or use expected name generated_ips_{size}_seed_12345.csv:\n  {names}"
        )

    return candidates[0]


# -----------------------------
# IP helpers
# -----------------------------

def prefix_key(ip: int, prefix_bits: int) -> int:
    return ip >> (32 - prefix_bits)


def random_ip_inside_prefix(prefix: int, prefix_bits: int, rng: random.Random) -> int:
    host_bits = 32 - prefix_bits

    if host_bits == 0:
        return prefix

    host_mask = (1 << host_bits) - 1
    host = rng.getrandbits(host_bits)

    return ((prefix << host_bits) | host) & 0xFFFFFFFF


def choose_prefixes_from_anchors(
    ips: List[int],
    prefix_bits: int,
    count: int,
    rng: random.Random,
    weighted: bool
) -> List[int]:
    if not ips:
        return []

    counts = Counter(prefix_key(ip, prefix_bits) for ip in ips)
    keys = list(counts.keys())
    count = min(count, len(keys))

    if count <= 0:
        return []

    if not weighted:
        return rng.sample(keys, count)

    selected: List[int] = []
    available = keys[:]

    for _ in range(count):
        weights = [counts[k] for k in available]
        chosen = rng.choices(available, weights=weights, k=1)[0]
        selected.append(chosen)
        available.remove(chosen)

        if not available:
            break

    return selected


# -----------------------------
# Traffic models
# -----------------------------

def build_enterprise(
    synthetic_ips: List[int],
    seed: int,
    business_p24_count: int = 128,
    department_p16_count: int = 16,
    business_ratio: float = 0.50,
    department_ratio: float = 0.35,
    random_ratio: float = 0.15,
    weighted_prefix_selection: bool = True,
    preserve_original_random_part: bool = True,
) -> List[int]:
    """
    Enterprise/campus traffic:
      - business_ratio: repeated traffic to selected /24 prefixes
      - department_ratio: repeated traffic to selected /16 regions
      - random_ratio: global background traffic

    Important:
      For /24 and /16 locality, the generator creates new random addresses inside
      selected prefixes. This avoids the common failure mode where a fully uniform
      synthetic dataset has only 1 address per /24 and hotspot generation degenerates
      into repeating the same few IP addresses.
    """
    if not synthetic_ips:
        return []

    rng = random.Random(seed)
    n = len(synthetic_ips)

    total = business_ratio + department_ratio + random_ratio
    if total <= 0:
        raise ValueError("At least one ratio must be positive")

    business_ratio /= total
    department_ratio /= total
    random_ratio /= total

    business_p24 = choose_prefixes_from_anchors(
        synthetic_ips, 24, business_p24_count, rng, weighted_prefix_selection
    )
    department_p16 = choose_prefixes_from_anchors(
        synthetic_ips, 16, department_p16_count, rng, weighted_prefix_selection
    )

    if not business_p24:
        business_p24 = [prefix_key(rng.choice(synthetic_ips), 24)]
    if not department_p16:
        department_p16 = [prefix_key(rng.choice(synthetic_ips), 16)]

    out: List[int] = []

    for _ in range(n):
        r = rng.random()

        if r < business_ratio:
            p = rng.choice(business_p24)
            out.append(random_ip_inside_prefix(p, 24, rng))

        elif r < business_ratio + department_ratio:
            p = rng.choice(department_p16)
            out.append(random_ip_inside_prefix(p, 16, rng))

        else:
            if preserve_original_random_part:
                out.append(rng.choice(synthetic_ips))
            else:
                p = rng.choice(synthetic_ips)
                out.append(p)

    rng.shuffle(out)
    return out


def build_hotspot(
    synthetic_ips: List[int],
    seed: int,
    hot_p24_count: int = 8,
    hotspot_ratio: float = 0.95,
    weighted_prefix_selection: bool = True,
    preserve_original_random_part: bool = True,
) -> List[int]:
    """
    Hotspot traffic:
      - hotspot_ratio of queries goes into a small set of hot /24 prefixes
      - remaining traffic is random background

    Generates random host addresses inside the selected /24 prefixes instead of
    only resampling existing IPs. This gives prefix locality without reducing the
    dataset to a few repeated identical addresses.
    """
    if not synthetic_ips:
        return []

    rng = random.Random(seed)
    n = len(synthetic_ips)

    if hotspot_ratio < 0 or hotspot_ratio > 1:
        raise ValueError("hotspot_ratio must be in range 0..1")

    hot_p24 = choose_prefixes_from_anchors(
        synthetic_ips, 24, hot_p24_count, rng, weighted_prefix_selection
    )

    if not hot_p24:
        hot_p24 = [prefix_key(rng.choice(synthetic_ips), 24)]

    out: List[int] = []

    for _ in range(n):
        if rng.random() < hotspot_ratio:
            p = rng.choice(hot_p24)
            out.append(random_ip_inside_prefix(p, 24, rng))
        else:
            if preserve_original_random_part:
                out.append(rng.choice(synthetic_ips))
            else:
                out.append(rng.choice(synthetic_ips))

    rng.shuffle(out)
    return out


# -----------------------------
# Analysis
# -----------------------------

def locality_stats(ips: List[int]) -> Tuple[int, float, int, float]:
    if not ips:
        return 0, 0.0, 0, 0.0

    p24_counts = Counter(prefix_key(ip, 24) for ip in ips)
    p16_counts = Counter(prefix_key(ip, 16) for ip in ips)

    top10_p24 = sum(c for _, c in p24_counts.most_common(10)) / len(ips)
    top10_p16 = sum(c for _, c in p16_counts.most_common(10)) / len(ips)

    return len(p24_counts), top10_p24, len(p16_counts), top10_p16


def print_stats(label: str, ips: List[int]) -> None:
    unique_p24, top10_p24, unique_p16, top10_p16 = locality_stats(ips)

    print(f"{label}:")
    print(f"  queries             : {len(ips)}")
    print(f"  unique /24 prefixes : {unique_p24}")
    print(f"  top-10 /24 share    : {top10_p24 * 100:.2f}%")
    print(f"  unique /16 prefixes : {unique_p16}")
    print(f"  top-10 /16 share    : {top10_p16 * 100:.2f}%")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate enterprise and hotspot BIN query datasets from synthetic CSV files."
    )

    parser.add_argument(
        "--input-csv-dir",
        type=Path,
        default=synthetic_ip_dir(),
        help="Directory containing generated_ips_<N>_seed_12345.csv files",
    )
    parser.add_argument(
        "--enterprise-out-dir",
        type=Path,
        default=enterprise_query_dir(),
        help="Output directory for enterprise BIN files",
    )
    parser.add_argument(
        "--hotspot-out-dir",
        type=Path,
        default=hotspot_query_dir(),
        help="Output directory for hotspot BIN files",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=TARGET_SIZES,
        help="Dataset sizes to generate",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--enterprise-business-p24", type=int, default=128)
    parser.add_argument("--enterprise-department-p16", type=int, default=16)
    parser.add_argument("--enterprise-business-ratio", type=float, default=0.50)
    parser.add_argument("--enterprise-department-ratio", type=float, default=0.35)
    parser.add_argument("--enterprise-random-ratio", type=float, default=0.15)

    parser.add_argument("--hotspot-p24", type=int, default=8)
    parser.add_argument("--hotspot-ratio", type=float, default=0.95)

    parser.add_argument(
        "--uniform-prefix-selection",
        action="store_true",
        help="Select anchor prefixes uniformly instead of frequency-weighted",
    )

    args = parser.parse_args()

    args.enterprise_out_dir.mkdir(parents=True, exist_ok=True)
    args.hotspot_out_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("Generating enterprise and hotspot BIN query datasets")
    print("============================================================")
    print(f"Input CSV dir      : {args.input_csv_dir}")
    print(f"Enterprise out dir : {args.enterprise_out_dir}")
    print(f"Hotspot out dir    : {args.hotspot_out_dir}")
    print(f"Sizes              : {args.sizes}")
    print(f"Seed               : {args.seed}")
    print("")

    for idx, size in enumerate(args.sizes):
        csv_path = find_csv_for_size(args.input_csv_dir, size)
        synthetic_ips = load_ip_uint32_csv(csv_path)

        if len(synthetic_ips) != size:
            print(f"WARNING: {csv_path.name} contains {len(synthetic_ips)} IPs, expected {size}")

        enterprise_ips = build_enterprise(
            synthetic_ips=synthetic_ips,
            seed=args.seed + idx * 1000 + 11,
            business_p24_count=args.enterprise_business_p24,
            department_p16_count=args.enterprise_department_p16,
            business_ratio=args.enterprise_business_ratio,
            department_ratio=args.enterprise_department_ratio,
            random_ratio=args.enterprise_random_ratio,
            weighted_prefix_selection=not args.uniform_prefix_selection,
        )

        hotspot_ips = build_hotspot(
            synthetic_ips=synthetic_ips,
            seed=args.seed + idx * 1000 + 22,
            hot_p24_count=args.hotspot_p24,
            hotspot_ratio=args.hotspot_ratio,
            weighted_prefix_selection=not args.uniform_prefix_selection,
        )

        enterprise_name = (
            f"queries_prefix_{len(enterprise_ips)}"
            f"_enterprise_b{int(args.enterprise_business_ratio * 100)}"
            f"_d{int(args.enterprise_department_ratio * 100)}"
            f"_r{int(args.enterprise_random_ratio * 100)}"
            f"_p24{args.enterprise_business_p24}"
            f"_p16{args.enterprise_department_p16}.bin"
        )

        hotspot_name = (
            f"queries_prefix_{len(hotspot_ips)}"
            f"_hotspot_hp{args.hotspot_p24}"
            f"_r{int(args.hotspot_ratio * 100)}.bin"
        )

        enterprise_path = args.enterprise_out_dir / enterprise_name
        hotspot_path = args.hotspot_out_dir / hotspot_name

        save_uint32_bin(enterprise_path, enterprise_ips)
        save_uint32_bin(hotspot_path, hotspot_ips)

        print("------------------------------------------------------------")
        print(f"Source CSV: {csv_path.name}")
        print_stats("Synthetic source locality", synthetic_ips)
        print("")
        print(f"Enterprise BIN: {enterprise_path.name}")
        print_stats("Enterprise locality", enterprise_ips)
        print("")
        print(f"Hotspot BIN: {hotspot_path.name}")
        print_stats("Hotspot locality", hotspot_ips)
        print("")

    print("DONE.")


if __name__ == "__main__":
    main()

