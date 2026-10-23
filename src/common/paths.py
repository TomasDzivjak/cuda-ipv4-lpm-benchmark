"""Shared project paths for the LPM benchmark scripts.

Set the LPM_PROJECT_ROOT environment variable to the repository root before
running a script. Paths are resolved at runtime, so they are independent of
the machine-specific location of the repository.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the absolute repository root from LPM_PROJECT_ROOT."""
    value = os.environ.get("LPM_PROJECT_ROOT")
    if not value:
        raise RuntimeError(
            "LPM_PROJECT_ROOT is not set. Set it to the repository root before "
            "running this script. In PowerShell: "
            "$env:LPM_PROJECT_ROOT = (Get-Location).Path"
        )

    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"LPM_PROJECT_ROOT is not an existing directory: {root}")
    return root


def data_dir() -> Path:
    return project_root() / "data"


def fib_dir() -> Path:
    return data_dir() / "fib"


def synthetic_ip_dir() -> Path:
    return data_dir() / "traffic" / "synthetic"


def enterprise_query_dir() -> Path:
    return data_dir() / "traffic" / "enterprise"


def hotspot_query_dir() -> Path:
    return data_dir() / "traffic" / "hotspot"


def results_dir() -> Path:
    return project_root() / "results"


def benchmark_results_dir() -> Path:
    return results_dir() / "benchmarks"


def cpu_dir24_8_results_dir() -> Path:
    return benchmark_results_dir() / "cpu-dir24-8"


def cpu_multibit_trie_results_dir() -> Path:
    return benchmark_results_dir() / "cpu-multibit-trie"


def cuda_dir24_8_results_dir() -> Path:
    return benchmark_results_dir() / "cuda-dir24-8"


def cuda_multibit_trie_results_dir() -> Path:
    return benchmark_results_dir() / "cuda-multibit-trie"


def hybrid_dir24_8_results_dir() -> Path:
    return benchmark_results_dir() / "hybrid-dir24-8"


def figure_results_dir() -> Path:
    return results_dir() / "figures"


def nsight_results_dir() -> Path:
    return results_dir() / "nsight-compute"


def ensure_output_dirs() -> None:
    """Create standard output folders if they do not already exist."""
    for directory in (
        cpu_dir24_8_results_dir(),
        cpu_multibit_trie_results_dir(),
        cuda_dir24_8_results_dir(),
        cuda_multibit_trie_results_dir(),
        hybrid_dir24_8_results_dir(),
        figure_results_dir(),
        nsight_results_dir(),
    ):
        directory.mkdir(parents=True, exist_ok=True)
