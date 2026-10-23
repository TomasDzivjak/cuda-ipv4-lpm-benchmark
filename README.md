# CPU, CUDA and Hybrid IPv4 Lookup Benchmarks

This repository accompanies a bachelor's thesis on IPv4 longest-prefix-match (LPM) lookup. It contains CPU and CUDA implementations of DIR-24-8 and multibit tries, a hybrid CPU/GPU DIR-24-8 implementation, the original analysis scripts, and measured benchmark results.

Large input datasets are generated locally and are not tracked. The supplied benchmark CSVs let you review the recorded measurements and regenerate the plots without rerunning the benchmarks.

## Repository layout

```text
analysis/plots/       Python scripts for processing CSV results and creating plots
data/                 Generated input data (ignored by Git)
docs/                 Thesis PDF and supporting documentation
docs/figures/         Optional explanatory diagrams for documentation
.gitignore            Excludes local build products and generated input datasets
results/benchmarks/   Recorded CSV results, grouped by implementation
results/figures/      Generated plots and derived tables
results/nsight-compute/ Optional Nsight Compute exports
src/common/           Shared project path helpers
src/cpu/              CPU implementations
src/cuda/             CUDA implementations
src/hybrid/           Hybrid CPU/GPU implementation
src/data-generation/  Input dataset generators
build/                Local compiler output; ignored by Git
```

## Selected results

These plots are generated from the included CSV files by the scripts in `analysis/plots/`.

![Enterprise traffic: median throughput comparison](results/figures/graphs_cpu_vs_gpu_enterprise_median/cpu_gpu_enterprise_realistic_priepustnost_median_line.png)

![Hotspot traffic: median throughput comparison](results/figures/graphs_cpu_vs_gpu_hotspot_median/cpu_gpu_hotspot_realistic_priepustnost_median_line.png)

![Synthetic traffic: throughput comparison](results/figures/graphs_cpu_vs_gpu_synthetic_median_fixed/cpu_gpu_all_synthetic_priepustnost_bez_kernel_only_ciarovy_graf.png)

![DIR-24-8 CPU, CUDA and hybrid comparison](results/figures/graphs_dir24_8_cpu_gpu_cuda_hybrid_all_traffic_v8_median/dir24_8_cpu_gpu_cuda_hybrid_vsetky_typy_prevadzky_priepustnost_ciarovy_graf.png)

The full set of generated figures is under `results/figures/`. The project workflow diagram from the thesis is included below. Results depend on the hardware and software environment used for each measurement; consult the thesis and CSV metadata when interpreting or comparing them.

![Project workflow and implementation overview](docs/figures/system-overview.png)

## Recreate the plots

Install Python 3 and the libraries listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

From the repository root, set the project root and run any analysis script. For example:

```powershell
$env:LPM_PROJECT_ROOT = (Get-Location).Path
python analysis/plots/process_cpu_gpu_all_lpm_enterprise.py
```

Each script reads the corresponding CSV files from `results/benchmarks/` and writes plots and derived tables to `results/figures/`.

## Generate input data

Generated data is excluded from Git because the files can be large and the generator programs are included.

- `generate_synthetic_FIB.cpp` writes FIB CSV files to `data/fib/` and synthetic IP CSV files to `data/traffic/synthetic/` by default.
- `dataset_generator_enterprise_hotspot.py` reads synthetic IP CSV files from `data/traffic/synthetic/` and writes enterprise and hotspot query BIN files to `data/traffic/enterprise/` and `data/traffic/hotspot/`.

These paths are resolved relative to `LPM_PROJECT_ROOT`. The C++ generator requires a C++ compiler. The benchmark implementations require an appropriate C++ or CUDA toolchain; CUDA execution requires a compatible GPU. The `data/samples/` directory is not used by the current generators. Add only small, redistributable example inputs there if they become useful.

`build/` is for local compiled programs and intermediate build files. It is not required for browsing the code or regenerating plots and should stay out of version control. `docs/figures/` is optional; use it for explanatory diagrams (for example, a workflow or system overview). Benchmark charts belong in `results/figures/`.

## Thesis and attribution

The thesis PDF is included at [`docs/thesis.pdf`](docs/thesis.pdf), so employers can read the complete work. Citation: Dzivjak, Tomáš. *Urýchlenie smerovania s použitím CUDA*. Bachelor's thesis, Slovak University of Technology in Bratislava, Faculty of Electrical Engineering and Information Technology, 2026. The presentation is unavailable and may be added later.

The author retains copyright subject to the signed publication licence and the rights in third-party material included in the thesis. Public availability in the thesis register does not by itself place the work in the public domain. The thesis PDF is provided for reading and citation; this repository does not grant a blanket license to reuse its text, figures, or third-party material. The project source code needs its own explicit license.

## License

No license has been selected for the source code yet. Add a `LICENSE` file after choosing the license. The thesis PDF, code, benchmark results, and any third-party material may need separate rights and attribution notices.



