# Benchmarky IPv4 LPM na CPU, CUDA a hybridnom CPU/GPU

Tento projekt som vytvoril v rámci bakalárskej práce o zrýchlení vyhľadávania najdlhšej zodpovedajúcej IPv4 trasy (LPM) pomocou CUDA. Na jednom mieste tu mám svoje CPU, CUDA a hybridné CPU/GPU implementácie DIR-24-8 a multibit trie, skripty na spracovanie meraní a výsledky benchmarkov.

Vstupné datasety sa dajú vytvoriť lokálne, preto veľké generované súbory v repozitári nie sú. Pridal som však namerané CSV a grafy, takže výsledky možno prezerať bez opakovania benchmarkov.

## Čo repozitár obsahuje

```text
analysis/plots/         Python skripty na spracovanie CSV a tvorbu grafov
data/                   Generované vstupné dáta (Git ich ignoruje)
docs/                   Bakalárska práca a súvisiaca dokumentácia
docs/figures/           Vysvetľujúce diagramy
results/benchmarks/     Namerané CSV rozdelené podľa implementácie
results/figures/        Grafy a odvodené tabuľky
results/nsight-compute/ Exporty z Nsight Compute
src/common/             Spoločné pomocné funkcie pre prácu s cestami
src/cpu/                CPU implementácie
src/cuda/               CUDA implementácie
src/hybrid/             Hybridná implementácia CPU/GPU
src/data-generation/    Generátory vstupných datasetov
build/                  Lokálne výstupy kompilátora; Git ich ignoruje
```

## Výsledky

Nasledujúce grafy som vytvoril z priložených CSV pomocou skriptov v `analysis/plots/`.

![Enterprise prevádzka: porovnanie mediánu priepustnosti](results/figures/graphs_cpu_vs_gpu_enterprise_median/cpu_gpu_enterprise_realistic_priepustnost_median_line.png)

![Hotspot prevádzka: porovnanie mediánu priepustnosti](results/figures/graphs_cpu_vs_gpu_hotspot_median/cpu_gpu_hotspot_realistic_priepustnost_median_line.png)

![Syntetická prevádzka: porovnanie priepustnosti](results/figures/graphs_cpu_vs_gpu_synthetic_median_fixed/cpu_gpu_all_synthetic_priepustnost_bez_kernel_only_ciarovy_graf.png)

![Porovnanie DIR-24-8 na CPU, CUDA a hybridnom CPU/GPU](results/figures/graphs_dir24_8_cpu_gpu_cuda_hybrid_all_traffic_v8_median/dir24_8_cpu_gpu_cuda_hybrid_vsetky_typy_prevadzky_priepustnost_ciarovy_graf.png)

Všetky grafy sú v `results/figures/`. Schéma nižšie zobrazuje systém a pracovný postup projektu. Výsledky závisia od hardvéru a softvérového prostredia pri meraní; kontext k nim uvádzam v bakalárskej práci a metadátach CSV.

![Prehľad pracovného postupu a implementácií](docs/figures/system-overview.png)

## Opätovné vytvorenie grafov

Nainštaluj Python 3 a knižnice uvedené v `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

V koreňovom priečinku repozitára nastav koreň projektu a spusti analytický skript. Napríklad:

```powershell
$env:LPM_PROJECT_ROOT = (Get-Location).Path
python analysis/plots/process_cpu_gpu_all_lpm_enterprise.py
```

Skripty načítajú príslušné CSV z `results/benchmarks/` a uložia grafy a odvodené tabuľky do `results/figures/`.

## Generovanie vstupných dát

Generované datasety nie sú v Gite, pretože môžu byť veľké. Pridal som však programy na ich vytvorenie:

- `generate_synthetic_FIB.cpp` predvolene ukladá FIB CSV do `data/fib/` a syntetické IP CSV do `data/traffic/synthetic/`.
- `dataset_generator_enterprise_hotspot.py` číta syntetické IP CSV z `data/traffic/synthetic/` a ukladá BIN súbory enterprise a hotspot prevádzky do `data/traffic/enterprise/` a `data/traffic/hotspot/`.

Tieto cesty sa určujú relatívne od `LPM_PROJECT_ROOT`. Generátor v C++ vyžaduje C++ kompilátor. Benchmarkové implementácie potrebujú vhodný C++ alebo CUDA nástrojový reťazec a na spustenie CUDA kompatibilnú grafickú kartu. Súčasné generátory priečinok `data/samples/` nepoužívajú.

Priečinok `build/` je určený na lokálne skompilované programy a dočasné build súbory; na prezeranie zdrojového kódu ani tvorbu grafov ho netreba. Vysvetľujúce diagramy dávam do `docs/figures/` a benchmarkové grafy do `results/figures/`.

## Bakalárska práca

Bakalársku prácu *Urýchlenie smerovania s použitím CUDA* som napísal na Slovenskej technickej univerzite v Bratislave, Fakulte elektrotechniky a informatiky (2026). PDF je v súbore [docs/thesis.pdf](docs/thesis.pdf).

Odporúčaná citácia: Dzivjak, Tomáš. *Urýchlenie smerovania s použitím CUDA*. Bakalárska práca, Slovenská technická univerzita v Bratislave, Fakulta elektrotechniky a informatiky, 2026.

PDF práce je tu na čítanie a citovanie. Autorské práva k nemu a práva k použitým materiálom tretích strán sa riadia príslušnou licenčnou zmluvou a ďalšími právami. Zdrojový kód zatiaľ nemá licenciu na ďalšie použitie.

## Licencia

K zdrojovému kódu som zatiaľ nepridal licenciu. Pokiaľ ju nedoplním, všetky práva zostávajú vyhradené. PDF práce a materiály tretích strán môžu mať samostatné podmienky.
