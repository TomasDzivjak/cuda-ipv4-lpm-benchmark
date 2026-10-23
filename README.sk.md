# Benchmarky vyhľadávania IPv4 na CPU, CUDA a hybridnom CPU/GPU

Tento repozitár patrí k bakalárskej práci o vyhľadávaní najdlhšej zodpovedajúcej IPv4 trasy (LPM). Obsahuje CPU a CUDA implementácie DIR-24-8 a multibit trie, hybridnú implementáciu DIR-24-8 pre CPU/GPU, pôvodné analytické skripty a namerané benchmarkové výsledky.

Veľké vstupné datasety sa generujú lokálne a Git ich nesleduje. Priložené benchmarkové CSV umožňujú prezerať namerané výsledky a znovu vytvoriť grafy bez opakovania benchmarkov.

## Štruktúra repozitára

```text
analysis/plots/       Python skripty na spracovanie CSV a tvorbu grafov
data/                 Generované vstupné dáta (Git ich ignoruje)
docs/                 PDF bakalárskej práce a podporná dokumentácia
docs/figures/         Voliteľné vysvetľujúce diagramy k dokumentácii
.gitignore            Vylúčenie lokálneho buildu a generovaných datasetov
results/benchmarks/   Namerané CSV rozdelené podľa implementácie
results/figures/      Vygenerované grafy a odvodené tabuľky
results/nsight-compute/ Voliteľné exporty z Nsight Compute
src/common/           Pomocné funkcie na tvorbu ciest
src/cpu/              CPU implementácie
src/cuda/             CUDA implementácie
src/hybrid/           Hybridná implementácia CPU/GPU
src/data-generation/  Generátory vstupných datasetov
build/                Lokálne výstupy kompilátora; Git ich ignoruje
```

## Vybrané výsledky

Tieto grafy vytvárajú skripty v `analysis/plots/` z priložených CSV.

![Enterprise prevádzka: porovnanie mediánu priepustnosti](results/figures/graphs_cpu_vs_gpu_enterprise_median/cpu_gpu_enterprise_realistic_priepustnost_median_line.png)

![Hotspot prevádzka: porovnanie mediánu priepustnosti](results/figures/graphs_cpu_vs_gpu_hotspot_median/cpu_gpu_hotspot_realistic_priepustnost_median_line.png)

![Syntetická prevádzka: porovnanie priepustnosti](results/figures/graphs_cpu_vs_gpu_synthetic_median_fixed/cpu_gpu_all_synthetic_priepustnost_bez_kernel_only_ciarovy_graf.png)

![Porovnanie DIR-24-8 na CPU, CUDA a hybridnom CPU/GPU](results/figures/graphs_dir24_8_cpu_gpu_cuda_hybrid_all_traffic_v8_median/dir24_8_cpu_gpu_cuda_hybrid_vsetky_typy_prevadzky_priepustnost_ciarovy_graf.png)

Všetky vytvorené grafy sú v `results/figures/`. Nižšie je aj bloková schéma riešenia z bakalárskej práce. Výsledky závisia od hardvéru a softvérového prostredia pri meraní; pri interpretácii alebo porovnávaní si pozrite bakalársku prácu a metadáta CSV.

![Prehľad pracovného postupu a implementácií](docs/figures/system-overview.png)

## Opätovné vytvorenie grafov

Nainštalujte Python 3 a knižnice uvedené v `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

V koreňovom priečinku repozitára nastavte koreň projektu a spustite analytický skript. Napríklad:

```powershell
$env:LPM_PROJECT_ROOT = (Get-Location).Path
python analysis/plots/process_cpu_gpu_all_lpm_enterprise.py
```

Každý skript načíta príslušné CSV z `results/benchmarks/` a uloží grafy a odvodené tabuľky do `results/figures/`.

## Generovanie vstupných dát

Generované dáta nie sú v Gite, pretože môžu byť veľké a repozitár obsahuje programy na ich vytvorenie.

- `generate_synthetic_FIB.cpp` predvolene ukladá FIB CSV do `data/fib/` a syntetické IP CSV do `data/traffic/synthetic/`.
- `dataset_generator_enterprise_hotspot.py` číta syntetické IP CSV z `data/traffic/synthetic/` a ukladá BIN súbory enterprise a hotspot prevádzky do `data/traffic/enterprise/` a `data/traffic/hotspot/`.

Cesty sa určujú relatívne od `LPM_PROJECT_ROOT`. Generátor v C++ vyžaduje C++ kompilátor. Benchmarkové implementácie vyžadujú vhodný C++ alebo CUDA nástrojový reťazec; CUDA programy vyžadujú kompatibilnú grafickú kartu. Priečinok `data/samples/` súčasné generátory nepoužívajú. Ak neskôr vznikne potreba ukážkových vstupov, patria tam iba malé súbory, ktoré možno ďalej šíriť.

`build/` slúži na lokálne skompilované programy a dočasné build súbory. Na prezeranie kódu ani opätovné vytvorenie grafov nie je potrebný a nemá byť vo verziovacom systéme. `docs/figures/` je voliteľný priečinok na vysvetľujúce diagramy, napríklad schému systému alebo pracovného postupu. Benchmarkové grafy patria do `results/figures/`.

## Bakalárska práca a uvedenie autora

PDF práce je súčasťou repozitára v [`docs/thesis.pdf`](docs/thesis.pdf), aby si ju zamestnávatelia mohli prečítať. Citácia: Dzivjak, Tomáš. *Urýchlenie smerovania s použitím CUDA*. Bakalárska práca, Slovenská technická univerzita v Bratislave, Fakulta elektrotechniky a informatiky, 2026. Prezentácia zatiaľ nie je dostupná a možno ju doplniť neskôr.

Autorské práva k práci zostávajú autorovi podľa podmienok podpísanej licenčnej zmluvy a s prihliadnutím na práva k materiálom tretích strán. Verejná dostupnosť v registri záverečných prác sama osebe neznamená, že je práca voľným dielom. PDF je tu na čítanie a citovanie; repozitár neudeľuje všeobecnú licenciu na opätovné použitie textu, obrázkov ani materiálov tretích strán. Zdrojový kód projektu potrebuje vlastnú výslovnú licenciu.

## Licencia

Licencia na zdrojový kód ešte nie je vybraná. Po výbere pridaj súbor `LICENSE`. PDF práce, kód, namerané výsledky a materiály tretích strán môžu vyžadovať samostatné práva a uvedenie zdrojov.



