#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/paths.hpp"
#include <omp.h>

// Skript pracuje so súbormi a priečinkami cez std::filesystem.
namespace fs = std::filesystem;

// ------------------------------------------------------------
// Štruktúra Prefix
// ------------------------------------------------------------
// Reprezentuje jeden záznam z FIB tabuľky.
// network  = sieťová adresa prefixu
// length   = dĺžka prefixu (napr. /24)
// mask     = maska prefixu v 32-bitovom tvare
// next_hop = výsledná hodnota, ktorú má lookup vrátiť
struct Prefix {
    uint32_t network = 0;
    uint8_t length = 0;
    uint32_t mask = 0;
    int32_t next_hop = -1;
};

// ------------------------------------------------------------
// Vytvorenie masky prefixu podľa jeho dĺžky
// ------------------------------------------------------------
// Príklady:
// /0  -> 0x00000000
// /24 -> 0xFFFFFF00
// /32 -> 0xFFFFFFFF
static uint32_t prefix_mask(uint8_t length) {
    if (length == 0) return 0u;
    if (length >= 32) return 0xFFFFFFFFu;
    return 0xFFFFFFFFu << (32u - length);
}

// ------------------------------------------------------------
// Rozdelenie jedného CSV riadku na polia
// ------------------------------------------------------------
// Funkcia berie do úvahy aj úvodzovky, aby sa čiarka vo vnútri
// úvodzoviek nechápala ako oddeľovač stĺpcov.
static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    current.reserve(line.size());
    bool in_quotes = false;

    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

// ------------------------------------------------------------
// Orezanie bielych znakov zo začiatku a konca reťazca
// ------------------------------------------------------------
static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// ------------------------------------------------------------
// Načítanie IP adries z CSV súboru
// ------------------------------------------------------------
// Očakávaný formát podľa generator.cpp:
// index,ip_uint32,ip_dotted
// Tu používame iba stĺpec ip_uint32.
static std::vector<uint32_t> load_ips_csv(const fs::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Nepodarilo sa otvorit IP CSV: " + file_path.string());
    }

    std::string line;

    // Preskočíme hlavičku CSV.
    if (!std::getline(file, line)) {
        throw std::runtime_error("Prazdny IP CSV: " + file_path.string());
    }

    std::vector<uint32_t> ips;
    ips.reserve(1024);

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        const auto fields = split_csv_line(line);
        if (fields.size() < 2) {
            throw std::runtime_error("Neplatny riadok v IP CSV: " + line);
        }

        // Druhý stĺpec je 32-bitová IP adresa.
        ips.push_back(static_cast<uint32_t>(std::stoul(trim(fields[1]))));
    }

    return ips;
}

// ------------------------------------------------------------
// Načítanie FIB tabuľky z CSV súboru
// ------------------------------------------------------------
// Očakávaný formát podľa generator.cpp:
// index,network_uint32,network_dotted,prefix_length,mask_uint32,mask_dotted,next_hop
// Z každého riadku si zoberieme network, length, mask a next_hop.
static std::vector<Prefix> load_fib_csv(const fs::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Nepodarilo sa otvorit FIB CSV: " + file_path.string());
    }

    std::string line;

    // Preskočíme hlavičku CSV.
    if (!std::getline(file, line)) {
        throw std::runtime_error("Prazdny FIB CSV: " + file_path.string());
    }

    std::vector<Prefix> fib;
    fib.reserve(1024);

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        const auto fields = split_csv_line(line);
        if (fields.size() < 7) {
            throw std::runtime_error("Neplatny riadok vo FIB CSV: " + line);
        }

        Prefix p;
        p.network = static_cast<uint32_t>(std::stoul(trim(fields[1])));
        p.length = static_cast<uint8_t>(std::stoul(trim(fields[3])));
        p.mask = static_cast<uint32_t>(std::stoul(trim(fields[4])));
        p.next_hop = static_cast<int32_t>(std::stol(trim(fields[6])));

        // Pre istotu si masku prepočítame z dĺžky prefixu, aby bola konzistentná.
        const uint32_t canonical_mask = prefix_mask(p.length);
        if (p.mask != canonical_mask) {
            p.mask = canonical_mask;
        }

        // Sieťová adresa sa ešte raz orezáva maskou.
        p.network &= p.mask;

        fib.push_back(p);
    }

    return fib;
}

// ------------------------------------------------------------
// Pomocná funkcia: vytiahne prvé číslo z názvu súboru
// ------------------------------------------------------------
// Napr. z názvu "fib_table_10000_seed_12345.csv" vytiahne 10000.
// Používa sa na rozumné zoradenie CSV súborov.
static std::size_t extract_first_number(const std::string& s) {
    std::string digits;
    bool started = false;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            digits.push_back(c);
            started = true;
        } else if (started) {
            break;
        }
    }
    if (digits.empty()) return 0;
    return static_cast<std::size_t>(std::stoull(digits));
}

// ------------------------------------------------------------
// Zoznam CSV súborov v priečinku, zoradený podľa čísla v názve
// ------------------------------------------------------------
static std::vector<fs::path> list_csv_files_sorted(const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("Priecinok neexistuje alebo nie je priecinok: " + dir.string());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".csv") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        const auto na = extract_first_number(a.filename().string());
        const auto nb = extract_first_number(b.filename().string());
        if (na != nb) return na < nb;
        return a.filename().string() < b.filename().string();
    });

    return files;
}

// ------------------------------------------------------------
// Trieda Dir24_8
// ------------------------------------------------------------
// Toto je samotná implementácia CPU lookup algoritmu DIR-24-8.
// Základná myšlienka:
// 1. Prvých 24 bitov IP adresy použijeme ako index do primary tabuľky.
// 2. Ak je prefix dlhší než /24, použije sa secondary tabuľka,
//    ktorá rozhoduje podľa posledných 8 bitov.
class Dir24_8 {
public:
    // --------------------------------------------------------
    // SecondaryTable
    // --------------------------------------------------------
    // Jedna sekundárna tabuľka má 256 položiek, lebo posledných
    // 8 bitov IP adresy môže nadobúdať 256 hodnôt (0 až 255).
    struct SecondaryTable {
        std::array<int32_t, 256> next_hop{};
        std::array<int16_t, 256> best_len{};

        SecondaryTable() {
            // -1 znamená, že zatiaľ nič nie je nastavené.
            next_hop.fill(-1);
            best_len.fill(-1);
        }
    };

    // --------------------------------------------------------
    // PrimaryEntry
    // --------------------------------------------------------
    // Jedna položka v primary tabuľke obsahuje:
    // - next_hop a best_len pre prefixy do /24
    // - alebo odkaz na secondary tabuľku, ak ju treba použiť
    struct PrimaryEntry {
        int32_t next_hop = -1;
        int16_t best_len = -1;
        uint32_t secondary_index = 0;
        uint8_t is_secondary = 0;
    };

    // --------------------------------------------------------
    // Build fáza DIR-24-8
    // --------------------------------------------------------
    // Zo vstupnej FIB tabuľky sa vytvorí primary tabuľka a podľa
    // potreby aj sekundárne tabuľky.
    void build(const std::vector<Prefix>& fib) {
        // Primary tabuľka má 2^24 položiek.
        primary_.assign(PRIMARY_SIZE, PrimaryEntry{});
        secondaries_.clear();

        // Prefixy sa stabilne zoradia podľa dĺžky vzostupne.
        // Dlhšie prefixy sa tak zapisujú neskôr a môžu prekryť kratšie.
        std::vector<Prefix> prefixes = fib;
        std::stable_sort(prefixes.begin(), prefixes.end(),
            [](const Prefix& a, const Prefix& b) {
                return a.length < b.length;
            });

        for (const Prefix& p : prefixes) {
            if (p.length <= 24) {
                fill_primary_range(p);
            } else {
                fill_secondary_range(p);
            }
        }
    }

    // --------------------------------------------------------
    // Lookup operácia
    // --------------------------------------------------------
    // 1. Zoberie sa horných 24 bitov IP adresy.
    // 2. Ak primary entry nemá secondary tabuľku, výsledok je hneď k dispozícii.
    // 3. Inak sa použije posledných 8 bitov a dohľadá sa v secondary tabuľke.
    int32_t lookup(uint32_t ip) const {
        const uint32_t idx24 = ip >> 8;
        const PrimaryEntry& e = primary_[idx24];

        if (!e.is_secondary) {
            return e.next_hop;
        }

        const uint8_t low8 = static_cast<uint8_t>(ip & 0xFFu);
        return secondaries_[e.secondary_index].next_hop[low8];
    }

private:
    // Primary tabuľka má veľkosť 2^24.
    static constexpr uint32_t PRIMARY_SIZE = 1u << 24;

    std::vector<PrimaryEntry> primary_;
    std::vector<SecondaryTable> secondaries_;

    // --------------------------------------------------------
    // Vyplnenie primary tabuľky pre prefixy dĺžky <= /24
    // --------------------------------------------------------
    // Prefix /24 ovplyvní presne 1 položku primary tabuľky.
    // Prefix /16 ovplyvní viac položiek, lebo pokrýva väčší rozsah.
    void fill_primary_range(const Prefix& p) {
        const uint32_t start = p.network >> 8;
        const uint32_t count = (p.length == 24)
            ? 1u
            : (1u << (24u - static_cast<uint32_t>(p.length)));

        for (uint32_t i = 0; i < count; ++i) {
            PrimaryEntry& entry = primary_[start + i];

            // Ak položka ešte nemá secondary tabuľku, zapisujeme priamo sem.
            if (!entry.is_secondary) {
                // Zapíše sa len vtedy, ak je prefix dlhší ako ten, ktorý tam už je.
                if (p.length > entry.best_len) {
                    entry.next_hop = p.next_hop;
                    entry.best_len = p.length;
                }
            } else {
                // Ak už secondary tabuľka existuje, kratší prefix musí byť zapísaný
                // aj do primary entry aj do všetkých 256 pozícií secondary tabuľky,
                // ale len vtedy, ak je dlhší než to, čo tam bolo doteraz.
                if (p.length > entry.best_len) {
                    entry.next_hop = p.next_hop;
                    entry.best_len = p.length;
                }

                SecondaryTable& sec = secondaries_[entry.secondary_index];
                for (int j = 0; j < 256; ++j) {
                    if (p.length > sec.best_len[static_cast<std::size_t>(j)]) {
                        sec.next_hop[static_cast<std::size_t>(j)] = p.next_hop;
                        sec.best_len[static_cast<std::size_t>(j)] = p.length;
                    }
                }
            }
        }
    }

    // --------------------------------------------------------
    // Vyplnenie secondary tabuľky pre prefixy dĺžky > /24
    // --------------------------------------------------------
    // Prefixy /25 až /32 sa mapujú do jedinej secondary tabuľky,
    // ktorá zodpovedá konkrétnej 24-bitovej hodnote v primary tabuľke.
    void fill_secondary_range(const Prefix& p) {
        const uint32_t idx24 = p.network >> 8;
        PrimaryEntry& entry = primary_[idx24];

        // Ak pre túto 24-bitovú hodnotu secondary tabuľka ešte neexistuje,
        // vytvoríme ju a inicializujeme hodnotami z primary entry.
        if (!entry.is_secondary) {
            SecondaryTable sec;
            sec.next_hop.fill(entry.next_hop);
            sec.best_len.fill(entry.best_len);
            secondaries_.push_back(sec);
            entry.secondary_index = static_cast<uint32_t>(secondaries_.size() - 1);
            entry.is_secondary = 1;
        }

        SecondaryTable& sec = secondaries_[entry.secondary_index];
        const uint8_t low = static_cast<uint8_t>(p.network & 0xFFu);
        const uint32_t range = 1u << (32u - static_cast<uint32_t>(p.length));

        // Rozsah sa vyplní podľa zvyšku dĺžky prefixu.
        // Napríklad /25 pokryje polovicu secondary tabuľky,
        // /32 pokryje iba jednu položku.
        for (uint32_t i = 0; i < range; ++i) {
            const uint8_t pos = static_cast<uint8_t>(low + i);
            if (p.length > sec.best_len[pos]) {
                sec.next_hop[pos] = p.next_hop;
                sec.best_len[pos] = p.length;
            }
        }
    }
};

// ------------------------------------------------------------
// Paralelný lookup cez OpenMP
// ------------------------------------------------------------
// Každé vlákno spracúva inú časť vstupného poľa IP adries.
static void dir_lookup_omp(
    const Dir24_8& dir,
    const std::vector<uint32_t>& ips,
    std::vector<int32_t>& out_next_hop,
    int num_threads
) {
    if (out_next_hop.size() != ips.size()) {
        throw std::runtime_error(
            "Vystupny buffer musi byt alokovany pred meranym lookupom. Ocakavana velkost=" +
            std::to_string(ips.size()) + ", aktualna=" + std::to_string(out_next_hop.size())
        );
    }

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(ips.size()); ++i) {
        out_next_hop[static_cast<std::size_t>(i)] = dir.lookup(ips[static_cast<std::size_t>(i)]);
    }
}

// ------------------------------------------------------------
// Predpripravenie vystupneho buffera mimo meraneho useku
// ------------------------------------------------------------
// Cielom je odstranit nahodny sum z alokacie a mapovania pamate
// z lookup_time_ms. Samotny merany lookup stale zahrna OpenMP paralelnu
// oblast, staticke rozdelenie prace medzi vlakna a zapis next-hop vysledkov
// do vystupneho pola. Nejde teda o izolovane volanie dir.lookup() bez rezie.
static void pre_touch_output_buffer(std::vector<int32_t>& out_next_hop) {
    std::fill(out_next_hop.begin(), out_next_hop.end(), 0);
}

// ------------------------------------------------------------
// Realistickejsi batch forwarding model CPU smerovaca
// ------------------------------------------------------------
// V realnom smerovaci sa pracovné buffre typicky nealokujú pre každý batch,
// ale recyklujú sa. Preto sa alokacia nerata do merania. Do merania vsak
// cielene zapocitame pripravu vstupnej davky: query IP adresy sa skopiruju
// do predalokovaneho ingress buffera, z ktoreho potom lookup faza cita.
// Tato metrika je vhodnejsia na prakticke porovnanie s GPU kernel+transfer,
// pretoze CPU tiez neukazuje iba izolovany scalar lookup.
static void router_forward_batch_cpu_model(
    const Dir24_8& dir,
    const std::vector<uint32_t>& source_ips,
    std::vector<uint32_t>& ingress_buffer,
    std::vector<int32_t>& out_next_hop,
    int num_threads
) {
    if (ingress_buffer.size() != source_ips.size()) {
        throw std::runtime_error(
            "Ingress buffer musi byt alokovany pred meranym batchom. Ocakavana velkost=" +
            std::to_string(source_ips.size()) + ", aktualna=" + std::to_string(ingress_buffer.size())
        );
    }
    if (out_next_hop.size() != source_ips.size()) {
        throw std::runtime_error(
            "Vystupny buffer musi byt alokovany pred meranym batchom. Ocakavana velkost=" +
            std::to_string(source_ips.size()) + ", aktualna=" + std::to_string(out_next_hop.size())
        );
    }

    // Sekvencna kopia reprezentuje pripravu davky v CPU pipeline. Nejde o
    // diskove I/O ani dynamicku alokaciu, ale o recyklovany pracovny buffer.
    std::copy(source_ips.begin(), source_ips.end(), ingress_buffer.begin());

    // Samotna lookup faza nad pripravenym ingress bufferom.
    dir_lookup_omp(dir, ingress_buffer, out_next_hop, num_threads);
}

// ------------------------------------------------------------
// Výpočet checksumu z výsledného poľa next-hopov
// ------------------------------------------------------------
// Používa sa na kontrolu, či lookup vracia pri rôznych behoch
// rovnaké výsledky.
static uint64_t checksum_vector(const std::vector<int32_t>& v) {
    uint64_t sum = 0;
    for (int32_t x : v) {
        sum += static_cast<uint64_t>(static_cast<uint32_t>(x));
    }
    return sum;
}


// ------------------------------------------------------------
// Typ prevádzky / query datasetu
// ------------------------------------------------------------
enum class TrafficType {
    SYNTHETIC,
    ENTERPRISE,
    HOTSPOT
};

static std::string traffic_type_to_string(TrafficType t) {
    switch (t) {
        case TrafficType::SYNTHETIC:  return "synthetic";
        case TrafficType::ENTERPRISE: return "enterprise";
        case TrafficType::HOTSPOT:    return "hotspot";
    }
    return "unknown";
}

// ------------------------------------------------------------
// Načítanie IP adries z BIN súboru
// ------------------------------------------------------------
// ENTERPRISE a HOTSPOT datasety sú binárne súbory obsahujúce iba
// za sebou uložené hodnoty uint32_t, teda IPv4 adresy v 32-bitovom tvare.
static std::vector<uint32_t> load_ips_bin(const fs::path& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Nepodarilo sa otvorit IP BIN: " + file_path.string());
    }

    const std::streamsize bytes = file.tellg();
    if (bytes < 0) {
        throw std::runtime_error("Nepodarilo sa zistit velkost IP BIN: " + file_path.string());
    }
    if (bytes % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) {
        throw std::runtime_error("Neplatna velkost IP BIN suboru, nie je delitelna 4: " + file_path.string());
    }

    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> ips(static_cast<std::size_t>(bytes) / sizeof(uint32_t));
    if (!ips.empty()) {
        file.read(reinterpret_cast<char*>(ips.data()), bytes);
        if (!file) {
            throw std::runtime_error("Nepodarilo sa nacitat cely IP BIN: " + file_path.string());
        }
    }

    return ips;
}

// ------------------------------------------------------------
// Konfigurácia jedného testovaného datasetu
// ------------------------------------------------------------
struct DatasetConfig {
    int id = 0;
    std::string label;
    std::string fib_filename;
    std::string ip_filename;
};

// ------------------------------------------------------------
// Štruktúra jedného výsledného riadku pre CSV
// ------------------------------------------------------------
struct RunMetrics {
    int run_id = 0;
    std::string traffic_type;
    int dataset_id = 0;
    std::string dataset_label;
    int threads = 12;
    std::string fib_file;
    std::string ip_file;
    std::size_t fib_size = 0;
    std::size_t ip_count = 0;

    // Inicializacny / rekonfiguracny naklad. V realnom smerovaci sa FIB
    // nestavia pred kazdou davkou paketov, preto sa neprepocitava do hlavnej
    // lookup priepustnosti. Zostava vsak v CSV ako samostatna metrika.
    double build_time_ms = 0.0;
    double build_median_time_ms = 0.0;

    // Stabilny forwarding stav: FIB je uz postavena, vstupny batch je uz
    // v pamati procesu a vystupny buffer je predalokovany. Meria sa OpenMP
    // lookup + citanie IP + zapis next-hop vysledkov.
    double lookup_time_ms = 0.0;
    double lookup_median_time_ms = 0.0;
    double lookup_throughput_mpps = 0.0;
    double lookup_median_throughput_mpps = 0.0;

    // Realistickejsi batch model CPU smerovaca: predalokovane buffre sa
    // recykluju, ale do casu je zahrnuta priprava/kopia vstupnej davky do
    // pracovneho ingress buffera + samotny lookup + zapis vystupu.
    double router_batch_time_ms = 0.0;
    double router_batch_median_time_ms = 0.0;
    double router_batch_throughput_mpps = 0.0;
    double router_batch_median_throughput_mpps = 0.0;

    // Pre kompatibilitu s existujucimi grafmi je hlavna throughput_mpps
    // nastavena na router_batch_throughput_mpps, teda na praktickejsiu CPU
    // batch metriku, nie na izolovanejsi steady-state lookup.
    double total_time_ms = 0.0;
    double total_median_time_ms = 0.0;
    double throughput_mpps = 0.0;
    double median_throughput_mpps = 0.0;
    double avg_time_ms = 0.0;
    double median_time_ms = 0.0;
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;
    double speedup_vs_linear = 0.0;
    double speedup_vs_single_thread = 0.0;
    uint64_t checksum = 0;
};

// ------------------------------------------------------------
// Pomocná štruktúra pre štatistiky z viacerých behov
// ------------------------------------------------------------
struct AggregateStats {
    double avg_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

// ------------------------------------------------------------
// Výpočet priemeru, mediánu, minima a maxima z poľa nameraných hodnôt
// ------------------------------------------------------------
static AggregateStats compute_stats(const std::vector<double>& values) {
    AggregateStats s;
    if (values.empty()) return s;

    s.min_ms = *std::min_element(values.begin(), values.end());
    s.max_ms = *std::max_element(values.begin(), values.end());
    s.avg_ms = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();
    if ((n % 2u) == 1u) {
        s.median_ms = sorted[n / 2u];
    } else {
        s.median_ms = 0.5 * (sorted[(n / 2u) - 1u] + sorted[n / 2u]);
    }

    return s;
}

// ------------------------------------------------------------
// Pomocná funkcia na výpočet času v milisekundách
// ------------------------------------------------------------
static double duration_ms(const std::chrono::steady_clock::time_point& a,
                          const std::chrono::steady_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ------------------------------------------------------------
// Zápis výsledkov do CSV súboru
// ------------------------------------------------------------
// Každý riadok reprezentuje jednu kombináciu FIB + query datasetu.
// Hodnoty sú agregované z measured_runs_per_config behov.
static void write_csv(const fs::path& file_path, const std::vector<RunMetrics>& rows) {
    std::ofstream out(file_path);
    if (!out.is_open()) {
        throw std::runtime_error("Nepodarilo sa zapisat vysledny CSV: " + file_path.string());
    }

    out << "run_id,traffic_type,dataset_id,dataset_label,threads,fib_file,ip_file,fib_size,ip_count,build_time_ms,build_median_time_ms,lookup_time_ms,lookup_median_time_ms,lookup_throughput_mpps,lookup_median_throughput_mpps,router_batch_time_ms,router_batch_median_time_ms,router_batch_throughput_mpps,router_batch_median_throughput_mpps,total_time_ms,total_median_time_ms,throughput_mpps,median_throughput_mpps,avg_time_ms,median_time_ms,min_time_ms,max_time_ms,speedup_vs_linear,speedup_vs_single_thread,checksum\n";
    out << std::fixed << std::setprecision(6);

    for (const auto& r : rows) {
        out << r.run_id << ','
            << '"' << r.traffic_type << '"' << ','
            << r.dataset_id << ','
            << '"' << r.dataset_label << '"' << ','
            << r.threads << ','
            << '"' << r.fib_file << '"' << ','
            << '"' << r.ip_file << '"' << ','
            << r.fib_size << ','
            << r.ip_count << ','
            << r.build_time_ms << ','
            << r.build_median_time_ms << ','
            << r.lookup_time_ms << ','
            << r.lookup_median_time_ms << ','
            << r.lookup_throughput_mpps << ','
            << r.lookup_median_throughput_mpps << ','
            << r.router_batch_time_ms << ','
            << r.router_batch_median_time_ms << ','
            << r.router_batch_throughput_mpps << ','
            << r.router_batch_median_throughput_mpps << ','
            << r.total_time_ms << ','
            << r.total_median_time_ms << ','
            << r.throughput_mpps << ','
            << r.median_throughput_mpps << ','
            << r.avg_time_ms << ','
            << r.median_time_ms << ','
            << r.min_time_ms << ','
            << r.max_time_ms << ','
            << r.speedup_vs_linear << ','
            << r.speedup_vs_single_thread << ','
            << r.checksum << '\n';
    }
}

// ------------------------------------------------------------
// Referenčné checksumy pre syntetické kombinácie
// ------------------------------------------------------------
// Pre ENTERPRISE / HOTSPOT sa referenčný checksum typicky nepoužíva,
// pretože ide o nové BIN query datasety.
static std::unordered_map<std::string, uint64_t> make_reference_checksum_map() {
    std::unordered_map<std::string, uint64_t> m;
    m["synthetic|fib_table_10000_seed_12345.csv|generated_ips_200000_seed_12345.csv"] = 93331372ull;
    m["synthetic|fib_table_20000_seed_12345.csv|generated_ips_400000_seed_12345.csv"] = 190556109ull;
    m["synthetic|fib_table_40000_seed_12345.csv|generated_ips_800000_seed_12345.csv"] = 382296155ull;
    m["synthetic|fib_table_50000_seed_12345.csv|generated_ips_1000000_seed_12345.csv"] = 478915957ull;
    m["synthetic|fib_table_50000_seed_12345.csv|generated_ips_1500000_seed_12345.csv"] = 717944741ull;
    m["synthetic|fib_table_50000_seed_12345.csv|generated_ips_2000000_seed_12345.csv"] = 956590446ull;
    m["synthetic|fib_table_50000_seed_12345.csv|generated_ips_3000000_seed_12345.csv"] = 1433780620ull;
    return m;
}

static std::vector<DatasetConfig> make_dataset_configs(TrafficType traffic_type) {
    if (traffic_type == TrafficType::SYNTHETIC) {
        return {
            {1, "synthetic_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "generated_ips_200000_seed_12345.csv"},
            {2, "synthetic_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "generated_ips_400000_seed_12345.csv"},
            {3, "synthetic_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "generated_ips_800000_seed_12345.csv"},
            {4, "synthetic_fib50k_ips1m",  "fib_table_50000_seed_12345.csv", "generated_ips_1000000_seed_12345.csv"},
            {5, "synthetic_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "generated_ips_1500000_seed_12345.csv"},
            {6, "synthetic_fib50k_ips2m",  "fib_table_50000_seed_12345.csv", "generated_ips_2000000_seed_12345.csv"},
            {7, "synthetic_fib50k_ips3m",  "fib_table_50000_seed_12345.csv", "generated_ips_3000000_seed_12345.csv"}
        };
    }

    if (traffic_type == TrafficType::ENTERPRISE) {
        return {
            {1, "enterprise_fib10k_ips200k",   "fib_table_10000_seed_12345.csv", "queries_prefix_200000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {2, "enterprise_fib20k_ips400k",   "fib_table_20000_seed_12345.csv", "queries_prefix_400000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {3, "enterprise_fib40k_ips800k",   "fib_table_40000_seed_12345.csv", "queries_prefix_800000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {4, "enterprise_fib50k_ips1m",     "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {5, "enterprise_fib50k_ips1_5m",   "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {6, "enterprise_fib50k_ips2m",     "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {7, "enterprise_fib50k_ips3m",     "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_enterprise_b50_d35_r15_p24128_p1616.bin"}
        };
    }

    return {
        {1, "hotspot_fib10k_ips200k",   "fib_table_10000_seed_12345.csv", "queries_prefix_200000_hotspot_hp8_r95.bin"},
        {2, "hotspot_fib20k_ips400k",   "fib_table_20000_seed_12345.csv", "queries_prefix_400000_hotspot_hp8_r95.bin"},
        {3, "hotspot_fib40k_ips800k",   "fib_table_40000_seed_12345.csv", "queries_prefix_800000_hotspot_hp8_r95.bin"},
        {4, "hotspot_fib50k_ips1m",     "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_hotspot_hp8_r95.bin"},
        {5, "hotspot_fib50k_ips1_5m",   "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_hotspot_hp8_r95.bin"},
        {6, "hotspot_fib50k_ips2m",     "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_hotspot_hp8_r95.bin"},
        {7, "hotspot_fib50k_ips3m",     "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_hotspot_hp8_r95.bin"}
    };
}

static std::string make_ids_suffix(const std::vector<int>& ids) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) oss << '_';
        oss << ids[i];
    }
    return oss.str();
}

static bool contains_id(const std::vector<int>& ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

int main() {
    try {
        // --------------------------------------------------------
        // Cesty k vstupným a výstupným priečinkom
        // --------------------------------------------------------
        const fs::path fib_dir = lpm_paths::fib_dir();
        const fs::path ips_dir = lpm_paths::synthetic_ip_dir();
        const fs::path enterprise_dir = lpm_paths::enterprise_query_dir();
        const fs::path hotspot_dir = lpm_paths::hotspot_query_dir();
        const fs::path output_dir = lpm_paths::cpu_dir24_8_results_dir();

        // ========================================================
        // TRAFFIC TYPE
        // SYNTHETIC  = pôvodné CSV generated_ips_*
        // ENTERPRISE = BIN query datasety z Q_BIN_enterprise
        // HOTSPOT    = BIN query datasety z Q_BIN_hotspot
        // ========================================================
        const TrafficType traffic_type = TrafficType::SYNTHETIC;

        // --------------------------------------------------------
        // Nastavenie benchmarku
        // --------------------------------------------------------
        const int threads = 12;
        const int measured_runs_per_config = 10;
        const int warmup_runs_per_config = 5;
        const bool enable_reference_checksum_validation = true;

        // Metodika "realny smerovac":
        // - FIB/DIR24-8 struktura sa nepocita do per-batch lookup throughputu,
        //   lebo v stabilnom stave smerovaca je uz postavena v pamati.
        // - Pracovne buffre sa predalokujú a recyklujú, podobne ako packet/ring
        //   buffre v realnom systeme. Dynamicka alokacia sa preto nemeria.
        // - Hlavna throughput_mpps metrika pouziva router_batch_time_ms:
        //   kopia vstupnej davky do ingress buffera + OpenMP lookup + zapis vysledkov.
        const bool preallocate_reusable_router_buffers = true;
        const bool use_router_batch_metric_as_main_throughput = true;

        // --------------------------------------------------------
        // Výber dataset konfigurácií, ktoré sa majú benchmarkovať
        // --------------------------------------------------------
        // SYNTHETIC:
        // 1 = 10k FIB + 200k IPs
        // 2 = 20k FIB + 400k IPs
        // 3 = 40k FIB + 800k IPs
        // 4 = 50k FIB + 1M IPs
        // 5 = 50k FIB + 1.5M IPs
        // 6 = 50k FIB + 2M IPs
        // 7 = 50k FIB + 3M IPs
        //
        // ENTERPRISE / HOTSPOT:
        // 1 = 10k FIB + 200k IPs
        // 2 = 20k FIB + 400k IPs
        // 3 = 40k FIB + 800k IPs
        // 4 = 50k FIB + 1M IPs
        // 5 = 50k FIB + 1.5M IPs
        // 6 = 50k FIB + 2M IPs
        // 7 = 50k FIB + 3M IPs
        //
        // Pre kompletné meranie použi {1, 2, 3, 4, 5, 6, 7}.
        const std::vector<int> selected_dataset_ids = {1, 2, 3, 4, 5, 6, 7};

        const int hw_threads = omp_get_max_threads();
        if (threads > hw_threads) {
            throw std::runtime_error(
                "Pozadovanych je 12 threads, ale OpenMP runtime hlasi max " +
                std::to_string(hw_threads) + " threads."
            );
        }

        fs::create_directories(output_dir);

        const std::string traffic_name = traffic_type_to_string(traffic_type);
        const fs::path query_dir =
            (traffic_type == TrafficType::SYNTHETIC)  ? ips_dir :
            (traffic_type == TrafficType::ENTERPRISE) ? enterprise_dir :
                                                        hotspot_dir;

        const std::vector<DatasetConfig> all_configs = make_dataset_configs(traffic_type);
        std::vector<DatasetConfig> selected_configs;
        for (const auto& cfg : all_configs) {
            if (contains_id(selected_dataset_ids, cfg.id)) {
                selected_configs.push_back(cfg);
            }
        }

        if (selected_configs.empty()) {
            throw std::runtime_error("Nebola vybrana ziadna platna dataset konfiguracia pre traffic_type=" + traffic_name);
        }

        std::unordered_map<std::string, std::vector<Prefix>> fib_cache;
        std::unordered_map<std::string, std::vector<uint32_t>> ip_cache;
        const auto reference_checksums = make_reference_checksum_map();
        std::vector<RunMetrics> summary_rows;

        std::cout << "CPU DIR24/8 benchmark\n";
        std::cout << "Traffic type: " << traffic_name << "\n";
        std::cout << "Threads: " << threads << "\n";
        std::cout << "Measured runs/config: " << measured_runs_per_config << "\n";
        std::cout << "Warmup runs/config: " << warmup_runs_per_config << "\n";
        std::cout << "Methodology: router-like steady state, prebuilt FIB, reusable buffers\n";
        std::cout << "Main throughput_mpps: input batch copy + OpenMP lookup + output writes\n";

        for (const auto& cfg : selected_configs) {
            const fs::path fib_path = fib_dir / cfg.fib_filename;
            const fs::path ip_path = query_dir / cfg.ip_filename;

            if (!fs::exists(fib_path)) {
                throw std::runtime_error("FIB subor neexistuje: " + fib_path.string());
            }
            if (!fs::exists(ip_path)) {
                throw std::runtime_error("Query/IP subor neexistuje: " + ip_path.string());
            }

            if (fib_cache.find(cfg.fib_filename) == fib_cache.end()) {
                fib_cache.emplace(cfg.fib_filename, load_fib_csv(fib_path));
            }
            if (ip_cache.find(cfg.ip_filename) == ip_cache.end()) {
                if (traffic_type == TrafficType::SYNTHETIC) {
                    ip_cache.emplace(cfg.ip_filename, load_ips_csv(ip_path));
                } else {
                    ip_cache.emplace(cfg.ip_filename, load_ips_bin(ip_path));
                }
            }

            const auto& fib = fib_cache.at(cfg.fib_filename);
            const auto& ips = ip_cache.at(cfg.ip_filename);

            const std::string combo_key = traffic_name + "|" + cfg.fib_filename + "|" + cfg.ip_filename;

            std::cout << "\nSpracovavam ID=" << cfg.id
                      << " | " << cfg.label
                      << " | " << cfg.fib_filename
                      << " x " << cfg.ip_filename
                      << "\n";

            std::vector<double> build_times;
            std::vector<double> lookup_times;
            std::vector<double> router_batch_times;
            build_times.reserve(measured_runs_per_config);
            lookup_times.reserve(measured_runs_per_config);
            router_batch_times.reserve(measured_runs_per_config);

            uint64_t stable_checksum = 0;
            bool checksum_initialized = false;

            // ----------------------------------------------------
            // 1) Samostatne meranie build/init nakladu
            // ----------------------------------------------------
            // Toto reprezentuje inicializaciu alebo rekonfiguraciu FIB struktury.
            // Nie je to per-packet forwarding naklad, preto sa nepouziva na hlavnu
            // throughput_mpps metriku.
            const int total_runs = warmup_runs_per_config + measured_runs_per_config;
            for (int run_id = 1; run_id <= total_runs; ++run_id) {
                auto b0 = std::chrono::steady_clock::now();
                Dir24_8 build_probe;
                build_probe.build(fib);
                auto b1 = std::chrono::steady_clock::now();

                if (run_id > warmup_runs_per_config) {
                    build_times.push_back(duration_ms(b0, b1));
                }
            }

            // Aktivna FIB struktura smerovaca: postavi sa raz a nasledne sa nad
            // nou vykonavaju opakovane lookupy, ako v stabilnom forwarding stave.
            Dir24_8 dir;
            dir.build(fib);

            // Recyklovane pracovne buffre. Alokacia nie je sucastou meranej batch
            // cesty, lebo realny smerovac by pre kazdu davku paketov nevytvaral
            // nove std::vector buffre.
            std::vector<int32_t> out(ips.size());
            std::vector<uint32_t> ingress_buffer(ips.size());
            if (preallocate_reusable_router_buffers) {
                pre_touch_output_buffer(out);
                std::fill(ingress_buffer.begin(), ingress_buffer.end(), 0u);
            }

            // ----------------------------------------------------
            // 2) Steady-state lookup meranie
            // ----------------------------------------------------
            // FIB aj input su uz v pamati, vystupny buffer je predalokovany.
            // Tato metrika je ponechana v CSV pre analyzu algoritmu.
            for (int run_id = 1; run_id <= total_runs; ++run_id) {
                auto l0 = std::chrono::steady_clock::now();
                dir_lookup_omp(dir, ips, out, threads);
                auto l1 = std::chrono::steady_clock::now();

                const double lookup_time_ms = duration_ms(l0, l1);
                const uint64_t checksum = checksum_vector(out);

                if (!checksum_initialized) {
                    stable_checksum = checksum;
                    checksum_initialized = true;
                } else if (stable_checksum != checksum) {
                    throw std::runtime_error(
                        "Nekonzistentny checksum medzi steady-state lookup behovami pre kombinaciu " +
                        combo_key + " pri threads=" + std::to_string(threads)
                    );
                }

                if (run_id > warmup_runs_per_config) {
                    lookup_times.push_back(lookup_time_ms);
                }
            }

            // ----------------------------------------------------
            // 3) Realistickejsi CPU router batch model
            // ----------------------------------------------------
            // Do merania sa zapocita priprava vstupneho batchu do predalokovaneho
            // ingress buffera + paralelny lookup + zapis vystupov. Toto je hlavna
            // prakticka metrika pre porovnanie s GPU kernel+transfer.
            for (int run_id = 1; run_id <= total_runs; ++run_id) {
                auto r0 = std::chrono::steady_clock::now();
                router_forward_batch_cpu_model(dir, ips, ingress_buffer, out, threads);
                auto r1 = std::chrono::steady_clock::now();

                const double router_batch_time_ms = duration_ms(r0, r1);
                const uint64_t checksum = checksum_vector(out);

                if (stable_checksum != checksum) {
                    throw std::runtime_error(
                        "Nekonzistentny checksum medzi router-batch behovami pre kombinaciu " +
                        combo_key + " pri threads=" + std::to_string(threads)
                    );
                }

                if (run_id > warmup_runs_per_config) {
                    router_batch_times.push_back(router_batch_time_ms);

                    std::cout << "  run=" << (run_id - warmup_runs_per_config)
                              << " | router_batch_ms=" << std::fixed << std::setprecision(6) << router_batch_time_ms
                              << " | checksum=" << checksum << "\n";
                }
            }

            if (enable_reference_checksum_validation) {
                auto it = reference_checksums.find(combo_key);
                if (it != reference_checksums.end()) {
                    const uint64_t expected = it->second;
                    if (stable_checksum != expected) {
                        throw std::runtime_error(
                            "DIR checksum nesedi s referenciou pre kombinaciu " +
                            combo_key + ". Ocakavany checksum=" + std::to_string(expected) +
                            ", namerany checksum=" + std::to_string(stable_checksum)
                        );
                    }
                }
            }

            const AggregateStats build_stats = compute_stats(build_times);
            const AggregateStats lookup_stats = compute_stats(lookup_times);
            const AggregateStats router_batch_stats = compute_stats(router_batch_times);

            const double lookup_throughput_mpps =
                (lookup_stats.avg_ms > 0.0)
                    ? (static_cast<double>(ips.size()) / lookup_stats.avg_ms) / 1000.0
                    : 0.0;

            const double lookup_median_throughput_mpps =
                (lookup_stats.median_ms > 0.0)
                    ? (static_cast<double>(ips.size()) / lookup_stats.median_ms) / 1000.0
                    : 0.0;

            const double router_batch_throughput_mpps =
                (router_batch_stats.avg_ms > 0.0)
                    ? (static_cast<double>(ips.size()) / router_batch_stats.avg_ms) / 1000.0
                    : 0.0;

            const double router_batch_median_throughput_mpps =
                (router_batch_stats.median_ms > 0.0)
                    ? (static_cast<double>(ips.size()) / router_batch_stats.median_ms) / 1000.0
                    : 0.0;

            const double throughput_mpps = use_router_batch_metric_as_main_throughput
                ? router_batch_throughput_mpps
                : lookup_throughput_mpps;

            const double median_throughput_mpps = use_router_batch_metric_as_main_throughput
                ? router_batch_median_throughput_mpps
                : lookup_median_throughput_mpps;

            RunMetrics row;
            row.run_id = 0;
            row.traffic_type = traffic_name;
            row.dataset_id = cfg.id;
            row.dataset_label = cfg.label;
            row.threads = threads;
            row.fib_file = cfg.fib_filename;
            row.ip_file = cfg.ip_filename;
            row.fib_size = fib.size();
            row.ip_count = ips.size();
            row.build_time_ms = build_stats.avg_ms;
            row.build_median_time_ms = build_stats.median_ms;
            row.lookup_time_ms = lookup_stats.avg_ms;
            row.lookup_median_time_ms = lookup_stats.median_ms;
            row.lookup_throughput_mpps = lookup_throughput_mpps;
            row.lookup_median_throughput_mpps = lookup_median_throughput_mpps;
            row.router_batch_time_ms = router_batch_stats.avg_ms;
            row.router_batch_median_time_ms = router_batch_stats.median_ms;
            row.router_batch_throughput_mpps = router_batch_throughput_mpps;
            row.router_batch_median_throughput_mpps = router_batch_median_throughput_mpps;
            row.total_time_ms = build_stats.avg_ms + router_batch_stats.avg_ms;
            row.total_median_time_ms = build_stats.median_ms + router_batch_stats.median_ms;
            row.throughput_mpps = throughput_mpps;
            row.median_throughput_mpps = median_throughput_mpps;
            row.avg_time_ms = router_batch_stats.avg_ms;
            row.median_time_ms = router_batch_stats.median_ms;
            row.min_time_ms = router_batch_stats.min_ms;
            row.max_time_ms = router_batch_stats.max_ms;
            row.speedup_vs_linear = 0.0;
            row.speedup_vs_single_thread = 0.0;
            row.checksum = stable_checksum;

            summary_rows.push_back(row);

            std::cout << "  AVG lookup_ms=" << std::fixed << std::setprecision(6) << lookup_stats.avg_ms
                      << " | MED lookup_ms=" << lookup_stats.median_ms
                      << " | AVG router_batch_ms=" << router_batch_stats.avg_ms
                      << " | MED router_batch_ms=" << router_batch_stats.median_ms
                      << " | router_batch_avg_mpps=" << router_batch_throughput_mpps
                      << " | router_batch_median_mpps=" << router_batch_median_throughput_mpps
                      << " | checksum=" << stable_checksum << "\n";
        }

        const fs::path summary_file =
            output_dir / (
                "cpu_dir24_8_lpm_" + traffic_name +
                "_ids" + make_ids_suffix(selected_dataset_ids) +
                "_threads12_summary.csv"
            );

        write_csv(summary_file, summary_rows);

        std::cout << "\nUlozeny vystup: " << summary_file.string() << "\n";
        std::cout << "Hotovo.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CHYBA: " << e.what() << "\n";
        return 1;
    }
}
