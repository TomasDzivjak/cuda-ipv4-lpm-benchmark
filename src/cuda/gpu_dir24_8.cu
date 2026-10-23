#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/paths.hpp"

#include <cuda_runtime.h>

// Tento program benchmarkuje GPU implementáciu DIR-24-8 LPM lookupu.
// Postup je v skratke takýto:
// 1. Načíta FIB a IP adresy z CSV súborov.
// 2. Na CPU zostaví DIR-24-8 dátovú štruktúru.
// 3. Štruktúru „sploští“ do podoby vhodnej na kopírovanie na GPU.
// 4. Raz ju nahrá do pamäte GPU.
// 5. Pre každú konfiguráciu vykoná warmup behy a následne ostré merania.
// 6. Meria samostatne H2D prenos vstupných IP, čas kernelu a D2H prenos výsledkov.
// 7. Výsledky zapíše do summary CSV aj do raw CSV s jednotlivými behmi.

namespace fs = std::filesystem;

// Makro na kontrolu CUDA volaní.
// Ak CUDA operácia zlyhá, okamžite sa vyhodí výnimka s popisom chyby a číslom riadku.

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::ostringstream oss__;                                                    \
            oss__ << "CUDA chyba: " << cudaGetErrorString(err__)                        \
                  << " (" << static_cast<int>(err__) << ") pri volani " << #call       \
                  << " na riadku " << __LINE__;                                          \
            throw std::runtime_error(oss__.str());                                       \
        }                                                                                \
    } while (0)

// Reprezentácia jedného prefixu z FIB tabuľky.
// network  = sieťová adresa prefixu
// length   = dĺžka prefixu (/8, /24, /32, ...)
// mask     = maska prefixu
// next_hop = výsledný next hop pre daný prefix
struct Prefix {
    uint32_t network = 0;
    uint8_t length = 0;
    uint32_t mask = 0;
    int32_t next_hop = -1;
};

// Vráti kanonickú 32-bit masku pre zadanú dĺžku prefixu.
// Napr. pre /24 vráti 255.255.255.0 v binárnej podobe.
static uint32_t prefix_mask(uint8_t length) {
    if (length == 0) return 0u;
    if (length >= 32) return 0xFFFFFFFFu;
    return 0xFFFFFFFFu << (32u - length);
}

// Jednoduché rozdelenie CSV riadku na polia.
// Ošetruje aj úvodzovky, aby sa čiarky vo vnútri textu nebrali ako oddeľovač.
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

// Odstráni medzery a konce riadkov zo začiatku a konca reťazca.
static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// Načíta CSV so vstupnými IP adresami.
// Predpokladá, že druhý stĺpec obsahuje IP adresu už v číselnej 32-bit forme.
static std::vector<uint32_t> load_ips_csv(const fs::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Nepodarilo sa otvorit IP CSV: " + file_path.string());
    }

    std::string line;
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

        ips.push_back(static_cast<uint32_t>(std::stoul(trim(fields[1]))));
    }

    return ips;
}

// Načíta FIB tabuľku z CSV.
// Pre každý riadok pripraví prefix, opraví masku na kanonický tvar
// a zároveň vynuluje host bity pomocou network &= mask.

// Načíta BIN súbor so vstupnými IP adresami.
// BIN formát musí obsahovať čistú sekvenciu uint32_t IPv4 adries bez hlavičky.
static std::vector<uint32_t> load_ips_bin(const fs::path& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Nepodarilo sa otvorit BIN dataset: " + file_path.string());
    }

    file.seekg(0, std::ios::end);
    const std::size_t file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if ((file_size % sizeof(uint32_t)) != 0u) {
        throw std::runtime_error("BIN dataset nema validnu velkost, nie je nasobok 4 B: " + file_path.string());
    }

    const std::size_t count = file_size / sizeof(uint32_t);
    std::vector<uint32_t> ips(count);

    if (count > 0u) {
        file.read(reinterpret_cast<char*>(ips.data()), static_cast<std::streamsize>(file_size));
        if (!file) {
            throw std::runtime_error("Chyba pri citani BIN datasetu: " + file_path.string());
        }
    }

    return ips;
}

static std::vector<Prefix> load_fib_csv(const fs::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Nepodarilo sa otvorit FIB CSV: " + file_path.string());
    }

    std::string line;
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

        const uint32_t canonical_mask = prefix_mask(p.length);
        if (p.mask != canonical_mask) {
            p.mask = canonical_mask;
        }
        p.network &= p.mask;

        fib.push_back(p);
    }

    return fib;
}

// Pomocná funkcia: z názvu súboru vytiahne prvé číslo.
// Používa sa pri prirodzenom triedení súborov podľa veľkosti datasetu.
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

// Nájde všetky CSV súbory v priečinku a zotriedi ich podľa prvého čísla v názve.
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


// Nájde všetky BIN súbory v priečinku a zotriedi ich podľa prvého čísla v názve.
static std::vector<fs::path> list_bin_files_sorted(const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("Priecinok neexistuje alebo nie je priecinok: " + dir.string());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".bin") {
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

// CPU reprezentácia DIR-24-8 tabuľky.
// Najprv sa vytvorí host verzia štruktúry, ktorá sa následne konvertuje
// do jednoduchších polí vhodných na kopírovanie na GPU.
class Dir24_8Host {
public:
    // Sekundárna tabuľka pre prefixy dlhšie ako /24.
    // Má presne 256 položiek, teda všetky možné kombinácie posledného bajtu.
    struct SecondaryTable {
        std::array<int32_t, 256> next_hop{};
        std::array<int16_t, 256> best_len{};

        SecondaryTable() {
            next_hop.fill(-1);
            best_len.fill(-1);
        }
    };

    // Jedna položka primárnej DIR-24-8 tabuľky.
    // next_hop        = výsledok pre prefixy do /24
    // best_len        = najdlhší prefix, ktorý tu zatiaľ platí
    // secondary_index = index do poľa sekundárnych tabuliek
    // is_secondary    = či sa má pokračovať do sekundárnej tabuľky
    struct PrimaryEntry {
        int32_t next_hop = -1;
        int16_t best_len = -1;
        uint32_t secondary_index = 0;
        uint8_t is_secondary = 0;
    };

    // Zostavenie celej DIR-24-8 štruktúry z FIB tabuľky.
    // Prefixy sa spracúvajú od kratších po dlhšie, aby dlhší prefix vedel prepísať kratší.
    void build(const std::vector<Prefix>& fib) {
        primary_.assign(PRIMARY_SIZE, PrimaryEntry{});
        secondaries_.clear();

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

    const std::vector<PrimaryEntry>& primary() const { return primary_; }
    const std::vector<SecondaryTable>& secondaries() const { return secondaries_; }

private:
    static constexpr uint32_t PRIMARY_SIZE = 1u << 24;

    std::vector<PrimaryEntry> primary_;
    std::vector<SecondaryTable> secondaries_;

    // Vyplní rozsah v primárnej tabuľke pre prefixy dĺžky /0 až /24.
    // Ak už položka ukazuje do sekundárnej tabuľky, základná hodnota sa aktualizuje
    // aj tam, aby sa zachovala správna longest-prefix-match logika.
    void fill_primary_range(const Prefix& p) {
        const uint32_t start = p.network >> 8;
        const uint32_t count = (p.length == 24)
            ? 1u
            : (1u << (24u - static_cast<uint32_t>(p.length)));

        for (uint32_t i = 0; i < count; ++i) {
            PrimaryEntry& entry = primary_[start + i];

            if (!entry.is_secondary) {
                if (p.length > entry.best_len) {
                    entry.next_hop = p.next_hop;
                    entry.best_len = p.length;
                }
            } else {
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

    // Vyplní príslušný rozsah v sekundárnej tabuľke pre prefixy /25 až /32.
    // Sekundárna tabuľka sa vytvorí len vtedy, keď je naozaj potrebná.
    void fill_secondary_range(const Prefix& p) {
        const uint32_t idx24 = p.network >> 8;
        PrimaryEntry& entry = primary_[idx24];

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

        for (uint32_t i = 0; i < range; ++i) {
            const uint8_t pos = static_cast<uint8_t>(low + i);
            if (p.length > sec.best_len[pos]) {
                sec.next_hop[pos] = p.next_hop;
                sec.best_len[pos] = p.length;
            }
        }
    }
};

// Kompaktnejšia reprezentácia položky primárnej tabuľky pre GPU.
// Na GPU nepotrebujeme všetky host polia, iba tie, ktoré kernel skutočne používa.
struct DevicePrimaryEntry {
    int32_t next_hop;
    int32_t secondary_index;
    uint8_t is_secondary;
    uint8_t pad[3];
};
static_assert(sizeof(DevicePrimaryEntry) == 12, "DevicePrimaryEntry must be 12 bytes");

// Konverzia host primárnej tabuľky do GPU-friendly poľa.
static std::vector<DevicePrimaryEntry> flatten_primary(const std::vector<Dir24_8Host::PrimaryEntry>& src) {
    std::vector<DevicePrimaryEntry> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[i].next_hop = src[i].next_hop;
        out[i].secondary_index = static_cast<int32_t>(src[i].secondary_index);
        out[i].is_secondary = src[i].is_secondary;
        out[i].pad[0] = out[i].pad[1] = out[i].pad[2] = 0;
    }
    return out;
}

// Konverzia všetkých sekundárnych tabuliek do jedného lineárneho poľa next-hop hodnôt.
// Na GPU sa tak dá indexovať jednoducho cez secondary_index * 256 + low8.
static std::vector<int32_t> flatten_secondaries_next_hop(const std::vector<Dir24_8Host::SecondaryTable>& src) {
    std::vector<int32_t> out(src.size() * 256ull, -1);
    for (std::size_t i = 0; i < src.size(); ++i) {
        for (std::size_t j = 0; j < 256; ++j) {
            out[i * 256ull + j] = src[i].next_hop[j];
        }
    }
    return out;
}

// Samotný GPU kernel pre DIR-24-8 lookup.
// Každé vlákno spracúva presne jednu IP adresu.
// 1. Zoberie horných 24 bitov a nájde položku v primárnej tabuľke.
// 2. Ak stačí primárna tabuľka, výsledok je hotový.
// 3. Ak treba, použije sa dolných 8 bitov na vstup do sekundárnej tabuľky.
__global__
void dir24_8_lookup_kernel(const DevicePrimaryEntry* __restrict__ primary,
                           const int32_t* __restrict__ secondaries_next_hop,
                           const uint32_t* __restrict__ ips,
                           int32_t* __restrict__ out_next_hop,
                           std::size_t ip_count) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= ip_count) return;

    const uint32_t ip = ips[idx];
    const uint32_t idx24 = ip >> 8;
    const DevicePrimaryEntry e = primary[idx24];

    int32_t result = e.next_hop;
    if (e.is_secondary) {
        const uint8_t low8 = static_cast<uint8_t>(ip & 0xFFu);
        result = secondaries_next_hop[static_cast<std::size_t>(e.secondary_index) * 256u + low8];
    }

    out_next_hop[idx] = result;
}

// Jednoduchý checksum nad výsledným poľom next-hopov.
// Slúži na kontrolu, že všetky behy dávajú rovnaký výsledok.
static uint64_t checksum_vector(const int32_t* data, std::size_t size) {
    uint64_t sum = 0;
    for (std::size_t i = 0; i < size; ++i) {
        sum += static_cast<uint64_t>(static_cast<uint32_t>(data[i]));
    }
    return sum;
}

// Agregované štatistiky z viacerých meraní tej istej konfigurácie.
struct AggregateStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double stddev = 0.0;
};

// Z vektora hodnôt vypočíta priemer, minimum, maximum, medián a smerodajnú odchýlku.
static AggregateStats compute_stats(const std::vector<double>& values) {
    AggregateStats s;
    if (values.empty()) return s;

    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    s.avg = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    if ((sorted.size() % 2u) == 1u) {
        s.median = sorted[sorted.size() / 2u];
    } else {
        const std::size_t right = sorted.size() / 2u;
        const std::size_t left = right - 1u;
        s.median = (sorted[left] + sorted[right]) / 2.0;
    }

    double variance = 0.0;
    for (double v : values) {
        const double d = v - s.avg;
        variance += d * d;
    }
    variance /= static_cast<double>(values.size());
    s.stddev = std::sqrt(variance);
    return s;
}

// Referenčné checksumy pre známe datasety.
// Ak je validácia zapnutá, program si overí, že GPU implementácia dáva očakávaný výsledok.
static std::unordered_map<std::string, uint64_t> make_reference_checksum_map() {
    std::unordered_map<std::string, uint64_t> m;
    m["fib_table_10000_seed_12345.csv|generated_ips_200000_seed_12345.csv"] = 93331372ull;
    m["fib_table_20000_seed_12345.csv|generated_ips_400000_seed_12345.csv"] = 190556109ull;
    m["fib_table_40000_seed_12345.csv|generated_ips_800000_seed_12345.csv"] = 382296155ull;
    m["fib_table_50000_seed_12345.csv|generated_ips_1000000_seed_12345.csv"] = 478915957ull;
    m["fib_table_50000_seed_12345.csv|generated_ips_1500000_seed_12345.csv"] = 717944741ull;
    m["fib_table_50000_seed_12345.csv|generated_ips_2000000_seed_12345.csv"] = 956590446ull;
    m["fib_table_50000_seed_12345.csv|generated_ips_3000000_seed_12345.csv"] = 1433780620ull;
    return m;
}

// Jeden riadok pre raw CSV, teda detailné údaje z jedného konkrétneho behu.
struct RawRunRow {
    std::string fib_file;
    std::string ip_file;
    std::size_t fib_size = 0;
    std::size_t ip_count = 0;
    bool warmup = false;
    int run_id = 0;
    int kernel_repeats_per_measurement = 0;
    double h2d_ips_ms = 0.0;
    double kernel_total_ms = 0.0;
    double kernel_avg_ms = 0.0;
    double d2h_ms = 0.0;
    double transfer_plus_kernel_effective_ms = 0.0;
    double total_end_to_end_effective_ms = 0.0;
    double kernel_throughput_mpps = 0.0;
    uint64_t checksum = 0;
};

// Jeden riadok pre summary CSV, teda agregované výsledky za celú konfiguráciu.
struct ResultRow {
    std::string fib_file;
    std::string ip_file;
    std::size_t fib_size = 0;
    std::size_t ip_count = 0;
    std::size_t primary_entries = 0;
    std::size_t secondary_tables = 0;
    std::size_t secondary_slots = 0;
    int warmup_runs = 0;
    int measured_runs = 0;
    int kernel_repeats_per_measurement = 0;
    int pre_measure_sleep_ms = 0;
    int block_size = 0;
    int grid_size = 0;

    double build_once_ms = 0.0;
    double flatten_once_ms = 0.0;
    double upload_primary_once_ms = 0.0;
    double upload_secondaries_once_ms = 0.0;
    double upload_tables_once_ms = 0.0;
    double setup_once_total_ms = 0.0;

    double h2d_ips_avg_ms = 0.0;
    double h2d_ips_min_ms = 0.0;
    double h2d_ips_max_ms = 0.0;
    double h2d_ips_median_ms = 0.0;
    double h2d_ips_stddev_ms = 0.0;

    double kernel_lookup_avg_ms = 0.0;
    double kernel_lookup_min_ms = 0.0;
    double kernel_lookup_max_ms = 0.0;
    double kernel_lookup_median_ms = 0.0;
    double kernel_lookup_stddev_ms = 0.0;
    double kernel_lookup_throughput_mpps_avg = 0.0;
    double kernel_lookup_throughput_mpps_median = 0.0;

    double d2h_avg_ms = 0.0;
    double d2h_min_ms = 0.0;
    double d2h_max_ms = 0.0;
    double d2h_median_ms = 0.0;
    double d2h_stddev_ms = 0.0;

    double transfer_plus_kernel_avg_ms = 0.0;
    double transfer_plus_kernel_min_ms = 0.0;
    double transfer_plus_kernel_max_ms = 0.0;
    double transfer_plus_kernel_median_ms = 0.0;
    double transfer_plus_kernel_stddev_ms = 0.0;
    double transfer_plus_kernel_throughput_mpps_avg = 0.0;
    double transfer_plus_kernel_throughput_mpps_median = 0.0;

    double total_end_to_end_avg_ms = 0.0;
    double total_end_to_end_min_ms = 0.0;
    double total_end_to_end_max_ms = 0.0;
    double total_end_to_end_median_ms = 0.0;
    double total_end_to_end_stddev_ms = 0.0;
    double total_end_to_end_throughput_mpps_avg = 0.0;
    double total_end_to_end_throughput_mpps_median = 0.0;

    uint64_t checksum = 0;
};

// Zapíše agregované výsledky do summary CSV.
static void write_summary_csv(const fs::path& file_path, const std::vector<ResultRow>& rows) {
    std::ofstream out(file_path);
    if (!out.is_open()) {
        throw std::runtime_error("Nepodarilo sa zapisat vysledny CSV: " + file_path.string());
    }

    out << "fib_file,ip_file,fib_size,ip_count,primary_entries,secondary_tables,secondary_slots,"
        << "warmup_runs,measured_runs,kernel_repeats_per_measurement,pre_measure_sleep_ms,block_size,grid_size,"
        << "build_once_ms,flatten_once_ms,upload_primary_once_ms,upload_secondaries_once_ms,upload_tables_once_ms,setup_once_total_ms,"
        << "h2d_ips_avg_ms,h2d_ips_min_ms,h2d_ips_max_ms,h2d_ips_median_ms,h2d_ips_stddev_ms,"
        << "kernel_lookup_avg_ms,kernel_lookup_min_ms,kernel_lookup_max_ms,kernel_lookup_median_ms,kernel_lookup_stddev_ms,kernel_lookup_throughput_mpps_avg,kernel_lookup_throughput_mpps_median,"
        << "d2h_avg_ms,d2h_min_ms,d2h_max_ms,d2h_median_ms,d2h_stddev_ms,"
        << "transfer_plus_kernel_avg_ms,transfer_plus_kernel_min_ms,transfer_plus_kernel_max_ms,transfer_plus_kernel_median_ms,transfer_plus_kernel_stddev_ms,transfer_plus_kernel_throughput_mpps_avg,transfer_plus_kernel_throughput_mpps_median,"
        << "total_end_to_end_avg_ms,total_end_to_end_min_ms,total_end_to_end_max_ms,total_end_to_end_median_ms,total_end_to_end_stddev_ms,total_end_to_end_throughput_mpps_avg,total_end_to_end_throughput_mpps_median,"
        << "checksum\n";

    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << '"' << r.fib_file << '"' << ','
            << '"' << r.ip_file << '"' << ','
            << r.fib_size << ','
            << r.ip_count << ','
            << r.primary_entries << ','
            << r.secondary_tables << ','
            << r.secondary_slots << ','
            << r.warmup_runs << ','
            << r.measured_runs << ','
            << r.kernel_repeats_per_measurement << ','
            << r.pre_measure_sleep_ms << ','
            << r.block_size << ','
            << r.grid_size << ','
            << r.build_once_ms << ','
            << r.flatten_once_ms << ','
            << r.upload_primary_once_ms << ','
            << r.upload_secondaries_once_ms << ','
            << r.upload_tables_once_ms << ','
            << r.setup_once_total_ms << ','
            << r.h2d_ips_avg_ms << ','
            << r.h2d_ips_min_ms << ','
            << r.h2d_ips_max_ms << ','
            << r.h2d_ips_median_ms << ','
            << r.h2d_ips_stddev_ms << ','
            << r.kernel_lookup_avg_ms << ','
            << r.kernel_lookup_min_ms << ','
            << r.kernel_lookup_max_ms << ','
            << r.kernel_lookup_median_ms << ','
            << r.kernel_lookup_stddev_ms << ','
            << r.kernel_lookup_throughput_mpps_avg << ','
            << r.kernel_lookup_throughput_mpps_median << ','
            << r.d2h_avg_ms << ','
            << r.d2h_min_ms << ','
            << r.d2h_max_ms << ','
            << r.d2h_median_ms << ','
            << r.d2h_stddev_ms << ','
            << r.transfer_plus_kernel_avg_ms << ','
            << r.transfer_plus_kernel_min_ms << ','
            << r.transfer_plus_kernel_max_ms << ','
            << r.transfer_plus_kernel_median_ms << ','
            << r.transfer_plus_kernel_stddev_ms << ','
            << r.transfer_plus_kernel_throughput_mpps_avg << ','
            << r.transfer_plus_kernel_throughput_mpps_median << ','
            << r.total_end_to_end_avg_ms << ','
            << r.total_end_to_end_min_ms << ','
            << r.total_end_to_end_max_ms << ','
            << r.total_end_to_end_median_ms << ','
            << r.total_end_to_end_stddev_ms << ','
            << r.total_end_to_end_throughput_mpps_avg << ','
            << r.total_end_to_end_throughput_mpps_median << ','
            << r.checksum << '\n';
    }
}

// Zapíše detailné výsledky jednotlivých behov do raw CSV.
static void write_raw_runs_csv(const fs::path& file_path, const std::vector<RawRunRow>& rows) {
    std::ofstream out(file_path);
    if (!out.is_open()) {
        throw std::runtime_error("Nepodarilo sa zapisat raw run CSV: " + file_path.string());
    }

    out << "fib_file,ip_file,fib_size,ip_count,warmup,run_id,kernel_repeats_per_measurement,"
        << "h2d_ips_ms,kernel_total_ms,kernel_avg_ms,d2h_ms,transfer_plus_kernel_effective_ms,total_end_to_end_effective_ms,kernel_throughput_mpps,checksum\n";

    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << '"' << r.fib_file << '"' << ','
            << '"' << r.ip_file << '"' << ','
            << r.fib_size << ','
            << r.ip_count << ','
            << (r.warmup ? 1 : 0) << ','
            << r.run_id << ','
            << r.kernel_repeats_per_measurement << ','
            << r.h2d_ips_ms << ','
            << r.kernel_total_ms << ','
            << r.kernel_avg_ms << ','
            << r.d2h_ms << ','
            << r.transfer_plus_kernel_effective_ms << ','
            << r.total_end_to_end_effective_ms << ','
            << r.kernel_throughput_mpps << ','
            << r.checksum << '\n';
    }
}

// Vráti čas medzi dvoma CUDA eventmi v milisekundách.
static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// Vráti čas medzi dvoma CPU časovými bodmi v milisekundách.
static double duration_ms(const std::chrono::high_resolution_clock::time_point& a,
                          const std::chrono::high_resolution_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}


// Typ query datasetu.
// SYNTHETIC  = pôvodné CSV datasety generated_ips_*
// ENTERPRISE = BIN datasety s enterprise locality modelom
// HOTSPOT    = BIN datasety s vysokou /24 locality
enum class TrafficType {
    SYNTHETIC,
    ENTERPRISE,
    HOTSPOT
};

// Jedna benchmark konfigurácia: ID, výstupný label, FIB súbor a query súbor.
struct DatasetConfig {
    int id = 0;
    std::string label;
    std::string fib_name;
    std::string ip_name;
};

// Vytvorí suffix pre názvy výstupných CSV v rovnakom štýle ako CPU benchmarky.
// Namiesto veľmi dlhého názvu s FIB/IP veľkosťami použije iba typ trafficu
// a ID vybraných datasetov, napr. hotspot_ids1_2_3_4_5_6_7.
static std::string make_output_suffix(const std::vector<DatasetConfig>& selected_configs) {
    if (selected_configs.empty()) {
        throw std::runtime_error("Nie je vybrana ziadna dataset konfiguracia.");
    }

    std::string traffic_prefix = "selected";
    const std::string& first_label = selected_configs.front().label;
    const std::size_t fib_pos = first_label.find("_fib");
    if (fib_pos != std::string::npos) {
        traffic_prefix = first_label.substr(0, fib_pos);
    }

    std::ostringstream oss;
    oss << traffic_prefix << "_ids";

    for (std::size_t i = 0; i < selected_configs.size(); ++i) {
        if (i > 0u) {
            oss << "_";
        }
        oss << selected_configs[i].id;
    }

    return oss.str();
}

int main() {
    // Hlavný benchmark scenár:
    // - pripraví cesty a parametre merania
    // - načíta datasety
    // - pre každú konfiguráciu postaví DIR-24-8 tabuľku
    // - nahrá tabuľky na GPU
    // - vykoná warmup a merané behy
    // - spočíta štatistiky a uloží CSV výstupy
    try {
        // Cesty k vstupným a výstupným priečinkom.
        const fs::path fib_dir = lpm_paths::fib_dir();
        const fs::path ips_dir = lpm_paths::synthetic_ip_dir();
        const fs::path enterprise_dir = lpm_paths::enterprise_query_dir();
        const fs::path hotspot_dir = lpm_paths::hotspot_query_dir();
        const fs::path output_dir = lpm_paths::cuda_dir24_8_results_dir();

        // ============================================================
        // TRAFFIC TYPE
        // SYNTHETIC  = pôvodné CSV generated_ips_*
        // ENTERPRISE = BIN query datasety z Q_BIN_enterprise
        // HOTSPOT    = BIN query datasety z Q_BIN_hotspot
        // ============================================================
        const TrafficType traffic_type = TrafficType::HOTSPOT;

        // Nastavenie benchmarku.
        const int measured_runs_per_config = 10;
        const int warmup_runs_per_config = 5;
        const int kernel_repeats_per_measurement = 20;
        const int pre_measure_sleep_ms = 25;
        const int block_size = 256;
        const bool enable_reference_checksum_validation = true;

        // Vyber dataset konfigurácií, ktoré sa majú benchmarkovať.
        //
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

        // Vytvorenie výstupného priečinka a inicializácia CUDA kontextu.
        fs::create_directories(output_dir);
        CUDA_CHECK(cudaFree(nullptr));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Global memory: " << (prop.totalGlobalMem / (1024ull * 1024ull)) << " MiB\n";
        std::cout << "Algorithm: GPU DIR24/8 LPM lookup\n";
        std::cout << "Traffic type: "
                  << (traffic_type == TrafficType::SYNTHETIC ? "SYNTHETIC" :
                      traffic_type == TrafficType::ENTERPRISE ? "ENTERPRISE" : "HOTSPOT")
                  << "\n";
        std::cout << "Warmup runs/config: " << warmup_runs_per_config << "\n";
        std::cout << "Measured runs/config: " << measured_runs_per_config << "\n";
        std::cout << "Kernel repeats/measurement: " << kernel_repeats_per_measurement << "\n";
        std::cout << "Pre-measure sleep [ms]: " << pre_measure_sleep_ms << "\n";

        // Získanie zoznamu FIB CSV súborov a query súborov podľa zvoleného traffic typu.
        const std::vector<fs::path> fib_files = list_csv_files_sorted(fib_dir);
        std::vector<fs::path> ip_files;

        if (traffic_type == TrafficType::SYNTHETIC) {
            ip_files = list_csv_files_sorted(ips_dir);
        } else if (traffic_type == TrafficType::ENTERPRISE) {
            ip_files = list_bin_files_sorted(enterprise_dir);
        } else {
            ip_files = list_bin_files_sorted(hotspot_dir);
        }

        if (fib_files.empty()) {
            throw std::runtime_error("V FIB priecinku sa nenasli ziadne CSV subory.");
        }
        if (ip_files.empty()) {
            throw std::runtime_error("V IP/BIN priecinku sa nenasli ziadne query subory pre zvoleny traffic type.");
        }

        std::unordered_map<std::string, fs::path> fib_by_name;
        std::unordered_map<std::string, fs::path> ip_by_name;
        for (const auto& f : fib_files) fib_by_name.emplace(f.filename().string(), f);
        for (const auto& f : ip_files) ip_by_name.emplace(f.filename().string(), f);

        // Presne definované kombinácie FIB a query datasetov.
        std::vector<DatasetConfig> all_dataset_configs;

        if (traffic_type == TrafficType::SYNTHETIC) {
            all_dataset_configs = {
                {1, "synthetic_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "generated_ips_200000_seed_12345.csv"},
                {2, "synthetic_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "generated_ips_400000_seed_12345.csv"},
                {3, "synthetic_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "generated_ips_800000_seed_12345.csv"},
                {4, "synthetic_fib50k_ips1m", "fib_table_50000_seed_12345.csv", "generated_ips_1000000_seed_12345.csv"},
                {5, "synthetic_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "generated_ips_1500000_seed_12345.csv"},
                {6, "synthetic_fib50k_ips2m", "fib_table_50000_seed_12345.csv", "generated_ips_2000000_seed_12345.csv"},
                {7, "synthetic_fib50k_ips3m", "fib_table_50000_seed_12345.csv", "generated_ips_3000000_seed_12345.csv"}
            };
        } else if (traffic_type == TrafficType::ENTERPRISE) {
            all_dataset_configs = {
                {1, "enterprise_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "queries_prefix_200000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {2, "enterprise_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "queries_prefix_400000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {3, "enterprise_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "queries_prefix_800000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {4, "enterprise_fib50k_ips1m", "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {5, "enterprise_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {6, "enterprise_fib50k_ips2m", "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {7, "enterprise_fib50k_ips3m", "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_enterprise_b50_d35_r15_p24128_p1616.bin"}
            };
        } else {
            all_dataset_configs = {
                {1, "hotspot_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "queries_prefix_200000_hotspot_hp8_r95.bin"},
                {2, "hotspot_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "queries_prefix_400000_hotspot_hp8_r95.bin"},
                {3, "hotspot_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "queries_prefix_800000_hotspot_hp8_r95.bin"},
                {4, "hotspot_fib50k_ips1m", "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_hotspot_hp8_r95.bin"},
                {5, "hotspot_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_hotspot_hp8_r95.bin"},
                {6, "hotspot_fib50k_ips2m", "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_hotspot_hp8_r95.bin"},
                {7, "hotspot_fib50k_ips3m", "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_hotspot_hp8_r95.bin"}
            };
        }

        if (selected_dataset_ids.empty()) {
            throw std::runtime_error("Nie je vybrata ziadna dataset konfiguracia.");
        }

        std::vector<DatasetConfig> dataset_configs;
        dataset_configs.reserve(selected_dataset_ids.size());

        for (const int selected_id : selected_dataset_ids) {
            auto cfg_it = std::find_if(all_dataset_configs.begin(), all_dataset_configs.end(),
                                       [selected_id](const DatasetConfig& cfg) { return cfg.id == selected_id; });
            if (cfg_it == all_dataset_configs.end()) {
                std::ostringstream oss;
                oss << "Neplatne ID dataset konfiguracie: " << selected_id
                    << ". Pre SYNTHETIC/ENTERPRISE/HOTSPOT pouzi ID 1 az 7.";
                throw std::runtime_error(oss.str());
            }
            dataset_configs.push_back(*cfg_it);
        }

        const std::string output_suffix = make_output_suffix(dataset_configs);
        const fs::path output_csv = output_dir / ("gpu_dir24_8_lpm_" + output_suffix + "_summary.csv");
        const fs::path output_raw_csv = output_dir / ("gpu_dir24_8_lpm_" + output_suffix + "_raw_runs.csv");

        std::cout << "Vybrane dataset konfiguracie:";
        for (const auto& cfg : dataset_configs) {
            std::cout << " " << cfg.id << "=" << cfg.label;
        }
        std::cout << "\n";
        std::cout << "Summary CSV: " << output_csv.string() << "\n";
        std::cout << "Raw runs CSV: " << output_raw_csv.string() << "\n";

        // Cache na uloženie už načítaných datasetov, aby sa ten istý súbor
        // pri opakovanom použití znovu nečítal z disku.
        std::unordered_map<std::string, std::vector<Prefix>> fib_cache;
        std::unordered_map<std::string, std::vector<uint32_t>> ip_cache;

        // Prednačítanie všetkých použitých FIB a query datasetov.
        for (const auto& cfg : dataset_configs) {
            const std::string& fib_name = cfg.fib_name;
            const std::string& ip_name = cfg.ip_name;

            auto fib_it = fib_by_name.find(fib_name);
            if (fib_it == fib_by_name.end()) {
                throw std::runtime_error("V FIB priecinku chyba subor: " + fib_name);
            }
            auto ip_it = ip_by_name.find(ip_name);
            if (ip_it == ip_by_name.end()) {
                throw std::runtime_error("V IP/BIN priecinku chyba query subor: " + ip_name);
            }

            if (fib_cache.find(fib_it->second.string()) == fib_cache.end()) {
                fib_cache.emplace(fib_it->second.string(), load_fib_csv(fib_it->second));
            }
            if (ip_cache.find(ip_it->second.string()) == ip_cache.end()) {
                if (traffic_type == TrafficType::SYNTHETIC) {
                    ip_cache.emplace(ip_it->second.string(), load_ips_csv(ip_it->second));
                } else {
                    ip_cache.emplace(ip_it->second.string(), load_ips_bin(ip_it->second));
                }
            }
        }

        const auto reference_checksums = make_reference_checksum_map();
        std::vector<ResultRow> summary_rows;
        std::vector<RawRunRow> raw_rows;
        summary_rows.reserve(dataset_configs.size());
        raw_rows.reserve(dataset_configs.size() * static_cast<std::size_t>(warmup_runs_per_config + measured_runs_per_config));

        // CUDA eventy používané na presné meranie H2D, kernelu a D2H.
        cudaEvent_t ev_start{}, ev_stop{};
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        for (const auto& cfg : dataset_configs) {
            const std::string& fib_name = cfg.fib_name;
            const std::string& ip_name = cfg.ip_name;

            const fs::path& fib_file = fib_by_name.at(fib_name);
            const fs::path& ip_file = ip_by_name.at(ip_name);
            const auto& fib = fib_cache.at(fib_file.string());
            const auto& ips = ip_cache.at(ip_file.string());
            const std::string combo_key = fib_file.filename().string() + "|" + ip_file.filename().string();

            std::cout << "\nSpracovavam: " << fib_file.filename().string()
                      << " x " << ip_file.filename().string() << "\n";

            // Výpočet počtu blokov tak, aby sa pokryli všetky IP adresy.
            const int grid_size = static_cast<int>((ips.size() + static_cast<std::size_t>(block_size) - 1) /
                                                   static_cast<std::size_t>(block_size));

            // 1. CPU build DIR-24-8 tabuľky.
            auto t_build_0 = std::chrono::high_resolution_clock::now();
            Dir24_8Host dir;
            dir.build(fib);
            auto t_build_1 = std::chrono::high_resolution_clock::now();
            const double build_once_ms = duration_ms(t_build_0, t_build_1);

            // 2. Prevod host štruktúr do lineárnych polí vhodných pre GPU.
            auto t_flatten_0 = std::chrono::high_resolution_clock::now();
            std::vector<DevicePrimaryEntry> primary_flat = flatten_primary(dir.primary());
            std::vector<int32_t> secondaries_flat = flatten_secondaries_next_hop(dir.secondaries());
            auto t_flatten_1 = std::chrono::high_resolution_clock::now();
            const double flatten_once_ms = duration_ms(t_flatten_0, t_flatten_1);

            const std::size_t primary_entries = primary_flat.size();
            const std::size_t secondary_tables = dir.secondaries().size();
            const std::size_t secondary_slots = secondaries_flat.size();

            // 3. Alokácia GPU pamäte pre tabuľky, vstupné IP aj výstupné next-hopy.
            DevicePrimaryEntry* d_primary = nullptr;
            int32_t* d_secondaries = nullptr;
            uint32_t* d_ips = nullptr;
            int32_t* d_out = nullptr;

            CUDA_CHECK(cudaMalloc(&d_primary, primary_flat.size() * sizeof(DevicePrimaryEntry)));
            if (!secondaries_flat.empty()) {
                CUDA_CHECK(cudaMalloc(&d_secondaries, secondaries_flat.size() * sizeof(int32_t)));
            }
            CUDA_CHECK(cudaMalloc(&d_ips, ips.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_out, ips.size() * sizeof(int32_t)));

            // 4. Jednorazový upload primárnej tabuľky na GPU.
            CUDA_CHECK(cudaEventRecord(ev_start));
            CUDA_CHECK(cudaMemcpy(d_primary, primary_flat.data(), primary_flat.size() * sizeof(DevicePrimaryEntry), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            const double upload_primary_once_ms = elapsed_ms(ev_start, ev_stop);

            // 5. Jednorazový upload sekundárnych tabuliek na GPU.
            double upload_secondaries_once_ms = 0.0;
            if (!secondaries_flat.empty()) {
                CUDA_CHECK(cudaEventRecord(ev_start));
                CUDA_CHECK(cudaMemcpy(d_secondaries, secondaries_flat.data(), secondaries_flat.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                upload_secondaries_once_ms = elapsed_ms(ev_start, ev_stop);
            }
            const double upload_tables_once_ms = upload_primary_once_ms + upload_secondaries_once_ms;
            // Celkový jednorazový setup pre danú konfiguráciu.
            const double setup_once_total_ms = build_once_ms + flatten_once_ms + upload_tables_once_ms;

            // 6. Alokácia pinned host pamäte.
            // Pinned pamäť zrýchľuje kopírovanie medzi CPU a GPU a znižuje jitter pri meraní.
            uint32_t* h_ips_pinned = nullptr;
            int32_t* h_out_pinned = nullptr;
            CUDA_CHECK(cudaMallocHost(&h_ips_pinned, ips.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMallocHost(&h_out_pinned, ips.size() * sizeof(int32_t)));
            std::copy(ips.begin(), ips.end(), h_ips_pinned);

            std::vector<double> h2d_ips_ms_values, kernel_ms_values, d2h_ms_values;
            std::vector<double> transfer_plus_kernel_ms_values, total_end_to_end_ms_values;

            uint64_t stable_checksum = 0;
            bool checksum_initialized = false;

            // Lambda reprezentujúca jeden benchmark beh.
            // warmup_phase = true  -> výsledok sa nezapočíta do štatistík
            // warmup_phase = false -> ostré meranie
            auto run_lookup = [&](bool warmup_phase, int run_id) {
                // Pred meraním sa čaká na dokončenie všetkých predošlých GPU operácií.
                CUDA_CHECK(cudaDeviceSynchronize());
                if (pre_measure_sleep_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(pre_measure_sleep_ms));
                }

                // 6a. Prenos vstupných IP adries z CPU do GPU.
                CUDA_CHECK(cudaEventRecord(ev_start));
                CUDA_CHECK(cudaMemcpy(d_ips, h_ips_pinned, ips.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                const double h2d_ips_ms = elapsed_ms(ev_start, ev_stop);

                // 6b. Meranie samotného kernelu.
                // Kernel sa spúšťa viackrát za sebou, aby bol výsledný čas stabilnejší.
                CUDA_CHECK(cudaEventRecord(ev_start));
                for (int repeat = 0; repeat < kernel_repeats_per_measurement; ++repeat) {
                    dir24_8_lookup_kernel<<<grid_size, block_size>>>(d_primary, d_secondaries, d_ips, d_out, ips.size());
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                const double kernel_total_ms = elapsed_ms(ev_start, ev_stop);
                const double kernel_avg_ms = kernel_total_ms / static_cast<double>(kernel_repeats_per_measurement);

                // 6c. Prenos výsledných next-hopov späť z GPU do CPU.
                CUDA_CHECK(cudaEventRecord(ev_start));
                CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, ips.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                const double d2h_ms = elapsed_ms(ev_start, ev_stop);

                // transfer_plus_kernel_effective_ms = praktický lookup čas dávky bez jednorazového buildu tabuliek.
                const double transfer_plus_kernel_effective_ms = h2d_ips_ms + kernel_avg_ms + d2h_ms;
                const double total_end_to_end_effective_ms = setup_once_total_ms + transfer_plus_kernel_effective_ms;
                const double kernel_throughput_mpps =
                    (kernel_avg_ms > 0.0) ? (static_cast<double>(ips.size()) / kernel_avg_ms) / 1000.0 : 0.0;

                // Overenie správnosti výsledku cez checksum.
                const uint64_t checksum = checksum_vector(h_out_pinned, ips.size());
                if (!checksum_initialized) {
                    stable_checksum = checksum;
                    checksum_initialized = true;
                } else if (checksum != stable_checksum) {
                    throw std::runtime_error("Nekonzistentny checksum medzi behovami pre kombinaciu " + combo_key);
                }

                raw_rows.push_back(RawRunRow{
                    fib_file.filename().string(),
                    ip_file.filename().string(),
                    fib.size(),
                    ips.size(),
                    warmup_phase,
                    run_id,
                    kernel_repeats_per_measurement,
                    h2d_ips_ms,
                    kernel_total_ms,
                    kernel_avg_ms,
                    d2h_ms,
                    transfer_plus_kernel_effective_ms,
                    total_end_to_end_effective_ms,
                    kernel_throughput_mpps,
                    checksum
                });

                if (!warmup_phase) {
                    h2d_ips_ms_values.push_back(h2d_ips_ms);
                    kernel_ms_values.push_back(kernel_avg_ms);
                    d2h_ms_values.push_back(d2h_ms);
                    transfer_plus_kernel_ms_values.push_back(transfer_plus_kernel_effective_ms);
                    total_end_to_end_ms_values.push_back(total_end_to_end_effective_ms);
                }

                std::cout << "  " << (warmup_phase ? "warmup" : "measured")
                          << " run=" << run_id
                          << " | h2d_ips_ms=" << std::fixed << std::setprecision(3) << h2d_ips_ms
                          << " | kernel_total_ms=" << kernel_total_ms
                          << " | kernel_avg_ms=" << kernel_avg_ms
                          << " | d2h_ms=" << d2h_ms
                          << " | transfer_plus_kernel_effective_ms=" << transfer_plus_kernel_effective_ms
                          << " | total_end_to_end_effective_ms=" << total_end_to_end_effective_ms
                          << " | kernel_mpps=" << kernel_throughput_mpps
                          << " | checksum=" << checksum << "\n";
            };

            // Najprv prebehnú warmup behy, ktoré majú stabilizovať GPU.
            for (int run = 1; run <= warmup_runs_per_config; ++run) {
                run_lookup(true, run);
            }
            for (int run = 1; run <= measured_runs_per_config; ++run) {
                run_lookup(false, run);
            }

            // Voliteľná validácia voči referenčným checksumom.
            if (enable_reference_checksum_validation) {
                auto it = reference_checksums.find(combo_key);
                if (it != reference_checksums.end() && stable_checksum != it->second) {
                    throw std::runtime_error("GPU DIR24/8 checksum nesedi s referenciou pre kombinaciu " + combo_key +
                                             ". Ockavany checksum=" + std::to_string(it->second) +
                                             ", namerany checksum=" + std::to_string(stable_checksum));
                }
            }

            // Uvoľnenie host aj device pamäte po dokončení konfigurácie.
            CUDA_CHECK(cudaFreeHost(h_ips_pinned));
            CUDA_CHECK(cudaFreeHost(h_out_pinned));
            CUDA_CHECK(cudaFree(d_primary));
            if (d_secondaries != nullptr) CUDA_CHECK(cudaFree(d_secondaries));
            CUDA_CHECK(cudaFree(d_ips));
            CUDA_CHECK(cudaFree(d_out));

            // Výpočet agregovaných štatistík zo všetkých ostrých behov.
            const AggregateStats h2d_ips_stats = compute_stats(h2d_ips_ms_values);
            const AggregateStats kernel_stats = compute_stats(kernel_ms_values);
            const AggregateStats d2h_stats = compute_stats(d2h_ms_values);
            const AggregateStats transfer_plus_kernel_stats = compute_stats(transfer_plus_kernel_ms_values);
            const AggregateStats total_end_to_end_stats = compute_stats(total_end_to_end_ms_values);

            ResultRow row;
            row.fib_file = fib_file.filename().string();
            row.ip_file = ip_file.filename().string();
            row.fib_size = fib.size();
            row.ip_count = ips.size();
            row.primary_entries = primary_entries;
            row.secondary_tables = secondary_tables;
            row.secondary_slots = secondary_slots;
            row.warmup_runs = warmup_runs_per_config;
            row.measured_runs = measured_runs_per_config;
            row.kernel_repeats_per_measurement = kernel_repeats_per_measurement;
            row.pre_measure_sleep_ms = pre_measure_sleep_ms;
            row.block_size = block_size;
            row.grid_size = grid_size;
            row.build_once_ms = build_once_ms;
            row.flatten_once_ms = flatten_once_ms;
            row.upload_primary_once_ms = upload_primary_once_ms;
            row.upload_secondaries_once_ms = upload_secondaries_once_ms;
            row.upload_tables_once_ms = upload_tables_once_ms;
            row.setup_once_total_ms = setup_once_total_ms;

            row.h2d_ips_avg_ms = h2d_ips_stats.avg;
            row.h2d_ips_min_ms = h2d_ips_stats.min;
            row.h2d_ips_max_ms = h2d_ips_stats.max;
            row.h2d_ips_median_ms = h2d_ips_stats.median;
            row.h2d_ips_stddev_ms = h2d_ips_stats.stddev;

            row.kernel_lookup_avg_ms = kernel_stats.avg;
            row.kernel_lookup_min_ms = kernel_stats.min;
            row.kernel_lookup_max_ms = kernel_stats.max;
            row.kernel_lookup_median_ms = kernel_stats.median;
            row.kernel_lookup_stddev_ms = kernel_stats.stddev;
            row.kernel_lookup_throughput_mpps_avg =
                (kernel_stats.avg > 0.0) ? (static_cast<double>(ips.size()) / kernel_stats.avg) / 1000.0 : 0.0;
            row.kernel_lookup_throughput_mpps_median =
                (kernel_stats.median > 0.0) ? (static_cast<double>(ips.size()) / kernel_stats.median) / 1000.0 : 0.0;

            row.d2h_avg_ms = d2h_stats.avg;
            row.d2h_min_ms = d2h_stats.min;
            row.d2h_max_ms = d2h_stats.max;
            row.d2h_median_ms = d2h_stats.median;
            row.d2h_stddev_ms = d2h_stats.stddev;

            row.transfer_plus_kernel_avg_ms = transfer_plus_kernel_stats.avg;
            row.transfer_plus_kernel_min_ms = transfer_plus_kernel_stats.min;
            row.transfer_plus_kernel_max_ms = transfer_plus_kernel_stats.max;
            row.transfer_plus_kernel_median_ms = transfer_plus_kernel_stats.median;
            row.transfer_plus_kernel_stddev_ms = transfer_plus_kernel_stats.stddev;
            row.transfer_plus_kernel_throughput_mpps_avg =
                (transfer_plus_kernel_stats.avg > 0.0) ? (static_cast<double>(ips.size()) / transfer_plus_kernel_stats.avg) / 1000.0 : 0.0;
            row.transfer_plus_kernel_throughput_mpps_median =
                (transfer_plus_kernel_stats.median > 0.0) ? (static_cast<double>(ips.size()) / transfer_plus_kernel_stats.median) / 1000.0 : 0.0;

            row.total_end_to_end_avg_ms = total_end_to_end_stats.avg;
            row.total_end_to_end_min_ms = total_end_to_end_stats.min;
            row.total_end_to_end_max_ms = total_end_to_end_stats.max;
            row.total_end_to_end_median_ms = total_end_to_end_stats.median;
            row.total_end_to_end_stddev_ms = total_end_to_end_stats.stddev;
            row.total_end_to_end_throughput_mpps_avg =
                (total_end_to_end_stats.avg > 0.0) ? (static_cast<double>(ips.size()) / total_end_to_end_stats.avg) / 1000.0 : 0.0;
            row.total_end_to_end_throughput_mpps_median =
                (total_end_to_end_stats.median > 0.0) ? (static_cast<double>(ips.size()) / total_end_to_end_stats.median) / 1000.0 : 0.0;

            row.checksum = stable_checksum;
            summary_rows.push_back(row);
        }

        // Upratanie CUDA eventov po skončení benchmarku.
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        // Zápis finálnych CSV výstupov.
        write_summary_csv(output_csv, summary_rows);
        write_raw_runs_csv(output_raw_csv, raw_rows);

        std::cout << "\nUlozeny summary CSV: " << output_csv.string() << "\n";
        std::cout << "Ulozeny raw-runs CSV: " << output_raw_csv.string() << "\n";
        std::cout << "Hotovo.\n";
        return 0;
    } catch (const std::exception& e) {
        // Ak nastane chyba, vypíše sa na stderr a program skončí s návratovou hodnotou 1.
        std::cerr << "CHYBA: " << e.what() << "\n";
        return 1;
    }
}
