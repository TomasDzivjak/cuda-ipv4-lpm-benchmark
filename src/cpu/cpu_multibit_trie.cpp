#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/paths.hpp"
#include <omp.h>

namespace fs = std::filesystem;

// Reprezentacia jedneho prefixu z FIB tabulky
// network  = sietova adresa prefixu v uint32 formate
// length   = dlzka prefixu (/0 az /32)
// mask     = maska prefixu v uint32 formate
// next_hop = vystupny next-hop / identifikator trasy
struct Prefix {
    uint32_t network = 0;
    uint8_t length = 0;
    uint32_t mask = 0;
    int32_t next_hop = -1;
};

// Vytvori kanonicku masku pre zadanu dlzku prefixu.
// Napr. pre /24 vrati 255.255.255.0 v uint32 tvare.
static uint32_t prefix_mask(uint8_t length) {
    if (length == 0) return 0u;
    if (length >= 32) return 0xFFFFFFFFu;
    return 0xFFFFFFFFu << (32u - length);
}

// Jednoduche rozdelenie riadku CSV na polia.
// Funkcia podporuje aj pripad, ked sa v riadku nachadzaju uvodzovky.
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

// Oreze medzery a konce riadku z textoveho retazca.
static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// Nacitanie IP adries zo CSV suboru.
// Zo suboru sa pouziva stlpec ip_uint32, teda ciselna reprezentacia IPv4 adresy.
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

// Nacitanie FIB tabulky zo CSV suboru.
// Zo suboru sa nacita network_uint32, prefix_length, mask_uint32 a next_hop.
// Zaroven sa prefix znormalizuje na kanonicky tvar: network = network & mask.
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

// Z nazvu suboru vytiahne prve cislo.
// Pouziva sa na usporiadanie suborov typu fib_table_10000..., generated_ips_200000...
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

// Vypise vsetky CSV subory v priecinku a zoradi ich podla prveho cisla v nazve.
// Vdaka tomu sa FIB a IP datasety spracuvaju v prirodzenom poradi podla velkosti.
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

// Implementacia MultiBit Trie pre IPv4 LPM.
// Pouziva stride = 4 bity, teda v kazdej urovni sa spracuje 4-bitovy usek adresy.
// Pre 32-bitovu IPv4 adresu to znamena maximalne 8 urovni trie.
class MultiBitTrie {
public:
    static constexpr int STRIDE = 4;
    static constexpr int FANOUT = 1 << STRIDE;
    static constexpr int MAX_DEPTH = 32 / STRIDE;

    // Jeden uzol trie.
    // Pre kazdy z 16 moznych slotov si drzi:
    // - next_hop: najlepsi next-hop pre dany slot
    // - best_len: dlzku najlepsieho prefixu pre dany slot
    // - child: index potomka, ak treba ist hlbsie do trie
    struct Node {
        std::array<int32_t, FANOUT> next_hop{};
        std::array<int16_t, FANOUT> best_len{};
        std::array<int32_t, FANOUT> child{};

        Node() {
            next_hop.fill(-1);
            best_len.fill(-1);
            child.fill(-1);
        }
    };

    // Build faza: vytvori trie z celej FIB tabulky.
    // Prefixy sa najprv zoradia od kratsich po dlhsie, aby sa pri zapise
    // prirodzene presadzoval princip longest prefix match.
    void build(const std::vector<Prefix>& fib) {
        nodes_.clear();
        nodes_.emplace_back();
        nodes_.reserve(fib.size() * 2 + 1);

        std::vector<Prefix> prefixes = fib;
        std::stable_sort(prefixes.begin(), prefixes.end(), [](const Prefix& a, const Prefix& b) {
            return a.length < b.length;
        });

        for (const Prefix& p : prefixes) {
            insert_prefix(p);
        }
    }

    // Lookup faza: postupne prechadza trie po 4 bitoch.
    // Popri zostupe si pamata najlepsi zatial najdeny prefix.
    // Ak sa dalej neda pokracovat, vrati posledny platny next-hop.
    int32_t lookup(uint32_t ip) const {
        int32_t node_idx = 0;
        int32_t best_next_hop = -1;
        int16_t best_length = -1;

        for (int depth = 0; depth < MAX_DEPTH && node_idx >= 0; ++depth) {
            const uint32_t slot = extract_stride_bits(ip, depth);
            const Node& node = nodes_[static_cast<std::size_t>(node_idx)];

            if (node.best_len[slot] > best_length) {
                best_length = node.best_len[slot];
                best_next_hop = node.next_hop[slot];
            }

            node_idx = node.child[slot];
        }

        return best_next_hop;
    }

private:
    // Vsetky uzly trie su ulozene v jednom vektore.
    // Namiesto pointerov sa medzi uzlami odkazujeme indexami.
    std::vector<Node> nodes_;

    // Vytvori novy uzol a vrati jeho index.
    int32_t make_node() {
        nodes_.emplace_back();
        return static_cast<int32_t>(nodes_.size() - 1);
    }

    // Z IPv4 adresy vyberie 4-bitovy usek pre zadanu uroven trie.
    // depth = 0 znamena najvyssie 4 bity, depth = 1 dalsie 4 bity atd.
    static uint32_t extract_stride_bits(uint32_t value, int depth) {
        const int shift = 32 - STRIDE * (depth + 1);
        return (value >> shift) & (FANOUT - 1u);
    }

    // Zapise prefix do urceneho rozsahu slotov v danom uzle.
    // Ak je prefix dlhsii ako aktualne ulozeny, prepise best_len aj next_hop.
    void apply_prefix_to_slots(int32_t node_idx,
                               uint32_t start_slot,
                               uint32_t slot_count,
                               int16_t prefix_length,
                               int32_t next_hop) {
        Node& node = nodes_[static_cast<std::size_t>(node_idx)];
        for (uint32_t i = 0; i < slot_count; ++i) {
            const uint32_t slot = start_slot + i;
            if (prefix_length > node.best_len[slot]) {
                node.best_len[slot] = prefix_length;
                node.next_hop[slot] = next_hop;
            }
        }
    }

    // Vlozenie jedneho prefixu do trie.
    // Prefix sa podla svojej dlzky rozdeli na:
    // - plne stride urovne (po 4 bity)
    // - zvysne bity v poslednej ciastocnej urovni
    void insert_prefix(const Prefix& p) {
        const int full_levels = p.length / STRIDE;
        const int remainder = p.length % STRIDE;

        int32_t node_idx = 0;

        // Specialny pripad pre default route /0.
        // Ta plati pre cely prvy uzol, teda pre vsetky sloty root uzla.
        if (p.length == 0) {
            apply_prefix_to_slots(0, 0, FANOUT, 0, p.next_hop);
            return;
        }

        // Prechod cez plne 4-bitove useky prefixu.
        for (int depth = 0; depth < full_levels; ++depth) {
            const uint32_t slot = extract_stride_bits(p.network, depth);

            // Ak prefix konci presne na hranici stride, zapis sa urobi iba do jedneho slotu.
            if (depth == full_levels - 1 && remainder == 0) {
                apply_prefix_to_slots(node_idx, slot, 1, p.length, p.next_hop);
                return;
            }

            // Ak pre dany slot este neexistuje potomok, vytvorime ho.
            if (nodes_[static_cast<std::size_t>(node_idx)].child[slot] < 0) {
                const int32_t child_idx = make_node();
                nodes_[static_cast<std::size_t>(node_idx)].child[slot] = child_idx;
            }
            node_idx = nodes_[static_cast<std::size_t>(node_idx)].child[slot];
        }

        // Ak prefix konci presne na hrane uzla, plati pre vsetky sloty v danom uzle.
        if (remainder == 0) {
            apply_prefix_to_slots(node_idx, 0, FANOUT, p.length, p.next_hop);
            return;
        }

        // Ak prefix nekonci na hranici stride, treba vypocitat, pre ktore sloty v uzle plati.
        // Napr. pri zvysku 2 bity plati pre 4 susedne sloty.
        const uint32_t partial_slot = extract_stride_bits(p.network, full_levels);
        const uint32_t prefix_part = partial_slot >> (STRIDE - remainder);
        const uint32_t slot_count = 1u << (STRIDE - remainder);
        const uint32_t start_slot = prefix_part << (STRIDE - remainder);

        apply_prefix_to_slots(node_idx, start_slot, slot_count, p.length, p.next_hop);
    }
};

// ------------------------------------------------------------
// Paralelny lookup cez OpenMP
// ------------------------------------------------------------
// Kazde vlakno spracovava inu cast vstupneho pola IP adries.
// Vystupny buffer musi byt vytvoreny mimo meraneho useku, aby lookup_time_ms
// nezahrnal dynamicku alokaciu ani resize operaciu.
static void multibit_lookup_omp(
    const MultiBitTrie& trie,
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
        out_next_hop[static_cast<std::size_t>(i)] = trie.lookup(ips[static_cast<std::size_t>(i)]);
    }
}

// ------------------------------------------------------------
// Predpripravenie vystupneho buffera mimo meraneho useku
// ------------------------------------------------------------
// Alokacia a prve dotknutie sa pamati sa vykona pred meranim, rovnako ako
// v metodike CPU DIR24/8. Samotny lookup stale obsahuje zapis next-hop
// vysledkov do vystupneho pola.
static void pre_touch_output_buffer(std::vector<int32_t>& out_next_hop) {
    std::fill(out_next_hop.begin(), out_next_hop.end(), 0);
}

// ------------------------------------------------------------
// Realistickejsi batch forwarding model CPU smerovaca
// ------------------------------------------------------------
// Pracovne buffre sa recykluju. Do merania sa zapocitava priprava vstupnej
// davky kopirovanim do predalokovaneho ingress buffera, paralelny lookup
// a zapis vystupnych next-hop hodnot. Ide o metodicky rovnaku cestu ako
// v CPU DIR24/8 benchmarku.
static void router_forward_batch_cpu_model(
    const MultiBitTrie& trie,
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

    std::copy(source_ips.begin(), source_ips.end(), ingress_buffer.begin());
    multibit_lookup_omp(trie, ingress_buffer, out_next_hop, num_threads);
}

// Jednoduchy checksum vysledkov lookupu.
// Pouziva sa na overenie, ze vsetky behy vratili rovnake a korektne vysledky.
static uint64_t checksum_vector(const std::vector<int32_t>& v) {
    uint64_t sum = 0;
    for (int32_t x : v) {
        sum += static_cast<uint64_t>(static_cast<uint32_t>(x));
    }
    return sum;
}


// Typ prevadzky / query datasetu.
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

// Nacitanie IP adries z BIN suboru.
// ENTERPRISE a HOTSPOT datasety su binarne subory obsahujuce iba
// za sebou ulozene hodnoty uint32_t, teda IPv4 adresy v 32-bitovom tvare.
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

// Konfiguracia jedneho testovaneho datasetu.
struct DatasetConfig {
    int id = 0;
    std::string label;
    std::string fib_filename;
    std::string ip_filename;
};

// ------------------------------------------------------------
// Struktura jedneho vysledneho riadku pre CSV
// ------------------------------------------------------------
// Kazdy riadok obsahuje agregovane hodnoty z measured_runs_per_config behov.
// Schema je rovnaka ako v CPU DIR24/8 benchmarku, aby boli vystupy priamo
// porovnatelne v rovnakych grafoch a tabulkach.
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

    // Inicializacny / rekonfiguracny naklad: zostavenie trie z FIB tabulky.
    // Nezahrna sa do hlavnej batch priepustnosti.
    double build_time_ms = 0.0;
    double build_median_time_ms = 0.0;

    // Steady-state lookup: aktivna trie je uz zostavena a vystupny buffer je
    // predalokovany mimo meraneho useku.
    double lookup_time_ms = 0.0;
    double lookup_median_time_ms = 0.0;
    double lookup_throughput_mpps = 0.0;
    double lookup_median_throughput_mpps = 0.0;

    // CPU router-batch model: kopia vstupnej davky do ingress buffera,
    // paralelny lookup a zapis vysledkov.
    double router_batch_time_ms = 0.0;
    double router_batch_median_time_ms = 0.0;
    double router_batch_throughput_mpps = 0.0;
    double router_batch_median_throughput_mpps = 0.0;

    // Hlavna throughput metrika je nastavena na router-batch throughput,
    // rovnako ako v CPU DIR24/8 skripte.
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

// Pomocna struktura na vypocet priemeru, medianu, minima a maxima.
struct AggregateStats {
    double avg_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

// Zo zoznamu casov vypocita priemer, median, minimum a maximum.
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

// Stabilny monotonne rastuci casovac pouzity rovnako ako v CPU DIR24/8.
static double duration_ms(const std::chrono::steady_clock::time_point& a,
                          const std::chrono::steady_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Zapis agregovanych vysledkov do CSV suboru.
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

// Referencne checksumy pre synteticke kombinacie.
// Pre ENTERPRISE / HOTSPOT sa referencny checksum typicky nepouziva,
// pretoze ide o nove BIN query datasety.
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
            {1, "enterprise_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "queries_prefix_200000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {2, "enterprise_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "queries_prefix_400000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {3, "enterprise_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "queries_prefix_800000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {4, "enterprise_fib50k_ips1m",  "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {5, "enterprise_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {6, "enterprise_fib50k_ips2m",  "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {7, "enterprise_fib50k_ips3m",  "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_enterprise_b50_d35_r15_p24128_p1616.bin"}
        };
    }

    return {
        {1, "hotspot_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "queries_prefix_200000_hotspot_hp8_r95.bin"},
        {2, "hotspot_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "queries_prefix_400000_hotspot_hp8_r95.bin"},
        {3, "hotspot_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "queries_prefix_800000_hotspot_hp8_r95.bin"},
        {4, "hotspot_fib50k_ips1m",  "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_hotspot_hp8_r95.bin"},
        {5, "hotspot_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_hotspot_hp8_r95.bin"},
        {6, "hotspot_fib50k_ips2m",  "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_hotspot_hp8_r95.bin"},
        {7, "hotspot_fib50k_ips3m",  "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_hotspot_hp8_r95.bin"}
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
        // Vstupne a vystupne priecinky.
        const fs::path fib_dir = lpm_paths::fib_dir();
        const fs::path ips_dir = lpm_paths::synthetic_ip_dir();
        const fs::path enterprise_dir = lpm_paths::enterprise_query_dir();
        const fs::path hotspot_dir = lpm_paths::hotspot_query_dir();
        const fs::path output_dir = lpm_paths::cpu_multibit_trie_results_dir();

        // ========================================================
        // TRAFFIC TYPE
        // SYNTHETIC  = povodne CSV generated_ips_*
        // ENTERPRISE = BIN query datasety z Q_BIN_enterprise
        // HOTSPOT    = BIN query datasety z Q_BIN_hotspot
        // ========================================================
        const TrafficType traffic_type = TrafficType::SYNTHETIC;

        // Nastavenie benchmarku.
        const int threads = 12;
        const int measured_runs_per_config = 10;
        const int warmup_runs_per_config = 5;
        const bool enable_reference_checksum_validation = true;

        // Metodika zhodna s CPU DIR24/8 benchmarkom:
        // - zostavenie trie sa meria samostatne,
        // - aktivna trie pre forwarding sa postavi iba raz,
        // - pracovne buffre sa predalokuju mimo meraneho useku,
        // - hlavna throughput_mpps metrika pouziva router_batch_time_ms.
        const bool preallocate_reusable_router_buffers = true;
        const bool use_router_batch_metric_as_main_throughput = true;

        // Vyber dataset konfiguracii, ktore sa maju benchmarkovat.
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
        // Pre full benchmark real traffic pouzi {1, 2, 3, 4, 5, 6, 7}.
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

        std::cout << "CPU MultiBit Trie benchmark\n";
        std::cout << "Traffic type: " << traffic_name << "\n";
        std::cout << "Threads: " << threads << "\n";
        std::cout << "Measured runs/config: " << measured_runs_per_config << "\n";
        std::cout << "Warmup runs/config: " << warmup_runs_per_config << "\n";
        std::cout << "Measurement model: steady-state trie + preallocated buffers\n";
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
            // 1) Samostatne meranie zostavenia MultiBit Trie
            // ----------------------------------------------------
            // Toto reprezentuje inicializaciu alebo rekonfiguraciu FIB struktury.
            // Nie je to per-batch forwarding naklad a preto sa nepouziva na
            // vypocet hlavnej throughput_mpps metriky.
            const int total_runs = warmup_runs_per_config + measured_runs_per_config;
            for (int run_id = 1; run_id <= total_runs; ++run_id) {
                auto b0 = std::chrono::steady_clock::now();
                MultiBitTrie build_probe;
                build_probe.build(fib);
                auto b1 = std::chrono::steady_clock::now();

                if (run_id > warmup_runs_per_config) {
                    build_times.push_back(duration_ms(b0, b1));
                }
            }

            // Aktivna trie struktura: postavi sa raz a nasledne sa pouziva pri
            // opakovanych lookupoch, teda v stabilnom forwarding stave.
            MultiBitTrie trie;
            trie.build(fib);

            // Recyklovane pracovne buffre: ich alokacia a prve dotknutie sa
            // pamati nie su sucastou lookup ani router-batch casu.
            std::vector<int32_t> out(ips.size());
            std::vector<uint32_t> ingress_buffer(ips.size());
            if (preallocate_reusable_router_buffers) {
                pre_touch_output_buffer(out);
                std::fill(ingress_buffer.begin(), ingress_buffer.end(), 0u);
            }

            // ----------------------------------------------------
            // 2) Steady-state lookup meranie
            // ----------------------------------------------------
            // Trie aj vstup su uz v pamati, vystupny buffer je predalokovany.
            // Metrika zostava v CSV ako izolovanejsie porovnanie lookup jadra.
            for (int run_id = 1; run_id <= total_runs; ++run_id) {
                auto l0 = std::chrono::steady_clock::now();
                multibit_lookup_omp(trie, ips, out, threads);
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
            // 3) CPU router-batch model zhodny s DIR24/8
            // ----------------------------------------------------
            // Do merania sa zapocita priprava vstupneho batchu do
            // predalokovaneho ingress buffera, paralelny lookup a zapis vystupu.
            // Toto je hlavna prakticka CPU throughput metrika.
            for (int run_id = 1; run_id <= total_runs; ++run_id) {
                auto r0 = std::chrono::steady_clock::now();
                router_forward_batch_cpu_model(trie, ips, ingress_buffer, out, threads);
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

            // Overenie korektnosti voci referencnym checksumom z DIR24/8 implementacie.
            // Plati iba pre synteticke datasety, ktore su v mape referencii.
            if (enable_reference_checksum_validation) {
                auto it = reference_checksums.find(combo_key);
                if (it != reference_checksums.end()) {
                    const uint64_t expected = it->second;
                    if (stable_checksum != expected) {
                        throw std::runtime_error(
                            "MultiBit Trie checksum nesedi s referenciou pre kombinaciu " +
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
                "cpu_multibit_trie_lpm_" + traffic_name +
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
