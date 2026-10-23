// hybrid_dir24_8_lpm_scheduler.cu
// Hybridny benchmark DIR24/8: CPU OpenMP + CUDA GPU + jednoduchy scheduler/dispatcher.
//
//
// Windows/NVCC priklad kompilacie:
// nvcc -O3 -std=c++17 -Xcompiler "/openmp" hybrid_dir24_8_lpm_scheduler.cu -o hybrid_dir24_8_lpm_scheduler.exe
//
// Poznamka: konfiguracia je v ma1n() v sekcii USER CONFIG.

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
#include <unordered_set>
#include <vector>

#include "common/paths.hpp"

#include <omp.h>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

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

using Clock = std::chrono::steady_clock;

static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}

struct Prefix {
    uint32_t network = 0;
    uint8_t length = 0;
    uint32_t mask = 0;
    int32_t next_hop = -1;
};

static uint32_t prefix_mask(uint8_t length) {
    if (length == 0) return 0u;
    if (length >= 32) return 0xFFFFFFFFu;
    return 0xFFFFFFFFu << (32u - length);
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    current.reserve(line.size());
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) { fields.push_back(current); current.clear(); }
        else current.push_back(c);
    }
    fields.push_back(current);
    return fields;
}

static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static std::vector<uint32_t> load_ips_csv(const fs::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) throw std::runtime_error("Nepodarilo sa otvorit IP CSV: " + file_path.string());
    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error("Prazdny IP CSV: " + file_path.string());
    std::vector<uint32_t> ips;
    ips.reserve(1024);
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto fields = split_csv_line(line);
        if (fields.size() < 2) throw std::runtime_error("Neplatny riadok v IP CSV: " + line);
        ips.push_back(static_cast<uint32_t>(std::stoul(trim(fields[1]))));
    }
    return ips;
}

static std::vector<uint32_t> load_ips_bin(const fs::path& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Nepodarilo sa otvorit BIN dataset: " + file_path.string());
    file.seekg(0, std::ios::end);
    const std::size_t file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    if ((file_size % sizeof(uint32_t)) != 0u) throw std::runtime_error("BIN dataset nema validnu velkost: " + file_path.string());
    std::vector<uint32_t> ips(file_size / sizeof(uint32_t));
    if (!ips.empty()) {
        file.read(reinterpret_cast<char*>(ips.data()), static_cast<std::streamsize>(file_size));
        if (!file) throw std::runtime_error("Chyba pri citani BIN datasetu: " + file_path.string());
    }
    return ips;
}

static std::vector<Prefix> load_fib_csv(const fs::path& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) throw std::runtime_error("Nepodarilo sa otvorit FIB CSV: " + file_path.string());
    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error("Prazdny FIB CSV: " + file_path.string());
    std::vector<Prefix> fib;
    fib.reserve(1024);
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto fields = split_csv_line(line);
        if (fields.size() < 7) throw std::runtime_error("Neplatny riadok vo FIB CSV: " + line);
        Prefix p;
        p.network = static_cast<uint32_t>(std::stoul(trim(fields[1])));
        p.length = static_cast<uint8_t>(std::stoul(trim(fields[3])));
        p.mask = static_cast<uint32_t>(std::stoul(trim(fields[4])));
        p.next_hop = static_cast<int32_t>(std::stol(trim(fields[6])));
        p.mask = prefix_mask(p.length);
        p.network &= p.mask;
        fib.push_back(p);
    }
    return fib;
}

static std::size_t extract_first_number(const std::string& s) {
    std::string digits; bool started = false;
    for (char c : s) {
        if (c >= '0' && c <= '9') { digits.push_back(c); started = true; }
        else if (started) break;
    }
    return digits.empty() ? 0ull : static_cast<std::size_t>(std::stoull(digits));
}

static std::vector<fs::path> list_files_sorted(const fs::path& dir, const std::string& ext) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("Priecinok neexistuje alebo nie je priecinok: " + dir.string());
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ext) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        const auto na = extract_first_number(a.filename().string());
        const auto nb = extract_first_number(b.filename().string());
        if (na != nb) return na < nb;
        return a.filename().string() < b.filename().string();
    });
    return files;
}

class Dir24_8Host {
public:
    struct SecondaryTable {
        std::array<int32_t, 256> next_hop{};
        std::array<int16_t, 256> best_len{};
        SecondaryTable() { next_hop.fill(-1); best_len.fill(-1); }
    };

    struct PrimaryEntry {
        int32_t next_hop = -1;
        int16_t best_len = -1;
        uint32_t secondary_index = 0;
        uint8_t is_secondary = 0;
    };

    void build(const std::vector<Prefix>& fib) {
        primary_.assign(PRIMARY_SIZE, PrimaryEntry{});
        secondaries_.clear();
        std::vector<Prefix> prefixes = fib;
        std::stable_sort(prefixes.begin(), prefixes.end(), [](const Prefix& a, const Prefix& b) {
            return a.length < b.length;
        });
        for (const Prefix& p : prefixes) {
            if (p.length <= 24) fill_primary_range(p);
            else fill_secondary_range(p);
        }
    }

    int32_t lookup(uint32_t ip) const {
        const uint32_t idx24 = ip >> 8;
        const PrimaryEntry& e = primary_[idx24];
        if (!e.is_secondary) return e.next_hop;
        const uint8_t low8 = static_cast<uint8_t>(ip & 0xFFu);
        return secondaries_[e.secondary_index].next_hop[low8];
    }

    const std::vector<PrimaryEntry>& primary() const { return primary_; }
    const std::vector<SecondaryTable>& secondaries() const { return secondaries_; }

private:
    static constexpr uint32_t PRIMARY_SIZE = 1u << 24;
    std::vector<PrimaryEntry> primary_;
    std::vector<SecondaryTable> secondaries_;

    void fill_primary_range(const Prefix& p) {
        const uint32_t start = p.network >> 8;
        const uint32_t count = (p.length == 24) ? 1u : (1u << (24u - static_cast<uint32_t>(p.length)));
        for (uint32_t i = 0; i < count; ++i) {
            PrimaryEntry& entry = primary_[start + i];
            if (!entry.is_secondary) {
                if (p.length > entry.best_len) { entry.next_hop = p.next_hop; entry.best_len = p.length; }
            } else {
                if (p.length > entry.best_len) { entry.next_hop = p.next_hop; entry.best_len = p.length; }
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
            if (p.length > sec.best_len[pos]) { sec.next_hop[pos] = p.next_hop; sec.best_len[pos] = p.length; }
        }
    }
};

static void dir24_8_lookup_cpu_omp_no_resize(const Dir24_8Host& dir,
                                             const std::vector<uint32_t>& ips,
                                             std::vector<int32_t>& out,
                                             int num_threads) {
    if (out.size() != ips.size()) {
        throw std::runtime_error("CPU output buffer nema rovnaku velkost ako vstupny batch.");
    }

    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(ips.size()); ++i) {
        out[static_cast<std::size_t>(i)] = dir.lookup(ips[static_cast<std::size_t>(i)]);
    }
}

static void pre_touch_output_buffer(std::vector<int32_t>& out) {
    std::fill(out.begin(), out.end(), 0);
}

static void pre_touch_ingress_buffer(std::vector<uint32_t>& ingress) {
    std::fill(ingress.begin(), ingress.end(), 0u);
}

static void router_forward_batch_cpu_model(const Dir24_8Host& dir,
                                           const std::vector<uint32_t>& source_ips,
                                           std::vector<uint32_t>& ingress_buffer,
                                           std::vector<int32_t>& out,
                                           int num_threads) {
    if (ingress_buffer.size() != source_ips.size()) {
        throw std::runtime_error("CPU ingress buffer nema rovnaku velkost ako vstupny batch.");
    }
    if (out.size() != source_ips.size()) {
        throw std::runtime_error("CPU output buffer nema rovnaku velkost ako vstupny batch.");
    }

    // Router model: batch je skopirovany do predalokovaneho ingress buffera,
    // nasledne sa nad tymto bufferom vykona paralelny DIR24/8 lookup.
    // Nezahŕňa dynamicku alokaciu ani build FIB pred kazdou davkou.
    std::copy(source_ips.begin(), source_ips.end(), ingress_buffer.begin());
    dir24_8_lookup_cpu_omp_no_resize(dir, ingress_buffer, out, num_threads);
}

struct DevicePrimaryEntry {
    int32_t next_hop;
    int32_t secondary_index;
    uint8_t is_secondary;
    uint8_t pad[3];
};
static_assert(sizeof(DevicePrimaryEntry) == 12, "DevicePrimaryEntry must be 12 bytes");

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

static std::vector<int32_t> flatten_secondaries_next_hop(const std::vector<Dir24_8Host::SecondaryTable>& src) {
    std::vector<int32_t> out(src.size() * 256ull, -1);
    for (std::size_t i = 0; i < src.size(); ++i) {
        for (std::size_t j = 0; j < 256; ++j) out[i * 256ull + j] = src[i].next_hop[j];
    }
    return out;
}

__global__
void dir24_8_lookup_kernel(const DevicePrimaryEntry* __restrict__ primary,
                           const int32_t* __restrict__ secondaries_next_hop,
                           const uint32_t* __restrict__ ips,
                           int32_t* __restrict__ out,
                           std::size_t ip_count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= ip_count) return;
    const uint32_t ip = ips[idx];
    const DevicePrimaryEntry e = primary[ip >> 8];
    int32_t result = e.next_hop;
    if (e.is_secondary) result = secondaries_next_hop[static_cast<std::size_t>(e.secondary_index) * 256u + static_cast<uint8_t>(ip & 0xFFu)];
    out[idx] = result;
}

static uint64_t checksum_vector(const int32_t* data, std::size_t size) {
    uint64_t sum = 0;
    for (std::size_t i = 0; i < size; ++i) sum += static_cast<uint64_t>(static_cast<uint32_t>(data[i]));
    return sum;
}

struct Stats {
    double avg = 0, min = 0, max = 0, median = 0, stddev = 0;
};

static Stats aggregate(std::vector<double> values) {
    Stats s{};
    if (values.empty()) return s;
    std::sort(values.begin(), values.end());
    s.min = values.front();
    s.max = values.back();
    s.avg = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    const std::size_t n = values.size();
    s.median = (n % 2u) ? values[n / 2u] : 0.5 * (values[n / 2u - 1u] + values[n / 2u]);
    double var = 0.0;
    for (double v : values) var += (v - s.avg) * (v - s.avg);
    s.stddev = std::sqrt(var / static_cast<double>(values.size()));
    return s;
}

struct LocalityInfo {
    std::size_t unique24 = 0;
    double avg_ips_per_unique24 = 0.0;
    double unique24_ratio = 0.0;
};

static LocalityInfo analyze_prefix_locality_24(const std::vector<uint32_t>& ips,
                                               double sample_fraction = 1.0,
                                               std::size_t sample_max_count = 0) {
    const double clamped_fraction = std::max(0.0, std::min(1.0, sample_fraction));
    std::size_t sample_count = ips.size();
    if (clamped_fraction > 0.0 && clamped_fraction < 1.0) {
        sample_count = static_cast<std::size_t>(std::ceil(static_cast<double>(ips.size()) * clamped_fraction));
        sample_count = std::max<std::size_t>(1, sample_count);
    }
    if (sample_max_count > 0) {
        sample_count = std::min(sample_count, sample_max_count);
    }
    sample_count = std::min(sample_count, ips.size());

    std::unordered_set<uint32_t> prefixes;
    prefixes.reserve(sample_count);
    for (std::size_t i = 0; i < sample_count; ++i) prefixes.insert(ips[i] >> 8);

    LocalityInfo info{};
    info.unique24 = prefixes.size();
    if (sample_count > 0 && info.unique24 > 0) {
        info.avg_ips_per_unique24 = static_cast<double>(sample_count) / static_cast<double>(info.unique24);
        info.unique24_ratio = static_cast<double>(info.unique24) / static_cast<double>(sample_count);
    }
    return info;
}

enum class TrafficType { SYNTHETIC, ENTERPRISE, HOTSPOT };
enum class Component { CPU, GPU };

static std::string traffic_to_string(TrafficType t) {
    if (t == TrafficType::SYNTHETIC) return "synthetic";
    if (t == TrafficType::ENTERPRISE) return "enterprise";
    return "hotspot";
}

static std::string component_to_string(Component c) {
    return c == Component::CPU ? "CPU" : "GPU";
}

struct DatasetConfig {
    int id = 0;
    std::string label;
    std::string fib_name;
    std::string ip_name;
};

static std::string make_output_suffix(const std::vector<DatasetConfig>& selected, int cpu_threads) {
    if (selected.empty()) throw std::runtime_error("Nie je vybrana ziadna dataset konfiguracia.");

    std::string traffic_prefix = "selected";
    const auto pos = selected.front().label.find("_fib");
    if (pos != std::string::npos) traffic_prefix = selected.front().label.substr(0, pos);

    std::ostringstream oss;
    oss << traffic_prefix << "_ids";
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (i > 0) oss << "_";
        oss << selected[i].id;
    }
    oss << "_threads" << cpu_threads;
    return oss.str();
}

static Component decide_component(const LocalityInfo& loc,
                                  std::size_t batch_size,
                                  std::size_t low_locality_cpu_max_batch_size,
                                  std::size_t enterprise_gpu_min_batch_size,
                                  double low_locality_gpu_threshold_avg_per_unique24,
                                  double high_locality_cpu_threshold_avg_per_unique24) {
    // Hybridny smerovacovy model bez znalosti nazvu prevadzky:
    // 1) Nizka prefixova lokalita => rozptylena/synthetic-like prevadzka:
    //      - velmi maly batch spracuje CPU, pretoze transfery na GPU sa neamortizuju,
    //      - vacsi batch spracuje GPU.
    // 2) Stredna prefixova lokalita => enterprise-like prevadzka:
    //      - mensi batch spracuje CPU,
    //      - vacsi batch spracuje GPU.
    // 3) Vysoka prefixova lokalita => hotspot-like prevadzka => CPU.
    //
    // Scheduler nepozna nazov prevadzky; rozhoduje iba podla vlastnosti batchu.
    const double locality = loc.avg_ips_per_unique24;

    if (locality < low_locality_gpu_threshold_avg_per_unique24) {
        if (batch_size <= low_locality_cpu_max_batch_size) {
            return Component::CPU;
        }
        return Component::GPU;
    }
    if (locality >= high_locality_cpu_threshold_avg_per_unique24) {
        return Component::CPU;
    }
    if (batch_size < enterprise_gpu_min_batch_size) {
        return Component::CPU;
    }
    return Component::GPU;
}

struct RawRow {
    std::string dataset_label;
    int dataset_id = 0;
    int run_id = 0;
    std::string phase;
    std::string selected_component;
    std::size_t fib_size = 0;
    std::size_t ip_count = 0;
    std::size_t unique24 = 0;
    double avg_ips_per_unique24 = 0.0;
    double unique24_ratio = 0.0;
    double decision_ms = 0.0;
    double processing_without_decision_ms = 0.0;
    double gpu_kernel_only_ms = 0.0;
    double processing_with_decision_ms = 0.0;
    double throughput_without_decision_mpps = 0.0;
    double throughput_with_decision_mpps = 0.0;
    double cpu_baseline_processing_ms = 0.0;
    double cpu_baseline_throughput_mpps = 0.0;
    double h2d_ms = 0.0;
    double kernel_avg_ms = 0.0;
    double d2h_ms = 0.0;
    uint64_t checksum = 0;
};

struct SummaryRow {
    std::string dataset_label;
    int dataset_id = 0;
    std::string selected_component;
    std::size_t fib_size = 0;
    std::size_t ip_count = 0;
    std::size_t unique24 = 0;
    double avg_ips_per_unique24 = 0.0;
    double unique24_ratio = 0.0;
    double decision_avg_ms = 0.0;
    double decision_median_ms = 0.0;
    double processing_avg_ms = 0.0;
    double processing_median_ms = 0.0;
    double gpu_kernel_only_avg_ms = 0.0;
    double gpu_kernel_only_median_ms = 0.0;
    double with_decision_avg_ms = 0.0;
    double with_decision_median_ms = 0.0;
    double throughput_without_decision_avg_mpps = 0.0;
    double throughput_without_decision_median_mpps = 0.0;
    double throughput_with_decision_avg_mpps = 0.0;
    double throughput_with_decision_median_mpps = 0.0;
    double cpu_baseline_avg_ms = 0.0;
    double cpu_baseline_median_ms = 0.0;
    double cpu_baseline_throughput_avg_mpps = 0.0;
    double cpu_baseline_throughput_median_mpps = 0.0;
    double h2d_avg_ms = 0.0;
    double kernel_avg_ms = 0.0;
    double d2h_avg_ms = 0.0;
    uint64_t checksum = 0;
};


struct CpuBaselineSummaryRow {
    int run_id = 0;
    std::string traffic_type;
    int dataset_id = 0;
    std::string dataset_label;
    int threads = 12;
    std::string fib_file;
    std::string ip_file;
    std::size_t fib_size = 0;
    std::size_t ip_count = 0;

    // CPU-only baseline v tom istom hybridnom executable nemeria build ani izolovany lookup.
    // Hlavna platna metrika je router_batch_*.
    double build_time_ms = 0.0;
    double build_median_time_ms = 0.0;
    double lookup_time_ms = 0.0;
    double lookup_median_time_ms = 0.0;
    double lookup_throughput_mpps = 0.0;
    double lookup_median_throughput_mpps = 0.0;
    double router_batch_time_ms = 0.0;
    double router_batch_median_time_ms = 0.0;
    double router_batch_throughput_mpps = 0.0;
    double router_batch_median_throughput_mpps = 0.0;
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

static void write_raw_csv(const fs::path& p, const std::vector<RawRow>& rows) {
    std::ofstream f(p);
    if (!f.is_open()) throw std::runtime_error("Nepodarilo sa zapisat raw CSV: " + p.string());
    f << "dataset_label,dataset_id,run_id,phase,selected_component,fib_size,ip_count,unique24,avg_ips_per_unique24,unique24_ratio,"
      << "decision_ms,processing_without_decision_ms,gpu_kernel_only_ms,processing_with_decision_ms,throughput_without_decision_mpps,throughput_with_decision_mpps,"
      << "cpu_baseline_processing_ms,cpu_baseline_throughput_mpps,"
      << "h2d_ms,kernel_avg_ms,d2h_ms,checksum\n";
    f << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        f << r.dataset_label << ',' << r.dataset_id << ',' << r.run_id << ',' << r.phase << ',' << r.selected_component << ','
          << r.fib_size << ',' << r.ip_count << ',' << r.unique24 << ',' << r.avg_ips_per_unique24 << ',' << r.unique24_ratio << ','
          << r.decision_ms << ',' << r.processing_without_decision_ms << ',' << r.gpu_kernel_only_ms << ',' << r.processing_with_decision_ms << ','
          << r.throughput_without_decision_mpps << ',' << r.throughput_with_decision_mpps << ','
          << r.cpu_baseline_processing_ms << ',' << r.cpu_baseline_throughput_mpps << ','
          << r.h2d_ms << ',' << r.kernel_avg_ms << ',' << r.d2h_ms << ',' << r.checksum << '\n';
    }
}

static void write_summary_csv(const fs::path& p, const std::vector<SummaryRow>& rows) {
    std::ofstream f(p);
    if (!f.is_open()) throw std::runtime_error("Nepodarilo sa zapisat summary CSV: " + p.string());
    f << "dataset_label,dataset_id,selected_component,fib_size,ip_count,unique24,avg_ips_per_unique24,unique24_ratio,"
      << "decision_avg_ms,decision_median_ms,processing_avg_ms,processing_median_ms,gpu_kernel_only_avg_ms,gpu_kernel_only_median_ms,with_decision_avg_ms,with_decision_median_ms,"
      << "throughput_without_decision_avg_mpps,throughput_without_decision_median_mpps,throughput_with_decision_avg_mpps,throughput_with_decision_median_mpps,"
      << "cpu_baseline_avg_ms,cpu_baseline_median_ms,cpu_baseline_throughput_avg_mpps,cpu_baseline_throughput_median_mpps,"
      << "h2d_avg_ms,kernel_avg_ms,d2h_avg_ms,checksum\n";
    f << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        f << r.dataset_label << ',' << r.dataset_id << ',' << r.selected_component << ',' << r.fib_size << ',' << r.ip_count << ','
          << r.unique24 << ',' << r.avg_ips_per_unique24 << ',' << r.unique24_ratio << ','
          << r.decision_avg_ms << ',' << r.decision_median_ms << ',' << r.processing_avg_ms << ',' << r.processing_median_ms << ','
          << r.gpu_kernel_only_avg_ms << ',' << r.gpu_kernel_only_median_ms << ','
          << r.with_decision_avg_ms << ',' << r.with_decision_median_ms << ','
          << r.throughput_without_decision_avg_mpps << ',' << r.throughput_without_decision_median_mpps << ','
          << r.throughput_with_decision_avg_mpps << ',' << r.throughput_with_decision_median_mpps << ','
          << r.cpu_baseline_avg_ms << ',' << r.cpu_baseline_median_ms << ','
          << r.cpu_baseline_throughput_avg_mpps << ',' << r.cpu_baseline_throughput_median_mpps << ','
          << r.h2d_avg_ms << ',' << r.kernel_avg_ms << ',' << r.d2h_avg_ms << ',' << r.checksum << '\n';
    }
}


static void write_cpu_baseline_summary_csv(const fs::path& p, const std::vector<CpuBaselineSummaryRow>& rows) {
    std::ofstream out(p);
    if (!out.is_open()) throw std::runtime_error("Nepodarilo sa zapisat CPU baseline summary CSV: " + p.string());

    // Format kompatibilny so samostatnym CPU router-model benchmarkom.
    // V tomto subore su platne najma router_batch_* a throughput_mpps/median_throughput_mpps.
    // build_* a lookup_* su diagnosticky nastavene na 0.0, pretoze tento baseline meria iba CPU router-batch vetvu.
    out << "run_id,traffic_type,dataset_id,dataset_label,threads,fib_file,ip_file,fib_size,ip_count,"
        << "build_time_ms,build_median_time_ms,lookup_time_ms,lookup_median_time_ms,lookup_throughput_mpps,lookup_median_throughput_mpps,"
        << "router_batch_time_ms,router_batch_median_time_ms,router_batch_throughput_mpps,router_batch_median_throughput_mpps,"
        << "total_time_ms,total_median_time_ms,throughput_mpps,median_throughput_mpps,avg_time_ms,median_time_ms,min_time_ms,max_time_ms,"
        << "speedup_vs_linear,speedup_vs_single_thread,checksum\n";
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

int main() {
    try {
        // ============================================================
        // USER CONFIG
        // ============================================================
        const fs::path fib_dir = lpm_paths::fib_dir();
        const fs::path ips_dir = lpm_paths::synthetic_ip_dir();
        const fs::path enterprise_dir = lpm_paths::enterprise_query_dir();
        const fs::path hotspot_dir = lpm_paths::hotspot_query_dir();
        const fs::path output_dir = lpm_paths::hybrid_dir24_8_results_dir();
        
        // SYNTHETIC  |  ENTERPRISE  |  HOTSPOT
        const TrafficType traffic_type = TrafficType::ENTERPRISE;
        const std::vector<int> selected_dataset_ids = {1, 2, 3, 4, 5, 6, 7};

        const int cpu_threads = 12;
        const int warmup_runs_per_config = 5;
        const int measured_runs_per_config = 10;
        const int pre_measure_sleep_ms = 0;
        const int block_size = 256;
        const int kernel_repeats_per_measurement = 20; // rovnaky princip ako v GPU benchmarku: kernel cas = priemer z opakovani

        // Scheduler heuristika bez znalosti nazvu prevadzky:
        // - nizka /24 prefixova lokalita: GPU nad 200k
        // - stredna /24 prefixova lokalita: CPU pre mensie batche, GPU pre vacsie batche
        // - vysoka /24 prefixova lokalita: CPU bez ohladu na velkost batchu
        //
        // Metrika lokality = avg_ips_per_unique24 v odobratej vzorke.
        // Y = low_locality_gpu_threshold_avg_per_unique24
        // Z = high_locality_cpu_threshold_avg_per_unique24
        const double low_locality_gpu_threshold_avg_per_unique24 = 2.0;
        const double high_locality_cpu_threshold_avg_per_unique24 = 15.0;

        // Pri nizkej lokalite (synthetic-like) vysiel 200k batch rychlejsie na CPU.
        const std::size_t low_locality_cpu_max_batch_size = 200000;

        // Pri strednej lokalite (enterprise-like): do 800k vratane CPU, od 1M GPU.
        const std::size_t enterprise_gpu_min_batch_size = 800001;
        const double locality_sample_fraction = 0.25;
        const std::size_t locality_sample_max_count = 32768; // scheduler analyzuje najviac prvych 32k IP adries bez ohladu na velkost datasetu
        const bool validate_cpu_gpu_checksum_once = true;

        fs::create_directories(output_dir);
        const std::string traffic_name = traffic_to_string(traffic_type);
        const fs::path log_path = output_dir / ("hybrid_dir24_8_lpm_" + traffic_name + "_log.txt");
        std::ofstream log(log_path);
        if (!log.is_open()) throw std::runtime_error("Nepodarilo sa vytvorit log subor: " + log_path.string());

        CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        auto log_both = [&](const std::string& s) {
            std::cout << s;
            log << s;
            log.flush();
        };

        log_both("Hybrid DIR24/8 benchmark\n");
        log_both("GPU: " + std::string(prop.name) + "\n");
        log_both("Traffic type: " + traffic_name + "\n");
        log_both("CPU threads: " + std::to_string(cpu_threads) + "\n");
        log_both("Warmup runs/config: " + std::to_string(warmup_runs_per_config) + "\n");
        log_both("Measured runs/config: " + std::to_string(measured_runs_per_config) + "\n");
        log_both("Low locality GPU threshold Y avg_ips_per_unique24: " + std::to_string(low_locality_gpu_threshold_avg_per_unique24) + "\n");
        log_both("High locality CPU threshold Z avg_ips_per_unique24: " + std::to_string(high_locality_cpu_threshold_avg_per_unique24) + "\n");
        log_both("Low-locality CPU max batch size: " + std::to_string(low_locality_cpu_max_batch_size) + "\n");
        log_both("Enterprise-like GPU min batch size: " + std::to_string(enterprise_gpu_min_batch_size) + "\n");
        log_both("Locality sample fraction: " + std::to_string(locality_sample_fraction) + "\n");
        log_both("Locality sample max count: " + std::to_string(locality_sample_max_count) + "\n\n");

        const auto fib_files = list_files_sorted(fib_dir, ".csv");
        std::vector<fs::path> ip_files;
        if (traffic_type == TrafficType::SYNTHETIC) ip_files = list_files_sorted(ips_dir, ".csv");
        else if (traffic_type == TrafficType::ENTERPRISE) ip_files = list_files_sorted(enterprise_dir, ".bin");
        else ip_files = list_files_sorted(hotspot_dir, ".bin");

        std::unordered_map<std::string, fs::path> fib_by_name, ip_by_name;
        for (const auto& f : fib_files) fib_by_name.emplace(f.filename().string(), f);
        for (const auto& f : ip_files) ip_by_name.emplace(f.filename().string(), f);

        std::vector<DatasetConfig> all;
        if (traffic_type == TrafficType::SYNTHETIC) {
            all = {
                {1, "synthetic_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "generated_ips_200000_seed_12345.csv"},
                {2, "synthetic_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "generated_ips_400000_seed_12345.csv"},
                {3, "synthetic_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "generated_ips_800000_seed_12345.csv"},
                {4, "synthetic_fib50k_ips1m", "fib_table_50000_seed_12345.csv", "generated_ips_1000000_seed_12345.csv"},
                {5, "synthetic_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "generated_ips_1500000_seed_12345.csv"},
                {6, "synthetic_fib50k_ips2m", "fib_table_50000_seed_12345.csv", "generated_ips_2000000_seed_12345.csv"},
                {7, "synthetic_fib50k_ips3m", "fib_table_50000_seed_12345.csv", "generated_ips_3000000_seed_12345.csv"}
            };
        } else if (traffic_type == TrafficType::ENTERPRISE) {
            all = {
                {1, "enterprise_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "queries_prefix_200000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {2, "enterprise_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "queries_prefix_400000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {3, "enterprise_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "queries_prefix_800000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {4, "enterprise_fib50k_ips1m", "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {5, "enterprise_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {6, "enterprise_fib50k_ips2m", "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
                {7, "enterprise_fib50k_ips3m", "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_enterprise_b50_d35_r15_p24128_p1616.bin"}
            };
        } else {
            all = {
                {1, "hotspot_fib10k_ips200k", "fib_table_10000_seed_12345.csv", "queries_prefix_200000_hotspot_hp8_r95.bin"},
                {2, "hotspot_fib20k_ips400k", "fib_table_20000_seed_12345.csv", "queries_prefix_400000_hotspot_hp8_r95.bin"},
                {3, "hotspot_fib40k_ips800k", "fib_table_40000_seed_12345.csv", "queries_prefix_800000_hotspot_hp8_r95.bin"},
                {4, "hotspot_fib50k_ips1m", "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_hotspot_hp8_r95.bin"},
                {5, "hotspot_fib50k_ips1_5m", "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_hotspot_hp8_r95.bin"},
                {6, "hotspot_fib50k_ips2m", "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_hotspot_hp8_r95.bin"},
                {7, "hotspot_fib50k_ips3m", "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_hotspot_hp8_r95.bin"}
            };
        }

        std::vector<DatasetConfig> selected;
        for (int id : selected_dataset_ids) {
            auto it = std::find_if(all.begin(), all.end(), [id](const DatasetConfig& c) { return c.id == id; });
            if (it == all.end()) throw std::runtime_error("Neplatne ID datasetu pre zvoleny traffic type: " + std::to_string(id));
            selected.push_back(*it);
        }

        const std::string suffix = make_output_suffix(selected, cpu_threads);
        const fs::path summary_path = output_dir / ("hybrid_dir24_8_lpm_" + suffix + "_summary.csv");
        const fs::path raw_path = output_dir / ("hybrid_dir24_8_lpm_" + suffix + "_raw_runs.csv");
        const fs::path cpu_baseline_summary_path = output_dir / ("cpu_dir24_8_lpm_" + suffix + "_summary.csv");

        log_both("Summary CSV: " + summary_path.string() + "\n");
        log_both("Raw CSV: " + raw_path.string() + "\n");
        log_both("CPU baseline summary CSV: " + cpu_baseline_summary_path.string() + "\n");
        log_both("Log: " + log_path.string() + "\n\n");

        std::vector<SummaryRow> summary_rows;
        std::vector<RawRow> raw_rows;
        std::vector<CpuBaselineSummaryRow> cpu_baseline_summary_rows;

        for (const auto& cfg : selected) {
            log_both("=== Dataset " + cfg.label + " ===\n");
            auto fib_it = fib_by_name.find(cfg.fib_name);
            auto ip_it = ip_by_name.find(cfg.ip_name);
            if (fib_it == fib_by_name.end()) throw std::runtime_error("Chyba FIB subor: " + cfg.fib_name);
            if (ip_it == ip_by_name.end()) throw std::runtime_error("Chyba IP/BIN subor: " + cfg.ip_name);

            log_both("Loading FIB: " + fib_it->second.string() + "\n");
            std::vector<Prefix> fib = load_fib_csv(fib_it->second);
            log_both("Loading IPs: " + ip_it->second.string() + "\n");
            std::vector<uint32_t> ips = (traffic_type == TrafficType::SYNTHETIC) ? load_ips_csv(ip_it->second) : load_ips_bin(ip_it->second);
            log_both("FIB size: " + std::to_string(fib.size()) + ", IP count: " + std::to_string(ips.size()) + "\n");

            const auto build_start = Clock::now();
            Dir24_8Host gpu_dir;
            gpu_dir.build(fib);
            const auto build_end = Clock::now();
            log_both("Build CPU DIR24/8: " + std::to_string(ms_between(build_start, build_end)) + " ms\n");

            const auto flat_start = Clock::now();
            const auto h_primary = flatten_primary(gpu_dir.primary());
            const auto h_secondaries = flatten_secondaries_next_hop(gpu_dir.secondaries());
            const auto flat_end = Clock::now();
            log_both("Flatten for GPU: " + std::to_string(ms_between(flat_start, flat_end)) + " ms\n");
            log_both("Secondary tables: " + std::to_string(gpu_dir.secondaries().size()) + "\n");

            DevicePrimaryEntry* d_primary = nullptr;
            int32_t* d_secondaries = nullptr;
            uint32_t* d_ips = nullptr;
            int32_t* d_out = nullptr;
            uint32_t* h_ips_pinned = nullptr;
            int32_t* h_out_pinned = nullptr;

            CUDA_CHECK(cudaMalloc(&d_primary, h_primary.size() * sizeof(DevicePrimaryEntry)));
            CUDA_CHECK(cudaMemcpy(d_primary, h_primary.data(), h_primary.size() * sizeof(DevicePrimaryEntry), cudaMemcpyHostToDevice));
            if (!h_secondaries.empty()) {
                CUDA_CHECK(cudaMalloc(&d_secondaries, h_secondaries.size() * sizeof(int32_t)));
                CUDA_CHECK(cudaMemcpy(d_secondaries, h_secondaries.data(), h_secondaries.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaMalloc(&d_ips, ips.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_out, ips.size() * sizeof(int32_t)));
            CUDA_CHECK(cudaMallocHost(&h_ips_pinned, ips.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMallocHost(&h_out_pinned, ips.size() * sizeof(int32_t)));
            std::copy(ips.begin(), ips.end(), h_ips_pinned);


            cudaEvent_t ev_start{}, ev_stop{};
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));

            auto run_gpu_once = [&](double& h2d_ms, double& kernel_avg_ms, double& d2h_ms, uint64_t& checksum) {
                CUDA_CHECK(cudaEventRecord(ev_start));
                CUDA_CHECK(cudaMemcpy(d_ips, h_ips_pinned, ips.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                h2d_ms = elapsed_ms(ev_start, ev_stop);

                const int grid_size = static_cast<int>((ips.size() + static_cast<std::size_t>(block_size) - 1u) / static_cast<std::size_t>(block_size));
                double kernel_total = 0.0;
                for (int rep = 0; rep < kernel_repeats_per_measurement; ++rep) {
                    CUDA_CHECK(cudaEventRecord(ev_start));
                    dir24_8_lookup_kernel<<<grid_size, block_size>>>(d_primary, d_secondaries, d_ips, d_out, ips.size());
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaEventRecord(ev_stop));
                    CUDA_CHECK(cudaEventSynchronize(ev_stop));
                    kernel_total += elapsed_ms(ev_start, ev_stop);
                }
                kernel_avg_ms = kernel_total / static_cast<double>(kernel_repeats_per_measurement);

                CUDA_CHECK(cudaEventRecord(ev_start));
                CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, ips.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                d2h_ms = elapsed_ms(ev_start, ev_stop);
                checksum = checksum_vector(h_out_pinned, ips.size());
            };

            // CPU vetva je metodicky zhodna s CPU router modelom:
            // FIB/DIR24_8 struktura je postavena pred meranim a opakovane sa pouziva,
            // buffre su predalokovane a v meranom case je zahrnute kopirovanie davky
            // do ingress buffera + paralelny lookup + zapis vysledkov.
            std::vector<uint32_t> cpu_ingress_buffer(ips.size());
            std::vector<int32_t> cpu_out(ips.size());
            pre_touch_ingress_buffer(cpu_ingress_buffer);
            pre_touch_output_buffer(cpu_out);

            auto run_cpu_once = [&](double& processing_ms, uint64_t& checksum) {
                const auto t0 = Clock::now();
                router_forward_batch_cpu_model(gpu_dir, ips, cpu_ingress_buffer, cpu_out, cpu_threads);
                const auto t1 = Clock::now();

                processing_ms = ms_between(t0, t1);
                checksum = checksum_vector(cpu_out.data(), cpu_out.size());
            };

            uint64_t reference_checksum = 0;
            if (validate_cpu_gpu_checksum_once) {
                log_both("Validating CPU/GPU checksum once...\n");
                uint64_t cpu_sum = 0, gpu_sum = 0;
                double vh2d = 0, vk = 0, vd2h = 0;
                double cpu_processing_ms = 0.0;
                run_cpu_once(cpu_processing_ms, cpu_sum);
                run_gpu_once(vh2d, vk, vd2h, gpu_sum);
                if (cpu_sum != gpu_sum) {
                    std::ostringstream oss;
                    oss << "Checksum mismatch CPU=" << cpu_sum << " GPU=" << gpu_sum << " for " << cfg.label;
                    throw std::runtime_error(oss.str());
                }
                reference_checksum = cpu_sum;
                log_both("Checksum OK: " + std::to_string(reference_checksum) + "\n");
            }

            std::vector<double> decision_ms_values, processing_ms_values, with_decision_ms_values;
            std::vector<double> thr_without_values, thr_with_values, h2d_values, kernel_values, d2h_values;
            std::vector<double> cpu_baseline_ms_values, cpu_baseline_thr_values;

            // Dolezite pre korektnu metodiku:
            // "Bez rozhodovania" sa meria v samostatnej slucke. Scheduler sa spusti iba raz
            // mimo meraneho useku, aby sa pred kazdym meranym runom nezohrieval vstupny batch
            // vypoctom prefixovej lokality.
            //
            // "S rozhodovanim" sa rozhodovacia rezia mera samostatne a pripocita
            // k normalizovanemu casu zvolenej vetvy. Pri GPU tak with_decision_ms
            // neobsahuje vsetky stabilizacne kernel opakovania, ale priemerny kernel cas.
            LocalityInfo scheduler_loc = analyze_prefix_locality_24(ips, locality_sample_fraction, locality_sample_max_count);
            Component scheduler_component = decide_component(scheduler_loc, ips.size(), low_locality_cpu_max_batch_size,
                                                             enterprise_gpu_min_batch_size,
                                                             low_locality_gpu_threshold_avg_per_unique24,
                                                             high_locality_cpu_threshold_avg_per_unique24);

            LocalityInfo last_loc = scheduler_loc;
            Component last_component = scheduler_component;
            uint64_t last_checksum = reference_checksum;

            auto run_selected_component_once = [&](Component component,
                                                   double& processing_ms,
                                                   double& h2d_ms,
                                                   double& kernel_ms,
                                                   double& d2h_ms,
                                                   uint64_t& checksum) {
                h2d_ms = 0.0;
                kernel_ms = 0.0;
                d2h_ms = 0.0;
                checksum = 0;
                processing_ms = 0.0;

                if (component == Component::CPU) {
                    run_cpu_once(processing_ms, checksum);
                } else {
                    run_gpu_once(h2d_ms, kernel_ms, d2h_ms, checksum);
                    processing_ms = h2d_ms + kernel_ms + d2h_ms;
                }
            };

            auto run_without_decision = [&](bool warmup, int run_id) {
                if (pre_measure_sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pre_measure_sleep_ms));

                double h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0;
                double processing_ms = 0.0;
                uint64_t checksum = 0;

                run_selected_component_once(scheduler_component, processing_ms, h2d_ms, kernel_ms, d2h_ms, checksum);

                if (reference_checksum != 0 && checksum != reference_checksum) {
                    throw std::runtime_error("Checksum mismatch v hybrid without-decision run pre " + cfg.label);
                }

                const double ips_m = static_cast<double>(ips.size()) / 1e6;
                const double thr_without = ips_m / (processing_ms / 1000.0);

                raw_rows.push_back(RawRow{
                    cfg.label, cfg.id, run_id,
                    warmup ? "without_decision_warmup" : "without_decision_measured",
                    component_to_string(scheduler_component),
                    fib.size(), ips.size(), scheduler_loc.unique24, scheduler_loc.avg_ips_per_unique24, scheduler_loc.unique24_ratio,
                    0.0, processing_ms, kernel_ms, processing_ms, thr_without, thr_without,
                    0.0, 0.0,
                    h2d_ms, kernel_ms, d2h_ms, checksum
                });

                last_checksum = checksum;

                if (!warmup) {
                    processing_ms_values.push_back(processing_ms);
                    thr_without_values.push_back(thr_without);
                    h2d_values.push_back(h2d_ms);
                    kernel_values.push_back(kernel_ms);
                    d2h_values.push_back(d2h_ms);
                }
            };

            auto run_with_decision = [&](bool warmup, int run_id) {
                if (pre_measure_sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pre_measure_sleep_ms));

                const auto dec_start = Clock::now();
                LocalityInfo loc = analyze_prefix_locality_24(ips, locality_sample_fraction, locality_sample_max_count);
                Component component = decide_component(loc, ips.size(), low_locality_cpu_max_batch_size,
                                                       enterprise_gpu_min_batch_size,
                                                       low_locality_gpu_threshold_avg_per_unique24,
                                                       high_locality_cpu_threshold_avg_per_unique24);
                const auto dec_end = Clock::now();
                const double decision_ms = ms_between(dec_start, dec_end);

                double h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0;
                double processing_ms = 0.0;
                uint64_t checksum = 0;

                run_selected_component_once(component, processing_ms, h2d_ms, kernel_ms, d2h_ms, checksum);

                // processing_ms je pri GPU vetve uz normalizovany na jedno
                // spracovanie davky (H2D + priemerny kernel cas + D2H).
                // Preto k nemu pripocitame iba samostatne zmeranu rozhodovaciu reziu.
                const double with_decision_ms = decision_ms + processing_ms;

                if (reference_checksum != 0 && checksum != reference_checksum) {
                    throw std::runtime_error("Checksum mismatch v hybrid with-decision run pre " + cfg.label);
                }

                const double ips_m = static_cast<double>(ips.size()) / 1e6;
                const double thr_without = ips_m / (processing_ms / 1000.0);
                const double thr_with = ips_m / (with_decision_ms / 1000.0);

                raw_rows.push_back(RawRow{
                    cfg.label, cfg.id, run_id,
                    warmup ? "with_decision_warmup" : "with_decision_measured",
                    component_to_string(component),
                    fib.size(), ips.size(), loc.unique24, loc.avg_ips_per_unique24, loc.unique24_ratio,
                    decision_ms, processing_ms, kernel_ms, with_decision_ms, thr_without, thr_with,
                    0.0, 0.0,
                    h2d_ms, kernel_ms, d2h_ms, checksum
                });

                last_loc = loc;
                last_component = component;
                last_checksum = checksum;

                if (!warmup) {
                    decision_ms_values.push_back(decision_ms);
                    with_decision_ms_values.push_back(with_decision_ms);
                    thr_with_values.push_back(thr_with);
                }
            };

            auto run_cpu_baseline = [&](bool warmup, int run_id) {
                if (pre_measure_sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pre_measure_sleep_ms));

                double processing_ms = 0.0;
                uint64_t checksum = 0;
                run_cpu_once(processing_ms, checksum);

                if (reference_checksum != 0 && checksum != reference_checksum) {
                    throw std::runtime_error("Checksum mismatch v CPU-only baseline run pre " + cfg.label);
                }

                const double ips_m = static_cast<double>(ips.size()) / 1e6;
                const double thr = ips_m / (processing_ms / 1000.0);

                raw_rows.push_back(RawRow{
                    cfg.label, cfg.id, run_id,
                    warmup ? "cpu_baseline_warmup" : "cpu_baseline_measured",
                    "CPU_BASELINE",
                    fib.size(), ips.size(), scheduler_loc.unique24, scheduler_loc.avg_ips_per_unique24, scheduler_loc.unique24_ratio,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    processing_ms, thr,
                    0.0, 0.0, 0.0, checksum
                });

                if (!warmup) {
                    cpu_baseline_ms_values.push_back(processing_ms);
                    cpu_baseline_thr_values.push_back(thr);
                }
            };

            for (int r = 0; r < warmup_runs_per_config; ++r) run_without_decision(true, r);
            for (int r = 0; r < measured_runs_per_config; ++r) run_without_decision(false, r);

            for (int r = 0; r < warmup_runs_per_config; ++r) run_with_decision(true, r);
            for (int r = 0; r < measured_runs_per_config; ++r) run_with_decision(false, r);

            // CPU-only baseline sa meria v tom istom hybridnom executable.
            // Zaradene je az po hybridnych meraniach, aby nemenilo cache/TLB stav
            // existujucich metrík without/with decision. Nepouziva scheduler ani GPU vetvu.
            // Slúzi ako diagnostika, ci rozdiely medzi samostatnym CPU benchmarkom a
            // hybridnou CPU vetvou sposobuje ina meracia sekvencia/toolchain/pamatove
            // rozlozenie, nie odlisna CPU lookup logika.
            for (int r = 0; r < warmup_runs_per_config; ++r) run_cpu_baseline(true, r);
            for (int r = 0; r < measured_runs_per_config; ++r) run_cpu_baseline(false, r);

            const Stats dec_s = aggregate(decision_ms_values);
            const Stats proc_s = aggregate(processing_ms_values);
            const Stats with_s = aggregate(with_decision_ms_values);
            const Stats thr_wo_s = aggregate(thr_without_values);
            const Stats thr_w_s = aggregate(thr_with_values);
            const Stats h2d_s = aggregate(h2d_values);
            const Stats ker_s = aggregate(kernel_values);
            const Stats d2h_s = aggregate(d2h_values);
            const Stats cpu_base_s = aggregate(cpu_baseline_ms_values);
            const Stats cpu_base_thr_s = aggregate(cpu_baseline_thr_values);

            summary_rows.push_back(SummaryRow{
                cfg.label, cfg.id, component_to_string(last_component), fib.size(), ips.size(),
                last_loc.unique24, last_loc.avg_ips_per_unique24, last_loc.unique24_ratio,
                dec_s.avg, dec_s.median, proc_s.avg, proc_s.median, ker_s.avg, ker_s.median, with_s.avg, with_s.median,
                thr_wo_s.avg, thr_wo_s.median, thr_w_s.avg, thr_w_s.median,
                cpu_base_s.avg, cpu_base_s.median, cpu_base_thr_s.avg, cpu_base_thr_s.median,
                h2d_s.avg, ker_s.avg, d2h_s.avg, reference_checksum
            });


            cpu_baseline_summary_rows.push_back(CpuBaselineSummaryRow{
                cfg.id,
                traffic_to_string(traffic_type),
                cfg.id,
                cfg.label,
                cpu_threads,
                cfg.fib_name,
                cfg.ip_name,
                fib.size(),
                ips.size(),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                cpu_base_s.avg,
                cpu_base_s.median,
                cpu_base_thr_s.avg,
                cpu_base_thr_s.median,
                cpu_base_s.avg,
                cpu_base_s.median,
                cpu_base_thr_s.avg,
                cpu_base_thr_s.median,
                cpu_base_s.avg,
                cpu_base_s.median,
                cpu_base_s.min,
                cpu_base_s.max,
                0.0,
                0.0,
                reference_checksum
            });

            log_both("Decision: unique24=" + std::to_string(last_loc.unique24) +
                     ", avg_ips_per_unique24=" + std::to_string(last_loc.avg_ips_per_unique24) +
                     ", batch_size=" + std::to_string(ips.size()) +
                     ", component=" + component_to_string(last_component) + "\n");
            log_both("Measured CPU-only baseline throughput avg [Mpps]: " + std::to_string(cpu_base_thr_s.avg) + "\n");
            log_both("Measured throughput without decision avg [Mpps]: " + std::to_string(thr_wo_s.avg) + "\n");
            log_both("Measured throughput with decision avg [Mpps]: " + std::to_string(thr_w_s.avg) + "\n\n");

            CUDA_CHECK(cudaEventDestroy(ev_start));
            CUDA_CHECK(cudaEventDestroy(ev_stop));
            CUDA_CHECK(cudaFree(d_primary));
            if (d_secondaries) CUDA_CHECK(cudaFree(d_secondaries));
            CUDA_CHECK(cudaFree(d_ips));
            CUDA_CHECK(cudaFree(d_out));
            CUDA_CHECK(cudaFreeHost(h_ips_pinned));
            CUDA_CHECK(cudaFreeHost(h_out_pinned));
        }

        write_summary_csv(summary_path, summary_rows);
        write_raw_csv(raw_path, raw_rows);
        write_cpu_baseline_summary_csv(cpu_baseline_summary_path, cpu_baseline_summary_rows);
        log_both("Hotovo. CSV vystupy ulozene.\n");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CHYBA: " << e.what() << "\n";
        return 1;
    }
}
