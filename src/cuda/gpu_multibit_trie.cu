#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstring>
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

static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

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


static std::vector<uint32_t> load_ips_bin(const fs::path& file_path) {
    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Nepodarilo sa otvorit BIN dataset: " + file_path.string());
    }

    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (file_size % sizeof(uint32_t) != 0) {
        throw std::runtime_error("BIN dataset nema validnu velkost: " + file_path.string());
    }

    size_t count = file_size / sizeof(uint32_t);

    std::vector<uint32_t> ips(count);

    file.read(reinterpret_cast<char*>(ips.data()), file_size);

    if (!file) {
        throw std::runtime_error("Chyba pri citani BIN datasetu: " + file_path.string());
    }

    return ips;
}

static std::vector<fs::path> list_bin_files_sorted(const fs::path& dir) {
    std::vector<fs::path> files;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;

        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());

    return files;
}


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

struct GpuTrieNode {
    int32_t child_base = -1;  // index prvej skupiny 16 deti, -1 ak neexistuju
    int32_t next_hop = -1;
    int16_t best_len = -1;
    uint16_t pad = 0;
};
static_assert(sizeof(GpuTrieNode) == 12, "GpuTrieNode must be 12 bytes");

class NibbleTrie16Host {
public:
    NibbleTrie16Host() {
        nodes_.push_back(GpuTrieNode{}); // root
    }

    int root_index() const { return 0; }
    std::size_t node_count() const { return nodes_.size(); }
    const std::vector<GpuTrieNode>& nodes() const { return nodes_; }

    void update_root(int32_t next_hop, int16_t best_len) {
        update_node(0, next_hop, best_len);
    }

    void insert_expanded_prefix(uint16_t key16, int depth_nibbles, int32_t next_hop, int16_t best_len) {
        int node_idx = 0;
        for (int level = 0; level < depth_nibbles; ++level) {
            const int shift = 12 - 4 * level;
            const int nibble = static_cast<int>((key16 >> shift) & 0xFu);
            node_idx = ensure_child(node_idx, nibble);
        }
        update_node(node_idx, next_hop, best_len);
    }

private:
    std::vector<GpuTrieNode> nodes_;

    void update_node(int node_idx, int32_t next_hop, int16_t best_len) {
        GpuTrieNode& n = nodes_[static_cast<std::size_t>(node_idx)];
        if (best_len > n.best_len) {
            n.best_len = best_len;
            n.next_hop = next_hop;
        }
    }

    int ensure_child(int node_idx, int nibble) {
        GpuTrieNode& n = nodes_[static_cast<std::size_t>(node_idx)];
        if (n.child_base < 0) {
            const int base = static_cast<int>(nodes_.size());
            n.child_base = base;
            nodes_.resize(nodes_.size() + 16u);
            return base + nibble;
        }
        return n.child_base + nibble;
    }
};

struct SplitStride4Tables {
    // Nazvy su zvolene tak, aby sedeli s existujucim DIR24/8 CSV formatom.
    // primary_entries = pocet uzlov prveho 16-bit trie
    // secondary_tables = pocet aktivnych bucketov pre dlhe prefixy
    // secondary_slots = celkovy pocet uzlov v trie pre druhe polovice
    std::vector<GpuTrieNode> first_half_nodes;
    std::vector<int32_t> second_half_roots;  // 65536 poloziek, root index alebo -1
    std::vector<GpuTrieNode> second_half_nodes;
    std::size_t active_second_half_buckets = 0;
};

static int ceil_to_nibbles(int bit_length) {
    if (bit_length <= 0) return 0;
    return (bit_length + 3) / 4;
}

static void expand_and_insert(NibbleTrie16Host& trie,
                              uint16_t value16,
                              int prefix_len_bits,
                              int32_t next_hop,
                              int16_t best_len) {
    if (prefix_len_bits <= 0) {
        trie.update_root(next_hop, best_len);
        return;
    }

    const int depth_nibbles = ceil_to_nibbles(prefix_len_bits);
    const int depth_bits = depth_nibbles * 4;
    const int extra_bits = depth_bits - prefix_len_bits;

    const uint32_t base_prefix = static_cast<uint32_t>(value16) >> (16 - prefix_len_bits);
    const uint32_t variants = 1u << extra_bits;

    for (uint32_t suffix = 0; suffix < variants; ++suffix) {
        const uint32_t expanded = (base_prefix << extra_bits) | suffix;
        const uint16_t key16 = static_cast<uint16_t>(expanded << (16 - depth_bits));
        trie.insert_expanded_prefix(key16, depth_nibbles, next_hop, best_len);
    }
}

static SplitStride4Tables build_split_stride4_tables(const std::vector<Prefix>& fib) {
    NibbleTrie16Host first_half;
    std::vector<int32_t> second_half_roots(1u << 16, -1);
    std::unordered_map<uint16_t, NibbleTrie16Host> bucket_tries;
    bucket_tries.reserve(fib.size());

    std::vector<Prefix> sorted = fib;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Prefix& a, const Prefix& b) {
        return a.length < b.length;
    });

    for (const Prefix& p : sorted) {
        if (p.length == 0) {
            first_half.update_root(p.next_hop, 0);
            continue;
        }

        const uint16_t hi16 = static_cast<uint16_t>(p.network >> 16);
        const uint16_t lo16 = static_cast<uint16_t>(p.network & 0xFFFFu);

        if (p.length <= 16) {
            expand_and_insert(first_half, hi16, static_cast<int>(p.length), p.next_hop, static_cast<int16_t>(p.length));
        } else {
            const uint16_t bucket = hi16;
            const int suffix_len = static_cast<int>(p.length) - 16;
            expand_and_insert(bucket_tries[bucket], lo16, suffix_len, p.next_hop, static_cast<int16_t>(p.length));
        }
    }

    SplitStride4Tables out;
    out.first_half_nodes = first_half.nodes();
    out.second_half_roots = std::move(second_half_roots);

    out.second_half_nodes.reserve(1024);
    std::vector<uint16_t> bucket_keys;
    bucket_keys.reserve(bucket_tries.size());
    for (const auto& kv : bucket_tries) bucket_keys.push_back(kv.first);
    std::sort(bucket_keys.begin(), bucket_keys.end());

    for (uint16_t bucket : bucket_keys) {
        const auto& trie = bucket_tries.at(bucket);
        const int base = static_cast<int>(out.second_half_nodes.size());
        out.second_half_roots[bucket] = base;

        std::vector<GpuTrieNode> adjusted = trie.nodes();
        for (auto& node : adjusted) {
            if (node.child_base >= 0) {
                node.child_base += base;
            }
        }
        out.second_half_nodes.insert(out.second_half_nodes.end(), adjusted.begin(), adjusted.end());
        ++out.active_second_half_buckets;
    }

    return out;
}

__device__ __forceinline__
void trie16_lookup_device(const GpuTrieNode* __restrict__ nodes,
                          int32_t root_idx,
                          uint16_t key16,
                          int32_t& best_hop,
                          int16_t& best_len) {
    if (root_idx < 0) return;

    int32_t node_idx = root_idx;
    GpuTrieNode node = nodes[node_idx];

    if (node.best_len > best_len) {
        best_len = node.best_len;
        best_hop = node.next_hop;
    }

#pragma unroll
    for (int level = 0; level < 4; ++level) {
        if (node.child_base < 0) break;
        const int shift = 12 - 4 * level;
        const int nibble = static_cast<int>((key16 >> shift) & 0xFu);
        node_idx = node.child_base + nibble;
        node = nodes[node_idx];

        if (node.best_len > best_len) {
            best_len = node.best_len;
            best_hop = node.next_hop;
        }
    }
}

__global__
void multibit_trie_stride4_split_lookup_kernel(const GpuTrieNode* __restrict__ first_half_nodes,
                                               const int32_t* __restrict__ second_half_roots,
                                               const GpuTrieNode* __restrict__ second_half_nodes,
                                               const uint32_t* __restrict__ ips,
                                               int32_t* __restrict__ out_next_hop,
                                               std::size_t ip_count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= ip_count) return;

    const uint32_t ip = ips[idx];
    const uint16_t hi16 = static_cast<uint16_t>(ip >> 16);
    const uint16_t lo16 = static_cast<uint16_t>(ip & 0xFFFFu);

    int32_t best_hop = -1;
    int16_t best_len = -1;

    trie16_lookup_device(first_half_nodes, 0, hi16, best_hop, best_len);

    const int32_t second_root = second_half_roots[hi16];
    if (second_root >= 0) {
        trie16_lookup_device(second_half_nodes, second_root, lo16, best_hop, best_len);
    }

    out_next_hop[idx] = best_hop;
}

static uint64_t checksum_vector(const int32_t* data, std::size_t size) {
    uint64_t sum = 0;
    for (std::size_t i = 0; i < size; ++i) {
        sum += static_cast<uint64_t>(static_cast<uint32_t>(data[i]));
    }
    return sum;
}

struct AggregateStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double stddev = 0.0;
};

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

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static double duration_ms(const std::chrono::high_resolution_clock::time_point& a,
                          const std::chrono::high_resolution_clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}


enum class TrafficType {
    SYNTHETIC,
    ENTERPRISE,
    HOTSPOT
};

static std::string traffic_type_to_string(TrafficType traffic_type) {
    switch (traffic_type) {
        case TrafficType::SYNTHETIC:
            return "synthetic";
        case TrafficType::ENTERPRISE:
            return "enterprise";
        case TrafficType::HOTSPOT:
            return "hotspot";
        default:
            return "unknown";
    }
}

struct DatasetConfig {
    int id = 0;
    std::string label;
    std::string fib_name;
    std::string ip_name;
};

static std::string make_output_suffix(TrafficType traffic_type,
                                      const std::vector<DatasetConfig>& selected_configs) {
    if (selected_configs.empty()) {
        throw std::runtime_error("Nie je vybrana ziadna dataset konfiguracia.");
    }

    std::ostringstream oss;
    oss << traffic_type_to_string(traffic_type) << "_ids";
    for (size_t i = 0; i < selected_configs.size(); ++i) {
        if (i > 0) oss << "_";
        oss << selected_configs[i].id;
    }
    return oss.str();
}

int main() {
    try {
        const fs::path fib_dir = lpm_paths::fib_dir();
        const fs::path ips_dir = lpm_paths::synthetic_ip_dir();
        const fs::path output_dir = lpm_paths::cuda_multibit_trie_results_dir();

    const fs::path enterprise_dir =
        lpm_paths::enterprise_query_dir();

    const fs::path hotspot_dir =
        lpm_paths::hotspot_query_dir();

    // ============================================================
    // TRAFFIC TYPE
    // SYNTHETIC / ENTERPRISE / HOTSPOT
    // ============================================================
    const TrafficType traffic_type = TrafficType::SYNTHETIC;



        // Kombinacie datasetov sa maju spustit:
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
        // pouzivaju rovnake ID ako syntetika:
        // 1 = 10k FIB + 200k IPs
        // 2 = 20k FIB + 400k IPs
        // 3 = 40k FIB + 800k IPs
        // 4 = 50k FIB + 1M IPs
        // 5 = 50k FIB + 1.5M IPs
        // 6 = 50k FIB + 2M IPs
        // 7 = 50k FIB + 3M IPs
        const std::vector<int> selected_dataset_ids = {1, 2, 3, 4, 5, 6, 7};

        const int measured_runs_per_config = 10;
        const int warmup_runs_per_config = 5;
        const int kernel_repeats_per_measurement = 20;
        const int pre_measure_sleep_ms = 25;
        const int block_size = 256;
        const bool enable_reference_checksum_validation = true;

        fs::create_directories(output_dir);
        CUDA_CHECK(cudaFree(nullptr));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Global memory: " << (prop.totalGlobalMem / (1024ull * 1024ull)) << " MiB\n";
        std::cout << "Algorithm: GPU MultiBit Trie stride 4 (split 16/16, flat array GPU layout)\n";
        std::cout << "Warmup runs/config: " << warmup_runs_per_config << "\n";
        std::cout << "Measured runs/config: " << measured_runs_per_config << "\n";
        std::cout << "Kernel repeats/measurement: " << kernel_repeats_per_measurement << "\n";
        std::cout << "Pre-measure sleep [ms]: " << pre_measure_sleep_ms << "\n";

        const std::vector<fs::path> fib_files = list_csv_files_sorted(fib_dir);
        std::vector<fs::path> ip_files;

        if (traffic_type == TrafficType::SYNTHETIC) {
            ip_files = list_csv_files_sorted(ips_dir);
        }
        else if (traffic_type == TrafficType::ENTERPRISE) {
            ip_files = list_bin_files_sorted(enterprise_dir);
        }
        else {
            ip_files = list_bin_files_sorted(hotspot_dir);
        }

        if (fib_files.empty()) {
            throw std::runtime_error("V FIB priecinku sa nenasli ziadne CSV subory.");
        }
        if (ip_files.empty()) {
            throw std::runtime_error("V IP/BIN priecinku sa nenasli ziadne subory pre zvoleny traffic type.");
        }

        std::unordered_map<std::string, fs::path> fib_by_name;
        std::unordered_map<std::string, fs::path> ip_by_name;
        for (const auto& f : fib_files) fib_by_name.emplace(f.filename().string(), f);
        for (const auto& f : ip_files) ip_by_name.emplace(f.filename().string(), f);

        
    std::vector<DatasetConfig> all_dataset_configs;

    if (traffic_type == TrafficType::SYNTHETIC) {

        all_dataset_configs = {
            {1, "synthetic_fib10k_ips200k",  "fib_table_10000_seed_12345.csv", "generated_ips_200000_seed_12345.csv"},
            {2, "synthetic_fib20k_ips400k",  "fib_table_20000_seed_12345.csv", "generated_ips_400000_seed_12345.csv"},
            {3, "synthetic_fib40k_ips800k",  "fib_table_40000_seed_12345.csv", "generated_ips_800000_seed_12345.csv"},
            {4, "synthetic_fib50k_ips1m",    "fib_table_50000_seed_12345.csv", "generated_ips_1000000_seed_12345.csv"},
            {5, "synthetic_fib50k_ips1_5m",  "fib_table_50000_seed_12345.csv", "generated_ips_1500000_seed_12345.csv"},
            {6, "synthetic_fib50k_ips2m",    "fib_table_50000_seed_12345.csv", "generated_ips_2000000_seed_12345.csv"},
            {7, "synthetic_fib50k_ips3m",    "fib_table_50000_seed_12345.csv", "generated_ips_3000000_seed_12345.csv"}
        };
    }
    
else if (traffic_type == TrafficType::ENTERPRISE) {

        all_dataset_configs = {
            {1, "enterprise_fib10k_ips200k",  "fib_table_10000_seed_12345.csv", "queries_prefix_200000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {2, "enterprise_fib20k_ips400k",  "fib_table_20000_seed_12345.csv", "queries_prefix_400000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {3, "enterprise_fib40k_ips800k",  "fib_table_40000_seed_12345.csv", "queries_prefix_800000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {4, "enterprise_fib50k_ips1m",    "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {5, "enterprise_fib50k_ips1_5m",  "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {6, "enterprise_fib50k_ips2m",    "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_enterprise_b50_d35_r15_p24128_p1616.bin"},
            {7, "enterprise_fib50k_ips3m",    "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_enterprise_b50_d35_r15_p24128_p1616.bin"}
        };
    }
    else {

        all_dataset_configs = {
            {1, "hotspot_fib10k_ips200k",  "fib_table_10000_seed_12345.csv", "queries_prefix_200000_hotspot_hp8_r95.bin"},
            {2, "hotspot_fib20k_ips400k",  "fib_table_20000_seed_12345.csv", "queries_prefix_400000_hotspot_hp8_r95.bin"},
            {3, "hotspot_fib40k_ips800k",  "fib_table_40000_seed_12345.csv", "queries_prefix_800000_hotspot_hp8_r95.bin"},
            {4, "hotspot_fib50k_ips1m",    "fib_table_50000_seed_12345.csv", "queries_prefix_1000000_hotspot_hp8_r95.bin"},
            {5, "hotspot_fib50k_ips1_5m",  "fib_table_50000_seed_12345.csv", "queries_prefix_1500000_hotspot_hp8_r95.bin"},
            {6, "hotspot_fib50k_ips2m",    "fib_table_50000_seed_12345.csv", "queries_prefix_2000000_hotspot_hp8_r95.bin"},
            {7, "hotspot_fib50k_ips3m",    "fib_table_50000_seed_12345.csv", "queries_prefix_3000000_hotspot_hp8_r95.bin"}
        };
    }


        std::vector<DatasetConfig> dataset_configs;
        dataset_configs.reserve(selected_dataset_ids.size());
        for (int selected_id : selected_dataset_ids) {
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

        const std::string output_suffix = make_output_suffix(traffic_type, dataset_configs);

        const fs::path output_csv =
            output_dir / ("gpu_multibit_trie_lpm_" + output_suffix + "_summary.csv");

        const fs::path output_raw_csv =
            output_dir / ("gpu_multibit_trie_lpm_" + output_suffix + "_raw_runs.csv");

        std::cout << "Vybrane dataset konfiguracie:";
        for (const auto& cfg : dataset_configs) {
            std::cout << " " << cfg.id << "=" << cfg.label;
        }
        std::cout << "\n";
        std::cout << "Summary CSV: " << output_csv.string() << "\n";
        std::cout << "Raw runs CSV: " << output_raw_csv.string() << "\n";

        std::unordered_map<std::string, std::vector<Prefix>> fib_cache;
        std::unordered_map<std::string, std::vector<uint32_t>> ip_cache;

        for (const auto& cfg : dataset_configs) {
            const std::string& fib_name = cfg.fib_name;
            const std::string& ip_name = cfg.ip_name;

            auto fib_it = fib_by_name.find(fib_name);
            if (fib_it == fib_by_name.end()) {
                throw std::runtime_error("V FIB priecinku chyba subor: " + fib_name);
            }
            auto ip_it = ip_by_name.find(ip_name);
            if (ip_it == ip_by_name.end()) {
                throw std::runtime_error("V IPs priecinku chyba subor: " + ip_name);
            }

            if (fib_cache.find(fib_it->second.string()) == fib_cache.end()) {
                fib_cache.emplace(fib_it->second.string(), load_fib_csv(fib_it->second));
            }
            if (ip_cache.find(ip_it->second.string()) == ip_cache.end()) {
                if (traffic_type == TrafficType::SYNTHETIC) {
                    ip_cache.emplace(ip_it->second.string(), load_ips_csv(ip_it->second));
                }
                else {
                    ip_cache.emplace(ip_it->second.string(), load_ips_bin(ip_it->second));
                }
            }
        }

        const auto reference_checksums = make_reference_checksum_map();
        std::vector<ResultRow> summary_rows;
        std::vector<RawRunRow> raw_rows;
        summary_rows.reserve(dataset_configs.size());
        raw_rows.reserve(dataset_configs.size() * static_cast<std::size_t>(warmup_runs_per_config + measured_runs_per_config));

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

            const int grid_size = static_cast<int>((ips.size() + static_cast<std::size_t>(block_size) - 1) /
                                                   static_cast<std::size_t>(block_size));

            std::cout << "  [1/5] Build split stride-4 tabuliek..." << std::flush;
            auto t_build_0 = std::chrono::high_resolution_clock::now();
            SplitStride4Tables tables = build_split_stride4_tables(fib);
            auto t_build_1 = std::chrono::high_resolution_clock::now();
            const double build_once_ms = duration_ms(t_build_0, t_build_1);
            std::cout << " hotovo (" << std::fixed << std::setprecision(3) << build_once_ms << " ms)\n" << std::flush;

            std::cout << "  [2/5] Flatten kopii..." << std::flush;

            auto t_flatten_0 = std::chrono::high_resolution_clock::now();
            std::vector<GpuTrieNode> first_half_nodes = tables.first_half_nodes;
            std::vector<int32_t> second_half_roots = tables.second_half_roots;
            std::vector<GpuTrieNode> second_half_nodes = tables.second_half_nodes;
            auto t_flatten_1 = std::chrono::high_resolution_clock::now();
            const double flatten_once_ms = duration_ms(t_flatten_0, t_flatten_1);
            std::cout << " hotovo (" << std::fixed << std::setprecision(3) << flatten_once_ms << " ms)\n" << std::flush;

            const std::size_t primary_entries = first_half_nodes.size();
            const std::size_t secondary_tables = tables.active_second_half_buckets;
            const std::size_t secondary_slots = second_half_nodes.size();

            std::cout << "  [3/5] Alokacia GPU pamate..." << std::flush;

            GpuTrieNode* d_first_half_nodes = nullptr;
            int32_t* d_second_half_roots = nullptr;
            GpuTrieNode* d_second_half_nodes = nullptr;
            uint32_t* d_ips = nullptr;
            int32_t* d_out = nullptr;

            CUDA_CHECK(cudaMalloc(&d_first_half_nodes, first_half_nodes.size() * sizeof(GpuTrieNode)));
            CUDA_CHECK(cudaMalloc(&d_second_half_roots, second_half_roots.size() * sizeof(int32_t)));
            if (!second_half_nodes.empty()) {
                CUDA_CHECK(cudaMalloc(&d_second_half_nodes, second_half_nodes.size() * sizeof(GpuTrieNode)));
            }
            CUDA_CHECK(cudaMalloc(&d_ips, ips.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_out, ips.size() * sizeof(int32_t)));

            CUDA_CHECK(cudaEventRecord(ev_start));
            CUDA_CHECK(cudaMemcpy(d_first_half_nodes, first_half_nodes.data(),
                                  first_half_nodes.size() * sizeof(GpuTrieNode), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            const double upload_primary_once_ms = elapsed_ms(ev_start, ev_stop);

            CUDA_CHECK(cudaEventRecord(ev_start));
            CUDA_CHECK(cudaMemcpy(d_second_half_roots, second_half_roots.data(),
                                  second_half_roots.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
            if (!second_half_nodes.empty()) {
                CUDA_CHECK(cudaMemcpy(d_second_half_nodes, second_half_nodes.data(),
                                      second_half_nodes.size() * sizeof(GpuTrieNode), cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            const double upload_secondaries_once_ms = elapsed_ms(ev_start, ev_stop);
            const double upload_tables_once_ms = upload_primary_once_ms + upload_secondaries_once_ms;
            const double setup_once_total_ms = build_once_ms + flatten_once_ms + upload_tables_once_ms;
            std::cout << " hotovo (primary=" << upload_primary_once_ms << " ms, secondary=" << upload_secondaries_once_ms << " ms)\n" << std::flush;

            std::cout << "  Uzly: first_half=" << primary_entries
                      << ", second_half_roots=" << second_half_roots.size()
                      << ", second_half_nodes=" << secondary_slots
                      << ", active_buckets=" << secondary_tables << "\n" << std::flush;

            std::cout << "  [5/5] Meranie behov...\n" << std::flush;

            // Alokacia page-locked (pinned) host pamate pre per-batch prenosy.
            // Pouziva rovnaky prenosovy model ako CUDA DIR24/8 benchmark, aby boli
            // metriky kernel + prenos metodicky porovnatelne medzi algoritmami.
            uint32_t* h_ips_pinned = nullptr;
            int32_t* h_out_pinned = nullptr;
            CUDA_CHECK(cudaMallocHost(&h_ips_pinned, ips.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMallocHost(&h_out_pinned, ips.size() * sizeof(int32_t)));
            std::copy(ips.begin(), ips.end(), h_ips_pinned);

            std::vector<double> h2d_ms_values;
            std::vector<double> kernel_avg_ms_values;
            std::vector<double> d2h_ms_values;
            std::vector<double> transfer_plus_kernel_ms_values;
            std::vector<double> total_end_to_end_ms_values;
            std::vector<double> kernel_throughput_values;

            h2d_ms_values.reserve(measured_runs_per_config);
            kernel_avg_ms_values.reserve(measured_runs_per_config);
            d2h_ms_values.reserve(measured_runs_per_config);
            transfer_plus_kernel_ms_values.reserve(measured_runs_per_config);
            total_end_to_end_ms_values.reserve(measured_runs_per_config);
            kernel_throughput_values.reserve(measured_runs_per_config);

            uint64_t final_checksum = 0;

            const int total_runs = warmup_runs_per_config + measured_runs_per_config;
            for (int run = 0; run < total_runs; ++run) {
                const bool warmup = run < warmup_runs_per_config;
                const int run_id = run + 1;

                if (!warmup) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(pre_measure_sleep_ms));
                }

                const auto t_total_0 = std::chrono::high_resolution_clock::now();

                CUDA_CHECK(cudaEventRecord(ev_start));
                CUDA_CHECK(cudaMemcpy(d_ips, h_ips_pinned, ips.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                const double h2d_ips_ms = elapsed_ms(ev_start, ev_stop);

                CUDA_CHECK(cudaEventRecord(ev_start));
                for (int repeat = 0; repeat < kernel_repeats_per_measurement; ++repeat) {
                    multibit_trie_stride4_split_lookup_kernel<<<grid_size, block_size>>>(
                        d_first_half_nodes,
                        d_second_half_roots,
                        d_second_half_nodes,
                        d_ips,
                        d_out,
                        ips.size());
                }
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cudaGetLastError());
                const double kernel_total_ms = elapsed_ms(ev_start, ev_stop);
                const double kernel_avg_ms = kernel_total_ms / static_cast<double>(kernel_repeats_per_measurement);

                CUDA_CHECK(cudaEventRecord(ev_start));
                CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, ips.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                const double d2h_ms = elapsed_ms(ev_start, ev_stop);

                CUDA_CHECK(cudaDeviceSynchronize());
                const auto t_total_1 = std::chrono::high_resolution_clock::now();
                const double total_end_to_end_ms = duration_ms(t_total_0, t_total_1);
                const double transfer_plus_kernel_ms = h2d_ips_ms + kernel_avg_ms + d2h_ms;
                const double kernel_throughput_mpps =
                    (static_cast<double>(ips.size()) / 1.0e6) / (kernel_avg_ms / 1000.0);
                const uint64_t checksum = checksum_vector(h_out_pinned, ips.size());
                final_checksum = checksum;

                raw_rows.push_back(RawRunRow{
                    fib_file.filename().string(),
                    ip_file.filename().string(),
                    fib.size(),
                    ips.size(),
                    warmup,
                    run_id,
                    kernel_repeats_per_measurement,
                    h2d_ips_ms,
                    kernel_total_ms,
                    kernel_avg_ms,
                    d2h_ms,
                    transfer_plus_kernel_ms,
                    total_end_to_end_ms,
                    kernel_throughput_mpps,
                    checksum
                });

                if (!warmup) {
                    h2d_ms_values.push_back(h2d_ips_ms);
                    kernel_avg_ms_values.push_back(kernel_avg_ms);
                    d2h_ms_values.push_back(d2h_ms);
                    transfer_plus_kernel_ms_values.push_back(transfer_plus_kernel_ms);
                    total_end_to_end_ms_values.push_back(total_end_to_end_ms);
                    kernel_throughput_values.push_back(kernel_throughput_mpps);
                }

                std::cout << "  Run " << run_id << (warmup ? " [warmup]" : " [measured]")
                          << ": H2D=" << std::fixed << std::setprecision(3) << h2d_ips_ms << " ms"
                          << ", kernel_avg=" << kernel_avg_ms << " ms"
                          << ", D2H=" << d2h_ms << " ms"
                          << ", checksum=" << checksum << "\n";
            }

            if (enable_reference_checksum_validation) {
                auto it = reference_checksums.find(combo_key);
                if (it != reference_checksums.end() && it->second != final_checksum) {
                    std::ostringstream oss;
                    oss << "Checksum mismatch pre " << combo_key
                        << ": ocakavane=" << it->second
                        << ", skutocne=" << final_checksum;
                    throw std::runtime_error(oss.str());
                }
            }

            const AggregateStats h2d_stats = compute_stats(h2d_ms_values);
            const AggregateStats kernel_stats = compute_stats(kernel_avg_ms_values);
            const AggregateStats d2h_stats = compute_stats(d2h_ms_values);
            const AggregateStats transfer_stats = compute_stats(transfer_plus_kernel_ms_values);
            const AggregateStats total_stats = compute_stats(total_end_to_end_ms_values);
            const AggregateStats kernel_thr_stats = compute_stats(kernel_throughput_values);

            std::vector<double> transfer_thr_values;
            std::vector<double> total_thr_values;
            transfer_thr_values.reserve(transfer_plus_kernel_ms_values.size());
            total_thr_values.reserve(total_end_to_end_ms_values.size());
            for (double ms : transfer_plus_kernel_ms_values) {
                transfer_thr_values.push_back((static_cast<double>(ips.size()) / 1.0e6) / (ms / 1000.0));
            }
            for (double ms : total_end_to_end_ms_values) {
                total_thr_values.push_back((static_cast<double>(ips.size()) / 1.0e6) / (ms / 1000.0));
            }
            const AggregateStats transfer_thr_stats = compute_stats(transfer_thr_values);
            const AggregateStats total_thr_stats = compute_stats(total_thr_values);

            summary_rows.push_back(ResultRow{
                fib_file.filename().string(),
                ip_file.filename().string(),
                fib.size(),
                ips.size(),
                primary_entries,
                secondary_tables,
                secondary_slots,
                warmup_runs_per_config,
                measured_runs_per_config,
                kernel_repeats_per_measurement,
                pre_measure_sleep_ms,
                block_size,
                grid_size,
                build_once_ms,
                flatten_once_ms,
                upload_primary_once_ms,
                upload_secondaries_once_ms,
                upload_tables_once_ms,
                setup_once_total_ms,
                h2d_stats.avg,
                h2d_stats.min,
                h2d_stats.max,
                h2d_stats.median,
                h2d_stats.stddev,
                kernel_stats.avg,
                kernel_stats.min,
                kernel_stats.max,
                kernel_stats.median,
                kernel_stats.stddev,
                kernel_thr_stats.avg,
                kernel_thr_stats.median,
                d2h_stats.avg,
                d2h_stats.min,
                d2h_stats.max,
                d2h_stats.median,
                d2h_stats.stddev,
                transfer_stats.avg,
                transfer_stats.min,
                transfer_stats.max,
                transfer_stats.median,
                transfer_stats.stddev,
                transfer_thr_stats.avg,
                transfer_thr_stats.median,
                total_stats.avg,
                total_stats.min,
                total_stats.max,
                total_stats.median,
                total_stats.stddev,
                total_thr_stats.avg,
                total_thr_stats.median,
                final_checksum
            });

            // Uvolnenie pinned host bufferov po dokonceni merani danej konfiguracie.
            CUDA_CHECK(cudaFreeHost(h_ips_pinned));
            CUDA_CHECK(cudaFreeHost(h_out_pinned));

            CUDA_CHECK(cudaFree(d_first_half_nodes));
            CUDA_CHECK(cudaFree(d_second_half_roots));
            if (d_second_half_nodes != nullptr) CUDA_CHECK(cudaFree(d_second_half_nodes));
            CUDA_CHECK(cudaFree(d_ips));
            CUDA_CHECK(cudaFree(d_out));
        }

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));

        write_summary_csv(output_csv, summary_rows);
        write_raw_runs_csv(output_raw_csv, raw_rows);

        std::cout << "\nHotovo. Summary CSV: " << output_csv.string() << "\n";
        std::cout << "Raw runs CSV: " << output_raw_csv.string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CHYBA: " << e.what() << "\n" << std::flush;
        return 1;
    } catch (...) {
        std::cerr << "CHYBA: neznamy pad programu\n" << std::flush;
        return 1;
    }
}
