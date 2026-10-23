#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/paths.hpp"

struct Prefix {
    uint32_t network;
    uint32_t mask;
    uint8_t length;
    int32_t next_hop;
};

static constexpr uint32_t DEFAULT_SEED = 12345u;

static void print_usage(const char* exe_name) {
    std::cout
        << "Pouzitie:\n"
        << "  " << exe_name << " ips <pocet> [seed] [vystupny_priecinok]\n"
        << "  " << exe_name << " fib <pocet> [seed] [vystupny_priecinok]\n\n"
        << "Priklady:\n"
        << "  " << exe_name << " ips 200000\n"
        << "  " << exe_name << " ips 5000000 12345\n"
        << "  " << exe_name << " fib 100000 12345\n"
        << "  " << exe_name << " fib 500000 777 D:\\data\\fib\n\n"
        << "Predvoleny seed: " << DEFAULT_SEED << "\n"
        << "Predvoleny vystup pre IP:  data/traffic/synthetic pod LPM_PROJECT_ROOT\n"
        << "Predvoleny vystup pre FIB: data/fib pod LPM_PROJECT_ROOT\n";
}

static uint32_t prefix_mask(uint8_t len) {
    if (len == 0) {
        return 0u;
    }
    if (len >= 32) {
        return 0xFFFFFFFFu;
    }
    return 0xFFFFFFFFu << (32 - len);
}

static std::string uint32_to_ip(uint32_t ip) {
    std::ostringstream oss;
    oss << ((ip >> 24) & 0xFF) << '.'
        << ((ip >> 16) & 0xFF) << '.'
        << ((ip >> 8) & 0xFF) << '.'
        << (ip & 0xFF);
    return oss.str();
}

static bool ensure_directory_exists(const std::string& dir) {
    try {
        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Chyba pri vytvarani priecinka: " << dir << "\n";
        std::cerr << e.what() << "\n";
        return false;
    }
}

static std::string join_path(const std::string& dir, const std::string& file) {
    return (std::filesystem::path(dir) / file).string();
}

static std::vector<uint32_t> generate_ips(std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0u, 0xFFFFFFFFu);

    std::vector<uint32_t> ips;
    ips.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        ips.push_back(dist(rng));
    }

    return ips;
}

static std::vector<Prefix> generate_fib(std::size_t m, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> ipdist(0u, 0xFFFFFFFFu);
    std::uniform_int_distribution<int> len_choice(0, 99);

    std::vector<Prefix> fib;
    fib.reserve(m + 1);

    for (std::size_t i = 0; i < m; ++i) {
        uint32_t base = ipdist(rng);

        int r = len_choice(rng);
        uint8_t len;
        if (r < 10) {
            len = 8;
        } else if (r < 30) {
            len = 16;
        } else if (r < 80) {
            len = 24;
        } else {
            len = 32;
        }

        Prefix p{};
        p.length = len;
        p.mask = prefix_mask(len);
        p.network = base & p.mask;
        p.next_hop = static_cast<int32_t>(i % 1024);

        fib.push_back(p);
    }

    Prefix def{};
    def.length = 0;
    def.mask = 0u;
    def.network = 0u;
    def.next_hop = 0;
    fib.push_back(def);

    return fib;
}

static bool save_ips_to_csv(const std::vector<uint32_t>& ips,
                            std::size_t n,
                            uint32_t seed,
                            const std::string& out_dir) {
    if (!ensure_directory_exists(out_dir)) {
        return false;
    }

    const std::string filename = "generated_ips_" + std::to_string(n) +
                                 "_seed_" + std::to_string(seed) + ".csv";
    const std::string full_path = join_path(out_dir, filename);

    std::ofstream file(full_path, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Chyba: nepodarilo sa vytvorit subor " << full_path << "\n";
        return false;
    }

    file << "index,ip_uint32,ip_dotted\n";
    for (std::size_t i = 0; i < ips.size(); ++i) {
        file << i << ',' << ips[i] << ',' << uint32_to_ip(ips[i]) << '\n';
    }

    file.close();
    std::cout << "Ulozene IP adresy do: " << full_path << "\n";
    std::cout << "Pocet IP adries: " << ips.size() << "\n";
    std::cout << "Seed: " << seed << "\n";
    return true;
}

static bool save_fib_to_csv(const std::vector<Prefix>& fib,
                            std::size_t m,
                            uint32_t seed,
                            const std::string& out_dir) {
    if (!ensure_directory_exists(out_dir)) {
        return false;
    }

    const std::string filename = "fib_table_" + std::to_string(m) +
                                 "_seed_" + std::to_string(seed) + ".csv";
    const std::string full_path = join_path(out_dir, filename);

    std::ofstream file(full_path, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Chyba: nepodarilo sa vytvorit subor " << full_path << "\n";
        return false;
    }

    file << "index,network_uint32,network_dotted,prefix_length,mask_uint32,mask_dotted,next_hop\n";
    for (std::size_t i = 0; i < fib.size(); ++i) {
        file << i << ','
             << fib[i].network << ','
             << uint32_to_ip(fib[i].network) << ','
             << static_cast<int>(fib[i].length) << ','
             << fib[i].mask << ','
             << uint32_to_ip(fib[i].mask) << ','
             << fib[i].next_hop << '\n';
    }

    file.close();
    std::cout << "Ulozena FIB tabulka do: " << full_path << "\n";
    std::cout << "Pocet FIB zaznamov bez default route: " << m << "\n";
    std::cout << "Pocet riadkov spolu vratane default route: " << fib.size() << "\n";
    std::cout << "Seed: " << seed << "\n";
    return true;
}

static bool parse_size_t(const std::string& text, std::size_t& value) {
    try {
        unsigned long long parsed = std::stoull(text);
        if (parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            return false;
        }
        value = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_u32(const std::string& text, uint32_t& value) {
    try {
        unsigned long long parsed = std::stoull(text);
        if (parsed > static_cast<unsigned long long>(std::numeric_limits<uint32_t>::max())) {
            return false;
        }
        value = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];

    std::size_t count = 0;
    if (!parse_size_t(argv[2], count)) {
        std::cerr << "Chyba: neplatny pocet zaznamov: " << argv[2] << "\n";
        return 1;
    }

    uint32_t seed = DEFAULT_SEED;
    if (argc >= 4) {
        if (!parse_u32(argv[3], seed)) {
            std::cerr << "Chyba: neplatny seed: " << argv[3] << "\n";
            return 1;
        }
    }

    std::string out_dir;
    if (mode == "ips") {
        out_dir = (argc >= 5) ? argv[4] : lpm_paths::synthetic_ip_dir().string();
        std::cout << "Generujem IP adresy...\n";
        std::cout << "Pocet: " << count << "\n";
        std::cout << "Seed: " << seed << "\n";
        std::cout << "Vystupny priecinok: " << out_dir << "\n";

        auto ips = generate_ips(count, seed);
        return save_ips_to_csv(ips, count, seed, out_dir) ? 0 : 2;
    }

    if (mode == "fib") {
        out_dir = (argc >= 5) ? argv[4] : lpm_paths::fib_dir().string();
        std::cout << "Generujem FIB tabulku...\n";
        std::cout << "Pocet prefixov bez default route: " << count << "\n";
        std::cout << "Seed: " << seed << "\n";
        std::cout << "Vystupny priecinok: " << out_dir << "\n";

        auto fib = generate_fib(count, seed);
        return save_fib_to_csv(fib, count, seed, out_dir) ? 0 : 2;
    }

    std::cerr << "Chyba: neznamy mod '" << mode << "'. Pouzi 'ips' alebo 'fib'.\n";
    print_usage(argv[0]);
    return 1;
}
