#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace lpm_paths {
namespace fs = std::filesystem;

// LPM_PROJECT_ROOT must point to the repository root at program runtime.
// This deliberately avoids embedding a machine-specific path at compile time.
inline fs::path project_root() {
    const char* value = std::getenv("LPM_PROJECT_ROOT");
    if (value == nullptr || std::string(value).empty()) {
        throw std::runtime_error(
            "LPM_PROJECT_ROOT is not set. Set it to the repository root before "
            "running this program.");
    }

    const fs::path root = fs::absolute(fs::path(value)).lexically_normal();
    if (!fs::exists(root) || !fs::is_directory(root)) {
        throw std::runtime_error(
            "LPM_PROJECT_ROOT is not an existing directory: " + root.string());
    }
    return root;
}

inline fs::path data_dir() {
    return project_root() / "data";
}

inline fs::path fib_dir() {
    return data_dir() / "fib";
}

inline fs::path synthetic_ip_dir() {
    return data_dir() / "traffic" / "synthetic";
}

inline fs::path enterprise_query_dir() {
    return data_dir() / "traffic" / "enterprise";
}

inline fs::path hotspot_query_dir() {
    return data_dir() / "traffic" / "hotspot";
}

inline fs::path results_dir() {
    return project_root() / "results";
}

inline fs::path benchmark_results_dir() {
    return results_dir() / "benchmarks";
}

inline fs::path cpu_dir24_8_results_dir() {
    return benchmark_results_dir() / "cpu-dir24-8";
}

inline fs::path cpu_multibit_trie_results_dir() {
    return benchmark_results_dir() / "cpu-multibit-trie";
}

inline fs::path cuda_dir24_8_results_dir() {
    return benchmark_results_dir() / "cuda-dir24-8";
}

inline fs::path cuda_multibit_trie_results_dir() {
    return benchmark_results_dir() / "cuda-multibit-trie";
}

inline fs::path hybrid_dir24_8_results_dir() {
    return benchmark_results_dir() / "hybrid-dir24-8";
}

inline fs::path figure_results_dir() {
    return results_dir() / "figures";
}

inline fs::path nsight_results_dir() {
    return results_dir() / "nsight-compute";
}

inline void ensure_output_dirs() {
    fs::create_directories(cpu_dir24_8_results_dir());
    fs::create_directories(cpu_multibit_trie_results_dir());
    fs::create_directories(cuda_dir24_8_results_dir());
    fs::create_directories(cuda_multibit_trie_results_dir());
    fs::create_directories(hybrid_dir24_8_results_dir());
    fs::create_directories(figure_results_dir());
    fs::create_directories(nsight_results_dir());
}

}  // namespace lpm_paths
