#pragma once
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "../spin_types.h"

namespace IO {

/**
 * Represents a magnetic species in the unit cell
 */
struct MagneticSpecies {
    std::string name;           // Unique identifier (e.g., "Fe_moment", "Te1_Orb")
    SpinType spin_type;         // ISING or HEISENBERG
    double local_pos[3];        // Position in unit cell (crystal coordinates)
    
    MagneticSpecies(const std::string& n, SpinType type, double x, double y, double z) 
        : name(n), spin_type(type) {
        local_pos[0] = x;
        local_pos[1] = y; 
        local_pos[2] = z;
    }
};

/**
 * Represents an exchange coupling between two species
 */
struct ExchangeCoupling {
    std::string species1_name;  // First species name
    std::string species2_name;  // Second species name
    int cell_offset[3];         // Unit cell offset (Rx, Ry, Rz)
    double J;                   // Exchange coupling strength
    
    ExchangeCoupling(const std::string& s1, const std::string& s2, 
                     int rx, int ry, int rz, double coupling)
        : species1_name(s1), species2_name(s2), J(coupling) {
        cell_offset[0] = rx;
        cell_offset[1] = ry;
        cell_offset[2] = rz;
    }
};

/**
 * Represents a Kugel-Khomskii coupling between two sites
 * KK coupling: K * (S_i · S_j) * (τ_i * τ_j)
 * where S is Heisenberg spin and τ is Ising spin at each site
 */
struct KKCoupling {
    std::string species1_name;  // First species name (identifies site)
    std::string species2_name;  // Second species name (identifies site)
    int cell_offset[3];         // Unit cell offset (Rx, Ry, Rz)
    double K;                   // KK coupling strength
    
    KKCoupling(const std::string& s1, const std::string& s2, 
               int rx, int ry, int rz, double coupling)
        : species1_name(s1), species2_name(s2), K(coupling) {
        cell_offset[0] = rx;
        cell_offset[1] = ry;
        cell_offset[2] = rz;
    }
};

/**
 * Monte Carlo simulation parameters
 */
struct MonteCarloConfig {
    int warmup_steps;
    int measurement_steps;
    int sampling_frequency;
    long int seed;
};

/**
 * Temperature configuration - supports both single temperature and scans
 */
struct TemperatureConfig {
    enum Type { SINGLE, SCAN };
    Type type;
    
    // For single temperature
    double value;
    
    // For temperature scan
    double max_temp;
    double min_temp;
    double temp_step;
};

/**
 * Output configuration
 */
struct OutputConfig {
    std::string base_name;
    std::string directory;
    
    // Observable output options
    bool output_energy_total = true;           // Output total energy (in addition to per-spin)
    bool output_onsite_magnetization = false;  // Output per-species magnetization
    bool output_correlations = true;           // Output spin-spin correlations with first spin
    
    // Legacy: always output basic observables (E/spin, M, Cv, chi, acceptance)
};

/**
 * Initial configuration options
 */
struct InitializationConfig {
    enum Type { RANDOM, CUSTOM };
    Type type = RANDOM;
    
    // For CUSTOM type: specify values for each spin in unit cell
    // For Ising: +1 or -1
    // For Heisenberg: will be interpreted as sz component (sx=sy=0, normalized)
    std::vector<double> pattern;
    
    // Random seed for RANDOM initialization (uses global seed if not set)
    long int random_seed = -1;
};

/**
 * Diagnostic and profiling options
 */
struct DiagnosticConfig {
    bool enable_profiling = false;                // Enable timing/profiling output
    bool enable_config_dump = false;              // Enable configuration dumps
    bool enable_observable_evolution = false;     // Track observable evolution during measurement
    
    // Configuration dump options
    std::vector<int> dump_ranks;                  // Which ranks to dump (empty = none)
    bool dump_all_ranks = false;                  // If true, dump all ranks
    int dump_every_n_measurements = 10;           // Dump every N measurements
    std::string dump_format = "text";             // "text" or "binary" (currently only text supported)
    
    // Profiling options
    bool estimate_autocorrelation = true;          // Estimate autocorrelation times (printed to stdout)
    bool recompute_observables_each_sample = false; // If true, recompute energy/mag from scratch each sample (slow but accurate)
                                                    // If false, track incrementally (fast, recommended)
    
    // Helper to check if a specific rank should dump
    bool should_dump_rank(int rank) const {
        if (!enable_config_dump && !enable_observable_evolution) return false;
        if (dump_all_ranks) return true;
        return std::find(dump_ranks.begin(), dump_ranks.end(), rank) != dump_ranks.end();
    }
};

/**
 * Complete simulation configuration
 */
struct SimulationConfig {
    // Simulation parameters
    std::string simulation_type;
    int lattice_size;
    
    // Temperature settings
    TemperatureConfig temperature;
    
    // Monte Carlo parameters
    MonteCarloConfig monte_carlo;
    
    // Output settings
    OutputConfig output;
    
    // Diagnostic settings (optional)
    DiagnosticConfig diagnostics;
    
    // Initialization settings (optional)
    InitializationConfig initialization;
    
    // Input file paths
    std::string species_file;
    std::string couplings_file;
    std::string kugel_khomskii_file;  // Optional, for future use
    
    // Parsed data
    std::vector<MagneticSpecies> species;
    std::vector<ExchangeCoupling> couplings;
    std::vector<KKCoupling> kk_couplings;
};

} // namespace IO