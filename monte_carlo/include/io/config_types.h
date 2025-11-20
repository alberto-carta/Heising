#pragma once
#include <string>
#include <vector>
#include <map>
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
    
    // Input file paths
    std::string species_file;
    std::string couplings_file;
    std::string kugel_khomskii_file;  // Optional, for future use
    
    // Parsed data
    std::vector<MagneticSpecies> species;
    std::vector<ExchangeCoupling> couplings;
};

} // namespace IO