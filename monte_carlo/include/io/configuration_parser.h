#pragma once
#include "config_types.h"
#include <string>
#include <stdexcept>

namespace IO {

/**
 * Exception for configuration parsing errors
 */
class ConfigurationError : public std::runtime_error {
public:
    ConfigurationError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * Main configuration parser class
 * 
 * Provides a simple interface to load all simulation configuration
 * from TOML and data files. Designed for clarity over performance.
 */
class ConfigurationParser {
public:
    /**
     * Load complete simulation configuration from TOML file
     * 
     * @param toml_file Path to main TOML configuration file
     * @return Complete configuration structure
     * @throws ConfigurationError if any parsing fails
     */
    static SimulationConfig load_configuration(const std::string& toml_file);
    
private:
    /**
     * Parse the main TOML configuration file
     */
    static void parse_toml_file(const std::string& toml_file, SimulationConfig& config);
    
    /**
     * Parse species definition file
     */
    static std::vector<MagneticSpecies> parse_species_file(const std::string& species_file);
    
    /**
     * Parse coupling definition file  
     */
    static std::vector<ExchangeCoupling> parse_couplings_file(const std::string& couplings_file);
    
    /**
     * Validate that all coupling species exist in species list
     */
    static void validate_configuration(const SimulationConfig& config);
    
    /**
     * Convert string to SpinType enum
     */
    static SpinType string_to_spin_type(const std::string& type_str);
    
    /**
     * Check for mixed Ising-Heisenberg couplings (not yet supported)
     */
    static void check_mixed_couplings(const std::vector<MagneticSpecies>& species,
                                      const std::vector<ExchangeCoupling>& couplings);
};

} // namespace IO