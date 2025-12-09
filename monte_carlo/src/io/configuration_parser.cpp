#include "../include/io/configuration_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

#include <toml.hpp>  // toml11 library for TOML parsing

namespace IO {

SimulationConfig ConfigurationParser::load_configuration(const std::string& toml_file) {
    SimulationConfig config;
    
    try {
        // Parse main TOML file
        parse_toml_file(toml_file, config);
        
        // Parse referenced data files (make paths relative to TOML file location)
        std::string toml_dir = toml_file.substr(0, toml_file.find_last_of("/\\"));
        if (toml_dir == toml_file) toml_dir = ".";  // No directory in path
        
        std::string species_path = (config.species_file[0] == '/') ? config.species_file : toml_dir + "/" + config.species_file;
        std::string couplings_path = (config.couplings_file[0] == '/') ? config.couplings_file : toml_dir + "/" + config.couplings_file;
        
        config.species = parse_species_file(species_path);
        config.couplings = parse_couplings_file(couplings_path);
        
        // Validate the complete configuration
        validate_configuration(config);
        
        std::cout << "Configuration loaded successfully:" << std::endl;
        std::cout << "  - " << config.species.size() << " magnetic species" << std::endl;
        std::cout << "  - " << config.couplings.size() << " exchange couplings" << std::endl;
        std::cout << "  - Lattice size: " << config.lattice_size << "³" << std::endl;
        
        return config;
        
    } catch (const std::exception& e) {
        throw ConfigurationError("Failed to load configuration: " + std::string(e.what()));
    }
}

void ConfigurationParser::parse_toml_file(const std::string& toml_file, SimulationConfig& config) {
    try {
        // Parse TOML file using toml11
        const auto data = toml::parse(toml_file);
        
        // Parse simulation section
        if (data.contains("simulation")) {
            const auto sim = toml::find(data, "simulation");
            config.simulation_type = toml::find_or<std::string>(sim, "type", "temperature_scan");
            config.monte_carlo.seed = toml::find_or<long>(sim, "seed", -12345);
        }
        
        // Parse lattice section  
        if (data.contains("lattice")) {
            const auto lattice = toml::find(data, "lattice");
            config.lattice_size = toml::find_or<int>(lattice, "size", 8);
        }
        
        // Parse monte_carlo section
        if (data.contains("monte_carlo")) {
            const auto mc = toml::find(data, "monte_carlo");
            config.monte_carlo.warmup_steps = toml::find_or<int>(mc, "warmup_steps", 8000);
            config.monte_carlo.measurement_steps = toml::find_or<int>(mc, "measurement_steps", 80000);
            config.monte_carlo.sampling_frequency = toml::find_or<int>(mc, "sampling_frequency", 100);
        }
        
        // Parse temperature section
        if (data.contains("temperature")) {
            const auto temp = toml::find(data, "temperature");
            
            if (temp.contains("value")) {
                // Single temperature mode
                config.temperature.type = TemperatureConfig::SINGLE;
                config.temperature.value = toml::find<double>(temp, "value");
            } else {
                // Temperature scan mode (default)
                config.temperature.type = TemperatureConfig::SCAN;
                config.temperature.max_temp = toml::find_or<double>(temp, "max", 6.0);
                config.temperature.min_temp = toml::find_or<double>(temp, "min", 0.5);
                config.temperature.temp_step = toml::find_or<double>(temp, "step", 0.2);
            }
        } else {
            // Default to temperature scan
            config.temperature.type = TemperatureConfig::SCAN;
            config.temperature.max_temp = 6.0;
            config.temperature.min_temp = 0.5; 
            config.temperature.temp_step = 0.2;
        }
        
        // Parse output section
        if (data.contains("output")) {
            const auto output = toml::find(data, "output");
            config.output.base_name = toml::find_or<std::string>(output, "base_name", "simulation");
            config.output.directory = toml::find_or<std::string>(output, "directory", ".");
        } else {
            config.output.base_name = "simulation";
            config.output.directory = ".";
        }
        
        // Parse input_files section
        if (data.contains("input_files")) {
            const auto files = toml::find(data, "input_files");
            config.species_file = toml::find_or<std::string>(files, "species", "species.dat");
            config.couplings_file = toml::find_or<std::string>(files, "couplings", "couplings.dat");
            config.kugel_khomskii_file = toml::find_or<std::string>(files, "kugel_khomskii", "");
        } else {
            config.species_file = "species.dat";
            config.couplings_file = "couplings.dat";
            config.kugel_khomskii_file = "";
        }
        
    } catch (const toml::syntax_error& e) {
        throw ConfigurationError("TOML syntax error in " + toml_file + ": " + e.what());
    } catch (const toml::type_error& e) {
        throw ConfigurationError("TOML type error in " + toml_file + ": " + e.what());
    } catch (const std::exception& e) {
        throw ConfigurationError("Error parsing TOML file " + toml_file + ": " + e.what());
    }
}

std::vector<MagneticSpecies> ConfigurationParser::parse_species_file(const std::string& species_file) {
    std::vector<MagneticSpecies> species;
    
    std::ifstream file(species_file);
    if (!file.is_open()) {
        throw ConfigurationError("Cannot open species file: " + species_file);
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string name, type_str;
        double x, y, z;
        
        if (!(iss >> name >> type_str >> x >> y >> z)) {
            throw ConfigurationError("Invalid species file format at line " + 
                                     std::to_string(line_number) + ": " + line);
        }
        
        SpinType spin_type = string_to_spin_type(type_str);
        species.emplace_back(name, spin_type, x, y, z);
    }
    
    if (species.empty()) {
        throw ConfigurationError("No species found in file: " + species_file);
    }
    
    return species;
}

std::vector<ExchangeCoupling> ConfigurationParser::parse_couplings_file(const std::string& couplings_file) {
    std::vector<ExchangeCoupling> couplings;
    
    std::ifstream file(couplings_file);
    if (!file.is_open()) {
        throw ConfigurationError("Cannot open couplings file: " + couplings_file);
    }
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string species1, species2;
        int rx, ry, rz;
        double J;
        
        if (!(iss >> species1 >> species2 >> rx >> ry >> rz >> J)) {
            throw ConfigurationError("Invalid couplings file format at line " + 
                                     std::to_string(line_number) + ": " + line);
        }
        
        couplings.emplace_back(species1, species2, rx, ry, rz, J);
    }
    
    if (couplings.empty()) {
        throw ConfigurationError("No couplings found in file: " + couplings_file);
    }
    
    return couplings;
}

void ConfigurationParser::validate_configuration(const SimulationConfig& config) {
    // Check that all coupling species exist in the species list
    for (const auto& coupling : config.couplings) {
        bool found_species1 = false, found_species2 = false;
        
        for (const auto& species : config.species) {
            if (species.name == coupling.species1_name) found_species1 = true;
            if (species.name == coupling.species2_name) found_species2 = true;
        }
        
        if (!found_species1) {
            throw ConfigurationError("Species '" + coupling.species1_name + 
                                     "' in couplings not found in species file");
        }
        if (!found_species2) {
            throw ConfigurationError("Species '" + coupling.species2_name + 
                                     "' in couplings not found in species file");
        }
    }
    
    // Note: Mixed Ising-Heisenberg couplings are now supported
    // No need to check for mixed couplings - the simulation engine handles them
    
    // Validate parameter ranges
    if (config.lattice_size <= 0) {
        throw ConfigurationError("Lattice size must be positive");
    }
    
    if (config.monte_carlo.warmup_steps <= 0 || config.monte_carlo.measurement_steps <= 0) {
        throw ConfigurationError("Monte Carlo steps must be positive");
    }
    
    if (config.temperature.type == TemperatureConfig::SCAN) {
        if (config.temperature.max_temp <= config.temperature.min_temp) {
            throw ConfigurationError("Maximum temperature must be greater than minimum");
        }
        if (config.temperature.temp_step <= 0) {
            throw ConfigurationError("Temperature step must be positive");
        }
    }
}

SpinType ConfigurationParser::string_to_spin_type(const std::string& type_str) {
    std::string lower_type = type_str;
    std::transform(lower_type.begin(), lower_type.end(), lower_type.begin(), ::tolower);
    
    if (lower_type == "ising") {
        return SpinType::ISING;
    } else if (lower_type == "heisenberg") {
        return SpinType::HEISENBERG;
    } else {
        throw ConfigurationError("Unknown spin type: " + type_str + 
                                 " (must be 'Ising' or 'Heisenberg')");
    }
}

void ConfigurationParser::check_mixed_couplings(const std::vector<MagneticSpecies>& species,
                                                 const std::vector<ExchangeCoupling>& couplings) {
    // Mixed couplings are now fully supported - this check is disabled
    (void)species;   // Suppress unused parameter warning
    (void)couplings; // Suppress unused parameter warning
    // No validation needed - simulation engine handles all spin type combinations
}

} // namespace IO