/*
 * Configuration Parser for Monte Carlo Simulations
 * 
 * TOML Configuration Reference:
 * =============================
 * 
 * [simulation]
 *   type = "temperature_scan"           # Simulation type ("single_temperature" or "temperature_scan", default: "temperature_scan")
 *   seed = -12345                       # Random number seed (default: -12345)
 * 
 * [lattice]
 *   size = 8                            # Lattice size (cubic lattice: L×L×L)
 * 
 * [monte_carlo]
 *   warmup_steps = 8000                 # Number of warmup sweeps
 *   measurement_steps = 80000           # Number of measurement sweeps
 *   sampling_frequency = 100            # Sample every N sweeps
 * 
 * [temperature]
 *   # For single temperature:
 *   value = 2.0                         # Single temperature value
 *   
 *   # For temperature scan (default):
 *   max = 6.0                           # Maximum temperature
 *   min = 0.5                           # Minimum temperature
 *   step = 0.2                          # Temperature step size
 *   restart_from_previous_T = false     # If true, start each T from previous T's final config (default: false)
 * 
 * [output]
 *   base_name = "simulation"            # Output file base name
 *   directory = "."                     # Output directory
 *   
 *   # Observable output options (all default to false):
 *   output_energy_total = false         # Output total energy (in addition to per-spin)
 *   output_onsite_magnetization = false # Output on-site magnetization for each species
 *   output_correlations = true          # Output spin correlations with first spin (default: true if not specified)
 * 
 * [input_files]
 *   species = "species.dat"             # Species definitions file
 *   couplings = "couplings.dat"         # Exchange couplings file
 *   kugel_khomskii = ""                 # Optional: Kugel-Khomskii interactions file
 * 
 * [initialization]
 *   type = "random"                     # Initialization type: "random", "custom", or "file"
 *   pattern = [1, -1, 1, -1]            # For type="custom": values for each spin in unit cell
 *                                       # Ising: +1/-1, Heisenberg: sz component (-1 to +1)
 *   initialization_file_path = "config.dat"  # For type="file": path to configuration file
 *   random_seed = -1                    # Optional: separate seed for initialization (default: use main seed)
 * 
 * [slab_tunnel]
 *   enabled = false                     # Enable slab tunnel move (default: false)
 *   pattern1 = [1, 1, -1, -1, 1, -1, 1, -1]   # First pattern (e.g., C-AFM+G-OO)
 *   pattern2 = [1, -1, 1, -1, 1, 1, -1, -1]   # Second pattern (e.g., G-AFM+C-OO)
 *   lateral_size = 11                   # Lateral size of slab (N×N)
 *   thickness = 1                       # Thickness of slab in z-direction
 *   burst_interval = 100                # Attempt slab moves every N MC sweeps
 *   burst_attempts = 50                 # Number of slab move attempts per burst
 *   debug = false                       # Enable debug output for slab tunnel moves
 * 
 * [diagnostics]
 *   enable_profiling = false            # Enable timing/performance profiling
 *   enable_config_dump = false          # Enable configuration snapshots
 *   enable_observable_evolution = false # Enable per-measurement observable tracking
 *   dump_initial_config = false         # Dump initial configuration after warmup
 *   
 *   dump_ranks = [0, 1]                 # Ranks to dump: array of integers, "all", or "none"
 *   dump_every_n_measurements = 10      # Dump configuration every N measurements
 *   dump_format = "text"                # Format: "text" (only text currently supported)
 *   estimate_autocorrelation = true     # Estimate autocorrelation times at end of each T
 *   
 *   # Performance options:
 *   recompute_observables_each_sample = false  # If true: recompute energy/magnetization from scratch each sample (slow but accurate)
 *                                              # If false (default): use incremental tracking (fast, recommended)
 *                                              # Note: Correlations always require full computation regardless of this setting
 * 
 * Species File Format (species.dat):
 *   # name  type  x  y  z
 *   Fe  Heisenberg  0.0  0.0  0.0
 *   Ni  Ising       0.5  0.5  0.5
 * 
 * Couplings File Format (couplings.dat):
 *   # species1  species2  Rx  Ry  Rz  J
 *   Fe  Fe   1   0   0   -1.0
 *   Fe  Ni   0   0   0    0.5
 */

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
        
        // Parse KK file if specified
        if (!config.kugel_khomskii_file.empty()) {
            std::string kk_path = (config.kugel_khomskii_file[0] == '/') ? config.kugel_khomskii_file : toml_dir + "/" + config.kugel_khomskii_file;
            config.kk_couplings = parse_kk_file(kk_path);
        }
        
        // Validate the complete configuration
        validate_configuration(config);
        
        std::cout << "Configuration loaded successfully:" << std::endl;
        std::cout << "  - " << config.species.size() << " magnetic species" << std::endl;
        std::cout << "  - " << config.couplings.size() << " exchange couplings" << std::endl;
        if (!config.kk_couplings.empty()) {
            std::cout << "  - " << config.kk_couplings.size() << " Kugel-Khomskii couplings" << std::endl;
        }
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
            
            // Validate simulation type
            if (config.simulation_type != "single_temperature" && config.simulation_type != "temperature_scan") {
                throw ConfigurationError("Invalid simulation type: '" + config.simulation_type + 
                                       "'. Allowed values: 'single_temperature', 'temperature_scan'");
            }
            
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
            
            // Parse restart_from_previous_T flag (applies to temperature scan)
            config.temperature.restart_from_previous_T = toml::find_or<bool>(temp, "restart_from_previous_T", false);
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
            
            // Parse observable output options
            config.output.output_energy_total = toml::find_or<bool>(output, "output_energy_total", false);
            config.output.output_onsite_magnetization = toml::find_or<bool>(output, "output_onsite_magnetization", false);
            config.output.output_correlations = toml::find_or<bool>(output, "output_correlations", false);
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
        
        // Parse diagnostics section (optional)
        if (data.contains("diagnostics")) {
            const auto diag = toml::find(data, "diagnostics");
            config.diagnostics.enable_profiling = toml::find_or<bool>(diag, "enable_profiling", false);
            config.diagnostics.enable_config_dump = toml::find_or<bool>(diag, "enable_config_dump", false);
            config.diagnostics.enable_observable_evolution = toml::find_or<bool>(diag, "enable_observable_evolution", false);
            config.diagnostics.dump_initial_config = toml::find_or<bool>(diag, "dump_initial_config", false);
            
            config.diagnostics.dump_every_n_measurements = toml::find_or<int>(diag, "dump_every_n_measurements", 10);
            config.diagnostics.dump_format = toml::find_or<std::string>(diag, "dump_format", "text");
            config.diagnostics.estimate_autocorrelation = toml::find_or<bool>(diag, "estimate_autocorrelation", true);
            config.diagnostics.recompute_observables_each_sample = toml::find_or<bool>(diag, "recompute_observables_each_sample", false);
            
            // Parse dump_ranks - can be "all", "none", or array of integers
            if (diag.contains("dump_ranks")) {
                const auto& dump_ranks_value = toml::find(diag, "dump_ranks");
                
                if (dump_ranks_value.is_string()) {
                    std::string dump_ranks_str = dump_ranks_value.as_string();
                    if (dump_ranks_str == "all") {
                        config.diagnostics.dump_all_ranks = true;
                    } else if (dump_ranks_str == "none") {
                        config.diagnostics.dump_all_ranks = false;
                        config.diagnostics.dump_ranks.clear();
                    } else {
                        throw ConfigurationError("Invalid dump_ranks string: " + dump_ranks_str + 
                                               " (expected 'all' or 'none')");
                    }
                } else if (dump_ranks_value.is_array()) {
                    const auto& ranks_array = dump_ranks_value.as_array();
                    for (const auto& rank : ranks_array) {
                        config.diagnostics.dump_ranks.push_back(rank.as_integer());
                    }
                } else {
                    throw ConfigurationError("dump_ranks must be a string ('all'/'none') or array of integers");
                }
            }
        }
        
        // Parse initialization section (optional)
        if (data.contains("initialization")) {
            const auto init = toml::find(data, "initialization");
            
            // Parse type
            std::string type_str = toml::find_or<std::string>(init, "type", "random");
            if (type_str == "random") {
                config.initialization.type = InitializationConfig::RANDOM;
            } else if (type_str == "custom") {
                config.initialization.type = InitializationConfig::CUSTOM;
            } else if (type_str == "file") {
                config.initialization.type = InitializationConfig::FILE;
            } else {
                throw ConfigurationError("Invalid initialization type: " + type_str + 
                                       " (expected 'random', 'custom', or 'file')");
            }
            
            // Parse pattern if provided
            if (init.contains("pattern")) {
                const auto& pattern_value = toml::find(init, "pattern");
                if (pattern_value.is_array()) {
                    const auto& pattern_array = pattern_value.as_array();
                    for (const auto& val : pattern_array) {
                        if (val.is_floating()) {
                            config.initialization.pattern.push_back(val.as_floating());
                        } else if (val.is_integer()) {
                            config.initialization.pattern.push_back(static_cast<double>(val.as_integer()));
                        }
                    }
                } else {
                    throw ConfigurationError("initialization.pattern must be an array of numbers");
                }
            }
            
            // Parse initialization_file_path if provided
            if (init.contains("initialization_file_path")) {
                config.initialization.initialization_file_path = toml::find<std::string>(init, "initialization_file_path");
            }
            
            // Parse random seed if provided
            config.initialization.random_seed = toml::find_or<long int>(init, "random_seed", -1);
        }
        
        // Parse slab_tunnel section (optional)
        if (data.contains("slab_tunnel")) {
            const auto slab = toml::find(data, "slab_tunnel");
            
            config.slab_tunnel.enabled = toml::find_or<bool>(slab, "enabled", false);
            config.slab_tunnel.lateral_size = toml::find_or<int>(slab, "lateral_size", 1);
            config.slab_tunnel.thickness = toml::find_or<int>(slab, "thickness", 1);
            config.slab_tunnel.burst_interval = toml::find_or<int>(slab, "burst_interval", 100);
            config.slab_tunnel.burst_attempts = toml::find_or<int>(slab, "burst_attempts", 50);
            config.slab_tunnel.debug = toml::find_or<bool>(slab, "debug", false);
            
            // Parse pattern1 if provided
            if (slab.contains("pattern1")) {
                const auto& pattern1_value = toml::find(slab, "pattern1");
                if (pattern1_value.is_array()) {
                    const auto& pattern_array = pattern1_value.as_array();
                    for (const auto& val : pattern_array) {
                        if (val.is_floating()) {
                            config.slab_tunnel.pattern1.push_back(val.as_floating());
                        } else if (val.is_integer()) {
                            config.slab_tunnel.pattern1.push_back(static_cast<double>(val.as_integer()));
                        }
                    }
                } else {
                    throw ConfigurationError("slab_tunnel.pattern1 must be an array of numbers");
                }
            }
            
            // Parse pattern2 if provided
            if (slab.contains("pattern2")) {
                const auto& pattern2_value = toml::find(slab, "pattern2");
                if (pattern2_value.is_array()) {
                    const auto& pattern_array = pattern2_value.as_array();
                    for (const auto& val : pattern_array) {
                        if (val.is_floating()) {
                            config.slab_tunnel.pattern2.push_back(val.as_floating());
                        } else if (val.is_integer()) {
                            config.slab_tunnel.pattern2.push_back(static_cast<double>(val.as_integer()));
                        }
                    }
                } else {
                    throw ConfigurationError("slab_tunnel.pattern2 must be an array of numbers");
                }
            }
            
            // Validate that if enabled, both patterns must be provided
            if (config.slab_tunnel.enabled) {
                if (config.slab_tunnel.pattern1.empty() || config.slab_tunnel.pattern2.empty()) {
                    throw ConfigurationError("slab_tunnel enabled but pattern1 or pattern2 not provided");
                }
                if (config.slab_tunnel.pattern1.size() != config.slab_tunnel.pattern2.size()) {
                    throw ConfigurationError("slab_tunnel.pattern1 and pattern2 must have the same size");
                }
            }
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

std::vector<KKCoupling> ConfigurationParser::parse_kk_file(const std::string& kk_file) {
    std::vector<KKCoupling> kk_couplings;
    
    std::ifstream file(kk_file);
    if (!file.is_open()) {
        throw ConfigurationError("Cannot open Kugel-Khomskii file: " + kk_file);
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
        double K;
        
        if (!(iss >> species1 >> species2 >> rx >> ry >> rz >> K)) {
            throw ConfigurationError("Invalid KK file format at line " + 
                                     std::to_string(line_number) + ": " + line);
        }
        
        kk_couplings.emplace_back(species1, species2, rx, ry, rz, K);
    }
    
    if (kk_couplings.empty()) {
        throw ConfigurationError("No KK couplings found in file: " + kk_file);
    }
    
    std::cout << "  Loaded " << kk_couplings.size() << " Kugel-Khomskii couplings from " << kk_file << std::endl;
    
    return kk_couplings;
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
    
    // Check that all KK coupling species exist in the species list
    for (const auto& kk : config.kk_couplings) {
        bool found_species1 = false, found_species2 = false;
        
        for (const auto& species : config.species) {
            if (species.name == kk.species1_name) found_species1 = true;
            if (species.name == kk.species2_name) found_species2 = true;
        }
        
        if (!found_species1) {
            throw ConfigurationError("Species '" + kk.species1_name + 
                                     "' in KK couplings not found in species file");
        }
        if (!found_species2) {
            throw ConfigurationError("Species '" + kk.species2_name + 
                                     "' in KK couplings not found in species file");
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
    
    // Validate initialization configuration
    if (config.initialization.type == InitializationConfig::FILE) {
        if (config.initialization.initialization_file_path.empty()) {
            throw ConfigurationError("initialization.type='file' requires initialization_file_path to be set");
        }
    }
    
    if (config.initialization.type == InitializationConfig::CUSTOM) {
        if (config.initialization.pattern.empty()) {
            throw ConfigurationError("initialization.type='custom' requires pattern array to be set");
        }
        if (config.initialization.pattern.size() != config.species.size()) {
            throw ConfigurationError("initialization.pattern size (" + 
                                   std::to_string(config.initialization.pattern.size()) + 
                                   ") must match number of species (" + 
                                   std::to_string(config.species.size()) + ")");
        }
    }
    
    // Validate and provide warnings for diagnostic configuration
    if (config.diagnostics.dump_initial_config) {
        if (config.diagnostics.dump_ranks.empty() && !config.diagnostics.dump_all_ranks) {
            std::cout << "WARNING: dump_initial_config=true but no dump_ranks specified." << std::endl;
            std::cout << "         No configurations will be dumped." << std::endl;
            std::cout << "         Set dump_ranks=[0] or dump_ranks=\"all\" to enable dumping." << std::endl;
        }
    }
    
    if (config.diagnostics.enable_config_dump || config.diagnostics.enable_observable_evolution) {
        if (config.diagnostics.dump_ranks.empty() && !config.diagnostics.dump_all_ranks) {
            std::cout << "WARNING: Configuration dumps enabled but no dump_ranks specified." << std::endl;
            std::cout << "         No configurations will be dumped." << std::endl;
            std::cout << "         Set dump_ranks=[0] or dump_ranks=\"all\" to enable dumping." << std::endl;
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