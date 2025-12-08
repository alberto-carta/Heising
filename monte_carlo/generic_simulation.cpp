/*
 * Generic Monte Carlo Simulation Program
 * 
 * Reads configuration from TOML files and performs Monte Carlo
 * simulations on arbitrary magnetic systems
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h" 
#include "../include/random.h"
#include "../include/io/configuration_parser.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

// Global random seed (will be set from configuration)
long int seed = -12345;

/**
 * Convert configuration data to simulation objects
 */
UnitCell create_unit_cell_from_config(const std::vector<IO::MagneticSpecies>& species) {
    UnitCell cell;
    
    for (const auto& spec : species) {
        // Use the UnitCell::add_spin method with proper signature
        cell.add_spin(spec.name, spec.spin_type, 1.0);  // Default magnitude = 1.0
        // Position information is stored but not currently used in simulation
        // Could be extended for more complex lattice structures
    }
    
    return cell;
}

/**
 * Create coupling matrix from configuration with range checking
 */
CouplingMatrix create_couplings_from_config(const std::vector<IO::ExchangeCoupling>& couplings,
                                            const std::vector<IO::MagneticSpecies>& species,
                                            int lattice_size) {
    int num_spins = species.size();
    
    // Determine maximum coupling offset from input
    int max_offset = 1;  // Default to nearest neighbor
    for (const auto& coupling : couplings) {
        int max_abs = std::max({std::abs(coupling.cell_offset[0]),
                                std::abs(coupling.cell_offset[1]),
                                std::abs(coupling.cell_offset[2])});
        max_offset = std::max(max_offset, max_abs);
    }
    
    CouplingMatrix coupling_matrix;
    coupling_matrix.initialize(num_spins, max_offset);
    
    // Create map of species names to spin indices
    std::map<std::string, int> species_to_index;
    for (size_t i = 0; i < species.size(); i++) {
        species_to_index[species[i].name] = static_cast<int>(i);
    }
    
    // Add all couplings with range checking
    int truncated_couplings = 0;
    for (const auto& coupling : couplings) {
        int atom1_id = species_to_index[coupling.species1_name];
        int atom2_id = species_to_index[coupling.species2_name];
        
        // Check if coupling extends beyond reasonable simulation range
        int max_abs = std::max({std::abs(coupling.cell_offset[0]),
                                std::abs(coupling.cell_offset[1]),
                                std::abs(coupling.cell_offset[2])});
        
        if (max_abs >= lattice_size / 2) {
            std::cout << "WARNING: Coupling " << coupling.species1_name << "-" << coupling.species2_name
                      << " at offset (" << coupling.cell_offset[0] << "," << coupling.cell_offset[1] 
                      << "," << coupling.cell_offset[2] << ") extends beyond half lattice size ("
                      << lattice_size/2 << "). Truncating." << std::endl;
            truncated_couplings++;
            continue;
        }
        
        // Add coupling for specified unit cell offset
        coupling_matrix.set_coupling(atom1_id, atom2_id, 
                                     coupling.cell_offset[0],
                                     coupling.cell_offset[1], 
                                     coupling.cell_offset[2],
                                     coupling.J);
    }
    
    if (truncated_couplings > 0) {
        std::cout << "WARNING: " << truncated_couplings << " couplings were truncated due to range limits." << std::endl;
    }
    
    return coupling_matrix;
}

/**
 * Run single temperature simulation
 */
void run_single_temperature(const IO::SimulationConfig& config) {
    std::cout << "Running single temperature simulation at T = " << config.temperature.value << std::endl;
    
    // Create simulation objects from configuration
    UnitCell unit_cell = create_unit_cell_from_config(config.species);
    CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, config.lattice_size);
    
    MonteCarloSimulation sim(unit_cell, couplings, config.lattice_size, config.temperature.value);
    sim.initialize_lattice();
    
    int total_spins = config.lattice_size * config.lattice_size * config.lattice_size * config.species.size();
    
    // Warmup
    std::cout << "Warmup phase..." << std::endl;
    for (int sweep = 0; sweep < config.monte_carlo.warmup_steps; sweep++) {
        for (int attempt = 0; attempt < total_spins; attempt++) {
            sim.run_monte_carlo_step();
        }
    }
    
    // Measurement
    std::cout << "Measurement phase..." << std::endl;
    sim.reset_statistics();
    
    double total_energy = 0.0;
    double total_energy_sq = 0.0;
    double total_magnetization = 0.0;
    double total_abs_magnetization = 0.0;
    double total_magnetization_sq = 0.0;
    int num_samples = 0;
    
    for (int sweep = 0; sweep < config.monte_carlo.measurement_steps; sweep++) {
        for (int attempt = 0; attempt < total_spins; attempt++) {
            sim.run_monte_carlo_step();
        }
        
        if (sweep % config.monte_carlo.sampling_frequency == 0) {
            double energy = sim.get_energy();
            double magnetization = sim.get_magnetization();
            double abs_magnetization = sim.get_absolute_magnetization();
            
            total_energy += energy;
            total_energy_sq += energy * energy;
            total_magnetization += magnetization;
            total_abs_magnetization += abs_magnetization;
            total_magnetization_sq += magnetization * magnetization;
            num_samples++;
        }
    }
    
    // Calculate results
    double avg_energy_per_spin = total_energy / num_samples / total_spins;
    double avg_energy_sq_per_spin = total_energy_sq / num_samples / (total_spins * total_spins);
    double avg_magnetization_per_spin = total_magnetization / num_samples / total_spins;
    double avg_abs_magnetization_per_spin = total_abs_magnetization / num_samples / total_spins;
    double avg_magnetization_sq_per_spin = total_magnetization_sq / num_samples / (total_spins * total_spins);
    
    double specific_heat = (avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (config.temperature.value * config.temperature.value);
    double susceptibility = (avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / config.temperature.value;
    double accept_rate = sim.get_acceptance_rate();
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Results:" << std::endl;
    std::cout << "  Energy per spin: " << avg_energy_per_spin << std::endl;
    std::cout << "  Magnetization per spin: " << avg_magnetization_per_spin << std::endl;
    std::cout << "  |Magnetization| per spin: " << avg_abs_magnetization_per_spin << std::endl;
    std::cout << "  Specific heat per spin: " << specific_heat << std::endl;
    std::cout << "  Susceptibility per spin: " << susceptibility << std::endl;
    std::cout << "  Acceptance rate: " << std::setprecision(6) << accept_rate << std::endl;
}

/**
 * Configuration storage for temperature continuity
 */
struct ConfigurationSnapshot {
    std::vector<double> ising_spins;           // Ising spin values 
    std::vector<spin3d> heisenberg_spins;      // Heisenberg spin vectors
    
    void clear() {
        ising_spins.clear();
        heisenberg_spins.clear();
    }
};

/**
 * Save current simulation configuration for all species
 */
void save_configuration(const MonteCarloSimulation& sim, 
                       const std::vector<IO::MagneticSpecies>& species,
                       int lattice_size,
                       ConfigurationSnapshot& snapshot) {
    snapshot.clear();
    
    int total_spins_per_cell = species.size();
    int total_cells = lattice_size * lattice_size * lattice_size;
    
    // Count how many of each type we need
    int ising_count = 0, heisenberg_count = 0;
    for (const auto& spec : species) {
        if (spec.spin_type == SpinType::ISING) ising_count++;
        else heisenberg_count++;
    }
    
    // Pre-allocate storage
    snapshot.ising_spins.reserve(total_cells * ising_count);
    snapshot.heisenberg_spins.reserve(total_cells * heisenberg_count);
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int atom_id = 0; atom_id < total_spins_per_cell; atom_id++) {
                    if (species[atom_id].spin_type == SpinType::ISING) {
                        snapshot.ising_spins.push_back(sim.get_ising_spin(x, y, z, atom_id));
                    } else {
                        spin3d heisenberg_spin = sim.get_heisenberg_spin(x, y, z, atom_id);
                        snapshot.heisenberg_spins.push_back(heisenberg_spin);
                    }
                }
            }
        }
    }
}

/**
 * Load configuration into simulation for all species
 */
void load_configuration(MonteCarloSimulation& sim,
                       const std::vector<IO::MagneticSpecies>& species,
                       int lattice_size,
                       const ConfigurationSnapshot& snapshot) {
    int total_spins_per_cell = species.size();
    
    int ising_index = 0;
    int heisenberg_index = 0;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int atom_id = 0; atom_id < total_spins_per_cell; atom_id++) {
                    if (species[atom_id].spin_type == SpinType::ISING) {
                        sim.set_ising_spin(x, y, z, atom_id, snapshot.ising_spins[ising_index]);
                        ising_index++;
                    } else {
                        const spin3d& heisenberg_spin = snapshot.heisenberg_spins[heisenberg_index];
                        sim.set_heisenberg_spin(x, y, z, atom_id, heisenberg_spin);
                        heisenberg_index++;
                    }
                }
            }
        }
    }
}

/**
 * Run temperature scan simulation
 */
void run_temperature_scan(const IO::SimulationConfig& config) {
    std::cout << "Running temperature scan simulation" << std::endl;
    std::cout << "Temperature range: " << config.temperature.max_temp << " to " 
              << config.temperature.min_temp << " (step: " << config.temperature.temp_step << ")" << std::endl;
    
    // Create simulation objects from configuration
    UnitCell unit_cell = create_unit_cell_from_config(config.species);
    CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, config.lattice_size);
    
    int total_spins = config.lattice_size * config.lattice_size * config.lattice_size * config.species.size();
    
    // Determine output file name based on system type
    std::string output_file = config.output.directory + "/" + config.output.base_name + "_";
    
    // Simple heuristic to determine system type for file naming
    bool has_ising = false, has_heisenberg = false;
    for (const auto& species : config.species) {
        if (species.spin_type == SpinType::ISING) has_ising = true;
        if (species.spin_type == SpinType::HEISENBERG) has_heisenberg = true;
    }
    
    if (has_ising && has_heisenberg) {
        output_file += "mixed_system.dat";
    } else if (has_ising) {
        output_file += "ising_system.dat";
    } else {
        output_file += "heisenberg_system.dat";
    }
    
    std::ofstream outfile(output_file);
    outfile << "# Monte Carlo simulation results" << std::endl;
    outfile << "# System: ";
    for (const auto& species : config.species) {
        outfile << species.name << "(" << (species.spin_type == SpinType::ISING ? "Ising" : "Heisenberg") << ") ";
    }
    outfile << std::endl;
    outfile << "# Lattice: " << config.lattice_size << "³" << std::endl;
    outfile << "# Columns: T          Energy/spin   Magnetization |Magnetization| SpecificHeat  Susceptibility AcceptanceRate" << std::endl;
    outfile << std::fixed << std::setprecision(8);
    
    std::cout << "T          Energy/spin   Magnetization |Magnetization| SpecificHeat  Susceptibility AcceptanceRate" << std::endl;
    std::cout << "---------- ------------- ------------- ------------- ------------- -------------- --------------" << std::endl;
    
    // Storage for configuration continuity
    ConfigurationSnapshot saved_config;
    bool first_temperature = true;
    
    for (double T = config.temperature.max_temp; T >= config.temperature.min_temp; T -= config.temperature.temp_step) {
        std::cout << "\\nT = " << std::fixed << std::setprecision(2) << T << std::endl;
        
        MonteCarloSimulation sim(unit_cell, couplings, config.lattice_size, T);
        
        if (first_temperature) {
            std::cout << "  Initializing with ferromagnetic ground state (first temperature)..." << std::endl;
            sim.initialize_lattice();
            first_temperature = false;
        } else {
            std::cout << "  Loading configuration from previous temperature..." << std::endl;
            sim.initialize_lattice();  // Initialize arrays first
            load_configuration(sim, config.species, config.lattice_size, saved_config);
            std::cout << "  Configuration loaded successfully." << std::endl;
        }
        
        // Warmup
        for (int sweep = 0; sweep < config.monte_carlo.warmup_steps; sweep++) {
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            if ((sweep + 1) % 1000 == 0) {
                std::cout << "  Warmup: " << (sweep + 1) << "/" << config.monte_carlo.warmup_steps 
                          << " (" << std::setprecision(1) << (100.0 * (sweep + 1)) / config.monte_carlo.warmup_steps << "%)" << std::endl;
            }
        }
        
        // Measurement
        sim.reset_statistics();
        double total_energy = 0.0, total_energy_sq = 0.0;
        double total_magnetization = 0.0, total_abs_magnetization = 0.0, total_magnetization_sq = 0.0;
        int num_samples = 0;
        
        for (int sweep = 0; sweep < config.monte_carlo.measurement_steps; sweep++) {
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            
            if (sweep % config.monte_carlo.sampling_frequency == 0) {
                double energy = sim.get_energy();
                double magnetization = sim.get_magnetization();
                double abs_magnetization = sim.get_absolute_magnetization();
                
                total_energy += energy;
                total_energy_sq += energy * energy;
                total_magnetization += magnetization;
                total_abs_magnetization += abs_magnetization;
                total_magnetization_sq += magnetization * magnetization;
                num_samples++;
            }
            
            if ((sweep + 1) % 10000 == 0) {
                std::cout << "  Measurement: " << (sweep + 1) << "/" << config.monte_carlo.measurement_steps 
                          << " (" << std::setprecision(1) << (100.0 * (sweep + 1)) / config.monte_carlo.measurement_steps << "%)" << std::endl;
            }
        }
        
        // Calculate results
        double avg_energy_per_spin = total_energy / num_samples / total_spins;
        double avg_energy_sq_per_spin = total_energy_sq / num_samples / (total_spins * total_spins);
        double avg_magnetization_per_spin = total_magnetization / num_samples / total_spins;
        double avg_abs_magnetization_per_spin = total_abs_magnetization / num_samples / total_spins;
        double avg_magnetization_sq_per_spin = total_magnetization_sq / num_samples / (total_spins * total_spins);
        
        double specific_heat = (avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (T * T);
        double susceptibility = (avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / T;
        double accept_rate = sim.get_acceptance_rate();
        
        // Output
        std::cout << std::fixed << std::setprecision(8);
        std::cout << std::setw(10) << T << " "
                  << std::setw(13) << avg_energy_per_spin << " "
                  << std::setw(13) << avg_magnetization_per_spin << " "
                  << std::setw(13) << avg_abs_magnetization_per_spin << " "
                  << std::setw(13) << specific_heat << " "
                  << std::setw(14) << susceptibility << " "
                  << std::setw(14) << std::setprecision(6) << accept_rate << std::endl;
        
        outfile << std::fixed << std::setprecision(8);
        outfile << std::setw(10) << T << " "
                << std::setw(13) << avg_energy_per_spin << " "
                << std::setw(13) << avg_magnetization_per_spin << " "
                << std::setw(13) << avg_abs_magnetization_per_spin << " "
                << std::setw(13) << specific_heat << " "
                << std::setw(14) << susceptibility << " "
                << std::setw(14) << std::setprecision(6) << accept_rate << std::endl;
        
        // Save configuration for next temperature step
        std::cout << "  Saving configuration for next temperature step..." << std::endl;
        save_configuration(sim, config.species, config.lattice_size, saved_config);
    }
    
    outfile.close();
    std::cout << "\\nResults saved to: " << output_file << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "    Generic Monte Carlo Simulation     " << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parse command line arguments
    std::string config_file = "simulation.toml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    try {
        // Load configuration
        std::cout << "Loading configuration from: " << config_file << std::endl;
        IO::SimulationConfig config = IO::ConfigurationParser::load_configuration(config_file);
        
        // Set global random seed
        seed = config.monte_carlo.seed;
        
        // Print configuration summary
        std::cout << "\\nConfiguration summary:" << std::endl;
        std::cout << "  Lattice size: " << config.lattice_size << "³" << std::endl;
        std::cout << "  Species: ";
        for (const auto& species : config.species) {
            std::cout << species.name << "(" << (species.spin_type == SpinType::ISING ? "I" : "H") << ") ";
        }
        std::cout << std::endl;
        std::cout << "  Couplings: " << config.couplings.size() << " exchange interactions" << std::endl;
        std::cout << "  Monte Carlo: " << config.monte_carlo.warmup_steps << " warmup + " 
                  << config.monte_carlo.measurement_steps << " measurement steps" << std::endl;
        std::cout << std::endl;
        
        // Run simulation based on type
        if (config.temperature.type == IO::TemperatureConfig::SINGLE) {
            run_single_temperature(config);
        } else {
            run_temperature_scan(config);
        }
        
    } catch (const IO::ConfigurationError& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\\nSimulation completed successfully!" << std::endl;
    return 0;
}