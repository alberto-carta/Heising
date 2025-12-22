/*
 * Heising Main - MPI Monte Carlo Simulation Program
 * 
 * Reads configuration from TOML files and performs Monte Carlo
 * simulations on arbitrary magnetic systems using MPI parallelization
 * 
 * Requires MPI for execution
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h" 
#include "../include/random.h"
#include "../include/io/configuration_parser.h"
#include "../include/mpi_wrapper.h"

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
        // Use position from config to automatically determine sites
        cell.add_spin(spec.name, spec.spin_type, 1.0, 
                     spec.local_pos[0], spec.local_pos[1], spec.local_pos[2]);
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
    std::cout << std::fixed << std::setprecision(4);
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
    double total_magnetization_sq = 0.0;
    int num_samples = 0;
    
    for (int sweep = 0; sweep < config.monte_carlo.measurement_steps; sweep++) {
        for (int attempt = 0; attempt < total_spins; attempt++) {
            sim.run_monte_carlo_step();
        }
        
        if (sweep % config.monte_carlo.sampling_frequency == 0) {
            double energy = sim.get_energy();
            double magnetization = sim.get_magnetization();
            
            total_energy += energy;
            total_energy_sq += energy * energy;
            total_magnetization += magnetization;
            total_magnetization_sq += magnetization * magnetization;
            num_samples++;
        }
    }
    
    // Calculate results
    double avg_energy_per_spin = total_energy / num_samples / total_spins;
    double avg_energy_sq_per_spin = total_energy_sq / num_samples / (total_spins * total_spins);
    double avg_magnetization_per_spin = total_magnetization / num_samples / total_spins;
    double avg_magnetization_sq_per_spin = total_magnetization_sq / num_samples / (total_spins * total_spins);
    
    double specific_heat = (avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (config.temperature.value * config.temperature.value);
    double susceptibility = (avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / config.temperature.value;
    double accept_rate = sim.get_acceptance_rate();
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Results:" << std::endl;
    std::cout << "  Energy per spin: " << avg_energy_per_spin << std::endl;
    std::cout << "  Magnetization per spin: " << avg_magnetization_per_spin << std::endl;
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
 * Average simulation configuration across all MPI ranks
 * This ensures all walkers start from the same averaged state at the next temperature
 */
void average_configuration_mpi(MonteCarloSimulation& sim,
                               const std::vector<IO::MagneticSpecies>& species,
                               MPIAccumulator& mpi_accumulator) {
    // Get mutable references to spin arrays
    auto& ising_array = sim.get_ising_array_mutable();
    auto& heis_x_array = sim.get_heisenberg_x_array_mutable();
    auto& heis_y_array = sim.get_heisenberg_y_array_mutable();
    auto& heis_z_array = sim.get_heisenberg_z_array_mutable();
    
    // Convert Eigen arrays to std::vector for MPI operations
    std::vector<double> ising_vec(ising_array.data(), ising_array.data() + ising_array.size());
    std::vector<double> heis_x_vec(heis_x_array.data(), heis_x_array.data() + heis_x_array.size());
    std::vector<double> heis_y_vec(heis_y_array.data(), heis_y_array.data() + heis_y_array.size());
    std::vector<double> heis_z_vec(heis_z_array.data(), heis_z_array.data() + heis_z_array.size());
    
    // Average across all ranks
    mpi_accumulator.average_configuration(ising_vec);
    mpi_accumulator.average_configuration_vectors(heis_x_vec, heis_y_vec, heis_z_vec);
    
    // Copy back to Eigen arrays
    for (size_t i = 0; i < ising_vec.size(); i++) {
        ising_array[i] = ising_vec[i];
    }
    for (size_t i = 0; i < heis_x_vec.size(); i++) {
        heis_x_array[i] = heis_x_vec[i];
        heis_y_array[i] = heis_y_vec[i];
        heis_z_array[i] = heis_z_vec[i];
    }
}

// Serial version removed - MPI is now required for all simulations

/**
 * MPI-parallel temperature scan simulation
 * Each rank runs an independent walker, results are accumulated on rank 0
 */
void run_temperature_scan(const IO::SimulationConfig& config,
                          MPIEnvironment& mpi_env,
                          MPIAccumulator& mpi_accumulator) {
    int rank = mpi_env.get_rank();
    int num_ranks = mpi_env.get_num_ranks();
    
    // Divide measurement steps among ranks to keep total samples constant
    int measurement_steps_per_rank = config.monte_carlo.measurement_steps / num_ranks;
    if (measurement_steps_per_rank == 0) {
        if (rank == 0) {
            std::cerr << "ERROR: measurement_steps (" << config.monte_carlo.measurement_steps 
                      << ") is less than num_ranks (" << num_ranks << ")" << std::endl;
            std::cerr << "Please increase measurement_steps or reduce number of MPI ranks." << std::endl;
        }
        return;
    }
    
    if (rank == 0) {
        std::cout << "Running MPI-parallel temperature scan with " << num_ranks << " walkers" << std::endl;
        std::cout << "Temperature range: " << config.temperature.max_temp << " to " 
                  << config.temperature.min_temp << " (step: " << config.temperature.temp_step << ")" << std::endl;
        std::cout << "Measurement steps per rank: " << measurement_steps_per_rank 
                  << " (total: " << measurement_steps_per_rank * num_ranks << ")" << std::endl;
    }
    
    // Create simulation objects from configuration (each rank creates its own)
    UnitCell unit_cell = create_unit_cell_from_config(config.species);
    CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, config.lattice_size);
    
    int total_spins = config.lattice_size * config.lattice_size * config.lattice_size * config.species.size();
    
    // Determine output file name (only rank 0 will write)
    std::string output_file;
    std::ofstream outfile;
    
    if (rank == 0) {
        output_file = config.output.directory + "/" + config.output.base_name + "_";
        
        bool has_ising = false, has_heisenberg = false;
        for (const auto& species : config.species) {
            if (species.spin_type == SpinType::ISING) has_ising = true;
            if (species.spin_type == SpinType::HEISENBERG) has_heisenberg = true;
        }
        
        output_file += "observables.out";
        
        outfile.open(output_file);
        outfile << "# Monte Carlo simulation results (MPI parallel, " << num_ranks << " walkers)" << std::endl;
        outfile << "# System: ";
        for (const auto& species : config.species) {
            outfile << species.name << "(" << (species.spin_type == SpinType::ISING ? "Ising" : "Heisenberg") << ") ";
        }
        outfile << std::endl;
        outfile << "# Lattice: " << config.lattice_size << "³" << std::endl;
        outfile << "# Columns: T Energy/spin Magnetization SpecificHeat Susceptibility AcceptanceRate";
        // Commented out: Mx, My, Mz per species (use correlations instead)
        // for (const auto& species : config.species) {
        //     outfile << " Mx[" << species.name << "] My[" << species.name << "] Mz[" << species.name << "]";
        // }
        // NOTE: Correlations are direction-independent and safe to average across walkers
        // Find first Ising and first Heisenberg spins for correlation labels
        std::string first_ising_name = "", first_heis_name = "";
        for (const auto& sp : config.species) {
            if (sp.spin_type == SpinType::ISING && first_ising_name.empty()) {
                first_ising_name = sp.name;
            }
            if (sp.spin_type == SpinType::HEISENBERG && first_heis_name.empty()) {
                first_heis_name = sp.name;
            }
        }
        for (size_t i = 0; i < config.species.size(); i++) {
            if (config.species[i].spin_type == SpinType::ISING) {
                outfile << " <" << first_ising_name << "*" << config.species[i].name << ">";
            } else {
                outfile << " <" << first_heis_name << "·" << config.species[i].name << ">";
            }
        }
        outfile << std::endl;
        outfile << std::fixed << std::setprecision(8);
        
        std::cout << "T          Energy/spin   Magnetization SpecificHeat  Susceptibility AcceptanceRate";
        // Commented out: Mx, My, Mz per species (use correlations instead)
        // for (const auto& species : config.species) {
        //     std::cout << " Mx[" << species.name << "] My[" << species.name << "] Mz[" << species.name << "]";
        // }
        for (size_t i = 0; i < config.species.size(); i++) {
            if (config.species[i].spin_type == SpinType::ISING) {
                std::cout << " <" << first_ising_name << "*" << config.species[i].name << ">";
            } else {
                std::cout << " <" << first_heis_name << "·" << config.species[i].name << ">";
            }
        }
        std::cout << std::endl;
        std::cout << "---------- ------------- ------------- ------------- -------------- --------------";
        // Commented out separator for Mx,My,Mz columns
        // for (size_t i = 0; i < config.species.size(); i++) {
        //     std::cout << " -------------- -------------- --------------";
        // }
        for (size_t i = 0; i < config.species.size(); i++) {
            std::cout << " --------------";
        }
        std::cout << std::endl;
    }
    
    bool first_temperature = true;
    
    // Temperature scan loop
    for (double T = config.temperature.max_temp; T >= config.temperature.min_temp; T -= config.temperature.temp_step) {
        if (rank == 0) {
            std::cout << "\\nT = " << std::fixed << std::setprecision(2) << T << std::endl;
        }
        
        // Each rank creates its own simulation with rank-specific seed
        long int rank_seed = get_rank_seed(config.monte_carlo.seed, rank);
        seed = rank_seed;
        
        MonteCarloSimulation sim(unit_cell, couplings, config.lattice_size, T);
        
        if (first_temperature) {
            if (rank == 0) {
                std::cout << "  Initializing all walkers with ferromagnetic ground state..." << std::endl;
            }
            sim.initialize_lattice();
            first_temperature = false;
        } else {
            // All ranks initialize and then average their configurations
            if (rank == 0) {
                std::cout << "  Averaging configurations from all walkers..." << std::endl;
            }
            sim.initialize_lattice();
            average_configuration_mpi(sim, config.species, mpi_accumulator);
            mpi_env.barrier();
            if (rank == 0) {
                std::cout << "  All walkers synchronized with averaged configuration." << std::endl;
            }
        }
        
        // Warmup - each rank runs independently
        for (int sweep = 0; sweep < config.monte_carlo.warmup_steps; sweep++) {
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            if (rank == 0 && (sweep + 1) % 1000 == 0) {
                std::cout << "  Warmup: " << (sweep + 1) << "/" << config.monte_carlo.warmup_steps 
                          << " (" << std::setprecision(1) << (100.0 * (sweep + 1)) / config.monte_carlo.warmup_steps << "%)" << std::endl;
            }
        }
        
        // Measurement phase - each rank accumulates local statistics
        // Each rank runs only its portion of the total measurement steps
        sim.reset_statistics();
        double local_energy = 0.0, local_energy_sq = 0.0;
        double local_magnetization = 0.0, local_magnetization_sq = 0.0;
        std::vector<spin3d> local_mag_vec_per_spin(config.species.size(), spin3d(0.0, 0.0, 0.0));
        std::vector<double> local_correlations(config.species.size(), 0.0);
        int num_samples = 0;
        
        for (int sweep = 0; sweep < measurement_steps_per_rank; sweep++) {
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            
            if (sweep % config.monte_carlo.sampling_frequency == 0) {
                double energy = sim.get_energy();
                double magnetization = sim.get_magnetization();
                std::vector<spin3d> mag_vectors = sim.get_magnetization_vector_per_spin();
                std::vector<double> correlations = sim.get_spin_correlation_with_first();
                
                local_energy += energy;
                local_energy_sq += energy * energy;
                local_magnetization += magnetization;
                local_magnetization_sq += magnetization * magnetization;
                for (size_t i = 0; i < mag_vectors.size(); i++) {
                    local_mag_vec_per_spin[i].x += mag_vectors[i].x;
                    local_mag_vec_per_spin[i].y += mag_vectors[i].y;
                    local_mag_vec_per_spin[i].z += mag_vectors[i].z;
                    local_correlations[i] += correlations[i];
                }
                num_samples++;
            }
            
            if (rank == 0 && (sweep + 1) % 5000 == 0) {
                std::cout << "  Measurement: " << (sweep + 1) << "/" << measurement_steps_per_rank 
                          << " (" << std::setprecision(1) << (100.0 * (sweep + 1)) / measurement_steps_per_rank << "%)" << std::endl;
            }
        }
        
        // Accumulate statistics across all ranks
        double global_energy = mpi_accumulator.accumulate_sum(local_energy);
        double global_energy_sq = mpi_accumulator.accumulate_sum(local_energy_sq);
        double global_magnetization = mpi_accumulator.accumulate_sum(local_magnetization);
        double global_magnetization_sq = mpi_accumulator.accumulate_sum(local_magnetization_sq);
        
        // For magnetization vectors, reduce x, y, z components separately
        std::vector<double> local_mag_x(config.species.size());
        std::vector<double> local_mag_y(config.species.size());
        std::vector<double> local_mag_z(config.species.size());
        for (size_t i = 0; i < config.species.size(); i++) {
            local_mag_x[i] = local_mag_vec_per_spin[i].x;
            local_mag_y[i] = local_mag_vec_per_spin[i].y;
            local_mag_z[i] = local_mag_vec_per_spin[i].z;
        }
        std::vector<double> global_mag_x = mpi_accumulator.accumulate_sum(local_mag_x);
        std::vector<double> global_mag_y = mpi_accumulator.accumulate_sum(local_mag_y);
        std::vector<double> global_mag_z = mpi_accumulator.accumulate_sum(local_mag_z);
        
        // Accumulate correlations (these are safe to average across walkers)
        std::vector<double> global_correlations = mpi_accumulator.accumulate_sum(local_correlations);
        
        double local_acceptance = sim.get_acceptance_rate();
        double global_acceptance = mpi_accumulator.accumulate_sum(local_acceptance);
        
        // Only rank 0 computes final statistics and outputs
        if (rank == 0) {
            int total_samples = num_samples * num_ranks;
            
            double avg_energy_per_spin = global_energy / total_samples / total_spins;
            double avg_energy_sq_per_spin = global_energy_sq / total_samples / (total_spins * total_spins);
            double avg_magnetization_per_spin = global_magnetization / total_samples / total_spins;
            double avg_magnetization_sq_per_spin = global_magnetization_sq / total_samples / (total_spins * total_spins);
            
            std::vector<spin3d> avg_mag_vec_per_spin(config.species.size());
            std::vector<double> avg_correlations(config.species.size());
            for (size_t i = 0; i < config.species.size(); i++) {
                avg_mag_vec_per_spin[i].x = global_mag_x[i] / total_samples;
                avg_mag_vec_per_spin[i].y = global_mag_y[i] / total_samples;
                avg_mag_vec_per_spin[i].z = global_mag_z[i] / total_samples;
                avg_correlations[i] = global_correlations[i] / total_samples;
            }
            
            double specific_heat = (avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (T * T);
            double susceptibility = (avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / T;
            double avg_accept_rate = global_acceptance / num_ranks;
            
            // Output to console
            std::cout << std::fixed << std::setprecision(8);
            std::cout << std::setw(10) << T << " "
                      << std::setw(13) << avg_energy_per_spin << " "
                      << std::setw(13) << avg_magnetization_per_spin << " "
                      << std::setw(13) << specific_heat << " "
                      << std::setw(14) << susceptibility << " "
                      << std::setw(14) << std::setprecision(6) << avg_accept_rate;
            // Commented out: Mx, My, Mz per species output (use correlations instead)
            // for (const auto& mag : avg_mag_vec_per_spin) {
            //     std::cout << " " << std::setw(14) << std::setprecision(8) << mag.x
            //               << " " << std::setw(14) << std::setprecision(8) << mag.y
            //               << " " << std::setw(14) << std::setprecision(8) << mag.z;
            // }
            for (const auto& corr : avg_correlations) {
                std::cout << " " << std::setw(14) << std::setprecision(8) << corr;
            }
            std::cout << std::endl;
            
            // Output to file
            outfile << std::fixed << std::setprecision(8);
            outfile << std::setw(10) << T << " "
                    << std::setw(13) << avg_energy_per_spin << " "
                    << std::setw(13) << avg_magnetization_per_spin << " "
                    << std::setw(13) << specific_heat << " "
                    << std::setw(14) << susceptibility << " "
                    << std::setw(14) << std::setprecision(6) << avg_accept_rate;
            // Commented out: Mx, My, Mz per species output (use correlations instead)
            // for (const auto& mag : avg_mag_vec_per_spin) {
            //     outfile << " " << std::setw(14) << std::setprecision(8) << mag.x
            //             << " " << std::setw(14) << std::setprecision(8) << mag.y
            //             << " " << std::setw(14) << std::setprecision(8) << mag.z;
            // }
            for (const auto& corr : avg_correlations) {
                outfile << " " << std::setw(14) << std::setprecision(8) << corr;
            }
            outfile << std::endl;
        }
        
        // Synchronize before next temperature
        mpi_env.barrier();
    }
    
    if (rank == 0) {
        outfile.close();
        std::cout << "\\nResults saved to: " << output_file << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI environment (required for this executable)
    MPIEnvironment mpi_env(argc, argv);
    MPIAccumulator mpi_accumulator(mpi_env);
    
    // Redirect stdout to /dev/null for non-master ranks to suppress verbose output
    std::streambuf* cout_backup = nullptr;
    std::ofstream devnull;
    if (!mpi_env.is_master()) {
        devnull.open("/dev/null");
        cout_backup = std::cout.rdbuf();
        std::cout.rdbuf(devnull.rdbuf());
    }
    
    if (mpi_env.is_master()) {
        std::cout << "========================================" << std::endl;
        std::cout << "  MPI Monte Carlo Simulation (" << mpi_env.get_num_ranks() << " ranks)" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    // Parse command line arguments
    std::string config_file = "simulation.toml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    try {
        // Only master loads and prints configuration initially
        if (mpi_env.is_master()) {
            // Load configuration
            std::cout << "Loading configuration from: " << config_file << std::endl;
        }
        
        // All ranks load configuration
        IO::SimulationConfig config = IO::ConfigurationParser::load_configuration(config_file);
        
        // Set global random seed (will be modified per rank in MPI version)
        seed = config.monte_carlo.seed;
        
        if (mpi_env.is_master()) {
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
        }
        
        // Run simulation
        if (config.temperature.type == IO::TemperatureConfig::SINGLE) {
            // For single temperature, still use serial version for now
            if (mpi_env.is_master()) {
                run_single_temperature(config);
            }
        } else {
            run_temperature_scan(config, mpi_env, mpi_accumulator);
        }
        
    } catch (const IO::ConfigurationError& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Restore stdout for non-master ranks before MPI finalize
    if (!mpi_env.is_master() && cout_backup) {
        std::cout.rdbuf(cout_backup);
        devnull.close();
    }
    
    if (mpi_env.is_master()) {
        std::cout << "\\nSimulation completed successfully!" << std::endl;
    }
    return 0;
}