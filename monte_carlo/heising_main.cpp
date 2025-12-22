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
#include "../include/io/diagnostic_utils.h"
#include "../include/mpi_wrapper.h"
#include "../include/profiling.h"
#include "../include/simulation_utils.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

// Global random seed (will be set from configuration)
long int seed = -12345;

/**
 * Initialize simulation based on configuration
 */
void initialize_simulation(MonteCarloSimulation& sim, const IO::InitializationConfig& init_config) {
    switch (init_config.type) {
        case IO::InitializationConfig::RANDOM:
            sim.initialize_lattice_random();
            break;
        case IO::InitializationConfig::CUSTOM:
            if (init_config.pattern.empty()) {
                std::cerr << "ERROR: CUSTOM initialization requires 'pattern' array in config" << std::endl;
                sim.initialize_lattice_random();  // Fall back to random
            } else {
                sim.initialize_lattice_custom(init_config.pattern);
            }
            break;
        default:
            sim.initialize_lattice_random();
            break;
    }
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
    initialize_simulation(sim, config.initialization);
    
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
    
    // IMPORTANT: Definition of Monte Carlo "sweep"
    // ============================================
    // One "sweep" = total_spins random update attempts
    // Each attempt:
    //   1. Pick a RANDOM spin (x, y, z, spin_id) - not sequential!
    //   2. Propose a move (flip for Ising, rotation for Heisenberg)
    //   3. Accept/reject via Metropolis criterion
    // 
    // This means:
    //   - Some spins may be updated multiple times per sweep
    //   - Some spins may not be updated at all in a given sweep
    //   - On average, each spin gets ~1 update attempt per sweep
    //   - This is the standard Monte Carlo "random site selection" algorithm
    
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
        
        // Build column header dynamically based on output options
        outfile << "# Columns: T";
        std::cout << "T         ";
        
        // Always output energy per spin
        outfile << " Energy/spin";
        std::cout << " Energy/spin  ";
        
        // Optional: total energy
        if (config.output.output_energy_total) {
            outfile << " Energy_total";
            std::cout << " Energy_total ";
        }
        
        // Always output basic thermodynamic quantities
        outfile << " Magnetization SpecificHeat Susceptibility AcceptanceRate";
        std::cout << " Magnetization SpecificHeat  Susceptibility AcceptanceRate";
        
        // Optional: on-site magnetization per species
        if (config.output.output_onsite_magnetization) {
            for (const auto& sp : config.species) {
                if (sp.spin_type == SpinType::ISING) {
                    outfile << " M[" << sp.name << "]";
                    std::cout << " M[" << sp.name << "]";
                } else {
                    outfile << " Mx[" << sp.name << "] My[" << sp.name << "] Mz[" << sp.name << "]";
                    std::cout << " Mx[" << sp.name << "] My[" << sp.name << "] Mz[" << sp.name << "]";
                }
            }
        }
        
        // Optional: correlations with first spin
        if (config.output.output_correlations) {
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
                    std::cout << " <" << first_ising_name << "*" << config.species[i].name << ">";
                } else {
                    outfile << " <" << first_heis_name << "·" << config.species[i].name << ">";
                    std::cout << " <" << first_heis_name << "·" << config.species[i].name << ">";
                }
            }
        }
        
        outfile << std::endl;
        std::cout << std::endl;
        
        outfile << std::fixed << std::setprecision(8);
        
        // Print separator line for console
        std::cout << "----------";
        std::cout << " -------------";
        if (config.output.output_energy_total) std::cout << " -------------";
        std::cout << " ------------- ------------- -------------- --------------";
        if (config.output.output_onsite_magnetization) {
            for (const auto& sp : config.species) {
                if (sp.spin_type == SpinType::ISING) {
                    std::cout << " --------------";
                } else {
                    std::cout << " -------------- -------------- --------------";
                }
            }
        }
        if (config.output.output_correlations) {
            for (size_t i = 0; i < config.species.size(); i++) {
                std::cout << " --------------";
            }
        }
        std::cout << std::endl;
    }
    
    bool first_temperature = true;
    
    // Diagnostic setup
    std::string dump_dir = config.output.directory + "/dumps";
    if (rank == 0 && (config.diagnostics.enable_config_dump || config.diagnostics.enable_observable_evolution)) {
        IO::create_directory(config.output.directory);
        IO::create_directory(dump_dir);
        std::cout << "  Dump directory created: " << dump_dir << std::endl;
    }
    mpi_env.barrier();  // Ensure directory exists before any rank tries to write
    
    // Profiling variables
    std::vector<TemperatureTimings> all_timings;  // Store timings for each temperature
    
    // Temperature scan loop
    for (double T = config.temperature.max_temp; T >= config.temperature.min_temp; T -= config.temperature.temp_step) {
        auto t_start = std::chrono::high_resolution_clock::now();
        TemperatureTimings timings;
        
        if (rank == 0) {
            std::cout << "\\nT = " << std::fixed << std::setprecision(2) << T << std::endl;
        }
        
        // Each rank creates its own simulation with rank-specific seed
        long int rank_seed = get_rank_seed(config.monte_carlo.seed, rank);
        seed = rank_seed;
        
        MonteCarloSimulation sim(unit_cell, couplings, config.lattice_size, T);
        
        if (first_temperature) {
            if (rank == 0) {
                std::cout << "  Initializing all walkers..." << std::endl;
            }
            initialize_simulation(sim, config.initialization);
            first_temperature = false;
        } else {
            // All ranks initialize and then average their configurations
            if (rank == 0) {
                std::cout << "  Averaging configurations from all walkers..." << std::endl;
            }
            initialize_simulation(sim, config.initialization);
            average_configuration_mpi(sim, config.species, mpi_accumulator);
            mpi_env.barrier();
            if (rank == 0) {
                std::cout << "  All walkers synchronized with averaged configuration." << std::endl;
            }
        }
        
        // Warmup - each rank runs independently
        auto warmup_start = std::chrono::high_resolution_clock::now();
        
        // Note: One sweep = total_spins RANDOM update attempts (not sequential)
        // Profile MC step timing during first 1000 steps of warmup
        if (config.diagnostics.enable_profiling && config.monte_carlo.warmup_steps >= 1000) {
            auto mc_timing_start = std::chrono::high_resolution_clock::now();
            for (int step_sample = 0; step_sample < 1000; step_sample++) {
                for (int attempt = 0; attempt < total_spins; attempt++) {
                    sim.run_monte_carlo_step();
                }
            }
            auto mc_timing_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> mc_elapsed = mc_timing_end - mc_timing_start;
            timings.mc_step_time_estimate = mc_elapsed.count() / (1000.0 * total_spins);  // Time per single MC step
            
            // Continue with remaining warmup
            for (int sweep = 1000; sweep < config.monte_carlo.warmup_steps; sweep++) {
                for (int attempt = 0; attempt < total_spins; attempt++) {
                    sim.run_monte_carlo_step();
                }
                if (rank == 0 && (sweep + 1) % 1000 == 0) {
                    std::cout << "  Warmup: " << (sweep + 1) << "/" << config.monte_carlo.warmup_steps 
                              << " (" << std::setprecision(1) << (100.0 * (sweep + 1)) / config.monte_carlo.warmup_steps << "%)" << std::endl;
                }
            }
        } else {
            // Regular warmup without timing
            for (int sweep = 0; sweep < config.monte_carlo.warmup_steps; sweep++) {
                for (int attempt = 0; attempt < total_spins; attempt++) {
                    sim.run_monte_carlo_step();
                }
                if (rank == 0 && (sweep + 1) % 1000 == 0) {
                    std::cout << "  Warmup: " << (sweep + 1) << "/" << config.monte_carlo.warmup_steps 
                              << " (" << std::setprecision(1) << (100.0 * (sweep + 1)) / config.monte_carlo.warmup_steps << "%)" << std::endl;
                }
            }
        }
        
        auto warmup_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> warmup_elapsed = warmup_end - warmup_start;
        timings.warmup_time = warmup_elapsed.count();
        
        // Measurement phase - each rank accumulates local statistics
        // Each rank runs only its portion of the total measurement steps
        auto measurement_start = std::chrono::high_resolution_clock::now();
        
        // Note: measurement_steps_per_rank sweeps, where each sweep = total_spins random update attempts
        sim.reset_statistics();
        double local_energy = 0.0, local_energy_sq = 0.0;
        double local_magnetization = 0.0, local_magnetization_sq = 0.0;
        std::vector<spin3d> local_mag_vec_per_spin(config.species.size(), spin3d(0.0, 0.0, 0.0));
        std::vector<double> local_correlations(config.species.size(), 0.0);
        int num_samples = 0;
        
        // For autocorrelation estimation
        std::vector<double> energy_series, magnetization_series;
        if (config.diagnostics.estimate_autocorrelation) {
            energy_series.reserve(measurement_steps_per_rank / config.monte_carlo.sampling_frequency);
            magnetization_series.reserve(measurement_steps_per_rank / config.monte_carlo.sampling_frequency);
        }
        
        // Setup observable evolution file if needed
        std::ofstream obs_evolution_file;
        if (config.diagnostics.enable_observable_evolution && config.diagnostics.should_dump_rank(rank)) {
            obs_evolution_file = IO::setup_observable_evolution_file(dump_dir, rank, T, config.species);
        }
        
        // Print warning if using expensive observable computation
        if (rank == 0 && config.diagnostics.recompute_observables_each_sample) {
            std::cout << "  WARNING: recompute_observables_each_sample=true" << std::endl;
            std::cout << "           This will recompute energy/magnetization from scratch each sample" << std::endl;
            std::cout << "           Consider using tracked observables (default) for better performance." << std::endl;
        }
        
        // Warn if correlations are computed (they are always expensive)
        if (rank == 0 && config.output.output_correlations) {
            std::cout << "  Note: Computing spin correlations requires full lattice traversal each sample" << std::endl;
            std::cout << "        Consider reducing sampling_frequency if simulation is slow" << std::endl;
        }
        
        // Main measurement loop
        for (int sweep = 0; sweep < measurement_steps_per_rank; sweep++) {
            // One sweep = total_spins random update attempts
            // Each call to run_monte_carlo_step() picks ONE random spin and attempts to update it
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();  // Pick random spin, propose move, accept/reject
            }
            
            if (sweep % config.monte_carlo.sampling_frequency == 0) {
                // Get observables - use tracked values unless recomputation requested
                double energy, magnetization;
                std::vector<spin3d> mag_vectors;
                
                if (config.diagnostics.recompute_observables_each_sample) {
                    energy = sim.get_energy();
                    magnetization = sim.get_magnetization();
                    mag_vectors = sim.get_magnetization_vector_per_spin();
                } else {
                    energy = sim.get_tracked_energy();
                    magnetization = sim.get_tracked_magnetization();
                    mag_vectors = sim.get_tracked_magnetization_vector_per_spin();  // Use tracked version (fast!)
                }
                
                // Correlations always need full computation
                std::vector<double> correlations;
                if (config.output.output_correlations) {
                    correlations = sim.get_spin_correlation_with_first();
                }
                
                local_energy += energy;
                local_energy_sq += energy * energy;
                local_magnetization += magnetization;
                local_magnetization_sq += magnetization * magnetization;
                if (config.output.output_onsite_magnetization) {
                    for (size_t i = 0; i < mag_vectors.size(); i++) {
                        local_mag_vec_per_spin[i].x += mag_vectors[i].x;
                        local_mag_vec_per_spin[i].y += mag_vectors[i].y;
                        local_mag_vec_per_spin[i].z += mag_vectors[i].z;
                    }
                }
                if (config.output.output_correlations) {
                    for (size_t i = 0; i < correlations.size(); i++) {
                        local_correlations[i] += correlations[i];
                    }
                }
                num_samples++;
                
                // Store for autocorrelation
                if (config.diagnostics.estimate_autocorrelation) {
                    energy_series.push_back(energy / total_spins);
                    magnetization_series.push_back(magnetization / total_spins);
                }
                
                // Dump configuration if requested
                if (config.diagnostics.enable_config_dump && 
                    config.diagnostics.should_dump_rank(rank) &&
                    num_samples % config.diagnostics.dump_every_n_measurements == 0) {
                    IO::dump_configuration_to_file(sim, config.species, config.lattice_size, 
                                                   T, sweep, rank, dump_dir);
                }
                
                // Write observable evolution if requested
                if (obs_evolution_file.is_open() &&
                    num_samples % config.diagnostics.dump_every_n_measurements == 0) {
                    double accept_rate = sim.get_acceptance_rate();
                    IO::write_observable_evolution(obs_evolution_file, sweep, 
                                                   energy / total_spins, 
                                                   magnetization / total_spins,
                                                   correlations, accept_rate);
                }
            }
            
            // Progress updates every 5000 sweeps
            if (rank == 0 && (sweep + 1) % 5000 == 0) {
                std::cout << "  Measurement: " << (sweep + 1) << "/" << measurement_steps_per_rank 
                          << " (" << std::setprecision(1) << (100.0 * (sweep + 1)) / measurement_steps_per_rank << "%)" << std::endl;
            }
        }
        
        if (obs_evolution_file.is_open()) {
            obs_evolution_file.close();
        }
        
        auto measurement_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> measurement_elapsed = measurement_end - measurement_start;
        timings.measurement_time = measurement_elapsed.count();
        
        // Debug: print per-rank timing before communication
        if (config.diagnostics.enable_profiling) {
            double rank_measurement_time = measurement_elapsed.count();
            std::vector<double> all_measurement_times;
            if (rank == 0) {
                all_measurement_times.resize(num_ranks);
            }
#ifdef USE_MPI
            MPI_Gather(&rank_measurement_time, 1, MPI_DOUBLE, all_measurement_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
            if (rank == 0) {
                double min_time = *std::min_element(all_measurement_times.begin(), all_measurement_times.end());
                double max_time = *std::max_element(all_measurement_times.begin(), all_measurement_times.end());
                double imbalance = ((max_time - min_time) / max_time) * 100.0;
                if (imbalance > 10.0) {
                    std::cout << "  WARNING: Measurement phase load imbalance: " << std::fixed << std::setprecision(1) 
                             << imbalance << "% (min=" << min_time << "s, max=" << max_time << "s)" << std::endl;
                    std::cout << "  Per-rank measurement times (seconds):" << std::endl;
                    for (int r = 0; r < num_ranks; r++) {
                        std::cout << "    Rank " << std::setw(2) << r << ": " << std::fixed << std::setprecision(2) 
                                 << all_measurement_times[r] << " s" << std::endl;
                    }
                }
            }
        }
        
        // Accumulate statistics across all ranks
        auto comm_start = std::chrono::high_resolution_clock::now();
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
        
        auto comm_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> comm_elapsed = comm_end - comm_start;
        timings.communication_time = comm_elapsed.count();
        
        // Barrier with timing to measure load imbalance
        double barrier_wait = 0.0;
        if (config.diagnostics.enable_profiling) {
            barrier_wait = mpi_env.barrier_with_timing();
        } else {
            mpi_env.barrier();
        }
        timings.barrier_wait_time = barrier_wait;
        
        // Collect barrier times from all ranks for statistics (only if profiling enabled)
        std::vector<double> all_barrier_times;
        if (config.diagnostics.enable_profiling && rank == 0) {
            all_barrier_times.resize(num_ranks);
        }
        if (config.diagnostics.enable_profiling) {
#ifdef USE_MPI
            MPI_Gather(&barrier_wait, 1, MPI_DOUBLE, all_barrier_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
        }
        
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
            
            // Calculate total energy
            double avg_total_energy = avg_energy_per_spin * total_spins;
            
            // Output to console
            std::cout << std::fixed << std::setprecision(8);
            std::cout << std::setw(10) << T << " "
                      << std::setw(13) << avg_energy_per_spin << " ";
            if (config.output.output_energy_total) {
                std::cout << std::setw(13) << avg_total_energy << " ";
            }
            std::cout << std::setw(13) << avg_magnetization_per_spin << " "
                      << std::setw(13) << specific_heat << " "
                      << std::setw(14) << susceptibility << " "
                      << std::setw(14) << std::setprecision(6) << avg_accept_rate;
            
            // Optional: on-site magnetization
            if (config.output.output_onsite_magnetization) {
                for (size_t i = 0; i < config.species.size(); i++) {
                    if (config.species[i].spin_type == SpinType::ISING) {
                        // For Ising: output single magnetization value
                        std::cout << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].z;
                    } else {
                        // For Heisenberg: output all three components
                        std::cout << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].x
                                  << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].y
                                  << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].z;
                    }
                }
            }
            
            // Optional: correlations
            if (config.output.output_correlations) {
                for (const auto& corr : avg_correlations) {
                    std::cout << " " << std::setw(14) << std::setprecision(8) << corr;
                }
            }
            std::cout << std::endl;
            
            // Output to file
            outfile << std::fixed << std::setprecision(8);
            outfile << std::setw(10) << T << " "
                    << std::setw(13) << avg_energy_per_spin << " ";
            if (config.output.output_energy_total) {
                outfile << std::setw(13) << avg_total_energy << " ";
            }
            outfile << std::setw(13) << avg_magnetization_per_spin << " "
                    << std::setw(13) << specific_heat << " "
                    << std::setw(14) << susceptibility << " "
                    << std::setw(14) << std::setprecision(6) << avg_accept_rate;
            
            // Optional: on-site magnetization
            if (config.output.output_onsite_magnetization) {
                for (size_t i = 0; i < config.species.size(); i++) {
                    if (config.species[i].spin_type == SpinType::ISING) {
                        outfile << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].z;
                    } else {
                        outfile << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].x
                                << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].y
                                << " " << std::setw(14) << std::setprecision(8) << avg_mag_vec_per_spin[i].z;
                    }
                }
            }
            
            // Optional: correlations
            if (config.output.output_correlations) {
                for (const auto& corr : avg_correlations) {
                    outfile << " " << std::setw(14) << std::setprecision(8) << corr;
                }
            }
            outfile << std::endl;
        }
        
        // Compute and print autocorrelation estimates if enabled
        if (config.diagnostics.estimate_autocorrelation && energy_series.size() > 2) {
            double rho_energy = estimate_autocorrelation(energy_series);
            double rho_mag = estimate_autocorrelation(magnetization_series);
            double tau_energy = estimate_autocorrelation_time(rho_energy);
            double tau_mag = estimate_autocorrelation_time(rho_mag);
            
            if (rank == 0) {
                std::cout << "  Autocorrelation ρ(1): Energy=" << std::fixed << std::setprecision(4) << rho_energy
                         << ", Magnetization=" << rho_mag << std::endl;
                std::cout << "  Estimated τ_auto: Energy≈" << std::fixed << std::setprecision(2) << tau_energy
                         << " sweeps, Magnetization≈" << tau_mag << " sweeps" << std::endl;
            }
        }
        
        // Store temperature timing
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_elapsed = t_end - t_start;
        timings.total_time = t_elapsed.count();
        all_timings.push_back(timings);
        
        // Print timing summary for this temperature if profiling enabled
        if (config.diagnostics.enable_profiling && rank == 0) {
            print_temperature_timing(timings);
            
            std::cout << "  Note: 'Comm' time includes waiting for slower ranks at MPI operations" << std::endl;
            std::cout << "        High comm time indicates load imbalance across walkers" << std::endl;
            
            if (all_barrier_times.size() > 0) {
                print_barrier_statistics(all_barrier_times);
            }
        }
        
        // Synchronize before next temperature
        mpi_env.barrier();
    }
    
    // Print comprehensive profiling report at the end
    if (config.diagnostics.enable_profiling && rank == 0 && all_timings.size() > 0) {
        print_profiling_report(all_timings, all_timings.size());
    }
    
    if (rank == 0) {
        outfile.close();
        std::cout << "\\nResults saved to: " << output_file << std::endl;
    }
}  // end run_temperature_scan

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