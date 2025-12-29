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
#include "../include/io/output_formatting.h"
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
    std::optional<KK_Matrix> kk_matrix = create_kk_matrix_from_config(config.kk_couplings, unit_cell, config.lattice_size);
    
    MonteCarloSimulation sim(unit_cell, couplings, config.lattice_size, config.temperature.value, kk_matrix);
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
        IO::print_section_separator("INITIALIZATION");
        std::cout << "Running MPI-parallel temperature scan with " << num_ranks << " walkers" << std::endl;
        std::cout << "Temperature range: " << config.temperature.max_temp << " to " 
                  << config.temperature.min_temp << " (step: " << config.temperature.temp_step << ")" << std::endl;
        std::cout << "Measurement steps per rank: " << measurement_steps_per_rank 
                  << " (total: " << measurement_steps_per_rank * num_ranks << ")" << std::endl;
    }
    
    // Create simulation objects from configuration (each rank creates its own)
    UnitCell unit_cell = create_unit_cell_from_config(config.species);
    CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, config.lattice_size);
    std::optional<KK_Matrix> kk_matrix = create_kk_matrix_from_config(config.kk_couplings, unit_cell, config.lattice_size);
    
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
    std::string output_file, stddev_file;
    std::ofstream outfile, stddev_outfile;
    
    if (rank == 0) {
        output_file = config.output.directory + "/" + config.output.base_name + "_";
        stddev_file = config.output.directory + "/" + config.output.base_name + "_";
        
        bool has_ising = false, has_heisenberg = false;
        for (const auto& species : config.species) {
            if (species.spin_type == SpinType::ISING) has_ising = true;
            if (species.spin_type == SpinType::HEISENBERG) has_heisenberg = true;
        }
        
        output_file += "observables.out";
        stddev_file += "observables_stddev.out";
        
        outfile.open(output_file);
        stddev_outfile.open(stddev_file);
        
        // Write headers for both files
        auto write_header = [&](std::ofstream& file, const std::string& desc) {
            file << "# Monte Carlo simulation " << desc << " (MPI parallel, " << num_ranks << " walkers)" << std::endl;
            file << "# System: ";
            for (const auto& species : config.species) {
                file << species.name << "(" << (species.spin_type == SpinType::ISING ? "Ising" : "Heisenberg") << ") ";
            }
            file << std::endl;
            file << "# Lattice: " << config.lattice_size << "³" << std::endl;
        };
        
        write_header(outfile, "results (mean values)");
        write_header(stddev_outfile, "standard deviations");
        
        // Build column header dynamically based on output options
        auto write_column_header = [&](std::ofstream& file) {
            file << "# Columns: T";
            file << " Energy/spin";
            if (config.output.output_energy_total) {
                file << " Energy_total";
            }
            file << " Magnetization SpecificHeat Susceptibility AcceptanceRate";
            
            if (config.output.output_onsite_magnetization) {
                for (const auto& sp : config.species) {
                    if (sp.spin_type == SpinType::ISING) {
                        file << " M[" << sp.name << "]";
                    } else {
                        file << " Mx[" << sp.name << "] My[" << sp.name << "] Mz[" << sp.name << "]";
                    }
                }
            }
            
            if (config.output.output_correlations) {
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
                        file << " <" << first_ising_name << "*" << config.species[i].name << ">";
                    } else {
                        file << " <" << first_heis_name << "·" << config.species[i].name << ">";
                    }
                }
            }
            file << std::endl;
        };
        
        write_column_header(outfile);
        write_column_header(stddev_outfile);
        
        outfile << std::fixed << std::setprecision(8);
        stddev_outfile << std::fixed << std::setprecision(8);
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
    
    if (rank == 0) {
        IO::print_section_separator("TEMPERATURE SCAN");
    }
    
    // Temperature scan loop
    for (double T = config.temperature.max_temp; T >= config.temperature.min_temp; T -= config.temperature.temp_step) {
        auto t_start = std::chrono::high_resolution_clock::now();
        TemperatureTimings timings;
        
        if (rank == 0) {
            std::cout << "\nT = " << std::fixed << std::setprecision(2) << T << std::endl;
        }
        
        // Each rank creates its own simulation with rank-specific seed
        long int rank_seed = get_rank_seed(config.monte_carlo.seed, rank);
        seed = rank_seed;
        
        MonteCarloSimulation sim(unit_cell, couplings, config.lattice_size, T, kk_matrix);
        
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
        
        // Measurement phase - each rank stores all measurement samples
        // Each rank runs only its portion of the total measurement steps
        auto measurement_start = std::chrono::high_resolution_clock::now();
        
        // Note: measurement_steps_per_rank sweeps, where each sweep = total_spins random update attempts
        sim.reset_statistics();
        
        // Store all measurements in arrays for proper statistics
        std::vector<double> energy_samples;
        std::vector<double> magnetization_samples;
        std::vector<std::vector<double>> mag_x_samples(config.species.size());
        std::vector<std::vector<double>> mag_y_samples(config.species.size());
        std::vector<std::vector<double>> mag_z_samples(config.species.size());
        std::vector<std::vector<double>> correlation_samples(config.species.size());
        std::vector<double> acceptance_samples;
        
        int expected_samples = measurement_steps_per_rank / config.monte_carlo.sampling_frequency;
        energy_samples.reserve(expected_samples);
        magnetization_samples.reserve(expected_samples);
        for (size_t i = 0; i < config.species.size(); i++) {
            mag_x_samples[i].reserve(expected_samples);
            mag_y_samples[i].reserve(expected_samples);
            mag_z_samples[i].reserve(expected_samples);
            if (config.output.output_correlations) {
                correlation_samples[i].reserve(expected_samples);
            }
        }
        acceptance_samples.reserve(expected_samples);
        
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
                
                // Store all samples in arrays
                energy_samples.push_back(energy);
                magnetization_samples.push_back(magnetization);
                if (config.output.output_onsite_magnetization) {
                    for (size_t i = 0; i < mag_vectors.size(); i++) {
                        mag_x_samples[i].push_back(mag_vectors[i].x);
                        mag_y_samples[i].push_back(mag_vectors[i].y);
                        mag_z_samples[i].push_back(mag_vectors[i].z);
                    }
                }
                if (config.output.output_correlations) {
                    for (size_t i = 0; i < correlations.size(); i++) {
                        correlation_samples[i].push_back(correlations[i]);
                    }
                }
                acceptance_samples.push_back(sim.get_acceptance_rate());
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
        
        // Gather all measurements from all ranks to rank 0
        auto comm_start = std::chrono::high_resolution_clock::now();
        std::vector<double> all_energy_samples = mpi_accumulator.gather_samples(energy_samples);
        std::vector<double> all_magnetization_samples = mpi_accumulator.gather_samples(magnetization_samples);
        
        // Gather magnetization vector components for each species
        std::vector<std::vector<double>> all_mag_x_samples(config.species.size());
        std::vector<std::vector<double>> all_mag_y_samples(config.species.size());
        std::vector<std::vector<double>> all_mag_z_samples(config.species.size());
        if (config.output.output_onsite_magnetization) {
            for (size_t i = 0; i < config.species.size(); i++) {
                all_mag_x_samples[i] = mpi_accumulator.gather_samples(mag_x_samples[i]);
                all_mag_y_samples[i] = mpi_accumulator.gather_samples(mag_y_samples[i]);
                all_mag_z_samples[i] = mpi_accumulator.gather_samples(mag_z_samples[i]);
            }
        }
        
        // Gather correlations
        std::vector<std::vector<double>> all_correlation_samples(config.species.size());
        if (config.output.output_correlations) {
            for (size_t i = 0; i < config.species.size(); i++) {
                all_correlation_samples[i] = mpi_accumulator.gather_samples(correlation_samples[i]);
            }
        }
        
        // Gather acceptance rates
        std::vector<double> all_acceptance_samples = mpi_accumulator.gather_samples(acceptance_samples);
        
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
        
        // Only rank 0 computes final statistics from all gathered samples
        if (rank == 0) {
            int total_samples = all_energy_samples.size();
            
            // Compute mean and standard deviation for all observables
            auto compute_stats = [](const std::vector<double>& samples) -> std::pair<double, double> {
                if (samples.empty()) return {0.0, 0.0};
                double sum = 0.0;
                for (double val : samples) sum += val;
                double mean = sum / samples.size();
                double sum_sq_diff = 0.0;
                for (double val : samples) {
                    double diff = val - mean;
                    sum_sq_diff += diff * diff;
                }
                double stddev = std::sqrt(sum_sq_diff / samples.size());
                return {mean, stddev};
            };
            
            // Energy statistics
            auto [avg_energy, stddev_energy] = compute_stats(all_energy_samples);
            double avg_energy_per_spin = avg_energy / total_spins;
            double stddev_energy_per_spin = stddev_energy / total_spins;
            
            // Total energy statistics
            double avg_total_energy = avg_energy;
            double stddev_total_energy = stddev_energy;
            
            // Magnetization statistics
            auto [avg_magnetization, stddev_magnetization] = compute_stats(all_magnetization_samples);
            double avg_magnetization_per_spin = avg_magnetization / total_spins;
            double stddev_magnetization_per_spin = stddev_magnetization / total_spins;
            
            // Specific heat and susceptibility (from fluctuations)
            double avg_energy_sq = 0.0;
            for (double e : all_energy_samples) avg_energy_sq += e * e;
            avg_energy_sq /= total_samples;
            double avg_energy_sq_per_spin = avg_energy_sq / (total_spins * total_spins);
            
            double avg_magnetization_sq = 0.0;
            for (double m : all_magnetization_samples) avg_magnetization_sq += m * m;
            avg_magnetization_sq /= total_samples;
            double avg_magnetization_sq_per_spin = avg_magnetization_sq / (total_spins * total_spins);
            
            double specific_heat = (avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (T * T);
            double susceptibility = (avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / T;
            
            // Compute stddev for specific heat and susceptibility (using error propagation)
            double stddev_specific_heat = 2.0 * stddev_energy_per_spin * std::sqrt(avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (T * T);
            double stddev_susceptibility = 2.0 * stddev_magnetization_per_spin * std::sqrt(avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / T;
            
            // Acceptance rate statistics
            auto [avg_accept_rate, stddev_accept_rate] = compute_stats(all_acceptance_samples);
            
            // Per-spin magnetization vector statistics
            std::vector<spin3d> avg_mag_vec_per_spin(config.species.size());
            std::vector<spin3d> stddev_mag_vec_per_spin(config.species.size());
            if (config.output.output_onsite_magnetization) {
                for (size_t i = 0; i < config.species.size(); i++) {
                    auto [mx_mean, mx_std] = compute_stats(all_mag_x_samples[i]);
                    auto [my_mean, my_std] = compute_stats(all_mag_y_samples[i]);
                    auto [mz_mean, mz_std] = compute_stats(all_mag_z_samples[i]);
                    avg_mag_vec_per_spin[i] = spin3d(mx_mean, my_mean, mz_mean);
                    stddev_mag_vec_per_spin[i] = spin3d(mx_std, my_std, mz_std);
                }
            }
            
            // Correlation statistics
            std::vector<double> avg_correlations(config.species.size());
            std::vector<double> stddev_correlations(config.species.size());
            if (config.output.output_correlations) {
                for (size_t i = 0; i < config.species.size(); i++) {
                    auto [corr_mean, corr_std] = compute_stats(all_correlation_samples[i]);
                    avg_correlations[i] = corr_mean;
                    stddev_correlations[i] = corr_std;
                }
            }
            
            // Output to console - formatted for readability
            IO::print_observables_formatted(
                T, total_spins,
                avg_energy, stddev_energy,
                avg_magnetization, stddev_magnetization,
                specific_heat, stddev_specific_heat,
                susceptibility, stddev_susceptibility,
                avg_accept_rate, stddev_accept_rate,
                config.species,
                avg_mag_vec_per_spin, stddev_mag_vec_per_spin,
                avg_correlations, stddev_correlations,
                config.output.output_onsite_magnetization,
                config.output.output_correlations
            );
            
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
            
            // Output standard deviations to separate file
            stddev_outfile << std::fixed << std::setprecision(8);
            stddev_outfile << std::setw(10) << T << " "
                          << std::setw(13) << stddev_energy_per_spin << " ";
            if (config.output.output_energy_total) {
                stddev_outfile << std::setw(13) << stddev_total_energy << " ";
            }
            stddev_outfile << std::setw(13) << stddev_magnetization_per_spin << " "
                          << std::setw(13) << stddev_specific_heat << " "
                          << std::setw(14) << stddev_susceptibility << " "
                          << std::setw(14) << std::setprecision(6) << stddev_accept_rate;
            
            // Optional: on-site magnetization stddev
            if (config.output.output_onsite_magnetization) {
                for (size_t i = 0; i < config.species.size(); i++) {
                    if (config.species[i].spin_type == SpinType::ISING) {
                        stddev_outfile << " " << std::setw(14) << std::setprecision(8) << stddev_mag_vec_per_spin[i].z;
                    } else {
                        stddev_outfile << " " << std::setw(14) << std::setprecision(8) << stddev_mag_vec_per_spin[i].x
                                      << " " << std::setw(14) << std::setprecision(8) << stddev_mag_vec_per_spin[i].y
                                      << " " << std::setw(14) << std::setprecision(8) << stddev_mag_vec_per_spin[i].z;
                    }
                }
            }
            
            // Optional: correlations stddev
            if (config.output.output_correlations) {
                for (const auto& corr_std : stddev_correlations) {
                    stddev_outfile << " " << std::setw(14) << std::setprecision(8) << corr_std;
                }
            }
            stddev_outfile << std::endl;
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
        stddev_outfile.close();
        std::cout << "\nResults saved to:" << std::endl;
        std::cout << "  Mean values: " << output_file << std::endl;
        std::cout << "  Std dev:     " << stddev_file << std::endl;
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
        IO::print_logo();
        IO::print_section_separator("MPI Monte Carlo Simulation (" + std::to_string(mpi_env.get_num_ranks()) + " ranks)");
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