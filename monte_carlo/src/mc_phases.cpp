/*
 * Monte Carlo Simulation Phases Implementation
 * 
 * This module contains the warmup and measurement phases that are common
 * to all Monte Carlo simulations. Both functions modify the simulation
 * object in-place, evolving the spin configuration through Monte Carlo updates.
 */

#include "../include/mc_phases.h"
#include "../include/io/diagnostic_utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

/**
 * Run the warmup/equilibration phase of a Monte Carlo simulation
 * 
 * Purpose:
 *   - Equilibrate the system from initial configuration to thermal equilibrium
 *   - Allow the simulation to "forget" the initial state
 *   - Optionally profile MC step timing for performance diagnostics
 * 
 * Implementation:
 *   - Performs warmup_steps Monte Carlo sweeps
 *   - Each sweep = total_spins random spin flip attempts
 *   - Does NOT collect statistics (measurements done in measurement phase)
 *   - Works for both Ising and Heisenberg spins (spin-type agnostic)
 * 
 * Side effects:
 *   - MODIFIES sim object IN-PLACE: spin configuration evolves through MC updates
 *   - Changes internal state of simulation (spin arrays, energy tracking, etc.)
 *   - Prints progress to stdout if rank == 0
 * 
 * @param sim Reference to simulation object (MODIFIED IN-PLACE)
 * @param warmup_steps Number of Monte Carlo sweeps to perform
 * @param total_spins Total number of spins in system (lattice_size³ × spins_per_cell)
 * @param enable_profiling If true, profile first 1000 steps for timing estimate
 * @param rank MPI rank (controls output: only rank 0 prints)
 * @return Pair of (warmup_time_seconds, mc_step_time_estimate_seconds)
 */
std::pair<double, double> run_warmup_phase(
    MonteCarloSimulation& sim,
    int warmup_steps,
    int total_spins,
    bool enable_profiling,
    int rank
) {
    double initial_energy = sim.get_energy();

    if (rank == 0) {
        std::cout << "Warmup phase" << std::endl;
        std::cout << " Energy of the initial configuration: " << std::fixed << std::setprecision(6)
                  << initial_energy << std::endl;
    }

    
    auto warmup_start = std::chrono::high_resolution_clock::now();
    double mc_step_time_estimate = 0.0;
    
    if (enable_profiling && warmup_steps >= 1000) {
        // Profile MC step timing during first 1000 steps for performance diagnostics
        auto mc_timing_start = std::chrono::high_resolution_clock::now();
        for (int step_sample = 0; step_sample < 1000; step_sample++) {
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
        }
        auto mc_timing_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> mc_elapsed = mc_timing_end - mc_timing_start;
        mc_step_time_estimate = mc_elapsed.count() / (1000.0 * total_spins);
        
        // Continue with remaining warmup
        for (int sweep = 1000; sweep < warmup_steps; sweep++) {
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            if (rank == 0 && (sweep + 1) % 1000 == 0) {
                std::cout << "  Warmup: " << (sweep + 1) << "/" << warmup_steps 
                          << " (" << std::setprecision(1) << std::fixed
                          << (100.0 * (sweep + 1)) / warmup_steps << "%)" << std::endl;
            }
        }
    } else {
        // Regular warmup without timing
        for (int sweep = 0; sweep < warmup_steps; sweep++) {
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            if (rank == 0 && (sweep + 1) % 1000 == 0) {
                std::cout << "  Warmup: " << (sweep + 1) << "/" << warmup_steps 
                          << " (" << std::setprecision(1) << std::fixed
                          << (100.0 * (sweep + 1)) / warmup_steps << "%)" << std::endl;
            }
        }
    }
    
    auto warmup_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> warmup_elapsed = warmup_end - warmup_start;
    
    return {warmup_elapsed.count(), mc_step_time_estimate};
}

/**
 * Run the measurement phase of a Monte Carlo simulation
 * 
 * Purpose:
 *   - Collect statistical samples of observables from equilibrated configurations
 *   - Sample energy, magnetization, correlations, and acceptance rates
 *   - Generate time series data for autocorrelation analysis
 *   - Optionally dump configurations and evolution data for post-processing
 * 
 * Implementation:
 *   - Performs measurement_steps_per_rank Monte Carlo sweeps
 *   - Each sweep = total_spins random spin flip attempts
 *   - Samples observables every sampling_frequency sweeps
 *   - Works for both Ising and Heisenberg spins:
 *       * Ising: magnetization vector has only z-component (x=y=0)
 *       * Heisenberg: magnetization vector has all x,y,z components
 *       * Mixed systems: each spin type handled independently
 *   - Can use tracked observables (fast) or recompute each sample (accurate but slow)
 * 
 * Side effects:
 *   - MODIFIES sim object IN-PLACE: spin configuration continuously evolves
 *   - Updates internal simulation statistics via reset_statistics()
 *   - May write to disk: configuration dumps, observable evolution files
 *   - Prints progress to stdout if rank == 0
 * 
 * Observable storage:
 *   - Stores raw samples (not averaged) in MeasurementData structure
 *   - Each MPI rank collects its own samples independently
 *   - Samples later gathered to rank 0 for final statistics computation
 * 
 * @param sim Reference to simulation object (MODIFIED IN-PLACE)
 * @param config Full simulation configuration (controls sampling, output, diagnostics)
 * @param measurement_steps_per_rank Number of MC sweeps for this MPI rank
 * @param total_spins Total number of spins in system
 * @param T Temperature (used for file naming in dumps)
 * @param rank MPI rank (controls output and determines dump behavior)
 * @param dump_dir Directory for writing configuration/evolution files
 * @return Pair of (MeasurementData with all samples, measurement_time_seconds)
 */
std::pair<MeasurementData, double> run_measurement_phase(
    MonteCarloSimulation& sim,
    const IO::SimulationConfig& config,
    int measurement_steps_per_rank,
    int total_spins,
    double T,
    int rank,
    const std::string& dump_dir
) {
    if (rank == 0) {
        std::cout << std::endl;
        std::cout << "Measurement phase" << std::endl;
    }

    auto measurement_start = std::chrono::high_resolution_clock::now();
    
    // Reset internal statistics tracking (important after warmup phase)
    sim.reset_statistics();
    
    // Initialize measurement data structure to hold all samples
    MeasurementData data;
    int expected_samples = measurement_steps_per_rank / config.monte_carlo.sampling_frequency;
    
    // Pre-allocate vectors for efficiency
    data.energy_samples.reserve(expected_samples);
    data.magnetization_samples.reserve(expected_samples);
    
    // Per-species magnetization components (x,y,z)
    // Note: For Ising spins, only z-component is non-zero; x=y=0
    data.mag_x_samples.resize(config.species.size());
    data.mag_y_samples.resize(config.species.size());
    data.mag_z_samples.resize(config.species.size());
    data.correlation_samples.resize(config.species.size());
    
    for (size_t i = 0; i < config.species.size(); i++) {
        data.mag_x_samples[i].reserve(expected_samples);
        data.mag_y_samples[i].reserve(expected_samples);
        data.mag_z_samples[i].reserve(expected_samples);
        if (config.output.output_correlations) {
            data.correlation_samples[i].reserve(expected_samples);
        }
    }
    data.acceptance_samples.reserve(expected_samples);
    
    // For autocorrelation estimation
    if (config.diagnostics.estimate_autocorrelation) {
        data.energy_series.reserve(expected_samples);
        data.magnetization_series.reserve(expected_samples);
    }
    
    // Setup observable evolution file if needed
    std::ofstream obs_evolution_file;
    if (config.diagnostics.enable_observable_evolution && config.diagnostics.should_dump_rank(rank)) {
        obs_evolution_file = IO::setup_observable_evolution_file(dump_dir, rank, T, config.species);
    }
    
    // Print warnings if using expensive options
    if (rank == 0 && config.diagnostics.recompute_observables_each_sample) {
        std::cout << "  WARNING: recompute_observables_each_sample=true (slower)" << std::endl;
    }
    if (rank == 0 && config.output.output_correlations) {
        std::cout << "  Note: Computing spin correlations (requires full lattice traversal)" << std::endl;
    }
    
    // Main measurement loop
    // Each iteration: (1) evolves spin config via MC, (2) samples observables
    int num_samples = 0;
    for (int sweep = 0; sweep < measurement_steps_per_rank; sweep++) {
        // One sweep = total_spins random spin flip attempts
        // This MODIFIES the spin configuration in sim object
        for (int attempt = 0; attempt < total_spins; attempt++) {
            sim.run_monte_carlo_step();  // IN-PLACE modification of spins
        }
        
        // Sample observables at specified frequency
        if (sweep % config.monte_carlo.sampling_frequency == 0) {
            // Get observables from current spin configuration
            double energy, magnetization;
            std::vector<spin3d> mag_vectors;  // Per-species magnetization vectors
            
            if (config.diagnostics.recompute_observables_each_sample) {
                // Recompute from scratch (slow but accurate)
                energy = sim.get_energy();
                magnetization = sim.get_magnetization();
                mag_vectors = sim.get_magnetization_vector_per_spin();
            } else {
                // Use tracked values (fast, uses incremental updates)
                energy = sim.get_tracked_energy();
                magnetization = sim.get_tracked_magnetization();
                mag_vectors = sim.get_tracked_magnetization_vector_per_spin();
            }
            
            // Correlations always need full computation (no tracking available)
            std::vector<double> correlations;
            if (config.output.output_correlations) {
                correlations = sim.get_spin_correlation_with_first();
            }
            
            // Store all samples (raw data, not averaged yet)
            data.energy_samples.push_back(energy);
            data.magnetization_samples.push_back(magnetization);
            
            // Store per-species magnetization components
            // For Ising: only mag_vectors[i].z is non-zero
            // For Heisenberg: all three components (x,y,z) are populated
            if (config.output.output_onsite_magnetization) {
                for (size_t i = 0; i < mag_vectors.size(); i++) {
                    data.mag_x_samples[i].push_back(mag_vectors[i].x);
                    data.mag_y_samples[i].push_back(mag_vectors[i].y);
                    data.mag_z_samples[i].push_back(mag_vectors[i].z);
                }
            }
            
            // Store correlation function samples if requested
            if (config.output.output_correlations) {
                for (size_t i = 0; i < correlations.size(); i++) {
                    data.correlation_samples[i].push_back(correlations[i]);
                }
            }
            
            // Store acceptance rate (fraction of attempted moves that were accepted)
            data.acceptance_samples.push_back(sim.get_acceptance_rate());
            num_samples++;
            
            // Store time series for autocorrelation analysis
            if (config.diagnostics.estimate_autocorrelation) {
                data.energy_series.push_back(energy / total_spins);
                data.magnetization_series.push_back(magnetization / total_spins);
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
                      << " (" << std::setprecision(1) << std::fixed
                      << (100.0 * (sweep + 1)) / measurement_steps_per_rank << "%)" << std::endl;
        }
    }
    
    if (obs_evolution_file.is_open()) {
        obs_evolution_file.close();
    }
    
    auto measurement_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> measurement_elapsed = measurement_end - measurement_start;
    
    return {data, measurement_elapsed.count()};
}
