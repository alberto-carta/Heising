/*
 * Monte Carlo Simulation Phases
 * 
 * Handles warmup and measurement phases of Monte Carlo simulations
 * Separated from main control flow for modularity and reusability
 */

#ifndef MC_PHASES_H
#define MC_PHASES_H

#include "simulation_engine.h"
#include "io/configuration_parser.h"
#include "mpi_wrapper.h"
#include "profiling.h"
#include <vector>
#include <string>
#include <fstream>

// Structure to hold measurement data during measurement phase
struct MeasurementData {
    std::vector<double> energy_samples;
    std::vector<double> magnetization_samples;
    std::vector<std::vector<double>> mag_x_samples; // will be 0 for ising spins
    std::vector<std::vector<double>> mag_y_samples; // will be 0 for ising spins
    std::vector<std::vector<double>> mag_z_samples; // only one active for ising spins
    std::vector<std::vector<double>> correlation_samples;
    std::vector<double> acceptance_samples;
    
    // For autocorrelation
    std::vector<double> energy_series;
    std::vector<double> magnetization_series;
};

/**
 * Run the warmup phase of the simulation
 * @param sim The simulation object
 * @param warmup_steps Number of warmup sweeps
 * @param total_spins Total number of spins in the system
 * @param enable_profiling Whether to profile MC step timing
 * @param rank MPI rank (for output control)
 * @return Time taken for warmup in seconds, and optionally MC step timing estimate
 */
std::pair<double, double> run_warmup_phase(
    MonteCarloSimulation& sim,
    int warmup_steps,
    int total_spins,
    bool enable_profiling,
    int rank
);

/**
 * Run the measurement phase of the simulation
 * @param sim The simulation object
 * @param config Simulation configuration
 * @param measurement_steps_per_rank Number of measurement sweeps for this rank
 * @param total_spins Total number of spins in the system
 * @param T Temperature
 * @param rank MPI rank
 * @param dump_dir Directory for dumping configurations
 * @return MeasurementData containing all sampled observables and measurement time
 */
std::pair<MeasurementData, double> run_measurement_phase(
    MonteCarloSimulation& sim,
    const IO::SimulationConfig& config,
    int measurement_steps_per_rank,
    int total_spins,
    double T,
    int rank,
    const std::string& dump_dir
);

#endif // MC_PHASES_H
