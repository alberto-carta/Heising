/*
 * Diagnostic Utilities for Monte Carlo Simulations
 * 
 * Functions for dumping configurations, tracking observable evolution,
 * and other diagnostic operations.
 */

#ifndef DIAGNOSTIC_UTILS_H
#define DIAGNOSTIC_UTILS_H

#include "../simulation_engine.h"
#include "config_types.h"
#include <string>
#include <vector>
#include <fstream>

namespace IO {

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
 * Create directory if it doesn't exist
 * 
 * @param path Path to directory
 */
void create_directory(const std::string& path);

/**
 * Save current simulation configuration for all species
 * 
 * @param sim Monte Carlo simulation object
 * @param species Vector of magnetic species
 * @param lattice_size Size of lattice
 * @param snapshot Output snapshot to store configuration
 */
void save_configuration(const MonteCarloSimulation& sim, 
                       const std::vector<MagneticSpecies>& species,
                       int lattice_size,
                       ConfigurationSnapshot& snapshot);

/**
 * Load configuration into simulation for all species
 * 
 * @param sim Monte Carlo simulation object
 * @param species Vector of magnetic species
 * @param lattice_size Size of lattice
 * @param snapshot Input snapshot containing configuration
 */
void load_configuration(MonteCarloSimulation& sim,
                       const std::vector<MagneticSpecies>& species,
                       int lattice_size,
                       const ConfigurationSnapshot& snapshot);

/**
 * Dump configuration to file with metadata
 * 
 * @param sim Monte Carlo simulation object
 * @param species Vector of magnetic species
 * @param lattice_size Size of lattice
 * @param temperature Current temperature
 * @param measurement_step Current measurement step number
 * @param rank MPI rank
 * @param dump_dir Directory for dump files
 */
void dump_configuration_to_file(const MonteCarloSimulation& sim,
                                const std::vector<MagneticSpecies>& species,
                                int lattice_size,
                                double temperature,
                                int measurement_step,
                                int rank,
                                const std::string& dump_dir);

/**
 * Setup and open observable evolution file
 * 
 * @param dump_dir Directory for output files
 * @param rank MPI rank
 * @param temperature Current temperature
 * @param species Vector of magnetic species
 * @return Opened output file stream
 */
std::ofstream setup_observable_evolution_file(const std::string& dump_dir,
                                              int rank,
                                              double temperature,
                                              const std::vector<MagneticSpecies>& species);

/**
 * Write observable evolution data to file
 * 
 * @param file Output file stream
 * @param measurement_step Current measurement step
 * @param energy Energy per spin
 * @param magnetization Magnetization per spin
 * @param correlations Vector of correlation values
 * @param acceptance_rate Acceptance rate
 */
void write_observable_evolution(std::ofstream& file,
                                int measurement_step,
                                double energy,
                                double magnetization,
                                const std::vector<double>& correlations,
                                double acceptance_rate);

} // namespace IO

#endif // DIAGNOSTIC_UTILS_H
