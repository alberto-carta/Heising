/*
 * Simulation Utilities
 * 
 * Helper functions for setting up and configuring Monte Carlo simulations
 */

#ifndef SIMULATION_UTILS_H
#define SIMULATION_UTILS_H

#include "simulation_engine.h"
#include "io/config_types.h"
#include "mpi_wrapper.h"
#include <vector>

/**
 * Convert configuration data to simulation UnitCell object
 * 
 * @param species Vector of magnetic species from configuration
 * @return Initialized UnitCell object
 */
UnitCell create_unit_cell_from_config(const std::vector<IO::MagneticSpecies>& species);

/**
 * Create coupling matrix from configuration with range checking
 * 
 * @param couplings Vector of exchange couplings from configuration
 * @param species Vector of magnetic species
 * @param lattice_size Size of the lattice for range checking
 * @return Initialized CouplingMatrix object
 */
CouplingMatrix create_couplings_from_config(const std::vector<IO::ExchangeCoupling>& couplings,
                                            const std::vector<IO::MagneticSpecies>& species,
                                            int lattice_size);

/**
 * Average simulation configuration across all MPI ranks
 * This ensures all walkers start from the same averaged state at the next temperature
 * 
 * @param sim Monte Carlo simulation object
 * @param species Vector of magnetic species
 * @param mpi_accumulator MPI accumulator for communication
 */
void average_configuration_mpi(MonteCarloSimulation& sim,
                               const std::vector<IO::MagneticSpecies>& species,
                               MPIAccumulator& mpi_accumulator);

#endif // SIMULATION_UTILS_H
