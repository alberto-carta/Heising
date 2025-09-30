/*
 * Unified Monte Carlo Simulation Engine
 * 
 * This class can handle both Ising and Heisenberg models using the same code structure.
 * The key idea is to use function pointers to switch between model-specific operations.
 */

#ifndef SIMULATION_ENGINE_H
#define SIMULATION_ENGINE_H

#include "spin_types.h"
#include <fstream>

class MonteCarloSimulation {
private:
    // Simulation parameters
    SpinType model_type;
    int lattice_size;
    double temperature;
    double coupling_J;
    double max_rotation_angle;  // For Heisenberg model local updates
    
    // Lattice data - we need both types since we don't know which model at compile time
    int** ising_lattice;        // For Ising model: +1 or -1 spins
    spin3d** heisenberg_lattice; // For Heisenberg model: 3D unit vectors
    
    // Function pointers - these will point to the right functions for the chosen model
    // Think of these as "pluggable" functions that we set once and then just call
    double (MonteCarloSimulation::*calculate_local_energy)(lat_type pos);
    void (MonteCarloSimulation::*propose_spin_flip)(lat_type pos);
    bool (MonteCarloSimulation::*accept_or_reject)(lat_type pos, double energy_change);
    double (MonteCarloSimulation::*calculate_total_energy)();
    double (MonteCarloSimulation::*calculate_total_magnetization)();

public:
    // Constructor - sets up the simulation for chosen model type
    MonteCarloSimulation(SpinType type, int size, double T, double J);
    
    // Destructor - cleans up allocated memory
    ~MonteCarloSimulation();
    
    // Main simulation methods
    void initialize_lattice();
    void run_transient_phase(int transient_steps);
    void run_monte_carlo_step();
    void run_full_simulation(int mc_steps);
    
    // Measurement methods
    double get_energy();
    double get_magnetization();
    double get_absolute_magnetization();
    
    // Parameter control
    void set_temperature(double T) { temperature = T; }
    void set_coupling(double J) { coupling_J = J; }
    
    // Output methods
    void print_lattice();
    void save_configuration(const std::string& filename);

private:
    // Model-specific implementations - these are what the function pointers will point to
    
    // Ising model methods
    double ising_local_energy(lat_type pos);
    void ising_propose_flip(lat_type pos);
    double ising_total_energy();
    double ising_total_magnetization();
    
    // Heisenberg model methods  
    double heisenberg_local_energy(lat_type pos);
    void heisenberg_propose_flip(lat_type pos);
    double heisenberg_total_energy();
    double heisenberg_total_magnetization();
    
    // Common methods
    bool metropolis_test(double energy_change);
    void setup_function_pointers();  // Sets the function pointers based on model_type
    void allocate_memory();
    void deallocate_memory();
};

#endif // SIMULATION_ENGINE_H