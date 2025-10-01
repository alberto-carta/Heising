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
#include <eigen3/Eigen/Dense>

class MonteCarloSimulation {
private:
    // Simulation parameters
    SpinType model_type;
    int lattice_size;
    double temperature;
    double coupling_J;
    double max_rotation_angle;  // For Heisenberg model local updates
    
    // Metropolis statistics
    long int total_attempts;
    long int total_acceptances;
    
    // Lattice data - using Eigen Array3D for NumPy-like functionality
    // We'll use a flattened 1D array and index it as 3D: index = x*(size*size) + y*size + z
    // Both models use the same 3D lattice geometry, but different spin types
    Eigen::Array<int, Eigen::Dynamic, 1> ising_lattice;        // For Ising model: flattened 3D array of +1 or -1 spins
    Eigen::Array<spin3d, Eigen::Dynamic, 1> heisenberg_lattice; // For Heisenberg model: flattened 3D array of 3D unit vectors
    
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
    void run_warmup_phase(int warmup_steps);
    void run_monte_carlo_step();
    void run_full_simulation(int mc_steps);
    
    // Measurement methods
    double get_energy();
    double get_magnetization();
    double get_absolute_magnetization();
    
    // Metropolis statistics methods
    double get_acceptance_rate() const;
    long int get_total_attempts() const { return total_attempts; }
    long int get_total_acceptances() const { return total_acceptances; }
    void reset_statistics();
    
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
    
    // Helper function to convert 3D indices (x,y,z) to 1D index for flattened array
    inline int flatten_index(int x, int y, int z) const {
        return x * (lattice_size + 1) * (lattice_size + 1) + y * (lattice_size + 1) + z;
    }
};

#endif // SIMULATION_ENGINE_H