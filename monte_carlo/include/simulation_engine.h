/*
 * Fast Simulation Engine Header
 * 
 * Performance-optimized simulation engine using simplified data structures
 * Focus on numerical efficiency over complex abstractions
 */

#ifndef SIMULATION_ENGINE_H
#define SIMULATION_ENGINE_H

#include "multi_atom.h"
#include "spin_types.h"
#include "random.h"
#include <eigen3/Eigen/Dense>

// External seed declaration
extern long int seed;

class MonteCarloSimulation {
private:
    // Core parameters
    int lattice_size;
    double temperature;
    double max_rotation_angle;
    
        // Multi-atom structure
    UnitCell unit_cell;
    CouplingMatrix coupling_matrix;
    
    // Lattice storage - Eigen arrays for performance
    Eigen::ArrayXd ising_spins;      // All Ising spins in flat array
    Eigen::ArrayXd heisenberg_x;     // Heisenberg x-components
    Eigen::ArrayXd heisenberg_y;     // Heisenberg y-components  
    Eigen::ArrayXd heisenberg_z;     // Heisenberg z-components
    
    // Statistics
    long int total_attempts;
    long int total_acceptances;
    
    // Fast indexing
    inline int flatten_index(int x, int y, int z, int atom_id) const {
        int num_atoms = unit_cell.num_atoms();
        return ((x - 1) * lattice_size * lattice_size + (y - 1) * lattice_size + (z - 1)) * num_atoms + atom_id;
    }
    
    // Fast energy calculation - direct array access
    double calculate_local_energy_fast(int x, int y, int z, int atom_id);
    
    // Fast Metropolis test
    inline bool metropolis_test_fast(double energy_change) {
        if (energy_change <= 0.0) return true;
        double probability = std::exp(-energy_change / temperature);
        return ran1(&seed) < probability;
    }

public:
    // Constructor
    MonteCarloSimulation(const UnitCell& uc, const CouplingMatrix& couplings, 
                        int size, double T);
    
    // Destructor
    ~MonteCarloSimulation() = default;
    
    // Main methods
    void initialize_lattice();
    void run_monte_carlo_step();
    void run_warmup_phase(int warmup_steps);
    
    // Measurements
    double get_energy();
    double get_magnetization();
    double get_absolute_magnetization();
    double get_acceptance_rate() const;
    void reset_statistics();
    
    // Fast spin access for testing
    int get_ising_spin(int x, int y, int z, int atom_id) const;
    void set_ising_spin(int x, int y, int z, int atom_id, int spin);
    spin3d get_heisenberg_spin(int x, int y, int z, int atom_id) const;
    void set_heisenberg_spin(int x, int y, int z, int atom_id, const spin3d& spin);
    
    // Direct energy calculation for testing
    double calculate_local_energy(int x, int y, int z, int atom_id) {
        return calculate_local_energy_fast(x, y, z, atom_id);
    }
    
    // Metropolis test for testing
    bool metropolis_test(double energy_change) {
        return metropolis_test_fast(energy_change);
    }
};

#endif // SIMULATION_ENGINE_H