/*
 * Fast Simulation Engine Header
 * 
 * Performance-optimized simulation engine using simplified data structures
 * Focus on numerical efficiency over complex abstractions
 */

#ifndef SIMULATION_ENGINE_H
#define SIMULATION_ENGINE_H

#include "multi_spin.h"
#include "spin_types.h"
#include "random.h"
#include <eigen3/Eigen/Dense>
#include <optional>  // C++17 feature

// External seed declaration
extern long int seed;

class MonteCarloSimulation {
private:
    // Core parameters
    int lattice_size;
    double temperature;
    double max_rotation_angle;
    
        // Multi-spin structure
    UnitCell unit_cell;
    CouplingMatrix coupling_matrix;
    std::optional<KK_Matrix> kk_matrix;  // Optional Kugel-Khomskii coupling matrix
    
    // Lattice storage - Eigen arrays for performance
    Eigen::ArrayXd ising_spins;      // All Ising spins in flat array
    Eigen::ArrayXd heisenberg_x;     // Heisenberg x-components
    Eigen::ArrayXd heisenberg_y;     // Heisenberg y-components  
    Eigen::ArrayXd heisenberg_z;     // Heisenberg z-components
    
    // Statistics
    long int total_attempts;
    long int total_acceptances;
    
    // Fast indexing
    inline int flatten_index(int x, int y, int z, int spin_id) const {
        int num_spins = unit_cell.num_spins();
        return ((x - 1) * lattice_size * lattice_size + (y - 1) * lattice_size + (z - 1)) * num_spins + spin_id;
    }
    
    // Fast energy calculation - direct array access
    double calculate_local_energy_fast(int x, int y, int z, int spin_id);
    
    // Fast Metropolis test
    inline bool metropolis_test_fast(double energy_change) {
        if (energy_change <= 0.0) return true;
        double probability = std::exp(-energy_change / temperature);
        return ran1(&seed) < probability;
    }

public:
    // Constructor
    MonteCarloSimulation(const UnitCell& uc,
                         const CouplingMatrix& couplings,
                         int size,
                         double T, 
                         std::optional<KK_Matrix> kk = std::nullopt);
    
    // Destructor
    ~MonteCarloSimulation() = default;
    
    // Check if Kugel-Khomskii coupling is present
    bool has_kugel_khomskii() const { return kk_matrix.has_value(); }
    
    // Main methods
    void initialize_lattice();
    void run_monte_carlo_step();
    void run_warmup_phase(int warmup_steps);
    
    // Measurements
    double get_energy();
    double get_magnetization();
    double get_absolute_magnetization();
    std::vector<double> get_magnetization_per_spin();  // Returns magnetization for each spin in unit cell
    double get_acceptance_rate() const;
    void reset_statistics();
    
    // Fast spin access for testing
    int get_ising_spin(int x, int y, int z, int spin_id) const;
    void set_ising_spin(int x, int y, int z, int spin_id, int spin);
    spin3d get_heisenberg_spin(int x, int y, int z, int spin_id) const;
    void set_heisenberg_spin(int x, int y, int z, int spin_id, const spin3d& spin);
    
    // Direct energy calculation for testing
    double calculate_local_energy(int x, int y, int z, int spin_id) {
        return calculate_local_energy_fast(x, y, z, spin_id);
    }
    
    // Metropolis test for testing
    bool metropolis_test(double energy_change) {
        return metropolis_test_fast(energy_change);
    }
};

#endif // SIMULATION_ENGINE_H