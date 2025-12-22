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
    
    // Tracked observables (updated incrementally during MC)
    double current_energy;
    double current_magnetization;
    std::vector<spin3d> current_mag_per_spin;
    
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
    void initialize_lattice_custom(const std::vector<double>& pattern);
    void initialize_lattice_random();
    void update_tracked_observables();  // Recompute tracked energy/magnetization
    void run_monte_carlo_step();
    void run_warmup_phase(int warmup_steps);
    
    // Measurements
    double get_energy();  // Recompute from scratch
    double get_magnetization();  // Recompute from scratch
    double get_tracked_energy() const { return current_energy; }  // Get tracked value (fast O(1))
    double get_tracked_magnetization() const { return current_magnetization; }  // Get tracked value (fast O(1))
    const std::vector<spin3d>& get_tracked_magnetization_vector_per_spin() const { return current_mag_per_spin; }  // Get tracked per-spin magnetization (fast O(1))
    double get_absolute_magnetization();
    std::vector<double> get_magnetization_per_spin();  // Returns z-component magnetization for each spin
    std::vector<spin3d> get_magnetization_vector_per_spin();  // Returns full vector <M> for each spin (slow O(N))
    std::vector<double> get_magnetization_magnitude_per_spin();  // Returns <|M|> for each spin type
    
    // Spin correlation measurements (for multi-walker simulations)
    // For Ising spins: returns <S_0 * S_i> where S_0 is first Ising spin (pairwise product)
    // For Heisenberg spins: returns <S_0 · S_i> where S_0 is first Heisenberg spin (dot product)
    // This is direction-independent and can be averaged across walkers with different ordering directions
    std::vector<double> get_spin_correlation_with_first();
    
    double get_acceptance_rate() const;
    void reset_statistics();
    
    // Fast spin access for testing
    int get_ising_spin(int x, int y, int z, int spin_id) const;
    void set_ising_spin(int x, int y, int z, int spin_id, int spin);
    spin3d get_heisenberg_spin(int x, int y, int z, int spin_id) const;
    void set_heisenberg_spin(int x, int y, int z, int spin_id, const spin3d& spin);
    
    // Configuration extraction for MPI averaging
    const Eigen::ArrayXd& get_ising_array() const { return ising_spins; }
    const Eigen::ArrayXd& get_heisenberg_x_array() const { return heisenberg_x; }
    const Eigen::ArrayXd& get_heisenberg_y_array() const { return heisenberg_y; }
    const Eigen::ArrayXd& get_heisenberg_z_array() const { return heisenberg_z; }
    
    Eigen::ArrayXd& get_ising_array_mutable() { return ising_spins; }
    Eigen::ArrayXd& get_heisenberg_x_array_mutable() { return heisenberg_x; }
    Eigen::ArrayXd& get_heisenberg_y_array_mutable() { return heisenberg_y; }
    Eigen::ArrayXd& get_heisenberg_z_array_mutable() { return heisenberg_z; }
    
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