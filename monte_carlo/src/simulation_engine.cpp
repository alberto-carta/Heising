/*
 * Fast Simulation Engine Implementation
 * 
 * Performance-optimized Monte Carlo simulation with direct array access
 * and simplified energy calculations
 */

#include "simulation_engine.h"
#include "spin_operations.h"
#include "random.h"
#include <iostream>
#include <cmath>
#include <iomanip>

// External seed declaration
extern long int seed;

// Constructor
MonteCarloSimulation::MonteCarloSimulation(const UnitCell& uc, 
                                          const CouplingMatrix& couplings,
                                          int size, double T,
                                          std::optional<KK_Matrix> kk)
    : lattice_size(size), temperature(T), max_rotation_angle(1.5),
      unit_cell(uc), coupling_matrix(couplings), kk_matrix(kk),
      total_attempts(0), total_acceptances(0) {
    
    std::cout << "Creating Monte Carlo simulation:" << std::endl;
    std::cout << "Unit cell: " << unit_cell.num_spins() << " spins" << std::endl;
    std::cout << "Lattice size: " << size << "x" << size << "x" << size << std::endl;
    std::cout << "Temperature: " << T << std::endl;
    
    // Print spin information
    for (int i = 0; i < unit_cell.num_spins(); i++) {
        const SpinInfo& spin = unit_cell.get_spin(i);
        std::cout << "  Spin " << i << ": " << spin.label 
                  << " (" << (spin.spin_type == SpinType::ISING ? "Ising" : "Heisenberg") 
                  << ", S=" << spin.spin_magnitude << ")" << std::endl;
    }
    
    coupling_matrix.print_summary();
    
    if (kk_matrix.has_value()) {
        std::cout << "Kugel-Khomskii coupling is enabled." << std::endl;
    }
    
    // Allocate memory
    int total_spins = size * size * size * unit_cell.num_spins();
    std::cout << "Allocating memory for " << total_spins << " spins..." << std::endl;
    
    ising_spins.resize(total_spins);
    heisenberg_x.resize(total_spins);
    heisenberg_y.resize(total_spins);
    heisenberg_z.resize(total_spins);
    
    std::cout << "FAST simulation engine initialized!" << std::endl;
}

// Initialize lattice with ordered spins (ferromagnetic ground state)
void MonteCarloSimulation::initialize_lattice() {
    std::cout << "Initializing lattice with ordered spins (ferromagnetic state)..." << std::endl;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < unit_cell.num_spins(); spin_id++) {
                    const SpinInfo& spin = unit_cell.get_spin(spin_id);
                    int idx = flatten_index(x, y, z, spin_id);
                    
                    if (spin.spin_type == SpinType::ISING) {
                        ising_spins[idx] = 1.0;  // All spins up for ferromagnet
                    } else {
                        // All spins pointing in +z direction for ferromagnet
                        heisenberg_x[idx] = 0.0;
                        heisenberg_y[idx] = 0.0;
                        heisenberg_z[idx] = 1.0;
                    }
                }
            }
        }
    }
    
    std::cout << "Lattice initialized in ferromagnetic ground state!" << std::endl;
}

// Local energy calculation 
double MonteCarloSimulation::calculate_local_energy_fast(int x, int y, int z, int spin_id) {
    double energy = 0.0;
    const SpinInfo& spin_i = unit_cell.get_spin(spin_id);
    int idx_i = flatten_index(x, y, z, spin_id);
    
    // Loop over all possible neighbor offsets efficiently
    int max_range = coupling_matrix.get_max_offset();
    for (int dx = -max_range; dx <= max_range; dx++) {
        for (int dy = -max_range; dy <= max_range; dy++) {
            for (int dz = -max_range; dz <= max_range; dz++) {
                // Skip if no coupling
                for (int spin_j = 0; spin_j < unit_cell.num_spins(); spin_j++) {
                    double J_ij = coupling_matrix.get_coupling(spin_id, spin_j, dx, dy, dz);
                    if (J_ij == 0.0) continue;
                    
                    // Calculate neighbor position with periodic boundaries
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    
                    // Apply periodic boundary conditions
                    if (nx < 1) nx += lattice_size;
                    if (nx > lattice_size) nx -= lattice_size;
                    if (ny < 1) ny += lattice_size;
                    if (ny > lattice_size) ny -= lattice_size;
                    if (nz < 1) nz += lattice_size;
                    if (nz > lattice_size) nz -= lattice_size;
                    
                    int idx_j = flatten_index(nx, ny, nz, spin_j);
                    const SpinInfo& spin_j_info = unit_cell.get_spin(spin_j);
                    
                    // Fast dot product calculation
                    double dot_product = 0.0;
                    if (spin_i.spin_type == SpinType::ISING && spin_j_info.spin_type == SpinType::ISING) {
                        dot_product = ising_spins[idx_i] * ising_spins[idx_j];
                    } else if (spin_i.spin_type == SpinType::HEISENBERG && spin_j_info.spin_type == SpinType::HEISENBERG) {
                        dot_product = heisenberg_x[idx_i] * heisenberg_x[idx_j] +
                                     heisenberg_y[idx_i] * heisenberg_y[idx_j] +
                                     heisenberg_z[idx_i] * heisenberg_z[idx_j];
                    } else {
                        // Mixed Ising-Heisenberg interaction
                        if (spin_i.spin_type == SpinType::ISING) {
                            dot_product = ising_spins[idx_i] * heisenberg_z[idx_j];  // Use z-component
                        } else {
                            dot_product = heisenberg_z[idx_i] * ising_spins[idx_j];  // Use z-component
                        }
                    }
                    
                    energy += J_ij * dot_product;
                }
            }
        }
    }
    
    return energy;
}

// Fast Monte Carlo step
void MonteCarloSimulation::run_monte_carlo_step() {
    // Random site selection
    int x = static_cast<int>(ran1(&seed) * lattice_size) + 1;
    int y = static_cast<int>(ran1(&seed) * lattice_size) + 1;
    int z = static_cast<int>(ran1(&seed) * lattice_size) + 1;
    int spin_id = static_cast<int>(ran1(&seed) * unit_cell.num_spins());
    
    const SpinInfo& spin = unit_cell.get_spin(spin_id);
    int idx = flatten_index(x, y, z, spin_id);
    
    // Store old values
    double old_ising = ising_spins[idx];
    double old_hx = heisenberg_x[idx];
    double old_hy = heisenberg_y[idx];
    double old_hz = heisenberg_z[idx];
    
    // Calculate energy before move
    double energy_before = calculate_local_energy_fast(x, y, z, spin_id);
    
    // Propose move
    if (spin.spin_type == SpinType::ISING) {
        ising_spins[idx] = -ising_spins[idx];  // Flip
    } else {
        // Rotate Heisenberg spin
        spin3d old_spin(old_hx, old_hy, old_hz);
        spin3d new_spin = small_random_change(old_spin, max_rotation_angle);
        heisenberg_x[idx] = new_spin.x;
        heisenberg_y[idx] = new_spin.y;
        heisenberg_z[idx] = new_spin.z;
    }
    
    // Calculate energy after move
    double energy_after = calculate_local_energy_fast(x, y, z, spin_id);
    double energy_change = energy_after - energy_before;
    
    total_attempts++;
    
    // Metropolis test
    if (!metropolis_test_fast(energy_change)) {
        // Reject - restore old values
        ising_spins[idx] = old_ising;
        heisenberg_x[idx] = old_hx;
        heisenberg_y[idx] = old_hy;
        heisenberg_z[idx] = old_hz;
    } else {
        total_acceptances++;
    }
}

// Warmup phase
void MonteCarloSimulation::run_warmup_phase(int warmup_steps) {
    std::cout << "Running fast warmup phase (" << warmup_steps << " steps)..." << std::endl;
    
    for (int step = 0; step < warmup_steps; step++) {
        run_monte_carlo_step();
        
        // Progress feedback every 10%
        if (warmup_steps > 1000 && step % (warmup_steps / 10) == 0) {
            double progress = 100.0 * step / warmup_steps;
            std::cout << "Warmup: " << step << "/" << warmup_steps 
                      << " (" << std::fixed << std::setprecision(2) << progress << "%)" << std::endl;
        }
    }
    
    std::cout << "Fast warmup phase completed." << std::endl;
}

// Energy calculation
double MonteCarloSimulation::get_energy() {
    double total_energy = 0.0;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < unit_cell.num_spins(); spin_id++) {
                    total_energy += calculate_local_energy_fast(x, y, z, spin_id);
                }
            }
        }
    }
    
    return total_energy / 2.0;  // Avoid double counting
}

// Magnetization calculation
double MonteCarloSimulation::get_magnetization() {
    double total_mag = 0.0;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < unit_cell.num_spins(); spin_id++) {
                    const SpinInfo& spin = unit_cell.get_spin(spin_id);
                    int idx = flatten_index(x, y, z, spin_id);
                    
                    if (spin.spin_type == SpinType::ISING) {
                        total_mag += ising_spins[idx] * spin.spin_magnitude;
                    } else {
                        // For Heisenberg, use z-component as convention (like Ising)
                        total_mag += heisenberg_z[idx] * spin.spin_magnitude;
                    }
                }
            }
        }
    }
    
    return total_mag;
}

// Absolute magnetization
double MonteCarloSimulation::get_absolute_magnetization() {
    double total_mag = 0.0;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < unit_cell.num_spins(); spin_id++) {
                    const SpinInfo& spin = unit_cell.get_spin(spin_id);
                    int idx = flatten_index(x, y, z, spin_id);
                    
                    if (spin.spin_type == SpinType::ISING) {
                        total_mag += std::abs(ising_spins[idx]) * spin.spin_magnitude;
                    } else {
                        // For Heisenberg, use magnitude of the 3D vector
                        double spin_mag = sqrt(heisenberg_x[idx]*heisenberg_x[idx] + 
                                             heisenberg_y[idx]*heisenberg_y[idx] + 
                                             heisenberg_z[idx]*heisenberg_z[idx]);
                        total_mag += spin_mag * spin.spin_magnitude;
                    }
                }
            }
        }
    }
    
    return total_mag;
}

// Acceptance rate
double MonteCarloSimulation::get_acceptance_rate() const {
    if (total_attempts == 0) return 0.0;
    return 100.0 * total_acceptances / total_attempts;
}

// Reset statistics
void MonteCarloSimulation::reset_statistics() {
    total_attempts = 0;
    total_acceptances = 0;
}

// Spin access methods (for testing)
int MonteCarloSimulation::get_ising_spin(int x, int y, int z, int spin_id) const {
    int idx = flatten_index(x, y, z, spin_id);
    return static_cast<int>(ising_spins[idx]);
}

void MonteCarloSimulation::set_ising_spin(int x, int y, int z, int spin_id, int spin) {
    int idx = flatten_index(x, y, z, spin_id);
    ising_spins[idx] = static_cast<double>(spin);
}

spin3d MonteCarloSimulation::get_heisenberg_spin(int x, int y, int z, int spin_id) const {
    int idx = flatten_index(x, y, z, spin_id);
    return spin3d(heisenberg_x[idx], heisenberg_y[idx], heisenberg_z[idx]);
}

void MonteCarloSimulation::set_heisenberg_spin(int x, int y, int z, int spin_id, const spin3d& spin) {
    int idx = flatten_index(x, y, z, spin_id);
    heisenberg_x[idx] = spin.x;
    heisenberg_y[idx] = spin.y;
    heisenberg_z[idx] = spin.z;
}