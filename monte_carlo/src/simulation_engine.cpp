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
    std::cout << std::fixed << std::setprecision(4);
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
        kk_matrix->print_summary();
        // Set energy dispatch pointer to KK version
        energy_dispatch_ptr = &MonteCarloSimulation::calculate_local_energy_implementation<true>;
    }

    else {
        std::cout << "No Kugel-Khomskii coupling." << std::endl;
        // Set energy dispatch pointer to non-KK version
        energy_dispatch_ptr = &MonteCarloSimulation::calculate_local_energy_implementation<false>;
    }
    
    // Allocate memory
    int total_spins = size * size * size * unit_cell.num_spins();
    std::cout << "Allocating memory for " << total_spins << " spins..." << std::endl;
    
    ising_spins.resize(total_spins);
    heisenberg_x.resize(total_spins);
    heisenberg_y.resize(total_spins);
    heisenberg_z.resize(total_spins);
    
    // Initialize tracked observables
    current_mag_per_spin.resize(unit_cell.num_spins(), spin3d(0, 0, 0));
    current_energy = 0.0;
    current_magnetization = 0.0;
    
    std::cout << "Simulation engine initialized!" << std::endl;
}

// Initialize lattice with ordered spins 
void MonteCarloSimulation::initialize_lattice_custom(const std::vector<double>& pattern) {
    std::cout << "Initializing lattice with custom pattern [";
    for (size_t i = 0; i < pattern.size(); i++) {
        std::cout << pattern[i];
        if (i < pattern.size() - 1) std::cout << ", ";
    }
    std::cout << "]..." << std::endl;
    
    int num_spins_per_cell = unit_cell.num_spins();
    if (pattern.size() != static_cast<size_t>(num_spins_per_cell)) {
        std::cerr << "ERROR: Pattern size (" << pattern.size() 
                  << ") does not match number of spins in unit cell (" 
                  << num_spins_per_cell << ")" << std::endl;
        throw std::runtime_error("Invalid initialization pattern size");
    }
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < num_spins_per_cell; spin_id++) {
                    const SpinInfo& spin = unit_cell.get_spin(spin_id);
                    int idx = flatten_index(x, y, z, spin_id);
                    double value = pattern[spin_id];
                    
                    if (spin.spin_type == SpinType::ISING) {
                        // For Ising: use sign of pattern value
                        ising_spins[idx] = (value >= 0) ? 1.0 : -1.0;
                    } else {
                        // For Heisenberg: use pattern value as sz, normalize
                        heisenberg_x[idx] = 0.0;
                        heisenberg_y[idx] = 0.0;
                        heisenberg_z[idx] = (value >= 0) ? 1.0 : -1.0;
                    }
                }
            }
        }
    }
    
    std::cout << "Lattice initialized with custom pattern!" << std::endl;
    update_tracked_observables();
}

void MonteCarloSimulation::initialize_lattice_random() {
    std::cout << "Initializing lattice with random spin orientations..." << std::endl;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < unit_cell.num_spins(); spin_id++) {
                    const SpinInfo& spin = unit_cell.get_spin(spin_id);
                    int idx = flatten_index(x, y, z, spin_id);
                    
                    if (spin.spin_type == SpinType::ISING) {
                        ising_spins[idx] = (ran1(&seed) < 0.5) ? 1.0 : -1.0;
                    } else {
                        // Random point on unit sphere
                        heisenberg_x[idx] = 0.0;
                        heisenberg_y[idx] = 0.0;
                        heisenberg_z[idx] = (ran1(&seed) < 0.5) ? 1.0 : -1.0;
                        // For true random, would use:
                        // double theta = acos(2.0 * ran1(&seed) - 1.0);
                        // double phi = 2.0 * M_PI * ran1(&seed);
                        // But keeping it simple for now
                    }
                }
            }
        }
    }
    
    std::cout << "Lattice initialized with random spins!" << std::endl;
    update_tracked_observables();
}

// Update tracked observables by recomputing from current configuration
void MonteCarloSimulation::update_tracked_observables() {
    current_energy = get_energy();
    current_magnetization = get_magnetization();
    current_mag_per_spin = get_magnetization_vector_per_spin();
}

// Local energy calculation 
template <bool use_kk>
double MonteCarloSimulation::calculate_local_energy_implementation(int x, int y, int z, int spin_id) {
    double energy = 0.0;
    
    // Standard coupling contribution (J_ij * S_i · S_j)
    energy += compute_coupling_contribution(x, y, z, spin_id);
    
    // Kugel-Khomskii coupling (if enabled)
    // This depends only on site_i and relative distances, not absolute position
    if constexpr (use_kk) {
        const SpinInfo& spin_i = unit_cell.get_spin(spin_id);
        int site_i = spin_i.site_id;
        double kk_energy = compute_kk_contribution(x, y, z, site_i);
        energy += kk_energy;
    }

    return energy;
}

// Standard coupling contribution to local energy (J_ij * S_i · S_j)
double MonteCarloSimulation::compute_coupling_contribution(int x, int y, int z, int spin_id) {
    double energy = 0.0;
    const SpinInfo& spin_i = unit_cell.get_spin(spin_id);
    int idx_i = flatten_index(x, y, z, spin_id);
    
    // Loop over all possible neighbor offsets efficiently
    int max_range = coupling_matrix.get_max_offset();
    for (int dx = -max_range; dx <= max_range; dx++) {
        for (int dy = -max_range; dy <= max_range; dy++) {
            for (int dz = -max_range; dz <= max_range; dz++) {
                // Loop over all spins in neighbor unit cells
                for (int spin_j = 0; spin_j < unit_cell.num_spins(); spin_j++) {
                    double J_ij = coupling_matrix.get_coupling(spin_id, spin_j, dx, dy, dz);
                    if (J_ij == 0.0) continue; // Skip if no coupling
                    
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

// Kugel-Khomskii contribution to local energy at site_i
// K(site_i, site_j, dx, dy, dz) * (S_i · S_j) * (τ_i * τ_j)
// where S is Heisenberg spin and τ is Ising spin at each site
// Note: This only depends on site_i and relative unit cell distances, not on absolute (x,y,z)
double MonteCarloSimulation::compute_kk_contribution(int x, int y, int z, int site_i) {
    double kk_energy = 0.0;
    
    if (!kk_matrix.has_value()) return 0.0;
    
    // Get spins at site_i - need exactly 2 (1 Heisenberg + 1 Ising)
    std::vector<int> spins_i = unit_cell.get_spins_at_site(site_i);
    if (spins_i.size() != 2) return 0.0;  // KK requires exactly 2 spins per site
    
    // Identify Heisenberg and Ising spins at site_i
    int heis_i = -1, ising_i = -1;
    for (int spin_id : spins_i) {
        const SpinInfo& spin = unit_cell.get_spin(spin_id);
        if (spin.spin_type == SpinType::HEISENBERG) heis_i = spin_id;
        else if (spin.spin_type == SpinType::ISING) ising_i = spin_id;
    }
    
    if (heis_i == -1 || ising_i == -1) return 0.0;  // Need both types
    
    // Get indices for site_i spins
    int idx_heis_i = flatten_index(x, y, z, heis_i);
    int idx_ising_i = flatten_index(x, y, z, ising_i);
    
    // Loop over all possible neighbor offsets
    int max_range = kk_matrix->get_max_offset();
    int num_sites = unit_cell.get_num_sites();
    
    for (int dx = -max_range; dx <= max_range; dx++) {
        for (int dy = -max_range; dy <= max_range; dy++) {
            for (int dz = -max_range; dz <= max_range; dz++) {
                
                // Calculate neighbor position with periodic boundaries
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                
                if (nx < 1) nx += lattice_size;
                if (nx > lattice_size) nx -= lattice_size;
                if (ny < 1) ny += lattice_size;
                if (ny > lattice_size) ny -= lattice_size;
                if (nz < 1) nz += lattice_size;
                if (nz > lattice_size) nz -= lattice_size;
                
                // Loop over all neighbor sites
                for (int site_j = 0; site_j < num_sites; site_j++) {
                    double K_ij = kk_matrix->get_coupling(site_i, site_j, dx, dy, dz);
                    if (K_ij == 0.0) continue;
                    
                    // Get spins at site_j
                    std::vector<int> spins_j = unit_cell.get_spins_at_site(site_j);
                    if (spins_j.size() != 2) continue;
                    
                    // Identify Heisenberg and Ising spins at site_j
                    int heis_j = -1, ising_j = -1;
                    for (int spin_id : spins_j) {
                        const SpinInfo& spin = unit_cell.get_spin(spin_id);
                        if (spin.spin_type == SpinType::HEISENBERG) heis_j = spin_id;
                        else if (spin.spin_type == SpinType::ISING) ising_j = spin_id;
                    }
                    
                    if (heis_j == -1 || ising_j == -1) continue;
                    
                    // Get array indices for site_j spins
                    int idx_heis_j = flatten_index(nx, ny, nz, heis_j);
                    int idx_ising_j = flatten_index(nx, ny, nz, ising_j);
                    
                    // Compute (S_i · S_j) - dot product of Heisenberg spins
                    double S_dot_S = heisenberg_x[idx_heis_i] * heisenberg_x[idx_heis_j] +
                                     heisenberg_y[idx_heis_i] * heisenberg_y[idx_heis_j] +
                                     heisenberg_z[idx_heis_i] * heisenberg_z[idx_heis_j];
                    
                    // Compute (τ_i * τ_j) - product of Ising spins
                    double tau_i_tau_j = ising_spins[idx_ising_i] * ising_spins[idx_ising_j];
                    
                    // KK contribution: K * (S_i · S_j) * (τ_i * τ_j)
                    kk_energy += K_ij * S_dot_S * tau_i_tau_j;
                }
            }
        }
    }
    
    return kk_energy;
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
        
        // Update tracked observables incrementally
        current_energy += energy_change;
        
        // Update magnetization (change from old to new spin)
        if (spin.spin_type == SpinType::ISING) {
            double mag_change = (ising_spins[idx] - old_ising) * spin.spin_magnitude;  // +2S or -2S
            current_magnetization += mag_change;
            current_mag_per_spin[spin_id].z += mag_change / (lattice_size * lattice_size * lattice_size);  // Track per-spin average
        } else {
            // For Heisenberg, mag change is vector difference
            double mag_change_x = heisenberg_x[idx] - old_hx;
            double mag_change_y = heisenberg_y[idx] - old_hy;
            double mag_change_z = heisenberg_z[idx] - old_hz;
            double old_mag = std::sqrt(old_hx*old_hx + old_hy*old_hy + old_hz*old_hz);
            double new_mag = std::sqrt(heisenberg_x[idx]*heisenberg_x[idx] + 
                                      heisenberg_y[idx]*heisenberg_y[idx] + 
                                      heisenberg_z[idx]*heisenberg_z[idx]);
            current_magnetization += (new_mag - old_mag) * spin.spin_magnitude;
            // Track per-spin average (divided by number of cells)
            double norm = 1.0 / (lattice_size * lattice_size * lattice_size);
            current_mag_per_spin[spin_id].x += mag_change_x * spin.spin_magnitude * norm;
            current_mag_per_spin[spin_id].y += mag_change_y * spin.spin_magnitude * norm;
            current_mag_per_spin[spin_id].z += mag_change_z * spin.spin_magnitude * norm;
        }
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
    
    std::cout << "Warmup phase completed." << std::endl;
}

// Energy calculation
double MonteCarloSimulation::get_energy() {
    double total_energy = 0.0;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < unit_cell.num_spins(); spin_id++) {
                    total_energy += calculate_local_energy_fast(x, y, z, spin_id); // this is double counting the KK contribution
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

// Per-spin magnetization - returns magnetization for each spin type in unit cell
std::vector<double> MonteCarloSimulation::get_magnetization_per_spin() {
    int num_spins = unit_cell.num_spins();
    std::vector<double> mag_per_spin(num_spins, 0.0);
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < num_spins; spin_id++) {
                    const SpinInfo& spin = unit_cell.get_spin(spin_id);
                    int idx = flatten_index(x, y, z, spin_id);
                    
                    if (spin.spin_type == SpinType::ISING) {
                        mag_per_spin[spin_id] += ising_spins[idx] * spin.spin_magnitude;
                    } else {
                        // For Heisenberg, use z-component (like total magnetization)
                        mag_per_spin[spin_id] += heisenberg_z[idx] * spin.spin_magnitude;
                    }
                }
            }
        }
    }
    
    // Normalize by number of unit cells
    int total_cells = lattice_size * lattice_size * lattice_size;
    for (int i = 0; i < num_spins; i++) {
        mag_per_spin[i] /= total_cells;
    }
    
    return mag_per_spin;
}

// Per-spin magnetization vector - returns full <M_x>, <M_y>, <M_z> for each spin
std::vector<spin3d> MonteCarloSimulation::get_magnetization_vector_per_spin() {
    int num_spins = unit_cell.num_spins();
    std::vector<spin3d> mag_vec_per_spin(num_spins, spin3d(0.0, 0.0, 0.0));
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int spin_id = 0; spin_id < num_spins; spin_id++) {
                    const SpinInfo& spin = unit_cell.get_spin(spin_id);
                    int idx = flatten_index(x, y, z, spin_id);
                    
                    if (spin.spin_type == SpinType::ISING) {
                        // Ising: only z-component
                        mag_vec_per_spin[spin_id].z += ising_spins[idx] * spin.spin_magnitude;
                    } else {
                        // Heisenberg: full vector
                        mag_vec_per_spin[spin_id].x += heisenberg_x[idx] * spin.spin_magnitude;
                        mag_vec_per_spin[spin_id].y += heisenberg_y[idx] * spin.spin_magnitude;
                        mag_vec_per_spin[spin_id].z += heisenberg_z[idx] * spin.spin_magnitude;
                    }
                }
            }
        }
    }
    
    // Normalize by number of unit cells
    int total_cells = lattice_size * lattice_size * lattice_size;
    for (int i = 0; i < num_spins; i++) {
        mag_vec_per_spin[i].x /= total_cells;
        mag_vec_per_spin[i].y /= total_cells;
        mag_vec_per_spin[i].z /= total_cells;
    }
    
    return mag_vec_per_spin;
}

// Average magnitude of magnetization per spin type: <|M|>
std::vector<double> MonteCarloSimulation::get_magnetization_magnitude_per_spin() {
    int num_spin_types = unit_cell.num_spins();
    std::vector<double> magnitudes(num_spin_types, 0.0);
    int total_cells = lattice_size * lattice_size * lattice_size;
    
    // For each spin type, compute average magnitude across all unit cells
    for (int spin_id = 0; spin_id < num_spin_types; spin_id++) {
        const SpinInfo& spin = unit_cell.get_spin(spin_id);
        double sum_magnitude = 0.0;
        
        for (int x = 1; x <= lattice_size; x++) {
            for (int y = 1; y <= lattice_size; y++) {
                for (int z = 1; z <= lattice_size; z++) {
                    int idx = flatten_index(x, y, z, spin_id);
                    
                    if (spin.spin_type == SpinType::ISING) {
                        sum_magnitude += std::abs(ising_spins[idx]);
                    } else {
                        // For Heisenberg, compute magnitude of this spin's vector
                        double mag = sqrt(heisenberg_x[idx]*heisenberg_x[idx] + 
                                        heisenberg_y[idx]*heisenberg_y[idx] + 
                                        heisenberg_z[idx]*heisenberg_z[idx]);
                        sum_magnitude += mag;
                    }
                }
            }
        }
        
        magnitudes[spin_id] = sum_magnitude / total_cells;
    }
    
    return magnitudes;
}

// Compute spin correlations for multi-walker simulations
// For Ising spins: returns <S_0 * S_i> where S_0 is first Ising spin (pairwise correlation)
// For Heisenberg spins: returns <S_0 · S_i> where S_0 is first Heisenberg spin (dot product)
// This is direction-independent and safe to average across walkers.
std::vector<double> MonteCarloSimulation::get_spin_correlation_with_first() {
    int num_spin_types = unit_cell.num_spins();
    std::vector<double> correlations(num_spin_types, 0.0);
    int total_cells = lattice_size * lattice_size * lattice_size;
    
    // Find first Ising and first Heisenberg spin for correlation reference
    int first_ising_idx = -1;
    int first_heisenberg_idx = -1;
    for (int i = 0; i < num_spin_types; i++) {
        if (unit_cell.get_spin(i).spin_type == SpinType::ISING && first_ising_idx == -1) {
            first_ising_idx = i;
        }
        if (unit_cell.get_spin(i).spin_type == SpinType::HEISENBERG && first_heisenberg_idx == -1) {
            first_heisenberg_idx = i;
        }
    }
    
    // For each spin type i, compute appropriate measurement
    for (int spin_i = 0; spin_i < num_spin_types; spin_i++) {
        const SpinInfo& spin_i_info = unit_cell.get_spin(spin_i);
        
        if (spin_i_info.spin_type == SpinType::ISING) {
            // For Ising: compute correlation <S_0 * S_i> with first Ising spin
            // This remains direction-independent across walkers (product of ±1 values)
            if (first_ising_idx == -1) {
                correlations[spin_i] = 0.0;
                continue;
            }
            
            double sum_products = 0.0;
            int num_pairs = 0;
            
            // Loop over all positions for first Ising spin
            for (int x0 = 1; x0 <= lattice_size; x0++) {
                for (int y0 = 1; y0 <= lattice_size; y0++) {
                    for (int z0 = 1; z0 <= lattice_size; z0++) {
                        int idx_0 = flatten_index(x0, y0, z0, first_ising_idx);
                        double s0 = ising_spins[idx_0];
                        
                        // Loop over all positions for spin_i
                        for (int xi = 1; xi <= lattice_size; xi++) {
                            for (int yi = 1; yi <= lattice_size; yi++) {
                                for (int zi = 1; zi <= lattice_size; zi++) {
                                    int idx_i = flatten_index(xi, yi, zi, spin_i);
                                    double si = ising_spins[idx_i];
                                    
                                    sum_products += s0 * si;
                                    num_pairs++;
                                }
                            }
                        }
                    }
                }
            }
            correlations[spin_i] = sum_products / num_pairs;
            
        } else {
            // For Heisenberg: compute correlation with first Heisenberg spin
            if (first_heisenberg_idx == -1) {
                // Should not happen, but handle gracefully
                correlations[spin_i] = 0.0;
                continue;
            }
            
            double sum_dot_products = 0.0;
            int num_pairs = 0;
            
            // Loop over all positions for first Heisenberg spin
            for (int x0 = 1; x0 <= lattice_size; x0++) {
                for (int y0 = 1; y0 <= lattice_size; y0++) {
                    for (int z0 = 1; z0 <= lattice_size; z0++) {
                        int idx_0 = flatten_index(x0, y0, z0, first_heisenberg_idx);
                        
                        // Get first Heisenberg spin components
                        double s0_x = heisenberg_x[idx_0];
                        double s0_y = heisenberg_y[idx_0];
                        double s0_z = heisenberg_z[idx_0];
                        
                        // Loop over all positions for spin_i
                        for (int xi = 1; xi <= lattice_size; xi++) {
                            for (int yi = 1; yi <= lattice_size; yi++) {
                                for (int zi = 1; zi <= lattice_size; zi++) {
                                    int idx_i = flatten_index(xi, yi, zi, spin_i);
                                    
                                    // Get spin_i components
                                    double si_x = heisenberg_x[idx_i];
                                    double si_y = heisenberg_y[idx_i];
                                    double si_z = heisenberg_z[idx_i];
                                    
                                    // Compute dot product
                                    double dot_product = s0_x * si_x + s0_y * si_y + s0_z * si_z;
                                    
                                    sum_dot_products += dot_product;
                                    num_pairs++;
                                }
                            }
                        }
                    }
                }
            }
            
            // Average over all pairs
            correlations[spin_i] = sum_dot_products / num_pairs;
        }
    }
    
    return correlations;
}

// Absolute magnetization

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