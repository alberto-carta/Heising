/*
 * Monte Carlo Simulation Engine Implementation
 * 
 * This file contains the actual physics implementations for both Ising and Heisenberg models.
 * The key idea is that we set up function pointers once in the constructor, then the same
 * Monte Carlo algorithm works for both models.
 */

#include "simulation_engine.h"
#include "spin_operations.h"
#include "random.h"
#include <iostream>
#include <iomanip>
#include <cmath>

extern long int seed;  // Use the same random seed as the rest of the program

// Constructor - this is where the magic happens!
MonteCarloSimulation::MonteCarloSimulation(SpinType type, int size, double T, double J) 
    : model_type(type), lattice_size(size), temperature(T), coupling_J(J), max_rotation_angle(0.5) {
    

    // this is equivalent to:
    // model_type = type;
    // lattice_size = size;
    // temperature = T;
    // coupling_J = J;
    // max_rotation_angle = 0.5;  // hardcoded for now

    std::cout << "Creating Monte Carlo simulation:" << std::endl;
    std::cout << "Model: ";
    switch (type) {
        case SpinType::ISING:
            std::cout << "Ising";
            break;
        case SpinType::HEISENBERG:
            std::cout << "Heisenberg";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Lattice size: " << size << "x" << size << std::endl;
    std::cout << "Temperature: " << T << ", Coupling: " << J << std::endl;
    
    // Allocate memory based on model type
    allocate_memory();
    
    // Set up function pointers for the Monte Carlo hot path
    setup_function_pointers();
    
    std::cout << "Simulation engine initialized successfully!" << std::endl;
}

// Set up function pointers for the Monte Carlo hot path - avoids if/else in inner loop
void MonteCarloSimulation::setup_function_pointers() {
    switch (model_type) {
        case SpinType::ISING:
            calculate_local_energy = &MonteCarloSimulation::ising_local_energy;
            propose_spin_flip = &MonteCarloSimulation::ising_propose_flip;
            break;
            
        case SpinType::HEISENBERG:
            calculate_local_energy = &MonteCarloSimulation::heisenberg_local_energy;
            propose_spin_flip = &MonteCarloSimulation::heisenberg_propose_flip;
            break;
    }
}

// Memory allocation - only allocate what we need based on model type
void MonteCarloSimulation::allocate_memory() {
    std::cout << "Allocating memory for " << lattice_size << "x" << lattice_size << " lattice..." << std::endl;
    
    // Initialize both pointers to nullptr for safety
    ising_lattice = nullptr;
    heisenberg_lattice = nullptr;
    
    switch (model_type) {
        case SpinType::ISING:
            std::cout << "Allocating memory for Ising spins (integers)..." << std::endl;
            // Allocate 2D array for Ising spins (integers: +1 or -1)
            ising_lattice = new int*[lattice_size + 1];  // +1 for boundary conditions
            for (int i = 0; i <= lattice_size; i++) {
                ising_lattice[i] = new int[lattice_size + 1];
            }
            break;
            
        case SpinType::HEISENBERG:
            std::cout << "Allocating memory for Heisenberg spins (3D vectors)..." << std::endl;
            // Allocate 2D array for Heisenberg spins (3D vectors)
            heisenberg_lattice = new spin3d*[lattice_size + 1];
            for (int i = 0; i <= lattice_size; i++) {
                heisenberg_lattice[i] = new spin3d[lattice_size + 1];
            }
            break;
            
        default:
            std::cerr << "Error: Cannot allocate memory for unknown model type!" << std::endl;
            exit(1);
    }
    
    std::cout << "Memory allocated successfully!" << std::endl;
}

// Destructor - clean up allocated memory
MonteCarloSimulation::~MonteCarloSimulation() {
    deallocate_memory();
}

void MonteCarloSimulation::deallocate_memory() {
    switch (model_type) {
        case SpinType::ISING:
            if (ising_lattice != nullptr) {
                for (int i = 0; i <= lattice_size; i++) {
                    delete[] ising_lattice[i];
                }
                delete[] ising_lattice;
                ising_lattice = nullptr;
            }
            break;
            
        case SpinType::HEISENBERG:
            if (heisenberg_lattice != nullptr) {
                for (int i = 0; i <= lattice_size; i++) {
                    delete[] heisenberg_lattice[i];
                }
                delete[] heisenberg_lattice;
                heisenberg_lattice = nullptr;
            }
            break;
            
        default:
            // Nothing to deallocate for unknown types
            break;
    }
}

// ========================================================================================
// ISING MODEL IMPLEMENTATIONS
// ========================================================================================

// Calculate the local energy of an Ising spin at position pos
// This is the energy due to interactions with nearest neighbors
double MonteCarloSimulation::ising_local_energy(lat_type pos) {
    // Get the spin value at this position (+1 or -1)
    int current_spin = ising_lattice[pos.x][pos.y];
    
    // Calculate neighbor positions with periodic boundary conditions
    int left   = (pos.x == 1) ? lattice_size : pos.x - 1;
    int right  = (pos.x == lattice_size) ? 1 : pos.x + 1;
    int up     = (pos.y == lattice_size) ? 1 : pos.y + 1;
    int down   = (pos.y == 1) ? lattice_size : pos.y - 1;
    
    // Sum the neighboring spins
    int neighbor_sum = ising_lattice[left][pos.y]  + ising_lattice[right][pos.y] +
                       ising_lattice[pos.x][up]   + ising_lattice[pos.x][down];
    
    // Ising interaction energy: E = +J * spin * sum_of_neighbors  
    // J > 0: antiferromagnetic (favors opposite neighbors)
    // J < 0: ferromagnetic (favors same neighbors)
    double energy = +coupling_J * current_spin * neighbor_sum;
    
    return energy;
}

// Propose a spin flip for Ising model - simply flip +1 to -1 or vice versa
void MonteCarloSimulation::ising_propose_flip(lat_type pos) {
    // Ising model: spin flip means multiply by -1
    // +1 becomes -1, -1 becomes +1
    ising_lattice[pos.x][pos.y] = -ising_lattice[pos.x][pos.y];
    
    // That's it! Ising moves are simple - just flip the spin
}

// Calculate total energy of the entire Ising lattice
double MonteCarloSimulation::ising_total_energy() {
    double total_energy = 0.0;
    lat_type pos;
    
    // Sum over all lattice sites
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            pos.x = x;
            pos.y = y;
            
            // Add the local energy at this position
            total_energy += ising_local_energy(pos);
        }
    }
    
    // Divide by 2 because each bond is counted twice
    // (once from each end of the bond)
    return total_energy / 2.0;
}

// Calculate total magnetization of the Ising lattice
double MonteCarloSimulation::ising_total_magnetization() {
    double total_magnetization = 0.0;
    
    // Sum all spin values (+1 or -1)
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            total_magnetization += ising_lattice[x][y];
        }
    }
    
    return total_magnetization;
}

// ========================================================================================
// HEISENBERG MODEL IMPLEMENTATIONS
// ========================================================================================

// Calculate the local energy of a Heisenberg spin at position pos
// This uses dot products between 3D spin vectors
double MonteCarloSimulation::heisenberg_local_energy(lat_type pos) {
    // Get the current 3D spin vector at this position
    spin3d current_spin = heisenberg_lattice[pos.x][pos.y];
    
    // Calculate neighbor positions with periodic boundary conditions
    int left   = (pos.x == 1) ? lattice_size : pos.x - 1;
    int right  = (pos.x == lattice_size) ? 1 : pos.x + 1;
    int up     = (pos.y == lattice_size) ? 1 : pos.y + 1;
    int down   = (pos.y == 1) ? lattice_size : pos.y - 1;
    
    // Sum the dot products with all 4 nearest neighbors
    double neighbor_dot_sum = 0.0;
    neighbor_dot_sum += current_spin.dot(heisenberg_lattice[left][pos.y]);   // Left neighbor
    neighbor_dot_sum += current_spin.dot(heisenberg_lattice[right][pos.y]);  // Right neighbor  
    neighbor_dot_sum += current_spin.dot(heisenberg_lattice[pos.x][up]);     // Up neighbor
    neighbor_dot_sum += current_spin.dot(heisenberg_lattice[pos.x][down]);   // Down neighbor
    
    // Heisenberg interaction energy: E = +J * (S_i · S_j) summed over neighbors
    // J > 0: antiferromagnetic (favors antiparallel spins, negative dot products)
    // J < 0: ferromagnetic (favors parallel spins, positive dot products)  
    double energy = +coupling_J * neighbor_dot_sum;
    
    return energy;
}

// Propose a spin flip for Heisenberg model - hybrid strategy with two move types
void MonteCarloSimulation::heisenberg_propose_flip(lat_type pos) {
    // Simple hybrid strategy: mix small rotations with spin flips
    // Hardcoded 10% spin flips for now (can be made adjustable later)
    
    if (ran1(&seed) < 0.1) {  // 10% of moves are spin flips
        // SPIN FLIP: Reverse the spin direction (multiply by -1)
        // This is analogous to Ising flip but in 3D: S → -S
        spin3d current_spin = heisenberg_lattice[pos.x][pos.y];
        heisenberg_lattice[pos.x][pos.y].x = -current_spin.x;
        heisenberg_lattice[pos.x][pos.y].y = -current_spin.y;
        heisenberg_lattice[pos.x][pos.y].z = -current_spin.z;
        
    } else {  // 90% of moves are small rotations
        // LOCAL ROTATION: Small rotation around current orientation
        spin3d current_spin = heisenberg_lattice[pos.x][pos.y];
        spin3d new_spin = small_random_change(current_spin, max_rotation_angle);
        heisenberg_lattice[pos.x][pos.y] = new_spin;
    }
}

// Calculate total energy of the entire Heisenberg lattice
double MonteCarloSimulation::heisenberg_total_energy() {
    double total_energy = 0.0;
    lat_type pos;
    
    // Sum over all lattice sites
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            pos.x = x;
            pos.y = y;
            
            // Add the local energy at this position
            total_energy += heisenberg_local_energy(pos);
        }
    }
    
    // Divide by 2 because each bond is counted twice (same as Ising)
    return total_energy / 2.0;
}

// Calculate total magnetization of the Heisenberg lattice
double MonteCarloSimulation::heisenberg_total_magnetization() {
    // For Heisenberg model, magnetization is the magnitude of the vector sum
    spin3d total_moment(0.0, 0.0, 0.0);  // Start with zero vector
    
    // Sum all spin vectors
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            spin3d current_spin = heisenberg_lattice[x][y];
            total_moment.x += current_spin.x;
            total_moment.y += current_spin.y;  
            total_moment.z += current_spin.z;
        }
    }
    
    // Return the magnitude of the total magnetization vector
    return total_moment.magnitude();
}

// ========================================================================================
// PUBLIC INTERFACE METHODS - These use the function pointers!
// ========================================================================================

// Initialize the lattice with random spin configurations
void MonteCarloSimulation::initialize_lattice() {
    std::cout << "Initializing lattice with random spins..." << std::endl;
    
    switch (model_type) {
        case SpinType::ISING:
            // Initialize Ising spins randomly to +1 or -1
            for (int x = 1; x <= lattice_size; x++) {
                for (int y = 1; y <= lattice_size; y++) {
                    if (ran1(&seed) >= 0.5) {
                        ising_lattice[x][y] = 1;
                    } else {
                        ising_lattice[x][y] = -1;
                    }
                }
            }
            break;
            
        case SpinType::HEISENBERG:
            // Initialize Heisenberg spins to random unit vectors
            for (int x = 1; x <= lattice_size; x++) {
                for (int y = 1; y <= lattice_size; y++) {
                    heisenberg_lattice[x][y] = random_unit_vector();
                }
            }
            break;
    }
    
    std::cout << "Lattice initialized!" << std::endl;
}

// Run a single Monte Carlo step - clear and simple!
void MonteCarloSimulation::run_monte_carlo_step() {
    lat_type pos;
    int total_sites = lattice_size * lattice_size;
    
    // Do one sweep: attempt to update every spin once on average
    for (int attempt = 0; attempt < total_sites; attempt++) {
        // Choose a random lattice position (indices from 1 to lattice_size)
        pos.x = 1 + (int)(ran1(&seed) * lattice_size);
        pos.y = 1 + (int)(ran1(&seed) * lattice_size);
        
        // Calculate energy before the proposed move
        double energy_before = (this->*calculate_local_energy)(pos);
        
        // Propose a move (this changes the lattice!)
        (this->*propose_spin_flip)(pos);
        
        // Calculate energy after the proposed move
        double energy_after = (this->*calculate_local_energy)(pos);
        
        // Metropolis acceptance criterion
        double energy_change = energy_after - energy_before;
        if (!metropolis_test(energy_change)) {
            // Reject: undo the move by flipping back
            (this->*propose_spin_flip)(pos);
        }
        // If accepted, we keep the new configuration
    }
}

// Metropolis acceptance test - same for both models!
bool MonteCarloSimulation::metropolis_test(double energy_change) {
    if (energy_change <= 0.0) {
        return true;  // Always accept moves that lower energy
    } else {
        // Accept with probability exp(-ΔE/T)
        double acceptance_probability = std::exp(-energy_change / temperature);
        return (ran1(&seed) < acceptance_probability);
    }
}

// Measurement methods - simple and clear!
double MonteCarloSimulation::get_energy() {
    switch (model_type) {
        case SpinType::ISING:
            return ising_total_energy();
        case SpinType::HEISENBERG:
            return heisenberg_total_energy();
        default:
            return 0.0;
    }
}

double MonteCarloSimulation::get_magnetization() {
    switch (model_type) {
        case SpinType::ISING:
            return ising_total_magnetization();
        case SpinType::HEISENBERG:
            return heisenberg_total_magnetization();
        default:
            return 0.0;
    }
}

double MonteCarloSimulation::get_absolute_magnetization() {
    return std::abs(get_magnetization());  // Just use the function above
}

// Run transient phase to equilibrate the system
void MonteCarloSimulation::run_transient_phase(int transient_steps) {
    std::cout << "Running transient phase (" << transient_steps << " steps)..." << std::endl;
    
    for (int step = 0; step < transient_steps; step++) {
        run_monte_carlo_step();
        
        // Print progress every 1000 steps
        if ((step + 1) % 1000 == 0) {
            std::cout << "Transient: " << (step + 1) << "/" << transient_steps 
                      << " (" << (100.0 * (step + 1) / transient_steps) << "%)" << std::endl;
        }
    }
    
    std::cout << "Transient phase completed." << std::endl;
}

// Run a full simulation with specified number of Monte Carlo steps
void MonteCarloSimulation::run_full_simulation(int mc_steps) {
    std::cout << "Running full simulation (" << mc_steps << " steps)..." << std::endl;
    
    for (int step = 0; step < mc_steps; step++) {
        run_monte_carlo_step();
        
        // Print progress every 10000 steps
        if ((step + 1) % 10000 == 0) {
            double energy = get_energy();
            double magnetization = get_magnetization();
            
            std::cout << "Step " << (step + 1) << "/" << mc_steps 
                      << " - E = " << std::fixed << std::setprecision(2) << energy
                      << ", M = " << magnetization << std::endl;
        }
    }
    
    std::cout << "Simulation completed!" << std::endl;
}