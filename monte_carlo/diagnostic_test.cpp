/*
 * Diagnostic test for Heisenberg magnetization at low temperature
 * This will help us understand what's going wrong with the magnetization calculation
 */

#include "../include/simulation_engine.h"
#include "../include/multi_atom.h" 
#include "../include/random.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Global random seed
long int seed = -12345;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "    Heisenberg Diagnostic Test          " << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Simple test parameters
    int lattice_size = 4;           // Smaller lattice for faster testing
    double T = 0.1;                 // Very low temperature
    double coupling_J = -1.0;       // Ferromagnetic coupling
    int warmup_steps = 1000;        // Short warmup for testing
    int measurement_steps = 1000;   // Short measurement
    
    std::cout << "Test Parameters:" << std::endl;
    std::cout << "  Lattice size: " << lattice_size << "³ (" << lattice_size*lattice_size*lattice_size << " spins)" << std::endl;
    std::cout << "  Temperature: " << T << " (very low - should be ferromagnetic)" << std::endl;
    std::cout << "  Coupling J: " << coupling_J << " (ferromagnetic)" << std::endl;
    std::cout << std::endl;
    
    // Create Heisenberg system
    UnitCell heisenberg_cell = create_unit_cell(SpinType::HEISENBERG);
    CouplingMatrix heisenberg_couplings = create_nn_couplings(1, coupling_J);
    
    MonteCarloSimulation sim(heisenberg_cell, heisenberg_couplings, lattice_size, T);
    sim.initialize_lattice();
    
    // Check initial state
    std::cout << "Initial state:" << std::endl;
    std::cout << "  Energy: " << sim.get_energy() << std::endl;
    std::cout << "  Magnetization (z-component): " << sim.get_magnetization() << std::endl;
    std::cout << "  |Magnetization|: " << sim.get_absolute_magnetization() << std::endl;
    std::cout << std::endl;
    
    // Warmup
    std::cout << "Running warmup..." << std::endl;
    int total_spins = lattice_size * lattice_size * lattice_size;
    for (int sweep = 0; sweep < warmup_steps; sweep++) {
        for (int attempt = 0; attempt < total_spins; attempt++) {
            sim.run_monte_carlo_step();
        }
        if (sweep % 100 == 0) {
            std::cout << "  Sweep " << sweep << ": E=" << std::fixed << std::setprecision(4) 
                      << sim.get_energy() << ", M=" << sim.get_magnetization() 
                      << ", |M|=" << sim.get_absolute_magnetization() << std::endl;
        }
    }
    
    // Final measurement
    std::cout << std::endl << "Final state after warmup:" << std::endl;
    std::cout << "  Energy per spin: " << sim.get_energy() / total_spins << std::endl;
    std::cout << "  Magnetization per spin: " << sim.get_magnetization() / total_spins << std::endl;
    std::cout << "  |Magnetization| per spin: " << sim.get_absolute_magnetization() / total_spins << std::endl;
    std::cout << "  Acceptance rate: " << sim.get_acceptance_rate() << "%" << std::endl;
    std::cout << std::endl;
    
    // Expected values for comparison
    std::cout << "Expected at T=" << T << " with J=" << coupling_J << ":" << std::endl;
    std::cout << "  |Magnetization| per spin should be close to 1.0 (fully ordered)" << std::endl;
    std::cout << "  Energy per spin should be close to " << 3.0 * coupling_J << " = " << 3.0 * coupling_J << " (6 neighbors, each pair counted once)" << std::endl;
    std::cout << std::endl;
    
    // Diagnostic: Let's manually check a few spin orientations
    std::cout << "Diagnostic - checking individual spin orientations:" << std::endl;
    // Note: We can't access individual spins directly from here, but we can infer from the results
    
    if (std::abs(sim.get_absolute_magnetization() / total_spins) < 0.5) {
        std::cout << "WARNING: Magnetization too small! Possible issues:" << std::endl;
        std::cout << "  1. Magnetization calculation only using z-component (not total magnitude)" << std::endl;
        std::cout << "  2. Spins not aligning properly (temperature too high or coupling wrong)" << std::endl;
        std::cout << "  3. Random initial conditions not equilibrating properly" << std::endl;
    } else {
        std::cout << "Magnetization looks reasonable for low temperature ferromagnet." << std::endl;
    }
    
    return 0;
}