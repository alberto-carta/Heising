/*
 * Comprehensive test for both Heisenberg and Ising models
 * Tests at characteristic temperatures: very low, very high, and near transition
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h" 
#include "../include/random.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Global random seed
long int seed = -12345;

void test_model(const std::string& model_name, SpinType spin_type, 
                double temperature, double coupling_J, int lattice_size = 4) {
    
    std::cout << "\n========================================" << std::endl;
    std::cout << model_name << " Model at T = " << temperature << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create system
    UnitCell unit_cell = create_unit_cell(spin_type);
    CouplingMatrix couplings = create_nn_couplings(1, coupling_J);
    
    MonteCarloSimulation sim(unit_cell, couplings, lattice_size, temperature);
    sim.initialize_lattice();
    
    int total_spins = lattice_size * lattice_size * lattice_size;
    int warmup_steps = 2000;  // Reasonable for testing
    
    std::cout << "Initial state:" << std::endl;
    std::cout << "  Energy/spin: " << std::fixed << std::setprecision(4) 
              << sim.get_energy() / total_spins << std::endl;
    std::cout << "  Magnetization/spin: " << sim.get_magnetization() / total_spins << std::endl;
    std::cout << "  |Magnetization|/spin: " << sim.get_absolute_magnetization() / total_spins << std::endl;
    
    // Warmup
    std::cout << "\nRunning equilibration..." << std::endl;
    for (int sweep = 0; sweep < warmup_steps; sweep++) {
        for (int attempt = 0; attempt < total_spins; attempt++) {
            sim.run_monte_carlo_step();
        }
        
        // Print progress every 500 sweeps
        if ((sweep + 1) % 500 == 0 || sweep < 10) {
            double energy_per_spin = sim.get_energy() / total_spins;
            double mag_per_spin = sim.get_magnetization() / total_spins;
            double abs_mag_per_spin = sim.get_absolute_magnetization() / total_spins;
            
            std::cout << "  Sweep " << std::setw(4) << (sweep + 1) 
                      << ": E/N=" << std::setw(8) << std::setprecision(4) << energy_per_spin
                      << ", M/N=" << std::setw(8) << std::setprecision(4) << mag_per_spin
                      << ", |M|/N=" << std::setw(8) << std::setprecision(4) << abs_mag_per_spin
                      << ", Accept=" << std::setw(6) << std::setprecision(2) << sim.get_acceptance_rate() << "%" << std::endl;
        }
    }
    
    // Final results
    std::cout << "\nFinal equilibrated state:" << std::endl;
    double final_energy = sim.get_energy() / total_spins;
    double final_mag = sim.get_magnetization() / total_spins;
    double final_abs_mag = sim.get_absolute_magnetization() / total_spins;
    double accept_rate = sim.get_acceptance_rate();
    
    std::cout << "  Energy per spin: " << std::setprecision(6) << final_energy << std::endl;
    std::cout << "  Magnetization per spin: " << final_mag << std::endl;
    std::cout << "  |Magnetization| per spin: " << final_abs_mag << std::endl;
    std::cout << "  Acceptance rate: " << std::setprecision(2) << accept_rate << "%" << std::endl;
    
    // Expected behavior analysis
    std::cout << "\nExpected behavior analysis:" << std::endl;
    if (coupling_J < 0) {  // Ferromagnetic
        if (temperature < 0.5) {
            std::cout << "  Low T ferromagnet: Should have high |M| (~1.0) and low energy" << std::endl;
            if (final_abs_mag > 0.8) {
                std::cout << "  ✓ PASS: High magnetization as expected" << std::endl;
            } else {
                std::cout << "  ✗ FAIL: Magnetization too low for low-T ferromagnet" << std::endl;
            }
        } else if (temperature > 5.0) {
            std::cout << "  High T: Should have low |M| (~0.0) due to thermal disorder" << std::endl;
            if (final_abs_mag < 0.2) {
                std::cout << "  ✓ PASS: Low magnetization as expected for high T" << std::endl;
            } else {
                std::cout << "  ✗ FAIL: Magnetization too high for high-T disordered state" << std::endl;
            }
        } else {
            std::cout << "  Intermediate T: Near transition region" << std::endl;
            std::cout << "  Expected Tc for 3D " << model_name << ": " 
                      << (spin_type == SpinType::ISING ? "~4.5" : "~1.4") << std::endl;
        }
    } else {  // Antiferromagnetic
        std::cout << "  Antiferromagnetic: Net magnetization should be small" << std::endl;
    }
    
    // Energy consistency check
    double expected_min_energy = 3.0 * coupling_J;  // 6 neighbors, each pair counted once
    std::cout << "  Ground state energy per spin: " << expected_min_energy << std::endl;
    if (temperature < 0.5 && final_energy < expected_min_energy + 0.1) {
        std::cout << "  ✓ PASS: Energy close to ground state" << std::endl;
    } else if (temperature < 0.5) {
        std::cout << "  ⚠ WARNING: Energy higher than expected ground state" << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Comprehensive Spin Model Validation   " << std::endl;
    std::cout << "========================================" << std::endl;
    
    int lattice_size = 4;  // Small for fast testing
    double J_ferro = -1.0;  // Ferromagnetic
    
    std::cout << "Testing both Ising and Heisenberg models" << std::endl;
    std::cout << "Lattice size: " << lattice_size << "³ (" << lattice_size*lattice_size*lattice_size << " spins)" << std::endl;
    std::cout << "Ferromagnetic coupling: J = " << J_ferro << std::endl;
    
    // Test temperatures
    std::vector<double> test_temperatures = {0.1, 1.0, 1.5, 2.0, 4.0, 8.0};
    
    for (double T : test_temperatures) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TEMPERATURE T = " << T << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Test Ising model
        test_model("Ising", SpinType::ISING, T, J_ferro, lattice_size);
        
        // Test Heisenberg model  
        test_model("Heisenberg", SpinType::HEISENBERG, T, J_ferro, lattice_size);
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "VALIDATION COMPLETE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Check the results above for:" << std::endl;
    std::cout << "1. Low T (0.1, 1.0): High |M| for both models" << std::endl;
    std::cout << "2. High T (8.0): Low |M| for both models" << std::endl;
    std::cout << "3. Transition region: Ising Tc~4.5, Heisenberg Tc~1.4" << std::endl;
    std::cout << "4. Acceptance rates: Should be reasonable (10-90%)" << std::endl;
    std::cout << "5. Energy consistency: Should approach ground state at low T" << std::endl;
    
    return 0;
}