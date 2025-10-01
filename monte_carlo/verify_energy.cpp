/*
 * Energy verification test for Heisenberg model
 */

#include "simulation_engine.h"
#include "spin_types.h"
#include "random.h"
#include <iostream>
#include <iomanip>

long int seed = -12345;

int main() {
    std::cout << "=== HEISENBERG ENERGY VERIFICATION TEST ===" << std::endl;
    
    // Create a small 2x2x2 system for manual verification
    int size = 2;
    double T = 1.0;  // Temperature doesn't matter for this test
    double J = 1.0; // '+' = AFM coupling, '-' = FM coupling
    
    MonteCarloSimulation sim(SpinType::HEISENBERG, size, T, J);
    
    std::cout << "Testing with 2x2x2 lattice for manual verification" << std::endl;
    std::cout << "J = " << J << " (antiferromagnetic)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test 1: All spins aligned (should give maximum AFM energy)
    std::cout << "Test 1: All spins pointing up (0,0,1)" << std::endl;
    
    // We need to manually set the lattice to a known state
    // This requires accessing the private members, so let's test with initialization
    sim.initialize_lattice();
    
    double initial_E = sim.get_energy();
    double initial_M = sim.get_magnetization();
    
    std::cout << "Random initialization:" << std::endl;
    std::cout << "Energy = " << initial_E << std::endl;
    std::cout << "Magnetization = " << initial_M << std::endl;
    
    // Run at very low temperature to see ordering
    std::cout << "\nTesting at T = 0.1 (very low temperature):" << std::endl;
    MonteCarloSimulation low_T_sim(SpinType::HEISENBERG, size, 0.1, J);
    low_T_sim.initialize_lattice();
    
    std::cout << "Before equilibration:" << std::endl;
    std::cout << "Energy = " << low_T_sim.get_energy() << std::endl;
    std::cout << "Magnetization = " << low_T_sim.get_magnetization() << std::endl;
    
    // Long equilibration
    low_T_sim.run_warmup_phase(50000);
    
    std::cout << "After 50,000 equilibration steps:" << std::endl;
    std::cout << "Energy = " << low_T_sim.get_energy() << std::endl;
    std::cout << "Magnetization = " << low_T_sim.get_magnetization() << std::endl;
    std::cout << "Acceptance rate = " << low_T_sim.get_acceptance_rate() << "%" << std::endl;
    
    // Expected for 2x2x2 = 8 spins:
    // If fully ordered: |M| should be ~8
    // If random: |M| should be ~sqrt(8) ≈ 2.8
    
    double expected_random = std::sqrt(8.0);
    double expected_ordered = 8.0;
    double actual = low_T_sim.get_magnetization();
    
    std::cout << "\nDiagnostic:" << std::endl;
    std::cout << "Expected if random: |M| ≈ " << expected_random << std::endl;
    std::cout << "Expected if ordered: |M| ≈ " << expected_ordered << std::endl;
    std::cout << "Actual: |M| = " << actual << std::endl;
    
    if (std::abs(actual) < expected_random * 1.5) {
        std::cout << ">>> PROBLEM: System appears to be random/disordered!" << std::endl;
    } else if (std::abs(actual) > expected_ordered * 0.7) {
        std::cout << ">>> GOOD: System appears to be well ordered!" << std::endl;
    } else {
        std::cout << ">>> PARTIAL: System is partially ordered." << std::endl;
    }
    
    return 0;
}