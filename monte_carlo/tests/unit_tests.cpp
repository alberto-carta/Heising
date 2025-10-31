/*
 * Comprehensive Unit Tests for Monte Carlo Multi-Atom Implementation
 * 
 * Tests core functionalities without trivial imports:
 * 1. Multi-atom lattice creation and spin access
 * 2. Coupling matrix setup and energy calculations
 * 3. Metropolis algorithm correctness 
 * 4. Energy conservation and physical properties
 * 5. Mixed spin type handling (Ising + Heisenberg)
 */

#include "../include/simulation_engine.h"
#include "../include/multi_atom.h"
#include "../include/random.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

long int seed = -12345;

// Helper function for floating point comparison
bool approx_equal(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) < tolerance;
}

// Test 1: Multi-atom lattice creation and spin access
bool test_lattice_creation() {
    std::cout << "\n=== Test 1: Multi-Atom Lattice Creation ===" << std::endl;
    
    // Create 2-atom unit cell (Heisenberg + Ising)
    UnitCell cell;
    cell.add_atom("H1", SpinType::HEISENBERG, 1.0);
    cell.add_atom("I1", SpinType::ISING, 1.0);
    
    // Simple coupling matrix
    CouplingMatrix couplings;
    couplings.initialize(2, 1);  // 2 atoms, max_offset = 1
    couplings.set_intra_coupling(0, 1, -1.0);  // Intra-cell FM coupling
    couplings.set_nn_couplings(0, 0, -0.5);   // H-H nearest neighbors
    couplings.set_nn_couplings(1, 1, -0.5);   // I-I nearest neighbors
    
    // Create simulation
    MonteCarloSimulation sim(cell, couplings, 3, 1.0);
    sim.initialize_lattice();
    
    // Test spin access - set and get known values
    spin3d test_heisenberg(0.5, 0.5, std::sqrt(0.5));
    test_heisenberg.normalize();
    sim.set_heisenberg_spin(1, 1, 1, 0, test_heisenberg);
    sim.set_ising_spin(1, 1, 1, 1, -1);
    
    spin3d retrieved_h = sim.get_heisenberg_spin(1, 1, 1, 0);
    int retrieved_i = sim.get_ising_spin(1, 1, 1, 1);
    
    bool h_correct = approx_equal(retrieved_h.x, test_heisenberg.x, 1e-10) &&
                     approx_equal(retrieved_h.y, test_heisenberg.y, 1e-10) &&
                     approx_equal(retrieved_h.z, test_heisenberg.z, 1e-10);
    
    bool i_correct = (retrieved_i == -1);
    
    if (h_correct && i_correct) {
        std::cout << "✓ Spin access works correctly" << std::endl;
        return true;
    } else {
        std::cout << "✗ Spin access failed" << std::endl;
        return false;
    }
}

// Test 2: Energy calculation correctness
bool test_energy_calculation() {
    std::cout << "\n=== Test 2: Energy Calculation Correctness ===" << std::endl;
    
    // Create simple 1-atom Heisenberg system for analytical verification
    UnitCell cell = create_unit_cell(SpinType::HEISENBERG);
    CouplingMatrix couplings = create_nn_couplings(1, -1.0);  // FM coupling
    
    MonteCarloSimulation sim(cell, couplings, 2, 1.0);  // 2x2x2 = 8 spins
    sim.initialize_lattice();
    
    // Set all spins to point up (0, 0, 1) for known energy
    spin3d up_spin(0, 0, 1);
    for (int x = 1; x <= 2; x++) {
        for (int y = 1; y <= 2; y++) {
            for (int z = 1; z <= 2; z++) {
                sim.set_heisenberg_spin(x, y, z, 0, up_spin);
            }
        }
    }
    
    // Calculate total energy - should be negative (FM coupling, aligned spins)
    double total_energy = sim.get_energy();
    
    // Each spin has 3 neighbors in 2x2x2 lattice with periodic boundary
    // Total pairs = 8 * 3 / 2 = 12 pairs
    // Energy = 12 * (-1.0) * (1 * 1) = -12.0
    double expected_energy = -12.0;
    
    if (approx_equal(total_energy, expected_energy, 1e-6)) {
        std::cout << "✓ Energy calculation correct: " << total_energy << std::endl;
        return true;
    } else {
        std::cout << "✗ Energy calculation failed: got " << total_energy 
                  << ", expected " << expected_energy << std::endl;
        return false;
    }
}

// Test 3: Metropolis algorithm correctness
bool test_metropolis_algorithm() {
    std::cout << "\n=== Test 3: Metropolis Algorithm Correctness ===" << std::endl;
    
    UnitCell cell = create_unit_cell(SpinType::HEISENBERG);
    CouplingMatrix couplings = create_nn_couplings(1, -1.0);  // FM coupling
    
    MonteCarloSimulation sim(cell, couplings, 3, 0.1);  // Very low temperature
    sim.initialize_lattice();
    
    // Set initial configuration - all spins aligned (minimum energy state)
    spin3d aligned_spin(0, 0, 1);
    for (int x = 1; x <= 3; x++) {
        for (int y = 1; y <= 3; y++) {
            for (int z = 1; z <= 3; z++) {
                sim.set_heisenberg_spin(x, y, z, 0, aligned_spin);
            }
        }
    }
    
    double initial_energy = sim.get_energy();
    
    // Calculate local energy for center spin
    double local_energy_before = sim.calculate_local_energy(2, 2, 2, 0);
    
    // Test energy-lowering move (should always be accepted at any temperature)
    // Since we're at minimum energy, any move will increase energy
    // So let's test the opposite: start from random and see if we reach lower energy
    
    sim.initialize_lattice();  // Random initialization
    double random_energy = sim.get_energy();
    
    // Run some Monte Carlo steps at low temperature
    sim.reset_statistics();
    for (int i = 0; i < 1000; i++) {
        sim.run_monte_carlo_step();
    }
    
    double final_energy = sim.get_energy();
    double acceptance_rate = sim.get_acceptance_rate();
    
    // At low temperature, energy should decrease or stay low, acceptance should be reasonable
    bool energy_improved = (final_energy <= random_energy);
    bool acceptance_reasonable = (acceptance_rate > 0.01 && acceptance_rate < 0.99);
    
    if (energy_improved && acceptance_reasonable) {
        std::cout << "✓ Metropolis algorithm working: E_initial=" << random_energy 
                  << " → E_final=" << final_energy << ", accept=" << acceptance_rate << std::endl;
        return true;
    } else {
        std::cout << "✗ Metropolis algorithm issue: E_initial=" << random_energy 
                  << " → E_final=" << final_energy << ", accept=" << acceptance_rate << std::endl;
        return false;
    }
}

// Test 4: Mixed spin types (Ising + Heisenberg)
bool test_mixed_spin_types() {
    std::cout << "\n=== Test 4: Mixed Spin Types ===" << std::endl;
    
    // Create unit cell with both Ising and Heisenberg atoms
    UnitCell mixed_cell;
    mixed_cell.add_atom("Heisenberg", SpinType::HEISENBERG, 1.0);
    mixed_cell.add_atom("Ising", SpinType::ISING, 1.0);
    
    CouplingMatrix mixed_couplings;
    mixed_couplings.initialize(2, 1);
    mixed_couplings.set_intra_coupling(0, 1, -1.0);  // H-I coupling
    mixed_couplings.set_nn_couplings(0, 0, -0.5);    // H-H coupling
    mixed_couplings.set_nn_couplings(1, 1, -0.5);    // I-I coupling
    
    MonteCarloSimulation mixed_sim(mixed_cell, mixed_couplings, 3, 1.0);
    mixed_sim.initialize_lattice();
    
    // Test that both spin types are properly handled
    double initial_energy = mixed_sim.get_energy();
    
    // Run some steps
    for (int i = 0; i < 100; i++) {
        mixed_sim.run_monte_carlo_step();
    }
    
    double final_energy = mixed_sim.get_energy();
    double acceptance_rate = mixed_sim.get_acceptance_rate();
    
    // System should be functional (finite energies, reasonable acceptance)
    bool energies_finite = (std::isfinite(initial_energy) && std::isfinite(final_energy));
    bool acceptance_ok = (acceptance_rate > 0.001 && acceptance_rate < 1.0);
    
    if (energies_finite && acceptance_ok) {
        std::cout << "✓ Mixed spin types work correctly" << std::endl;
        return true;
    } else {
        std::cout << "✗ Mixed spin types failed: energies=" << energies_finite 
                  << ", acceptance=" << acceptance_rate << std::endl;
        return false;
    }
}

// Test 5: Dynamic coupling matrix scaling
bool test_coupling_matrix_scaling() {
    std::cout << "\n=== Test 5: Dynamic Coupling Matrix Scaling ===" << std::endl;
    
    // Test different max_offset values
    UnitCell cell = create_unit_cell(SpinType::ISING);
    
    // Test max_offset = 1 (nearest neighbors only)
    CouplingMatrix couplings_nn;
    couplings_nn.initialize(1, 1);
    couplings_nn.set_nn_couplings(0, 0, -1.0);
    
    // Should have exactly 6 non-zero couplings (6 NN directions)
    int nn_couplings = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (couplings_nn.get_coupling(0, 0, dx, dy, dz) != 0.0) {
                    nn_couplings++;
                }
            }
        }
    }
    
    // Test max_offset = 2 (includes next-nearest neighbors)
    CouplingMatrix couplings_nnn;
    couplings_nnn.initialize(1, 2);
    couplings_nnn.set_nn_couplings(0, 0, -1.0);
    couplings_nnn.set_coupling(0, 0, 2, 0, 0, -0.5);  // Should work
    
    double long_range_coupling = couplings_nnn.get_coupling(0, 0, 2, 0, 0);
    double out_of_range = couplings_nnn.get_coupling(0, 0, 3, 0, 0);  // Should be 0
    
    bool nn_correct = (nn_couplings == 6);
    bool nnn_works = (long_range_coupling == -0.5);
    bool range_limited = (out_of_range == 0.0);
    
    if (nn_correct && nnn_works && range_limited) {
        std::cout << "✓ Dynamic coupling matrix scaling works" << std::endl;
        return true;
    } else {
        std::cout << "✗ Coupling matrix scaling failed: nn=" << nn_couplings 
                  << ", nnn=" << long_range_coupling << ", out=" << out_of_range << std::endl;
        return false;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   COMPREHENSIVE MONTE CARLO TESTS     " << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 5;
    
    if (test_lattice_creation()) passed++;
    if (test_energy_calculation()) passed++;
    if (test_metropolis_algorithm()) passed++;
    if (test_mixed_spin_types()) passed++;
    if (test_coupling_matrix_scaling()) passed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS: " << passed << "/" << total << " tests passed" << std::endl;
    
    if (passed == total) {
        std::cout << "✓ ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ " << (total - passed) << " tests failed" << std::endl;
        return 1;
    }
}