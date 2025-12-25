/*
 * Comprehensive Unit Tests for Monte Carlo Multi-Spin Implementation
 * 
 * Tests core functionalities without trivial imports:
 * 1. Multi-spin lattice creation and spin access
 * 2. Coupling matrix setup and energy calculations
 * 3. Metropolis algorithm correctness 
 * 4. Energy conservation and physical properties
 * 5. Mixed spin type handling (Ising + Heisenberg)
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h"
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

// Test 1: Multi-spin lattice creation and spin access
bool test_lattice_creation() {
    std::cout << "\n=== Test 1: Multi-Spin Lattice Creation ===" << std::endl;
    
    // Create 2-spin unit cell (Heisenberg + Ising at same position for KK coupling)
    UnitCell cell;
    cell.add_spin("H1", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);  // Site 0
    cell.add_spin("I1", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);        // Site 0 (same position!)
    
    // Simple coupling matrix
    CouplingMatrix couplings;
    couplings.initialize(2, 1);  // 2 spins, max_offset = 1
    couplings.set_intra_coupling(0, 1, -1.0);  // Intra-cell FM coupling
    couplings.set_nn_couplings(0, 0, -0.5);   // H-H nearest neighbors
    couplings.set_nn_couplings(1, 1, -0.5);   // I-I nearest neighbors

    // Add Kugel-Khomskii coupling matrix
    KK_Matrix kk_couplings;
    kk_couplings.initialize(cell, 1);  // unit cell, max_offset = 1
    // KK coupling between site 0 in this cell and site 0 in neighbor cell
    kk_couplings.set_coupling(0, 0, 1, 0, 0, 0.1);  // KK along +x direction
    
    // Create simulation with KK coupling
    MonteCarloSimulation sim(cell, couplings, 3, 1.0, kk_couplings);
    sim.initialize_lattice_custom(std::vector<double>{1.0, 1.0});  
    
    // Verify KK coupling is present
    if (!sim.has_kugel_khomskii()) {
        std::cout << "✗ KK coupling not detected" << std::endl;
        return false;
    }
    
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
        std::cout << "✓ Spin access works correctly with KK coupling" << std::endl;
        return true;
    } else {
        std::cout << "✗ Spin access failed" << std::endl;
        return false;
    }
}

// Test 2: Energy calculation correctness
bool test_energy_calculation() {
    std::cout << "\n=== Test 2: Energy Calculation Correctness ===" << std::endl;
    
    // Create simple 1-spin Heisenberg system for analytical verification
    UnitCell cell = create_unit_cell(SpinType::HEISENBERG);
    CouplingMatrix couplings = create_nn_couplings(1, -1.0);  // FM coupling
    
    MonteCarloSimulation sim(cell, couplings, 2, 1.0);  // 2x2x2 = 8 spins
    sim.initialize_lattice_custom(std::vector<double>(cell.num_spins(), 1.0));
    
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
    
    // Each spin has 6 neighbors in 2x2x2 lattice with periodic boundary
    // Total pairs = 8 * 6 / 2 = 24 pairs
    // Energy = 24 * (-1.0) * (1 * 1) = -24.0
    double expected_energy = -24.0;
    
    if (approx_equal(total_energy, expected_energy, 1e-6)) {
        std::cout << "✓ Energy calculation correct: " << total_energy << std::endl;
        return true;
    } else {
        std::cout << "✗ Energy calculation failed: got " << total_energy 
                  << ", expected " << expected_energy << std::endl;
        return false;
    }
}

// Test 3: Metropolis algorithm correctness at T=0
bool test_metropolis_algorithm() {
    std::cout << "\n=== Test 3: Metropolis Algorithm Correctness (T=0) ===" << std::endl;
    
    UnitCell cell = create_unit_cell(SpinType::ISING);  // Use Ising for determinism
    CouplingMatrix couplings = create_nn_couplings(1, -1.0);  // FM coupling
    
    // Test at T=0: only energy-lowering moves should be accepted
    MonteCarloSimulation sim(cell, couplings, 3, 0.0);
    
    // Set a configuration with ONE spin misaligned (pointing down)
    // All others point up - this creates a high-energy defect
    for (int x = 1; x <= 3; x++) {
        for (int y = 1; y <= 3; y++) {
            for (int z = 1; z <= 3; z++) {
                sim.set_ising_spin(x, y, z, 0, 1);  // All up
            }
        }
    }
    sim.set_ising_spin(2, 2, 2, 0, -1);  // Center spin down (defect)
    
    double initial_energy = sim.get_energy();
    
    // At T=0, attempting to flip the center spin should be accepted
    // because it lowers energy (aligns with neighbors)
    // Store the energy change
    double local_energy_before = sim.calculate_local_energy(2, 2, 2, 0);
    
    // Calculate what energy would be if center spin flipped to align
    // Current: center=-1, neighbors=+1, E_local = -J * (-1) * 6 * (+1) = +6J = -6.0
    // After flip: center=+1, neighbors=+1, E_local = -J * (+1) * 6 * (+1) = -6J = +6.0
    // Delta E = -12J = +12.0 (energy decreases by 12.0)
    
    // Now test: at T=0, energy-increasing move should be rejected
    // Create ground state (all aligned)
    for (int x = 1; x <= 3; x++) {
        for (int y = 1; y <= 3; y++) {
            for (int z = 1; z <= 3; z++) {
                sim.set_ising_spin(x, y, z, 0, 1);  // All up
            }
        }
    }
    
    double ground_energy = sim.get_energy();
    
    // Manually test Metropolis: try to flip a spin in ground state
    // This should ALWAYS be rejected at T=0
    sim.reset_statistics();
    int accepted_count = 0;
    for (int i = 0; i < 100; i++) {
        double energy_before = sim.get_energy();
        sim.run_monte_carlo_step();
        double energy_after = sim.get_energy();
        
        // At T=0 in ground state, no move should be accepted
        if (energy_after != energy_before) {
            accepted_count++;
        }
    }
    
    double acceptance_rate = sim.get_acceptance_rate();
    
    // At T=0 in ground state, acceptance should be exactly 0
    if (accepted_count == 0 && approx_equal(acceptance_rate, 0.0, 1e-10)) {
        std::cout << "✓ Metropolis at T=0: no energy-raising moves accepted" << std::endl;
        return true;
    } else {
        std::cout << "✗ Metropolis at T=0 failed: " << accepted_count 
                  << " moves accepted, rate=" << acceptance_rate << std::endl;
        return false;
    }
}

// Test 4: Mixed spin types (Ising + Heisenberg) with deterministic configuration
bool test_mixed_spin_types() {
    std::cout << "\n=== Test 4: Mixed Spin Types (Deterministic) ===" << std::endl;
    
    // Create unit cell with both Ising and Heisenberg spins at same position
    UnitCell mixed_cell;
    mixed_cell.add_spin("Heisenberg", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);
    mixed_cell.add_spin("Ising", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);
    
    CouplingMatrix mixed_couplings;
    mixed_couplings.initialize(2, 1);

    // set symmetrically on-site anti-alignment couplings
    mixed_couplings.set_intra_coupling(0, 1, 10.0);  // H-I coupling
    mixed_couplings.set_intra_coupling(1, 0, 10.0);  // H-I coupling
    // set ising and heisenberg couplings
    mixed_couplings.set_nn_couplings(0, 0, -1.0);    // H-H coupling
    mixed_couplings.set_nn_couplings(1, 1, -1.0);    // I-I coupling


    
    MonteCarloSimulation mixed_sim(mixed_cell, mixed_couplings, 2, 1.0);
    
    // Set a known configuration: all Heisenberg spins up (0,0,1), all Ising up (+1)
    spin3d heisenberg_up(0, 0, 1);
    for (int x = 1; x <= 2; x++) {
        for (int y = 1; y <= 2; y++) {
            for (int z = 1; z <= 2; z++) {
                mixed_sim.set_heisenberg_spin(x, y, z, 0, heisenberg_up);
                mixed_sim.set_ising_spin(x, y, z, 1, 1);
            }
        }
    }
    
    // Calculate energy analytically for this configuration
    // Each site has 2 spins. Total sites = 2x2x2 = 8
    // E contribution from HH = -24
    // E contribution from II = -24
    // E contribution from local HI = 10 * 8
    // 
    // Total expected: -48 + 80 = +32
    double expected_energy = +32.0;
    double calculated_energy = mixed_sim.get_energy();
    
    // Test local energy calculation for a Heisenberg spin
    double local_h = mixed_sim.calculate_local_energy(1, 1, 1, 0);
    // Site (1,1,1) spin 0 has:
    //   - 1 intra-cell coupling: +10
    //   - 6 NN H-H couplings: 6 * 1 = -6
    // Total: +4.0
    double expected_local_h = 4.0;
    
    // Test local energy for an Ising spin
    double local_i = mixed_sim.calculate_local_energy(1, 1, 1, 1);
    // Site (1,1,1) spin 1 has:
    //   - 1 intra-cell coupling: +10
    //   - 6 NN I-I couplings: -6
    // Total: +4.0
    double expected_local_i = 4.0;
    
    bool energy_correct = approx_equal(calculated_energy, expected_energy, 1e-6);
    bool local_h_correct = approx_equal(local_h, expected_local_h, 1e-6);
    bool local_i_correct = approx_equal(local_i, expected_local_i, 1e-6);
    
    if (energy_correct && local_h_correct && local_i_correct) {
        std::cout << "✓ Mixed spin types: E_total=" << calculated_energy 
                  << ", E_local_H=" << local_h << ", E_local_I=" << local_i << std::endl;
        return true;
    } else {
        std::cout << "✗ Mixed spin types failed:" << std::endl;
        std::cout << "  Total energy: " << calculated_energy << " (expected " << expected_energy << ")" << std::endl;
        std::cout << "  Local H: " << local_h << " (expected " << expected_local_h << ")" << std::endl;
        std::cout << "  Local I: " << local_i << " (expected " << expected_local_i << ")" << std::endl;
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

// Test 6: Kugel-Khomskii energy calculation
bool test_kk_energy_calculation() {
    std::cout << "\n=== Test 6: Kugel-Khomskii Energy Calculation ===" << std::endl;
    
    // Create 2-spin unit cell (Heisenberg + Ising at same site)
    UnitCell cell;
    cell.add_spin("H", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);  // Site 0
    cell.add_spin("I", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);        // Site 0
    
    // Standard J couplings
    CouplingMatrix couplings;
    couplings.initialize(2, 1);
    couplings.set_nn_couplings(0, 0, -1.0);  // H-H FM (all NN)
    couplings.set_nn_couplings(1, 1, -1.0);  // I-I FM (all NN)
    
    // KK coupling: K * (S_i · S_j) * (τ_i * τ_j)
    KK_Matrix kk_couplings;
    kk_couplings.initialize(cell, 1);
    // Set KK couplings in all NN directions
    kk_couplings.set_coupling(0, 0,  1,  0,  0, 0.5);  // +x
    kk_couplings.set_coupling(0, 0, -1,  0,  0, 0.5);  // -x
    kk_couplings.set_coupling(0, 0,  0,  1,  0, 0.5);  // +y
    kk_couplings.set_coupling(0, 0,  0, -1,  0, 0.5);  // -y
    kk_couplings.set_coupling(0, 0,  0,  0,  1, 0.5);  // +z
    kk_couplings.set_coupling(0, 0,  0,  0, -1, 0.5);  // -z
    
    // Create 2x2x2 lattice with KK
    MonteCarloSimulation sim(cell, couplings, 2, 1.0, kk_couplings);
    
    // Test case 1: All spins aligned (FM state)
    // H = (0,0,1), I = +1
    sim.initialize_lattice_custom({1.0, 1.0});
    double E_aligned = sim.get_energy();
    
    // For 2x2x2 lattice with 2 spins per cell:
    // - 8 unit cells total, 16 spins (8 H + 8 I)
    // - Each H spin has 3 NN: E_J_H = -1.0 * 3 * 8 / 2 = -12
    // - Each I spin has 3 NN: E_J_I = -0.5 * 3 * 8 / 2 = -6
    // - Each site has 3 NN sites: E_KK = -0.2 * (1*1) * (+1*+1) * 3 * 8 / 2 = -2.4
    // Total: -12 - 6 - 2.4 = -20.4
    
    double expected_E_aligned = -20.4;
    
    if (!approx_equal(E_aligned, expected_E_aligned, 0.01)) {
        std::cout << "✗ Aligned state energy incorrect: expected " << expected_E_aligned 
                  << ", got " << E_aligned << std::endl;
        return false;
    }
    
    // Test case 2: Flip one Ising spin (breaks KK alignment)
    // This should increase energy because KK coupling is FM
    sim.set_ising_spin(1, 1, 1, 1, -1);  // Flip I spin at (1,1,1)
    double E_flipped = sim.get_energy();
    
    // Flipping one I spin:
    // - Changes I-I energy: was -0.5 * 3, now mixed (some parallel, some antiparallel)
    //   Δ = 2 * 0.5 * 3 = 3.0 (approximately, depends on neighbors)
    // - Changes KK energy: was -0.2 * (1) * (+1) for 3 bonds, now -0.2 * (1) * (-1)
    //   Δ_KK = 2 * 0.2 * (S·S) * 3 = 1.2 per bond direction
    // Energy should increase
    
    if (E_flipped <= E_aligned) {
        std::cout << "✗ Flipping spin with FM KK should increase energy" << std::endl;
        std::cout << "  E_aligned=" << E_aligned << ", E_flipped=" << E_flipped << std::endl;
        return false;
    }
    
    // Test case 3: Verify KK contribution is computed
    // Create system without KK for comparison
    MonteCarloSimulation sim_no_kk(cell, couplings, 2, 1.0);
    sim_no_kk.initialize_lattice_custom({1.0, 1.0});
    double E_no_kk = sim_no_kk.get_energy();
    
    // E_aligned should be lower than E_no_kk by the KK contribution
    double kk_contribution = E_aligned - E_no_kk;
    double expected_kk = -2.4;
    
    if (!approx_equal(kk_contribution, expected_kk, 0.01)) {
        std::cout << "✗ KK contribution incorrect: expected " << expected_kk 
                  << ", got " << kk_contribution << std::endl;
        std::cout << "  E_with_kk=" << E_aligned << ", E_no_kk=" << E_no_kk << std::endl;
        return false;
    }
    
    std::cout << "✓ KK energy: E_aligned=" << E_aligned << ", E_no_kk=" << E_no_kk 
              << ", KK_contrib=" << kk_contribution << std::endl;
    
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   COMPREHENSIVE MONTE CARLO TESTS     " << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 6;
    
    if (test_lattice_creation()) passed++;
    if (test_energy_calculation()) passed++;
    if (test_metropolis_algorithm()) passed++;
    if (test_mixed_spin_types()) passed++;
    if (test_coupling_matrix_scaling()) passed++;
    if (test_kk_energy_calculation()) passed++;
    
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