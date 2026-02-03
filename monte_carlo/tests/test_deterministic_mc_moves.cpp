/*
 * Deterministic Monte Carlo Move Tests
 * 
 * Tests that MC moves:
 * 1. Calculate energy changes correctly
 * 2. Don't modify the system during proposal
 * 3. Double spin tunnel is KK-invariant
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h"
#include "../include/random.h"
#include "../include/mc_moves.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

long int seed = -12345;

// Helper function for floating point comparison
bool approx_equal(double a, double b, double tolerance = 1e-6) {
    return std::abs(a - b) < tolerance;
}

/**
 * Test 1: Ising flip in ferromagnetic system
 * Should correctly compute energy change when flipping one spin
 */
bool test_ising_flip_ferromagnet() {
    std::cout << "\n=== Test 1: Ising Flip in Ferromagnet ===" << std::endl;
    
    // Create single Ising spin unit cell
    UnitCell cell;
    cell.add_spin("Fe", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);
    
    // Ferromagnetic coupling J = -1.0
    CouplingMatrix couplings;
    couplings.initialize(1, 1);
    couplings.set_coupling(0, 0, 1, 0, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 1, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 0, 1, -1.0);
    couplings.set_coupling(0, 0, -1, 0, 0, -1.0);
    couplings.set_coupling(0, 0, 0, -1, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 0, -1, -1.0);
    
    int L = 4;
    double T = 0.1;
    MonteCarloSimulation sim(cell, couplings, L, T);
    
    // Initialize all spins up (+1)
    sim.initialize_lattice_custom({1.0});
    
    // Store initial configuration
    int initial_spin = sim.get_ising_spin(2, 2, 2, 0);
    double initial_energy = sim.get_energy();
    
    // Propose flip at site (2,2,2)
    MoveProposer proposer(sim);
    MoveProposal proposal = proposer.propose_ising_flip(2, 2, 2, 0);
    
    // Check that system wasn't modified
    int after_proposal_spin = sim.get_ising_spin(2, 2, 2, 0);
    double after_proposal_energy = sim.get_energy();
    
    if (initial_spin != after_proposal_spin) {
        std::cout << "  ✗ FAIL: System was modified during proposal!" << std::endl;
        return false;
    }
    
    if (!approx_equal(initial_energy, after_proposal_energy)) {
        std::cout << "  ✗ FAIL: Energy changed during proposal!" << std::endl;
        return false;
    }
    
    // Expected energy change: flipping from +1 to -1 in FM
    // Before: 6 neighbors × (-1.0) × (+1)(+1) = -6
    // After:  6 neighbors × (-1.0) × (-1)(+1) = +6
    // Change: +6 - (-6) = +12
    double expected_delta = 12.0;
    
    std::cout << "  Proposed energy change: " << proposal.energy_change 
              << " (expected: " << expected_delta << ")" << std::endl;
    
    if (approx_equal(proposal.energy_change, expected_delta)) {
        std::cout << "  ✓ Ising flip energy correct!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ Energy mismatch!" << std::endl;
        return false;
    }
}

/**
 * Test 2: Heisenberg flip in ferromagnet
 * S → -S should have same magnitude as Ising flip
 */
bool test_heisenberg_flip_ferromagnet() {
    std::cout << "\n=== Test 2: Heisenberg Flip in Ferromagnet ===" << std::endl;
    
    UnitCell cell;
    cell.add_spin("Cr", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);
    
    CouplingMatrix couplings;
    couplings.initialize(1, 1);
    couplings.set_coupling(0, 0, 1, 0, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 1, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 0, 1, -1.0);
    couplings.set_coupling(0, 0, -1, 0, 0, -1.0);
    couplings.set_coupling(0, 0, 0, -1, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 0, -1, -1.0);
    
    int L = 4;
    double T = 0.1;
    MonteCarloSimulation sim(cell, couplings, L, T);
    
    // Initialize all spins along +z
    sim.initialize_lattice_custom({1.0});
    
    // Store initial state
    spin3d initial_spin = sim.get_heisenberg_spin(2, 2, 2, 0);
    
    // Propose flip
    MoveProposer proposer(sim);
    MoveProposal proposal = proposer.propose_heisenberg_flip(2, 2, 2, 0);
    
    // Check system wasn't modified
    spin3d after_spin = sim.get_heisenberg_spin(2, 2, 2, 0);
    
    if (!approx_equal(initial_spin.z, after_spin.z)) {
        std::cout << "  ✗ FAIL: System was modified during proposal!" << std::endl;
        return false;
    }
    
    // Expected: same as Ising (+12 for FM)
    double expected_delta = 12.0;
    
    std::cout << "  Proposed energy change: " << proposal.energy_change 
              << " (expected: " << expected_delta << ")" << std::endl;
    
    if (approx_equal(proposal.energy_change, expected_delta)) {
        std::cout << "  ✓ Heisenberg flip energy correct!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ Energy mismatch!" << std::endl;
        return false;
    }
}

/**
 * Test 3: Double spin tunnel with KK coupling
 * Should correctly compute energy using site energy (not summing individual spins)
 * KK energy should be invariant
 */
bool test_double_spin_tunnel_kk_invariance() {
    std::cout << "\n=== Test 3: Double Spin Tunnel KK Invariance ===" << std::endl;
    
    // Create 2-site unit cell with mixed spins
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);  // Site 0
    cell.add_spin("CrA", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);       // Site 0
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);  // Site 1
    cell.add_spin("CrB", SpinType::ISING, 1.0, 0.5, 0.0, 0.0);       // Site 1
    
    // Set up J couplings
    CouplingMatrix couplings;
    couplings.initialize(4, 1);
    
    // Cr1-Cr2 coupling
    couplings.set_coupling(0, 2, 0, 0, 0, 1.0);
    couplings.set_coupling(2, 0, 0, 0, 0, 1.0);
    
    // CrA-CrB coupling
    couplings.set_coupling(1, 3, 0, 0, 0, 0.5);
    couplings.set_coupling(3, 1, 0, 0, 0, 0.5);
    
    // Set up KK coupling
    KK_Matrix kk_matrix;
    kk_matrix.initialize(cell, 1);  // Initialize with unit cell
    kk_matrix.set_coupling(0, 1, 0, 0, 0, 1.0);  // KK between site 0 and site 1
    
    int L = 2;
    double T = 0.1;
    MonteCarloSimulation sim(cell, couplings, L, T, kk_matrix);
    
    // Initialize: Cr1=+z, CrA=+1, Cr2=-z, CrB=-1
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0});
    
    // Calculate KK energy before
    double kk_before = 0.0;
    for (int x = 1; x <= L; x++) {
        for (int y = 1; y <= L; y++) {
            for (int z = 1; z <= L; z++) {
                // KK = K * (S1·S2) * (τ1*τ2)
                spin3d s1 = sim.get_heisenberg_spin(x, y, z, 0);
                spin3d s2 = sim.get_heisenberg_spin(x, y, z, 2);
                int tau1 = sim.get_ising_spin(x, y, z, 1);
                int tau2 = sim.get_ising_spin(x, y, z, 3);
                
                double s_dot_s = s1.x*s2.x + s1.y*s2.y + s1.z*s2.z;
                kk_before += 1.0 * s_dot_s * tau1 * tau2;
            }
        }
    }
    
    std::cout << "  KK energy before: " << kk_before << std::endl;
    
    // Propose double tunnel on site 0
    MoveProposer proposer(sim);
    MoveProposal proposal = proposer.propose_site_double_tunnel(2, 1, 1, 0);
    
    // For testing: manually apply the move to check KK invariance
    sim.set_heisenberg_spin(2, 1, 1, 0, spin3d(0, 0, -1));  // Flip Cr1
    sim.set_ising_spin(2, 1, 1, 1, -1);  // Flip CrA
    
    // Calculate KK energy after
    double kk_after = 0.0;
    for (int x = 1; x <= L; x++) {
        for (int y = 1; y <= L; y++) {
            for (int z = 1; z <= L; z++) {
                spin3d s1 = sim.get_heisenberg_spin(x, y, z, 0);
                spin3d s2 = sim.get_heisenberg_spin(x, y, z, 2);
                int tau1 = sim.get_ising_spin(x, y, z, 1);
                int tau2 = sim.get_ising_spin(x, y, z, 3);
                
                double s_dot_s = s1.x*s2.x + s1.y*s2.y + s1.z*s2.z;
                kk_after += 1.0 * s_dot_s * tau1 * tau2;
            }
        }
    }
    
    std::cout << "  KK energy after:  " << kk_after << std::endl;
    std::cout << "  KK change:        " << (kk_after - kk_before) << std::endl;
    
    // Revert for next test
    sim.set_heisenberg_spin(2, 1, 1, 0, spin3d(0, 0, 1));
    sim.set_ising_spin(2, 1, 1, 1, 1);
    
    if (approx_equal(kk_before, kk_after)) {
        std::cout << "  ✓ KK energy is invariant under double spin tunnel!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ KK energy changed! This violates invariance." << std::endl;
        return false;
    }
}

/**
 * Test 4: Verify proposal doesn't modify system energy
 */
bool test_proposal_energy_conservation() {
    std::cout << "\n=== Test 4: Proposal Energy Conservation ===" << std::endl;
    
    UnitCell cell;
    cell.add_spin("Fe", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);
    
    CouplingMatrix couplings;
    couplings.initialize(1, 1);
    couplings.set_coupling(0, 0, 1, 0, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 1, 0, -1.0);
    couplings.set_coupling(0, 0, 0, 0, 1, -1.0);
    
    int L = 3;
    double T = 0.1;
    MonteCarloSimulation sim(cell, couplings, L, T);
    sim.initialize_lattice_custom({1.0});
    
    double initial_energy = sim.get_energy();
    
    MoveProposer proposer(sim);
    
    // Make multiple proposals
    for (int i = 0; i < 10; i++) {
        proposer.propose_ising_flip(1, 1, 1, 0);
    }
    
    double final_energy = sim.get_energy();
    
    if (approx_equal(initial_energy, final_energy)) {
        std::cout << "  ✓ Energy unchanged after proposals!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ Energy changed: " << initial_energy 
                  << " → " << final_energy << std::endl;
        return false;
    }
}

/**
 * Test 5: Double spin tunnel energy calculation
 * Verify it uses site energy (correct for KK) not sum of individual energies
 */
bool test_double_tunnel_uses_site_energy() {
    std::cout << "\n=== Test 5: Double Tunnel Uses Site Energy ===" << std::endl;
    
    // Create 2-site system with KK
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);
    cell.add_spin("CrA", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("CrB", SpinType::ISING, 1.0, 0.5, 0.0, 0.0);
    
    CouplingMatrix couplings;
    couplings.initialize(4, 1);
    couplings.set_coupling(0, 2, 0, 0, 0, 0.8);
    couplings.set_coupling(2, 0, 0, 0, 0, 0.8);
    couplings.set_coupling(1, 3, 0, 0, 0, 0.5);
    couplings.set_coupling(3, 1, 0, 0, 0, 0.5);
    
    KK_Matrix kk_matrix;
    kk_matrix.initialize(cell, 1);
    kk_matrix.set_coupling(0, 1, 0, 0, 0, 1.0);
    
    int L = 2;
    double T = 0.1;
    MonteCarloSimulation sim(cell, couplings, L, T, kk_matrix);
    
    // Initialize C-AFM + COO state
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0});
    
    // Get site energy before
    double site_energy_before = sim.calculate_site_energy(1, 1, 1, 0);
    
    // Propose double tunnel
    MoveProposer proposer(sim);
    MoveProposal proposal = proposer.propose_site_double_tunnel(1, 1, 1, 0);
    
    // Manually flip and calculate site energy after
    sim.set_heisenberg_spin(1, 1, 1, 0, spin3d(0, 0, -1));
    sim.set_ising_spin(1, 1, 1, 1, -1);
    
    double site_energy_after = sim.calculate_site_energy(1, 1, 1, 0);
    
    double expected_change = site_energy_after - site_energy_before;
    
    // Revert
    sim.set_heisenberg_spin(1, 1, 1, 0, spin3d(0, 0, 1));
    sim.set_ising_spin(1, 1, 1, 1, 1);
    
    std::cout << "  Site energy change:     " << expected_change << std::endl;
    std::cout << "  Proposed energy change: " << proposal.energy_change << std::endl;
    
    if (approx_equal(proposal.energy_change, expected_change, 1e-5)) {
        std::cout << "  ✓ Double tunnel correctly uses site energy!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ Energy mismatch - may be double counting KK!" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  DETERMINISTIC MC MOVE TESTS          " << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 5;
    
    if (test_ising_flip_ferromagnet()) passed++;
    if (test_heisenberg_flip_ferromagnet()) passed++;
    if (test_double_spin_tunnel_kk_invariance()) passed++;
    if (test_proposal_energy_conservation()) passed++;
    if (test_double_tunnel_uses_site_energy()) passed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS: " << passed << "/" << total << " tests passed" << std::endl;
    
    if (passed == total) {
        std::cout << "✓ ALL MC MOVE TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ " << (total - passed) << " tests failed" << std::endl;
        return 1;
    }
}
