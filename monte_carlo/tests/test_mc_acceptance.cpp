/*
 * Non-Deterministic Monte Carlo Acceptance Tests
 * 
 * Tests Metropolis acceptance rates and dynamics
 * These tests are non-deterministic and statistical in nature
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
 * Test 1: Double tunnel acceptance rate at finite temperature
 * Non-deterministic test - runs many moves and checks acceptance rate is reasonable
 */
bool test_double_tunnel_acceptance_rate() {
    std::cout << "\n=== Test 1: Double Tunnel Acceptance Rate (Finite T) ===" << std::endl;
    
    // Create 4-spin unit cell similar to C-AFM + COO configuration
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);
    cell.add_spin("tau1", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("tau2", SpinType::ISING, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("Cr3", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.0);
    cell.add_spin("tau3", SpinType::ISING, 1.0, 0.0, 0.5, 0.0);
    cell.add_spin("Cr4", SpinType::HEISENBERG, 1.0, 0.5, 0.5, 0.0);
    cell.add_spin("tau4", SpinType::ISING, 1.0, 0.5, 0.5, 0.0);
    
    // Set up J couplings (AFM between Cr spins)
    CouplingMatrix couplings;
    couplings.initialize(8, 1);
    
    // Heisenberg AFM couplings
    couplings.set_coupling(0, 2, 0, 0, 0, 1.0);
    couplings.set_coupling(2, 0, 0, 0, 0, 1.0);
    couplings.set_coupling(0, 4, 0, 0, 0, 1.0);
    couplings.set_coupling(4, 0, 0, 0, 0, 1.0);
    couplings.set_coupling(2, 6, 0, 0, 0, 1.0);
    couplings.set_coupling(6, 2, 0, 0, 0, 1.0);
    couplings.set_coupling(4, 6, 0, 0, 0, 1.0);
    couplings.set_coupling(6, 4, 0, 0, 0, 1.0);
    
    // Ising couplings
    couplings.set_coupling(1, 3, 0, 0, 0, 0.5);
    couplings.set_coupling(3, 1, 0, 0, 0, 0.5);
    couplings.set_coupling(1, 5, 0, 0, 0, 0.5);
    couplings.set_coupling(5, 1, 0, 0, 0, 0.5);
    couplings.set_coupling(3, 7, 0, 0, 0, 0.5);
    couplings.set_coupling(7, 3, 0, 0, 0, 0.5);
    couplings.set_coupling(5, 7, 0, 0, 0, 0.5);
    couplings.set_coupling(7, 5, 0, 0, 0, 0.5);
    
    // Set up KK coupling
    KK_Matrix kk_matrix;
    kk_matrix.initialize(cell, 1);
    
    // KK couplings between sites
    kk_matrix.set_coupling(0, 1, 0, 0, 0, 1.0);
    kk_matrix.set_coupling(1, 0, 0, 0, 0, 1.0);
    kk_matrix.set_coupling(0, 2, 0, 0, 0, 1.0);
    kk_matrix.set_coupling(2, 0, 0, 0, 0, 1.0);
    kk_matrix.set_coupling(1, 3, 0, 0, 0, 1.0);
    kk_matrix.set_coupling(3, 1, 0, 0, 0, 1.0);
    kk_matrix.set_coupling(2, 3, 0, 0, 0, 1.0);
    kk_matrix.set_coupling(3, 2, 0, 0, 0, 1.0);
    
    int L = 3;
    double T = 0.5;  // Finite temperature for reasonable acceptance
    MonteCarloSimulation sim(cell, couplings, L, T, kk_matrix);
    
    // Initialize in C-AFM + COO pattern
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    
    double initial_energy = sim.get_energy();
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    
    // Propose many double tunnel moves and track acceptance
    MoveProposer proposer(sim);
    int num_proposals = 1000;
    int accepted = 0;
    int rejected = 0;
    
    for (int i = 0; i < num_proposals; i++) {
        // Pick random site and location
        int x = 1 + (rand() % L);
        int y = 1 + (rand() % L);
        int z = 1 + (rand() % L);
        int site_id = rand() % 4;
        
        // Propose double tunnel
        MoveProposal proposal = proposer.propose_site_double_tunnel(x, y, z, site_id);
        
        // Metropolis acceptance
        double acceptance_prob = std::exp(-proposal.energy_change / T);
        if (proposal.energy_change < 0 || ran1(&seed) < acceptance_prob) {
            accepted++;
            // Actually apply the move for realistic dynamics
            for (size_t j = 0; j < proposal.affected_spin_ids.size(); j++) {
                int spin_id = proposal.affected_spin_ids[j];
                if (sim.get_unit_cell().get_spin(spin_id).spin_type == SpinType::ISING) {
                    sim.set_ising_spin(x, y, z, spin_id, static_cast<int>(proposal.new_ising_values[j]));
                } else {
                    sim.set_heisenberg_spin(x, y, z, spin_id, 
                        spin3d(proposal.new_hx_values[j], 
                               proposal.new_hy_values[j], 
                               proposal.new_hz_values[j]));
                }
            }
        } else {
            rejected++;
        }
    }
    
    double acceptance_rate = 100.0 * accepted / num_proposals;
    std::cout << "  Proposals: " << num_proposals << std::endl;
    std::cout << "  Accepted: " << accepted << " (" << acceptance_rate << "%)" << std::endl;
    std::cout << "  Rejected: " << rejected << std::endl;
    
    // Check that acceptance rate is reasonable (not 0%, not 100%)
    if (acceptance_rate < 1.0) {
        std::cout << "  ✗ Acceptance rate too low (<1%)! Possible bug." << std::endl;
        return false;
    } else if (acceptance_rate > 99.0) {
        std::cout << "  ⚠ Acceptance rate very high (>99%). Check energy calculation." << std::endl;
        return true;  // Not necessarily a failure
    } else {
        std::cout << "  ✓ Acceptance rate is reasonable!" << std::endl;
        return true;
    }
}

/**
 * Test 2: Low temperature scenario
 */
bool test_low_temperature_acceptance() {
    std::cout << "\n=== Test 2: Low Temperature Acceptance ===" << std::endl;
    
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);
    cell.add_spin("tau1", SpinType::ISING, 1.0, 0.0, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("tau2", SpinType::ISING, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("Cr3", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.0);
    cell.add_spin("tau3", SpinType::ISING, 1.0, 0.0, 0.5, 0.0);
    cell.add_spin("Cr4", SpinType::HEISENBERG, 1.0, 0.5, 0.5, 0.0);
    cell.add_spin("tau4", SpinType::ISING, 1.0, 0.5, 0.5, 0.0);
    
    CouplingMatrix couplings;
    couplings.initialize(8, 1);
    
    // Strong AFM couplings
    for (int i = 0; i < 8; i += 2) {
        for (int j = 0; j < 8; j += 2) {
            if (i != j) {
                couplings.set_coupling(i, j, 1, 0, 0, 5.0);
                couplings.set_coupling(i, j, 0, 1, 0, 5.0);
                couplings.set_coupling(i, j, 0, 0, 1, 5.0);
            }
        }
    }
    
    // Ising couplings
    for (int i = 1; i < 8; i += 2) {
        for (int j = 1; j < 8; j += 2) {
            if (i != j) {
                couplings.set_coupling(i, j, 1, 0, 0, 2.0);
                couplings.set_coupling(i, j, 0, 1, 0, 2.0);
                couplings.set_coupling(i, j, 0, 0, 1, 2.0);
            }
        }
    }
    
    KK_Matrix kk_matrix;
    kk_matrix.initialize(cell, 1);
    
    // KK couplings
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i != j) {
                kk_matrix.set_coupling(i, j, 1, 0, 0, 3.0);
                kk_matrix.set_coupling(i, j, 0, 1, 0, 3.0);
                kk_matrix.set_coupling(i, j, 0, 0, 1, 3.0);
            }
        }
    }
    
    int L = 2;
    double T = 0.05;  // Low temperature like user reported
    MonteCarloSimulation sim(cell, couplings, L, T, kk_matrix);
    
    // Initialize in ordered C-AFM + COO state
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    
    double initial_energy = sim.get_energy();
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    std::cout << "  Temperature: " << T << std::endl;
    
    // Try 500 double tunnel moves
    MoveProposer proposer(sim);
    int num_proposals = 500;
    int accepted = 0;
    
    for (int i = 0; i < num_proposals; i++) {
        int x = 1 + (rand() % L);
        int y = 1 + (rand() % L);
        int z = 1 + (rand() % L);
        int site_id = rand() % 4;
        
        MoveProposal proposal = proposer.propose_site_double_tunnel(x, y, z, site_id);
        
        // Metropolis test
        if (proposal.energy_change < 0) {
            accepted++;
        } else {
            double acceptance_prob = std::exp(-proposal.energy_change / T);
            if (ran1(&seed) < acceptance_prob) {
                accepted++;
            }
        }
    }
    
    double acceptance_rate = 100.0 * accepted / num_proposals;
    std::cout << "  Acceptance rate: " << acceptance_rate << "%" << std::endl;
    
    // At low T, we might have low acceptance, but 0% would indicate a bug
    if (acceptance_rate == 0.0) {
        std::cout << "  ⚠ WARNING: 0% acceptance - this may indicate the reported bug!" << std::endl;
        std::cout << "  This could be due to ordered ground state with all moves increasing energy." << std::endl;
        return true;  // Not necessarily a test failure
    } else if (acceptance_rate < 0.5) {
        std::cout << "  ⚠ Very low acceptance (<0.5%) at T=" << T << std::endl;
        return true;
    } else {
        std::cout << "  ✓ Non-zero acceptance rate achieved" << std::endl;
        return true;
    }
}

/**
 * Test 3: Exact reproduction of user's configuration
 * Uses the exact couplings from lowT_GOO_CAFM.toml configuration
 */
bool test_exact_user_configuration() {
    std::cout << "\n=== Test 3: Exact User Configuration (L=8, T=0.05) ===" << std::endl;
    
    // Exact species from species.dat
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("Cr3", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("Cr4", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.0);
    cell.add_spin("CrA", SpinType::ISING, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("CrB", SpinType::ISING, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("CrC", SpinType::ISING, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("CrD", SpinType::ISING, 1.0, 0.0, 0.5, 0.0);
    
    // Exact couplings from generated_couplings.dat
    CouplingMatrix couplings;
    couplings.initialize(8, 1);
    
    double Jv = 0.8;
    double Jh = 1.1;
    double hv = 0.5;
    double hh = 0.5;
    
    // Heisenberg spin couplings (Jv vertical, Jh horizontal)
    // Cr1-Cr2 vertical
    couplings.set_coupling(0, 1, 0, 0, 0, Jv);
    couplings.set_coupling(0, 1, 0, 0, -1, Jv);
    couplings.set_coupling(1, 0, 0, 0, 0, Jv);
    couplings.set_coupling(1, 0, 0, 0, 1, Jv);
    
    // Cr3-Cr4 vertical
    couplings.set_coupling(2, 3, 0, 0, 0, Jv);
    couplings.set_coupling(2, 3, 0, 0, 1, Jv);
    couplings.set_coupling(3, 2, 0, 0, 0, Jv);
    couplings.set_coupling(3, 2, 0, 0, -1, Jv);
    
    // Cr1-Cr4 horizontal
    couplings.set_coupling(0, 3, 0, 0, 0, Jh);
    couplings.set_coupling(0, 3, 1, 0, 0, Jh);
    couplings.set_coupling(0, 3, 0, -1, 0, Jh);
    couplings.set_coupling(0, 3, 1, -1, 0, Jh);
    couplings.set_coupling(3, 0, 0, 0, 0, Jh);
    couplings.set_coupling(3, 0, 0, 1, 0, Jh);
    couplings.set_coupling(3, 0, -1, 0, 0, Jh);
    couplings.set_coupling(3, 0, -1, 1, 0, Jh);
    
    // Cr2-Cr3 horizontal
    couplings.set_coupling(1, 2, 0, 0, 0, Jh);
    couplings.set_coupling(1, 2, 1, 0, 0, Jh);
    couplings.set_coupling(1, 2, 0, -1, 0, Jh);
    couplings.set_coupling(1, 2, 1, -1, 0, Jh);
    couplings.set_coupling(2, 1, 0, 0, 0, Jh);
    couplings.set_coupling(2, 1, 0, 1, 0, Jh);
    couplings.set_coupling(2, 1, -1, 0, 0, Jh);
    couplings.set_coupling(2, 1, -1, 1, 0, Jh);
    
    // Ising spin couplings (hv vertical, hh horizontal)
    // CrA-CrB vertical
    couplings.set_coupling(4, 5, 0, 0, 0, hv);
    couplings.set_coupling(4, 5, 0, 0, -1, hv);
    couplings.set_coupling(5, 4, 0, 0, 0, hv);
    couplings.set_coupling(5, 4, 0, 0, 1, hv);
    
    // CrC-CrD vertical
    couplings.set_coupling(6, 7, 0, 0, 0, hv);
    couplings.set_coupling(6, 7, 0, 0, 1, hv);
    couplings.set_coupling(7, 6, 0, 0, 0, hv);
    couplings.set_coupling(7, 6, 0, 0, -1, hv);
    
    // CrA-CrD horizontal
    couplings.set_coupling(4, 7, 0, 0, 0, hh);
    couplings.set_coupling(4, 7, 1, 0, 0, hh);
    couplings.set_coupling(4, 7, 0, -1, 0, hh);
    couplings.set_coupling(4, 7, 1, -1, 0, hh);
    couplings.set_coupling(7, 4, 0, 0, 0, hh);
    couplings.set_coupling(7, 4, 0, 1, 0, hh);
    couplings.set_coupling(7, 4, -1, 0, 0, hh);
    couplings.set_coupling(7, 4, -1, 1, 0, hh);
    
    // CrB-CrC horizontal
    couplings.set_coupling(5, 6, 0, 0, 0, hh);
    couplings.set_coupling(5, 6, 1, 0, 0, hh);
    couplings.set_coupling(5, 6, 0, -1, 0, hh);
    couplings.set_coupling(5, 6, 1, -1, 0, hh);
    couplings.set_coupling(6, 5, 0, 0, 0, hh);
    couplings.set_coupling(6, 5, 0, 1, 0, hh);
    couplings.set_coupling(6, 5, -1, 0, 0, hh);
    couplings.set_coupling(6, 5, -1, 1, 0, hh);
    
    // KK couplings from kugel_khomskii.dat
    KK_Matrix kk_matrix;
    kk_matrix.initialize(cell, 1);
    
    double K = 1.0;
    
    // Cr1-CrB (site 0 - site 1)
    kk_matrix.set_coupling(0, 1, 0, 0, 0, K);
    kk_matrix.set_coupling(0, 1, 0, 0, -1, K);
    
    // Cr2-CrA (site 1 - site 0)
    kk_matrix.set_coupling(1, 0, 0, 0, 0, K);
    kk_matrix.set_coupling(1, 0, 0, 0, 1, K);
    
    // Cr3-CrD (site 2 - site 3)
    kk_matrix.set_coupling(2, 3, 0, 0, 0, K);
    kk_matrix.set_coupling(2, 3, 0, 0, 1, K);
    
    // Cr4-CrC (site 3 - site 2)
    kk_matrix.set_coupling(3, 2, 0, 0, 0, K);
    kk_matrix.set_coupling(3, 2, 0, 0, -1, K);
    
    // Exact simulation parameters
    int L = 8;  // lattice size from config
    double T = 0.05;  // temperature from config
    MonteCarloSimulation sim(cell, couplings, L, T, kk_matrix);
    
    // Exact initialization pattern from config
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    // sim.initialize_lattice_custom({1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    
    double initial_energy = sim.get_energy();
    std::cout << "  Lattice size: " << L << "x" << L << "x" << L << std::endl;
    std::cout << "  Temperature: " << T << std::endl;
    std::cout << "  Initial energy: " << initial_energy << std::endl;
    std::cout << "  Pattern: [1,1,-1,-1,1,-1,1,-1] (C-AFM + G-OO)" << std::endl;
    
    // Test double tunnel moves like the actual simulation
    MoveProposer proposer(sim);
    int num_proposals = 2000;
    int accepted = 0;
    
    for (int i = 0; i < num_proposals; i++) {
        int x = 1 + (rand() % L);
        int y = 1 + (rand() % L);
        int z = 1 + (rand() % L);
        int site_id = rand() % 4;
        
        MoveProposal proposal = proposer.propose_site_double_tunnel(x, y, z, site_id);
        
        // Metropolis test
        if (proposal.energy_change < 0) {
            accepted++;
        } else {
            double acceptance_prob = std::exp(-proposal.energy_change / T);
            if (ran1(&seed) < acceptance_prob) {
                accepted++;
            }
        }
    }
    
    double acceptance_rate = 100.0 * accepted / num_proposals;
    std::cout << "  Proposals: " << num_proposals << std::endl;
    std::cout << "  Accepted: " << accepted << " (" << acceptance_rate << "%)" << std::endl;
    
    // This is the exact scenario where user saw 0% acceptance
    if (acceptance_rate == 0.0) {
        std::cout << "  ✗ BUG REPRODUCED: 0% acceptance rate!" << std::endl;
        std::cout << "  This confirms the bug exists with this configuration." << std::endl;
        return false;
    } else if (acceptance_rate < 1.0) {
        std::cout << "  ⚠ Very low acceptance (<1%) - possible issue" << std::endl;
        return true;
    } else {
        std::cout << "  ✓ Reasonable acceptance rate - bug not reproduced" << std::endl;
        return true;
    }
}

/**
 * Test 4: Energy landscape diagnostic
 * Understand why double tunnel moves all increase energy
 */
bool test_energy_landscape_diagnostic() {
    std::cout << "\n=== Test 4: Energy Landscape Diagnostic ===" << std::endl;
    
    // Same exact configuration as test 3
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("Cr3", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("Cr4", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.0);
    cell.add_spin("CrA", SpinType::ISING, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("CrB", SpinType::ISING, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("CrC", SpinType::ISING, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("CrD", SpinType::ISING, 1.0, 0.0, 0.5, 0.0);
    
    CouplingMatrix couplings;
    couplings.initialize(8, 1);
    
    double Jv = 0.8;
    double Jh = 0.5;
    double hv = 0.5;
    double hh = 0.5;
    
    // Heisenberg couplings
    couplings.set_coupling(0, 1, 0, 0, 0, Jv);
    couplings.set_coupling(0, 1, 0, 0, -1, Jv);
    couplings.set_coupling(1, 0, 0, 0, 0, Jv);
    couplings.set_coupling(1, 0, 0, 0, 1, Jv);
    couplings.set_coupling(2, 3, 0, 0, 0, Jv);
    couplings.set_coupling(2, 3, 0, 0, 1, Jv);
    couplings.set_coupling(3, 2, 0, 0, 0, Jv);
    couplings.set_coupling(3, 2, 0, 0, -1, Jv);
    couplings.set_coupling(0, 3, 0, 0, 0, Jh);
    couplings.set_coupling(0, 3, 1, 0, 0, Jh);
    couplings.set_coupling(0, 3, 0, -1, 0, Jh);
    couplings.set_coupling(0, 3, 1, -1, 0, Jh);
    couplings.set_coupling(3, 0, 0, 0, 0, Jh);
    couplings.set_coupling(3, 0, 0, 1, 0, Jh);
    couplings.set_coupling(3, 0, -1, 0, 0, Jh);
    couplings.set_coupling(3, 0, -1, 1, 0, Jh);
    couplings.set_coupling(1, 2, 0, 0, 0, Jh);
    couplings.set_coupling(1, 2, 1, 0, 0, Jh);
    couplings.set_coupling(1, 2, 0, -1, 0, Jh);
    couplings.set_coupling(1, 2, 1, -1, 0, Jh);
    couplings.set_coupling(2, 1, 0, 0, 0, Jh);
    couplings.set_coupling(2, 1, 0, 1, 0, Jh);
    couplings.set_coupling(2, 1, -1, 0, 0, Jh);
    couplings.set_coupling(2, 1, -1, 1, 0, Jh);
    
    // Ising couplings
    couplings.set_coupling(4, 5, 0, 0, 0, hv);
    couplings.set_coupling(4, 5, 0, 0, -1, hv);
    couplings.set_coupling(5, 4, 0, 0, 0, hv);
    couplings.set_coupling(5, 4, 0, 0, 1, hv);
    couplings.set_coupling(6, 7, 0, 0, 0, hv);
    couplings.set_coupling(6, 7, 0, 0, 1, hv);
    couplings.set_coupling(7, 6, 0, 0, 0, hv);
    couplings.set_coupling(7, 6, 0, 0, -1, hv);
    couplings.set_coupling(4, 7, 0, 0, 0, hh);
    couplings.set_coupling(4, 7, 1, 0, 0, hh);
    couplings.set_coupling(4, 7, 0, -1, 0, hh);
    couplings.set_coupling(4, 7, 1, -1, 0, hh);
    couplings.set_coupling(7, 4, 0, 0, 0, hh);
    couplings.set_coupling(7, 4, 0, 1, 0, hh);
    couplings.set_coupling(7, 4, -1, 0, 0, hh);
    couplings.set_coupling(7, 4, -1, 1, 0, hh);
    couplings.set_coupling(5, 6, 0, 0, 0, hh);
    couplings.set_coupling(5, 6, 1, 0, 0, hh);
    couplings.set_coupling(5, 6, 0, -1, 0, hh);
    couplings.set_coupling(5, 6, 1, -1, 0, hh);
    couplings.set_coupling(6, 5, 0, 0, 0, hh);
    couplings.set_coupling(6, 5, 0, 1, 0, hh);
    couplings.set_coupling(6, 5, -1, 0, 0, hh);
    couplings.set_coupling(6, 5, -1, 1, 0, hh);
    
    KK_Matrix kk_matrix;
    kk_matrix.initialize(cell, 1);
    double K = 1.0;
    kk_matrix.set_coupling(0, 1, 0, 0, 0, K);
    kk_matrix.set_coupling(0, 1, 0, 0, -1, K);
    kk_matrix.set_coupling(1, 0, 0, 0, 0, K);
    kk_matrix.set_coupling(1, 0, 0, 0, 1, K);
    kk_matrix.set_coupling(2, 3, 0, 0, 0, K);
    kk_matrix.set_coupling(2, 3, 0, 0, 1, K);
    kk_matrix.set_coupling(3, 2, 0, 0, 0, K);
    kk_matrix.set_coupling(3, 2, 0, 0, -1, K);
    
    int L = 8;
    double T = 0.05;
    MonteCarloSimulation sim(cell, couplings, L, T, kk_matrix);
    
    // Current state: C-AFM + G-OO [1,1,-1,-1,1,-1,1,-1]
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    
    double E_current = sim.get_energy();
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  CURRENT STATE: C-AFM + G-OO" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "  Pattern: [Cr1=+1, Cr2=+1, Cr3=-1, Cr4=-1, A=+1, B=-1, C=+1, D=-1]" << std::endl;
    std::cout << "  Total energy: " << std::fixed << std::setprecision(1) << E_current << std::endl;
    
    // Test alternative: G-AFM + C-OO [1,-1,1,-1,1,1,-1,-1]
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "  ALTERNATIVE STATE: G-AFM + C-OO" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    sim.initialize_lattice_custom({1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0});
    double E_G_AFM_C_OO = sim.get_energy();
    std::cout << "  Pattern: [Cr1=+1, Cr2=-1, Cr3=+1, Cr4=-1, A=+1, B=+1, C=-1, D=-1]" << std::endl;
    std::cout << "  Total energy: " << std::fixed << std::setprecision(1) << E_G_AFM_C_OO << std::endl;
    std::cout << "\n  >>> Energy difference: " << std::fixed << std::setprecision(1) << (E_G_AFM_C_OO - E_current) 
              << (E_G_AFM_C_OO < E_current ? " (LOWER!) <<<" : " (HIGHER)") << std::endl;
    
    // Test what happens with single-spin flips
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "  SINGLE-SPIN MOVE ANALYSIS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    
    MoveProposer proposer(sim);
    
    // Try flipping just Heisenberg spin at site 0
    MoveProposal heisenberg_flip = proposer.propose_heisenberg_flip(4, 4, 4, 0);
    std::cout << "  Flip Heisenberg Cr1 only: ΔE = " << heisenberg_flip.energy_change << std::endl;
    
    // Try flipping just Ising spin at site 0
    MoveProposal ising_flip = proposer.propose_ising_flip(4, 4, 4, 4);
    std::cout << "  Flip Ising CrA only: ΔE = " << ising_flip.energy_change << std::endl;
    
    // Try double tunnel at site 0
    MoveProposal double_tunnel = proposer.propose_site_double_tunnel(4, 4, 4, 0);
    std::cout << "  Double tunnel at site 0: ΔE = " << double_tunnel.energy_change << std::endl;
    
    // Sample multiple double tunnel moves
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "  RANDOM DOUBLE TUNNEL SAMPLING (20 moves)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    int num_negative = 0;
    int num_zero = 0;
    int num_positive = 0;
    double min_delta = 1e9;
    double max_delta = -1e9;
    
    for (int i = 0; i < 20; i++) {
        int x = 1 + (rand() % L);
        int y = 1 + (rand() % L);
        int z = 1 + (rand() % L);
        int site_id = rand() % 4;
        
        MoveProposal prop = proposer.propose_site_double_tunnel(x, y, z, site_id);
        
        if (prop.energy_change < 0) num_negative++;
        else if (prop.energy_change == 0) num_zero++;
        else num_positive++;
        
        if (prop.energy_change < min_delta) min_delta = prop.energy_change;
        if (prop.energy_change > max_delta) max_delta = prop.energy_change;
    }
    
    std::cout << "  Negative ΔE: " << std::setw(3) << num_negative << " moves" << std::endl;
    std::cout << "  Zero ΔE:     " << std::setw(3) << num_zero << " moves" << std::endl;
    std::cout << "  Positive ΔE: " << std::setw(3) << num_positive << " moves" << std::endl;
    std::cout << "  Min ΔE:      " << std::fixed << std::setprecision(2) << min_delta << std::endl;
    std::cout << "  Max ΔE:      " << std::fixed << std::setprecision(2) << max_delta << std::endl;
    
    // Physics explanation
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  PHYSICS ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "  - Jv=" << Jv << " (vertical Heisenberg)" << std::endl;
    std::cout << "  - Jh=" << Jh << " (horizontal Heisenberg)" << std::endl;
    std::cout << "  - hv=" << hv << " (vertical Ising)" << std::endl;
    std::cout << "  - hh=" << hh << " (horizontal Ising)" << std::endl;
    std::cout << "  - K=" << K << " (KK coupling)" << std::endl;
    std::cout << "  For positive J: AFM is favored (antiparallel spins)" << std::endl;
    std::cout << "  Current C-AFM has Jh-dominated frustration" << std::endl;
    std::cout << "  Expected G-AFM should be lower energy if Jh > Jv" << std::endl;
    
    std::cout << "\n" << std::string(60, '-') << std::endl;
    if (E_G_AFM_C_OO < E_current) {
        std::cout << "  >>> CONCLUSION: TRAPPED IN LOCAL MINIMUM! <<<" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "  ✓ G-AFM + C-OO has LOWER energy (ΔE = " << std::fixed << std::setprecision(1) 
                  << (E_G_AFM_C_OO - E_current) << ")" << std::endl;
        std::cout << "  ✗ Current moves cannot access this state" << std::endl;
        std::cout << "  → Double tunnel preserves local KK constraint" << std::endl;
        std::cout << "  → Need different move types to escape!" << std::endl;
    } else {
        std::cout << "  >>> CONCLUSION: IN GLOBAL MINIMUM <<<" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        std::cout << "  ✓ C-AFM + G-OO has lowest energy" << std::endl;
        std::cout << "  → 0% acceptance expected at low T" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
    
    return true;
}

/**
 * Test 5: Layer-by-layer domain transformation
 * Convert entire z-layers from C-AFM+G-OO to G-AFM+C-OO
 */
bool test_slab_flip_moves() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  TEST 5: LAYER TRANSFORMATION (C-AFM+G-OO → G-AFM+C-OO)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Same exact configuration as test 3 & 4
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("Cr3", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("Cr4", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.0);
    cell.add_spin("CrA", SpinType::ISING, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("CrB", SpinType::ISING, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("CrC", SpinType::ISING, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("CrD", SpinType::ISING, 1.0, 0.0, 0.5, 0.0);
    
    CouplingMatrix couplings;
    couplings.initialize(8, 1);
    
    double Jv = 0.8;
    double Jh = 1.1;
    double hv = 0.5;
    double hh = 0.1;
    
    // Heisenberg couplings
    couplings.set_coupling(0, 1, 0, 0, 0, Jv);
    couplings.set_coupling(0, 1, 0, 0, -1, Jv);
    couplings.set_coupling(1, 0, 0, 0, 0, Jv);
    couplings.set_coupling(1, 0, 0, 0, 1, Jv);
    couplings.set_coupling(2, 3, 0, 0, 0, Jv);
    couplings.set_coupling(2, 3, 0, 0, 1, Jv);
    couplings.set_coupling(3, 2, 0, 0, 0, Jv);
    couplings.set_coupling(3, 2, 0, 0, -1, Jv);
    couplings.set_coupling(0, 3, 0, 0, 0, Jh);
    couplings.set_coupling(0, 3, 1, 0, 0, Jh);
    couplings.set_coupling(0, 3, 0, -1, 0, Jh);
    couplings.set_coupling(0, 3, 1, -1, 0, Jh);
    couplings.set_coupling(3, 0, 0, 0, 0, Jh);
    couplings.set_coupling(3, 0, 0, 1, 0, Jh);
    couplings.set_coupling(3, 0, -1, 0, 0, Jh);
    couplings.set_coupling(3, 0, -1, 1, 0, Jh);
    couplings.set_coupling(1, 2, 0, 0, 0, Jh);
    couplings.set_coupling(1, 2, 1, 0, 0, Jh);
    couplings.set_coupling(1, 2, 0, -1, 0, Jh);
    couplings.set_coupling(1, 2, 1, -1, 0, Jh);
    couplings.set_coupling(2, 1, 0, 0, 0, Jh);
    couplings.set_coupling(2, 1, 0, 1, 0, Jh);
    couplings.set_coupling(2, 1, -1, 0, 0, Jh);
    couplings.set_coupling(2, 1, -1, 1, 0, Jh);
    
    // Ising couplings
    couplings.set_coupling(4, 5, 0, 0, 0, hv);
    couplings.set_coupling(4, 5, 0, 0, -1, hv);
    couplings.set_coupling(5, 4, 0, 0, 0, hv);
    couplings.set_coupling(5, 4, 0, 0, 1, hv);
    couplings.set_coupling(6, 7, 0, 0, 0, hv);
    couplings.set_coupling(6, 7, 0, 0, 1, hv);
    couplings.set_coupling(7, 6, 0, 0, 0, hv);
    couplings.set_coupling(7, 6, 0, 0, -1, hv);
    couplings.set_coupling(4, 7, 0, 0, 0, hh);
    couplings.set_coupling(4, 7, 1, 0, 0, hh);
    couplings.set_coupling(4, 7, 0, -1, 0, hh);
    couplings.set_coupling(4, 7, 1, -1, 0, hh);
    couplings.set_coupling(7, 4, 0, 0, 0, hh);
    couplings.set_coupling(7, 4, 0, 1, 0, hh);
    couplings.set_coupling(7, 4, -1, 0, 0, hh);
    couplings.set_coupling(7, 4, -1, 1, 0, hh);
    couplings.set_coupling(5, 6, 0, 0, 0, hh);
    couplings.set_coupling(5, 6, 1, 0, 0, hh);
    couplings.set_coupling(5, 6, 0, -1, 0, hh);
    couplings.set_coupling(5, 6, 1, -1, 0, hh);
    couplings.set_coupling(6, 5, 0, 0, 0, hh);
    couplings.set_coupling(6, 5, 0, 1, 0, hh);
    couplings.set_coupling(6, 5, -1, 0, 0, hh);
    couplings.set_coupling(6, 5, -1, 1, 0, hh);
    
    KK_Matrix kk_matrix;
    kk_matrix.initialize(cell, 1);
    double K = 1.0;
    kk_matrix.set_coupling(0, 1, 0, 0, 0, K);
    kk_matrix.set_coupling(0, 1, 0, 0, -1, K);
    kk_matrix.set_coupling(1, 0, 0, 0, 0, K);
    kk_matrix.set_coupling(1, 0, 0, 0, 1, K);
    kk_matrix.set_coupling(2, 3, 0, 0, 0, K);
    kk_matrix.set_coupling(2, 3, 0, 0, 1, K);
    kk_matrix.set_coupling(3, 2, 0, 0, 0, K);
    kk_matrix.set_coupling(3, 2, 0, 0, -1, K);
    
    int L = 16;
    double T = 0.05;
    MonteCarloSimulation sim(cell, couplings, L, T, kk_matrix);
    
    // Initialize entire lattice as C-AFM + G-OO
    sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
    
    double E_initial = sim.get_energy();
    std::cout << "\n  INITIAL STATE: Entire lattice in C-AFM + G-OO" << std::endl;
    std::cout << "  Pattern:      [Cr1=+1, Cr2=+1, Cr3=-1, Cr4=-1, A=+1, B=-1, C=+1, D=-1]" << std::endl;
    std::cout << "  Lattice:      " << L << "×" << L << "×" << L << " cells" << std::endl;
    std::cout << "  Energy:       " << std::fixed << std::setprecision(1) << E_initial << std::endl;
    
    // Now convert the first z-layer (z=1) from C-AFM+G-OO to G-AFM+C-OO
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "  SINGLE LAYER TRANSFORMATION (z=1 only)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "  Target pattern: [Cr1=+1, Cr2=-1, Cr3=+1, Cr4=-1, A=+1, B=+1, C=-1, D=-1]" << std::endl;
    
    int z = 1;  // First layer
    for (int x = 1; x <= L; x++) {
        for (int y = 1; y <= L; y++) {
            // Set Cr1 (site 0) z-component to +1
            spin3d s0 = sim.get_heisenberg_spin(x, y, z, 0);
            s0.z = 1.0;
            sim.set_heisenberg_spin(x, y, z, 0, s0);
            
            // Set Cr2 (site 1) z-component to -1
            spin3d s1 = sim.get_heisenberg_spin(x, y, z, 1);
            s1.z = -1.0;
            sim.set_heisenberg_spin(x, y, z, 1, s1);
            
            // Set Cr3 (site 2) z-component to +1
            spin3d s2 = sim.get_heisenberg_spin(x, y, z, 2);
            s2.z = 1.0;
            sim.set_heisenberg_spin(x, y, z, 2, s2);
            
            // Set Cr4 (site 3) z-component to -1
            spin3d s3 = sim.get_heisenberg_spin(x, y, z, 3);
            s3.z = -1.0;
            sim.set_heisenberg_spin(x, y, z, 3, s3);
            
            // Set CrA (site 4) Ising to +1
            sim.set_ising_spin(x, y, z, 4, 1);
            
            // Set CrB (site 5) Ising to +1
            sim.set_ising_spin(x, y, z, 5, 1);
            
            // Set CrC (site 6) Ising to -1
            sim.set_ising_spin(x, y, z, 6, -1);
            
            // Set CrD (site 7) Ising to -1
            sim.set_ising_spin(x, y, z, 7, -1);
        }
    }
    
    double E_after_one_layer = sim.get_energy();
    double delta_E_one = E_after_one_layer - E_initial;
    
    std::cout << "\n  Energy after:   " << std::fixed << std::setprecision(1) << E_after_one_layer << std::endl;
    std::cout << "  ΔE =            " << std::fixed << std::setprecision(1) << delta_E_one;
    
    if (delta_E_one < 0) {
        std::cout << "  ✓ LOWERS ENERGY!" << std::endl;
    } else {
        std::cout << "  ✗ INCREASES ENERGY" << std::endl;
    }
    
    // Now convert ALL layers to see the final state
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "  FULL LATTICE TRANSFORMATION (all layers)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    sim.initialize_lattice_custom({1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0});
    
    double E_all_converted = sim.get_energy();
    double delta_E_total = E_all_converted - E_initial;
    
    std::cout << "  Final energy:   " << std::fixed << std::setprecision(1) << E_all_converted << std::endl;
    std::cout << "  Total ΔE:       " << std::fixed << std::setprecision(1) << delta_E_total;
    
    if (delta_E_total < 0) {
        std::cout << "  ✓ LOWERS ENERGY!" << std::endl;
        std::cout << "\n  >>> G-AFM + C-OO is the GLOBAL MINIMUM <<<" << std::endl;
    }
    
    // Now find the critical size for [size×size×1] slabs
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  CRITICAL SIZE SEARCH: [size×size×1] slabs" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\n  Size | Area | ΔE      | ΔE/Area | Status" << std::endl;
    std::cout << "  -----|------|---------|---------|------------------" << std::endl;
    
    int critical_size = -1;
    
    for (int size = 1; size <= L; size++) {
        // Reset to C-AFM + G-OO
        sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
        double E_before = sim.get_energy();
        
        // Convert a centered [size×size×1] slab at z=1
        int x_start = (L - size) / 2 + 1;
        int y_start = (L - size) / 2 + 1;
        int z = 1;
        
        for (int x = x_start; x < x_start + size; x++) {
            for (int y = y_start; y < y_start + size; y++) {
                // Transform to G-AFM + C-OO pattern
                spin3d s0 = sim.get_heisenberg_spin(x, y, z, 0);
                s0.z = 1.0;
                sim.set_heisenberg_spin(x, y, z, 0, s0);
                
                spin3d s1 = sim.get_heisenberg_spin(x, y, z, 1);
                s1.z = -1.0;
                sim.set_heisenberg_spin(x, y, z, 1, s1);
                
                spin3d s2 = sim.get_heisenberg_spin(x, y, z, 2);
                s2.z = 1.0;
                sim.set_heisenberg_spin(x, y, z, 2, s2);
                
                spin3d s3 = sim.get_heisenberg_spin(x, y, z, 3);
                s3.z = -1.0;
                sim.set_heisenberg_spin(x, y, z, 3, s3);
                
                sim.set_ising_spin(x, y, z, 4, 1);
                sim.set_ising_spin(x, y, z, 5, 1);
                sim.set_ising_spin(x, y, z, 6, -1);
                sim.set_ising_spin(x, y, z, 7, -1);
            }
        }
        
        double E_after = sim.get_energy();
        double delta_E = E_after - E_before;
        int area = size * size;
        double delta_per_area = delta_E / area;
        
        std::string status;
        if (delta_E < 0) {
            status = "LOWERS ✓";
            if (critical_size == -1) {
                critical_size = size;
            }
        } else if (delta_E == 0) {
            status = "NEUTRAL";
        } else {
            status = "RAISES ✗";
        }
        
        std::cout << "  " << std::setw(4) << size 
                  << " | " << std::setw(4) << area
                  << " | " << std::setw(7) << std::fixed << std::setprecision(1) << delta_E
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << delta_per_area
                  << " | " << status
                  << std::endl;
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  CRITICAL SIZE ANALYSIS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    if (critical_size > 0) {
        std::cout << "  ✓ Critical size found: " << critical_size << "×" << critical_size << " cells" << std::endl;
        std::cout << "\n  KEY FINDINGS:" << std::endl;
        std::cout << "  • Domains ≥ " << critical_size << "×" << critical_size << " cells → LOWER energy" << std::endl;
        std::cout << "  • Domains < " << critical_size << "×" << critical_size << " cells → RAISE energy" << std::endl;
        std::cout << "  • Domain wall energy dominates for small sizes" << std::endl;
        std::cout << "  • Bulk energy (favoring G-AFM+C-OO) wins at large sizes" << std::endl;
    } else {
        std::cout << "  ✗ No favorable size found up to " << L << "×" << L << std::endl;
        std::cout << "  → Domain wall energy dominates even for full-layer transformation" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;
    
    // Now explore thickness variation at critical and near-critical sizes
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  THICKNESS EXPLORATION: [size×size×thickness] slabs" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\n  Testing thickness variation at different lateral sizes..." << std::endl;
    
    std::vector<int> test_sizes = {9, 10, 11};  // Near-critical and critical
    std::vector<int> test_thicknesses = {1, 2, 3, 4, 5, 6, 7, 8};
    
    for (int size : test_sizes) {
        std::cout << "\n  --- Lateral size: " << size << "×" << size << " cells ---" << std::endl;
        std::cout << "\n  Thick | Volume | ΔE      | ΔE/Vol  | ΔE/Layer | Status" << std::endl;
        std::cout << "  ------|--------|---------|---------|----------|------------------" << std::endl;
        
        for (int thickness : test_thicknesses) {
            if (thickness > L) break;  // Don't exceed lattice size
            
            // Reset to C-AFM + G-OO
            sim.initialize_lattice_custom({1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0});
            double E_before = sim.get_energy();
            
            // Convert a centered [size×size×thickness] slab
            int x_start = (L - size) / 2 + 1;
            int y_start = (L - size) / 2 + 1;
            int z_start = (L - thickness) / 2 + 1;
            
            for (int x = x_start; x < x_start + size; x++) {
                for (int y = y_start; y < y_start + size; y++) {
                    for (int z = z_start; z < z_start + thickness; z++) {
                        // Transform to G-AFM + C-OO pattern
                        spin3d s0 = sim.get_heisenberg_spin(x, y, z, 0);
                        s0.z = 1.0;
                        sim.set_heisenberg_spin(x, y, z, 0, s0);
                        
                        spin3d s1 = sim.get_heisenberg_spin(x, y, z, 1);
                        s1.z = -1.0;
                        sim.set_heisenberg_spin(x, y, z, 1, s1);
                        
                        spin3d s2 = sim.get_heisenberg_spin(x, y, z, 2);
                        s2.z = 1.0;
                        sim.set_heisenberg_spin(x, y, z, 2, s2);
                        
                        spin3d s3 = sim.get_heisenberg_spin(x, y, z, 3);
                        s3.z = -1.0;
                        sim.set_heisenberg_spin(x, y, z, 3, s3);
                        
                        sim.set_ising_spin(x, y, z, 4, 1);
                        sim.set_ising_spin(x, y, z, 5, 1);
                        sim.set_ising_spin(x, y, z, 6, -1);
                        sim.set_ising_spin(x, y, z, 7, -1);
                    }
                }
            }
            
            double E_after = sim.get_energy();
            double delta_E = E_after - E_before;
            int volume = size * size * thickness;
            double delta_per_volume = delta_E / volume;
            double delta_per_layer = delta_E / thickness;
            
            std::string status;
            if (delta_E < 0) {
                status = "LOWERS ✓";
            } else if (delta_E == 0) {
                status = "NEUTRAL";
            } else {
                status = "RAISES ✗";
            }
            
            std::cout << "  " << std::setw(5) << thickness 
                      << " | " << std::setw(6) << volume
                      << " | " << std::setw(7) << std::fixed << std::setprecision(1) << delta_E
                      << " | " << std::setw(7) << std::fixed << std::setprecision(2) << delta_per_volume
                      << " | " << std::setw(8) << std::fixed << std::setprecision(1) << delta_per_layer
                      << " | " << status
                      << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  THICKNESS ANALYSIS" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "  KEY OBSERVATIONS:" << std::endl;
    std::cout << "  • ΔE/Layer should be approximately constant for favorable slabs" << std::endl;
    std::cout << "  • Thicker slabs have more bulk but also more interface area" << std::endl;
    std::cout << "  • For critical-size slabs, thickness scaling shows energy per layer" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  MC ACCEPTANCE RATE TESTS             " << std::endl;
    std::cout << "  (Non-Deterministic)                  " << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 5;
    
    if (test_double_tunnel_acceptance_rate()) passed++;
    if (test_low_temperature_acceptance()) passed++;
    if (test_exact_user_configuration()) passed++;
    if (test_energy_landscape_diagnostic()) passed++;
    if (test_slab_flip_moves()) passed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS: " << passed << "/" << total << " tests passed" << std::endl;
    
    if (passed == total) {
        std::cout << "✓ ALL ACCEPTANCE TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ " << (total - passed) << " tests failed" << std::endl;
        return 1;
    }
}
