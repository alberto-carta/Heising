/*
 * Multi-Atom Cell Energy Validation Tests
 * 
 * Tests energy calculations for various magnetic systems:
 * 1. Single-atom ferromagnet (FM)
 * 2. Single-atom antiferromagnet (AFM)
 * 3. 4-atom G-type antiferromagnet (G-AFM)
 * 
 * Each test verifies that computed energies match analytical expectations
 * for known ground states.
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
bool approx_equal(double a, double b, double tolerance = 1e-6) {
    return std::abs(a - b) < tolerance;
}

/**
 * Test 1: Single-atom ferromagnet with J < 0
 * Ground state: all spins aligned (e.g., all pointing up)
 * Expected energy per spin: -3J (negative, favorable)
 */
bool test_single_atom_ferromagnet() {
    std::cout << "\n=== Test 1: Single-Atom Ferromagnet (J=-1.0) ===" << std::endl;
    
    // Create single-spin unit cell at origin
    UnitCell cell;
    cell.add_spin("Fe", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);
    
    // Set up ferromagnetic NN couplings (J = -1.0)
    CouplingMatrix couplings;
    couplings.initialize(1, 1);  // 1 spin, max_offset = 1
    
    // 6 nearest neighbors in cubic lattice: ±x, ±y, ±z
    couplings.set_coupling(0, 0, 1, 0, 0, -1.0);   // +x
    couplings.set_coupling(0, 0, 0, 1, 0, -1.0);   // +y
    couplings.set_coupling(0, 0, 0, 0, 1, -1.0);   // +z
    couplings.set_coupling(0, 0, -1, 0, 0, -1.0);  // -x
    couplings.set_coupling(0, 0, 0, -1, 0, -1.0);  // -y
    couplings.set_coupling(0, 0, 0, 0, -1, -1.0);  // -z
    
    // Create simulation
    int L = 4;  // 4x4x4 lattice
    double T = 0.01;  // Low temperature
    MonteCarloSimulation sim(cell, couplings, L, T);
    
    // Initialize in ferromagnetic state (all spins pointing up: z=+1)
    sim.initialize_lattice_custom({1.0});
    
    double energy_per_spin = sim.get_energy() / (L * L * L);
    double expected_energy = -3.0;  // H = Σ J S·S, J=-1, parallel: 6 neighbors × (-1.0) × (+1) / 2 = -3.0
    
    std::cout << "  Energy per spin: " << std::fixed << std::setprecision(6) 
              << energy_per_spin << " (expected: " << expected_energy << ")" << std::endl;
    
    if (approx_equal(energy_per_spin, expected_energy)) {
        std::cout << "  ✓ Ferromagnet energy correct!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ Energy mismatch: got " << energy_per_spin 
                  << ", expected " << expected_energy << std::endl;
        return false;
    }
}

/**
 * Test 2: Single-atom antiferromagnet with J > 0
 * Ground state: checkerboard pattern (alternating up/down)
 * Expected energy per spin: positive (unfavorable for parallel, but this is the AFM ground state)
 */
bool test_single_atom_antiferromagnet() {
    std::cout << "\n=== Test 2: Single-Atom Antiferromagnet (J=+1.0) ===" << std::endl;
    
    // Create single-spin unit cell
    UnitCell cell;
    cell.add_spin("Cr", SpinType::HEISENBERG, 1.0, 0.0, 0.0, 0.0);
    
    // Set up antiferromagnetic NN couplings (J = +1.0)
    CouplingMatrix couplings;
    couplings.initialize(1, 1);
    
    couplings.set_coupling(0, 0, 1, 0, 0, 1.0);   // +x
    couplings.set_coupling(0, 0, 0, 1, 0, 1.0);   // +y
    couplings.set_coupling(0, 0, 0, 0, 1, 1.0);   // +z
    couplings.set_coupling(0, 0, -1, 0, 0, 1.0);  // -x
    couplings.set_coupling(0, 0, 0, -1, 0, 1.0);  // -y
    couplings.set_coupling(0, 0, 0, 0, -1, 1.0);  // -z
    
    int L = 4;
    double T = 0.01;
    MonteCarloSimulation sim(cell, couplings, L, T);
    
    // Initialize in checkerboard AFM state
    // Spin up if (x+y+z) is even, down if odd
    for (int x = 1; x <= L; x++) {
        for (int y = 1; y <= L; y++) {
            for (int z = 1; z <= L; z++) {
                double sz = ((x + y + z) % 2 == 0) ? 1.0 : -1.0;
                sim.set_heisenberg_spin(x, y, z, 0, spin3d(0.0, 0.0, sz));
            }
        }
    }
    
    double energy_per_spin = sim.get_energy() / (L * L * L);
    double expected_energy = -3.0;  // H = Σ J S·S, J=+1, antiparallel (S·S=-1): 6 × (+1) × (-1) / 2 = -3.0
    
    std::cout << "  Energy per spin: " << std::fixed << std::setprecision(6) 
              << energy_per_spin << " (expected: " << expected_energy << ")" << std::endl;
    
    if (approx_equal(energy_per_spin, expected_energy)) {
        std::cout << "  ✓ Antiferromagnet energy correct!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ Energy mismatch: got " << energy_per_spin 
                  << ", expected " << expected_energy << std::endl;
        return false;
    }
}

/**
 * Test 3: 4-atom cell with ferromagnetic couplings
 * Same geometry as G-AFM but with J < 0 (ferromagnetic)
 * Ground state: all spins aligned
 * Should give same energy per spin as single-atom ferromagnet
 */
bool test_4atom_ferromagnet() {
    std::cout << "\n=== Test 3: 4-Atom Ferromagnet (J=-1.0) ===" << std::endl;
    
    // Create 4-spin unit cell matching examples/4-atom_cell_GAFM/species.dat
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("Cr3", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("Cr4", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.0);
    
    // Set up ferromagnetic couplings (J = -1.0)
    CouplingMatrix couplings;
    couplings.initialize(4, 1);
    
    // Cr1-Cr2 pairs
    couplings.set_coupling(0, 1, 0, 0, 0, -1.0);
    couplings.set_coupling(0, 1, 0, 0, -1, -1.0);
    
    // Cr1-Cr4 pairs
    couplings.set_coupling(0, 3, 0, 0, 0, -1.0);
    couplings.set_coupling(0, 3, 1, 0, 0, -1.0);
    couplings.set_coupling(0, 3, 0, -1, 0, -1.0);
    couplings.set_coupling(0, 3, 1, -1, 0, -1.0);
    
    // Cr2-Cr1 pairs (reciprocal)
    couplings.set_coupling(1, 0, 0, 0, 0, -1.0);
    couplings.set_coupling(1, 0, 0, 0, 1, -1.0);
    
    // Cr2-Cr3 pairs
    couplings.set_coupling(1, 2, 0, 0, 0, -1.0);
    couplings.set_coupling(1, 2, 1, 0, 0, -1.0);
    couplings.set_coupling(1, 2, 0, -1, 0, -1.0);
    couplings.set_coupling(1, 2, 1, -1, 0, -1.0);
    
    // Cr3-Cr4 pairs
    couplings.set_coupling(2, 3, 0, 0, 0, -1.0);
    couplings.set_coupling(2, 3, 0, 0, 1, -1.0);
    
    // Cr3-Cr2 pairs (reciprocal)
    couplings.set_coupling(2, 1, 0, 0, 0, -1.0);
    couplings.set_coupling(2, 1, -1, 0, 0, -1.0);
    couplings.set_coupling(2, 1, -1, 1, 0, -1.0);
    couplings.set_coupling(2, 1, 0, 1, 0, -1.0);
    
    // Cr4-Cr3 pairs (reciprocal)
    couplings.set_coupling(3, 2, 0, 0, 0, -1.0);
    couplings.set_coupling(3, 2, 0, 0, -1, -1.0);
    
    // Cr4-Cr1 pairs (reciprocal)
    couplings.set_coupling(3, 0, 0, 0, 0, -1.0);
    couplings.set_coupling(3, 0, 0, 1, 0, -1.0);
    couplings.set_coupling(3, 0, -1, 0, 0, -1.0);
    couplings.set_coupling(3, 0, -1, 1, 0, -1.0);
    
    int L = 2;  // 2x2x2 lattice
    double T = 0.01;
    MonteCarloSimulation sim(cell, couplings, L, T);
    
    // Initialize in ferromagnetic state (all spins up)
    sim.initialize_lattice_custom({1.0, 1.0, 1.0, 1.0});
}


bool test_4atom_antiferromagnet() {
    std::cout << "\n=== Test 3: 4-Atom Antiferromagnet (J=1.0) ===" << std::endl;
    
    // Create 4-spin unit cell matching examples/4-atom_cell_GAFM/species.dat
    UnitCell cell;
    cell.add_spin("Cr1", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.0);
    cell.add_spin("Cr2", SpinType::HEISENBERG, 1.0, 0.5, 0.0, 0.5);
    cell.add_spin("Cr3", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.5);
    cell.add_spin("Cr4", SpinType::HEISENBERG, 1.0, 0.0, 0.5, 0.0);
    
    // Set up ferromagnetic couplings (J = -1.0)
    CouplingMatrix couplings;
    couplings.initialize(4, 1);
    
    // Cr1-Cr2 pairs
    couplings.set_coupling(0, 1, 0, 0, 0, 1.0);
    couplings.set_coupling(0, 1, 0, 0, -1, 1.0);
    
    // Cr1-Cr4 pairs
    couplings.set_coupling(0, 3, 0, 0, 0, 1.0);
    couplings.set_coupling(0, 3, 1, 0, 0, 1.0);
    couplings.set_coupling(0, 3, 0, -1, 0, 1.0);
    couplings.set_coupling(0, 3, 1, -1, 0, 1.0);
    
    // Cr2-Cr1 pairs (reciprocal)
    couplings.set_coupling(1, 0, 0, 0, 0, 1.0);
    couplings.set_coupling(1, 0, 0, 0, 1, 1.0);
    
    // Cr2-Cr3 pairs
    couplings.set_coupling(1, 2, 0, 0, 0, 1.0);
    couplings.set_coupling(1, 2, 1, 0, 0, 1.0);
    couplings.set_coupling(1, 2, 0, -1, 0, 1.0);
    couplings.set_coupling(1, 2, 1, -1, 0, 1.0);
    
    // Cr3-Cr4 pairs
    couplings.set_coupling(2, 3, 0, 0, 0, 1.0);
    couplings.set_coupling(2, 3, 0, 0, 1, 1.0);
    
    // Cr3-Cr2 pairs (reciprocal)
    couplings.set_coupling(2, 1, 0, 0, 0, 1.0);
    couplings.set_coupling(2, 1, -1, 0, 0, 1.0);
    couplings.set_coupling(2, 1, -1, 1, 0, 1.0);
    couplings.set_coupling(2, 1, 0, 1, 0, 1.0);
    
    // Cr4-Cr3 pairs (reciprocal)
    couplings.set_coupling(3, 2, 0, 0, 0, 1.0);
    couplings.set_coupling(3, 2, 0, 0, -1, 1.0);
    
    // Cr4-Cr1 pairs (reciprocal)
    couplings.set_coupling(3, 0, 0, 0, 0, 1.0);
    couplings.set_coupling(3, 0, 0, 1, 0, 1.0);
    couplings.set_coupling(3, 0, -1, 0, 0, 1.0);
    couplings.set_coupling(3, 0, -1, 1, 0, 1.0);
    
    int L = 2;  // 2x2x2 lattice
    double T = 0.01;
    MonteCarloSimulation sim(cell, couplings, L, T);
    
    // Initialize in AF ordered state (Cr1/Cr3 up, Cr2/Cr4 down)
    sim.initialize_lattice_custom({1.0, -1.0, 1.0, -1.0});
    
    double total_energy = sim.get_energy();
    double energy_per_spin = total_energy / (L * L * L * 4);
    
    double expected_energy = -3.0;
    
    std::cout << "  Total energy: " << std::fixed << std::setprecision(6) << total_energy << std::endl;
    std::cout << "  Energy per spin: " << energy_per_spin 
              << " (expected: " << expected_energy << ")" << std::endl;
    
    // Also check sublattice magnetizations (should all be +1)
    std::vector<double> mag_per_spin = sim.get_magnetization_per_spin();
    std::cout << "  Sublattice magnetizations (z-component):" << std::endl;
    std::cout << "    Cr1: " << mag_per_spin[0] << " (expected: +1.0)" << std::endl;
    std::cout << "    Cr2: " << mag_per_spin[1] << " (expected: +1.0)" << std::endl;
    std::cout << "    Cr3: " << mag_per_spin[2] << " (expected: +1.0)" << std::endl;
    std::cout << "    Cr4: " << mag_per_spin[3] << " (expected: +1.0)" << std::endl;
    
    bool energy_correct = approx_equal(energy_per_spin, expected_energy);
    bool mag_correct = approx_equal(mag_per_spin[0], 1.0) && 
                       approx_equal(mag_per_spin[1], 1.0) &&
                       approx_equal(mag_per_spin[2], 1.0) &&
                       approx_equal(mag_per_spin[3], 1.0);
    
    if (energy_correct && mag_correct) {
        std::cout << "  ✓ 4-atom antiferromagnet energy and magnetizations correct!" << std::endl;
        return true;
    } else {
        std::cout << "  ✗ Test failed: ";
        if (!energy_correct) std::cout << "energy mismatch ";
        if (!mag_correct) std::cout << "magnetization mismatch";
        std::cout << std::endl;
        return false;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  MULTI-ATOM CELL ENERGY VALIDATION   " << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 4;
    
    if (test_single_atom_ferromagnet()) passed++;
    if (test_single_atom_antiferromagnet()) passed++;
    if (test_4atom_ferromagnet()) passed++;
    if (test_4atom_ferromagnet()) passed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS: " << passed << "/" << total << " tests passed" << std::endl;
    
    if (passed == total) {
        std::cout << "✓ ALL ENERGY TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ " << (total - passed) << " tests failed" << std::endl;
        return 1;
    }
}
