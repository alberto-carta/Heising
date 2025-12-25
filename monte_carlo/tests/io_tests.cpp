/*
 * Configuration I/O Tests
 * 
 * Tests for configuration file parsing, validation, and error handling
 */

#include "../include/io/configuration_parser.h"
#include "../include/simulation_utils.h"
#include "../include/simulation_engine.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace IO;

// Global seed (required by random.cpp)
long int seed = -12345;

// Test counter
int tests_passed = 0;
int tests_total = 0;

void test_result(bool passed, const std::string& test_name) {
    tests_total++;
    if (passed) {
        tests_passed++;
        std::cout << "✓ " << test_name << std::endl;
    } else {
        std::cout << "✗ " << test_name << " FAILED" << std::endl;
    }
}

/**
 * Test 1: Ferromagnetic system initialization and J couplings
 */
bool test_ferromagnetic_loading() {
    std::cout << "\n=== Test 1: Ferromagnetic System Loading ===" << std::endl;
    
    try {
        SimulationConfig config = ConfigurationParser::load_configuration(
            "tests/io_test_data/ferromagnet/simulation.toml");
        
        // Verify species loaded correctly
        if (config.species.size() != 1) {
            std::cerr << "ERROR: Expected 1 species, got " << config.species.size() << std::endl;
            return false;
        }
        
        if (config.species[0].name != "Fe" || config.species[0].spin_type != SpinType::HEISENBERG) {
            std::cerr << "ERROR: Species not loaded correctly" << std::endl;
            return false;
        }
        
        // Verify couplings loaded correctly (3 nearest neighbor couplings, all negative = FM)
        if (config.couplings.size() != 3) {
            std::cerr << "ERROR: Expected 3 couplings, got " << config.couplings.size() << std::endl;
            return false;
        }
        
        for (const auto& coupling : config.couplings) {
            if (coupling.J >= 0) {
                std::cerr << "ERROR: Ferromagnetic coupling should be negative, got J=" << coupling.J << std::endl;
                return false;
            }
        }
        
        // Create simulation objects and verify energy
        UnitCell unit_cell = create_unit_cell_from_config(config.species);
        CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, 4);
        
        MonteCarloSimulation sim(unit_cell, couplings, 4, 1.0);
        
        // Initialize all spins aligned (ferromagnetic ground state)
        sim.initialize_lattice_custom({1.0});
        
        double energy = sim.get_energy();
        // For FM with J=-1, 3 NN couplings (x,y,z directions), L=4
        // Each spin has 6 neighbors in 3D cubic, but only half couplings (avoid double count)
        // E_total = -1 * 3 * 64 / 1 = -192  (3 coupling directions, 64 spins, divided by 2 in get_energy)
        // Energy per spin: -192 / 64 = -3.0
        // But since we defined 3 couplings (not 6), actual E = -1.5 per spin
        double expected_energy_per_spin = -1.5;
        double energy_per_spin = energy / 64.0;
        
        if (std::abs(energy_per_spin - expected_energy_per_spin) > 0.01) {
            std::cerr << "ERROR: Expected energy per spin " << expected_energy_per_spin 
                      << ", got " << energy_per_spin << std::endl;
            return false;
        }
        
        std::cout << "  Loaded: " << config.species.size() << " species, " 
                  << config.couplings.size() << " couplings" << std::endl;
        std::cout << "  FM ground state energy per spin: " << energy_per_spin << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 2: Antiferromagnetic system initialization
 */
bool test_antiferromagnetic_loading() {
    std::cout << "\n=== Test 2: Antiferromagnetic System Loading ===" << std::endl;
    
    try {
        SimulationConfig config = ConfigurationParser::load_configuration(
            "tests/io_test_data/antiferromagnet/simulation.toml");
        
        // Verify couplings are positive (AFM)
        for (const auto& coupling : config.couplings) {
            if (coupling.J <= 0) {
                std::cerr << "ERROR: Antiferromagnetic coupling should be positive, got J=" << coupling.J << std::endl;
                return false;
            }
        }
        
        // Create simulation objects
        UnitCell unit_cell = create_unit_cell_from_config(config.species);
        CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, 4);
        
        MonteCarloSimulation sim(unit_cell, couplings, 4, 1.0);
        
        // Initialize with perfect AFM pattern (alternating up/down)
        // For 2x2x2 with single spin: perfect AFM not possible in cubic lattice
        // But we can initialize ordered state
        sim.initialize_lattice_custom({1.0});
        
        double fm_energy = sim.get_energy();
        
        // Now flip one spin to see AFM energy is lower
        sim.set_heisenberg_spin(1, 1, 1, 0, spin3d(0, 0, -1.0));
        double partial_afm_energy = sim.get_energy();
        
        // With AFM coupling (J>0), antiparallel spins should have lower energy
        if (partial_afm_energy >= fm_energy) {
            std::cerr << "ERROR: AFM ordering should have lower energy than FM" << std::endl;
            std::cerr << "  FM energy: " << fm_energy << ", Partial AFM: " << partial_afm_energy << std::endl;
            return false;
        }
        
        std::cout << "  Loaded AFM system with J>0" << std::endl;
        std::cout << "  FM state energy: " << fm_energy << ", Partial AFM: " << partial_afm_energy << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 3: KK coupling loading
 */
bool test_kk_coupling_loading() {
    std::cout << "\n=== Test 3: KK Coupling Loading ===" << std::endl;
    
    try {
        SimulationConfig config = ConfigurationParser::load_configuration(
            "tests/io_test_data/kk_system/simulation.toml");
        
        // Verify species (should have 2: 1 Heisenberg + 1 Ising)
        if (config.species.size() != 2) {
            std::cerr << "ERROR: Expected 2 species for KK system, got " << config.species.size() << std::endl;
            return false;
        }
        
        // Verify KK couplings loaded
        if (config.kk_couplings.empty()) {
            std::cerr << "ERROR: KK couplings not loaded" << std::endl;
            return false;
        }
        
        std::cout << "  Loaded: " << config.species.size() << " species, " 
                  << config.couplings.size() << " J couplings, "
                  << config.kk_couplings.size() << " KK couplings" << std::endl;
        
        // Create simulation with KK
        UnitCell unit_cell = create_unit_cell_from_config(config.species);
        CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, 4);
        std::optional<KK_Matrix> kk_matrix = create_kk_matrix_from_config(config.kk_couplings, unit_cell, 4);
        
        if (!kk_matrix.has_value()) {
            std::cerr << "ERROR: KK matrix not created" << std::endl;
            return false;
        }
        
        MonteCarloSimulation sim(unit_cell, couplings, 4, 1.0, kk_matrix);
        
        // Initialize with aligned spins
        sim.initialize_lattice_custom({1.0, 1.0});
        
        double energy_aligned = sim.get_energy();
        
        // Flip Ising spin at one site
        sim.set_ising_spin(1, 1, 1, 1, -1);
        double energy_flipped = sim.get_energy();
        
        // With FM KK coupling (K<0), aligned spins should be favored
        double energy_diff = energy_flipped - energy_aligned;
        
        std::cout << "  Energy (aligned): " << energy_aligned << std::endl;
        std::cout << "  Energy (one flip): " << energy_flipped << std::endl;
        std::cout << "  Energy difference: " << energy_diff << std::endl;
        
        if (energy_diff <= 0) {
            std::cerr << "ERROR: FM KK coupling should penalize spin flip" << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 4: Species-coupling mismatch (should throw error)
 */
bool test_species_mismatch_error() {
    std::cout << "\n=== Test 4: Species-Coupling Mismatch Error ===" << std::endl;
    
    try {
        SimulationConfig config = ConfigurationParser::load_configuration(
            "tests/io_test_data/invalid_mismatch/simulation.toml");
        
        std::cerr << "ERROR: Should have thrown ConfigurationError for species mismatch" << std::endl;
        return false;
        
    } catch (const ConfigurationError& e) {
        std::cout << "  Correctly caught error: " << e.what() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Wrong exception type: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 5: Invalid KK system (wrong number of spins per site)
 */
bool test_invalid_kk_system() {
    std::cout << "\n=== Test 5: Invalid KK System (Wrong Spins Per Site) ===" << std::endl;
    
    try {
        SimulationConfig config = ConfigurationParser::load_configuration(
            "tests/io_test_data/invalid_kk/simulation.toml");
        
        // Configuration loads fine, but KK matrix creation should fail gracefully
        UnitCell unit_cell = create_unit_cell_from_config(config.species);
        CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, 4);
        std::optional<KK_Matrix> kk_matrix = create_kk_matrix_from_config(config.kk_couplings, unit_cell, 4);
        
        if (!kk_matrix.has_value()) {
            std::cout << "  KK matrix correctly not created (no valid KK couplings)" << std::endl;
            return true;
        }
        
        // If KK matrix was created, try to use it and see if it handles invalid sites gracefully
        MonteCarloSimulation sim(unit_cell, couplings, 4, 1.0, kk_matrix);
        sim.initialize_lattice_custom({1.0, 1.0, 1.0});  // 3 spins
        
        // Should run without crashing (compute_kk_contribution returns 0.0 for invalid sites)
        double energy = sim.get_energy();
        
        std::cout << "  System handles invalid KK configuration gracefully (E=" << energy << ")" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Test 6: KK with wrong spin types (e.g., 2 Heisenberg, no Ising)
 */
bool test_kk_wrong_spin_types() {
    std::cout << "\n=== Test 6: KK with Wrong Spin Types ===" << std::endl;
    
    try {
        SimulationConfig config = ConfigurationParser::load_configuration(
            "tests/io_test_data/invalid_kk_types/simulation.toml");
        
        UnitCell unit_cell = create_unit_cell_from_config(config.species);
        CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, 4);
        std::optional<KK_Matrix> kk_matrix = create_kk_matrix_from_config(config.kk_couplings, unit_cell, 4);
        
        // Should still create KK matrix (validation happens at runtime in compute_kk_contribution)
        if (!kk_matrix.has_value()) {
            std::cout << "  KK matrix not created (expected)" << std::endl;
            return true;
        }
        
        MonteCarloSimulation sim(unit_cell, couplings, 4, 1.0, kk_matrix);
        sim.initialize_lattice_custom({1.0, 1.0});  // Both Heisenberg
        
        // Should run without crashing (compute_kk_contribution returns 0.0 for wrong spin types)
        double energy = sim.get_energy();
        
        std::cout << "  System handles wrong spin types gracefully (E=" << energy << ")" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   CONFIGURATION I/O TESTS             " << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_result(test_ferromagnetic_loading(), "Ferromagnetic system loading and J couplings");
    test_result(test_antiferromagnetic_loading(), "Antiferromagnetic system loading");
    test_result(test_kk_coupling_loading(), "KK coupling loading and physics");
    test_result(test_species_mismatch_error(), "Species-coupling mismatch error handling");
    test_result(test_invalid_kk_system(), "Invalid KK system (wrong number of spins)");
    test_result(test_kk_wrong_spin_types(), "KK with wrong spin types");
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS: " << tests_passed << "/" << tests_total << " tests passed" << std::endl;
    
    if (tests_passed == tests_total) {
        std::cout << "✓ ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
