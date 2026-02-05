/*
 * Test Slab Tunnel Move
 * 
 * Verifies that slab tunnel move produces identical energy changes
 * to those computed by the slab flip analyzer
 */

#include "../include/simulation_engine.h"
#include "../include/simulation_utils.h"
#include "../include/mc_moves.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

// Global random seed
long int seed = -12345;

// Simple file parsers (same as slab analyzer)
std::vector<IO::MagneticSpecies> parse_species(const std::string& filename) {
    std::vector<IO::MagneticSpecies> species;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string name, type;
        double x, y, z;
        if (iss >> name >> type >> x >> y >> z) {
            SpinType spin_type = (type == "Heisenberg") ? SpinType::HEISENBERG : SpinType::ISING;
            species.emplace_back(name, spin_type, x, y, z);
        }
    }
    return species;
}

std::vector<IO::ExchangeCoupling> parse_couplings(const std::string& filename) {
    std::vector<IO::ExchangeCoupling> couplings;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string s1, s2;
        int rx, ry, rz;
        double J;
        if (iss >> s1 >> s2 >> rx >> ry >> rz >> J) {
            couplings.emplace_back(s1, s2, rx, ry, rz, J);
        }
    }
    return couplings;
}

std::vector<IO::KKCoupling> parse_kk(const std::string& filename) {
    std::vector<IO::KKCoupling> kk_couplings;
    std::ifstream file(filename);
    if (!file.good()) return kk_couplings;
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string s1, s2;
        int rx, ry, rz;
        double K;
        if (iss >> s1 >> s2 >> rx >> ry >> rz >> K) {
            kk_couplings.emplace_back(s1, s2, rx, ry, rz, K);
        }
    }
    return kk_couplings;
}

int main() {
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "  SLAB TUNNEL MOVE TEST" << std::endl;
    std::cout << "======================================================================\n" << std::endl;
    
    // Parse data files (same as analyzer)
    std::string base_path = "../examples/4-atom_cell_kk_anisotropic/";
    
    auto species = parse_species(base_path + "species.dat");
    auto couplings = parse_couplings(base_path + "generated_couplings.dat");
    auto kk_couplings = parse_kk(base_path + "kugel_khomskii.dat");
    
    // Build simulation (L=16 to match analyzer)
    int L = 16;
    UnitCell cell = create_unit_cell_from_config(species);
    CouplingMatrix coupling_matrix = create_couplings_from_config(couplings, species, L);
    auto kk_opt = create_kk_matrix_from_config(kk_couplings, cell, L);
    KK_Matrix kk_matrix;
    if (kk_opt.has_value()) {
        kk_matrix = kk_opt.value();
    } else {
        int max_offset = 1;
        for (const auto& k : kk_couplings) {
            max_offset = std::max(max_offset, std::abs(k.cell_offset[0]));
            max_offset = std::max(max_offset, std::abs(k.cell_offset[1]));
            max_offset = std::max(max_offset, std::abs(k.cell_offset[2]));
        }
        kk_matrix.initialize(cell, max_offset);
    }
    
    MonteCarloSimulation sim(cell, coupling_matrix, L, 0.0, kk_matrix);
    
    // Define patterns (same as analyzer)
    std::vector<double> pattern1 = {1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0};  // C-AFM + G-OO
    std::vector<double> pattern2 = {1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0};  // G-AFM + C-OO
    
    // Set slab tunnel parameters (burst mode not used in tests)
    sim.set_slab_tunnel_parameters(pattern1, pattern2, 1, 1, 100, 50);  // size 1x1x1, burst: interval=100, attempts=50
    
    // Initialize with pattern1
    sim.initialize_lattice_custom(pattern1);
    
    std::cout << "Testing various slab sizes (comparing to analyzer output):\n" << std::endl;
    std::cout << "  Size | ΔE (MC Move) | ΔE (Analyzer) | Match?" << std::endl;
    std::cout << "  -----|--------------|----------------|-------" << std::endl;
    
    // Expected values from analyzer output
    std::vector<int> test_sizes = {1, 2, 3, 4, 5, 10, 11, 12, 16};
    std::vector<double> expected_energies = {16.8, 35.2, 48.8, 57.6, 61.6, 9.6, -15.2, -44.8, -614.4};
    
    int passed = 0;
    int total = test_sizes.size();
    
    for (size_t i = 0; i < test_sizes.size(); i++) {
        int size = test_sizes[i];
        double expected = expected_energies[i];
        
        // Update slab size (burst parameters don't affect energy calculation)
        sim.set_slab_tunnel_parameters(pattern1, pattern2, size, 1, 100, 50);
        
        // Calculate energy change for centered slab using move proposer
        int x_center = L/2 - size/2 + 1;
        int y_center = L/2 - size/2 + 1;
        int z_start = 1;
        
        MoveProposal proposal = sim.get_move_proposer()->propose_slab_tunnel(x_center, y_center, z_start);
        double delta_E = proposal.energy_change;
        
        // Check if matches (within tolerance)
        double tolerance = 0.2;
        bool matches = std::abs(delta_E - expected) < tolerance;
        
        if (matches) passed++;
        
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  " << std::setw(4) << size 
                  << " | " << std::setw(12) << delta_E
                  << " | " << std::setw(14) << expected
                  << " | " << (matches ? "✓" : "✗") << std::endl;
    }
    
    std::cout << "\n======================================================================" << std::endl;
    std::cout << "Test Result: " << passed << "/" << total << " tests passed" << std::endl;
    
    if (passed == total) {
        std::cout << "✓ ALL TESTS PASSED - Slab tunnel moves match analyzer!" << std::endl;
        std::cout << "======================================================================\n" << std::endl;
        return 0;
    } else {
        std::cout << "✗ SOME TESTS FAILED" << std::endl;
        std::cout << "======================================================================\n" << std::endl;
        return 1;
    }
}
