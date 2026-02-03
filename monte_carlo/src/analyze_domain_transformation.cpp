/*
 * Slab Flip Analyzer - Domain Transformation Energy Analysis
 * 
 * Simple tool that reads species/couplings/KK from files and tests energy
 * differences between two spin patterns at various slab sizes.
 * 
 * Usage: ./slab_flip_analyzer.x <config.toml>
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h"
#include "../include/simulation_utils.h"
#include "../include/io/configuration_parser.h"
#include <toml.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

// Global random seed
long int seed = -12345;

// Simple parsers for data files
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

void apply_pattern(MonteCarloSimulation& sim, const std::vector<double>& pattern,
                  int x_start, int x_end, int y_start, int y_end, int z_start, int z_end) {
    // Pattern format: [Cr1_z, Cr2_z, Cr3_z, Cr4_z, CrA_tau, CrB_tau, CrC_tau, CrD_tau]
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            for (int z = z_start; z <= z_end; z++) {
                for (int site = 0; site < 4; site++) {
                    spin3d s = sim.get_heisenberg_spin(x, y, z, site);
                    s.z = pattern[site];
                    sim.set_heisenberg_spin(x, y, z, site, s);
                }
                for (int site = 4; site < 8; site++) {
                    sim.set_ising_spin(x, y, z, site, static_cast<int>(pattern[site]));
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config.toml>" << std::endl;
        return 1;
    }
    
    std::string config_file = argv[1];
    std::filesystem::path config_path(config_file);
    std::filesystem::path config_dir = config_path.parent_path();
    
    // Parse config
    auto data = toml::parse(config_file);
    int L = toml::find<int>(data, "lattice_size");
    std::vector<double> pattern1 = toml::find<std::vector<double>>(data, "pattern1");
    std::vector<double> pattern2 = toml::find<std::vector<double>>(data, "pattern2");
    std::vector<int> lateral_sizes = toml::find<std::vector<int>>(data, "lateral_sizes");
    
    // Get file paths
    std::string species_file = (config_dir / toml::find<std::string>(data, "species_file")).string();
    std::string couplings_file = (config_dir / toml::find<std::string>(data, "couplings_file")).string();
    std::string kk_file = (config_dir / toml::find<std::string>(data, "kugel_khomskii_file")).string();
    
    // Parse data files
    auto species = parse_species(species_file);
    auto couplings = parse_couplings(couplings_file);
    auto kk_couplings = parse_kk(kk_file);
    
    // Build simulation
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
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  SLAB FLIP ANALYZER" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\n  Lattice: " << L << "×" << L << "×" << L << " cells" << std::endl;
    
    // Pattern 1 energy
    sim.initialize_lattice_custom(pattern1);
    double E1 = sim.get_energy();
    std::cout << "\n  Pattern 1 energy: " << std::fixed << std::setprecision(1) << E1 << std::endl;
    
    // Pattern 2 energy  
    sim.initialize_lattice_custom(pattern2);
    double E2 = sim.get_energy();
    double delta_total = E2 - E1;
    std::cout << "  Pattern 2 energy: " << std::fixed << std::setprecision(1) << E2 << std::endl;
    std::cout << "  ΔE (2-1):         " << std::fixed << std::setprecision(1) << delta_total;
    if (delta_total < 0) std::cout << "  ✓ Pattern 2 LOWER" << std::endl;
    else if (delta_total > 0) std::cout << "  ✗ Pattern 2 HIGHER" << std::endl;
    else std::cout << "  = DEGENERATE" << std::endl;
    
    // Single layer flip
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "  SINGLE LAYER TRANSFORMATION (z=1)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    sim.initialize_lattice_custom(pattern1);
    double E_before_layer = sim.get_energy();
    apply_pattern(sim, pattern2, 1, L, 1, L, 1, 1);
    double E_after_layer = sim.get_energy();
    double delta_layer = E_after_layer - E_before_layer;
    
    std::cout << "  ΔE = " << std::fixed << std::setprecision(1) << delta_layer;
    if (delta_layer < 0) std::cout << "  ✓ LOWERS ENERGY" << std::endl;
    else std::cout << "  ✗ RAISES ENERGY" << std::endl;
    
    // Critical size search
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  CRITICAL SIZE SEARCH: [size×size×1] slabs at z=1" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\n  Size | Area | ΔE      | ΔE/Area | Status" << std::endl;
    std::cout << "  -----|------|---------|---------|------------------" << std::endl;
    
    int critical = -1;
    for (int size : lateral_sizes) {
        if (size > L) break;
        
        sim.initialize_lattice_custom(pattern1);
        double E_before = sim.get_energy();
        
        int x_start = (L - size) / 2 + 1;
        int y_start = (L - size) / 2 + 1;
        apply_pattern(sim, pattern2, x_start, x_start + size - 1, y_start, y_start + size - 1, 1, 1);
        
        double E_after = sim.get_energy();
        double delta = E_after - E_before;
        int area = size * size;
        
        std::string status = (delta < 0) ? "LOWERS ✓" : "RAISES ✗";
        if (delta < 0 && critical == -1) critical = size;
        
        std::cout << "  " << std::setw(4) << size 
                  << " | " << std::setw(4) << area
                  << " | " << std::setw(7) << std::fixed << std::setprecision(1) << delta
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << delta/area
                  << " | " << status << std::endl;
    }
    
    if (critical > 0) {
        std::cout << "\n  ✓ Critical size: " << critical << "×" << critical << " cells" << std::endl;
    } else {
        std::cout << "\n  ✗ No favorable size found" << std::endl;
    }
    
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
