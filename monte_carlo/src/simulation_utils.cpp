/*
 * Simulation Utilities Implementation
 */

#include "../include/simulation_utils.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <algorithm>

UnitCell create_unit_cell_from_config(const std::vector<IO::MagneticSpecies>& species) {
    UnitCell cell;
    
    for (const auto& spec : species) {
        // Use position from config to automatically determine sites
        cell.add_spin(spec.name, spec.spin_type, 1.0, 
                     spec.local_pos[0], spec.local_pos[1], spec.local_pos[2]);
    }
    
    return cell;
}

CouplingMatrix create_couplings_from_config(const std::vector<IO::ExchangeCoupling>& couplings,
                                            const std::vector<IO::MagneticSpecies>& species,
                                            int lattice_size) {
    int num_spins = species.size();
    
    // Determine maximum coupling offset from input
    int max_offset = 1;  // Default to nearest neighbor
    for (const auto& coupling : couplings) {
        int max_abs = std::max({std::abs(coupling.cell_offset[0]),
                                std::abs(coupling.cell_offset[1]),
                                std::abs(coupling.cell_offset[2])});
        max_offset = std::max(max_offset, max_abs);
    }
    
    CouplingMatrix coupling_matrix;
    coupling_matrix.initialize(num_spins, max_offset);
    
    // Create map of species names to spin indices
    std::map<std::string, int> species_to_index;
    for (size_t i = 0; i < species.size(); i++) {
        species_to_index[species[i].name] = static_cast<int>(i);
    }
    
    // Add all couplings with range checking
    int truncated_couplings = 0;
    for (const auto& coupling : couplings) {
        int atom1_id = species_to_index[coupling.species1_name];
        int atom2_id = species_to_index[coupling.species2_name];
        
        // Check if coupling extends beyond reasonable simulation range
        int max_abs = std::max({std::abs(coupling.cell_offset[0]),
                                std::abs(coupling.cell_offset[1]),
                                std::abs(coupling.cell_offset[2])});
        
        if (max_abs >= lattice_size / 2) {
            std::cout << "WARNING: Coupling " << coupling.species1_name << "-" << coupling.species2_name
                      << " at offset (" << coupling.cell_offset[0] << "," << coupling.cell_offset[1] 
                      << "," << coupling.cell_offset[2] << ") extends beyond half lattice size ("
                      << lattice_size/2 << "). Truncating." << std::endl;
            truncated_couplings++;
            continue;
        }
        
        // Add coupling for specified unit cell offset
        coupling_matrix.set_coupling(atom1_id, atom2_id, 
                                     coupling.cell_offset[0],
                                     coupling.cell_offset[1], 
                                     coupling.cell_offset[2],
                                     coupling.J);
    }
    
    if (truncated_couplings > 0) {
        std::cout << "WARNING: " << truncated_couplings << " couplings were truncated due to range limits." << std::endl;
    }
    
    return coupling_matrix;
}

std::optional<KK_Matrix> create_kk_matrix_from_config(const std::vector<IO::KKCoupling>& kk_couplings,
                                                        const UnitCell& unit_cell,
                                                        int lattice_size) {
    // Return nullopt if no KK couplings
    if (kk_couplings.empty()) {
        return std::nullopt;
    }
    
    int num_sites = unit_cell.get_num_sites();
    
    // Determine maximum KK coupling offset
    int max_offset = 1;
    for (const auto& kk : kk_couplings) {
        int max_abs = std::max({std::abs(kk.cell_offset[0]),
                                std::abs(kk.cell_offset[1]),
                                std::abs(kk.cell_offset[2])});
        max_offset = std::max(max_offset, max_abs);
    }
    
    KK_Matrix kk_matrix;
    kk_matrix.initialize(unit_cell, max_offset);
    
    // For each KK coupling, we need to determine which site each species belongs to
    // Assumption: Species at same position belong to the same site
    std::map<std::string, int> species_to_site;
    for (int spin_id = 0; spin_id < unit_cell.num_spins(); spin_id++) {
        const SpinInfo& spin = unit_cell.get_spin(spin_id);
        species_to_site[spin.label] = spin.site_id;
    }
    
    // Add all KK couplings with range checking
    int truncated_kk = 0;
    for (const auto& kk : kk_couplings) {
        // Get site IDs for the two species
        auto it1 = species_to_site.find(kk.species1_name);
        auto it2 = species_to_site.find(kk.species2_name);
        
        if (it1 == species_to_site.end()) {
            std::cerr << "ERROR: Species '" << kk.species1_name << "' in KK coupling not found" << std::endl;
            continue;
        }
        if (it2 == species_to_site.end()) {
            std::cerr << "ERROR: Species '" << kk.species2_name << "' in KK coupling not found" << std::endl;
            continue;
        }
        
        int site1_id = it1->second;
        int site2_id = it2->second;
        
        // Check if coupling extends beyond reasonable simulation range
        int max_abs = std::max({std::abs(kk.cell_offset[0]),
                                std::abs(kk.cell_offset[1]),
                                std::abs(kk.cell_offset[2])});
        
        if (max_abs >= lattice_size / 2) {
            std::cout << "WARNING: KK coupling " << kk.species1_name << "-" << kk.species2_name
                      << " at offset (" << kk.cell_offset[0] << "," << kk.cell_offset[1] 
                      << "," << kk.cell_offset[2] << ") extends beyond half lattice size ("
                      << lattice_size/2 << "). Truncating." << std::endl;
            truncated_kk++;
            continue;
        }
        
        // Add KK coupling for specified unit cell offset
        try {
            kk_matrix.set_coupling(site1_id, site2_id,
                                  kk.cell_offset[0],
                                  kk.cell_offset[1],
                                  kk.cell_offset[2],
                                  kk.K);
        } catch (const std::runtime_error& e) {
            // Skip invalid KK couplings (e.g., sites without mixed spin types)
            std::cerr << "WARNING: Skipping invalid KK coupling: " << e.what() << std::endl;
            continue;
        }
    }
    
    if (truncated_kk > 0) {
        std::cout << "WARNING: " << truncated_kk << " KK couplings were truncated due to range limits." << std::endl;
    }
    
    return kk_matrix;
}

void average_configuration_mpi(MonteCarloSimulation& sim,
                               const std::vector<IO::MagneticSpecies>& species,
                               MPIAccumulator& mpi_accumulator) {
    // Get mutable references to spin arrays
    auto& ising_array = sim.get_ising_array_mutable();
    auto& heis_x_array = sim.get_heisenberg_x_array_mutable();
    auto& heis_y_array = sim.get_heisenberg_y_array_mutable();
    auto& heis_z_array = sim.get_heisenberg_z_array_mutable();
    
    // Convert Eigen arrays to std::vector for MPI operations
    std::vector<double> ising_vec(ising_array.data(), ising_array.data() + ising_array.size());
    std::vector<double> heis_x_vec(heis_x_array.data(), heis_x_array.data() + heis_x_array.size());
    std::vector<double> heis_y_vec(heis_y_array.data(), heis_y_array.data() + heis_y_array.size());
    std::vector<double> heis_z_vec(heis_z_array.data(), heis_z_array.data() + heis_z_array.size());
    
    // Average across all ranks
    mpi_accumulator.average_configuration(ising_vec);
    mpi_accumulator.average_configuration_vectors(heis_x_vec, heis_y_vec, heis_z_vec);
    
    // Copy back to Eigen arrays
    for (size_t i = 0; i < ising_vec.size(); i++) {
        ising_array[i] = ising_vec[i];
    }
    for (size_t i = 0; i < heis_x_vec.size(); i++) {
        heis_x_array[i] = heis_x_vec[i];
        heis_y_array[i] = heis_y_vec[i];
        heis_z_array[i] = heis_z_vec[i];
    }
}
