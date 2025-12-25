/*
 * Fast Multi-Spin Data Structures
 * 
 * Simplified, performance-focused implementation for multi-spin Monte Carlo
 * Uses simple arrays and direct indexing instead of complex containers
 */

#ifndef MULTI_SPIN_H
#define MULTI_SPIN_H

#include "spin_types.h"
#include <vector>
#include <string>
#include <iostream>

// Simple spin information
struct SpinInfo {
    SpinType spin_type;
    double spin_magnitude;
    std::string label;
    int site_id;      // Which physical site this spin belongs to
    double x, y, z;   // Position within unit cell (fractional coordinates)
    
    SpinInfo() : spin_type(SpinType::ISING), spin_magnitude(1.0), label(""), 
                 site_id(0), x(0.0), y(0.0), z(0.0) {} // Constructor for dummy objects
    SpinInfo(const std::string& lbl, SpinType type, double mag, int site, 
             double px, double py, double pz) 
        : spin_type(type), spin_magnitude(mag), label(lbl), site_id(site),
          x(px), y(py), z(pz) {} // Proper constructor
};

// Structure to represent a physical site with its position
struct Site {
    double x, y, z;  // Position in unit cell
    std::vector<int> spin_indices;  // Indices of spins at this site
    
    Site(double px, double py, double pz) : x(px), y(py), z(pz) {}
    
    // Check if this position matches (with tolerance for floating point)
    bool matches_position(double px, double py, double pz, double tolerance = 1e-6) const {
        return std::abs(x - px) < tolerance && 
               std::abs(y - py) < tolerance && 
               std::abs(z - pz) < tolerance;
    }
};

// Unit cell - container for spins with position-based site management
class UnitCell {
private:
    std::vector<SpinInfo> spins;
    std::vector<Site> sites;  // List of physical sites
    
    // Find site at given position, return -1 if not found
    int find_site_at_position(double x, double y, double z) const {
        for (size_t i = 0; i < sites.size(); i++) {
            if (sites[i].matches_position(x, y, z)) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    // Given a spin index, return its site ID
    int get_site_id_of_spin(int spin_index) const {
        if (spin_index < 0 || spin_index >= static_cast<int>(spins.size())) {
            return -1;
        }
        return spins[spin_index].site_id;
    }
    
public:
    UnitCell() {}
    
    // Add spin with position - automatically assigns to correct site
    void add_spin(const std::string& label, SpinType type, double magnitude,
                  double x = 0.0, double y = 0.0, double z = 0.0) {
        // Check if site already exists at this position
        int site_id = find_site_at_position(x, y, z);
        
        if (site_id == -1) {
            // Create new site at this position
            site_id = static_cast<int>(sites.size());
            sites.emplace_back(x, y, z);
        }
        
        // Add spin index to site
        sites[site_id].spin_indices.push_back(static_cast<int>(spins.size()));
        
        // Create and store spin
        spins.emplace_back(label, type, magnitude, site_id, x, y, z);
    }
    
    int num_spins() const { return static_cast<int>(spins.size()); }
    int get_num_sites() const { return static_cast<int>(sites.size()); }
    const SpinInfo& get_spin(int id) const { return spins[id]; }
    const Site& get_site(int site_id) const { return sites[site_id]; }
    
    // Get all spins belonging to a given site
    std::vector<int> get_spins_at_site(int site_id) const {
        if (site_id < 0 || site_id >= static_cast<int>(sites.size())) {
            return std::vector<int>();
        }
        return sites[site_id].spin_indices;
    }

    
    // Check if a site has mixed spin types
    bool site_has_mixed_types(int site_id) const {
        std::vector<int> site_spins = get_spins_at_site(site_id);
        if (site_spins.size() <= 1) return false;
        
        SpinType first_type = spins[site_spins[0]].spin_type;
        for (size_t i = 1; i < site_spins.size(); i++) {
            if (spins[site_spins[i]].spin_type != first_type) return true;
        }
        return false;
    }
    
    bool has_mixed_spin_types() const {
        if (spins.empty()) return false;
        SpinType first_type = spins[0].spin_type;
        for (size_t i = 1; i < spins.size(); i++) {
            if (spins[i].spin_type != first_type) return true;
        }
        return false;
    }
};

// Coupling matrix - dynamic 5D array: [spin_i][spin_j][dx+max_offset][dy+max_offset][dz+max_offset]
class CouplingMatrix {
private:
    int num_spins;
    int max_offset;        // Dynamically determined maximum offset
    int offset_size;       // 2 * max_offset + 1
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> J;  // J[i][j][dx][dy][dz]
    
    // Convert offset to array index
    int offset_to_index(int offset) const {
        return offset + max_offset;
    }
    
public:
    CouplingMatrix() : num_spins(0), max_offset(0), offset_size(1) {}
    
    // Initialize with specified maximum offset range
    void initialize(int n_spins, int max_coupling_offset = 1) {
        num_spins = n_spins;
        max_offset = max_coupling_offset;
        offset_size = 2 * max_offset + 1;
        
        std::cout << "Initializing coupling matrix: " << n_spins << " spins, max_offset = " 
                  << max_offset << " (array size: " << offset_size << "³)" << std::endl;
        
        // Resize to [n_spins][n_spins][offset_size][offset_size][offset_size] and initialize to 0.0
        J.assign(n_spins,                                                                    // [spin_i]
            std::vector<std::vector<std::vector<std::vector<double>>>>(n_spins,             // [spin_j]
                std::vector<std::vector<std::vector<double>>>(offset_size,                  // [dx]
                    std::vector<std::vector<double>>(offset_size,                           // [dy]
                        std::vector<double>(offset_size, 0.0)))));                          // [dz]
        
        size_t total_elements = (size_t)n_spins * n_spins * offset_size * offset_size * offset_size;
        size_t memory_mb = total_elements * sizeof(double) / (1024 * 1024);
        std::cout << "Coupling matrix memory: " << memory_mb << " MB (" << total_elements << " elements)" << std::endl;
    }
    
    // Set coupling value - direct array access
    void set_coupling(int spin_i, int spin_j, int dx, int dy, int dz, double coupling_value) {
        if (std::abs(dx) > max_offset || std::abs(dy) > max_offset || std::abs(dz) > max_offset) {
            std::cerr << "Error: Coupling offset (" << dx << "," << dy << "," << dz 
                      << ") exceeds max_offset = " << max_offset << std::endl;
            return;
        }
        
        int idx_x = offset_to_index(dx);
        int idx_y = offset_to_index(dy);
        int idx_z = offset_to_index(dz);
        
        J[spin_i][spin_j][idx_x][idx_y][idx_z] = coupling_value;
    }
    
    // Get coupling value - direct array access
    double get_coupling(int spin_i, int spin_j, int dx, int dy, int dz) const {
        if (std::abs(dx) > max_offset || std::abs(dy) > max_offset || std::abs(dz) > max_offset) {
            return 0.0;  // No coupling beyond max range
        }
        
        int idx_x = offset_to_index(dx);
        int idx_y = offset_to_index(dy);
        int idx_z = offset_to_index(dz);
        
        return J[spin_i][spin_j][idx_x][idx_y][idx_z];
    }
    
    // Convenience method for symmetric nearest-neighbor couplings
    void set_nn_couplings(int spin_i, int spin_j, double coupling_value) {
        // 6 nearest neighbors
        set_coupling(spin_i, spin_j,  1, 0, 0, coupling_value);  // +x
        set_coupling(spin_i, spin_j, -1, 0, 0, coupling_value);  // -x
        set_coupling(spin_i, spin_j,  0, 1, 0, coupling_value);  // +y
        set_coupling(spin_i, spin_j,  0,-1, 0, coupling_value);  // -y
        set_coupling(spin_i, spin_j,  0, 0, 1, coupling_value);  // +z
        set_coupling(spin_i, spin_j,  0, 0,-1, coupling_value);  // -z
    }
    
    // Set intra-cell coupling (same cell, dx=dy=dz=0)
    void set_intra_coupling(int spin_i, int spin_j, double coupling_value) {
        set_coupling(spin_i, spin_j, 0, 0, 0, coupling_value);
    }
    
    int get_num_spins() const { return num_spins; }
    int get_max_offset() const { return max_offset; }
    
    // Print coupling matrix for debugging
    void print_summary() const {
        std::cout << "Coupling matrix summary:" << std::endl;
        std::cout << "  Spins: " << num_spins << std::endl;
        std::cout << "  Max offset range: ±" << max_offset << std::endl;
        
        // Count non-zero couplings
        int non_zero = 0;
        for (int i = 0; i < num_spins; i++) {
            for (int j = 0; j < num_spins; j++) {
                for (int dx = -max_offset; dx <= max_offset; dx++) {
                    for (int dy = -max_offset; dy <= max_offset; dy++) {
                        for (int dz = -max_offset; dz <= max_offset; dz++) {
                            if (get_coupling(i, j, dx, dy, dz) != 0.0) {
                                non_zero++;
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  Non-zero couplings: " << non_zero << std::endl;
    }
};

// Helper functions for creating simple systems
inline UnitCell create_unit_cell(SpinType model_type) {
    UnitCell cell;
    cell.add_spin("Spin1", model_type, 1.0, 0.0, 0.0, 0.0);  // At origin
    return cell;
}

inline CouplingMatrix create_nn_couplings(int num_spins, double J, int max_range = 1) {
    CouplingMatrix couplings;
    couplings.initialize(num_spins, max_range);  // Only allocate what we need
    
    // For single spin, add nearest neighbor couplings
    if (num_spins == 1) {
        couplings.set_nn_couplings(0, 0, J);
    }
    
    return couplings;
}

// 5D matrix containing Kugel-Khomskii couplings using SITE indices
// K[site_i][site_j][dx][dy][dz] represents coupling between sites, not individual spins
class KK_Matrix {
private:
    int num_sites;         // Number of physical sites in unit cell
    int max_offset;        
    int offset_size;       
    UnitCell unit_cell;    
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> K;  // K[site_i][site_j][dx][dy][dz]
    
    // Convert offset to array index
    int offset_to_index(int offset) const {
        return offset + max_offset;
    }
    

public:
    KK_Matrix() : num_sites(0), max_offset(0), offset_size(1) {}
    
    // Initialize with specified maximum offset range
    void initialize(UnitCell given_unit_cell, int max_coupling_offset = 1) { 
        unit_cell = given_unit_cell;
        num_sites = unit_cell.get_num_sites();
        max_offset = max_coupling_offset;
        offset_size = 2 * max_offset + 1;   
        
        std::cout << "Initializing KK matrix: " << num_sites << " sites, max_offset = " 
                  << max_offset << " (array size: " << offset_size << "³)" << std::endl;

        // Verify that we have sites with mixed spin types
        bool has_mixed_site = false;
        for (int site = 0; site < num_sites; site++) {
            if (unit_cell.site_has_mixed_types(site)) {
                has_mixed_site = true;
                std::cout << "  Site " << site << " has mixed spin types (KK-capable)" << std::endl;
                break;
            }
        }
        
        if (!has_mixed_site) {
            std::cerr << "Warning: No site has mixed spin types. KK coupling requires both spin types at each site." << std::endl;
        }

        // Resize to [num_sites][num_sites][offset_size][offset_size][offset_size]
        K.assign(num_sites,                                                                  // [site_i]
            std::vector<std::vector<std::vector<std::vector<double>>>>(num_sites,           // [site_j]
                std::vector<std::vector<std::vector<double>>>(offset_size,                  // [dx]
                    std::vector<std::vector<double>>(offset_size,                           // [dy]
                        std::vector<double>(offset_size, 0.0)))));                          // [dz]
        
        size_t total_elements = (size_t)num_sites * num_sites * offset_size * offset_size * offset_size;
        size_t memory_mb = total_elements * sizeof(double) / (1024 * 1024);
        std::cout << "KK matrix memory: " << memory_mb << " MB (" << total_elements << " elements)" << std::endl;
    }

    // Set coupling value between two SITES
    void set_coupling(int site_i, int site_j, int dx, int dy, int dz, double coupling_value) {
        // Verify both sites have mixed spin types
        if (!unit_cell.site_has_mixed_types(site_i) || !unit_cell.site_has_mixed_types(site_j)) {
            std::cerr << "Error: KK coupling requires both sites to have mixed spin types. "
                      << "Site " << site_i << " and site " << site_j << " must each have both Ising and Heisenberg spins." << std::endl;
            return;
        }

        if (std::abs(dx) > max_offset || std::abs(dy) > max_offset || std::abs(dz) > max_offset) {
            std::cerr << "Error: Coupling offset (" << dx << "," << dy << "," << dz 
                      << ") exceeds max_offset = " << max_offset << std::endl;
            return;
        }
        
        int idx_x = offset_to_index(dx);
        int idx_y = offset_to_index(dy);
        int idx_z = offset_to_index(dz);
        
        K[site_i][site_j][idx_x][idx_y][idx_z] = coupling_value;
    }
    
    // Get coupling value between two SITES
    double get_coupling(int site_i, int site_j, int dx, int dy, int dz) const {
        if (std::abs(dx) > max_offset || std::abs(dy) > max_offset || std::abs(dz) > max_offset) {
            return 0.0;
        }
        
        int idx_x = offset_to_index(dx);
        int idx_y = offset_to_index(dy);
        int idx_z = offset_to_index(dz);
        
        return K[site_i][site_j][idx_x][idx_y][idx_z];
    }
    
    int get_num_sites() const { return num_sites; }

    int get_max_offset() const { return max_offset; }

    void print_summary() const {
        std::cout << "KK Coupling matrix summary:" << std::endl;
        std::cout << "  Sites: " << num_sites << std::endl;
        std::cout << "  Max offset range: ±" << max_offset << std::endl;
        
        // Count non-zero couplings
        int non_zero = 0;
        for (int i = 0; i < num_sites; i++) {
            for (int j = 0; j < num_sites; j++) {
                for (int dx = -max_offset; dx <= max_offset; dx++) {
                    for (int dy = -max_offset; dy <= max_offset; dy++) {
                        for (int dz = -max_offset; dz <= max_offset; dz++) {
                            if (get_coupling(i, j, dx, dy, dz) != 0.0) {
                                non_zero++;
                            }
                        }
                    }
                }
            }
        }
        std::cout << "  Non-zero KK couplings: " << non_zero << std::endl;
    }
};


#endif // MULTI_SPIN_H