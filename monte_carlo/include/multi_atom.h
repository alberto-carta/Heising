/*
 * Fast Multi-Atom Data Structures
 * 
 * Simplified, performance-focused implementation for multi-atom Monte Carlo
 * Uses simple arrays and direct indexing instead of complex containers
 */

#ifndef MULTI_ATOM_H
#define MULTI_ATOM_H

#include "spin_types.h"
#include <vector>
#include <string>
#include <iostream>

// Simple atom information
struct AtomInfo {
    SpinType spin_type;
    double spin_magnitude;
    std::string label;
    
    AtomInfo() : spin_type(SpinType::ISING), spin_magnitude(1.0), label("") {}
    AtomInfo(const std::string& lbl, SpinType type, double mag) 
        : spin_type(type), spin_magnitude(mag), label(lbl) {}
};

// Unit cell - simple container for atoms
class UnitCell {
private:
    std::vector<AtomInfo> atoms;
    
public:
    void add_atom(const std::string& label, SpinType type, double magnitude) {
        atoms.emplace_back(label, type, magnitude);
    }
    
    int num_atoms() const { return static_cast<int>(atoms.size()); }
    const AtomInfo& get_atom(int id) const { return atoms[id]; }
    
    bool has_mixed_spin_types() const {
        if (atoms.empty()) return false;
        SpinType first_type = atoms[0].spin_type;
        for (size_t i = 1; i < atoms.size(); i++) {
            if (atoms[i].spin_type != first_type) return true;
        }
        return false;
    }
};

// Coupling matrix - dynamic 5D array: [atom_i][atom_j][dx+max_offset][dy+max_offset][dz+max_offset]
class CouplingMatrix {
private:
    int num_atoms;
    int max_offset;        // Dynamically determined maximum offset
    int offset_size;       // 2 * max_offset + 1
    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> J;  // J[i][j][dx][dy][dz]
    
    // Convert offset to array index
    int offset_to_index(int offset) const {
        return offset + max_offset;
    }
    
public:
    CouplingMatrix() : num_atoms(0), max_offset(0), offset_size(1) {}
    
    // Initialize with specified maximum offset range
    void initialize(int n_atoms, int max_coupling_offset = 1) {
        num_atoms = n_atoms;
        max_offset = max_coupling_offset;
        offset_size = 2 * max_offset + 1;
        
        std::cout << "Initializing coupling matrix: " << n_atoms << " atoms, max_offset = " 
                  << max_offset << " (array size: " << offset_size << "³)" << std::endl;
        
        // Resize to [n_atoms][n_atoms][offset_size][offset_size][offset_size] and initialize to 0.0
        J.assign(n_atoms, 
            std::vector<std::vector<std::vector<std::vector<double>>>>(n_atoms,
                std::vector<std::vector<std::vector<double>>>(offset_size,
                    std::vector<std::vector<double>>(offset_size,
                        std::vector<double>(offset_size, 0.0)))));
        
        size_t total_elements = (size_t)n_atoms * n_atoms * offset_size * offset_size * offset_size;
        size_t memory_mb = total_elements * sizeof(double) / (1024 * 1024);
        std::cout << "Coupling matrix memory: " << memory_mb << " MB (" << total_elements << " elements)" << std::endl;
    }
    
    // Set coupling value - direct array access
    void set_coupling(int atom_i, int atom_j, int dx, int dy, int dz, double coupling_value) {
        if (std::abs(dx) > max_offset || std::abs(dy) > max_offset || std::abs(dz) > max_offset) {
            std::cerr << "Error: Coupling offset (" << dx << "," << dy << "," << dz 
                      << ") exceeds max_offset = " << max_offset << std::endl;
            return;
        }
        
        int idx_x = offset_to_index(dx);
        int idx_y = offset_to_index(dy);
        int idx_z = offset_to_index(dz);
        
        J[atom_i][atom_j][idx_x][idx_y][idx_z] = coupling_value;
    }
    
    // Get coupling value - direct array access
    double get_coupling(int atom_i, int atom_j, int dx, int dy, int dz) const {
        if (std::abs(dx) > max_offset || std::abs(dy) > max_offset || std::abs(dz) > max_offset) {
            return 0.0;  // No coupling beyond max range
        }
        
        int idx_x = offset_to_index(dx);
        int idx_y = offset_to_index(dy);
        int idx_z = offset_to_index(dz);
        
        return J[atom_i][atom_j][idx_x][idx_y][idx_z];
    }
    
    // Convenience method for symmetric nearest-neighbor couplings
    void set_nn_couplings(int atom_i, int atom_j, double coupling_value) {
        // 6 nearest neighbors
        set_coupling(atom_i, atom_j,  1, 0, 0, coupling_value);  // +x
        set_coupling(atom_i, atom_j, -1, 0, 0, coupling_value);  // -x
        set_coupling(atom_i, atom_j,  0, 1, 0, coupling_value);  // +y
        set_coupling(atom_i, atom_j,  0,-1, 0, coupling_value);  // -y
        set_coupling(atom_i, atom_j,  0, 0, 1, coupling_value);  // +z
        set_coupling(atom_i, atom_j,  0, 0,-1, coupling_value);  // -z
    }
    
    // Set intra-cell coupling (same cell, dx=dy=dz=0)
    void set_intra_coupling(int atom_i, int atom_j, double coupling_value) {
        set_coupling(atom_i, atom_j, 0, 0, 0, coupling_value);
    }
    
    int get_num_atoms() const { return num_atoms; }
    int get_max_offset() const { return max_offset; }
    
    // Print coupling matrix for debugging
    void print_summary() const {
        std::cout << "Coupling matrix summary:" << std::endl;
        std::cout << "  Atoms: " << num_atoms << std::endl;
        std::cout << "  Max offset range: ±" << max_offset << std::endl;
        
        // Count non-zero couplings
        int non_zero = 0;
        for (int i = 0; i < num_atoms; i++) {
            for (int j = 0; j < num_atoms; j++) {
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
    cell.add_atom("Atom1", model_type, 1.0);
    return cell;
}

inline CouplingMatrix create_nn_couplings(int num_atoms, double J, int max_range = 1) {
    CouplingMatrix couplings;
    couplings.initialize(num_atoms, max_range);  // Only allocate what we need
    
    // For single atom, add nearest neighbor couplings
    if (num_atoms == 1) {
        couplings.set_nn_couplings(0, 0, J);
    }
    
    return couplings;
}

#endif // MULTI_ATOM_H