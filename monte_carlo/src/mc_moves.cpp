/*
 * Monte Carlo Move Proposer Implementation
 * 
 * Implements move proposal logic separated from simulation engine
 */

#include "mc_moves.h"
#include "simulation_engine.h"
#include "spin_operations.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

// Constructor
MoveProposer::MoveProposer(MonteCarloSimulation& simulation) 
    : sim(simulation) {}

// Propose Ising flip - returns energy change and new spin value
MoveProposal MoveProposer::propose_ising_flip(int x, int y, int z, int spin_id) {
    // Get direct access to arrays
    auto& ising_spins = sim.get_ising_array_mutable();
    int idx = sim.flatten_index(x, y, z, spin_id);
    
    // Calculate energy before move
    double energy_before = sim.calculate_local_energy_fast(x, y, z, spin_id);
    
    // Propose flip (temporarily modify to calculate energy)
    double old_ising = ising_spins[idx];
    ising_spins[idx] = -ising_spins[idx];
    
    // Calculate energy after move
    double energy_after = sim.calculate_local_energy_fast(x, y, z, spin_id);
    
    // Restore original configuration
    double proposed_ising = ising_spins[idx];
    ising_spins[idx] = old_ising;
    
    // Return proposal
    MoveProposal proposal;
    proposal.energy_change = energy_after - energy_before;
    proposal.affected_spin_ids.push_back(spin_id);
    proposal.affected_x.push_back(x);
    proposal.affected_y.push_back(y);
    proposal.affected_z.push_back(z);
    proposal.new_ising_values.push_back(proposed_ising);
    proposal.new_hx_values.push_back(0.0);
    proposal.new_hy_values.push_back(0.0);
    proposal.new_hz_values.push_back(0.0);
    
    return proposal;
}

// Propose Heisenberg rotation - returns energy change and new spin direction
MoveProposal MoveProposer::propose_heisenberg_rotation(int x, int y, int z, int spin_id) {
    // Get direct access to arrays
    auto& heisenberg_x = sim.get_heisenberg_x_array_mutable();
    auto& heisenberg_y = sim.get_heisenberg_y_array_mutable();
    auto& heisenberg_z = sim.get_heisenberg_z_array_mutable();
    int idx = sim.flatten_index(x, y, z, spin_id);
    
    // Calculate energy before move
    double energy_before = sim.calculate_local_energy_fast(x, y, z, spin_id);
    
    // Propose rotation: select angle and apply
    spin3d old_spin(heisenberg_x[idx], heisenberg_y[idx], heisenberg_z[idx]);
    double angle = select_small_angle(sim.get_max_rotation_angle());
    spin3d new_spin = apply_rotation(old_spin, angle);
    
    // Temporarily apply to calculate energy
    heisenberg_x[idx] = new_spin.x;
    heisenberg_y[idx] = new_spin.y;
    heisenberg_z[idx] = new_spin.z;
    
    // Calculate energy after move
    double energy_after = sim.calculate_local_energy_fast(x, y, z, spin_id);
    
    // Restore original configuration
    heisenberg_x[idx] = old_spin.x;
    heisenberg_y[idx] = old_spin.y;
    heisenberg_z[idx] = old_spin.z;
    
    // Return proposal
    MoveProposal proposal;
    proposal.energy_change = energy_after - energy_before;
    proposal.affected_spin_ids.push_back(spin_id);
    proposal.affected_x.push_back(x);
    proposal.affected_y.push_back(y);
    proposal.affected_z.push_back(z);
    proposal.new_ising_values.push_back(0.0);
    proposal.new_hx_values.push_back(new_spin.x);
    proposal.new_hy_values.push_back(new_spin.y);
    proposal.new_hz_values.push_back(new_spin.z);
    
    return proposal;
}

// Propose Heisenberg flip - returns energy change and new spin direction
MoveProposal MoveProposer::propose_heisenberg_flip(int x, int y, int z, int spin_id) {
    // Get direct access to arrays
    auto& heisenberg_x = sim.get_heisenberg_x_array_mutable();
    auto& heisenberg_y = sim.get_heisenberg_y_array_mutable();
    auto& heisenberg_z = sim.get_heisenberg_z_array_mutable();
    int idx = sim.flatten_index(x, y, z, spin_id);
    
    // Calculate energy before move
    double energy_before = sim.calculate_local_energy_fast(x, y, z, spin_id);
    
    // Propose flip (reverse direction)
    spin3d old_spin(heisenberg_x[idx], heisenberg_y[idx], heisenberg_z[idx]);
    spin3d new_spin = flip_heisenberg_spin(old_spin);
    
    // Temporarily apply to calculate energy
    heisenberg_x[idx] = new_spin.x;
    heisenberg_y[idx] = new_spin.y;
    heisenberg_z[idx] = new_spin.z;
    
    // Calculate energy after move
    double energy_after = sim.calculate_local_energy_fast(x, y, z, spin_id);
    
    // Restore original configuration
    heisenberg_x[idx] = old_spin.x;
    heisenberg_y[idx] = old_spin.y;
    heisenberg_z[idx] = old_spin.z;
    
    // Return proposal
    MoveProposal proposal;
    proposal.energy_change = energy_after - energy_before;
    proposal.affected_spin_ids.push_back(spin_id);
    proposal.affected_x.push_back(x);
    proposal.affected_y.push_back(y);
    proposal.affected_z.push_back(z);
    proposal.new_ising_values.push_back(0.0);
    proposal.new_hx_values.push_back(new_spin.x);
    proposal.new_hy_values.push_back(new_spin.y);
    proposal.new_hz_values.push_back(new_spin.z);
    
    return proposal;
}

// Propose site-level double tunnel - flips all spins at a site
MoveProposal MoveProposer::propose_site_double_tunnel(int x, int y, int z, int site_id) {
    // Get direct access to arrays
    auto& ising_spins = sim.get_ising_array_mutable();
    auto& heisenberg_x = sim.get_heisenberg_x_array_mutable();
    auto& heisenberg_y = sim.get_heisenberg_y_array_mutable();
    auto& heisenberg_z = sim.get_heisenberg_z_array_mutable();
    
    // Get all spins at this site
    const UnitCell& unit_cell = sim.get_unit_cell();
    std::vector<int> spins_at_site = unit_cell.get_spins_at_site(site_id);
    
    MoveProposal proposal;
    proposal.affected_spin_ids = spins_at_site;
    
    // Store old configurations
    std::vector<double> old_ising_vals;
    std::vector<double> old_hx_vals;
    std::vector<double> old_hy_vals;
    std::vector<double> old_hz_vals;
    
    for (int spin_id : spins_at_site) {
        int idx = sim.flatten_index(x, y, z, spin_id);
        old_ising_vals.push_back(ising_spins[idx]);
        old_hx_vals.push_back(heisenberg_x[idx]);
        old_hy_vals.push_back(heisenberg_y[idx]);
        old_hz_vals.push_back(heisenberg_z[idx]);
    }
    
    // Calculate site energy before (avoids KK double counting)
    double energy_before = sim.calculate_site_energy(x, y, z, site_id);
    
    // Apply flips to all spins at this site
    for (size_t i = 0; i < spins_at_site.size(); i++) {
        int spin_id = spins_at_site[i];
        const SpinInfo& spin = unit_cell.get_spin(spin_id);
        int idx = sim.flatten_index(x, y, z, spin_id);
        
        if (spin.spin_type == SpinType::ISING) {
            // Flip Ising spin
            double new_val = -ising_spins[idx];
            ising_spins[idx] = new_val;
            proposal.new_ising_values.push_back(new_val);
            proposal.new_hx_values.push_back(0.0);
            proposal.new_hy_values.push_back(0.0);
            proposal.new_hz_values.push_back(0.0);
        } else {
            // Flip Heisenberg spin (reverse direction)
            spin3d old_spin(heisenberg_x[idx], heisenberg_y[idx], heisenberg_z[idx]);
            spin3d new_spin = flip_heisenberg_spin(old_spin);
            
            heisenberg_x[idx] = new_spin.x;
            heisenberg_y[idx] = new_spin.y;
            heisenberg_z[idx] = new_spin.z;
            
            proposal.new_ising_values.push_back(0.0);
            proposal.new_hx_values.push_back(new_spin.x);
            proposal.new_hy_values.push_back(new_spin.y);
            proposal.new_hz_values.push_back(new_spin.z);
        }
    }
    
    // Calculate site energy after all flips (avoids KK double counting)
    double energy_after = sim.calculate_site_energy(x, y, z, site_id);
    
    // Restore original configuration
    for (size_t i = 0; i < spins_at_site.size(); i++) {
        int spin_id = spins_at_site[i];
        int idx = sim.flatten_index(x, y, z, spin_id);
        
        ising_spins[idx] = old_ising_vals[i];
        heisenberg_x[idx] = old_hx_vals[i];
        heisenberg_y[idx] = old_hy_vals[i];
        heisenberg_z[idx] = old_hz_vals[i];
    }
    
    proposal.energy_change = energy_after - energy_before;
    
    // Add position for all affected spins (same x,y,z for all spins at this site)
    for (size_t i = 0; i < spins_at_site.size(); i++) {
        proposal.affected_x.push_back(x);
        proposal.affected_y.push_back(y);
        proposal.affected_z.push_back(z);
    }
    
    return proposal;
}

// Propose slab tunnel - flips entire slab region according to pre-computed flip mask
MoveProposal MoveProposer::propose_slab_tunnel(int x_start, int y_start, int z_start) {
    // Get MPI rank for debug output control
    int rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    bool debug = sim.get_slab_tunnel_debug();
    
    // Get direct access to arrays
    auto& ising_spins = sim.get_ising_array_mutable();
    auto& heisenberg_x = sim.get_heisenberg_x_array_mutable();
    auto& heisenberg_y = sim.get_heisenberg_y_array_mutable();
    auto& heisenberg_z = sim.get_heisenberg_z_array_mutable();
    
    const UnitCell& unit_cell = sim.get_unit_cell();
    const auto& flip_mask = sim.get_slab_flip_mask();
    int lateral_size = sim.get_slab_lateral_size();
    int thickness = sim.get_slab_thickness();
    int num_spins = unit_cell.num_spins();
    
    MoveProposal proposal;
    
    // Calculate slab bounds (with periodic boundary conditions handled by flatten_index)
    int x_end = x_start + lateral_size - 1;
    int y_end = y_start + lateral_size - 1;
    int z_end = z_start + thickness - 1;
    
    // Store old configuration for all spins in slab
    std::vector<int> all_indices;
    std::vector<double> old_ising_vals;
    std::vector<double> old_hx_vals;
    std::vector<double> old_hy_vals;
    std::vector<double> old_hz_vals;
    
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            for (int z = z_start; z <= z_end; z++) {
                for (int spin_id = 0; spin_id < num_spins; spin_id++) {
                    int idx = sim.flatten_index(x, y, z, spin_id);
                    all_indices.push_back(idx);
                    old_ising_vals.push_back(ising_spins[idx]);
                    old_hx_vals.push_back(heisenberg_x[idx]);
                    old_hy_vals.push_back(heisenberg_y[idx]);
                    old_hz_vals.push_back(heisenberg_z[idx]);
                }
            }
        }
    }
    
    // Calculate energy before
    // Note: For large slabs, this can be slow as it computes full system energy
    double energy_before = sim.get_energy();
    
    // Apply flips according to flip mask
    for (int x = x_start; x <= x_end; x++) {
        for (int y = y_start; y <= y_end; y++) {
            for (int z = z_start; z <= z_end; z++) {
                for (int spin_id = 0; spin_id < num_spins; spin_id++) {
                    if (flip_mask[spin_id]) {
                        int idx = sim.flatten_index(x, y, z, spin_id);
                        const SpinInfo& spin = unit_cell.get_spin(spin_id);
                        
                        if (spin.spin_type == SpinType::HEISENBERG) {
                            // Flip Heisenberg spin (reverse direction)
                            spin3d old_spin(heisenberg_x[idx], heisenberg_y[idx], heisenberg_z[idx]);
                            spin3d new_spin = flip_heisenberg_spin(old_spin);
                            
                            heisenberg_x[idx] = new_spin.x;
                            heisenberg_y[idx] = new_spin.y;
                            heisenberg_z[idx] = new_spin.z;
                            
                            proposal.affected_spin_ids.push_back(spin_id);
                            proposal.affected_x.push_back(x);
                            proposal.affected_y.push_back(y);
                            proposal.affected_z.push_back(z);
                            proposal.new_ising_values.push_back(0.0);
                            proposal.new_hx_values.push_back(new_spin.x);
                            proposal.new_hy_values.push_back(new_spin.y);
                            proposal.new_hz_values.push_back(new_spin.z);
                        } else {
                            // Ising spin - flip
                            double new_val = -ising_spins[idx];
                            ising_spins[idx] = new_val;
                            
                            proposal.affected_spin_ids.push_back(spin_id);
                            proposal.affected_x.push_back(x);
                            proposal.affected_y.push_back(y);
                            proposal.affected_z.push_back(z);
                            proposal.new_ising_values.push_back(new_val);
                            proposal.new_hx_values.push_back(0.0);
                            proposal.new_hy_values.push_back(0.0);
                            proposal.new_hz_values.push_back(0.0);
                        }
                    }
                }
            }
        }
    }
    
    // Calculate energy after
    double energy_after = sim.get_energy();
    
    // Restore original configuration
    for (size_t i = 0; i < all_indices.size(); i++) {
        int idx = all_indices[i];
        ising_spins[idx] = old_ising_vals[i];
        heisenberg_x[idx] = old_hx_vals[i];
        heisenberg_y[idx] = old_hy_vals[i];
        heisenberg_z[idx] = old_hz_vals[i];
    }
    
    proposal.energy_change = energy_after - energy_before;
    
    if (debug && rank == 0) {
        std::cout << "[DEBUG: SLAB_TUNNEL] pos=(" << x_start << "," << y_start << "," << z_start 
                  << ") size=" << lateral_size << "x" << lateral_size << "x" << thickness
                  << " flips=" << proposal.affected_spin_ids.size()
                  << " E_before=" << energy_before 
                  << " E_after=" << energy_after 
                  << " ΔE=" << proposal.energy_change << std::endl;
    }
    
    return proposal;
}
