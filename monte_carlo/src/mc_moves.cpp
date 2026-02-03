/*
 * Monte Carlo Move Proposer Implementation
 * 
 * Implements move proposal logic separated from simulation engine
 */

#include "mc_moves.h"
#include "simulation_engine.h"
#include "spin_operations.h"

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
    return proposal;
}
