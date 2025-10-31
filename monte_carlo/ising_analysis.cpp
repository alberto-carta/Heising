/*
 * Ising Ferromagnet Phase Transition Analysis
 * 
 * Non-interactive program to study the ferromagnetic transition
 * in the 3D Ising model using Monte Carlo simulation
 */

#include "../include/simulation_engine.h"
#include "../include/multi_atom.h" 
#include "../include/random.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

// Global random seed
long int seed = -12345;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "     Ising Ferromagnet Transition      " << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Simulation parameters optimized for transition detection
    int lattice_size = 8;           // 8³ = 512 spins (good compromise)
    double T_max = 6.0;             // Above critical temperature (Ising Tc ~ 4.5)
    double T_min = 0.5;             // Well below critical temperature
    double T_step = 0.2;            // Fine temperature steps
    double coupling_J = -1.0;       // Ferromagnetic coupling (J < 0)
    int warmup_steps = 8000;        // Equilibration steps
    int measurement_steps = 80000;  // Good statistics for transition
    
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Lattice size: " << lattice_size << "³ (" << lattice_size*lattice_size*lattice_size << " spins)" << std::endl;
    std::cout << "  Temperature range: " << T_max << " to " << T_min << " (step: " << T_step << ")" << std::endl;
    std::cout << "  Coupling J: " << coupling_J << " (ferromagnetic)" << std::endl;
    std::cout << "  Warmup: " << warmup_steps << " steps" << std::endl;
    std::cout << "  Measurement: " << measurement_steps << " steps" << std::endl;
    std::cout << "  Expected Tc (3D Ising): ~4.5" << std::endl;
    std::cout << std::endl;
    
    // Create ferromagnetic Ising system
    UnitCell ising_cell = create_unit_cell(SpinType::ISING);
    CouplingMatrix ising_couplings = create_nn_couplings(1, coupling_J);  // Ferromagnetic
    
    // Output file for analysis
    std::ofstream outfile("ising_ferromagnet_transition.dat");
    outfile << "# Ising Ferromagnet Transition Data" << std::endl;
    outfile << "# Lattice: " << lattice_size << "³, J = " << coupling_J << std::endl;
    outfile << "# Columns: T          Energy/spin   Magnetization |Magnetization| SpecificHeat  Susceptibility AcceptanceRate" << std::endl;
    outfile << std::fixed << std::setprecision(8);
    
    std::cout << "T          Energy/spin   Magnetization |Magnetization| SpecificHeat  Susceptibility AcceptanceRate" << std::endl;
    std::cout << "---------- ------------- ------------- ------------- ------------- -------------- --------------" << std::endl;
    
    double prev_energy = 0.0;
    double prev_mag = 0.0;
    
    // Storage for previous temperature configuration (Ising spins)
    std::vector<double> saved_ising_spins;
    bool first_temperature = true;
    int total_spins = lattice_size * lattice_size * lattice_size;
    
    for (double T = T_max; T >= T_min; T -= T_step) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Temperature T = " << std::fixed << std::setprecision(2) << T << std::endl;
        std::cout << "========================================" << std::endl;
        
        MonteCarloSimulation sim(ising_cell, ising_couplings, lattice_size, T);
        
        if (first_temperature) {
            // Initialize with aligned configuration for first temperature
            std::cout << "Initializing with ferromagnetic ground state (first temperature)..." << std::endl;
            sim.initialize_lattice();
            first_temperature = false;
        } else {
            // Start from previous temperature configuration
            std::cout << "Loading configuration from previous temperature..." << std::endl;
            sim.initialize_lattice(); // Initialize arrays first
            
            // Copy saved Ising configuration
            int idx = 0;
            for (int x = 1; x <= lattice_size; x++) {
                for (int y = 1; y <= lattice_size; y++) {
                    for (int z = 1; z <= lattice_size; z++) {
                        for (int atom_id = 0; atom_id < ising_cell.num_atoms(); atom_id++) {
                            sim.set_ising_spin(x, y, z, atom_id, saved_ising_spins[idx]);
                            idx++;
                        }
                    }
                }
            }
            std::cout << "Configuration loaded from previous temperature." << std::endl;
        }
        
        // Warmup phase
        std::cout << "Warmup phase:" << std::endl;
        for (int sweep = 0; sweep < warmup_steps; sweep++) {
            // One sweep = N attempts where N = total number of spins
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            if ((sweep + 1) % 1000 == 0) {
                std::cout << "  Warmup sweep " << (sweep + 1) << "/" << warmup_steps 
                          << " (" << std::fixed << std::setprecision(1) 
                          << (100.0 * (sweep + 1)) / warmup_steps << "%)" << std::endl;
            }
        }
        
        // Measurement phase
        std::cout << "Measurement phase:" << std::endl;
        sim.reset_statistics();
        double total_energy = 0.0;
        double total_energy_sq = 0.0;
        double total_magnetization = 0.0;
        double total_abs_magnetization = 0.0;
        double total_magnetization_sq = 0.0;
        int num_samples = 0;
        
        for (int sweep = 0; sweep < measurement_steps; sweep++) {
            // One sweep = N attempts where N = total number of spins
            for (int attempt = 0; attempt < total_spins; attempt++) {
                sim.run_monte_carlo_step();
            }
            
            if (sweep % 100 == 0) {  // Sample every 100 sweeps to reduce correlation
                double energy = sim.get_energy();
                double magnetization = sim.get_magnetization();
                double abs_magnetization = sim.get_absolute_magnetization();
                
                total_energy += energy;
                total_energy_sq += energy * energy;
                total_magnetization += magnetization;
                total_abs_magnetization += abs_magnetization;
                total_magnetization_sq += magnetization * magnetization;
                num_samples++;
            }
            
            // Progress feedback every 10000 sweeps
            if ((sweep + 1) % 10000 == 0) {
                std::cout << "  Measurement sweep " << (sweep + 1) << "/" << measurement_steps 
                          << " (" << std::fixed << std::setprecision(1) 
                          << (100.0 * (sweep + 1)) / measurement_steps << "%)" << std::endl;
            }
        }
        
        // Calculate averages and fluctuations
        double avg_energy_per_spin = total_energy / num_samples / total_spins;
        double avg_energy_sq_per_spin = total_energy_sq / num_samples / (total_spins * total_spins);
        double avg_magnetization_per_spin = total_magnetization / num_samples / total_spins;
        double avg_abs_magnetization_per_spin = total_abs_magnetization / num_samples / total_spins;
        double avg_magnetization_sq_per_spin = total_magnetization_sq / num_samples / (total_spins * total_spins);
        
        // Calculate thermodynamic quantities per spin
        // Specific heat: C_v = (⟨E²⟩ - ⟨E⟩²) / T² per spin
        double specific_heat = (avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (T * T);
        
        // Susceptibility: χ = (⟨M²⟩ - ⟨M⟩²) / T per spin  
        double susceptibility = (avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / T;
        double accept_rate = sim.get_acceptance_rate();
        
        // Display results to 8 decimal places
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "T = " << T << std::endl;
        std::cout << "Average Energy per spin = " << avg_energy_per_spin << std::endl;
        std::cout << "Average Magnetization per spin = " << avg_magnetization_per_spin << std::endl;
        std::cout << "Average Absolute Magnetization per spin = " << avg_abs_magnetization_per_spin << std::endl;
        std::cout << "Specific Heat per spin = " << specific_heat << std::endl;
        std::cout << "Susceptibility per spin = " << susceptibility << std::endl;
        std::cout << "Acceptance Rate = " << std::setprecision(6) << accept_rate << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Write to file with consistent formatting
        outfile << std::fixed << std::setprecision(8);  // Reset to 8 decimals for all values
        outfile << std::setw(10) << T << " "
                << std::setw(13) << avg_energy_per_spin << " "
                << std::setw(13) << avg_magnetization_per_spin << " "
                << std::setw(13) << avg_abs_magnetization_per_spin << " "
                << std::setw(13) << specific_heat << " "
                << std::setw(14) << susceptibility << " "
                << std::setw(14) << std::setprecision(6) << accept_rate << std::endl;
        
        // Save current configuration for next temperature step
        std::cout << "Saving configuration for next temperature step..." << std::endl;
        saved_ising_spins.clear();
        saved_ising_spins.reserve(total_spins);
        
        for (int x = 1; x <= lattice_size; x++) {
            for (int y = 1; y <= lattice_size; y++) {
                for (int z = 1; z <= lattice_size; z++) {
                    for (int atom_id = 0; atom_id < ising_cell.num_atoms(); atom_id++) {
                        saved_ising_spins.push_back(sim.get_ising_spin(x, y, z, atom_id));
                    }
                }
            }
        }
        
        // Estimate critical temperature from specific heat maximum (rough)
        if (T < T_max - T_step) {  // Skip first point
            if (specific_heat > 10.0 && prev_energy != 0.0) {
                std::cout << "*** Possible transition region detected near T = " << T << " ***" << std::endl;
            }
        }
        
        prev_energy = avg_energy_per_spin;
        prev_mag = avg_abs_magnetization_per_spin;
    }
    
    outfile.close();
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Analysis completed!" << std::endl;
    std::cout << "Results saved to: ising_ferromagnet_transition.dat" << std::endl;
    std::cout << std::endl;
    std::cout << "Analysis tips:" << std::endl;
    std::cout << "- Critical temperature (Tc): Look for peaks in specific heat and susceptibility" << std::endl;
    std::cout << "- Magnetization: Should drop from ~1 to ~0 at the transition" << std::endl;
    std::cout << "- 3D Ising model: Tc/|J| ≈ 4.5 (theory), so expect Tc ≈ 4.5 for |J|=1" << std::endl;
    std::cout << "- Compare with Heisenberg model: Ising should have sharper transition" << std::endl;
    std::cout << std::endl;
    
    return 0;
}