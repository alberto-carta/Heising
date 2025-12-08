/*
 * Monte Carlo Simulation Program
 * 
 * Comprehensive example demonstrating multi-atom Monte Carlo capabilities:
 * - Single atom Ising and Heisenberg models
 * - Multi-atom systems with mixed spin types
 * - Temperature sweep studies
 * - Phase transition analysis
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h"
#include "../include/random.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

// Global random seed
long int seed = -12345;

// Simulation parameters
struct SimParams {
    int lattice_size = 8;
    double T_max = 6.0;
    double T_min = 0.1;
    double T_step = 0.1;
    double coupling_J = 1.0;
    int warmup_steps = 20000;
    int measurement_steps = 50000;
};

void print_header() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Multi-Atom Monte Carlo Simulation   " << std::endl;
    std::cout << "========================================" << std::endl;
}

void print_simulation_info(const SimParams& params) {
    std::cout << "Lattice size: " << params.lattice_size << "³" << std::endl;
    std::cout << "Temperature range: " << params.T_max << " to " << params.T_min 
              << " (step: " << params.T_step << ")" << std::endl;
    std::cout << "Coupling J: " << params.coupling_J << std::endl;
    std::cout << "Warmup: " << params.warmup_steps << " steps" << std::endl;
    std::cout << "Measurement: " << params.measurement_steps << " steps" << std::endl;
    std::cout << std::endl;
}

// Example 1: Single-atom Ising model temperature sweep
void run_ising_sweep(const SimParams& params) {
    std::cout << "=== Ising Model Temperature Sweep ===" << std::endl;
    
    std::ofstream ising_file("ising_results.dat");
    ising_file << "# T\t\tEnergy\t\tMagnetization\tAcceptRate" << std::endl;
    
    UnitCell ising_cell = create_unit_cell(SpinType::ISING);
    CouplingMatrix ising_couplings = create_nn_couplings(1, params.coupling_J);
    
    for (double T = params.T_max; T >= params.T_min; T -= params.T_step) {
        MonteCarloSimulation sim(ising_cell, ising_couplings, params.lattice_size, T);
        sim.initialize_lattice();
        
        // Warmup
        for (int i = 0; i < params.warmup_steps; i++) {
            sim.run_monte_carlo_step();
        }
        
        // Reset statistics and measure
        sim.reset_statistics();
        double total_energy = 0.0;
        double total_magnetization = 0.0;
        
        for (int i = 0; i < params.measurement_steps; i++) {
            sim.run_monte_carlo_step();
            if (i % 100 == 0) {  // Sample every 100 steps
                total_energy += sim.get_energy();
                total_magnetization += sim.get_absolute_magnetization();
            }
        }
        
        int num_samples = params.measurement_steps / 100;
        double avg_energy = total_energy / num_samples;
        double avg_magnetization = total_magnetization / num_samples;
        double accept_rate = sim.get_acceptance_rate();
        
        // Normalize per spin
        int total_spins = params.lattice_size * params.lattice_size * params.lattice_size;
        avg_energy /= total_spins;
        avg_magnetization /= total_spins;
        
        std::cout << "T=" << std::setw(5) << std::setprecision(2) << T 
                  << " E=" << std::setw(8) << std::setprecision(4) << avg_energy
                  << " M=" << std::setw(8) << std::setprecision(4) << avg_magnetization
                  << " A=" << std::setw(6) << std::setprecision(1) << accept_rate * 100 << "%" << std::endl;
        
        ising_file << T << "\t" << avg_energy << "\t" << avg_magnetization << "\t" << accept_rate << std::endl;
    }
    
    ising_file.close();
    std::cout << "Results saved to ising_results.dat" << std::endl << std::endl;
}

// Example 2: Single-atom Heisenberg model temperature sweep  
void run_heisenberg_sweep(const SimParams& params) {
    std::cout << "=== Heisenberg Model Temperature Sweep ===" << std::endl;
    
    std::ofstream heisenberg_file("heisenberg_results.dat");
    heisenberg_file << "# T\t\tEnergy\t\tMagnetization\tAcceptRate" << std::endl;
    
    UnitCell heisenberg_cell = create_unit_cell(SpinType::HEISENBERG);
    CouplingMatrix heisenberg_couplings = create_nn_couplings(1, params.coupling_J);
    
    for (double T = params.T_max; T >= params.T_min; T -= params.T_step) {
        MonteCarloSimulation sim(heisenberg_cell, heisenberg_couplings, params.lattice_size, T);
        sim.initialize_lattice();
        
        // Warmup (Heisenberg needs more equilibration)
        for (int i = 0; i < params.warmup_steps * 2; i++) {
            sim.run_monte_carlo_step();
        }
        
        // Reset statistics and measure
        sim.reset_statistics();
        double total_energy = 0.0;
        double total_magnetization = 0.0;
        
        for (int i = 0; i < params.measurement_steps; i++) {
            sim.run_monte_carlo_step();
            if (i % 100 == 0) {  // Sample every 100 steps
                total_energy += sim.get_energy();
                total_magnetization += sim.get_absolute_magnetization();
            }
        }
        
        int num_samples = params.measurement_steps / 100;
        double avg_energy = total_energy / num_samples;
        double avg_magnetization = total_magnetization / num_samples;
        double accept_rate = sim.get_acceptance_rate();
        
        // Normalize per spin
        int total_spins = params.lattice_size * params.lattice_size * params.lattice_size;
        avg_energy /= total_spins;
        avg_magnetization /= total_spins;
        
        std::cout << "T=" << std::setw(5) << std::setprecision(2) << T 
                  << " E=" << std::setw(8) << std::setprecision(4) << avg_energy
                  << " M=" << std::setw(8) << std::setprecision(4) << avg_magnetization
                  << " A=" << std::setw(6) << std::setprecision(1) << accept_rate * 100 << "%" << std::endl;
        
        heisenberg_file << T << "\t" << avg_energy << "\t" << avg_magnetization << "\t" << accept_rate << std::endl;
    }
    
    heisenberg_file.close();
    std::cout << "Results saved to heisenberg_results.dat" << std::endl << std::endl;
}

// Example 3: Multi-atom system demonstration
void run_multi_atom_demo(const SimParams& params) {
    std::cout << "=== Multi-Spin System Demonstration ===" << std::endl;
    
    // Create 4-spin unit cell with mixed spin types
    UnitCell multi_cell;
    multi_cell.add_spin("H1", SpinType::HEISENBERG, 1.0);
    multi_cell.add_spin("H2", SpinType::HEISENBERG, 1.0);
    multi_cell.add_spin("I1", SpinType::ISING, 1.0);
    multi_cell.add_spin("I2", SpinType::ISING, 1.0);
    
    // Create coupling matrix with extended range
    CouplingMatrix multi_couplings;
    multi_couplings.initialize(4, 2);  // 4 spins, max_offset = 2
    
    // Intra-cell couplings (within same unit cell)
    multi_couplings.set_intra_coupling(0, 1, -1.0);  // H1-H2 FM
    multi_couplings.set_intra_coupling(2, 3, -1.0);  // I1-I2 FM
    multi_couplings.set_intra_coupling(0, 2, -0.5);  // H1-I1 coupling
    multi_couplings.set_intra_coupling(1, 3, -0.5);  // H2-I2 coupling
    
    // Inter-cell nearest neighbor couplings
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            multi_couplings.set_nn_couplings(i, j, -0.3);
        }
    }
    
    multi_couplings.print_summary();
    
    // Run at a few key temperatures
    std::vector<double> temperatures = {4.0, 2.0, 1.0, 0.5};
    
    std::cout << "Temperature | Energy/spin | Magnetization | Accept Rate" << std::endl;
    std::cout << "------------|-------------|---------------|------------" << std::endl;
    
    for (double T : temperatures) {
        MonteCarloSimulation sim(multi_cell, multi_couplings, 6, T);  // Smaller lattice for demo
        sim.initialize_lattice();
        
        // Equilibrate
        for (int i = 0; i < params.warmup_steps; i++) {
            sim.run_monte_carlo_step();
        }
        
        // Measure
        sim.reset_statistics();
        double total_energy = 0.0;
        double total_magnetization = 0.0;
        int measurements = 0;
        
        for (int i = 0; i < params.measurement_steps / 2; i++) {  // Shorter run for demo
            sim.run_monte_carlo_step();
            if (i % 50 == 0) {
                total_energy += sim.get_energy();
                total_magnetization += sim.get_absolute_magnetization();
                measurements++;
            }
        }
        
        double avg_energy = total_energy / measurements;
        double avg_magnetization = total_magnetization / measurements;
        double accept_rate = sim.get_acceptance_rate();
        
        // Normalize per spin (6³ × 4 = 864 total spins)
        int total_spins = 6 * 6 * 6 * 4;
        avg_energy /= total_spins;
        avg_magnetization /= total_spins;
        
        std::cout << std::setw(11) << std::setprecision(1) << T << " | "
                  << std::setw(11) << std::setprecision(4) << avg_energy << " | "
                  << std::setw(13) << std::setprecision(4) << avg_magnetization << " | "
                  << std::setw(10) << std::setprecision(1) << accept_rate * 100 << "%" << std::endl;
    }
    
    std::cout << std::endl;
}

// Menu system for interactive use
void show_menu() {
    std::cout << "Available simulations:" << std::endl;
    std::cout << "1. Ising model temperature sweep" << std::endl;
    std::cout << "2. Heisenberg model temperature sweep" << std::endl;
    std::cout << "3. Multi-atom system demonstration" << std::endl;
    std::cout << "4. Run all simulations" << std::endl;
    std::cout << "5. Exit" << std::endl;
    std::cout << "Enter choice (1-5): ";
}

int main(int argc, char* argv[]) {
    print_header();
    
    SimParams params;
    
    // Check for command line arguments to modify parameters
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            params.lattice_size = std::stoi(argv[++i]);
        } else if (arg == "--tmax" && i + 1 < argc) {
            params.T_max = std::stod(argv[++i]);
        } else if (arg == "--tmin" && i + 1 < argc) {
            params.T_min = std::stod(argv[++i]);
        } else if (arg == "--coupling" && i + 1 < argc) {
            params.coupling_J = std::stod(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --size N       Set lattice size (default: 8)" << std::endl;
            std::cout << "  --tmax T       Set maximum temperature (default: 6.0)" << std::endl;
            std::cout << "  --tmin T       Set minimum temperature (default: 0.1)" << std::endl;
            std::cout << "  --coupling J   Set coupling strength (default: 1.0)" << std::endl;
            std::cout << "  --help         Show this help" << std::endl;
            return 0;
        }
    }
    
    print_simulation_info(params);
    
    // Interactive menu
    int choice;
    while (true) {
        show_menu();
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                run_ising_sweep(params);
                break;
            case 2:
                run_heisenberg_sweep(params);
                break;
            case 3:
                run_multi_atom_demo(params);
                break;
            case 4:
                run_ising_sweep(params);
                run_heisenberg_sweep(params);
                run_multi_atom_demo(params);
                break;
            case 5:
                std::cout << "Goodbye!" << std::endl;
                return 0;
            default:
                std::cout << "Invalid choice. Please try again." << std::endl;
                break;
        }
    }
    
    return 0;
}