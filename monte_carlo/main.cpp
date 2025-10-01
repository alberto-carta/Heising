/*
 * Temperature Sweep Monte Carlo Simulations
 * 
 * This program performs temperature sweeps for both Ising and Heisenberg models
 * to study phase transitions and compare their critical behavior.
 */

#include "simulation_engine.h"
#include "spin_types.h"
#include "random.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// Global random seed - required by the random number generator
long int seed = -12345;  // Negative seed for initialization

// Simulation parameters
const int lattice_size = 8;            // Lattice size (8x8x8 for 3D)
const double T_max = 6.0;              // Starting temperature
const double T_min = 0.1;              // Final temperature
const double T_step = 0.1;             // Temperature step
const double coupling_J = 1.0;        // Antiferromagnetic coupling (J > 0)
const int warmup_steps = 50000;       // Warmup steps for equilibration (increased for Heisenberg)
const int sweeps = 100000;             // Monte Carlo sweeps for averaging

void print_header() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Monte Carlo Temperature Sweeps      " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Lattice size: " << lattice_size << "x" << lattice_size << "x" << lattice_size << " (3D)" << std::endl;
    std::cout << "Temperature range: " << T_max << " to " << T_min << " (step: " << T_step << ")" << std::endl;
    std::cout << "Coupling J: " << coupling_J << " (antiferromagnetic)" << std::endl;
    std::cout << "Warmup: " << warmup_steps << " steps" << std::endl;
    std::cout << "Sweeps: " << sweeps << " steps" << std::endl;
    std::cout << std::endl;
}

void run_temperature_sweep(SpinType model_type, const std::string& filename) {
    std::string model_name = (model_type == SpinType::ISING) ? "Ising" : "Heisenberg";
    
    std::cout << "=== " << model_name << " MODEL TEMPERATURE SWEEP ===" << std::endl;
    std::cout << std::endl;
    
    // Open output file
    std::ofstream data_file(filename);
    if (!data_file.is_open()) {
        std::cerr << "Error: Cannot open output file " << filename << std::endl;
        return;
    }
    
    // Write header to data file
    data_file << "# " << model_name << " Model Monte Carlo Temperature Sweep" << std::endl;
    data_file << "# Lattice size: " << lattice_size << "x" << lattice_size << "x" << lattice_size 
              << " (3D), J = " << coupling_J << std::endl;
    data_file << "# Columns: T, M_avg, |M|_avg, M^2_avg, E_avg, E^2_avg" << std::endl;
    data_file << "Temperature,Magnetization,AbsMagnetization,MagSqAvg,Energy,EnergySqAvg" << std::endl;
    
    // Temperature loop
    for (double T = T_max; T >= T_min; T -= T_step) {
        std::cout << "T = " << std::fixed << std::setprecision(2) << T << "... ";
        std::cout.flush();
        
        // Create simulation for this temperature
        MonteCarloSimulation sim(model_type, lattice_size, T, coupling_J);
        sim.initialize_lattice();
        
        // Warmup phase
        sim.run_warmup_phase(warmup_steps);
        
        // Reset statistics after warmup (we only want production statistics)
        sim.reset_statistics();
        
        // Sweep phase - collect statistics
        double M_sum = 0.0, M_abs_sum = 0.0, M_sq_sum = 0.0;
        double E_sum = 0.0, E_sq_sum = 0.0;
        
        for (int step = 0; step < sweeps; step++) {
            sim.run_monte_carlo_step();
            
            // Collect measurements
            double M = sim.get_magnetization();
            double M_abs = sim.get_absolute_magnetization();
            double E = sim.get_energy();
            
            M_sum += M;
            M_abs_sum += M_abs;
            M_sq_sum += M * M;
            E_sum += E;
            E_sq_sum += E * E;
        }
        
        // Calculate averages
        double M_avg = M_sum / sweeps;
        double M_abs_avg = M_abs_sum / sweeps;
        double M_sq_avg = M_sq_sum / sweeps;
        double E_avg = E_sum / sweeps;
        double E_sq_avg = E_sq_sum / sweeps;
        
        // Write results to file
        data_file << std::fixed << std::setprecision(6)
                  << T << "," << M_avg << "," << M_abs_avg << "," << M_sq_avg
                  << "," << E_avg << "," << E_sq_avg << std::endl;
        
        // Print progress with acceptance rate
        double acceptance_rate = sim.get_acceptance_rate();
        std::cout << "<|M|> = " << std::setprecision(3) << M_abs_avg 
                  << ", <E> = " << E_avg 
                  << ", Accept = " << std::setprecision(1) << acceptance_rate << "%" << std::endl;
    }
    
    data_file.close();
    std::cout << "Results saved to " << filename << std::endl;
    std::cout << std::endl;
}

void analyze_critical_behavior() {
    std::cout << "=== SIMULATION COMPLETED ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Output files generated:" << std::endl;
    std::cout << "• ising_results.dat - Ising model temperature sweep" << std::endl;
    std::cout << "• heisenberg_results.dat - Heisenberg model temperature sweep" << std::endl;
    std::cout << std::endl;
}

int main() {
    print_header();
    
    try {
        // Run temperature sweep for Ising model
        run_temperature_sweep(SpinType::ISING, "ising_results.dat");
        
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::endl;
        
        // Run temperature sweep for Heisenberg model
        run_temperature_sweep(SpinType::HEISENBERG, "heisenberg_results.dat");
        
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::endl;
        
        // Analysis information
        analyze_critical_behavior();
        
        std::cout << "========================================" << std::endl;
        std::cout << "  Temperature sweeps completed!        " << std::endl;
        std::cout << "  Data files: ising_results.dat        " << std::endl;
        std::cout << "              heisenberg_results.dat   " << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}