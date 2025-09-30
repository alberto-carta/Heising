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
const int lattice_size = 16;           // Lattice size (16x16)
const double T_max = 6.0;              // Starting temperature
const double T_min = 0.1;              // Final temperature
const double T_step = 0.1;             // Temperature step
const double coupling_J = -1.0;        // Antiferromagnetic coupling (J > 0)
const int warmup_steps = 8000;        // Warmup steps for equilibration
const int sweeps = 40000;             // Monte Carlo sweeps for averaging

void print_header() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Monte Carlo Temperature Sweeps      " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Lattice size: " << lattice_size << "x" << lattice_size << std::endl;
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
    data_file << "# Lattice size: " << lattice_size << "x" << lattice_size 
              << ", J = " << coupling_J << std::endl;
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
        sim.run_transient_phase(warmup_steps);
        
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
        
        // Print progress
        std::cout << "<|M|> = " << std::setprecision(3) << M_abs_avg 
                  << ", <E> = " << E_avg << std::endl;
    }
    
    data_file.close();
    std::cout << "Results saved to " << filename << std::endl;
    std::cout << std::endl;
}

void analyze_critical_behavior() {
    std::cout << "=== CRITICAL BEHAVIOR ANALYSIS ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Expected critical temperatures:" << std::endl;
    std::cout << "• Ising 2D:       T_c ≈ 2.27 (exact: 2/ln(1+√2) ≈ 2.269)" << std::endl;
    std::cout << "• Heisenberg 2D:  T_c ≈ ∞ (no finite-T transition)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "What to look for in the data:" << std::endl;
    std::cout << "• Ising: Sharp drop in |M| near T_c ≈ 2.27" << std::endl;
    std::cout << "• Heisenberg: Gradual decrease in |M| (no sharp transition)" << std::endl;
    std::cout << "• Energy: Smooth variation for both models" << std::endl;
    std::cout << std::endl;
    
    std::cout << "To find T_c for Ising model:" << std::endl;
    std::cout << "1. Plot |M| vs T from ising_results.dat" << std::endl;
    std::cout << "2. Look for the steepest drop in magnetization" << std::endl;
    std::cout << "3. Calculate susceptibility χ = (⟨M²⟩ - ⟨|M|⟩²)/T" << std::endl;
    std::cout << "4. T_c is where χ peaks" << std::endl;
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