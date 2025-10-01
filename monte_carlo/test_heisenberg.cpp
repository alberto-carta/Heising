/*
 * Quick test for Heisenberg model convergence
 */

#include "simulation_engine.h"
#include "spin_types.h"
#include "random.h"
#include <iostream>
#include <iomanip>

long int seed = -12345;

int main() {
    // Test single temperature
    double T = 0.5;  // Very low temperature
    int size = 8;
    double J = -1.0;  // AFM coupling
    
    std::cout << "Testing Heisenberg model at T = " << T << std::endl;
    std::cout << "Expected: High magnetization if properly converged" << std::endl;
    std::cout << "DEBUG: This should show updated version" << std::endl;
    std::cout << "========================================" << std::endl;
    
    MonteCarloSimulation sim(SpinType::HEISENBERG, size, T, J);
    sim.initialize_lattice();
    
    // Check initial state
    double initial_M = sim.get_magnetization();
    double initial_E = sim.get_energy();
    std::cout << "Initial: M = " << initial_M << ", E = " << initial_E << std::endl;
    
    // Long warmup
    std::cout << "Running 20,000 warmup steps..." << std::endl;
    sim.run_warmup_phase(20000);
    sim.reset_statistics();
    
    // Check after warmup
    double warmup_M = sim.get_magnetization();
    double warmup_E = sim.get_energy();
    std::cout << "After warmup: M = " << warmup_M << ", E = " << warmup_E << std::endl;
    
    // Production run
    std::cout << "Running 10,000 production steps..." << std::endl;
    double M_sum = 0.0, M_abs_sum = 0.0, E_sum = 0.0;
    
    for (int step = 0; step < 10000; step++) {
        sim.run_monte_carlo_step();
        
        double M = sim.get_magnetization();
        double E = sim.get_energy();
        
        M_sum += M;
        M_abs_sum += std::abs(M);
        E_sum += E;
        
        if (step % 1000 == 0) {
            std::cout << "Step " << step << ": M = " << M << ", E = " << E 
                      << ", Accept = " << sim.get_acceptance_rate() << "%" << std::endl;
        }
    }
    
    // Final averages
    double M_avg = M_sum / 10000;
    double M_abs_avg = M_abs_sum / 10000;
    double E_avg = E_sum / 10000;
    
    std::cout << "========================================" << std::endl;
    std::cout << "Final results:" << std::endl;
    std::cout << "Average M = " << M_avg << std::endl;
    std::cout << "Average |M| = " << M_abs_avg << std::endl;
    std::cout << "Average E = " << E_avg << std::endl;
    std::cout << "Acceptance rate = " << sim.get_acceptance_rate() << "%" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Diagnostic info
    std::cout << "Diagnostic info:" << std::endl;
    std::cout << "Total spins = " << size*size*size << " = " << (size*size*size) << std::endl;
    std::cout << "If fully ordered, |M| should be close to " << (size*size*size) << std::endl;
    std::cout << "Current |M|/N = " << (M_abs_avg/(size*size*size)) << " (should be ~1 if ordered)" << std::endl;
    
    return 0;
}