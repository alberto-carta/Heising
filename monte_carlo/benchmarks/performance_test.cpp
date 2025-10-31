/*
 * Performance Benchmarks for Monte Carlo Implementation
 * 
 * Tests computational performance of key operations:
 * 1. Local energy calculation time vs. number of neighbors
 * 2. Lattice setup time for different system sizes
 * 3. Monte Carlo step time (proposal + acceptance)
 * 4. Memory usage scaling with coupling range
 */

#include "../include/simulation_engine.h"
#include "../include/multi_atom.h"
#include "../include/random.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

long int seed = -12345;

// Timing utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }
};

// Test 1: Local energy calculation performance vs. neighbor count
void benchmark_local_energy() {
    std::cout << "\n=== Benchmark 1: Local Energy Calculation ===" << std::endl;
    std::cout << "Testing energy calculation time vs. coupling range" << std::endl;
    std::cout << "Max Offset | Couplings | Time per call (μs)" << std::endl;
    std::cout << "-----------|-----------|------------------" << std::endl;
    
    const int num_calls = 10000;
    Timer timer;
    
    for (int max_offset = 1; max_offset <= 3; max_offset++) {
        // Create system with specified coupling range
        UnitCell cell = create_unit_cell(SpinType::HEISENBERG);
        CouplingMatrix couplings;
        couplings.initialize(1, max_offset);
        
        // Add couplings up to max_offset
        for (int dx = -max_offset; dx <= max_offset; dx++) {
            for (int dy = -max_offset; dy <= max_offset; dy++) {
                for (int dz = -max_offset; dz <= max_offset; dz++) {
                    if (dx != 0 || dy != 0 || dz != 0) {  // Skip self-interaction
                        couplings.set_coupling(0, 0, dx, dy, dz, -1.0);
                    }
                }
            }
        }
        
        MonteCarloSimulation sim(cell, couplings, 8, 1.0);
        sim.initialize_lattice();
        
        // Count actual non-zero couplings
        int coupling_count = 0;
        for (int dx = -max_offset; dx <= max_offset; dx++) {
            for (int dy = -max_offset; dy <= max_offset; dy++) {
                for (int dz = -max_offset; dz <= max_offset; dz++) {
                    if (couplings.get_coupling(0, 0, dx, dy, dz) != 0.0) {
                        coupling_count++;
                    }
                }
            }
        }
        
        // Time many local energy calculations
        timer.start();
        for (int i = 0; i < num_calls; i++) {
            // Calculate energy at center of lattice
            sim.calculate_local_energy(4, 4, 4, 0);
        }
        double total_time = timer.elapsed_ms();
        double time_per_call = (total_time * 1000.0) / num_calls;  // Convert to microseconds
        
        std::cout << std::setw(10) << max_offset << " | " 
                  << std::setw(9) << coupling_count << " | "
                  << std::setw(17) << std::fixed << std::setprecision(2) << time_per_call << std::endl;
    }
}

// Test 2: Lattice setup time vs. system size
void benchmark_setup_time() {
    std::cout << "\n=== Benchmark 2: Lattice Setup Time ===" << std::endl;
    std::cout << "Testing initialization time vs. system size" << std::endl;
    std::cout << "Lattice Size | Total Spins | Setup Time (ms)" << std::endl;
    std::cout << "-------------|-------------|----------------" << std::endl;
    
    Timer timer;
    std::vector<int> sizes = {4, 8, 16, 32};
    
    for (int size : sizes) {
        UnitCell cell = create_unit_cell(SpinType::HEISENBERG);
        CouplingMatrix couplings = create_nn_couplings(1, -1.0);
        
        timer.start();
        MonteCarloSimulation sim(cell, couplings, size, 1.0);
        sim.initialize_lattice();
        double setup_time = timer.elapsed_ms();
        
        int total_spins = size * size * size;
        
        std::cout << std::setw(12) << size << " | " 
                  << std::setw(11) << total_spins << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << setup_time << std::endl;
    }
}

// Test 3: Monte Carlo step performance (proposal + acceptance)
void benchmark_mc_steps() {
    std::cout << "\n=== Benchmark 3: Monte Carlo Step Performance ===" << std::endl;
    std::cout << "Testing MC step time for different spin types" << std::endl;
    std::cout << "Spin Type   | Steps/sec | Accept Rate | Time/step (μs)" << std::endl;
    std::cout << "------------|-----------|-------------|---------------" << std::endl;
    
    Timer timer;
    const int num_steps = 50000;
    
    // Test Ising model
    {
        UnitCell ising_cell = create_unit_cell(SpinType::ISING);
        CouplingMatrix ising_couplings = create_nn_couplings(1, -1.0);
        MonteCarloSimulation ising_sim(ising_cell, ising_couplings, 8, 2.0);
        ising_sim.initialize_lattice();
        
        timer.start();
        for (int i = 0; i < num_steps; i++) {
            ising_sim.run_monte_carlo_step();
        }
        double ising_time = timer.elapsed_ms();
        
        double steps_per_sec = num_steps / (ising_time / 1000.0);
        double time_per_step = (ising_time * 1000.0) / num_steps;
        double accept_rate = ising_sim.get_acceptance_rate();
        
        std::cout << std::setw(11) << "Ising" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(0) << steps_per_sec << " | "
                  << std::setw(11) << std::setprecision(1) << accept_rate * 100 << "% | "
                  << std::setw(14) << std::setprecision(2) << time_per_step << std::endl;
    }
    
    // Test Heisenberg model
    {
        UnitCell heisenberg_cell = create_unit_cell(SpinType::HEISENBERG);
        CouplingMatrix heisenberg_couplings = create_nn_couplings(1, -1.0);
        MonteCarloSimulation heisenberg_sim(heisenberg_cell, heisenberg_couplings, 8, 2.0);
        heisenberg_sim.initialize_lattice();
        
        timer.start();
        for (int i = 0; i < num_steps; i++) {
            heisenberg_sim.run_monte_carlo_step();
        }
        double heisenberg_time = timer.elapsed_ms();
        
        double steps_per_sec = num_steps / (heisenberg_time / 1000.0);
        double time_per_step = (heisenberg_time * 1000.0) / num_steps;
        double accept_rate = heisenberg_sim.get_acceptance_rate();
        
        std::cout << std::setw(11) << "Heisenberg" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(0) << steps_per_sec << " | "
                  << std::setw(11) << std::setprecision(1) << accept_rate * 100 << "% | "
                  << std::setw(14) << std::setprecision(2) << time_per_step << std::endl;
    }
    
    // Test multi-atom system
    {
        UnitCell multi_cell;
        multi_cell.add_atom("H1", SpinType::HEISENBERG, 1.0);
        multi_cell.add_atom("I1", SpinType::ISING, 1.0);
        
        CouplingMatrix multi_couplings;
        multi_couplings.initialize(2, 1);
        multi_couplings.set_intra_coupling(0, 1, -1.0);
        multi_couplings.set_nn_couplings(0, 0, -0.5);
        multi_couplings.set_nn_couplings(1, 1, -0.5);
        
        MonteCarloSimulation multi_sim(multi_cell, multi_couplings, 6, 2.0);
        multi_sim.initialize_lattice();
        
        timer.start();
        for (int i = 0; i < num_steps; i++) {
            multi_sim.run_monte_carlo_step();
        }
        double multi_time = timer.elapsed_ms();
        
        double steps_per_sec = num_steps / (multi_time / 1000.0);
        double time_per_step = (multi_time * 1000.0) / num_steps;
        double accept_rate = multi_sim.get_acceptance_rate();
        
        std::cout << std::setw(11) << "Multi-atom" << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(0) << steps_per_sec << " | "
                  << std::setw(11) << std::setprecision(1) << accept_rate * 100 << "% | "
                  << std::setw(14) << std::setprecision(2) << time_per_step << std::endl;
    }
}

// Test 4: Memory usage analysis
void benchmark_memory_usage() {
    std::cout << "\n=== Benchmark 4: Memory Usage Analysis ===" << std::endl;
    std::cout << "Coupling matrix memory usage by configuration" << std::endl;
    std::cout << "Atoms | Max Offset | Array Size | Memory (KB)" << std::endl;
    std::cout << "------|------------|------------|------------" << std::endl;
    
    std::vector<std::pair<int, int>> configs = {{1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {4, 2}};
    
    for (auto& config : configs) {
        int num_atoms = config.first;
        int max_offset = config.second;
        
        int array_size = 2 * max_offset + 1;
        size_t total_elements = (size_t)num_atoms * num_atoms * array_size * array_size * array_size;
        size_t memory_bytes = total_elements * sizeof(double);
        double memory_kb = memory_bytes / 1024.0;
        
        std::cout << std::setw(5) << num_atoms << " | "
                  << std::setw(10) << max_offset << " | "
                  << std::setw(10) << array_size << "³ | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << memory_kb << std::endl;
    }
    
    // Compare with old fixed approach
    std::cout << "\nComparison with old fixed 7³ approach:" << std::endl;
    size_t old_fixed_elements = 343;  // 7³
    double old_memory_kb = old_fixed_elements * sizeof(double) / 1024.0;
    
    std::cout << "Old fixed (7³): " << std::fixed << std::setprecision(1) << old_memory_kb << " KB per atom pair" << std::endl;
    std::cout << "New dynamic (3³): " << (27 * sizeof(double) / 1024.0) << " KB per atom pair (NN only)" << std::endl;
    std::cout << "Memory reduction: " << std::setprecision(1) << (old_memory_kb / (27 * sizeof(double) / 1024.0)) << "x" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   MONTE CARLO PERFORMANCE BENCHMARKS  " << std::endl;
    std::cout << "========================================" << std::endl;
    
    benchmark_local_energy();
    benchmark_setup_time();
    benchmark_mc_steps();
    benchmark_memory_usage();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "         BENCHMARKS COMPLETED          " << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}