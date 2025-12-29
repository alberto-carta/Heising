/*
 * MPI Wrapper for Monte Carlo Parallelization
 * 
 * Provides a clean interface for MPI operations with fallback to serial execution
 * when MPI is not available. Handles walker-based parallelization where each
 * MPI rank runs independent Monte Carlo simulations and results are accumulated.
 */

#ifndef MPI_WRAPPER_H
#define MPI_WRAPPER_H

#include <vector>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * MPI Environment Manager
 * 
 * Handles MPI initialization, finalization, and provides rank/size information.
 * Falls back gracefully to serial execution when compiled without MPI.
 */
class MPIEnvironment {
private:
    int rank;
    int num_ranks;
    bool is_initialized;
    
public:
    MPIEnvironment(int argc, char** argv);
    ~MPIEnvironment();
    
    // Accessors
    int get_rank() const { return rank; }
    int get_num_ranks() const { return num_ranks; }
    bool is_master() const { return rank == 0; }
    bool using_mpi() const { 
#ifdef USE_MPI
        return is_initialized;
#else
        return false;
#endif
    }
    
    // Barrier synchronization
    void barrier();
    
    /**
     * Barrier with timing - returns time spent waiting at barrier (seconds)
     * All ranks measure their wait time, rank 0 collects statistics
     */
    double barrier_with_timing();
    
    // Print info about MPI setup
    void print_info() const;
};

/**
 * MPI Statistics Accumulator
 * 
 * Handles accumulation of Monte Carlo statistics across MPI ranks.
 * Each rank computes local statistics, then they are reduced to rank 0.
 */
class MPIAccumulator {
private:
    const MPIEnvironment& mpi_env;
    
public:
    explicit MPIAccumulator(const MPIEnvironment& env) : mpi_env(env) {}
    
    /**
     * Accumulate scalar statistics across all ranks
     * Returns the sum on rank 0, undefined on other ranks
     */
    double accumulate_sum(double local_value);
    
    /**
     * Accumulate vector statistics across all ranks
     * Returns the sum on rank 0, undefined on other ranks
     */
    std::vector<double> accumulate_sum(const std::vector<double>& local_values);
    
    /**
     * Gather all measurement samples from all ranks to rank 0
     * Each rank sends its local samples, rank 0 receives concatenated array
     * Returns concatenated vector on rank 0, empty vector on other ranks
     */
    std::vector<double> gather_samples(const std::vector<double>& local_samples);
    
    /**
     * Average configuration data across all ranks for temperature continuity
     * All ranks receive the averaged configuration
     */
    void average_configuration(std::vector<double>& ising_spins);
    void average_configuration_vectors(std::vector<double>& heisenberg_x,
                                      std::vector<double>& heisenberg_y,
                                      std::vector<double>& heisenberg_z);
};

/**
 * Helper function to generate rank-specific random seed
 */
inline long int get_rank_seed(long int base_seed, int rank) {
    // Simple but effective: add rank offset to base seed
    // Negative seeds are common for ran1, so preserve sign
    if (base_seed < 0) {
        return base_seed - rank * 12345;
    } else {
        return base_seed + rank * 12345;
    }
}

#endif // MPI_WRAPPER_H
