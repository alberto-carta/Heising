/*
 * MPI Wrapper Implementation
 */

#include "mpi_wrapper.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

// MPIEnvironment Implementation
MPIEnvironment::MPIEnvironment(int argc, char** argv) 
    : rank(0), num_ranks(1), is_initialized(false) {
    
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    is_initialized = true;
    
    if (rank == 0) {
        std::cout << "MPI initialized with " << num_ranks << " ranks" << std::endl;
        std::cout << "Thread support level: " << provided << std::endl;
    }
#else
    // Serial execution
    rank = 0;
    num_ranks = 1;
    is_initialized = false;
    std::cout << "Running in serial mode (MPI not available)" << std::endl;
#endif
}

MPIEnvironment::~MPIEnvironment() {
#ifdef USE_MPI
    if (is_initialized) {
        MPI_Finalize();
    }
#endif
}

void MPIEnvironment::barrier() {
#ifdef USE_MPI
    if (is_initialized) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}

double MPIEnvironment::barrier_with_timing() {
#ifdef USE_MPI
    if (is_initialized) {
        double start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        return end_time - start_time;
    }
#endif
    return 0.0;
}

void MPIEnvironment::print_info() const {
#ifdef USE_MPI
    if (is_initialized) {
        std::cout << "Rank " << rank << "/" << num_ranks << " ready" << std::endl;
    } else {
        std::cout << "Serial execution (no MPI)" << std::endl;
    }
#else
    std::cout << "Serial execution (compiled without MPI)" << std::endl;
#endif
}

// MPIAccumulator Implementation
double MPIAccumulator::accumulate_sum(double local_value) {
#ifdef USE_MPI
    if (mpi_env.using_mpi()) {
        double global_sum = 0.0;
        MPI_Reduce(&local_value, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 
                   0, MPI_COMM_WORLD);
        return global_sum;
    } else {
        return local_value;
    }
#else
    return local_value;
#endif
}

std::vector<double> MPIAccumulator::accumulate_sum(const std::vector<double>& local_values) {
#ifdef USE_MPI
    if (mpi_env.using_mpi()) {
        std::vector<double> global_sum(local_values.size(), 0.0);
        MPI_Reduce(const_cast<double*>(local_values.data()), global_sum.data(), 
                   local_values.size(), MPI_DOUBLE, MPI_SUM, 
                   0, MPI_COMM_WORLD);
        return global_sum;
    } else {
        return local_values;
    }
#else
    return local_values;
#endif
}

void MPIAccumulator::average_configuration(std::vector<double>& ising_spins) {
#ifdef USE_MPI
    if (mpi_env.using_mpi()) {
        // Create buffer for averaged data
        std::vector<double> averaged(ising_spins.size());
        
        // All-reduce to average across all ranks
        MPI_Allreduce(ising_spins.data(), averaged.data(), 
                     ising_spins.size(), MPI_DOUBLE, MPI_SUM, 
                     MPI_COMM_WORLD);
        
        // Divide by number of ranks to get average
        double scale = 1.0 / mpi_env.get_num_ranks();
        for (size_t i = 0; i < averaged.size(); i++) {
            ising_spins[i] = averaged[i] * scale;
        }
    }
    // If not using MPI, configuration is already what we want
#endif
}

void MPIAccumulator::average_configuration_vectors(std::vector<double>& heisenberg_x,
                                                   std::vector<double>& heisenberg_y,
                                                   std::vector<double>& heisenberg_z) {
#ifdef USE_MPI
    if (mpi_env.using_mpi()) {
        size_t size = heisenberg_x.size();
        
        // Average x components
        std::vector<double> avg_x(size);
        MPI_Allreduce(heisenberg_x.data(), avg_x.data(), 
                     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Average y components
        std::vector<double> avg_y(size);
        MPI_Allreduce(heisenberg_y.data(), avg_y.data(), 
                     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Average z components
        std::vector<double> avg_z(size);
        MPI_Allreduce(heisenberg_z.data(), avg_z.data(), 
                     size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Scale and normalize
        double scale = 1.0 / mpi_env.get_num_ranks();
        for (size_t i = 0; i < size; i++) {
            double x = avg_x[i] * scale;
            double y = avg_y[i] * scale;
            double z = avg_z[i] * scale;
            
            // Renormalize to unit vector for Heisenberg spins
            double norm = std::sqrt(x*x + y*y + z*z);
            if (norm > 1e-10) {
                heisenberg_x[i] = x / norm;
                heisenberg_y[i] = y / norm;
                heisenberg_z[i] = z / norm;
            } else {
                // If norm is too small, set to z-direction
                heisenberg_x[i] = 0.0;
                heisenberg_y[i] = 0.0;
                heisenberg_z[i] = 1.0;
            }
        }
    }
#endif
}
