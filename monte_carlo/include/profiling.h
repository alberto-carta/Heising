/*
 * Profiling and Diagnostics Module
 * 
 * Tools for profiling Monte Carlo simulations including:
 * - Timing measurements
 * - Autocorrelation estimation
 * - Performance statistics
 */

#ifndef PROFILING_H
#define PROFILING_H

#include <vector>
#include <chrono>

/**
 * Timing data structure for temperature scan profiling
 */
struct TemperatureTimings {
    double warmup_time = 0.0;
    double measurement_time = 0.0;
    double communication_time = 0.0;
    double total_time = 0.0;
    double barrier_wait_time = 0.0;
    double mc_step_time_estimate = 0.0;  // Time per MC step from warmup sampling
};

/**
 * Simple autocorrelation estimator
 * Computes lag-1 autocorrelation coefficient from time series
 * 
 * @param series Time series data
 * @return Lag-1 autocorrelation coefficient ρ(1)
 */
double estimate_autocorrelation(const std::vector<double>& series);

/**
 * Estimate autocorrelation time from lag-1 autocorrelation
 * Uses approximation: τ ≈ (1 + ρ) / (1 - ρ) for small ρ
 * 
 * @param rho Lag-1 autocorrelation coefficient
 * @return Estimated autocorrelation time in sweeps
 */
double estimate_autocorrelation_time(double rho);

/**
 * Print comprehensive profiling report
 * 
 * @param all_timings Vector of timing data for each temperature
 * @param num_temperatures Number of temperature points
 */
void print_profiling_report(const std::vector<TemperatureTimings>& all_timings, 
                           int num_temperatures);

/**
 * Print barrier synchronization statistics
 * 
 * @param all_barrier_times Barrier wait times from all MPI ranks
 */
void print_barrier_statistics(const std::vector<double>& all_barrier_times);

/**
 * Print per-temperature timing summary
 * 
 * @param timings Timing data for current temperature
 */
void print_temperature_timing(const TemperatureTimings& timings);

#endif // PROFILING_H
