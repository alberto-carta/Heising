/*
 * Profiling and Diagnostics Module Implementation
 */

#include "../include/profiling.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

double estimate_autocorrelation(const std::vector<double>& series) {
    if (series.size() < 3) return 0.0;
    
    // Compute mean
    double mean = 0.0;
    for (double val : series) mean += val;
    mean /= series.size();
    
    // Compute lag-0 and lag-1 covariance
    double c0 = 0.0, c1 = 0.0;
    for (size_t i = 0; i < series.size(); i++) {
        c0 += (series[i] - mean) * (series[i] - mean);
    }
    for (size_t i = 0; i < series.size() - 1; i++) {
        c1 += (series[i] - mean) * (series[i+1] - mean);
    }
    
    c0 /= series.size();
    c1 /= (series.size() - 1);
    
    if (c0 > 1e-15) {
        return c1 / c0;
    }
    return 0.0;
}

double estimate_autocorrelation_time(double rho) {
    if (rho >= 1.0) return 1e10;  // Infinite correlation
    if (rho <= -1.0) return 1.0;   // Anti-correlated
    return (1.0 + rho) / (1.0 - rho);
}

void print_profiling_report(const std::vector<TemperatureTimings>& all_timings, 
                           int num_temperatures) {
    if (all_timings.empty()) return;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  PROFILING AND DIAGNOSTICS SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    
    double total_walltime = 0.0;
    double total_warmup = 0.0;
    double total_measurement = 0.0;
    double total_comm = 0.0;
    
    for (const auto& t : all_timings) {
        total_walltime += t.total_time;
        total_warmup += t.warmup_time;
        total_measurement += t.measurement_time;
        total_comm += t.communication_time;
    }
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total simulation time: " << total_walltime << " s (" 
              << total_walltime / 60.0 << " min)" << std::endl;
    std::cout << "  Warmup time:       " << total_warmup << " s (" 
              << 100.0 * total_warmup / total_walltime << "%)" << std::endl;
    std::cout << "  Measurement time:  " << total_measurement << " s (" 
              << 100.0 * total_measurement / total_walltime << "%)" << std::endl;
    std::cout << "  Communication time:" << total_comm << " s (" 
              << 100.0 * total_comm / total_walltime << "%)" << std::endl;
    
    if (all_timings[0].mc_step_time_estimate > 0.0) {
        std::cout << "\nMonte Carlo performance:" << std::endl;
        std::cout << "  Time per MC step: " << std::scientific << std::setprecision(3) 
                 << all_timings[0].mc_step_time_estimate << " s" << std::endl;
        std::cout << "  MC steps per second: " << std::fixed << std::setprecision(0) 
                 << 1.0 / all_timings[0].mc_step_time_estimate << std::endl;
    }
    
    std::cout << "\nNumber of temperature points: " << num_temperatures << std::endl;
    std::cout << "Average time per temperature: " << total_walltime / num_temperatures 
              << " s" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void print_barrier_statistics(const std::vector<double>& all_barrier_times) {
    if (all_barrier_times.empty()) return;
    
    double max_wait = *std::max_element(all_barrier_times.begin(), all_barrier_times.end());
    double avg_wait = 0.0;
    for (double w : all_barrier_times) avg_wait += w;
    avg_wait /= all_barrier_times.size();
    
    double variance = 0.0;
    for (double w : all_barrier_times) {
        variance += (w - avg_wait) * (w - avg_wait);
    }
    variance /= all_barrier_times.size();
    
    std::cout << "  Barrier sync: Max=" << std::fixed << std::setprecision(4) << max_wait << "s"
             << ", Avg=" << avg_wait << "s"
             << ", StdDev=" << std::sqrt(variance) << "s" << std::endl;
}

void print_temperature_timing(const TemperatureTimings& timings) {
    std::cout << "  Timing: Total=" << std::fixed << std::setprecision(2) << timings.total_time << "s"
             << " (Warmup=" << timings.warmup_time << "s"
             << ", Measurement=" << timings.measurement_time << "s"
             << ", Comm=" << timings.communication_time << "s)" << std::endl;
    
    if (timings.mc_step_time_estimate > 0.0) {
        std::cout << "  MC step time: " << std::scientific << std::setprecision(2) 
                 << timings.mc_step_time_estimate << " s/step" << std::endl;
    }
}
