/*
 * Heising Main - MPI Monte Carlo Simulation Program
 * 
 * Reads configuration from TOML files and performs Monte Carlo
 * simulations on arbitrary magnetic systems using MPI parallelization
 * 
 * Requires MPI for execution
 */

#include "../include/simulation_engine.h"
#include "../include/multi_spin.h" 
#include "../include/random.h"
#include "../include/io/configuration_parser.h"
#include "../include/io/diagnostic_utils.h"
#include "../include/io/output_formatting.h"
#include "../include/mpi_wrapper.h"
#include "../include/profiling.h"
#include "../include/simulation_utils.h"
#include "../include/mc_phases.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

// Global random seed (will be set from configuration)
long int seed = -12345;

// Structure to hold results from a single temperature simulation
struct TemperatureResults {
    double T;
    int total_spins;
    double avg_energy;
    double stddev_energy;
    double avg_magnetization;
    double stddev_magnetization;
    double specific_heat;
    double stddev_specific_heat;
    double susceptibility;
    double stddev_susceptibility;
    double avg_accept_rate;
    double stddev_accept_rate;
    std::vector<spin3d> avg_mag_vectors;
    std::vector<spin3d> stddev_mag_vectors;
    std::vector<double> avg_correlations;
    std::vector<double> stddev_correlations;
    TemperatureTimings timings;
};

/**
 * Initialize simulation based on configuration
 */
void initialize_simulation(MonteCarloSimulation& sim, const IO::InitializationConfig& init_config) {
    switch (init_config.type) {
        case IO::InitializationConfig::RANDOM:
            sim.initialize_lattice_random();
            break;
        case IO::InitializationConfig::CUSTOM:
            if (init_config.pattern.empty()) {
                std::cerr << "ERROR: CUSTOM initialization requires 'pattern' array in config" << std::endl;
                sim.initialize_lattice_random();  // Fall back to random
            } else {
                sim.initialize_lattice_custom(init_config.pattern);
            }
            break;
        default:
            sim.initialize_lattice_random();
            break;
    }
}

/**
 * Run simulation at a single temperature with MPI parallelization
 * This is the core function called by both single_temperature and temperature_scan modes
 */
TemperatureResults run_temperature_point(
    double T,
    const IO::SimulationConfig& config,
    const UnitCell& unit_cell,
    const CouplingMatrix& couplings,
    const std::optional<KK_Matrix>& kk_matrix,
    MPIEnvironment& mpi_env,
    MPIAccumulator& mpi_accumulator,
    bool first_temperature,
    const std::string& dump_dir)
{
    int rank = mpi_env.get_rank();
    int num_ranks = mpi_env.get_num_ranks();
    int total_spins = config.lattice_size * config.lattice_size * config.lattice_size * config.species.size();
    
    // Divide measurement steps among ranks
    int measurement_steps_per_rank = config.monte_carlo.measurement_steps / num_ranks;
    
    TemperatureResults results;
    results.T = T;
    results.total_spins = total_spins;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Each rank creates its own simulation with rank-specific seed
    long int rank_seed = get_rank_seed(config.monte_carlo.seed, rank);
    seed = rank_seed;
    
    MonteCarloSimulation sim(unit_cell, couplings, config.lattice_size, T, kk_matrix);
    
    // Initialization
    if (first_temperature) {
        if (rank == 0) {
            std::cout << "  Initializing all walkers..." << std::endl;
        }
        initialize_simulation(sim, config.initialization);
    } else {
        // All ranks initialize and then average their configurations
        if (rank == 0) {
            std::cout << "  Averaging configurations from all walkers..." << std::endl;
        }
        initialize_simulation(sim, config.initialization);
        average_configuration_mpi(sim, config.species, mpi_accumulator);
        mpi_env.barrier();
        if (rank == 0) {
            std::cout << "  All walkers synchronized with averaged configuration." << std::endl;
        }
    }
    
    if (rank == 0) {
        IO::print_subsection_separator("Starting simulation");
        std::cout << std::endl;
    }
    
    // Warmup phase
    auto [warmup_time, mc_step_estimate] = run_warmup_phase(sim, config.monte_carlo.warmup_steps, 
                                                              total_spins, config.diagnostics.enable_profiling, rank);
    results.timings.warmup_time = warmup_time;
    results.timings.mc_step_time_estimate = mc_step_estimate;
    
    // Measurement phase
    auto [measurement_data, measurement_time] = run_measurement_phase(sim, config, measurement_steps_per_rank,
                                                                        total_spins, T, rank, dump_dir);
    results.timings.measurement_time = measurement_time;
    
    // Gather all measurements from all ranks to rank 0
    auto comm_start = std::chrono::high_resolution_clock::now();
    std::vector<double> all_energy_samples = mpi_accumulator.gather_samples(measurement_data.energy_samples);
    std::vector<double> all_magnetization_samples = mpi_accumulator.gather_samples(measurement_data.magnetization_samples);
    
    // Gather magnetization vector components for each species
    std::vector<std::vector<double>> all_mag_x_samples(config.species.size());
    std::vector<std::vector<double>> all_mag_y_samples(config.species.size());
    std::vector<std::vector<double>> all_mag_z_samples(config.species.size());
    if (config.output.output_onsite_magnetization) {
        for (size_t i = 0; i < config.species.size(); i++) {
            all_mag_x_samples[i] = mpi_accumulator.gather_samples(measurement_data.mag_x_samples[i]);
            all_mag_y_samples[i] = mpi_accumulator.gather_samples(measurement_data.mag_y_samples[i]);
            all_mag_z_samples[i] = mpi_accumulator.gather_samples(measurement_data.mag_z_samples[i]);
        }
    }
    
    // Gather correlations
    std::vector<std::vector<double>> all_correlation_samples(config.species.size());
    if (config.output.output_correlations) {
        for (size_t i = 0; i < config.species.size(); i++) {
            all_correlation_samples[i] = mpi_accumulator.gather_samples(measurement_data.correlation_samples[i]);
        }
    }
    
    // Gather acceptance rates
    std::vector<double> all_acceptance_samples = mpi_accumulator.gather_samples(measurement_data.acceptance_samples);
    
    auto comm_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> comm_elapsed = comm_end - comm_start;
    results.timings.communication_time = comm_elapsed.count();
    
    // Barrier with timing to measure load imbalance
    double barrier_wait = 0.0;
    if (config.diagnostics.enable_profiling) {
        barrier_wait = mpi_env.barrier_with_timing();
    } else {
        mpi_env.barrier();
    }
    results.timings.barrier_wait_time = barrier_wait;
    
    // Only rank 0 computes final statistics
    if (rank == 0) {
        int total_samples = all_energy_samples.size();
        
        // Compute mean and standard deviation
        auto compute_stats = [](const std::vector<double>& samples) -> std::pair<double, double> {
            if (samples.empty()) return {0.0, 0.0};
            double sum = 0.0;
            for (double val : samples) sum += val;
            double mean = sum / samples.size();
            double sum_sq_diff = 0.0;
            for (double val : samples) {
                double diff = val - mean;
                sum_sq_diff += diff * diff;
            }
            double stddev = std::sqrt(sum_sq_diff / samples.size());
            return {mean, stddev};
        };
        
        // Energy statistics
        auto [avg_energy, stddev_energy] = compute_stats(all_energy_samples);
        results.avg_energy = avg_energy;
        results.stddev_energy = stddev_energy;
        
        // Magnetization statistics
        auto [avg_magnetization, stddev_magnetization] = compute_stats(all_magnetization_samples);
        results.avg_magnetization = avg_magnetization;
        results.stddev_magnetization = stddev_magnetization;
        
        // Specific heat and susceptibility
        double avg_energy_sq = 0.0;
        for (double e : all_energy_samples) avg_energy_sq += e * e;
        avg_energy_sq /= total_samples;
        double avg_energy_sq_per_spin = avg_energy_sq / (total_spins * total_spins);
        double avg_energy_per_spin = avg_energy / total_spins;
        
        double avg_magnetization_sq = 0.0;
        for (double m : all_magnetization_samples) avg_magnetization_sq += m * m;
        avg_magnetization_sq /= total_samples;
        double avg_magnetization_sq_per_spin = avg_magnetization_sq / (total_spins * total_spins);
        double avg_magnetization_per_spin = avg_magnetization / total_spins;
        
        results.specific_heat = (avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (T * T);
        results.susceptibility = (avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / T;
        
        // Compute stddev for specific heat and susceptibility
        double stddev_energy_per_spin = stddev_energy / total_spins;
        double stddev_magnetization_per_spin = stddev_magnetization / total_spins;
        results.stddev_specific_heat = 2.0 * stddev_energy_per_spin * std::sqrt(avg_energy_sq_per_spin - avg_energy_per_spin * avg_energy_per_spin) / (T * T);
        results.stddev_susceptibility = 2.0 * stddev_magnetization_per_spin * std::sqrt(avg_magnetization_sq_per_spin - avg_magnetization_per_spin * avg_magnetization_per_spin) / T;
        
        // Acceptance rate statistics
        auto [avg_accept_rate, stddev_accept_rate] = compute_stats(all_acceptance_samples);
        results.avg_accept_rate = avg_accept_rate;
        results.stddev_accept_rate = stddev_accept_rate;
        
        // Per-spin magnetization vector statistics
        results.avg_mag_vectors.resize(config.species.size());
        results.stddev_mag_vectors.resize(config.species.size());
        if (config.output.output_onsite_magnetization) {
            for (size_t i = 0; i < config.species.size(); i++) {
                auto [mx_mean, mx_std] = compute_stats(all_mag_x_samples[i]);
                auto [my_mean, my_std] = compute_stats(all_mag_y_samples[i]);
                auto [mz_mean, mz_std] = compute_stats(all_mag_z_samples[i]);
                results.avg_mag_vectors[i] = spin3d(mx_mean, my_mean, mz_mean);
                results.stddev_mag_vectors[i] = spin3d(mx_std, my_std, mz_std);
            }
        }
        
        // Correlation statistics
        results.avg_correlations.resize(config.species.size());
        results.stddev_correlations.resize(config.species.size());
        if (config.output.output_correlations) {
            for (size_t i = 0; i < config.species.size(); i++) {
                auto [corr_mean, corr_std] = compute_stats(all_correlation_samples[i]);
                results.avg_correlations[i] = corr_mean;
                results.stddev_correlations[i] = corr_std;
            }
        }
        
        // Autocorrelation estimates if enabled
        if (config.diagnostics.estimate_autocorrelation && measurement_data.energy_series.size() > 2) {
            double rho_energy = estimate_autocorrelation(measurement_data.energy_series);
            double rho_mag = estimate_autocorrelation(measurement_data.magnetization_series);
            double tau_energy = estimate_autocorrelation_time(rho_energy);
            double tau_mag = estimate_autocorrelation_time(rho_mag);
            
            std::cout << "  Autocorrelation ρ(1): Energy=" << std::fixed << std::setprecision(4) << rho_energy
                     << ", Magnetization=" << rho_mag << std::endl;
            std::cout << "  Estimated τ_auto: Energy≈" << std::fixed << std::setprecision(2) << tau_energy
                     << " sweeps, Magnetization≈" << tau_mag << " sweeps" << std::endl;
        }
    }
    
    // Store temperature timing
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_end - t_start;
    results.timings.total_time = t_elapsed.count();
    
    // Print timing summary if profiling enabled
    if (config.diagnostics.enable_profiling && rank == 0) {
        print_temperature_timing(results.timings);
    }
    
    // Synchronize before returning
    mpi_env.barrier();
    
    return results;
}

/**
 * Run single temperature mode
 */
void run_single_temperature(const IO::SimulationConfig& config,
                           MPIEnvironment& mpi_env,
                           MPIAccumulator& mpi_accumulator) {
    int rank = mpi_env.get_rank();
    
    if (rank == 0) {
        IO::print_section_separator("SINGLE TEMPERATURE SIMULATION");
        std::cout << "Temperature: T = " << config.temperature.value << std::endl;
        std::cout << "MPI walkers: " << mpi_env.get_num_ranks() << std::endl;
    }
    
    // Create simulation objects from configuration
    UnitCell unit_cell = create_unit_cell_from_config(config.species);
    CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, config.lattice_size);
    std::optional<KK_Matrix> kk_matrix = create_kk_matrix_from_config(config.kk_couplings, unit_cell, config.lattice_size);
    
    // Setup diagnostics
    std::string dump_dir = config.output.directory + "/dumps";
    if (rank == 0 && (config.diagnostics.enable_config_dump || config.diagnostics.enable_observable_evolution)) {
        IO::create_directory(config.output.directory);
        IO::create_directory(dump_dir);
    }
    mpi_env.barrier();
    
    // Run simulation at single temperature
    TemperatureResults results = run_temperature_point(
        config.temperature.value,
        config, unit_cell, couplings, kk_matrix,
        mpi_env, mpi_accumulator,
        true,  // first_temperature
        dump_dir
    );
    
    // Output results (only rank 0)
    if (rank == 0) {
        IO::print_observables_formatted(
            results.T, results.total_spins,
            results.avg_energy, results.stddev_energy,
            results.avg_magnetization, results.stddev_magnetization,
            results.specific_heat, results.stddev_specific_heat,
            results.susceptibility, results.stddev_susceptibility,
            results.avg_accept_rate, results.stddev_accept_rate,
            config.species,
            results.avg_mag_vectors, results.stddev_mag_vectors,
            results.avg_correlations, results.stddev_correlations,
            config.output.output_onsite_magnetization,
            config.output.output_correlations
        );
    }
}

/**
 * Run temperature scan mode
 */
void run_temperature_scan(const IO::SimulationConfig& config,
                          MPIEnvironment& mpi_env,
                          MPIAccumulator& mpi_accumulator) {
    int rank = mpi_env.get_rank();
    int num_ranks = mpi_env.get_num_ranks();
    
    // Divide measurement steps among ranks
    int measurement_steps_per_rank = config.monte_carlo.measurement_steps / num_ranks;
    if (measurement_steps_per_rank == 0) {
        if (rank == 0) {
            std::cerr << "ERROR: measurement_steps (" << config.monte_carlo.measurement_steps 
                      << ") is less than num_ranks (" << num_ranks << ")" << std::endl;
            std::cerr << "Please increase measurement_steps or reduce number of MPI ranks." << std::endl;
        }
        return;
    }
    
    if (rank == 0) {
        IO::print_section_separator("INITIALIZATION");
        std::cout << "Running MPI-parallel temperature scan with " << num_ranks << " walkers" << std::endl;
        std::cout << "Temperature range: " << config.temperature.max_temp << " to " 
                  << config.temperature.min_temp << " (step: " << config.temperature.temp_step << ")" << std::endl;
        std::cout << "Measurement steps per rank: " << measurement_steps_per_rank 
                  << " (total: " << measurement_steps_per_rank * num_ranks << ")" << std::endl;
    }
    
    // Create simulation objects from configuration
    UnitCell unit_cell = create_unit_cell_from_config(config.species);
    CouplingMatrix couplings = create_couplings_from_config(config.couplings, config.species, config.lattice_size);
    std::optional<KK_Matrix> kk_matrix = create_kk_matrix_from_config(config.kk_couplings, unit_cell, config.lattice_size);
    
    // Setup output files (rank 0 only)
    std::string output_file, stddev_file;
    std::ofstream outfile, stddev_outfile;
    
    if (rank == 0) {
        output_file = config.output.directory + "/" + config.output.base_name + "_observables.out";
        stddev_file = config.output.directory + "/" + config.output.base_name + "_observables_stddev.out";
        
        outfile.open(output_file);
        stddev_outfile.open(stddev_file);
        
        // Write headers
        auto write_header = [&](std::ofstream& file, const std::string& desc) {
            file << "# Monte Carlo simulation " << desc << " (MPI parallel, " << num_ranks << " walkers)" << std::endl;
            file << "# System: ";
            for (const auto& species : config.species) {
                file << species.name << "(" << (species.spin_type == SpinType::ISING ? "Ising" : "Heisenberg") << ") ";
            }
            file << std::endl;
            file << "# Lattice: " << config.lattice_size << "³" << std::endl;
        };
        
        write_header(outfile, "results (mean values)");
        write_header(stddev_outfile, "standard deviations");
        
        // Build column header
        auto write_column_header = [&](std::ofstream& file) {
            file << "# Columns: T Energy/spin";
            if (config.output.output_energy_total) {
                file << " Energy_total";
            }
            file << " Magnetization SpecificHeat Susceptibility AcceptanceRate";
            
            if (config.output.output_onsite_magnetization) {
                for (const auto& sp : config.species) {
                    if (sp.spin_type == SpinType::ISING) {
                        file << " M[" << sp.name << "]";
                    } else {
                        file << " Mx[" << sp.name << "] My[" << sp.name << "] Mz[" << sp.name << "]";
                    }
                }
            }
            
            if (config.output.output_correlations) {
                std::string first_ising_name = "", first_heis_name = "";
                for (const auto& sp : config.species) {
                    if (sp.spin_type == SpinType::ISING && first_ising_name.empty()) {
                        first_ising_name = sp.name;
                    }
                    if (sp.spin_type == SpinType::HEISENBERG && first_heis_name.empty()) {
                        first_heis_name = sp.name;
                    }
                }
                for (size_t i = 0; i < config.species.size(); i++) {
                    if (config.species[i].spin_type == SpinType::ISING) {
                        file << " <" << first_ising_name << "*" << config.species[i].name << ">";
                    } else {
                        file << " <" << first_heis_name << "·" << config.species[i].name << ">";
                    }
                }
            }
            file << std::endl;
        };
        
        write_column_header(outfile);
        write_column_header(stddev_outfile);
        
        outfile << std::fixed << std::setprecision(8);
        stddev_outfile << std::fixed << std::setprecision(8);
    }
    
    // Setup diagnostics
    std::string dump_dir = config.output.directory + "/dumps";
    if (rank == 0 && (config.diagnostics.enable_config_dump || config.diagnostics.enable_observable_evolution)) {
        IO::create_directory(config.output.directory);
        IO::create_directory(dump_dir);
        std::cout << "  Dump directory created: " << dump_dir << std::endl;
    }
    mpi_env.barrier();
    
    // Profiling
    std::vector<TemperatureTimings> all_timings;
    
    if (rank == 0) {
        IO::print_section_separator("TEMPERATURE SCAN");
    }
    
    // Temperature scan loop
    bool first_temperature = true;
    for (double T = config.temperature.max_temp; T >= config.temperature.min_temp; T -= config.temperature.temp_step) {
        if (rank == 0) {
            std::cout << "\nT = " << std::fixed << std::setprecision(2) << T << std::endl;
        }
        
        // Run simulation at this temperature
        TemperatureResults results = run_temperature_point(
            T, config, unit_cell, couplings, kk_matrix,
            mpi_env, mpi_accumulator,
            first_temperature,
            dump_dir
        );
        
        first_temperature = false;
        
        // Output results (rank 0 only)
        if (rank == 0) {
            // Console output - formatted
            IO::print_observables_formatted(
                results.T, results.total_spins,
                results.avg_energy, results.stddev_energy,
                results.avg_magnetization, results.stddev_magnetization,
                results.specific_heat, results.stddev_specific_heat,
                results.susceptibility, results.stddev_susceptibility,
                results.avg_accept_rate, results.stddev_accept_rate,
                config.species,
                results.avg_mag_vectors, results.stddev_mag_vectors,
                results.avg_correlations, results.stddev_correlations,
                config.output.output_onsite_magnetization,
                config.output.output_correlations
            );
            
            // File output - compact format
            double avg_energy_per_spin = results.avg_energy / results.total_spins;
            double stddev_energy_per_spin = results.stddev_energy / results.total_spins;
            double avg_magnetization_per_spin = results.avg_magnetization / results.total_spins;
            double stddev_magnetization_per_spin = results.stddev_magnetization / results.total_spins;
            
            outfile << std::setw(10) << T << " " << std::setw(13) << avg_energy_per_spin << " ";
            if (config.output.output_energy_total) {
                outfile << std::setw(13) << results.avg_energy << " ";
            }
            outfile << std::setw(13) << avg_magnetization_per_spin << " "
                    << std::setw(13) << results.specific_heat << " "
                    << std::setw(14) << results.susceptibility << " "
                    << std::setw(14) << std::setprecision(6) << results.avg_accept_rate;
            
            if (config.output.output_onsite_magnetization) {
                for (size_t i = 0; i < config.species.size(); i++) {
                    if (config.species[i].spin_type == SpinType::ISING) {
                        outfile << " " << std::setw(14) << std::setprecision(8) << results.avg_mag_vectors[i].z;
                    } else {
                        outfile << " " << std::setw(14) << std::setprecision(8) << results.avg_mag_vectors[i].x
                                << " " << std::setw(14) << std::setprecision(8) << results.avg_mag_vectors[i].y
                                << " " << std::setw(14) << std::setprecision(8) << results.avg_mag_vectors[i].z;
                    }
                }
            }
            
            if (config.output.output_correlations) {
                for (const auto& corr : results.avg_correlations) {
                    outfile << " " << std::setw(14) << std::setprecision(8) << corr;
                }
            }
            outfile << std::endl;
            
            // Stddev file
            stddev_outfile << std::setw(10) << T << " " << std::setw(13) << stddev_energy_per_spin << " ";
            if (config.output.output_energy_total) {
                stddev_outfile << std::setw(13) << results.stddev_energy << " ";
            }
            stddev_outfile << std::setw(13) << stddev_magnetization_per_spin << " "
                          << std::setw(13) << results.stddev_specific_heat << " "
                          << std::setw(14) << results.stddev_susceptibility << " "
                          << std::setw(14) << std::setprecision(6) << results.stddev_accept_rate;
            
            if (config.output.output_onsite_magnetization) {
                for (size_t i = 0; i < config.species.size(); i++) {
                    if (config.species[i].spin_type == SpinType::ISING) {
                        stddev_outfile << " " << std::setw(14) << std::setprecision(8) << results.stddev_mag_vectors[i].z;
                    } else {
                        stddev_outfile << " " << std::setw(14) << std::setprecision(8) << results.stddev_mag_vectors[i].x
                                      << " " << std::setw(14) << std::setprecision(8) << results.stddev_mag_vectors[i].y
                                      << " " << std::setw(14) << std::setprecision(8) << results.stddev_mag_vectors[i].z;
                    }
                }
            }
            
            if (config.output.output_correlations) {
                for (const auto& corr_std : results.stddev_correlations) {
                    stddev_outfile << " " << std::setw(14) << std::setprecision(8) << corr_std;
                }
            }
            stddev_outfile << std::endl;
        }
        
        // Store timings
        all_timings.push_back(results.timings);
    }
    
    // Print comprehensive profiling report
    if (config.diagnostics.enable_profiling && rank == 0 && all_timings.size() > 0) {
        print_profiling_report(all_timings, all_timings.size());
    }
    
    if (rank == 0) {
        outfile.close();
        stddev_outfile.close();
        std::cout << "\nResults saved to:" << std::endl;
        std::cout << "  Mean values: " << output_file << std::endl;
        std::cout << "  Std dev:     " << stddev_file << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI environment
    MPIEnvironment mpi_env(argc, argv);
    MPIAccumulator mpi_accumulator(mpi_env);
    
    // Redirect stdout to /dev/null for non-master ranks
    std::streambuf* cout_backup = nullptr;
    std::ofstream devnull;
    if (!mpi_env.is_master()) {
        devnull.open("/dev/null");
        cout_backup = std::cout.rdbuf();
        std::cout.rdbuf(devnull.rdbuf());
    }
    
    if (mpi_env.is_master()) {
        IO::print_logo();
        IO::print_section_separator("MPI Monte Carlo Simulation (" + std::to_string(mpi_env.get_num_ranks()) + " ranks)");
    }
    
    // Parse command line arguments
    std::string config_file = "simulation.toml";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    try {
        if (mpi_env.is_master()) {
            std::cout << "Loading configuration from: " << config_file << std::endl;
        }
        
        // All ranks load configuration
        IO::SimulationConfig config = IO::ConfigurationParser::load_configuration(config_file);
        
        // Set global random seed
        seed = config.monte_carlo.seed;
        
        if (mpi_env.is_master()) {
            std::cout << "\nConfiguration summary:" << std::endl;
            std::cout << "  Lattice size: " << config.lattice_size << "³" << std::endl;
            std::cout << "  Species: ";
            for (const auto& species : config.species) {
                std::cout << species.name << "(" << (species.spin_type == SpinType::ISING ? "I" : "H") << ") ";
            }
            std::cout << std::endl;
            std::cout << "  Couplings: " << config.couplings.size() << " exchange interactions" << std::endl;
            std::cout << "  Monte Carlo: " << config.monte_carlo.warmup_steps << " warmup + " 
                      << config.monte_carlo.measurement_steps << " measurement steps" << std::endl;
            std::cout << std::endl;
        }
        
        // Run simulation
        if (config.simulation_type == "single_temperature") {
            run_single_temperature(config, mpi_env, mpi_accumulator);
        } else if (config.simulation_type == "temperature_scan") {
            run_temperature_scan(config, mpi_env, mpi_accumulator);
        } else {
            if (mpi_env.is_master()) {
                std::cerr << "ERROR: Unknown simulation type: " << config.simulation_type << std::endl;
            }
            return 1;
        }
        
    } catch (const IO::ConfigurationError& e) {
        std::cerr << "Configuration error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Restore stdout for non-master ranks
    if (!mpi_env.is_master() && cout_backup) {
        std::cout.rdbuf(cout_backup);
        devnull.close();
    }
    
    if (mpi_env.is_master()) {
        std::cout << "\nSimulation completed successfully!" << std::endl;
    }
    return 0;
}
