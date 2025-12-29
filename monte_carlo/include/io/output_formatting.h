/*
 * Output Formatting Utilities
 * 
 * Functions for formatted console and file output
 */

#ifndef OUTPUT_FORMATTING_H
#define OUTPUT_FORMATTING_H

#include "../spin_types.h"
#include "config_types.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace IO {

/**
 * Print the Heising logo
 */
inline void print_logo() {
    std::cout << "\n";
    std::cout << "          _______ _________ _______ _________ _        _______ \n";
    std::cout << "|\\     /|(  ____ \\\\__   __/(  ____ \\\\__   __/( (    /|(  ____ \\\n";
    std::cout << "| )   ( || (    \\/   ) (   | (    \\/   ) (   |  \\  ( || (    \\/\n";
    std::cout << "| (___) || (__       | |   | (_____    | |   |   \\ | || |      \n";
    std::cout << "|  ___  ||  __)      | |   (_____  )   | |   | (\\ \\) || | ____ \n";
    std::cout << "| (   ) || (         | |         ) |   | |   | | \\   || | \\_  )\n";
    std::cout << "| )   ( || (____/\\___) (___/\\____) |___) (___| )  \\  || (___) |\n";
    std::cout << "|/     \\|(_______/\\_______/\\_______)\\_______/|/    )_)(_______)\n";
    std::cout << "\n";
    std::cout << "        Monte Carlo Simulations for Mixed Magnetic Systems\n";
    std::cout << "        Work by Alberto Carta\n";
    std::cout << "        Alpha Version 0.0.2 \n";
    std::cout << "\n";
}

/**
 * Print a section separator
 */
inline void print_section_separator(const std::string& title = "") {
    std::cout << "\n";
    std::cout << "========================================";
    if (!title.empty()) {
        std::cout << "\n  " << title;
    }
    std::cout << "\n========================================\n";
}

/**
 * Print a subsection separator
 */
inline void print_subsection_separator(const std::string& title = "") {
    std::cout << "\n----------------------------------------\n";
    if (!title.empty()) {
        std::cout << "  " << title << "\n";
        std::cout << "----------------------------------------\n";
    }
}

/**
 * Print observables in a structured, readable format
 * 
 * @param T Temperature
 * @param total_spins Total number of spins
 * @param avg_energy Average total energy
 * @param stddev_energy Standard deviation of total energy
 * @param avg_magnetization Average total magnetization
 * @param stddev_magnetization Standard deviation of total magnetization
 * @param specific_heat Specific heat per spin
 * @param stddev_specific_heat Standard deviation of specific heat
 * @param susceptibility Susceptibility per spin
 * @param stddev_susceptibility Standard deviation of susceptibility
 * @param avg_accept_rate Average acceptance rate
 * @param stddev_accept_rate Standard deviation of acceptance rate
 * @param species List of magnetic species
 * @param avg_mag_vectors Per-spin average magnetization vectors
 * @param stddev_mag_vectors Per-spin standard deviation magnetization vectors
 * @param avg_correlations Average correlations with first spin
 * @param stddev_correlations Standard deviation of correlations
 * @param output_onsite_mag Whether to output per-spin magnetization
 * @param output_correlations Whether to output correlations
 */
inline void print_observables_formatted(
    double T,
    int total_spins,
    double avg_energy,
    double stddev_energy,
    double avg_magnetization,
    double stddev_magnetization,
    double specific_heat,
    double stddev_specific_heat,
    double susceptibility,
    double stddev_susceptibility,
    double avg_accept_rate,
    double stddev_accept_rate,
    const std::vector<MagneticSpecies>& species,
    const std::vector<spin3d>& avg_mag_vectors,
    const std::vector<spin3d>& stddev_mag_vectors,
    const std::vector<double>& avg_correlations,
    const std::vector<double>& stddev_correlations,
    bool output_onsite_mag,
    bool output_correlations)
{
    std::cout << std::fixed;
    
    // Temperature header
    print_subsection_separator("Temperature: T = " + std::to_string(T));
    
    // Global observables
    std::cout << "\n  Global Observables:\n";
    std::cout << "  -------------------\n";
    std::cout << std::setprecision(8);
    std::cout << "    Energy/spin:         " << std::setw(14) << avg_energy / total_spins 
              << " ± " << std::setw(14) << stddev_energy / total_spins << "\n";
    std::cout << "    Total Energy:        " << std::setw(14) << avg_energy 
              << " ± " << std::setw(14) << stddev_energy << "\n";
    std::cout << "    Magnetization:       " << std::setw(14) << avg_magnetization / total_spins 
              << " ± " << std::setw(14) << stddev_magnetization / total_spins << "\n";
    std::cout << "    Specific Heat:       " << std::setw(14) << specific_heat 
              << " ± " << std::setw(14) << stddev_specific_heat << "\n";
    std::cout << "    Susceptibility:      " << std::setw(14) << susceptibility 
              << " ± " << std::setw(14) << stddev_susceptibility << "\n";
    std::cout << std::setprecision(4);
    std::cout << "    Acceptance Rate:     " << std::setw(14) << avg_accept_rate 
              << " ± " << std::setw(14) << stddev_accept_rate << "\n";
    
    // Per-spin magnetization
    if (output_onsite_mag && !avg_mag_vectors.empty()) {
        std::cout << "\n  Per-Spin Magnetization:\n";
        std::cout << "  -----------------------\n";
        std::cout << std::setprecision(8);
        for (size_t i = 0; i < species.size(); i++) {
            std::cout << "    " << std::setw(8) << std::left << species[i].name << std::right;
            if (species[i].spin_type == SpinType::ISING) {
                std::cout << " M:  " << std::setw(14) << avg_mag_vectors[i].z 
                          << " ± " << std::setw(14) << stddev_mag_vectors[i].z << "\n";
            } else {
                std::cout << " Mx: " << std::setw(14) << avg_mag_vectors[i].x 
                          << " ± " << std::setw(14) << stddev_mag_vectors[i].x << "\n";
                std::cout << "             My: " << std::setw(14) << avg_mag_vectors[i].y 
                          << " ± " << std::setw(14) << stddev_mag_vectors[i].y << "\n";
                std::cout << "             Mz: " << std::setw(14) << avg_mag_vectors[i].z 
                          << " ± " << std::setw(14) << stddev_mag_vectors[i].z << "\n";
            }
        }
    }
    
    // Correlations
    if (output_correlations && !avg_correlations.empty()) {
        std::cout << "\n  Inter-Site Correlations:\n";
        std::cout << "  ------------------------\n";
        std::cout << std::setprecision(8);
        
        // Determine first spin of each type for correlation reference
        std::string first_ising_name = "", first_heis_name = "";
        for (const auto& sp : species) {
            if (sp.spin_type == SpinType::ISING && first_ising_name.empty()) {
                first_ising_name = sp.name;
            }
            if (sp.spin_type == SpinType::HEISENBERG && first_heis_name.empty()) {
                first_heis_name = sp.name;
            }
        }
        
        for (size_t i = 0; i < species.size(); i++) {
            if (species[i].spin_type == SpinType::ISING) {
                std::cout << "    <" << first_ising_name << "*" << species[i].name << ">: "
                          << std::setw(14) << avg_correlations[i] 
                          << " ± " << std::setw(14) << stddev_correlations[i] << "\n";
            } else {
                std::cout << "    <" << first_heis_name << "·" << species[i].name << ">: "
                          << std::setw(14) << avg_correlations[i] 
                          << " ± " << std::setw(14) << stddev_correlations[i] << "\n";
            }
        }
    }
    
    std::cout << std::endl;
}

} // namespace IO

#endif // OUTPUT_FORMATTING_H
