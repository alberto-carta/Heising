/*
 * Diagnostic Utilities Implementation
 */

#include "../../include/io/diagnostic_utils.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace IO {

void create_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        // Directory doesn't exist, create it
        mkdir(path.c_str(), 0755);
    }
}

void save_configuration(const MonteCarloSimulation& sim, 
                       const std::vector<MagneticSpecies>& species,
                       int lattice_size,
                       ConfigurationSnapshot& snapshot) {
    snapshot.clear();
    
    int total_spins_per_cell = species.size();
    int total_cells = lattice_size * lattice_size * lattice_size;
    
    // Count how many of each type we need
    int ising_count = 0, heisenberg_count = 0;
    for (const auto& spec : species) {
        if (spec.spin_type == SpinType::ISING) ising_count++;
        else heisenberg_count++;
    }
    
    // Pre-allocate storage
    snapshot.ising_spins.reserve(total_cells * ising_count);
    snapshot.heisenberg_spins.reserve(total_cells * heisenberg_count);
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int atom_id = 0; atom_id < total_spins_per_cell; atom_id++) {
                    if (species[atom_id].spin_type == SpinType::ISING) {
                        snapshot.ising_spins.push_back(sim.get_ising_spin(x, y, z, atom_id));
                    } else {
                        spin3d heisenberg_spin = sim.get_heisenberg_spin(x, y, z, atom_id);
                        snapshot.heisenberg_spins.push_back(heisenberg_spin);
                    }
                }
            }
        }
    }
}

void load_configuration(MonteCarloSimulation& sim,
                       const std::vector<MagneticSpecies>& species,
                       int lattice_size,
                       const ConfigurationSnapshot& snapshot) {
    int total_spins_per_cell = species.size();
    
    int ising_index = 0;
    int heisenberg_index = 0;
    
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (int atom_id = 0; atom_id < total_spins_per_cell; atom_id++) {
                    if (species[atom_id].spin_type == SpinType::ISING) {
                        sim.set_ising_spin(x, y, z, atom_id, snapshot.ising_spins[ising_index]);
                        ising_index++;
                    } else {
                        const spin3d& heisenberg_spin = snapshot.heisenberg_spins[heisenberg_index];
                        sim.set_heisenberg_spin(x, y, z, atom_id, heisenberg_spin);
                        heisenberg_index++;
                    }
                }
            }
        }
    }
}

void dump_configuration_to_file(const MonteCarloSimulation& sim,
                                const std::vector<MagneticSpecies>& species,
                                int lattice_size,
                                double temperature,
                                int measurement_step,
                                int rank,
                                const std::string& dump_dir) {
    std::ostringstream filename;
    filename << dump_dir << "/config_rank" << rank 
             << "_T" << std::fixed << std::setprecision(2) << temperature
             << "_meas" << measurement_step << ".dat";
    
    std::ofstream outfile(filename.str());
    if (!outfile.is_open()) {
        std::cerr << "Warning: Could not open dump file: " << filename.str() << std::endl;
        return;
    }
    
    // Write header
    outfile << "# Configuration dump" << std::endl;
    outfile << "# Temperature: " << std::fixed << std::setprecision(4) << temperature << std::endl;
    outfile << "# Measurement step: " << measurement_step << std::endl;
    outfile << "# Rank: " << rank << std::endl;
    outfile << "# Lattice size: " << lattice_size << "x" << lattice_size << "x" << lattice_size << std::endl;
    outfile << "# Species:";
    for (const auto& sp : species) {
        outfile << " " << sp.name;
    }
    outfile << std::endl;
    outfile << "# Format: x y z spin_id spin_name spin_type value(s)" << std::endl;
    outfile << "# For Ising: x y z spin_id name Ising value" << std::endl;
    outfile << "# For Heisenberg: x y z spin_id name Heisenberg sx sy sz" << std::endl;
    outfile << "#" << std::endl;
    
    // Write configuration
    for (int x = 1; x <= lattice_size; x++) {
        for (int y = 1; y <= lattice_size; y++) {
            for (int z = 1; z <= lattice_size; z++) {
                for (size_t spin_id = 0; spin_id < species.size(); spin_id++) {
                    const auto& sp = species[spin_id];
                    outfile << x << " " << y << " " << z << " " << spin_id << " " 
                            << sp.name << " ";
                    
                    if (sp.spin_type == SpinType::ISING) {
                        double val = sim.get_ising_spin(x, y, z, spin_id);
                        outfile << "Ising " << std::fixed << std::setprecision(6) << val << std::endl;
                    } else {
                        spin3d s = sim.get_heisenberg_spin(x, y, z, spin_id);
                        outfile << "Heisenberg " << std::fixed << std::setprecision(6) 
                                << s.x << " " << s.y << " " << s.z << std::endl;
                    }
                }
            }
        }
    }
    
    outfile.close();
}

std::ofstream setup_observable_evolution_file(const std::string& dump_dir,
                                              int rank,
                                              double temperature,
                                              const std::vector<MagneticSpecies>& species) {
    std::ostringstream obs_filename;
    obs_filename << dump_dir << "/observables_rank" << rank 
                << "_T" << std::fixed << std::setprecision(2) << temperature << ".dat";
    
    std::ofstream file(obs_filename.str());
    
    if (file.is_open()) {
        file << "# Observable evolution for T=" << temperature << ", Rank=" << rank << std::endl;
        file << "# measurement_step energy magnetization";
        for (const auto& sp : species) {
            file << " corr_" << sp.name;
        }
        file << " acceptance_rate" << std::endl;
        file << std::fixed << std::setprecision(8);
    }
    
    return file;
}

void write_observable_evolution(std::ofstream& file,
                                int measurement_step,
                                double energy,
                                double magnetization,
                                const std::vector<double>& correlations,
                                double acceptance_rate) {
    if (!file.is_open()) return;
    
    file << measurement_step << " " << energy << " " << magnetization;
    for (const auto& corr : correlations) {
        file << " " << corr;
    }
    file << " " << acceptance_rate << std::endl;
}

} // namespace IO
