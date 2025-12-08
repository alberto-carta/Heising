# Monte Carlo Simulation for Magnetic Systems

A Monte Carlo simulation engine for magnetic systems supporting Ising and Heisenberg spins with multi-atom unit cells and flexible coupling configurations.

## Requirements

- **C++11 compatible compiler** (g++, clang++)
- **Eigen3 library** (linear algebra)
- **toml11 library** (header-only, included in project)
- **Standard math library** (libm)

### Ubuntu/Debian Installation
```bash
sudo apt update
sudo apt install build-essential libeigen3-dev
```

### Other Linux Distributions
- **Fedora/RHEL**: `sudo dnf install gcc-c++ eigen3-devel`
- **Arch**: `sudo pacman -S gcc eigen`
- **macOS**: `brew install gcc eigen` (or use clang)

## Quick Start

### 1. Build the Code
```bash
make
```

This compiles the simulation executable: `build/generic_simulation`

### 2. Run a Simulation

The simulation is configured using TOML files. Two examples are provided:

#### Example 1: Ising Ferromagnet
```bash
./build/generic_simulation examples/ising_ferromagnet/simulation.toml
```

This runs a 3D Ising model with ferromagnetic nearest-neighbor interactions. The simulation:
- Uses an 8×8×8 lattice
- Scans temperatures from 6.0 down to 0.5 (step: 0.2)
- Performs 8,000 warmup steps and 80,000 measurement steps per temperature
- Outputs data to `ising_ferromagnet_ising_system.dat`

#### Example 2: Heisenberg Ferromagnet
```bash
./build/generic_simulation examples/heisenberg_ferromagnet/simulation.toml
```

This runs a 3D Heisenberg model with ferromagnetic interactions. Configuration similar to Ising but with continuous 3D spins instead of discrete ±1 spins.

### 3. Analyze Results

Use the provided Python scripts to plot the results:
```bash
cd ../analysis_tools
python analyze_ferromagnets.py
```

This generates plots showing magnetization, energy, specific heat, and susceptibility as functions of temperature.

## Project Structure

```
monte_carlo/
├── src/                    # Core implementation
│   ├── simulation_engine.cpp
│   ├── spin_operations.cpp
│   └── random.cpp
├── include/                # Header files
│   ├── simulation_engine.h
│   ├── multi_atom.h
│   ├── spin_operations.h
│   ├── spin_types.h
│   └── random.h
├── tests/                  # Unit tests
│   └── unit_tests.cpp
├── benchmarks/             # Performance tests
│   └── performance_test.cpp
├── examples/               # Example programs
│   └── main.cpp
├── build/                  # Compiled objects (created during build)
└── analysis_tools/         # Python analysis scripts
    ├── analyze_mc.py
    ├── analyze_results.py
    └── plots/
```

## Usage Examples

### Single Atom Systems

```cpp
#include "simulation_engine.h"
#include "multi_atom.h"

// Create Ising model
UnitCell ising_cell = create_unit_cell(SpinType::ISING);
CouplingMatrix ising_couplings = create_nn_couplings(1, -1.0);  // Ferromagnetic
MonteCarloSimulation ising_sim(ising_cell, ising_couplings, 8, 2.0);

// Create Heisenberg model  
UnitCell heisenberg_cell = create_unit_cell(SpinType::HEISENBERG);
CouplingMatrix heisenberg_couplings = create_nn_couplings(1, 1.0);  // Antiferromagnetic
MonteCarloSimulation heisenberg_sim(heisenberg_cell, heisenberg_couplings, 8, 2.0);
```

### Multi-Atom Systems

```cpp
// Create 4-atom unit cell
UnitCell multi_cell;
multi_cell.add_atom("Fe1", SpinType::HEISENBERG, 2.0);
multi_cell.add_atom("Fe2", SpinType::HEISENBERG, 2.0);
multi_cell.add_atom("Mn1", SpinType::ISING, 1.0);
multi_cell.add_atom("Mn2", SpinType::ISING, 1.0);

// Flexible coupling matrix
CouplingMatrix multi_couplings;
multi_couplings.initialize(4, 2);  // 4 atoms, max_offset = 2

// Intra-cell couplings
multi_couplings.set_intra_coupling(0, 1, -2.0);  // Fe1-Fe2 strong FM
multi_couplings.set_intra_coupling(2, 3, -1.0);  // Mn1-Mn2 FM
multi_couplings.set_intra_coupling(0, 2, 0.5);   // Fe1-Mn1 weak AFM

// Inter-cell nearest neighbors
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        multi_couplings.set_nn_couplings(i, j, -0.1);  // Weak inter-cell FM
    }
}

MonteCarloSimulation multi_sim(multi_cell, multi_couplings, 6, 1.5);
```

### Running Simulations

```cpp
// Initialize and equilibrate
sim.initialize_lattice();
for (int i = 0; i < 10000; i++) {
    sim.run_monte_carlo_step();  // Warmup
}

// Measure properties
sim.reset_statistics();
for (int i = 0; i < 100000; i++) {
    sim.run_monte_carlo_step();
    
    if (i % 1000 == 0) {
        double energy = sim.get_energy();
        double magnetization = sim.get_absolute_magnetization();
        double acceptance = sim.get_acceptance_rate();
        
        // Process measurements...
    }
}
```

## Build System

### Available Targets
- `make all` - Build everything (main program, tests, benchmarks)
- `make main` - Build main simulation program only
- `make test` - Build and run unit tests
- `make benchmark` - Build and run performance benchmarks  
- `make run` - Build and run interactive simulation
- `make clean` - Remove all build artifacts
- `make check-deps` - Verify required dependencies
- `make help` - Show detailed help

### Build Configurations
- `make debug` - Debug build with symbols and assertions
- `make profile` - Profiling build with gprof support
- `make install-deps` - Install dependencies (Ubuntu/Debian only)

### Command Line Options
```bash
./build/monte_carlo_sim --size 16 --tmax 5.0 --tmin 0.5 --coupling -1.0
```

## Performance Characteristics

### Computational Efficiency
- **Local energy calculation**: ~2-5 μs per call (depends on coupling range)
- **Monte Carlo steps**: 
  - Ising: ~50,000 steps/sec  
  - Heisenberg: ~30,000 steps/sec
  - Multi-atom: ~20,000 steps/sec
- **Memory usage**: Scales as (2×max_offset+1)³ per atom pair

### Memory Optimization
| System Type | Old Fixed | New Dynamic | Reduction |
|------------|-----------|-------------|-----------|
| Nearest neighbors | 2.7 KB | 0.2 KB | 13x |
| Next-nearest neighbors | 2.7 KB | 1.0 KB | 3x |
| Extended range (±3) | 2.7 KB | 2.7 KB | 1x |

## Algorithm Details

### Energy Calculation
- **Ising**: E = -∑ᵢⱼ Jᵢⱼ σᵢ σⱼ 
- **Heisenberg**: E = -∑ᵢⱼ Jᵢⱼ **S**ᵢ · **S**ⱼ
- **Mixed**: Handles Ising-Heisenberg interactions naturally

### Monte Carlo Updates
- **Ising**: Random spin flip (σ → -σ)
- **Heisenberg**: Small random rotation with adjustable angle
- **Metropolis acceptance**: P = min(1, exp(-ΔE/kT))

### Lattice Structure
- **3D cubic lattice** with periodic boundary conditions
- **Multi-atom unit cells** with flexible atom positioning
- **Configurable interaction ranges** up to any offset

## Testing

### Unit Tests (`make test`)
- Multi-atom lattice creation and spin access
- Energy calculation correctness vs. analytical results
- Metropolis algorithm validation
- Mixed spin type handling
- Dynamic coupling matrix scaling

### Performance Tests (`make benchmark`)
- Local energy computation time vs. neighbor count
- Lattice setup time vs. system size  
- Monte Carlo step performance for different spin types
- Memory usage analysis and scaling

## Data Analysis

Python analysis tools are provided in `analysis_tools/`:
- `analyze_mc.py` - Process Monte Carlo output data
- `analyze_results.py` - Generate plots and statistics
- `plots/` - Visualization utilities

## Contributing

1. **Code Style**: Follow existing conventions (PEP 8 for Python, Google C++ style)
2. **Testing**: Add tests for new functionality
3. **Performance**: Profile before and after changes
4. **Documentation**: Update README and inline comments

## License

This project is part of the Heising magnetic system calculations. See project documentation for licensing details.

## References

- Monte Carlo Methods in Statistical Physics (Newman & Barkema)
- Computational Physics of Phase Transitions (Landau & Binder)  
- Eigen3 Documentation: https://eigen.tuxfamily.org/