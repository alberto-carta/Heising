# Multi-Atom Monte Carlo Simulation

A high-performance Monte Carlo simulation engine for magnetic systems with support for multi-atom unit cells, mixed spin types (Ising and Heisenberg), and flexible coupling configurations.

## Features

- **Multi-Atom Support**: Arbitrary number of atoms per magnetic unit cell
- **Mixed Spin Types**: Ising (discrete ±1) and Heisenberg (continuous 3D) spins
- **Flexible Couplings**: Intra-cell and inter-cell interactions with configurable range
- **Dynamic Memory**: Coupling matrix size scales with actual interaction range
- **High Performance**: Optimized for numerical efficiency with direct array access
- **Temperature Sweeps**: Built-in phase transition studies

## Requirements

### System Dependencies
- **C++11 compatible compiler** (g++, clang++)
- **Eigen3 linear algebra library** 
- **Standard math library** (libm)

### Ubuntu/Debian Installation
```bash
sudo apt update
sudo apt install build-essential libeigen3-dev
```

### Other Linux Distributions
- **Fedora/RHEL**: `sudo dnf install gcc-c++ eigen3-devel`
- **Arch**: `sudo pacman -S gcc eigen`
- **macOS**: `brew install eigen`

## Quick Start

### 1. Check Dependencies
```bash
make check-deps
```

### 2. Build and Test
```bash
make test           # Build and run unit tests
make benchmark      # Build and run performance tests
make all            # Build everything
```

### 3. Run Simulation
```bash
make run            # Interactive menu system
```

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