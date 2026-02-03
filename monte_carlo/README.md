# Heising - Monte Carlo Simulation for Magnetic Systems

```
          _______ _________ _______ _________ _        _______ 
|\     /|(  ____ \\__   __/(  ____ \\__   __/( (    /|(  ____ \
| )   ( || (    \/   ) (   | (    \/   ) (   |  \  ( || (    \/
| (___) || (__       | |   | (_____    | |   |   \ | || |      
|  ___  ||  __)      | |   (_____  )   | |   | (\ \) || | ____ 
| (   ) || (         | |         ) |   | |   | | \   || | \_  )
| )   ( || (____/\___) (___/\____) |___) (___| )  \  || (___) |
|/     \|(_______/\_______/\_______)\_______/|/    )_)(_______)

```

**Version: 0.1.0-alpha**

A high-performance Monte Carlo simulation engine for magnetic systems supporting Ising and Heisenberg spins with multi-atom unit cells, Kugel-Khomskii interactions, and MPI parallelization.

**Features:**
- Support for Ising and Heisenberg spins
- Multi-spin unit cells with flexible coupling matrices
- Kugel-Khomskii (orbital-spin) interactions
- TOML-based configuration system
- MPI parallelization with independent walkers
- Comprehensive diagnostics and profiling
- Statistical analysis with proper error bars

## Status

**Alpha Release** - Core functionality implemented and tested. API may change.

## TODO / Roadmap

* **Enhanced Sampling:**
  - Consider parallel tempering for improved sampling at low temperatures
  - Replica exchange Monte Carlo for phase transition studies
  
* **Phonon Coupling (Exploratory):**
  - Effective phonon treatment using phonon density of states
  - Modified acceptance criterion weighted by phonon DOS D(ΔE)

## Requirements

### Core Dependencies
- **C++17 compatible compiler** (g++ 7+, clang++ 5+)
- **CMake 3.10+** (build system)
- **Eigen3** (linear algebra, header-only)
- **toml11** (TOML parser, header-only)
- **MPI** (OpenMPI or MPICH)

### Ubuntu/Debian Installation
```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libeigen3-dev libopenmpi-dev openmpi-bin libtoml11-dev
```

### Fedora/RHEL
```bash
sudo dnf install gcc-c++ cmake eigen3-devel openmpi-devel
```

### macOS
```bash
brew install cmake eigen open-mpi
```

## Building with CMake

### Quick Start

```bash
# From monte_carlo directory
mkdir build
cd build
cmake ..
make heising       # Build main executable (heising.x)
make tests         # Build test suite
make run_tests     # Build and run all tests
```

### Build Targets

- `make heising.x` or `make heising` - Build main MPI-parallel executable
- `make slab_flip_analyzer.x` - Build domain transformation analyzer
- `make tests` - Build all test executables (unit, io, deterministic MC, acceptance)
- `make run_tests` - Build and run all tests with output
- `make test` - Run tests using CTest
- `make unit_tests` - Build unit tests
- `make io_tests` - Build I/O tests
- `make test_deterministic_mc_moves` - Build deterministic MC validation tests
- `make test_mc_acceptance` - Build MC acceptance ratio tests

### CMake Configuration

CMake will automatically detect all dependencies. If a dependency is missing, you'll see a clear error message with installation instructions.

**Verify configuration:**
```bash
cmake ..
```

You should see:
```
========================================
Heising Configuration Summary
========================================
MPI found:        TRUE
Eigen3 found:     TRUE
Eigen3 location:  /usr/include/eigen3
toml11 location:  /usr/include
========================================
```

## Usage

### Running Simulations

```bash
# Run with MPI (2 walkers)
mpirun -np 2 ./heising.x configuration.toml

# Run single walker (still requires MPI)
./heising.x configuration.toml
```

### Slab Flip Analyzer

The `slab_flip_analyzer.x` tool analyzes domain transformation energetics between two spin patterns, useful for studying phase competition and critical nucleation sizes.

**Build:**
```bash
cd build
make slab_flip_analyzer.x
```

**Usage:**
```bash
./slab_flip_analyzer.x slab_analysis.toml
```

**Configuration (`slab_analysis.toml`):**
```toml
lattice_size = 16
species_file = "species.dat"
couplings_file = "couplings.dat"
kugel_khomskii_file = "kugel_khomskii.dat"

# Pattern format: [Cr1_z, Cr2_z, Cr3_z, Cr4_z, CrA_tau, CrB_tau, CrC_tau, CrD_tau]
pattern1 = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # C-AFM + G-OO
pattern2 = [1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0]  # G-AFM + C-OO

lateral_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
thicknesses = [1, 2, 3, 4, 5, 6, 7, 8]
```

**Output:**
- Energy comparison between full-lattice Pattern 1 and Pattern 2
- Single layer transformation energy
- Critical size scan showing when slab transformation becomes energetically favorable
- Identifies minimum domain size needed for phase nucleation

### Configuration Files

See `examples/` directory for sample configurations:
- `heisenberg_ferromagnet/` - Simple Heisenberg ferromagnet
- `heisenberg_afm/` - Antiferromagnetic Heisenberg system
- `ising_ferromagnet/` - Ising model ferromagnet
- `mixed_ferromagnet/` - Mixed Heisenberg+Ising spins
- `kk_ferromagnet/` - System with Kugel-Khomskii interactions
- `4-atom_cell_*` - Multi-atom unit cell examples (ferro, AFM, frustrated, KK, etc.)
- `example_initialization_from_file/` - Custom pattern initialization

### Basic Configuration

Create a `simulation.toml` file:

```toml
[simulation]
type = "temperature_scan"  # Options: "temperature_scan", "single_temperature"
seed = -12345

[lattice]
size = 8  # 8×8×8 lattice

[monte_carlo]
warmup_steps = 8000
measurement_steps = 80000
sampling_frequency = 100

[temperature]
max = 6.0
min = 0.5
step = 0.2

[output]
base_name = "my_simulation"
directory = "."
output_energy_total = true
output_onsite_magnetization = true
output_correlations = true

[input_files]
species = "species.dat"
couplings = "couplings.dat"

[initialization]
type = "random"  # Options: "random", "custom", "from_file"
# For custom: pattern = [1.0, -1.0, ...]
# For from_file: file = "initialization.dat"
```

## Output

The simulation produces:
- `{base_name}.observables` - Mean values and standard deviations for all observables
- `{base_name}.diagnostics` - Detailed per-walker diagnostics (when enabled)
- `{base_name}.timing` - Profiling information (when enabled)

Observables include:
- Energy per spin
- Total magnetization (and per-spin type)
- Specific heat
- Magnetic susceptibility
- Acceptance rate
- Per-site magnetization vectors (optional)
- Spin-spin correlations (optional)
- Phase classification (C-AFM, G-AFM, etc. for mixed systems)

## Testing

Run the test suite to verify installation:

```bash
cd build
make run_tests  # Build and run all tests
# Or use CTest:
ctest
```

Tests include:
- Unit tests: Basic spin operations, energy calculations
- I/O tests: Configuration parsing, file handling
- Deterministic MC tests: Reproducible move sequences
- Acceptance tests: MC acceptance ratio validation

## Performance Tips

- Use MPI parallelization with multiple walkers (typically 2-8)
- Adjust `sampling_frequency` based on autocorrelation time
- Enable profiling to identify bottlenecks: `enable_profiling = true`
- For large systems, reduce `output_correlations` if not needed

## Troubleshooting

### MPI not found
```bash
sudo apt install libopenmpi-dev openmpi-bin
```

### Eigen3 not found
```bash
sudo apt install libeigen3-dev
```

### toml11 not found
```bash
# Option 1: System package
sudo apt install libtoml11-dev

# Option 2: Manual installation
git clone https://github.com/ToruNiina/toml11.git
sudo cp toml11/toml.hpp /usr/local/include/
```

### Build fails with C++17 errors
Ensure you have a recent compiler:
```bash
g++ --version  # Should be 7.0 or newer
```

## Citation

If you use this code in your research, please cite:

```
Heising Monte Carlo Simulation Package (v0.1.0-alpha)
https://github.com/alberto-carta/Heising
```

## License

[Add license information]

## Authors

Alberto Carta

## Acknowledgments

This project uses:
- [Eigen3](https://eigen.tuxfamily.org/) for linear algebra
- [toml11](https://github.com/ToruNiina/toml11) for TOML parsing
- [OpenMPI](https://www.open-mpi.org/) for parallelization
