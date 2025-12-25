# Configuration I/O Test Data

This directory contains test configurations for validating the configuration parsing and simulation setup.

## Test Cases

### 1. `ferromagnet/`
- **Purpose**: Test ferromagnetic (J < 0) coupling loading and energy calculation
- **System**: Single Heisenberg spin, nearest-neighbor FM couplings
- **Validation**: Energy matches theoretical FM ground state

### 2. `antiferromagnet/`
- **Purpose**: Test antiferromagnetic (J > 0) coupling loading
- **System**: Single Heisenberg spin, nearest-neighbor AFM couplings
- **Validation**: AFM ordering has lower energy than FM ordering

### 3. `kk_system/`
- **Purpose**: Test Kugel-Khomskii coupling file parsing and physics
- **System**: 1 Heisenberg + 1 Ising spin at same site
- **Validation**: 
  - KK file correctly loaded
  - KK coupling affects energy as expected (K < 0 favors aligned spins)
  - Flipping one Ising spin increases energy

### 4. `invalid_mismatch/`
- **Purpose**: Test error handling for species-coupling mismatch
- **System**: Species file has only Fe, couplings reference non-existent Ni
- **Validation**: ConfigurationError thrown with clear message

### 5. `invalid_kk/`
- **Purpose**: Test graceful handling of invalid KK configuration (wrong number of spins per site)
- **System**: 3 spins at same site (KK requires exactly 2)
- **Validation**: System runs without crashing, compute_kk_contribution returns 0.0

### 6. `invalid_kk_types/`
- **Purpose**: Test graceful handling of wrong spin types for KK
- **System**: 2 Heisenberg spins at same site (KK requires 1 Heisenberg + 1 Ising)
- **Validation**: Warning issued, system runs without crashing

## Running Tests

```bash
make tests
./build/io_tests
```

All tests should pass with no errors.
