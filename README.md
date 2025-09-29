# Mean Field Theory Library

This library provides a modular implementation for solving mean field equations in mixed Ising-Heisenberg magnetic systems. It's a refactored version of the original `playaround_mean_field.py` with improved modularity and flexibility.

## Library Structure

```
meanfield/
├── __init__.py          # Main library interface
├── spintypes.py         # Individual spin type implementations (Ising, Heisenberg)
├── fields.py            # Effective field calculation strategies  
├── solvers.py           # Self-consistent equation solvers
├── systems.py           # Complete system definitions
└── visualization.py     # Plotting and analysis tools
```

## Key Components

### 1. Spin Types (`spintypes.py`)
- `IsingSpinType`: Individual classical spins (±1) as used in Ising models
- `HeisenbergSpinType`: Individual quantum spins with arbitrary S
- `brillouin_function`: Numerically stable Brillouin function implementation

### 2. Field Calculation (`fields.py`)
**Simple and flexible** - easy to modify effective field calculations:

- `calculate_effective_field`: Core function implementing h_eff = Σ_j J_ij * m_j
- `FieldCalculator`: Simple wrapper class for repeated calculations

### 3. Solvers (`solvers.py`) 
- `MeanFieldSolver`: Self-consistent iterative solver with convergence checking
- Supports both single temperature and temperature sweep calculations

### 4. System Definition (`systems.py`)
- `SublatticeDef`: Definition of individual sublattices
- `MagneticSystem`: Complete system with sublattices and interactions

### 5. Visualization (`visualization.py`)
- `plot_magnetizations`: Plot magnetization vs temperature (reproduces original plotting)
- `find_critical_temperature`: Find critical points

## Basic Usage

```python
import numpy as np
from meanfield import MagneticSystem, SublatticeDef

# Define sublattices
sublattices = [
    SublatticeDef('ising', initial_direction=+1),
    SublatticeDef('heisenberg', S=0.5, initial_direction=[0, 0, -1])
]

# Define interactions
J_matrix = np.array([[0, +1], [+1, 0]])  # Antiferromagnetic coupling
z_matrix = np.array([[0, 1], [1, 0]])    # Coordination numbers

# Note: Coupling convention - J < 0 = ferromagnetic, J > 0 = antiferromagnetic

# Create system
system = MagneticSystem(sublattices, J_matrix, z_matrix)

# Solve at single temperature
magnetizations, info = system.solve_at_temperature(1.0)

# Solve over temperature range  
temperatures = np.linspace(0.1, 5.0, 50)
mags_vs_T, infos = system.solve_temperature_range(temperatures)
```

## Customizing Field Calculations

The field calculation is kept simple and flexible. You can either:

1. **Modify the core function directly** in `fields.py`:
```python
def calculate_effective_field(coupling_matrix, magnetizations, sublattice_idx):
    # Standard: h_eff = Σ_j J_ij * m_j
    return coupling_matrix[sublattice_idx] @ magnetizations
```

2. **Create custom field functions**:
```python
import numpy as np
from meanfield import FieldCalculator

def my_custom_field(coupling_matrix, magnetizations, sublattice_idx):
    # Your custom calculation - add anisotropy, external fields, etc.
    h_standard = coupling_matrix[sublattice_idx] @ magnetizations
    h_external = np.array([0, 0, 0.1])  # External field example
    return h_standard + h_external

# Use it
field_calc = FieldCalculator(calculate_field_func=my_custom_field)
```

This approach allows easy implementation of:
- External magnetic fields
- Anisotropy terms  
- Different interaction types
- Custom neighbor structures

## Running Examples

The `examples.py` file reproduces all the calculations from the original code:

```bash
python examples.py
```

## Testing

Run the basic tests to verify installation:

```bash
python test_library.py
```

## Coupling Convention

**Important**: This library uses the convention where:
- **J < 0 (negative)** = **ferromagnetic coupling**
- **J > 0 (positive)** = **antiferromagnetic coupling**

This is implemented by using `-J[i,j]` in the effective field calculation: `H_eff[i] = Σ_j z[i,j] * (-J[i,j]) * <m_j>/S_j`

## Design Principles

This library follows the guidelines in `Agents.md`:
- **Numerical Focus**: Emphasis on stability and precision
- **Clean Code**: PEP 8 compliance and vectorized operations
- **Documentation**: NumPy-style docstrings throughout
- **Modularity**: Clear separation of concerns for easy modification
