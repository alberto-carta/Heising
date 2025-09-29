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
**This is the most flexible component** - designed for easy modification of how effective fields are computed:

- `BaseFieldCalculator`: Abstract base class for field calculation strategies
- `StandardFieldCalculator`: Standard mean-field theory using coupling/coordination matrices

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
J_matrix = np.array([[0, -1], [-1, 0]])  # Antiferromagnetic coupling  
z_matrix = np.array([[0, 1], [1, 0]])    # Coordination numbers

# Create system
system = MagneticSystem(sublattices, J_matrix, z_matrix)

# Solve at single temperature
magnetizations, info = system.solve_at_temperature(1.0)

# Solve over temperature range  
temperatures = np.linspace(0.1, 5.0, 50)
mags_vs_T, infos = system.solve_temperature_range(temperatures)
```

## Customizing Field Calculations

The field calculation is designed to be easily modifiable. To create custom field calculation strategies, inherit from `BaseFieldCalculator`:

```python
from meanfield.fields import BaseFieldCalculator

class MyCustomFieldCalculator(BaseFieldCalculator):
    def calculate_effective_field(self, sublattice_index, magnetizations, **kwargs):
        # Your custom field calculation here
        # Return scalar for Ising targets, np.array([x,y,z]) for Heisenberg
        pass
```

This flexibility allows you to easily implement:
- Different interaction types (exchange, dipolar, anisotropy, etc.)
- Custom geometries and neighbor structures  
- External field contributions
- Non-linear field dependencies

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

## Design Principles

This library follows the guidelines in `Agents.md`:
- **Numerical Focus**: Emphasis on stability and precision
- **Clean Code**: PEP 8 compliance and vectorized operations
- **Documentation**: NumPy-style docstrings throughout
- **Modularity**: Clear separation of concerns for easy modification
