"""
Mean Field Theory Library for Mixed Ising-Heisenberg Systems

This library provides modular components for solving mean field equations
in multi-sublattice magnetic systems with mixed Ising and Heisenberg models.

Main components:
- spintypes: Individual spin type implementations (Ising, Heisenberg)
- fields: Effective field calculation strategies
- solvers: Self-consistent equation solvers
- systems: Complete system definitions and examples
- visualization: Plotting and analysis tools
"""

from .spintypes import IsingSpinType, HeisenbergSpinType
from .fields import BaseFieldCalculator, StandardFieldCalculator
from .solvers import MeanFieldSolver
from .systems import MagneticSystem, SublatticeDef
from .visualization import plot_magnetizations, find_critical_temperature

__version__ = "0.1.0"
__all__ = [
    "IsingSpinType", 
    "HeisenbergSpinType",
    "BaseFieldCalculator", 
    "StandardFieldCalculator",
    "MeanFieldSolver",
    "MagneticSystem", 
    "SublatticeDef",
    "plot_magnetizations", 
    "find_critical_temperature"
]