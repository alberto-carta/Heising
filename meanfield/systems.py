"""
Magnetic System Definitions

This module provides classes to define complete magnetic systems
with sublattices, interactions, and solution methods.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from .spintypes import IsingSpinType, HeisenbergSpinType
from .fields import StandardFieldCalculator
from .solvers import MeanFieldSolver


class SublatticeDef:
    """
    Definition of a single sublattice.
    
    Parameters
    ----------
    model_type : str
        Type of model ('ising' or 'heisenberg')
    **kwargs
        Parameters for the model (e.g., S for Heisenberg, initial_direction)
        
    Examples
    --------
    >>> ising_sub = SublatticeDef('ising', initial_direction=1)
    >>> heisenberg_sub = SublatticeDef('heisenberg', S=0.5, initial_direction=[0,0,-1])
    """
    
    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type.lower()
        self.params = kwargs
        
        if self.model_type not in ['ising', 'heisenberg']:
            raise ValueError("model_type must be 'ising' or 'heisenberg'")


class MagneticSystem:
    """
    Complete magnetic system with multiple sublattices.
    
    This class encapsulates the full system definition including sublattices,
    interactions, and provides methods to solve the mean field equations.
    
    Parameters
    ----------
    sublattice_defs : List[SublatticeDef]
        List of sublattice definitions
    coupling_matrix : np.ndarray
        Coupling matrix J[i,j] between sublattices
    coordination_matrix : np.ndarray
        Coordination matrix z[i,j] between sublattices
        
    Examples
    --------
    >>> sublattices = [
    ...     SublatticeDef('ising', initial_direction=1),
    ...     SublatticeDef('heisenberg', S=0.5, initial_direction=[0,0,-1])
    ... ]
    >>> J = np.array([[0, -1], [-1, 0]])
    >>> z = np.array([[0, 2], [2, 0]])
    >>> system = MagneticSystem(sublattices, J, z)
    """
    
    def __init__(self,
                 sublattice_defs: List[SublatticeDef],
                 coupling_matrix: np.ndarray,
                 coordination_matrix: np.ndarray):
        
        self.sublattice_defs = sublattice_defs
        self.coupling_matrix = np.array(coupling_matrix)
        self.coordination_matrix = np.array(coordination_matrix)
        
        # Create spin types
        self.spin_types = []
        self.sublattice_types = []
        self.sublattice_params = []
        
        for sub_def in sublattice_defs:
            self.sublattice_types.append(sub_def.model_type)
            
            if sub_def.model_type == 'ising':
                initial_dir = sub_def.params.get('initial_direction', 1.0)
                spin_type = IsingSpinType(initial_direction=initial_dir)
                params = {'model': 'ising'}
            elif sub_def.model_type == 'heisenberg':
                S = sub_def.params.get('S', 0.5)
                initial_dir = sub_def.params.get('initial_direction', [0, 0, 1])
                spin_type = HeisenbergSpinType(S=S, initial_direction=initial_dir)
                params = {'model': 'heisenberg', 'S': S}
            
            self.spin_types.append(spin_type)
            self.sublattice_params.append(params)
        
        # Create field calculator
        self.field_calculator = StandardFieldCalculator(
            coupling_matrix=self.coupling_matrix,
            coordination_matrix=self.coordination_matrix,
            sublattice_types=self.sublattice_types,
            sublattice_params=self.sublattice_params
        )
        
        # Create solver
        self.solver = MeanFieldSolver(self.spin_types, self.field_calculator)
    
    def solve_at_temperature(self, temperature: float):
        """
        Solve system at given temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature in energy units
            
        Returns
        -------
        Tuple[List[Union[float, np.ndarray]], Dict[str, Any]]
            (magnetizations, convergence_info)
        """
        return self.solver.solve_at_temperature(temperature)
    
    def solve_temperature_range(self, temperatures: np.ndarray):
        """
        Solve system over temperature range.
        
        Parameters
        ----------
        temperatures : np.ndarray
            Array of temperatures
            
        Returns
        -------
        Tuple[np.ndarray, List[Dict[str, Any]]]
            (magnetizations_vs_T, convergence_infos)
        """
        return self.solver.solve_temperature_sweep(temperatures)