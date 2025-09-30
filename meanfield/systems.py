"""
Magnetic System Definitions

This module provides classes to define complete magnetic systems
with sublattices, interactions, and solution methods.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from .spintypes import IsingSpinType, HeisenbergSpinType
from .fields import FieldCalculator
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
        Convention: J < 0 = ferromagnetic, J > 0 = antiferromagnetic
    coordination_matrix : np.ndarray
        Coordination matrix z[i,j] between sublattices
    field_method : str, optional
        Field calculation method: 'standard' or 'kugel_khomskii', by default 'standard'
    kugel_khomskii_coupling_matrix : np.ndarray, optional
        K[i,j] coupling matrix for Kugel-Khomskii method between atomic sites
        Required when field_method='kugel_khomskii'
    max_iterations : int, optional
        Maximum number of solver iterations, by default 500
        
    Examples
    --------
    >>> # Standard system
    >>> sublattices = [
    ...     SublatticeDef('ising', initial_direction=1),
    ...     SublatticeDef('heisenberg', S=0.5, initial_direction=[0,0,-1])
    ... ]
    >>> J = np.array([[0, -1], [-1, 0]])
    >>> z = np.array([[0, 2], [2, 0]])
    >>> system = MagneticSystem(sublattices, J, z)
    >>> 
    >>> # Kugel-Khomskii system
    >>> K = np.array([[0, 0.5], [0.5, 0]])  # Site-based coupling
    >>> system = MagneticSystem(sublattices, J, z, 
    ...                        field_method='kugel_khomskii',
    ...                        kugel_khomskii_coupling_matrix=K)
    """
    
    def __init__(self,
                 sublattice_defs: List[SublatticeDef],
                 coupling_matrix: np.ndarray,
                 coordination_matrix: np.ndarray,
                 field_method: str = 'standard',
                 kugel_khomskii_coupling_matrix: Optional[np.ndarray] = None,
                 max_iterations: int = 500):
        
        self.sublattice_defs = sublattice_defs
        self.coupling_matrix = np.array(coupling_matrix)
        self.coordination_matrix = np.array(coordination_matrix)
        self.field_method = field_method
        self.kugel_khomskii_coupling_matrix = kugel_khomskii_coupling_matrix
        self.max_iterations = max_iterations
        
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
        
        # Create field calculator with appropriate method
        self.field_calculator = FieldCalculator(
            coupling_matrix=self.coupling_matrix,
            coordination_matrix=self.coordination_matrix,
            sublattice_types=self.sublattice_types,
            sublattice_params=self.sublattice_params,
            field_method=self.field_method,
            kugel_khomskii_coupling_matrix=self.kugel_khomskii_coupling_matrix
        )
        
        # Create solver with specified max iterations
        self.solver = MeanFieldSolver(self.spin_types, self.field_calculator, max_iterations=self.max_iterations)
    
    def solve_at_temperature(self, temperature: float, **kwargs):
        """
        Solve system at given temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature in energy units
        **kwargs
            Additional solver parameters (e.g., rattle_iterations, rattle_strength)
            
        Returns
        -------
        Tuple[List[Union[float, np.ndarray]], Dict[str, Any]]
            (magnetizations, convergence_info)
        """
        return self.solver.solve_at_temperature(temperature, **kwargs)
    
    def solve_temperature_range(self, temperatures: np.ndarray, **kwargs):
        """
        Solve system over temperature range.
        
        Parameters
        ----------
        temperatures : np.ndarray
            Array of temperatures
        **kwargs
            Additional solver parameters (e.g., rattle_iterations, rattle_strength, reverse_order)
            
        Returns
        -------
        Tuple[np.ndarray, List[Dict[str, Any]]]
            (magnetizations_vs_T, convergence_infos)
        """
        return self.solver.solve_temperature_sweep(temperatures, **kwargs)