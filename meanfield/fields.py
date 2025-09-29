"""
Effective Field Calculation Strategies

This module provides flexible strategies for calculating effective magnetic fields
from sublattice magnetizations. The base class allows easy customization of
field calculation methods for different physical scenarios.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import numpy as np


class BaseFieldCalculator(ABC):
    """
    Abstract base class for effective field calculations.
    
    This class defines the interface for calculating effective magnetic fields
    acting on each sublattice due to interactions with other sublattices.
    Subclass this to implement custom field calculation strategies.
    
    The flexibility of this approach allows for:
    - Different interaction types (exchange, dipolar, anisotropy, etc.)
    - Custom geometries and neighbor structures
    - External field contributions
    - Non-linear field dependencies
    """
    
    @abstractmethod
    def calculate_effective_field(self,
                                sublattice_index: int,
                                magnetizations: List[Union[float, np.ndarray]],
                                **kwargs) -> Union[float, np.ndarray]:
        """
        Calculate the effective field acting on a specific sublattice.
        
        Parameters
        ----------
        sublattice_index : int
            Index of the target sublattice
        magnetizations : List[Union[float, np.ndarray]]
            Current magnetizations of all sublattices
            - Ising: float values
            - Heisenberg: np.ndarray of shape (3,)
        **kwargs
            Additional parameters for field calculation
            
        Returns
        -------
        Union[float, np.ndarray]
            Effective field for the target sublattice
            - For Ising target: scalar field
            - For Heisenberg target: vector field [Hx, Hy, Hz]
        """
        pass
    
    def add_external_field(self, 
                          sublattice_index: int, 
                          external_field: Union[float, np.ndarray]) -> None:
        """
        Add an external magnetic field to a sublattice.
        
        Parameters
        ----------
        sublattice_index : int
            Index of the sublattice
        external_field : Union[float, np.ndarray]
            External field to add
        """
        pass


class StandardFieldCalculator(BaseFieldCalculator):
    """
    Standard mean-field calculator using coupling and coordination matrices.
    
    This implements the standard mean-field approximation where the effective
    field on sublattice i is:
    H_eff[i] = Σ_j z[i,j] * J[i,j] * <m_j> / S_j
    
    Parameters
    ----------
    coupling_matrix : np.ndarray
        Coupling strength matrix J[i,j] between sublattices
    coordination_matrix : np.ndarray
        Number of neighbors z[i,j] between sublattices
    sublattice_types : List[str]
        List of sublattice types ('ising' or 'heisenberg')
    sublattice_params : List[Dict[str, Any]]
        List of sublattice parameters (must contain 'S' for Heisenberg)
    external_fields : Dict[int, Union[float, np.ndarray]], optional
        External fields for specific sublattices
        
    Examples
    --------
    >>> J = np.array([[ 0, -1], [-1,  0]])  # Antiferromagnetic coupling
    >>> z = np.array([[0,  2], [ 2,  0]])   # 2 neighbors each
    >>> types = ['ising', 'heisenberg']
    >>> params = [{'model': 'ising'}, {'model': 'heisenberg', 'S': 0.5}]
    >>> calculator = StandardFieldCalculator(J, z, types, params)
    """
    
    def __init__(self,
                 coupling_matrix: np.ndarray,
                 coordination_matrix: np.ndarray,
                 sublattice_types: List[str],
                 sublattice_params: List[Dict[str, Any]],
                 external_fields: Dict[int, Union[float, np.ndarray]] = None):
        
        self.J_matrix = np.array(coupling_matrix)
        self.z_matrix = np.array(coordination_matrix)
        self.sublattice_types = sublattice_types
        self.sublattice_params = sublattice_params
        self.external_fields = external_fields or {}
        
        # Validate matrix dimensions
        n_sublattices = len(sublattice_types)
        if (self.J_matrix.shape != (n_sublattices, n_sublattices) or
            self.z_matrix.shape != (n_sublattices, n_sublattices)):
            raise ValueError("Matrix dimensions must match number of sublattices")
    
    def calculate_effective_field(self,
                                sublattice_index: int,
                                magnetizations: List[Union[float, np.ndarray]],
                                **kwargs) -> Union[float, np.ndarray]:
        """
        Calculate effective field using standard mean-field theory.
        
        The effective field includes:
        1. Exchange interactions with other sublattices
        2. External fields (if specified)
        
        Parameters
        ----------
        sublattice_index : int
            Target sublattice index
        magnetizations : List[Union[float, np.ndarray]]
            Current magnetizations of all sublattices
            
        Returns
        -------
        Union[float, np.ndarray]
            Effective field for target sublattice
        """
        i = sublattice_index
        target_type = self.sublattice_types[i]
        
        # Initialize effective field based on target type
        if target_type == 'heisenberg':
            h_eff = np.array([0.0, 0.0, 0.0])
        else:  # ising
            h_eff = 0.0
        
        # Sum contributions from all other sublattices
        for j, (source_type, mag_j) in enumerate(zip(self.sublattice_types, magnetizations)):
            if i == j or abs(self.J_matrix[i, j]) < 1e-12:
                continue  # Skip self-interaction or zero coupling
            
            # Get normalized magnetization from source sublattice
            if source_type == 'heisenberg':
                S_j = self.sublattice_params[j]['S']
                normalized_mag_j = mag_j / S_j
            else:  # ising
                normalized_mag_j = mag_j
            
            # Calculate coupling contribution
            coupling_strength = self.z_matrix[i, j] * self.J_matrix[i, j]
            contribution = coupling_strength * normalized_mag_j
            
            # Add contribution based on target and source types
            if target_type == 'heisenberg':
                if isinstance(contribution, np.ndarray):
                    h_eff += contribution
                else:
                    # Scalar contribution to vector field (z-component only)
                    h_eff[2] += contribution
            else:  # target is ising
                if isinstance(contribution, np.ndarray):
                    # Vector contribution to scalar field (use z-component)
                    h_eff += contribution[2] if len(contribution) > 2 else np.linalg.norm(contribution)
                else:
                    h_eff += contribution
        
        # Add external field if specified
        if i in self.external_fields:
            ext_field = self.external_fields[i]
            if target_type == 'heisenberg':
                if isinstance(ext_field, np.ndarray):
                    h_eff += ext_field
                else:
                    h_eff[2] += ext_field
            else:  # ising
                if isinstance(ext_field, np.ndarray):
                    h_eff += ext_field[2] if len(ext_field) > 2 else np.linalg.norm(ext_field)
                else:
                    h_eff += ext_field
        
        return h_eff
    
    def add_external_field(self, 
                          sublattice_index: int, 
                          external_field: Union[float, np.ndarray]) -> None:
        """Add or update external field for a sublattice."""
        self.external_fields[sublattice_index] = external_field
    
    def remove_external_field(self, sublattice_index: int) -> None:
        """Remove external field from a sublattice."""
        self.external_fields.pop(sublattice_index, None)
    
    def update_coupling(self, i: int, j: int, new_coupling: float) -> None:
        """Update coupling matrix element J[i,j] = J[j,i] = new_coupling."""
        self.J_matrix[i, j] = new_coupling
        self.J_matrix[j, i] = new_coupling
    
    def update_coordination(self, i: int, j: int, new_coordination: float) -> None:
        """Update coordination matrix element z[i,j]."""
        self.z_matrix[i, j] = new_coordination


