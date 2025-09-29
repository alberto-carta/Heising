"""
Effective Field Calculation

Simple and flexible effective field calculation for mean field theory.
"""

import numpy as np


def calculate_effective_field(sublattice_index, 
                            magnetizations, 
                            coupling_matrix, 
                            coordination_matrix, 
                            sublattice_types, 
                            sublattice_params,
                            external_fields=None):
    """
    Calculate the effective field acting on a specific sublattice.
    
    This is the core function for mean field theory. The effective field on sublattice i is:
    H_eff[i] = Σ_j z[i,j] * (-J[i,j]) * <m_j>/S_j + H_external[i]
    
    Note: We use the convention where negative J = ferromagnetic, positive J = antiferromagnetic
    
    Parameters
    ----------
    sublattice_index : int
        Which sublattice to calculate field for (0, 1, 2, ...)
    magnetizations : list
        Current magnetizations of all sublattices:
        - For Ising: numbers like [0.8, -0.5, ...]
        - For Heisenberg: arrays like [[0,0,0.3], [0.1,0,0.2], ...]
    coupling_matrix : array
        J[i,j] coupling strengths between sublattices
        Convention: J < 0 = ferromagnetic, J > 0 = antiferromagnetic
    coordination_matrix : array  
        z[i,j] number of neighbors between sublattices
    sublattice_types : list of str
        Type of each sublattice: ['ising', 'heisenberg', ...]
    sublattice_params : list of dict
        Parameters for each sublattice, must contain 'S' for Heisenberg
    external_fields : dict, optional
        External fields: {sublattice_index: field_value}
        
    Returns
    -------
    For Ising targets: float
        Effective field strength
    For Heisenberg targets: array [Hx, Hy, Hz]
        Effective field vector
        
    Examples
    --------
    >>> # Simple 2-sublattice system
    >>> mags = [0.8, -0.6]  # Two Ising spins
    >>> J = [[0, 1], [1, 0]]  # Antiferromagnetic (positive coupling)
    >>> z = [[0, 1], [1, 0]]    # Single neighbors
    >>> types = ['ising', 'ising']
    >>> params = [{'model': 'ising'}, {'model': 'ising'}]
    >>> 
    >>> # Field on sublattice 0
    >>> h_eff = calculate_effective_field(0, mags, J, z, types, params)
    >>> # Result: 0.6 (from neighbor with mag = -0.6, coupling = +1 but negated)
    """
    i = sublattice_index
    target_type = sublattice_types[i]
    
    # Initialize effective field based on target type
    if target_type == 'heisenberg':
        h_eff = np.array([0.0, 0.0, 0.0])
    else:  # ising
        h_eff = 0.0
    
    # Sum contributions from all other sublattices
    for j, (source_type, mag_j) in enumerate(zip(sublattice_types, magnetizations)):
        if i == j or abs(coupling_matrix[i][j]) < 1e-12:
            continue  # Skip self-interaction or zero coupling
        
        # Get normalized magnetization from source sublattice
        if source_type == 'heisenberg':
            S_j = sublattice_params[j]['S']
            normalized_mag_j = mag_j / S_j
        else:  # ising
            normalized_mag_j = mag_j
        
        # Calculate coupling contribution
        # Note: We use negative coupling to flip the convention:
        # negative J = ferromagnetic, positive J = antiferromagnetic
        coupling_strength = coordination_matrix[i][j] * (-coupling_matrix[i][j])
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
    if external_fields and i in external_fields:
        ext_field = external_fields[i]
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


class FieldCalculator:
    """
    Simple wrapper class for field calculations.
    
    This is just a convenient way to store system parameters and calculate fields.
    You can also use the calculate_effective_field function directly.
    
    Parameters
    ----------
    coupling_matrix : array
        J[i,j] coupling strengths between sublattices
        Convention: J < 0 = ferromagnetic, J > 0 = antiferromagnetic
    coordination_matrix : array
        z[i,j] number of neighbors between sublattices  
    sublattice_types : list of str
        ['ising', 'heisenberg', ...] for each sublattice
    sublattice_params : list of dict
        Parameters for each sublattice (must have 'S' for Heisenberg)
    external_fields : dict, optional
        {sublattice_index: field} for external fields
        
    Examples
    --------
    >>> # Create calculator for 2-spin system
    >>> J = [[0, 1], [1, 0]]  # Antiferromagnetic (positive coupling)
    >>> z = [[0, 1], [1, 0]]    # Single neighbors
    >>> types = ['ising', 'ising']
    >>> params = [{'model': 'ising'}, {'model': 'ising'}]  
    >>> calc = FieldCalculator(J, z, types, params)
    >>> 
    >>> # Calculate field
    >>> mags = [0.8, -0.6]
    >>> h_eff = calc.calculate_field(0, mags)
    """
    
    def __init__(self, coupling_matrix, coordination_matrix, sublattice_types, 
                 sublattice_params, external_fields=None):
        
        self.coupling_matrix = coupling_matrix
        self.coordination_matrix = coordination_matrix  
        self.sublattice_types = sublattice_types
        self.sublattice_params = sublattice_params
        self.external_fields = external_fields or {}
    
    def calculate_field(self, sublattice_index, magnetizations):
        """Calculate effective field for a sublattice."""
        return calculate_effective_field(
            sublattice_index, magnetizations,
            self.coupling_matrix, self.coordination_matrix,
            self.sublattice_types, self.sublattice_params,
            self.external_fields
        )
    
    def add_external_field(self, sublattice_index, external_field):
        """Add external field to a sublattice."""
        self.external_fields[sublattice_index] = external_field
    
    def remove_external_field(self, sublattice_index):
        """Remove external field from a sublattice."""
        self.external_fields.pop(sublattice_index, None)


