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

def calculate_kugel_khomskii_field( sublattice_index, 
                            magnetizations, 
                            coupling_matrix, 
                            coordination_matrix, 
                            sublattice_types, 
                            sublattice_params,
                            kugel_khomskii_coupling_matrix=None,
                            external_fields=None):
    """
    Computes Kugel-Khomskii type effective field for a specific sublattice.
    
    This function implements the Kugel-Khomskii coupling scheme where sublattices
    are organized in pairs: each Heisenberg spin is coupled to a corresponding
    Ising (orbital) spin on the same atomic site.
    
    Parameters
    ----------
    sublattice_index : int
        Index of sublattice to calculate field for
    magnetizations : list
        Current magnetizations of all sublattices
    coupling_matrix : array
        J[i,j] standard exchange coupling matrix
    coordination_matrix : array
        z[i,j] coordination number matrix
    sublattice_types : list
        Types of each sublattice
    sublattice_params : list
        Parameters for each sublattice
    kugel_khomskii_coupling_matrix : array
        K[i,j] Kugel-Khomskii coupling strengths between atomic sites
        Shape: (N_sites, N_sites) where N_sites = N_sublattices/2
    external_fields : dict, optional
        External fields
    
    Convention:
    - First half of sublattices: Heisenberg spins (sites 0, 1, 2, ...)
    - Second half of sublattices: Ising spins (sites 0, 1, 2, ...)
    - Sublattice i (Heisenberg) corresponds to sublattice i+N/2 (Ising)
      where N is the total number of sublattices
    
    Example: For 4 sublattices total:
    - Sublattice 0: Heisenberg spin on atom 0
    - Sublattice 1: Heisenberg spin on atom 1  
    - Sublattice 2: Ising spin on atom 0
    - Sublattice 3: Ising spin on atom 1
    """

    # Split sublattices into Heisenberg and Ising groups
    N = len(sublattice_types)
    if N % 2 != 0:
        raise ValueError("Total number of sublattices must be even for Kugel-Khomskii coupling.")
    half_N = N // 2
    heisenberg_indices = list(range(half_N))
    ising_indices = list(range(half_N, N))

    # first thing is to compute the effective field as usual
    h_eff = calculate_effective_field(sublattice_index,
                            magnetizations,
                            coupling_matrix,
                            coordination_matrix,
                            sublattice_types,
                            sublattice_params,
                            external_fields)   
     
    # we then apply the Kugel-Khomskii coupling
    # this looks like for the Heisenberg sublattices
    #  H_eff_KK = Σ_j z[i,j] * K[i,j] * <m_j>/S_j * <σ_j> * <σ_i>
    # and for the Ising sublattices
    #  H_eff_KK = Σ_j z[i,j] * K[i,j] * <m_j>/S_j * <m_i>/S_i * <σ_j>
    if kugel_khomskii_coupling_matrix is None:
        raise ValueError("kugel_khomskii_coupling_matrix must be provided for Kugel-Khomskii coupling.")
    K_matrix = kugel_khomskii_coupling_matrix
    i = sublattice_index
    target_type = sublattice_types[i]

    match target_type:
        case 'heisenberg':
            # Heisenberg sublattice i corresponds to site i
            site_i = i
            sigma_i = magnetizations[i + half_N]  # Corresponding Ising magnetization
            for jheis, jsing in zip(heisenberg_indices, ising_indices):
                if jheis == i:
                    continue  # Skip self-interaction
                else:
                    site_j = jheis  # Site index for source Heisenberg
                    S_j = sublattice_params[jheis]['S']
                    mag_j = magnetizations[jheis] / S_j  # normalized magnetization of source
                    sigma_j = magnetizations[jsing]  # Ising magnetization (±1)
                    coupling_strength = coordination_matrix[i][jheis] * K_matrix[site_i][site_j]
                    
                    # For Heisenberg target: contribution is vector * scalars = vector
                    contribution = coupling_strength * mag_j * sigma_j * sigma_i
                    h_eff += contribution  # Add full vector contribution
        
        case 'ising':
            # Ising sublattice i corresponds to site i-half_N
            site_i = i - half_N
            mag_i = magnetizations[i - half_N] / sublattice_params[i - half_N]['S']  # Corresponding Heisenberg normalized mag
            for jheis, jsing in zip(heisenberg_indices, ising_indices):
                if jsing == i:
                    continue  # Skip self-interaction
                else:             
                    site_j = jheis  # Site index for source Heisenberg
                    S_j = sublattice_params[jheis]['S']
                    mag_j = magnetizations[jheis] / S_j  # normalized magnetization of source
                    sigma_j = magnetizations[jsing]  # Ising magnetization (±1)
                    coupling_strength = coordination_matrix[i][jsing] * K_matrix[site_i][site_j]
                    
                    # For Ising target: take dot product to get scalar contribution
                    # contribution = K[site_i,site_j] * <m_j> · <m_i> * sigma_j
                    contribution = coupling_strength * np.dot(mag_j, mag_i) * sigma_j
                    h_eff += contribution  # scalar field

        case _:
            raise ValueError(f"Unknown sublattice type: {target_type}")
    
    return h_eff


class FieldCalculator:
    """
    Unified field calculator supporting both standard and Kugel-Khomskii methods.

    
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
    field_method : str, optional
        'standard' or 'kugel_khomskii'. Default is 'standard'
    kugel_khomskii_coupling_matrix : array, optional
        K[i,j] coupling matrix for Kugel-Khomskii method
        Shape: (N_sites, N_sites) where N_sites = N_sublattices/2
    external_fields : dict, optional
        {sublattice_index: field} for external fields
        
    Examples
    --------
    >>> # Standard field calculator
    >>> J = [[0, 1], [1, 0]]
    >>> z = [[0, 1], [1, 0]]
    >>> calc = FieldCalculator(J, z, types, params, field_method='standard')
    >>> 
    >>> # Kugel-Khomskii field calculator
    >>> K = [[0, 0.5], [0.5, 0]]  # Site-based K matrix
    >>> calc = FieldCalculator(J, z, types, params, 
    ...                       field_method='kugel_khomskii',
    ...                       kugel_khomskii_coupling_matrix=K)
    """
    
    def __init__(self, coupling_matrix, coordination_matrix, sublattice_types, 
                 sublattice_params, field_method='standard',
                 kugel_khomskii_coupling_matrix=None, external_fields=None):
        
        self.coupling_matrix = coupling_matrix
        self.coordination_matrix = coordination_matrix  
        self.sublattice_types = sublattice_types
        self.sublattice_params = sublattice_params
        self.field_method = field_method
        self.kugel_khomskii_coupling_matrix = kugel_khomskii_coupling_matrix
        self.external_fields = external_fields or {}
        
        # Validate Kugel-Khomskii setup
        if field_method == 'kugel_khomskii':
            if kugel_khomskii_coupling_matrix is None:
                raise ValueError("kugel_khomskii_coupling_matrix must be provided when field_method='kugel_khomskii'")
            
            N = len(sublattice_types)
            if N % 2 != 0:
                raise ValueError("Total number of sublattices must be even for Kugel-Khomskii coupling")
                
            expected_k_shape = (N // 2, N // 2)
            k_shape = np.array(kugel_khomskii_coupling_matrix).shape
            if k_shape != expected_k_shape:
                raise ValueError(f"kugel_khomskii_coupling_matrix shape {k_shape} doesn't match expected {expected_k_shape}")
    
    def calculate_field(self, sublattice_index, magnetizations):
        """
        Calculate effective field for a sublattice using the configured method.
        
        Parameters
        ----------
        sublattice_index : int
            Index of sublattice to calculate field for
        magnetizations : list
            Current magnetizations of all sublattices
            
        Returns
        -------
        Field value (scalar for Ising, vector for Heisenberg)
        """
        if self.field_method == 'standard':
            return calculate_effective_field(
                sublattice_index, magnetizations,
                self.coupling_matrix, self.coordination_matrix,
                self.sublattice_types, self.sublattice_params,
                self.external_fields
            )
        elif self.field_method == 'kugel_khomskii':
            return calculate_kugel_khomskii_field(
                sublattice_index, magnetizations,
                self.coupling_matrix, self.coordination_matrix,
                self.sublattice_types, self.sublattice_params,
                self.kugel_khomskii_coupling_matrix,
                self.external_fields
            )
        else:
            raise ValueError(f"Unknown field_method: {self.field_method}")
    
    def add_external_field(self, sublattice_index, external_field):
        """Add external field to a sublattice."""
        self.external_fields[sublattice_index] = external_field
    
    def remove_external_field(self, sublattice_index):
        """Remove external field from a sublattice."""
        self.external_fields.pop(sublattice_index, None)


