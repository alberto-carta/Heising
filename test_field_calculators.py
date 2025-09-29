#!/usr/bin/env python3
"""
Test Field Calculators


"""
#%%
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from meanfield.fields import calculate_effective_field, calculate_kugel_khomskii_field


def setup_kugel_khomskii_system_4sites(heisenberg_S=1.0, J_coupling=1.0, K_coupling=0.5):
    """
    Setup a 4-site Kugel-Khomskii system (8 sublattices total).
    
    Convention: 
    - Sublattices 0,1,2,3: Heisenberg spins on sites 0,1,2,3
    - Sublattices 4,5,6,7: Ising spins on sites 0,1,2,3
    """
    # Sublattice types: first half Heisenberg, second half Ising
    sublattice_types = ['heisenberg', 'heisenberg', 'heisenberg', 'heisenberg',
                       'ising', 'ising', 'ising', 'ising']
    
    # Parameters
    sublattice_params = [
        {'S': heisenberg_S}, {'S': heisenberg_S}, {'S': heisenberg_S}, {'S': heisenberg_S},  # H0-H3
        {'model': 'ising'}, {'model': 'ising'}, {'model': 'ising'}, {'model': 'ising'}     # I0-I3
    ]
    # Standard coupling matrix - same pattern as examples (4-sublattice connectivity)
    # Antiferromagnetic coupling (positive J) in square lattice pattern
    J_matrix = np.array([
        # H0   H1   H2   H3   I0   I1   I2   I3
        [ 0.0, +J_coupling, +J_coupling,  0.0,  0.0,  0.0,  0.0,  0.0],  # H0: couples to H1, H2
        [+J_coupling,  0.0,  0.0, +J_coupling,  0.0,  0.0,  0.0,  0.0],  # H1: couples to H0, H3
        [+J_coupling,  0.0,  0.0, +J_coupling,  0.0,  0.0,  0.0,  0.0],  # H2: couples to H0, H3
        [ 0.0, +J_coupling, +J_coupling,  0.0,  0.0,  0.0,  0.0,  0.0],  # H3: couples to H1, H2
        [ 0.0,  0.0,  0.0,  0.0,  0.0, +J_coupling, +J_coupling,  0.0],  # I0: couples to I1, I2
        [ 0.0,  0.0,  0.0,  0.0, +J_coupling,  0.0,  0.0, +J_coupling],  # I1: couples to I0, I3
        [ 0.0,  0.0,  0.0,  0.0, +J_coupling,  0.0,  0.0, +J_coupling],  # I2: couples to I0, I3
        [ 0.0,  0.0,  0.0,  0.0,  0.0, +J_coupling, +J_coupling,  0.0]   # I3: couples to I1, I2
    ])
    
    # Coordination matrix - same pattern as examples
    z_matrix = np.array([
        [0, 2, 2, 0, 0, 0, 0, 0],  # H0 has 2 neighbors of H1, 2 of H2
        [2, 0, 0, 2, 0, 0, 0, 0],  # H1 has 2 neighbors of H0, 2 of H3
        [2, 0, 0, 2, 0, 0, 0, 0],  # H2 has 2 neighbors of H0, 2 of H3
        [0, 2, 2, 0, 0, 0, 0, 0],  # H3 has 2 neighbors of H1, 2 of H2
        [0, 0, 0, 0, 0, 2, 2, 0],  # I0 has 2 neighbors of I1, 2 of I2
        [0, 0, 0, 0, 2, 0, 0, 2],  # I1 has 2 neighbors of I0, 2 of I3
        [0, 0, 0, 0, 2, 0, 0, 2],  # I2 has 2 neighbors of I0, 2 of I3
        [0, 0, 0, 0, 0, 2, 2, 0]   # I3 has 2 neighbors of I1, 2 of I2
    ])
    
    # Kugel-Khomskii coupling matrix - between atomic sites (4x4 for 4 sites)
    # K[site_i, site_j] coupling between sites i and j  
    K_matrix = np.array([
        # Site0  Site1  Site2  Site3
        [ 0.0,  +K_coupling, +K_coupling,  0.0],  # Site 0: couples to sites 1,2
        [+K_coupling,  0.0,  0.0, +K_coupling],  # Site 1: couples to sites 0,3
        [+K_coupling,  0.0,  0.0, +K_coupling],  # Site 2: couples to sites 0,3
        [ 0.0,  +K_coupling, +K_coupling,  0.0]   # Site 3: couples to sites 1,2
    ])
    
    # Initial magnetizations - alternating pattern
    S = heisenberg_S  

    magnetizations = [
        np.array([0.0, 0.0, +1])*S,  # H0: +z direction
        np.array([0.0, 0.0, -1])*S,  # H1: -z direction
        np.array([0.0, 0.0, -1])*S,  # H2: -z direction
        np.array([0.0, 0.0, +1])*S,  # H3: +z direction
        +1,                        # I0: +1 Ising
        -1,                        # I1: -1 Ising
        -1,                        # I2: -1 Ising
        +1                         # I3: +1 Ising
    ]
    
    return {
        'total_sublattices': 8,
        'n_sites': 4,
        'sublattice_types': sublattice_types,
        'sublattice_params': sublattice_params,
        'coupling_matrix': J_matrix,
        'coordination_matrix': z_matrix,
        'kk_coupling_matrix': K_matrix,
        'magnetizations': magnetizations,
        'K_coupling': K_coupling
    }






# %%

# Example: Create a custom system
print("Custom 4-site Kugel-Khomskii system:")
heisemberg_S = 100
S_magnitudes = [heisemberg_S, heisemberg_S, heisemberg_S, heisemberg_S, 1, 1, 1, 1]

system = setup_kugel_khomskii_system_4sites(heisenberg_S=100,
                                                J_coupling=1,
                                                K_coupling=-1)

print(f"Magnetizations:")
for i, (typ, mag) in enumerate(zip(system['sublattice_types'], system['magnetizations'])):
    print(f"  {i}: {typ:10s} {mag/S_magnitudes[i]}")

# Test on middle Heisenberg sublattice
test_idx = 0

h_kk = calculate_kugel_khomskii_field(
    test_idx,
    system['magnetizations'],
    system['coupling_matrix'],
    system['coordination_matrix'],
    system['sublattice_types'],
    system['sublattice_params'],
    kugel_khomskii_coupling_matrix=system['kk_coupling_matrix']
)
# run standard field for comparison
h_std = calculate_effective_field(
    test_idx,
    system['magnetizations'],
    system['coupling_matrix'],
        system['coordination_matrix'],
    system['sublattice_types'], 
    system['sublattice_params']
)
    
print(f"\nKugel-Khomskii field on sublattice {test_idx}: {h_kk}, normal field: {h_std}")
print(f"Difference: {h_kk - h_std}")

# You can add more experiments here...
print("\n" + "-"*50)
# %%
