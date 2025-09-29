#%%
"""
Example Usage of Mean Field Library

This script demonstrates how to use the new library structure to reproduce
the results from the original playaround_mean_field.py file.
"""

import numpy as np
from meanfield import (
    MagneticSystem, 
    SublatticeDef, 
    plot_magnetizations, 
    find_critical_temperature
)



def example_pure_heisenberg():
    """Example 2: Pure Heisenberg system (reproduces original example)."""
    
    # Define sublattices
    sublattice_defs = [
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, 0.1]),
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, 0]),
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, -0.6]),
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, 0])
    ]
    
    # Same coupling and coordination as before
    J_matrix = np.array([
        [ 0.0, -1.0, -1.0,  0.0],
        [-1.0,  0.0,  0.0, -1.0],
        [-1.0,  0.0,  0.0, -1.0],
        [ 0.0, -1.0, -1.0,  0.0]
    ])
    
    z_matrix = np.array([
        [0, 2, 2, 0],
        [2, 0, 0, 2],
        [2, 0, 0, 2],
        [0, 2, 2, 0]
    ])
    
    # Create and solve system
    system = MagneticSystem(sublattice_defs, J_matrix, z_matrix)
    temperatures = np.linspace(0.01, 8.0, 80)
    
    print("Solving pure classical system...")
    magnetizations_array, _ = system.solve_temperature_range(temperatures)
    
    # Extract and plot
    magnetizations_list = []
    for i in range(len(sublattice_defs)):
        mags = [magnetizations_array[j, i] for j in range(len(temperatures))]
        magnetizations_list.append(np.array(mags))
    
    plot_magnetizations(temperatures, magnetizations_list, system.sublattice_params,
                       'Pure Heisenberg 4-Sublattice System (Different S values)')


def example_pure_ising():
    """Example 3: Pure Ising system (reproduces original example)."""
    
    # Define sublattices
    sublattice_defs = [
        SublatticeDef('ising', initial_direction=+1),
        SublatticeDef('ising', initial_direction=-1),
        SublatticeDef('ising', initial_direction=-1),
        SublatticeDef('ising', initial_direction=+1)
    ]
    
    # Same matrices
    J_matrix = np.array([
        [ 0.0, -1.0, -1.0,  0.0],
        [-1.0,  0.0,  0.0, -1.0],
        [-1.0,  0.0,  0.0, -1.0],
        [ 0.0, -1.0, -1.0,  0.0]
    ])
    
    z_matrix = np.array([
        [0, 2, 2, 0],
        [2, 0, 0, 2],
        [2, 0, 0, 2],
        [0, 2, 2, 0]
    ])
    
    # Create and solve system
    system = MagneticSystem(sublattice_defs, J_matrix, z_matrix)
    temperatures = np.linspace(0.01, 8.0, 80)
    
    print("Solving pure Ising 4-sublattice system...")
    magnetizations_array, _ = system.solve_temperature_range(temperatures)
    
    # Extract magnetizations
    magnetizations_list = []
    for i in range(len(sublattice_defs)):
        mags = [magnetizations_array[j, i] for j in range(len(temperatures))]
        magnetizations_list.append(np.array(mags))
    
    plot_magnetizations(temperatures, magnetizations_list, system.sublattice_params,
                       'Pure Ising 4-Sublattice System')
    
    # Find critical temperature
    abs_mag = np.abs(magnetizations_list[0])
    Tc = find_critical_temperature(temperatures, magnetizations_list[0])
    if Tc:
        print(f"Critical temperature (4-sublattice Ising): {Tc:.3f}")
    else:
        print("Critical temperature (4-sublattice Ising): > 8.0")


def example_8_sublattice_separated():
    """Example 4: 8 sublattices - 4 Heisenberg + 4 Ising with no cross-coupling (reproduces original example)."""
    
    print("Solving 8-sublattice system: 4 Heisenberg + 4 Ising with no cross-coupling...")
    
    temperatures = np.linspace(0.01, 5, 100)  # Extended range to see full transitions
    
    # Define sublattice properties: 4 Heisenberg + 4 Ising (separate groups)
    sublattice_defs = [
        # Heisenberg sublattices (0-3) - interact only with each other
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, 1]),   # H0
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, -1]),  # H1
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, -1]),  # H2
        SublatticeDef('heisenberg', S=100.0, initial_direction=[0, 0, 1]),   # H3
        # Ising sublattices (4-7) - interact only with each other
        SublatticeDef('ising', initial_direction=+1),  # I0
        SublatticeDef('ising', initial_direction=-1),  # I1
        SublatticeDef('ising', initial_direction=-1),  # I2
        SublatticeDef('ising', initial_direction=+1)   # I3
    ]
    
    # 8x8 coupling matrix: Heisenberg (0-3) only with Heisenberg, Ising (4-7) only with Ising
    J_matrix_8 = np.array([
        # H0   H1   H2   H3   I0   I1   I2   I3
        [ 0.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # H0: couples to H1, H2
        [-1.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0],  # H1: couples to H0, H3  
        [-1.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0,  0.0],  # H2: couples to H0, H3
        [ 0.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # H3: couples to H1, H2
        [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0, -1.0,  0.0],  # I0: couples to I1, I2
        [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0],  # I1: couples to I0, I3
        [ 0.0,  0.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0],  # I2: couples to I0, I3
        [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0, -1.0,  0.0]   # I3: couples to I1, I2
    ])
    
    # Coordination matrix: same connectivity pattern within each group
    z_matrix_8 = np.array([
        [0, 2, 2, 0, 0, 0, 0, 0],  # H0 has 2 neighbors of H1, 2 of H2
        [2, 0, 0, 2, 0, 0, 0, 0],  # H1 has 2 neighbors of H0, 2 of H3
        [2, 0, 0, 2, 0, 0, 0, 0],  # H2 has 2 neighbors of H0, 2 of H3  
        [0, 2, 2, 0, 0, 0, 0, 0],  # H3 has 2 neighbors of H1, 2 of H2
        [0, 0, 0, 0, 0, 2, 2, 0],  # I0 has 2 neighbors of I1, 2 of I2
        [0, 0, 0, 0, 2, 0, 0, 2],  # I1 has 2 neighbors of I0, 2 of I3
        [0, 0, 0, 0, 2, 0, 0, 2],  # I2 has 2 neighbors of I0, 2 of I3
        [0, 0, 0, 0, 0, 2, 2, 0]   # I3 has 2 neighbors of I1, 2 of I2
    ])
    
    # Create and solve system
    system = MagneticSystem(sublattice_defs, J_matrix_8, z_matrix_8)
    magnetizations_array, _ = system.solve_temperature_range(temperatures)
    
    # Extract magnetizations for plotting
    magnetizations_list = []
    for i in range(len(sublattice_defs)):
        mags = [magnetizations_array[j, i] for j in range(len(temperatures))]
        magnetizations_list.append(np.array(mags))
    
    plot_magnetizations(temperatures, magnetizations_list, system.sublattice_params,
                       '8-Sublattice System: 4 Heisenberg + 4 Ising (No Cross-coupling)')
    
    # Find critical temperature for 8-sublattice Ising part (sublattice 4)
    Tc_8 = find_critical_temperature(temperatures, magnetizations_list[4])
    if Tc_8:
        print(f"Critical temperature (8-sublattice Ising part): {Tc_8:.3f}")
    else:
        print("Critical temperature (8-sublattice Ising part): > 5.0")


if __name__ == '__main__':
    # Run all examples
    example_pure_heisenberg()
    example_pure_ising()
    example_8_sublattice_separated()
# %%
