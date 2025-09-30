# %%
"""
Example Usage of Mean Field Library

This script demonstrates how to use the new library structure to reproduce
the results from the original playaround_mean_field.py file.

Each section can be run independently using the #%% cell markers.
"""

import numpy as np
from meanfield import (
    MagneticSystem, 
    SublatticeDef, 
    plot_magnetizations, 
    find_critical_temperature
)

# %%
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
    # Using new convention: positive J = antiferromagnetic
    J_matrix = np.array([
        [ 0.0, +1.0, +1.0,  0.0],
        [+1.0,  0.0,  0.0, +1.0],
        [+1.0,  0.0,  0.0, +1.0],
        [ 0.0, +1.0, +1.0,  0.0]
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

example_pure_heisenberg()

# %%
def example_pure_ising():
    """Example 3: Pure Ising system (reproduces original example)."""
    
    # Define sublattices
    sublattice_defs = [
        SublatticeDef('ising', initial_direction=+1),
        SublatticeDef('ising', initial_direction=-1),
        SublatticeDef('ising', initial_direction=-1),
        SublatticeDef('ising', initial_direction=+1)
    ]
    
    # Same matrices - using new convention: positive J = antiferromagnetic
    J_matrix = np.array([
        [ 0.0, +1.0, +1.0,  0.0],
        [+1.0,  0.0,  0.0, +1.0],
        [+1.0,  0.0,  0.0, +1.0],
        [ 0.0, +1.0, +1.0,  0.0]
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

example_pure_ising()

# %%
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
    # Using new convention: positive J = antiferromagnetic
    J_matrix_8 = np.array([
        # H0   H1   H2   H3   I0   I1   I2   I3
        [ 0.0, +1.0, +1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # H0: couples to H1, H2
        [+1.0,  0.0,  0.0, +1.0,  0.0,  0.0,  0.0,  0.0],  # H1: couples to H0, H3  
        [+1.0,  0.0,  0.0, +1.0,  0.0,  0.0,  0.0,  0.0],  # H2: couples to H0, H3
        [ 0.0, +1.0, +1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # H3: couples to H1, H2
        [ 0.0,  0.0,  0.0,  0.0,  0.0, +1.0, +1.0,  0.0],  # I0: couples to I1, I2
        [ 0.0,  0.0,  0.0,  0.0, +1.0,  0.0,  0.0, +1.0],  # I1: couples to I0, I3
        [ 0.0,  0.0,  0.0,  0.0, +1.0,  0.0,  0.0, +1.0],  # I2: couples to I0, I3
        [ 0.0,  0.0,  0.0,  0.0,  0.0, +1.0, +1.0,  0.0]   # I3: couples to I1, I2
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

example_8_sublattice_separated()

# %%
def example_kugel_khomskii():
    """Example 5: Kugel-Khomskii system with coupled Heisenberg and Ising spins on the same atomic sites."""
    
    print("Solving Kugel-Khomskii system: Coupled Heisenberg-Ising spins on atomic sites...")
    
    temperatures = np.linspace(0.01, 2.5, 80)
    
    # Define sublattice properties for 4 atomic sites
    # First half (0-3): Heisenberg spins on sites 0, 1, 2, 3
    # Second half (4-7): Ising (orbital) spins on sites 0, 1, 2, 3
    sublattice_defs = [
        # Heisenberg spins (site indices 0-3)
        SublatticeDef('heisenberg', S=100, initial_direction=[0, 0, 1]),   # Site 0 spin
        SublatticeDef('heisenberg', S=100, initial_direction=[0, 0, 1]),  # Site 1 spin
        SublatticeDef('heisenberg', S=100, initial_direction=[0, 0, -1]),  # Site 2 spin
        SublatticeDef('heisenberg', S=100, initial_direction=[0, 0, -1]),   # Site 3 spin
        # Ising orbital spins (site indices 0-3)
        SublatticeDef('ising', initial_direction=+1),  # Site 0 orbital
        SublatticeDef('ising', initial_direction=-1),  # Site 1 orbital
        SublatticeDef('ising', initial_direction=-1),  # Site 2 orbital
        SublatticeDef('ising', initial_direction=+1)   # Site 3 orbital
    ]
    
    # Standard exchange coupling matrix J[i,j] between sublattices  
    # Same pattern as previous examples: separate Heisenberg and Ising groups
    # Using new convention: positive J = antiferromagnetic
    J_heis = 1.1
    Jnext_heis = 0.1
    J_ising = 0.6
    Jnext_ising = 0.2

    J_matrix = np.array([
        # H0      H1       H2       H3       I0        I1        I2        I3
        [ 0.0,  +J_heis, +J_heis, +Jnext_heis,     0.0,      0.0,      0.0,      0.0],  # H0: couples to H1, H2
        [+J_heis,  0.0,     Jnext_heis,  +J_heis,   0.0,      0.0,      0.0,      0.0],  # H1: couples to H0, H3  
        [+J_heis,  Jnext_heis,     0.0,  +J_heis,   0.0,      0.0,      0.0,      0.0],  # H2: couples to H0, H3
        [ Jnext_heis,  +J_heis, +J_heis,   0.0,     0.0,      0.0,      0.0,      0.0],  # H3: couples to H1, H2
        [ 0.0,     0.0,     0.0,     0.0,    0.0,   +J_ising, +J_ising,   Jnext_ising],  # I0: couples to I1, I2
        [ 0.0,     0.0,     0.0,     0.0, +J_ising,    0.0,      Jnext_ising,   +J_ising],  # I1: couples to I0, I3
        [ 0.0,     0.0,     0.0,     0.0, +J_ising,    Jnext_ising,      0.0,   +J_ising],  # I2: couples to I0, I3
        [ 0.0,     0.0,     0.0,     0.0,  +Jnext_ising,   +J_ising, +J_ising,   0.0]   # I3: couples to I1, I2
    ])
    
    # Coordination matrix: same connectivity pattern within each group  
    z_matrix = np.array([
        [0, 2, 2, 4, 0, 0, 0, 0],  # H0 has 2 neighbors of H1, 2 of H2
        [2, 0, 4, 2, 0, 0, 0, 0],  # H1 has 2 neighbors of H0, 2 of H3
        [2, 4, 0, 2, 0, 0, 0, 0],  # H2 has 2 neighbors of H0, 2 of H3  
        [4, 2, 2, 0, 0, 0, 0, 0],  # H3 has 2 neighbors of H1, 2 of H2
        [0, 0, 0, 0, 0, 2, 2, 4],  # I0 has 2 neighbors of I1, 2 of I2
        [0, 0, 0, 0, 2, 0, 4, 2],  # I1 has 2 neighbors of I0, 2 of I3
        [0, 0, 0, 0, 2, 4, 0, 2],  # I2 has 2 neighbors of I0, 2 of I3
        [0, 0, 0, 0, 4, 2, 2, 0]   # I3 has 2 neighbors of I1, 2 of I2
    ])
    
    # Kugel-Khomskii coupling matrix K[i,j] between atomic sites (4x4 for 4 sites)
    # This couples Heisenberg and Ising spins on the same and neighboring sites
    k_mag = -0.55
    K_matrix = np.array([
        # Site 0, Site 1, Site 2, Site 3
        [ 0.0,  k_mag,   k_mag,   0.0 ],  # Site 0: KK coupling with sites 1,2
        [k_mag,  0.0,    0.0,   k_mag ],  # Site 1: KK coupling with sites 0,3
        [k_mag,  0.0,    0.0,   k_mag ],  # Site 2: KK coupling with sites 0,3
        [ 0.0,  k_mag,   k_mag,   0.0 ]   # Site 3: KK coupling with sites 1,2
    ])
    
    # Create systems for comparison
    print("Creating standard system...")
    # system_standard = MagneticSystem(sublattice_defs, J_matrix, z_matrix, 
    #                                field_method='standard',
    #                                max_iterations=1000)
    
    print("Creating Kugel-Khomskii system...")
    system_kk = MagneticSystem(sublattice_defs, J_matrix, z_matrix,
                              field_method='kugel_khomskii',
                              kugel_khomskii_coupling_matrix=K_matrix,
                              max_iterations=10000)
    
    # Solve both systems with rattling to break symmetry
    # print("Solving standard system...")
    # mags_standard, _ = system_standard.solve_temperature_range(temperatures, 
    #                                                          rattle_iterations=30, 
    #                                                          rattle_strength=0.05)
    
    print("Solving Kugel-Khomskii system...")
    mags_kk, _ = system_kk.solve_temperature_range(temperatures,
                                                 reverse_order=True,
                                                 rattle_iterations=10, 
                                                 rattle_strength=0.1)
    
    # Extract magnetizations for plotting
    mags_kk_list = []
    
    for i in range(len(sublattice_defs)):
        kk_mags = [mags_kk[j, i] for j in range(len(temperatures))]
        mags_kk_list.append(np.array(kk_mags))
    
    # # Plot standard system
    # plot_magnetizations(temperatures, mags_std_list, system_standard.sublattice_params,
    #                    'Kugel-Khomskii System: Standard Exchange Only')
    
    # Plot Kugel-Khomskii system
    plot_magnetizations(temperatures, mags_kk_list, system_kk.sublattice_params,
                       'Kugel-Khomskii System: With K-K Coupling')
    
    # Compare critical temperatures
    # std_Tc = find_critical_temperature(temperatures, mags_std_list[0])  # Heisenberg site 0
    kk_Tc = find_critical_temperature(temperatures, mags_kk_list[0])    # Heisenberg site 0
    
    # std_Tc_ising = find_critical_temperature(temperatures, mags_std_list[4])  # Ising site 0
    kk_Tc_ising = find_critical_temperature(temperatures, mags_kk_list[4])    # Ising site 0
    
    print("\n" + "="*60)
    print("KUGEL-KHOMSKII COUPLING EFFECTS:")
    print("="*60)
    
    print(f"Heisenberg Tc (Kugel-Khomskii): {kk_Tc:.3f}")
    print(f"Ising Tc (Kugel-Khomskii):     {kk_Tc_ising:.3f}")
    # if std_Tc and kk_Tc:
    #     print(f"Heisenberg Tc (standard):      {std_Tc:.3f}")
    #     print(f"Heisenberg Tc (Kugel-Khomskii): {kk_Tc:.3f}")
    #     # print(f"Heisenberg Tc change:          {kk_Tc - std_Tc:+.3f}")
    
    # if std_Tc_ising and kk_Tc_ising:
    #     print(f"Ising Tc (standard):           {std_Tc_ising:.3f}")
    #     print(f"Ising Tc (Kugel-Khomskii):     {kk_Tc_ising:.3f}")
    #     print(f"Ising Tc change:               {kk_Tc_ising - std_Tc_ising:+.3f}")
    
    # Show magnetization values at T=0.5
    print(f"\nMagnetization values at T=0.5:")
    T_idx = np.argmin(np.abs(temperatures - 0.5))
    
    for i in range(4):  # First 4 are Heisenberg
        kk_mag = mags_kk_list[i][T_idx]
        print(f"  Heisenberg site {i}: m_kkz={kk_mag[2]:.4f}")
    
    for i in range(4, 8):  # Next 4 are Ising
        kk_mag = mags_kk_list[i][T_idx]
        print(f"  Ising site {i-4}:     m_kk={kk_mag:.4f}")
    
    print("="*60)
# Example 4: Run Kugel-Khomskii System
example_kugel_khomskii()

# %%
# Run all examples at once (optional)
if __name__ == '__main__':
    print("Running all examples...")
    example_pure_heisenberg()
    example_pure_ising()
    example_8_sublattice_separated()
    example_kugel_khomskii()
    print("All examples completed!")
