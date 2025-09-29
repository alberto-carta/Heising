#%%
import numpy as np
import matplotlib.pyplot as plt

def brillouin_function(x, S):
    """
    Calculates the Brillouin function B_S(x).
    Handles the x=0 case to avoid division by zero.
    """
    if S == 0:
        return np.zeros_like(x)
    
    # Avoid division by zero at x=0
    # For small x, coth(x) -> 1/x. We add a tiny epsilon.
    eps = 1e-8
    x = np.where(x == 0, eps, x)

    coth_term1 = 1 / np.tanh(((2 * S + 1) / (2 * S)) * x)
    term1 = ((2 * S + 1) / (2 * S)) * coth_term1
    
    coth_term2 = 1 / np.tanh(x / (2 * S))
    term2 = (1 / (2 * S)) * coth_term2
    
    return term1 - term2

def solve_system(temperatures, J_matrix, z_matrix, sublattice_params):
    """
    Solves the coupled mean-field equations for a multi-sublattice system with mixed statistics.

    Args:
        temperatures (np.array): Array of temperatures to solve for.
        J_matrix (np.array): Coupling matrix J[i,j] between sublattices i and j.
        z_matrix (np.array): Coordination matrix z[i,j] for neighbors between sublattices i and j.
        sublattice_params (list): List of dictionaries, each containing:
            - 'model': 'ising' or 'heisenberg'
            - 'S': spin quantum number (for Heisenberg) or ignored (for Ising)
            - 'initial_direction': initial magnetization direction 
                * For Heisenberg: [x,y,z] vector
                * For Ising: scalar (+1 or -1) for initial spin direction

    Returns:
        list: List of magnetizations for each sublattice as numpy arrays.
    """
    n_sublattices = len(sublattice_params)
    
    # Store results for all sublattices
    magnetizations_vs_T = [[] for _ in range(n_sublattices)]

    for T in temperatures:
        # Initial guess for the magnetizations based on each sublattice's properties
        magnetizations = []
        for i, params in enumerate(sublattice_params):
            if params['model'] == 'heisenberg':
                # Vector guess for Heisenberg model
                if 'initial_direction' in params:
                    initial_dir = np.array(params['initial_direction'])
                else:
                    # Default alternating z-direction
                    sign = (-1) ** i
                    initial_dir = np.array([0.0, 0.0, sign])
                # Normalize and scale by a small initial value
                initial_dir = initial_dir / np.linalg.norm(initial_dir) if np.linalg.norm(initial_dir) > 0 else np.array([0.0, 0.0, 1.0])
                magnetizations.append(initial_dir * 0.8)  # Start with 80% of max magnetization
            else:  # Ising model
                # Scalar guess for Ising model
                if 'initial_direction' in params:
                    # Use specified initial direction (+1 or -1)
                    sign = np.sign(params['initial_direction']) if params['initial_direction'] != 0 else 1
                else:
                    # Default alternating signs
                    sign = (-1) ** i
                magnetizations.append(sign * 0.8)
        
        beta = 1.0 / T if T > 0 else np.inf

        # Iterate to find a self-consistent solution
        for iteration in range(500): # Max 500 iterations for convergence
            # Calculate effective fields for all sublattices
            # Normalize magnetizations so J represents pure energy scale
            h_eff = []
            for i in range(n_sublattices):
                # Initialize h_field based on the target sublattice type
                if sublattice_params[i]['model'] == 'heisenberg':
                    h_field = np.array([0.0, 0.0, 0.0])  # Vector field for Heisenberg
                else:
                    h_field = 0.0  # Scalar field for Ising
                
                for j in range(n_sublattices):
                    if i != j and abs(J_matrix[i, j]) > 1e-12:  # Only non-zero couplings
                        # Get normalized magnetization from sublattice j
                        if sublattice_params[j]['model'] == 'heisenberg':
                            # For Heisenberg: normalize by S_j
                            norm_mag_j = magnetizations[j] / sublattice_params[j]['S']
                        else:
                            # For Ising: already normalized (max = 1)
                            norm_mag_j = magnetizations[j]
                        
                        coupling_contribution = z_matrix[i, j] * J_matrix[i, j] * norm_mag_j
                        
                        # Add contribution based on target sublattice type
                        if sublattice_params[i]['model'] == 'heisenberg':
                            # Target is Heisenberg - field should be vector
                            if isinstance(coupling_contribution, np.ndarray):
                                h_field += coupling_contribution
                            else:
                                # Scalar contribution to vector - add to z-component only
                                h_field[2] += coupling_contribution
                        else:
                            # Target is Ising - field should be scalar
                            if isinstance(coupling_contribution, np.ndarray):
                                # Vector contribution to scalar - take z-component only
                                h_field += coupling_contribution[2] if len(coupling_contribution) > 2 else np.linalg.norm(coupling_contribution)
                            else:
                                h_field += coupling_contribution
                
                h_eff.append(h_field)
            
            # Update magnetizations based on each sublattice's model
            new_magnetizations = []
            for i, (params, h_field) in enumerate(zip(sublattice_params, h_eff)):
                if params['model'] == 'ising':
                    # Ising update: m = tanh(βh)
                    # h_field is now normalized, so it represents the proper energy scale directly
                    if isinstance(h_field, np.ndarray):
                        # If h_field is a vector (from Heisenberg neighbors), take its magnitude
                        h_magnitude = np.linalg.norm(h_field)
                        sign = np.sign(h_field[2]) if abs(h_field[2]) > 1e-9 else 1
                    else:
                        h_magnitude = abs(h_field)
                        sign = np.sign(h_field) if abs(h_field) > 1e-9 else 1
                    new_m = np.tanh(beta * h_magnitude) * sign
                    new_magnetizations.append(new_m)
                    
                elif params['model'] == 'heisenberg':
                    # Heisenberg update: m = S * B_S(βh) * h/|h|
                    # h_field is normalized, so energy scale is determined purely by J, not S
                    S = params['S']
                    if isinstance(h_field, np.ndarray):
                        h_magnitude = np.linalg.norm(h_field)
                    else:
                        # If h_field is scalar (from Ising neighbors), make it a z-vector
                        h_field = np.array([0.0, 0.0, float(h_field)])
                        h_magnitude = abs(h_field[2])
                    
                    if h_magnitude < 1e-9:  # Avoid division by zero
                        new_m = np.array([0.0, 0.0, 0.0])
                    else:
                        # Use h_magnitude directly - energy scale set by J, not S
                        brillouin_val = brillouin_function(beta * h_magnitude, S)
                        new_m = S * brillouin_val * (h_field / h_magnitude)
                    new_magnetizations.append(new_m)

            # Check for convergence
            converged = True
            for old_m, new_m in zip(magnetizations, new_magnetizations):
                if not np.allclose(old_m, new_m, atol=1e-6):
                    converged = False
                    break
            
            if converged:
                break

            # Simple mixing to improve stability
            mix = 0.1
            for i in range(n_sublattices):
                magnetizations[i] = (1 - mix) * magnetizations[i] + mix * new_magnetizations[i]

        # Store results for this temperature
        for i in range(n_sublattices):
            magnetizations_vs_T[i].append(magnetizations[i])
    
    # Convert to numpy arrays
    result = []
    for i in range(n_sublattices):
        result.append(np.array(magnetizations_vs_T[i]))
    
    return result

def plot_results(T, magnetizations_list, sublattice_params, title):
    """Helper function to plot the results for multiple sublattices."""
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 'x', 's', '^', 'v', 'd', '+', '*']

    for i, (magnetizations, params) in enumerate(zip(magnetizations_list, sublattice_params)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        if params['model'] == 'heisenberg':
            # For Heisenberg spins, plot z-component normalized by S
            if magnetizations.ndim == 2: # Vector case (shape is [temps, 3])
                mag_values = magnetizations[:, 2] / params['S']  # z-component normalized by S
                label = f'$m_{{{i+1},z}}/S$ (Heisenberg S={params["S"]})'
            else:
                # Fallback if somehow scalar
                mag_values = magnetizations / params['S']
                label = f'$m_{{{i+1}}}/S$ (Heisenberg S={params["S"]})'
        else:  # Ising
            # For Ising spins, magnetization is already normalized (max = 1)
            mag_values = magnetizations
            label = f'$m_{{{i+1}}}$ (Ising)'
        

        plt.plot(T, mag_values, marker=marker, linestyle='-', 
                label=label, markersize=4, color=color, alpha=0.8)
    
    plt.xlabel('Temperature (T/|J|)')
    plt.ylabel('Normalized Sublattice Magnetization')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- System Parameters for 4-sublattice system ---
    n_sublattices = 4
    
    # Temperature range (in units of |J|) - extended to see full transition
    temperatures = np.linspace(0.01, 8.0, 80)

    # Example 1: Mixed Ising-Heisenberg system
    # Define coupling matrix (4x4) - antiferromagnetic nearest neighbor coupling
    J_matrix = np.array([
        [ 0.0, -1.0, -1.0,  0.0],  # Sublattice 0 couplings
        [-1.0,  0.0,  0.0, -1.0],  # Sublattice 1 couplings  
        [-1.0,  0.0,  0.0, -1.0],  # Sublattice 2 couplings
        [ 0.0, -1.0, -1.0,  0.0]   # Sublattice 3 couplings
    ])
    
    # Coordination numbers matrix (how many neighbors of each type)
    z_matrix = np.array([
        [0, 2, 2, 0],  # Sublattice 0 has 2 neighbors of type 1, 2 of type 2
        [2, 0, 0, 2],  # Sublattice 1 has 2 neighbors of type 0, 2 of type 3
        [2, 0, 0, 2],  # Sublattice 2 has 2 neighbors of type 0, 2 of type 3  
        [0, 2, 2, 0]   # Sublattice 3 has 2 neighbors of type 1, 2 of type 2
    ])
    
    # Define sublattice properties: mix of Ising and Heisenberg
    sublattice_params = [
        {'model': 'ising', 'initial_direction': +1},                          # Sublattice 0: Ising, starts +
        {'model': 'heisenberg', 'S': 1.0, 'initial_direction': [0, 0, -1]},  # Sublattice 1: S=1 Heisenberg
        {'model': 'ising', 'initial_direction': -1},                          # Sublattice 2: Ising, starts -  
        {'model': 'heisenberg', 'S': 0.5, 'initial_direction': [0, 0, 1]}   # Sublattice 3: S=1/2 Heisenberg
    ]
    
    print("Solving mixed Ising-Heisenberg 4-sublattice system...")
    magnetizations = solve_system(temperatures, J_matrix, z_matrix, sublattice_params)
    plot_results(temperatures, magnetizations, sublattice_params, 
                'Mixed Ising-Heisenberg 4-Sublattice System')

    # Example 2: Pure Heisenberg system with different S values
    sublattice_params_heisenberg = [
        {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, 0.1]},
        {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, 0]},
        {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, -0.6]},
        {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, 0]}
    ]
    
    print("Solving pure classical system ...")
    magnetizations_heis = solve_system(temperatures, J_matrix, z_matrix, sublattice_params_heisenberg)
    plot_results(temperatures, magnetizations_heis, sublattice_params_heisenberg,
                'Pure Heisenberg 4-Sublattice System (Different S values)')
                
    # Example 3: Pure Ising system for comparison
    sublattice_params_ising = [
        {'model': 'ising', 'initial_direction': +1},
        {'model': 'ising', 'initial_direction': -1}, 
        {'model': 'ising', 'initial_direction': -1},
        {'model': 'ising', 'initial_direction': +1}
    ]
    

    print("Solving pure Ising 4-sublattice system...")
    magnetizations_ising = solve_system(temperatures, J_matrix, z_matrix, sublattice_params_ising)
    plot_results(temperatures, magnetizations_ising, sublattice_params_ising,
                'Pure Ising 4-Sublattice System')
    
    # Find critical temperature for 4-sublattice Ising
    abs_mag_4 = np.abs(magnetizations_ising[0])
    critical_idx_4 = np.where(abs_mag_4 < 0.1)[0]
    if len(critical_idx_4) > 0:
        Tc_4 = temperatures[critical_idx_4[0]]
        print(f"Critical temperature (4-sublattice Ising): {Tc_4:.3f}")
    else:
        print("Critical temperature (4-sublattice Ising): > 8.0")



# %%
# Example 4: 8 sublattices - 4 Heisenberg + 4 Ising with coupling only between different types
print("Solving 8-sublattice system: 4 Heisenberg + 4 Ising with no cross-coupling...")

temperatures = np.linspace(0.01, 5, 100)  # Extended range to see full transitions

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

# Define sublattice properties: 4 Heisenberg + 4 Ising (separate groups)
sublattice_params_8 = [
    # Heisenberg sublattices (0-3) - interact only with each other
    {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, 1]},   # H0
    {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, -1]},  # H1
    {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, -1]},  # H2
    {'model': 'heisenberg', 'S': 100.0, 'initial_direction': [0, 0, 1]},   # H3
    # Ising sublattices (4-7) - interact only with each other
    {'model': 'ising', 'initial_direction': +1},  # I0
    {'model': 'ising', 'initial_direction': -1},  # I1
    {'model': 'ising', 'initial_direction': -1},  # I2
    {'model': 'ising', 'initial_direction': +1}   # I3
]

magnetizations_8 = solve_system(temperatures, J_matrix_8, z_matrix_8, sublattice_params_8)
plot_results(temperatures, magnetizations_8, sublattice_params_8,
            '8-Sublattice System: 4 Heisenberg + 4 Ising (Cross-coupling Only)')

# Find critical temperature for 8-sublattice Ising part (sublattice 4)
abs_mag_8_ising = np.abs(magnetizations_8[4])  # Ising sublattice I0
critical_idx_8 = np.where(abs_mag_8_ising < 0.1)[0]
if len(critical_idx_8) > 0:
    Tc_8 = temperatures[critical_idx_8[0]]
    print(f"Critical temperature (8-sublattice Ising part): {Tc_8:.3f}")
else:
    print("Critical temperature (8-sublattice Ising part): > 8.0")


# %%
