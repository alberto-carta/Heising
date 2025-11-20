#%%
"""
Ferromagnet Phase Transition Analysis

Simple analysis of Monte Carlo simulation results for Ising and Heisenberg ferromagnets.
Just run the cells with Shift+Enter to plot the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#%% Load data function
def load_data(filename):
    """Load ferromagnet data from output file."""
    data = pd.read_csv(filename, comment='#', delim_whitespace=True,
                      names=['T', 'E', 'M', 'AbsM', 'Cv', 'Chi', 'AcceptRate'])
    print(f"Loaded {len(data)} points from {filename}")
    return data

def plot_results(data, title, theoretical_tc=None):
    """Plot the four main quantities."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    T = data['T']
    
    # Magnetization
    ax1.plot(T, data['M'], 'ro-', markersize=3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    if theoretical_tc:
        ax1.axvline(x=theoretical_tc, color='red', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel('Magnetization per spin')
    ax1.set_title('Magnetization')
    ax1.grid(True, alpha=0.3)
    
    # Energy
    ax2.plot(T, data['E'], 'go-', markersize=3)
    ax2.axhline(y=-3.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('Energy per spin')
    ax2.set_title('Energy')
    ax2.grid(True, alpha=0.3)
    
    # Specific Heat
    ax3.plot(T, data['Cv'], 'bo-', markersize=3)
    if theoretical_tc:
        ax3.axvline(x=theoretical_tc, color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Temperature T')
    ax3.set_ylabel('Specific Heat')
    ax3.set_title('Specific Heat')
    ax3.grid(True, alpha=0.3)
    
    # Susceptibility  
    ax4.plot(T, data['Chi'], 'mo-', markersize=3)
    if theoretical_tc:
        ax4.axvline(x=theoretical_tc, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Temperature T')
    ax4.set_ylabel('Susceptibility')
    ax4.set_title('Susceptibility')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

#%% Load Heisenberg ferromagnet data
heisenberg_file = "../monte_carlo/heisenberg_ferromagnet_proper_heisenberg_system.dat"
if not os.path.exists(heisenberg_file):
    heisenberg_file = "../monte_carlo/heisenberg_ferromagnet_transition.dat"
heisenberg_data = load_data(heisenberg_file)

#%% Plot Heisenberg ferromagnet results  
fig = plot_results(heisenberg_data, "Heisenberg Ferromagnet Phase Transition", theoretical_tc=1.44)
plt.show()

# Find critical temperature from peaks
max_cv_idx = heisenberg_data['Cv'].idxmax()
max_chi_idx = heisenberg_data['Chi'].idxmax()
T_cv = heisenberg_data.iloc[max_cv_idx]['T']
T_chi = heisenberg_data.iloc[max_chi_idx]['T']
print(f"Heisenberg Tc estimate: Cv peak at T={T_cv:.3f}, Chi peak at T={T_chi:.3f}")
print(f"Average: {(T_cv + T_chi)/2:.3f} (theory: 1.44)")

#%% Load Ising ferromagnet data  
ising_file = "../monte_carlo/ising_ferromagnet_transition.dat"
ising_data = load_data(ising_file)

#%% Plot Ising ferromagnet results
fig = plot_results(ising_data, "Ising Ferromagnet Phase Transition", theoretical_tc=4.5)
plt.show()

# Find critical temperature from peaks
max_cv_idx = ising_data['Cv'].idxmax()
max_chi_idx = ising_data['Chi'].idxmax()
T_cv = ising_data.iloc[max_cv_idx]['T']
T_chi = ising_data.iloc[max_chi_idx]['T']
print(f"Ising Tc estimate: Cv peak at T={T_cv:.3f}, Chi peak at T={T_chi:.3f}")
print(f"Average: {(T_cv + T_chi)/2:.3f} (theory: 4.5)")

#%% Compare both models side by side
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Ising vs Heisenberg Ferromagnet Comparison', fontsize=14)

# Magnetization
ax1.plot(ising_data['T'], ising_data['M'], 'b-', label='Ising')
ax1.plot(heisenberg_data['T'], heisenberg_data['M'], 'r-', label='Heisenberg')
ax1.axvline(x=4.5, color='blue', linestyle=':', alpha=0.7)
ax1.axvline(x=1.44, color='red', linestyle=':', alpha=0.7)
ax1.set_xlabel('T'); ax1.set_ylabel('Magnetization'); ax1.legend(); ax1.grid(True)

# Energy
ax2.plot(ising_data['T'], ising_data['E'], 'b-', label='Ising')
ax2.plot(heisenberg_data['T'], heisenberg_data['E'], 'r-', label='Heisenberg')
ax2.set_xlabel('T'); ax2.set_ylabel('Energy'); ax2.legend(); ax2.grid(True)

# Specific Heat
ax3.plot(ising_data['T'], ising_data['Cv'], 'b-', label='Ising')
ax3.plot(heisenberg_data['T'], heisenberg_data['Cv'], 'r-', label='Heisenberg')
ax3.set_xlabel('T'); ax3.set_ylabel('Specific Heat'); ax3.legend(); ax3.grid(True)

# Susceptibility
ax4.plot(ising_data['T'], ising_data['Chi'], 'b-', label='Ising')
ax4.plot(heisenberg_data['T'], heisenberg_data['Chi'], 'r-', label='Heisenberg')
ax4.set_xlabel('T'); ax4.set_ylabel('Susceptibility'); ax4.legend(); ax4.grid(True)

plt.tight_layout()
plt.show()

# %%
