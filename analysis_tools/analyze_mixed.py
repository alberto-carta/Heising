#%%
"""
Mixed System Analysis - Independent Ising and Heisenberg Ferromagnets

Analyzes Monte Carlo results for a mixed system with uncoupled Ising and Heisenberg spins.
Each spin type should show its own phase transition at different temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Load the mixed system data
filename = "../monte_carlo/examples/mixed_system/independent_ferromagnets_highres_mixed_system.dat"

# Load data - columns are: T, E/spin, M, |M|, Cv, Chi, AcceptRate, Mag[H], Mag[I]
data = pd.read_csv(filename, comment='#', delim_whitespace=True,
                  names=['T', 'E', 'M', 'AbsM', 'Cv', 'Chi', 'AcceptRate', 'Mag_H', 'Mag_I'])

print(f"Loaded {len(data)} temperature points")
print(f"Temperature range: {data['T'].min():.2f} to {data['T'].max():.2f}")

#%% Plot absolute magnetizations for both species
plt.figure(figsize=(10, 6))

# Heisenberg in red
plt.plot(data['T'], np.abs(data['Mag_H']), 'ro-', markersize=4, label='Heisenberg', linewidth=1.5)

# Ising in blue  
plt.plot(data['T'], np.abs(data['Mag_I']), 'bs-', markersize=4, label='Ising', linewidth=1.5)

# Mark theoretical critical temperatures
plt.axvline(x=1.44, color='red', linestyle=':', alpha=0.5, label='Heisenberg Tc (theory)')
plt.axvline(x=4.5, color='blue', linestyle=':', alpha=0.5, label='Ising Tc (theory)')

plt.xlabel('Temperature T', fontsize=12)
plt.ylabel('|Magnetization| per spin', fontsize=12)
plt.title('Independent Ferromagnets: Ising vs Heisenberg', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot energy per spin
plt.figure(figsize=(10, 6))

plt.plot(data['T'], data['E'], 'go-', markersize=4, linewidth=1.5)
plt.axhline(y=-3.0, color='k', linestyle='--', alpha=0.3, label='Ground state (J=-1)')

plt.xlabel('Temperature T', fontsize=12)
plt.ylabel('Energy per spin', fontsize=12)
plt.title('Total Energy (Mixed System)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot specific heat
plt.figure(figsize=(10, 6))

plt.plot(data['T'], data['Cv'], 'mo-', markersize=4, linewidth=1.5)
plt.axvline(x=1.44, color='red', linestyle=':', alpha=0.5, label='Heisenberg Tc')
plt.axvline(x=4.5, color='blue', linestyle=':', alpha=0.5, label='Ising Tc')

plt.xlabel('Temperature T', fontsize=12)
plt.ylabel('Specific Heat', fontsize=12)
plt.title('Specific Heat (Should show two peaks)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Plot susceptibility
plt.figure(figsize=(10, 6))

plt.plot(data['T'], data['Chi'], 'co-', markersize=4, linewidth=1.5)
plt.axvline(x=1.44, color='red', linestyle=':', alpha=0.5, label='Heisenberg Tc')
plt.axvline(x=4.5, color='blue', linestyle=':', alpha=0.5, label='Ising Tc')

plt.xlabel('Temperature T', fontsize=12)
plt.ylabel('Susceptibility', fontsize=12)
plt.title('Susceptibility (Should show two peaks)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Find critical temperatures from magnetization drops
# For Heisenberg: find where |Mag_H| drops below 0.5
heisenberg_ordered = data[np.abs(data['Mag_H']) > 0.1]
if len(heisenberg_ordered) > 0:
    Tc_heisenberg = heisenberg_ordered['T'].min()
    print(f"Heisenberg Tc estimate: {Tc_heisenberg:.2f} (theory: 1.44)")

# For Ising: find where |Mag_I| drops below 0.5
ising_ordered = data[np.abs(data['Mag_I']) > 0.5]
if len(ising_ordered) > 0:
    Tc_ising = ising_ordered['T'].min()
    print(f"Ising Tc estimate: {Tc_ising:.2f} (theory: 4.5)")

#%% 4-panel overview plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Independent Ising and Heisenberg Ferromagnets', fontsize=16, fontweight='bold')

# Panel 1: Magnetizations
ax1.plot(data['T'], np.abs(data['Mag_H']), 'ro-', markersize=3, label='Heisenberg', linewidth=1.5)
ax1.plot(data['T'], np.abs(data['Mag_I']), 'bs-', markersize=3, label='Ising', linewidth=1.5)
ax1.axvline(x=1.44, color='red', linestyle=':', alpha=0.4)
ax1.axvline(x=4.5, color='blue', linestyle=':', alpha=0.4)
ax1.set_xlabel('Temperature T', fontsize=11)
ax1.set_ylabel('|Magnetization| per spin', fontsize=11)
ax1.set_title('Magnetization', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Energy
ax2.plot(data['T'], data['E'], 'go-', markersize=3, linewidth=1.5)
ax2.axhline(y=-3.0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Temperature T', fontsize=11)
ax2.set_ylabel('Energy per spin', fontsize=11)
ax2.set_title('Energy', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel 3: Specific Heat
ax3.plot(data['T'], data['Cv'], 'mo-', markersize=3, linewidth=1.5)
ax3.axvline(x=1.44, color='red', linestyle=':', alpha=0.4, label='Heisenberg Tc')
ax3.axvline(x=4.5, color='blue', linestyle=':', alpha=0.4, label='Ising Tc')
ax3.set_xlabel('Temperature T', fontsize=11)
ax3.set_ylabel('Specific Heat', fontsize=11)
ax3.set_title('Specific Heat', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Susceptibility
ax4.plot(data['T'], data['Chi'], 'co-', markersize=3, linewidth=1.5)
ax4.axvline(x=1.44, color='red', linestyle=':', alpha=0.4, label='Heisenberg Tc')
ax4.axvline(x=4.5, color='blue', linestyle=':', alpha=0.4, label='Ising Tc')
ax4.set_xlabel('Temperature T', fontsize=11)
ax4.set_ylabel('Susceptibility', fontsize=11)
ax4.set_title('Susceptibility', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
