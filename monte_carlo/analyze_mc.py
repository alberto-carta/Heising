#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data file with headers
# The file now starts with comment lines (starting with #) and then has a header row
data = pd.read_csv('DATA.1.dat', sep=',', comment='#')

# Display basic information about the data
print("Data shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nFirst few rows:")
print(data.head())
print("\nData summary:")
print(data.describe())

# --- Create the plots ---
plt.style.use('seaborn-v0_8-whitegrid') # Sets a nice plot style
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Ising Model Monte Carlo Simulation Results', fontsize=16)

# 1. Plot Energy vs. Temperature
axes[0, 0].plot(data['Temperature'], data['Energy'], 'o-', label='Energy', markersize=4)
axes[0, 0].set_xlabel('Temperature (T)')
axes[0, 0].set_ylabel('Average Energy $\langle E \\rangle$')
axes[0, 0].set_title('Energy vs. Temperature')
axes[0, 0].grid(True, alpha=0.3)

# 2. Plot Magnetization vs. Temperature
axes[0, 1].plot(data['Temperature'], data['AbsMagnetization'], 'o-', color='r', label='Magnetization', markersize=4)
axes[0, 1].set_xlabel('Temperature (T)')
axes[0, 1].set_ylabel('Average Absolute Magnetization $\langle |M| \\rangle$')
axes[0, 1].set_title('Magnetization vs. Temperature')
axes[0, 1].grid(True, alpha=0.3)

# 3. Plot Specific Heat vs. Temperature
axes[1, 0].plot(data['Temperature'], data['SpecificHeat'], 'o-', color='g', label='Specific Heat', markersize=4)
axes[1, 0].set_xlabel('Temperature (T)')
axes[1, 0].set_ylabel('Specific Heat ($C_v$)')
axes[1, 0].set_title('Specific Heat vs. Temperature')
axes[1, 0].grid(True, alpha=0.3)

# 4. Plot Susceptibility vs. Temperature
axes[1, 1].plot(data['Temperature'], data['Susceptibility_M'], 'o-', color='m', label='Susceptibility', markersize=4)
axes[1, 1].set_xlabel('Temperature (T)')
axes[1, 1].set_ylabel('Susceptibility ($\\chi$)')
axes[1, 1].set_title('Susceptibility vs. Temperature')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('ising_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plots generated and saved as ising_model_results.png")

# Additional analysis
print(f"\nAdditional Analysis:")
if len(data) > 0:
    print(f"Temperature range: {data['Temperature'].min():.3f} to {data['Temperature'].max():.3f}")
    
    # Find temperature with maximum specific heat (likely near critical temperature)
    max_cv_idx = data['SpecificHeat'].idxmax()
    T_max_cv = data.loc[max_cv_idx, 'Temperature']
    max_cv = data.loc[max_cv_idx, 'SpecificHeat']
    print(f"Maximum specific heat: {max_cv:.3f} at T = {T_max_cv:.3f}")
    
    # Find temperature with maximum susceptibility
    max_chi_idx = data['Susceptibility_M'].idxmax()
    T_max_chi = data.loc[max_chi_idx, 'Temperature']
    max_chi = data.loc[max_chi_idx, 'Susceptibility_M']
    print(f"Maximum susceptibility: {max_chi:.3f} at T = {T_max_chi:.3f}")

# %%
