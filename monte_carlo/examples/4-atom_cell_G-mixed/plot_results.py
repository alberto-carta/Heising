#%%
import pandas as pd
import io

def load_monte_carlo_results(file_path):
    """
    Loads Monte Carlo simulation results, automatically identifying column names
    from the metadata header.
    """
    column_names = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Look for the specific header line defining the columns
            if line.startswith("# Columns:"):
                # Remove the prefix and split by whitespace
                header_content = line.replace("# Columns:", "").strip()
                column_names = header_content.split()
                break
    
    if not column_names:
        raise ValueError("Could not find '# Columns:' header in the file.")

    # Read the data, skipping lines that start with '#' (comments/metadata)
    # Using 'engine=python' to handle potential whitespace variations
    df = pd.read_csv(
        file_path, 
        sep=r'\s+', 
        comment='#', 
        names=column_names, 
        header=None
    )
    
    return df

df = load_monte_carlo_results('test_diagnostics_observables.out')
#%%
import matplotlib.pyplot as plt
Temp = df['T']
Cv = df['SpecificHeat']

plt.plot(Temp, Cv, marker='o')
plt.xlabel('Temperature (T)')
plt.ylabel('Specific Heat (Cv)')
plt.title('Specific Heat vs Temperature')
plt.grid()
plt.show()  


#%% 3 panel plot for Energy/spin, Specific Heat, and Mz[Cr1], Mz[Cr2], Mz[Cr3], Mz[Cr4]

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


# set font size to 14 for all plots
plt.rcParams.update({'font.size': 14})

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
# Plot Energy/spin
axs[0].plot(df['T'], df['Energy/spin'], marker='o', color='g')
axs[0].set_xlabel('Temperature (T)')
axs[0].set_ylabel('Energy per Spin')
axs[0].set_title('Energy per Spin vs Temperature')
axs[0].grid()       

# Plot Specific Heat
axs[1].plot(df['T'], df['SpecificHeat'], marker='o', color='g')
axs[1].set_xlabel('Temperature (T)')
axs[1].set_ylabel('Specific Heat (Cv)') 
axs[1].set_title('Specific Heat vs Temperature')
axs[1].grid()   
# Plot Mz[Cr1], Mz[Cr2], Mz[Cr3], Mz[Cr4]
axs[2].plot(df['T'], df['Mz[Cr1]'], marker='o', label='Mz[Cr1]', color=colorcycle[0])
axs[2].plot(df['T'], df['M[CrA]'], marker='s', label='M[CrA]', color=colorcycle[0], linestyle='--', alpha=0.5)
axs[2].plot(df['T'], df['Mz[Cr2]'], marker='o', label='Mz[Cr2]', color=colorcycle[1])
axs[2].plot(df['T'], df['M[CrB]'], marker='s', label='M[CrB]', color=colorcycle[1], linestyle='--', alpha=0.5)
# axs[2].plot(df['T'], df['Mz[Cr3]'], marker='o', label='Mz[Cr3]')
# axs[2].plot(df['T'], df['Mz[Cr4]'], marker='o', label='Mz[Cr4]')
axs[2].set_xlabel('Temperature (T)')
axs[2].set_ylabel('Magnetization (Mz)')
axs[2].set_title('Magnetization vs Temperature')
axs[2].legend()
axs[2].grid()   

# add vertical spacing between subplots

plt.tight_layout()



# Example usage:
# df = load_monte_carlo_results('test_diagnostics_observables.out')
# print(df.head())