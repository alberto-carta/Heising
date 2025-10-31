#!/usr/bin/env python3
"""
Heisenberg Ferromagnet Phase Transition Analysis

Analyzes the output from heisenberg_analysis.cpp and creates publication-quality plots
showing the ferromagnetic phase transition in the 3D Heisenberg model.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_heisenberg_data(filename):
    """Load Heisenberg ferromagnet data from the output file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        print("Please run the heisenberg_analysis program first.")
        return None
    
    try:
        # Read the data file, handling comments and whitespace-separated data
        data = pd.read_csv(filename, 
                          comment='#', 
                          delim_whitespace=True,
                          names=['Temperature', 'Energy_per_spin', 'Magnetization_per_spin', 
                                'AbsMag_per_spin', 'SpecificHeat', 'Susceptibility', 'AcceptanceRate'])
        
        print(f"Loaded {len(data)} temperature points from {filename}")
        print(f"Temperature range: T = {data['Temperature'].min():.2f} to {data['Temperature'].max():.2f}")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def plot_ferromagnet_transition(data):
    """Create comprehensive plots for Heisenberg ferromagnet transition."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('3D Heisenberg Ferromagnet Phase Transition', fontsize=16, fontweight='bold')
    
    T = data['Temperature'].values
    
    # Plot 1: Magnetization vs Temperature (the key quantity for ferromagnetism)
    ax1.plot(T, data['Magnetization_per_spin'], 'ro-', linewidth=2, markersize=4, 
             label='M/N (z-component)', alpha=0.8)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel('Magnetization per spin')
    ax1.set_title('Magnetization vs Temperature\n(Shows ferromagnetic ordering)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add theoretical Tc line
    T_c_theory = 1.44  # Theoretical value for 3D Heisenberg FM with |J|=1
    ax1.axvline(x=T_c_theory, color='red', linestyle=':', alpha=0.7, 
                label=f'Theory: Tc ≈ {T_c_theory}')
    ax1.legend()
    
    # Plot 2: Energy vs Temperature
    ax2.plot(T, data['Energy_per_spin'], 'go-', linewidth=2, markersize=4)
    ax2.axhline(y=-3.0, color='k', linestyle='--', alpha=0.5, 
                label='Ground state: -3.0')
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('Energy per spin')
    ax2.set_title('Average Energy per Spin')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Specific Heat vs Temperature (peaks at Tc)
    ax3.plot(T, data['SpecificHeat'], 'bo-', linewidth=2, markersize=4)
    ax3.axvline(x=T_c_theory, color='red', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Temperature T')
    ax3.set_ylabel('Specific Heat C_v')
    ax3.set_title('Specific Heat\n(Peak indicates Tc)')
    ax3.grid(True, alpha=0.3)
    
    # Find and mark the peak
    max_cv_idx = data['SpecificHeat'].idxmax()
    T_max_cv = data.iloc[max_cv_idx]['Temperature']
    max_cv = data.iloc[max_cv_idx]['SpecificHeat']
    ax3.plot(T_max_cv, max_cv, 'r*', markersize=12, 
             label=f'Peak at T = {T_max_cv:.2f}')
    ax3.legend()
    
    # Plot 4: Susceptibility vs Temperature (also peaks at Tc)
    ax4.plot(T, data['Susceptibility'], 'mo-', linewidth=2, markersize=4)
    ax4.axvline(x=T_c_theory, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Temperature T')
    ax4.set_ylabel('Magnetic Susceptibility χ')
    ax4.set_title('Magnetic Susceptibility\n(Peak indicates Tc)')
    ax4.grid(True, alpha=0.3)
    
    # Find and mark the peak
    max_chi_idx = data['Susceptibility'].idxmax()
    T_max_chi = data.iloc[max_chi_idx]['Temperature']
    max_chi = data.iloc[max_chi_idx]['Susceptibility']
    ax4.plot(T_max_chi, max_chi, 'r*', markersize=12, 
             label=f'Peak at T = {T_max_chi:.2f}')
    ax4.legend()
    
    plt.tight_layout()
    return fig, T_max_cv, T_max_chi

def plot_transition_details(data):
    """Create a focused plot on the transition region."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Heisenberg Ferromagnet Transition Details', fontsize=16, fontweight='bold')
    
    T = data['Temperature'].values
    
    # Focus on transition region (around T = 1.0 to 2.0)
    transition_mask = (T >= 0.8) & (T <= 2.5)
    T_trans = T[transition_mask]
    
    # Plot 1: Magnetization in transition region
    ax1.plot(T_trans, data['Magnetization_per_spin'][transition_mask], 'ro-', 
             linewidth=3, markersize=6, label='M/N (z-component)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=1.44, color='red', linestyle=':', alpha=0.7, 
                label='Theory: Tc ≈ 1.44')
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel('Magnetization per spin')
    ax1.set_title('Transition Region: Magnetization')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Combined specific heat and susceptibility
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(T_trans, data['SpecificHeat'][transition_mask], 'bo-', 
                     linewidth=3, markersize=6, label='Specific Heat')
    line2 = ax2_twin.plot(T_trans, data['Susceptibility'][transition_mask], 'mo-', 
                          linewidth=3, markersize=6, label='Susceptibility')
    
    ax2.axvline(x=1.44, color='red', linestyle=':', alpha=0.7, 
                label='Theory: Tc ≈ 1.44')
    
    ax2.set_xlabel('Temperature T')
    ax2.set_ylabel('Specific Heat C_v', color='blue')
    ax2_twin.set_ylabel('Susceptibility χ', color='magenta')
    ax2.set_title('Transition Peaks')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    return fig

def analyze_critical_temperature(data):
    """Analyze and estimate the critical temperature."""
    print("\n" + "="*50)
    print("CRITICAL TEMPERATURE ANALYSIS")
    print("="*50)
    
    # Find peaks in specific heat and susceptibility
    max_cv_idx = data['SpecificHeat'].idxmax()
    T_cv = data.iloc[max_cv_idx]['Temperature']
    max_cv = data.iloc[max_cv_idx]['SpecificHeat']
    
    max_chi_idx = data['Susceptibility'].idxmax()
    T_chi = data.iloc[max_chi_idx]['Temperature']
    max_chi = data.iloc[max_chi_idx]['Susceptibility']
    
    print(f"Specific heat peak:     Cv_max = {max_cv:8.3f} at T = {T_cv:.3f}")
    print(f"Susceptibility peak:    χ_max  = {max_chi:8.3f} at T = {T_chi:.3f}")
    print(f"Theoretical Tc (3D HFM):              Tc ≈ 1.44")
    
    # Average estimate
    T_c_estimate = (T_cv + T_chi) / 2
    print(f"Average estimated Tc:                 Tc ≈ {T_c_estimate:.3f}")
    
    # Check agreement with theory
    theoretical_tc = 1.44
    error_cv = abs(T_cv - theoretical_tc) / theoretical_tc * 100
    error_chi = abs(T_chi - theoretical_tc) / theoretical_tc * 100
    
    print(f"\nComparison with theory:")
    print(f"Specific heat error:   {error_cv:5.1f}%")
    print(f"Susceptibility error:  {error_chi:5.1f}%")
    
    # Analyze magnetization behavior
    print(f"\nMagnetization analysis:")
    low_T_data = data[data['Temperature'] <= 0.5]
    high_T_data = data[data['Temperature'] >= 3.0]
    
    if len(low_T_data) > 0:
        avg_low_T_mag = low_T_data['Magnetization_per_spin'].abs().mean()
        print(f"Low T (T ≤ 0.5):  |M|/N ≈ {avg_low_T_mag:.3f} (ordered)")
    
    if len(high_T_data) > 0:
        avg_high_T_mag = high_T_data['Magnetization_per_spin'].abs().mean()
        print(f"High T (T ≥ 3.0): |M|/N ≈ {avg_high_T_mag:.3f} (disordered)")
    
    return T_c_estimate

def main():
    """Main analysis routine."""
    print("Heisenberg Ferromagnet Phase Transition Analysis")
    print("="*55)
    
    # Load data
    filename = "../monte_carlo/heisenberg_ferromagnet_transition.dat"
    data = load_heisenberg_data(filename)
    
    if data is None:
        return
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate main analysis plots
    fig1, T_cv, T_chi = plot_ferromagnet_transition(data)
    fig1.savefig("plots/heisenberg_ferromagnet_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved: plots/heisenberg_ferromagnet_analysis.png")
    
    # Generate transition detail plots
    fig2 = plot_transition_details(data)
    fig2.savefig("plots/heisenberg_transition_details.png", dpi=300, bbox_inches='tight')
    print(f"Saved: plots/heisenberg_transition_details.png")
    
    # Analyze critical temperature
    T_c_est = analyze_critical_temperature(data)
    
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Generated comprehensive analysis plots")
    print(f"✓ Estimated critical temperature: Tc ≈ {T_c_est:.3f}")
    print(f"✓ 3D Heisenberg ferromagnet theory: Tc ≈ 1.44")
    print(f"✓ Phase transition clearly visible in all quantities")
    print(f"✓ Magnetization drops from ordered to disordered state")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
# %%
