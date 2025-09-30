#!/usr/bin/env python3
#%%
"""
Monte Carlo Results Visualization

This script analyzes and plots the results from the Monte Carlo temperature sweeps
for both Ising and Heisenberg models, highlighting phase transition behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def load_data(filename):
    """Load Monte Carlo results from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found!")
        return None
    
    try:
        # Read CSV file, skipping comment lines
        data = pd.read_csv(filename, comment='#')
        print(f"Loaded {len(data)} temperature points from {filename}")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def calculate_thermodynamic_quantities(data):
    """Calculate derived thermodynamic quantities."""
    T = data['Temperature'].values
    M_avg = data['Magnetization'].values
    M_abs_avg = data['AbsMagnetization'].values
    M_sq_avg = data['MagSqAvg'].values
    E_avg = data['Energy'].values
    E_sq_avg = data['EnergySqAvg'].values
    
    # Calculate susceptibility: χ = (⟨M²⟩ - ⟨|M|⟩²) / T
    # Note: Using |M| instead of M for better finite-size behavior
    chi = (M_sq_avg - M_abs_avg**2) / T
    
    # Calculate specific heat: C = (⟨E²⟩ - ⟨E⟩²) / T²
    C_v = (E_sq_avg - E_avg**2) / (T**2)
    
    return chi, C_v

def plot_ising_analysis(data):
    """Create comprehensive plots for Ising model analysis."""
    T = data['Temperature'].values
    M_abs_avg = data['AbsMagnetization'].values / (16 * 16)  # Normalize by number of sites
    E_avg = data['Energy'].values / (16 * 16)  # Normalize by number of sites
    
    chi, C_v = calculate_thermodynamic_quantities(data)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ising Model - Temperature Sweep Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Magnetization vs Temperature
    ax1.plot(T, M_abs_avg, 'bo-', linewidth=2, markersize=4)
    # Note: T_c for AFM Ising is different from FM case
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('⟨|M|⟩ per site')
    ax1.set_title('Absolute Magnetization per Site')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Energy vs Temperature
    ax2.plot(T, E_avg, 'go-', linewidth=2, markersize=4)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('⟨E⟩ per site')
    ax2.set_title('Average Energy per Site')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Magnetic Susceptibility
    ax3.plot(T, chi, 'mo-', linewidth=2, markersize=4)
    # Note: T_c for AFM Ising will be determined from susceptibility peak
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('χ = (⟨M²⟩ - ⟨|M|⟩²)/T')
    ax3.set_title('Magnetic Susceptibility')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Specific Heat
    ax4.plot(T, C_v, 'co-', linewidth=2, markersize=4)
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('C_v = (⟨E²⟩ - ⟨E⟩²)/T²')
    ax4.set_title('Specific Heat')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_heisenberg_analysis(data):
    """Create comprehensive plots for Heisenberg model analysis."""
    T = data['Temperature'].values
    M_abs_avg = data['AbsMagnetization'].values / (16 * 16)  # Normalize by number of sites
    E_avg = data['Energy'].values / (16 * 16)  # Normalize by number of sites
    
    chi, C_v = calculate_thermodynamic_quantities(data)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Heisenberg Model - Temperature Sweep Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Magnetization vs Temperature
    ax1.plot(T, M_abs_avg, 'ro-', linewidth=2, markersize=4)
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('⟨|M|⟩ per site')
    ax1.set_title('Absolute Magnetization per Site')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy vs Temperature
    ax2.plot(T, E_avg, 'go-', linewidth=2, markersize=4)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('⟨E⟩')
    ax2.set_title('Average Energy')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Magnetic Susceptibility
    ax3.plot(T, chi, 'mo-', linewidth=2, markersize=4)
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('χ = (⟨M²⟩ - ⟨|M|⟩²)/T')
    ax3.set_title('Magnetic Susceptibility (Broad Peak)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Specific Heat
    ax4.plot(T, C_v, 'co-', linewidth=2, markersize=4)
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('C_v = (⟨E²⟩ - ⟨E⟩²)/T²')
    ax4.set_title('Specific Heat')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_comparison(ising_data, heisenberg_data):
    """Create side-by-side comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ising vs Heisenberg: Phase Transition Comparison', fontsize=16, fontweight='bold')
    
    # Magnetization comparison (normalized per site)
    T_ising = ising_data['Temperature'].values
    M_ising = ising_data['AbsMagnetization'].values / (16 * 16)
    T_heisenberg = heisenberg_data['Temperature'].values
    M_heisenberg = heisenberg_data['AbsMagnetization'].values / (16 * 16)
    
    ax1.plot(T_ising, M_ising, 'b-', linewidth=2, label='Ising Model', marker='o', markersize=3)
    ax1.plot(T_heisenberg, M_heisenberg, 'r-', linewidth=2, label='Heisenberg Model', marker='s', markersize=3)
# T_c will be determined from data for AFM models
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('⟨|M|⟩ per site')
    ax1.set_title('Magnetization per Site vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Susceptibility comparison
    chi_ising, _ = calculate_thermodynamic_quantities(ising_data)
    chi_heisenberg, _ = calculate_thermodynamic_quantities(heisenberg_data)
    
    ax2.plot(T_ising, chi_ising, 'b-', linewidth=2, label='Ising Model', marker='o', markersize=3)
    ax2.plot(T_heisenberg, chi_heisenberg, 'r-', linewidth=2, label='Heisenberg Model', marker='s', markersize=3)
# T_c will be determined from susceptibility peak
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Magnetic Susceptibility χ')
    ax2.set_title('Susceptibility vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def find_critical_temperature(data, quantity='susceptibility'):
    """Estimate critical temperature from peak in susceptibility or specific heat."""
    T = data['Temperature'].values
    
    if quantity == 'susceptibility':
        chi, _ = calculate_thermodynamic_quantities(data)
        peak_idx = np.argmax(chi)
        T_c = T[peak_idx]
        peak_value = chi[peak_idx]
        print(f"Peak susceptibility: χ_max = {peak_value:.3f} at T = {T_c:.3f}")
    else:
        _, C_v = calculate_thermodynamic_quantities(data)
        peak_idx = np.argmax(C_v)
        T_c = T[peak_idx]
        peak_value = C_v[peak_idx]
        print(f"Peak specific heat: C_v_max = {peak_value:.3f} at T = {T_c:.3f}")
    
    return T_c

def main():
    """Main analysis and plotting routine."""
    print("Monte Carlo Results Analysis")
    print("=" * 40)
    
    # Load data files
    ising_data = load_data("ising_results.dat")
    heisenberg_data = load_data("heisenberg_results.dat")
    
    if ising_data is None and heisenberg_data is None:
        print("No data files found! Please run the Monte Carlo simulation first.")
        return
    
    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Analyze Ising model
    if ising_data is not None:
        print("\nIsing Model Analysis:")
        print("-" * 20)
        T_c_ising = find_critical_temperature(ising_data, 'susceptibility')
        print(f"Estimated T_c from susceptibility peak: {T_c_ising:.3f}")
        print("Note: This is for AFM Ising (J > 0), different from FM case")
        
        fig_ising = plot_ising_analysis(ising_data)
        fig_ising.savefig("plots/ising_analysis.png", dpi=300, bbox_inches='tight')
        print("Saved: plots/ising_analysis.png")
    
    # Analyze Heisenberg model
    if heisenberg_data is not None:
        print("\nHeisenberg Model Analysis:")
        print("-" * 26)
        T_c_heisenberg = find_critical_temperature(heisenberg_data, 'susceptibility')
        print(f"Susceptibility peak at T = {T_c_heisenberg:.3f}")
        print("Note: 2D Heisenberg AFM can have finite-T Néel transition")
        print("Unlike FM case, AFM Heisenberg can break discrete Z2 symmetry")
        
        fig_heisenberg = plot_heisenberg_analysis(heisenberg_data)
        fig_heisenberg.savefig("plots/heisenberg_analysis.png", dpi=300, bbox_inches='tight')
        print("Saved: plots/heisenberg_analysis.png")
    
    # Comparison plot
    if ising_data is not None and heisenberg_data is not None:
        print("\nComparison Analysis:")
        print("-" * 19)
        fig_comparison = plot_comparison(ising_data, heisenberg_data)
        fig_comparison.savefig("plots/model_comparison.png", dpi=300, bbox_inches='tight')
        print("Saved: plots/model_comparison.png")
        
        print("\nKey Observations:")
        print("• Both AFM models can show finite-T phase transitions")
        print("• Ising AFM: Sharp transition (Ising universality class)")
        print("• Heisenberg AFM: May show Néel transition (different from FM case)")
        print("• Susceptibility peaks indicate critical temperatures")
    
    print(f"\nAll plots saved in the 'plots/' directory!")
    print("Analysis complete!")

if __name__ == "__main__":
    main()
# %%
