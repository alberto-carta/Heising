"""
Visualization and Analysis Tools

This module provides functions for plotting magnetization results
and finding critical temperatures, directly translated from the original code.
"""

from typing import List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_magnetizations(temperatures: np.ndarray,
                       magnetizations_list: List[np.ndarray],
                       sublattice_params: List[dict],
                       title: str = "Magnetization vs Temperature") -> None:
    """
    Plot magnetization vs temperature for multiple sublattices.
    
    Enhanced version with improved markers: squares for Ising, circles for Heisenberg,
    larger marker sizes for better visibility.
    
    Parameters
    ----------
    temperatures : np.ndarray
        Array of temperatures
    magnetizations_list : List[np.ndarray]
        List of magnetization arrays for each sublattice
    sublattice_params : List[dict]
        List of sublattice parameters
    title : str, optional
        Plot title, by default "Magnetization vs Temperature"
        
    Examples
    --------
    >>> temps = np.linspace(0.1, 5.0, 50)
    >>> mags_list = [mag_array_0, mag_array_1, ...]
    >>> params = [{'model': 'ising'}, {'model': 'heisenberg', 'S': 0.5}]
    >>> plot_magnetizations(temps, mags_list, params)
    """
    plt.figure(figsize=(12, 8))
    
    # Enhanced color palette with better contrast
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, (magnetizations, params) in enumerate(zip(magnetizations_list, sublattice_params)):
        color = colors[i % len(colors)]
        
        # Choose marker based on model type
        if params['model'] == 'heisenberg':
            marker = 'o'  # Circle for Heisenberg
            # For Heisenberg spins, plot z-component normalized by S
            if magnetizations.ndim == 2:  # Vector case (shape is [temps, 3])
                mag_values = magnetizations[:, 2] / params['S']  # z-component normalized by S
                label = f'$m_{{{i+1},z}}/S$ (Heisenberg S={params["S"]})'
            else:
                # Fallback if somehow scalar
                mag_values = magnetizations / params['S']
                label = f'$m_{{{i+1}}}/S$ (Heisenberg S={params["S"]})'
        else:  # Ising
            marker = 's'  # Square for Ising
            # For Ising spins, magnetization is already normalized (max = 1)
            mag_values = magnetizations
            label = f'$m_{{{i+1}}}$ (Ising)'
        
        plt.plot(temperatures, mag_values, marker=marker, linestyle='-', 
                label=label, markersize=8, color=color, alpha=0.8, 
                linewidth=2, markeredgecolor='white', markeredgewidth=0.5)
    
    plt.xlabel('Temperature (T/|J|)', fontsize=14)
    plt.ylabel('Normalized Sublattice Magnetization', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()


def find_critical_temperature(temperatures: np.ndarray,
                            magnetizations: np.ndarray,
                            threshold: float = 0.1) -> Optional[float]:
    """
    Find critical temperature where magnetization drops below threshold.
    
    This translates the critical temperature finding logic from the original code.
    
    Parameters
    ----------
    temperatures : np.ndarray
        Temperature array
    magnetizations : np.ndarray
        Magnetization array (can be scalar or vector magnetizations)
    threshold : float, optional
        Magnetization threshold for critical point, by default 0.1
        
    Returns
    -------
    Optional[float]
        Critical temperature, or None if not found in range
        
    Examples
    --------
    >>> Tc = find_critical_temperature(temps, mags[:, 0], threshold=0.1)
    >>> if Tc:
    ...     print(f"Critical temperature: {Tc:.3f}")
    """
    # Calculate magnetization magnitude
    if magnetizations.ndim == 1:
        # Scalar magnetizations (Ising)
        abs_mag = np.abs(magnetizations)
    else:
        # Vector magnetizations (Heisenberg) - use magnitude
        abs_mag = np.linalg.norm(magnetizations, axis=1)
    
    # Find first point below threshold
    critical_indices = np.where(abs_mag < threshold)[0]
    
    if len(critical_indices) > 0:
        return temperatures[critical_indices[0]]
    else:
        return None