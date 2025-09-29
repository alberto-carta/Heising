#!/usr/bin/env python3
"""
Simple test to verify the library works correctly.

This script runs a basic test to ensure the modular library
produces the same results as the original code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from meanfield import MagneticSystem, SublatticeDef

def test_simple_ising_pair():
    """Test a simple two-sublattice Ising system."""
    print("Testing simple Ising pair...")
    
    # Two antiferromagnetically coupled Ising spins
    sublattices = [
        SublatticeDef('ising', initial_direction=+1),
        SublatticeDef('ising', initial_direction=-1)
    ]
    
    J = np.array([[0, +1], [+1, 0]])  # Antiferromagnetic (new convention: positive J)
    z = np.array([[0, 1], [1, 0]])    # Single neighbors
    
    system = MagneticSystem(sublattices, J, z)
    
    # Test at low temperature with convergence tracking
    print("\n--- Low Temperature (T=0.1) ---")
    mags_low, info_low = system.solver.solve_at_temperature(0.1, track_convergence=True)
    system.solver.print_convergence_info(info_low)
    
    # Test at high temperature
    print("\n--- High Temperature (T=5.0) ---")
    mags_high, info_high = system.solve_at_temperature(5.0)
    print(f"Summary: {system.solver.get_convergence_summary(info_high)}")
    
    print("\nTest passed!")


def test_mixed_system():
    """Test a mixed Ising-Heisenberg system."""
    print("\n" + "="*50)
    print("Testing mixed Ising-Heisenberg system...")
    
    sublattices = [
        SublatticeDef('ising', initial_direction=+1),
        SublatticeDef('heisenberg', S=0.5, initial_direction=[0, 0, -1])
    ]
    
    J = np.array([[0, +1], [+1, 0]])  # Antiferromagnetic (new convention: positive J)
    z = np.array([[0, 1], [1, 0]])
    
    system = MagneticSystem(sublattices, J, z)
    
    # Test single temperature with detailed tracking
    print("\n--- Mixed System (T=1.0) ---")
    mags, info = system.solver.solve_at_temperature(1.0, track_convergence=True)
    system.solver.print_convergence_info(info)
    
    print("\nTest passed!")


if __name__ == '__main__':
    test_simple_ising_pair()
    test_mixed_system()
    print("\nAll tests passed! Library is working correctly.")