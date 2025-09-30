"""
Mean Field Equation Solvers

This module contains solvers for self-consistent mean field equations.
The solvers use iterative methods to find equilibrium magnetizations.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .spintypes import BaseSpinType
from .fields import FieldCalculator


class MeanFieldSolver:
    """
    Self-consistent mean field equation solver with detailed convergence tracking.
    
    This solver iterates between calculating effective fields and updating
    magnetizations until convergence is achieved. It provides detailed
    information about the convergence process.
    
    Parameters
    ----------
    spin_types : list
        List of spin types for each sublattice. Can mix:
        - IsingSpinType: produces scalar magnetizations (float)
        - HeisenbergSpinType: produces vector magnetizations (np.ndarray)
    field_calculator : FieldCalculator
        Calculator for effective fields
    max_iterations : int, optional
        Maximum number of iterations, by default 500
    tolerance : float, optional
        Convergence tolerance, by default 1e-6
    mixing_parameter : float, optional
        Mixing parameter for stability (0 < α ≤ 1), by default 0.1
        New magnetization = (1-α)*old + α*calculated
        
    Examples
    --------
    >>> from meanfield import IsingSpinType, HeisenbergSpinType, MeanFieldSolver
    >>> spin_types = [IsingSpinType(), HeisenbergSpinType(S=0.5)]
    >>> # ... set up field calculator ...
    >>> solver = MeanFieldSolver(spin_types, field_calculator)
    >>> 
    >>> # Solve with convergence tracking
    >>> mags, info = solver.solve_at_temperature(1.0, track_convergence=True)
    >>> solver.print_convergence_info(info)
    """
    
    def __init__(self,
                 spin_types,
                 field_calculator,
                 max_iterations=500,
                 tolerance=1e-6,
                 mixing_parameter=0.1):
        
        self.spin_types = spin_types
        self.field_calculator = field_calculator
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.mixing_parameter = mixing_parameter
        
        if not (0 < mixing_parameter <= 1):
            raise ValueError("Mixing parameter must be in (0, 1]")
    
    def _initialize_magnetizations(self, initial_guess: Optional[List] = None) -> List:
        """
        Initialize magnetizations for iteration.
        
        Parameters
        ----------
        initial_guess : List[Union[float, np.ndarray]], optional
            Initial magnetization guess. If None, uses model defaults.
            
        Returns
        -------
        List[Union[float, np.ndarray]]
            Initial magnetizations for all sublattices
        """
        if initial_guess is not None:
            if len(initial_guess) != len(self.spin_types):
                raise ValueError("Initial guess length must match number of spin types")
            return [np.copy(mag) if isinstance(mag, np.ndarray) else mag 
                   for mag in initial_guess]
        
        # Use default initialization for each spin type
        magnetizations = []
        for i, spin_type in enumerate(self.spin_types):
            if hasattr(spin_type, 'initial_direction'):
                if isinstance(spin_type.initial_direction, np.ndarray):
                    # Heisenberg: scale by 80% of max magnetization
                    mag = 0.8 * spin_type.get_max_magnetization() * spin_type.initial_direction
                else:
                    # Ising: use initial direction with 80% magnitude
                    mag = 0.8 * spin_type.initial_direction
            else:
                # Fallback: alternating pattern
                if isinstance(spin_type.get_max_magnetization(), (int, float)):
                    mag = 0.8 * ((-1) ** i)  # Ising-like
                else:
                    mag = 0.8 * np.array([0.0, 0.0, (-1) ** i])  # Heisenberg-like
            
            magnetizations.append(mag)
        
        return magnetizations
    
    def _check_convergence(self, old_mags: List, new_mags: List) -> bool:
        """
        Check if magnetizations have converged.
        
        Parameters
        ----------
        old_mags : List[Union[float, np.ndarray]]
            Previous magnetizations
        new_mags : List[Union[float, np.ndarray]]
            New magnetizations
            
        Returns
        -------
        bool
            True if converged within tolerance
        """
        for old_mag, new_mag in zip(old_mags, new_mags):
            if not np.allclose(old_mag, new_mag, atol=self.tolerance):
                return False
        return True
    
    def _rattle_magnetizations(self, magnetizations: List, rattle_strength: float = 0.1) -> List:
        """
        Add random perturbations to average magnetization values used in effective field calculations.
        
        This rattles the magnetization values that feed into the effective field calculations,
        rather than the spin directions themselves. This is more physically meaningful as it
        perturbs the mean field values while respecting the underlying spin constraints.
        
        Parameters
        ----------
        magnetizations : List[Union[float, np.ndarray]]
            Current magnetizations
        rattle_strength : float, optional
            Strength of random perturbation (0 to 1), by default 0.1
            
        Returns
        -------
        List[Union[float, np.ndarray]]
            Rattled magnetizations
        """
        rattled_mags = []
        
        for i, (mag, spin_type) in enumerate(zip(magnetizations, self.spin_types)):
            if isinstance(mag, np.ndarray):
                # Heisenberg magnetization: add random vector perturbation
                random_vec = np.random.normal(0, rattle_strength, size=3)
                new_mag = mag + random_vec
                # Keep within reasonable bounds (don't exceed maximum possible magnetization)
                max_mag = spin_type.get_max_magnetization()
                if np.linalg.norm(new_mag) > max_mag:
                    new_mag = new_mag / np.linalg.norm(new_mag) * max_mag
                rattled_mags.append(new_mag)
            else:
                # Ising magnetization: add random perturbation, keep in valid range
                random_val = np.random.normal(0, rattle_strength)
                new_mag = mag + random_val
                # Keep within physical bounds [-1, +1] for Ising
                new_mag = np.clip(new_mag, -1.0, 1.0)
                rattled_mags.append(new_mag)
        
        return rattled_mags
    
    def solve_at_temperature(self,
                           temperature: float,
                           initial_guess: Optional[List] = None,
                           track_convergence: bool = False,
                           rattle_iterations: int = 0,
                           rattle_strength: float = 0.1,
                           **field_kwargs) -> Tuple[List, Dict[str, Any]]:
        """
        Solve mean field equations at a given temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature (in energy units)
        initial_guess : List, optional
            Initial magnetization guess. Each element can be:
            - float (for Ising spins)
            - np.ndarray([x,y,z]) (for Heisenberg spins)
        track_convergence : bool, optional
            If True, store magnetization history for each iteration
        rattle_iterations : int, optional
            Number of iterations with spin rattling at the beginning, by default 0
        rattle_strength : float, optional
            Strength of random perturbations (0 to 1), by default 0.1
        **field_kwargs
            Additional parameters for field calculation
            
        Returns
        -------
        Tuple[List, Dict[str, Any]]
            (final_magnetizations, convergence_info)
            
        Examples
        --------
        >>> mags, info = solver.solve_at_temperature(1.0)
        >>> print(f"Converged in {info['iterations']} iterations")
        >>> if info['converged']:
        >>>     print("Final magnetizations:", mags)
        """
        # Initialize magnetizations
        magnetizations = self._initialize_magnetizations(initial_guess)
        
        # Convergence tracking
        convergence_history = []
        field_history = []
        if track_convergence:
            convergence_history.append([np.copy(mag) if isinstance(mag, np.ndarray) else mag 
                                      for mag in magnetizations])
        
        # Iteration loop
        for iteration in range(self.max_iterations):
            # Apply rattling for the first few iterations
            if iteration < rattle_iterations:
                magnetizations = self._rattle_magnetizations(magnetizations, rattle_strength)
            # Calculate effective fields for all sublattices
            effective_fields = []
            for i in range(len(self.spin_types)):
                h_eff = self.field_calculator.calculate_field(i, magnetizations)
                effective_fields.append(h_eff)
            
            if track_convergence:
                field_history.append([np.copy(field) if isinstance(field, np.ndarray) else field 
                                    for field in effective_fields])
            
            # Update magnetizations using spin types
            new_magnetizations = []
            for spin_type, h_field in zip(self.spin_types, effective_fields):
                new_mag = spin_type.calculate_magnetization(h_field, temperature)
                new_magnetizations.append(new_mag)
            
            # Check convergence
            converged = self._check_convergence(magnetizations, new_magnetizations)
            
            # Apply mixing for stability
            mixed_magnetizations = []
            for i in range(len(magnetizations)):
                old_mag = magnetizations[i]
                new_mag = new_magnetizations[i]
                mixed_mag = ((1 - self.mixing_parameter) * old_mag + 
                           self.mixing_parameter * new_mag)
                mixed_magnetizations.append(mixed_mag)
            
            magnetizations = mixed_magnetizations
            
            if track_convergence:
                convergence_history.append([np.copy(mag) if isinstance(mag, np.ndarray) else mag 
                                          for mag in magnetizations])
            
            if converged:
                convergence_info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_fields': effective_fields,
                    'temperature': temperature,
                    'final_magnetizations': magnetizations
                }
                
                if track_convergence:
                    convergence_info['magnetization_history'] = convergence_history
                    convergence_info['field_history'] = field_history
                
                return magnetizations, convergence_info
        
        # Did not converge
        convergence_info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_fields': effective_fields,
            'temperature': temperature,
            'warning': f'Did not converge within {self.max_iterations} iterations',
            'final_magnetizations': magnetizations
        }
        
        if track_convergence:
            convergence_info['magnetization_history'] = convergence_history
            convergence_info['field_history'] = field_history
        
        return magnetizations, convergence_info
    
    def solve_temperature_sweep(self,
                              temperatures: np.ndarray,
                              initial_guess: Optional[List] = None,
                              use_previous: bool = True,
                              reverse_order: bool = False,
                              rattle_iterations: int = 0,
                              rattle_strength: float = 0.1,
                              **field_kwargs) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Solve equations for a range of temperatures.
        
        Parameters
        ----------
        temperatures : np.ndarray
            Array of temperatures to solve
        initial_guess : List[Union[float, np.ndarray]], optional
            Initial guess for first temperature in the sweep
        use_previous : bool, optional
            Whether to use previous solution as next initial guess (adiabatic continuation), by default True
        reverse_order : bool, optional
            If True, sweep temperatures in descending order (high to low), by default False
        rattle_iterations : int, optional
            Number of rattling iterations at each temperature, by default 0
        rattle_strength : float, optional
            Strength of random perturbations (0 to 1), by default 0.1
        **field_kwargs
            Additional parameters for field calculation
            
        Returns
        -------
        Tuple[np.ndarray, List[Dict[str, Any]]]
            (magnetizations_array, convergence_info_list)
            magnetizations_array has shape (n_temps, n_sublattices, ...)
            
        Examples
        --------
        >>> temps = np.linspace(0.1, 5.0, 50)
        >>> mags_vs_T, infos = solver.solve_temperature_sweep(temps)
        """
        n_temps = len(temperatures)
        n_sublattices = len(self.spin_types)
        
        # Determine sweep order
        if reverse_order:
            temp_indices = list(reversed(range(n_temps)))
            temp_sequence = temperatures[temp_indices]
        else:
            temp_indices = list(range(n_temps))
            temp_sequence = temperatures
        
        # Arrays to store results in original temperature order
        magnetizations_vs_T = [None] * n_temps
        convergence_infos = [None] * n_temps
        
        current_guess = initial_guess
        
        for seq_idx, temp_idx in enumerate(temp_indices):
            T = temp_sequence[seq_idx]
            mags, info = self.solve_at_temperature(T, current_guess, 
                                                 rattle_iterations=rattle_iterations,
                                                 rattle_strength=rattle_strength,
                                                 **field_kwargs)
            
            # Store results in original temperature order
            magnetizations_vs_T[temp_idx] = [np.copy(mag) if isinstance(mag, np.ndarray) else mag 
                                           for mag in mags]
            convergence_infos[temp_idx] = info
            
            # Use current solution as next initial guess (adiabatic continuation)
            if use_previous:
                current_guess = mags
            
            # Check for convergence issues
            if not info['converged']:
                print(f"Warning: Did not converge at T = {T:.3f}")
        
        # Convert to structured array
        result_array = np.empty((n_temps, n_sublattices), dtype=object)
        for i, mags_at_T in enumerate(magnetizations_vs_T):
            for j, mag in enumerate(mags_at_T):
                result_array[i, j] = mag
        
        return result_array, convergence_infos
    
    def print_convergence_info(self, convergence_info: Dict[str, Any]) -> None:
        """
        Print detailed convergence information in a readable format.
        
        Parameters
        ----------
        convergence_info : Dict[str, Any]
            Convergence information from solve_at_temperature
        """
        print(f"Temperature: {convergence_info['temperature']:.4f}")
        print(f"Converged: {convergence_info['converged']}")
        print(f"Iterations: {convergence_info['iterations']}")
        
        if 'warning' in convergence_info:
            print(f"Warning: {convergence_info['warning']}")
        
        print("Final magnetizations:")
        for i, mag in enumerate(convergence_info['final_magnetizations']):
            if isinstance(mag, np.ndarray):
                print(f"  Sublattice {i}: [{mag[0]:8.5f}, {mag[1]:8.5f}, {mag[2]:8.5f}] (Heisenberg)")
            else:
                print(f"  Sublattice {i}: {mag:8.5f} (Ising)")
        
        if 'magnetization_history' in convergence_info:
            print(f"\nConvergence history ({len(convergence_info['magnetization_history'])} steps):")
            history = convergence_info['magnetization_history']
            for step, mags in enumerate(history):
                if step % 10 == 0 or step == len(history) - 1:  # Print every 10th step + final
                    print(f"  Step {step:3d}:", end="")
                    for i, mag in enumerate(mags):
                        if isinstance(mag, np.ndarray):
                            mag_norm = np.linalg.norm(mag)
                            print(f" |m{i}|={mag_norm:6.4f}", end="")
                        else:
                            print(f" m{i}={mag:7.4f}", end="")
                    print()
    
    def get_convergence_summary(self, convergence_info: Dict[str, Any]) -> str:
        """Return a brief convergence summary as a string."""
        status = "✓" if convergence_info['converged'] else "✗"
        return (f"{status} T={convergence_info['temperature']:.3f}, "
               f"iter={convergence_info['iterations']}")