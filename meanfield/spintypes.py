"""
Individual Spin Type Implementations

This module contains classes for different types of individual magnetic spins.
Each spin type implements methods to calculate its magnetization given an effective field.
Note: These are individual spin behaviors, not many-body models (e.g., not the full Ising model).
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


def brillouin_function(x: np.ndarray, S: float) -> np.ndarray:
    """
    Calculate the Brillouin function B_S(x).
    
    The Brillouin function describes the magnetization of a quantum spin S
    in a magnetic field, given by:
    B_S(x) = (2S+1)/(2S) * coth((2S+1)x/(2S)) - 1/(2S) * coth(x/(2S))
    
    Parameters
    ----------
    x : np.ndarray
        Dimensionless field parameter (β * μ * H / ℏ)
    S : float
        Spin quantum number (must be positive)
        
    Returns
    -------
    np.ndarray
        Brillouin function values, same shape as x
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0, 2.0])
    >>> B_half = brillouin_function(x, 0.5)  # S = 1/2 case
    """
    if S <= 0:
        return np.zeros_like(x)
    
    # Handle x=0 case to avoid division by zero
    eps = 1e-12
    x_safe = np.where(np.abs(x) < eps, eps, x)
    
    # Calculate terms separately for numerical stability
    alpha = (2 * S + 1) / (2 * S)
    beta = 1 / (2 * S)
    
    # Use coth(y) = 1/tanh(y), with protection against overflow
    arg1 = alpha * x_safe
    arg2 = beta * x_safe
    
    # Protect against overflow in tanh
    arg1_clipped = np.clip(arg1, -700, 700)
    arg2_clipped = np.clip(arg2, -700, 700)
    
    term1 = alpha / np.tanh(arg1_clipped)
    term2 = beta / np.tanh(arg2_clipped)
    
    return term1 - term2


class BaseSpinType(ABC):
    """
    Abstract base class for individual magnetic spin types.
    
    All spin types should inherit from this class and implement
    the calculate_magnetization method.
    """
    
    @abstractmethod
    def calculate_magnetization(self, 
                              effective_field: Union[float, np.ndarray],
                              temperature: float) -> Union[float, np.ndarray]:
        """
        Calculate equilibrium magnetization given effective field and temperature.
        
        Parameters
        ----------
        effective_field : float or np.ndarray
            Effective magnetic field acting on the spin
        temperature : float
            Temperature (in energy units)
            
        Returns
        -------
        float or np.ndarray
            Equilibrium magnetization
        """
        pass
    
    @abstractmethod
    def get_max_magnetization(self) -> Union[float, np.ndarray]:
        """Return the maximum possible magnetization for this model."""
        pass


class IsingSpinType(BaseSpinType):
    """
    Individual classical spin (S = ±1) as used in Ising models.
    
    Represents a single classical spin that can only point up (+1) or down (-1).
    The magnetization follows: m = tanh(βh)
    
    Parameters
    ----------
    initial_direction : float, optional
        Initial spin direction (+1 or -1), by default +1
        
    Examples
    --------
    >>> spin = IsingSpinType(initial_direction=-1)
    >>> m = spin.calculate_magnetization(effective_field=2.0, temperature=1.0)
    """
    
    def __init__(self, initial_direction: float = 1.0):
        self.initial_direction = np.sign(initial_direction) if initial_direction != 0 else 1.0
    
    def calculate_magnetization(self, 
                              effective_field: Union[float, np.ndarray], 
                              temperature: float) -> float:
        """
        Calculate Ising magnetization: m = tanh(βh).
        
        Parameters
        ----------
        effective_field : float or np.ndarray
            Effective magnetic field. If array, uses magnitude and z-component sign
        temperature : float
            Temperature (must be positive)
            
        Returns
        -------
        float
            Magnetization between -1 and +1
        """
        if temperature <= 0:
            return self.initial_direction
            
        beta = 1.0 / temperature
        
        # Handle both scalar and vector effective fields
        if isinstance(effective_field, np.ndarray):
            h_magnitude = np.linalg.norm(effective_field)
            # Use z-component sign if available, otherwise use magnitude sign
            if len(effective_field) > 2:
                sign = np.sign(effective_field[2]) if abs(effective_field[2]) > 1e-9 else 1
            else:
                sign = np.sign(h_magnitude) if h_magnitude > 1e-9 else 1
        else:
            h_magnitude = abs(effective_field)
            sign = np.sign(effective_field) if abs(effective_field) > 1e-9 else 1
        
        return np.tanh(beta * h_magnitude) * sign
    
    def get_max_magnetization(self) -> float:
        """Return maximum Ising spin magnetization (always 1)."""
        return 1.0


class HeisenbergSpinType(BaseSpinType):
    """
    Individual quantum Heisenberg spin with arbitrary spin S.
    
    Represents a single quantum spin with magnitude S that can point in any direction.
    The magnetization follows the Brillouin function: m = S * B_S(βh) * h/|h|
    
    Parameters
    ----------
    S : float
        Spin quantum number (must be positive)
    initial_direction : np.ndarray, optional
        Initial spin direction vector [x, y, z], by default [0, 0, 1]
        
    Examples
    --------
    >>> spin = HeisenbergSpinType(S=0.5, initial_direction=[0, 0, -1])
    >>> field = np.array([0, 0, 1.5])
    >>> m = spin.calculate_magnetization(field, temperature=1.0)
    """
    
    def __init__(self, S: float, initial_direction: np.ndarray = None):
        if S <= 0:
            raise ValueError("Spin quantum number S must be positive")
        
        self.S = float(S)
        
        if initial_direction is None:
            self.initial_direction = np.array([0.0, 0.0, 1.0])
        else:
            self.initial_direction = np.array(initial_direction, dtype=float)
            # Normalize if non-zero
            norm = np.linalg.norm(self.initial_direction)
            if norm > 1e-9:
                self.initial_direction = self.initial_direction / norm
            else:
                self.initial_direction = np.array([0.0, 0.0, 1.0])
    
    def calculate_magnetization(self, 
                              effective_field: Union[float, np.ndarray], 
                              temperature: float) -> np.ndarray:
        """
        Calculate Heisenberg magnetization: m = S * B_S(βh) * h/|h|.
        
        Parameters
        ----------
        effective_field : float or np.ndarray
            Effective magnetic field vector or scalar
        temperature : float
            Temperature (must be positive)
            
        Returns
        -------
        np.ndarray
            Magnetization vector [mx, my, mz]
        """
        if temperature <= 0:
            return self.S * self.initial_direction
            
        beta = 1.0 / temperature
        
        # Convert scalar field to vector (z-direction)
        if isinstance(effective_field, (int, float)):
            field_vector = np.array([0.0, 0.0, float(effective_field)])
        else:
            field_vector = np.array(effective_field, dtype=float)
        
        h_magnitude = np.linalg.norm(field_vector)
        
        # Handle zero field case
        if h_magnitude < 1e-12:
            return np.array([0.0, 0.0, 0.0])
        
        # Calculate Brillouin function value
        brillouin_val = brillouin_function(beta * h_magnitude, self.S)
        
        # Magnetization points along field direction
        field_direction = field_vector / h_magnitude
        return self.S * brillouin_val * field_direction
    
    def get_max_magnetization(self) -> float:
        """Return maximum Heisenberg spin magnetization (S)."""
        return self.S