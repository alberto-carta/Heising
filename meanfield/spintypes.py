"""
Individual Spin Type Implementations

This module contains classes for different types of individual magnetic spins.
Each spin type implements methods to calculate its magnetization given an effective field.
Note: These are individual spin behaviors, not many-body models (e.g., not the full Ising model).
"""

from abc import ABC, abstractmethod
import numpy as np


def brillouin_function(x, S):
    """
    Calculate the Brillouin function for quantum spins.
    
    This function determines how strongly a quantum spin aligns with a magnetic field.
    It's the quantum mechanical equivalent of tanh() for classical spins.
    
    Mathematical form:
    B_S(x) = (2S+1)/(2S) * coth((2S+1)x/(2S)) - 1/(2S) * coth(x/(2S))
    
    Parameters
    ----------
    x : float or array
        Field strength divided by temperature: field/temperature
    S : float
        Spin quantum number (0.5 for electrons, 1.0 for typical ions, etc.)
        
    Returns
    -------
    float or array
        Brillouin function values between -1 and +1
        - Same shape as input x
        - Approaches ±1 for strong fields or low temperature
        - Approaches 0 for weak fields or high temperature
        
    Examples
    --------
    >>> # Electron spin (S=1/2) in moderate field
    >>> B = brillouin_function(1.0, S=0.5)
    >>> 
    >>> # Classical limit (large S) behaves like tanh(x)
    >>> B_classical = brillouin_function(1.0, S=100)  # ≈ tanh(1.0)
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
    def calculate_magnetization(self, effective_field, temperature):
        """
        Calculate equilibrium magnetization given effective field and temperature.
        
        Parameters
        ----------
        effective_field : float or list/array [x, y, z]
            Effective magnetic field acting on the spin:
            - For Ising spins: provide a number (e.g., 1.5)
            - For Heisenberg spins: provide [Hx, Hy, Hz] or just a number for Hz
        temperature : float
            Temperature in energy units (must be positive)
            
        Returns
        -------
        For Ising spins: float
            Magnetization value between -1 and +1
        For Heisenberg spins: array [mx, my, mz]
            Magnetization vector components
        """
        pass
    
    @abstractmethod
    def get_max_magnetization(self):
        """
        Return the maximum possible magnetization for this spin type.
        
        Returns
        -------
        For Ising spins: float
            Always returns 1.0
        For Heisenberg spins: float  
            Returns the spin quantum number S
        """
        pass


class IsingSpinType(BaseSpinType):
    """
    Individual classical Ising spin that can point up (+1) or down (-1).
    
    This represents a single classical magnetic moment that can only be in two states.
    The equilibrium magnetization follows: m = tanh(field/temperature)
    
    Parameters
    ----------
    initial_direction : float, optional
        Starting spin direction: +1 (up) or -1 (down)
        Default is +1 (spin up)
        
    Examples
    --------
    >>> # Create a spin that starts pointing down
    >>> spin = IsingSpinType(initial_direction=-1)
    >>> 
    >>> # Calculate magnetization at T=1.0 with field H=2.0
    >>> magnetization = spin.calculate_magnetization(effective_field=2.0, temperature=1.0)
    >>> print(f"Magnetization: {magnetization:.3f}")  # Should be close to +1
    """
    
    def __init__(self, initial_direction=1.0):
        """
        Create an Ising spin.
        
        Parameters
        ----------
        initial_direction : float
            Starting direction: +1 for up, -1 for down
        """
        self.initial_direction = np.sign(initial_direction) if initial_direction != 0 else 1.0
    
    def calculate_magnetization(self, effective_field, temperature):
        """
        Calculate the equilibrium magnetization of this Ising spin.
        
        Parameters
        ----------
        effective_field : float or array-like
            Magnetic field acting on the spin:
            - If number: field strength (positive = favors spin up)
            - If array [Hx, Hy, Hz]: uses Hz component and overall magnitude
        temperature : float
            Temperature (must be positive)
            Higher temperature → more disorder → magnetization closer to 0
            
        Returns
        -------
        float
            Magnetization value between -1 (fully down) and +1 (fully up)
            
        Examples
        --------
        >>> spin = IsingSpinType()
        >>> m_hot = spin.calculate_magnetization(1.0, temperature=10.0)   # ≈ 0.1 (disordered)
        >>> m_cold = spin.calculate_magnetization(1.0, temperature=0.1)   # ≈ 1.0 (ordered)
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
    Individual quantum Heisenberg spin that can point in any direction.
    
    This represents a quantum magnetic moment with spin quantum number S.
    Unlike Ising spins, it can point in any 3D direction.
    The equilibrium magnetization follows the Brillouin function.
    
    Parameters
    ----------
    S : float
        Spin quantum number (must be positive):
        - S = 0.5: electron spin
        - S = 1.0: typical magnetic ion
        - S = 100: classical limit (large S)
    initial_direction : list [x, y, z], optional
        Starting spin direction vector
        Default is [0, 0, 1] (pointing up along z-axis)
        Will be automatically normalized
        
    Examples
    --------
    >>> # Create electron spin (S=1/2) pointing down initially
    >>> spin = HeisenbergSpinType(S=0.5, initial_direction=[0, 0, -1])
    >>> 
    >>> # Apply field in +z direction
    >>> field = [0, 0, 1.5]  # Field along z-axis
    >>> magnetization = spin.calculate_magnetization(field, temperature=1.0)
    >>> print(f"Magnetization vector: {magnetization}")  # [0, 0, some_positive_value]
    """
    
    def __init__(self, S, initial_direction=None):
        """
        Create a Heisenberg spin.
        
        Parameters
        ----------
        S : float
            Spin quantum number (must be positive)
            Common values: 0.5 (electron), 1.0 (typical ion), 100 (classical)
        initial_direction : list [x, y, z], optional
            Starting direction vector, default is [0, 0, 1] (up)
        """
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
    
    def calculate_magnetization(self, effective_field, temperature):
        """
        Calculate the equilibrium magnetization vector of this Heisenberg spin.
        
        Parameters
        ----------
        effective_field : float or list/array [Hx, Hy, Hz]
            Magnetic field acting on the spin:
            - If number: field along z-axis (e.g., 1.5 means field = [0, 0, 1.5])
            - If list/array: field vector [Hx, Hy, Hz]
        temperature : float
            Temperature (must be positive)
            Higher temperature → smaller magnetization magnitude
            
        Returns
        -------
        array [mx, my, mz]
            Magnetization vector components:
            - Points in same direction as the effective field
            - Magnitude depends on field strength, temperature, and spin S
            - Each component is between -S and +S
            
        Examples
        --------
        >>> spin = HeisenbergSpinType(S=1.0)
        >>> 
        >>> # Field along z-axis
        >>> m = spin.calculate_magnetization(2.0, temperature=0.5)
        >>> # Result: [0, 0, some_value] - only z-component non-zero
        >>> 
        >>> # Field in xy-plane  
        >>> m = spin.calculate_magnetization([1.0, 1.0, 0], temperature=1.0)
        >>> # Result: [mx, my, 0] - magnetization in xy-plane
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