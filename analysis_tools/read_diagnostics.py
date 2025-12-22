#%%
"""
Diagnostic Data Reader for Monte Carlo Simulations

Tools for reading configuration dumps and observable evolution data
from MPI Monte Carlo simulations with diagnostics enabled.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class ConfigurationReader:
    """
    Read and parse configuration dump files from Monte Carlo simulations.
    
    Configuration files have format:
        x y z spin_id spin_name spin_type value(s)
    
    For Ising spins:
        x y z spin_id name Ising value
    
    For Heisenberg spins:
        x y z spin_id name Heisenberg sx sy sz
    """
    
    def __init__(self, filepath: str):
        """
        Initialize reader with a configuration dump file.
        
        Args:
            filepath: Path to configuration dump file
        """
        self.filepath = Path(filepath)
        self.metadata = {}
        self.data = None
        self._read_file()
    
    def _read_file(self):
        """Read and parse the configuration file."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header metadata
        data_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Parse metadata from comments
                if ':' in line:
                    key_val = line[1:].split(':', 1)
                    if len(key_val) == 2:
                        key = key_val[0].strip()
                        value = key_val[1].strip()
                        self.metadata[key] = value
            elif line:
                data_lines.append(line)
        
        # Parse data into structured format
        configs = []
        for line in data_lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            
            x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
            spin_id = int(parts[3])
            spin_name = parts[4]
            spin_type = parts[5]
            
            if spin_type == 'Ising':
                value = float(parts[6])
                configs.append({
                    'x': x, 'y': y, 'z': z,
                    'spin_id': spin_id,
                    'spin_name': spin_name,
                    'spin_type': spin_type,
                    'value': value,
                    'sx': np.nan, 'sy': np.nan, 'sz': np.nan
                })
            elif spin_type == 'Heisenberg':
                sx, sy, sz = float(parts[6]), float(parts[7]), float(parts[8])
                configs.append({
                    'x': x, 'y': y, 'z': z,
                    'spin_id': spin_id,
                    'spin_name': spin_name,
                    'spin_type': spin_type,
                    'value': np.nan,
                    'sx': sx, 'sy': sy, 'sz': sz
                })
        
        self.data = pd.DataFrame(configs)
    
    def get_temperature(self) -> float:
        """Get temperature from metadata."""
        return float(self.metadata.get('Temperature', 0.0))
    
    def get_rank(self) -> int:
        """Get MPI rank from metadata."""
        return int(self.metadata.get('Rank', 0))
    
    def get_measurement_step(self) -> int:
        """Get measurement step from metadata."""
        return int(self.metadata.get('Measurement step', 0))
    
    def get_lattice_size(self) -> Tuple[int, int, int]:
        """Get lattice dimensions from metadata."""
        size_str = self.metadata.get('Lattice size', '0x0x0')
        sizes = size_str.replace('x', ' ').split()
        if len(sizes) == 3:
            return tuple(map(int, sizes))
        return (0, 0, 0)
    
    def get_spins(self, spin_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get spin data, optionally filtered by type.
        
        Args:
            spin_type: Filter by 'Ising' or 'Heisenberg', or None for all
            
        Returns:
            DataFrame with spin configurations
        """
        if spin_type:
            return self.data[self.data['spin_type'] == spin_type].copy()
        return self.data.copy()
    
    def get_spin_grid(self, spin_name: str, component: Optional[str] = None) -> np.ndarray:
        """
        Get spins arranged in 3D grid for visualization.
        
        Args:
            spin_name: Name of the spin species
            component: For Heisenberg: 'sx', 'sy', 'sz', or None for magnitude
                      For Ising: None (returns value)
        
        Returns:
            3D numpy array of spin values
        """
        Lx, Ly, Lz = self.get_lattice_size()
        spin_data = self.data[self.data['spin_name'] == spin_name]
        
        if spin_data.empty:
            return np.zeros((Lx, Ly, Lz))
        
        grid = np.zeros((Lx, Ly, Lz))
        
        spin_type = spin_data.iloc[0]['spin_type']
        
        for _, row in spin_data.iterrows():
            x, y, z = row['x'] - 1, row['y'] - 1, row['z'] - 1  # Convert to 0-indexed
            
            if spin_type == 'Ising':
                grid[x, y, z] = row['value']
            elif spin_type == 'Heisenberg':
                if component == 'sx':
                    grid[x, y, z] = row['sx']
                elif component == 'sy':
                    grid[x, y, z] = row['sy']
                elif component == 'sz':
                    grid[x, y, z] = row['sz']
                else:  # magnitude
                    grid[x, y, z] = np.sqrt(row['sx']**2 + row['sy']**2 + row['sz']**2)
        
        return grid


class ObservableReader:
    """
    Read and parse observable evolution files from Monte Carlo simulations.
    
    File format:
        measurement_step energy magnetization corr_species1 corr_species2 ... acceptance_rate
    """
    
    def __init__(self, filepath: str):
        """
        Initialize reader with observable evolution file.
        
        Args:
            filepath: Path to observable evolution file
        """
        self.filepath = Path(filepath)
        self.metadata = {}
        self.data = None
        self._read_file()
    
    def _read_file(self):
        """Read and parse the observable file."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header for metadata and column names
        column_names = None
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Check if this is the column header line
                if 'measurement_step' in line:
                    column_names = line[1:].strip().split()
                else:
                    # Parse other metadata
                    parts = line[1:].split('for')
                    if len(parts) == 2:
                        info = parts[1].strip().split(',')
                        for item in info:
                            if '=' in item:
                                key, val = item.split('=')
                                self.metadata[key.strip()] = val.strip()
        
        # Read data
        if column_names:
            self.data = pd.read_csv(self.filepath, comment='#', 
                                   names=column_names, sep=r'\s+')
        else:
            self.data = pd.read_csv(self.filepath, comment='#', sep=r'\s+')
    
    def get_temperature(self) -> float:
        """Get temperature from metadata."""
        t_str = self.metadata.get('T', '0')
        return float(t_str)
    
    def get_rank(self) -> int:
        """Get MPI rank from metadata."""
        r_str = self.metadata.get('Rank', '0')
        return int(r_str)
    
    def get_observables(self) -> pd.DataFrame:
        """Get full observable DataFrame."""
        return self.data.copy()
    
    def get_column(self, column_name: str) -> np.ndarray:
        """
        Get specific observable column.
        
        Args:
            column_name: Name of column (e.g., 'energy', 'magnetization', 'acceptance_rate')
        
        Returns:
            Numpy array of values
        """
        if column_name in self.data.columns:
            return self.data[column_name].values
        raise KeyError(f"Column '{column_name}' not found in data")
    
    def get_correlations(self) -> pd.DataFrame:
        """Get all correlation columns."""
        corr_cols = [col for col in self.data.columns if col.startswith('corr_')]
        return self.data[corr_cols].copy()
    
    def plot_evolution(self, columns: Optional[List[str]] = None, ax=None):
        """
        Plot observable evolution over time.
        
        Args:
            columns: List of column names to plot, or None for default observables
            ax: Matplotlib axis to plot on, or None to create new figure
        """
        import matplotlib.pyplot as plt
        
        if columns is None:
            columns = ['energy', 'magnetization']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in columns:
            if col in self.data.columns:
                ax.plot(self.data['measurement_step'], self.data[col], 
                       label=col, marker='o', markersize=3)
        
        ax.set_xlabel('Measurement Step')
        ax.set_ylabel('Observable Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        T = self.get_temperature()
        rank = self.get_rank()
        ax.set_title(f'Observable Evolution (T={T}, Rank={rank})')
        
        return ax


def load_all_configurations(dump_dir: str, pattern: str = "config_*.dat") -> List[ConfigurationReader]:
    """
    Load all configuration dump files from a directory.
    
    Args:
        dump_dir: Path to dumps directory
        pattern: Glob pattern to match configuration files
    
    Returns:
        List of ConfigurationReader objects
    """
    dump_path = Path(dump_dir)
    config_files = sorted(dump_path.glob(pattern))
    return [ConfigurationReader(f) for f in config_files]


def load_all_observables(dump_dir: str, pattern: str = "observables_*.dat") -> List[ObservableReader]:
    """
    Load all observable evolution files from a directory.
    
    Args:
        dump_dir: Path to dumps directory
        pattern: Glob pattern to match observable files
    
    Returns:
        List of ObservableReader objects
    """
    dump_path = Path(dump_dir)
    obs_files = sorted(dump_path.glob(pattern))
    return [ObservableReader(f) for f in obs_files]


def extract_file_info(filepath: str) -> Dict[str, any]:
    """
    Extract rank, temperature, and measurement info from filename.
    
    Args:
        filepath: Path to dump file
    
    Returns:
        Dictionary with 'rank', 'temperature', and optionally 'measurement'
    """
    filename = Path(filepath).name
    info = {}
    
    # Extract rank
    rank_match = re.search(r'rank(\d+)', filename)
    if rank_match:
        info['rank'] = int(rank_match.group(1))
    
    # Extract temperature
    temp_match = re.search(r'T([\d.]+)', filename)
    if temp_match:
        info['temperature'] = float(temp_match.group(1))
    
    # Extract measurement step (for config files)
    meas_match = re.search(r'meas(\d+)', filename)
    if meas_match:
        info['measurement'] = int(meas_match.group(1))
    
    return info


# ============================================================================
# Example Usage - Notebook Style
# ============================================================================

# %% Load all diagnostic files from a directory
# Set the path to your dumps directory
# dump_dir = "../monte_carlo/examples/heisenberg_ferromagnet/dumps"
# dump_dir = "../monte_carlo/examples/ising_ferromagnet/dumps"
dump_dir = "../monte_carlo/examples/4-atom_cell_GAFM_ising/dumps"

# Load all configuration dumps and observable files
configs = load_all_configurations(dump_dir)
obs_list = load_all_observables(dump_dir)

print(f"Found {len(configs)} configuration files")
print(f"Found {len(obs_list)} observable files")

# %% Explore a single configuration file
# Read a specific configuration dump
config = ConfigurationReader(f"{dump_dir}/config_rank0_T2.00_meas400.dat")

print(f"Temperature: {config.get_temperature()}")
print(f"Rank: {config.get_rank()}")
print(f"Measurement step: {config.get_measurement_step()}")
print(f"Lattice size: {config.get_lattice_size()}")
print(f"Total spins: {len(config.data)}")

# View the first few spins
print("\nFirst few spins:")
print(config.data.head())

# %% Get spin configuration as 3D grid
# Extract Heisenberg spins as a 3D grid (z-component)
spin_grid_z = config.get_spin_grid('Fe_moment')
print(f"Spin grid shape: {spin_grid_z.shape}")
print(f"Average sz: {spin_grid_z.mean():.4f}")

# Get full magnitude
spin_magnitude = config.get_spin_grid('Fe_moment')
print(f"Average magnitude: {spin_magnitude.mean():.4f}")

# %% Read observable evolution for a specific rank and temperature
obs = ObservableReader(f"{dump_dir}/observables_rank0_T2.00.dat")

print(f"Temperature: {obs.get_temperature()}")
print(f"Rank: {obs.get_rank()}")
print(f"Number of measurements: {len(obs.data)}")
print(f"\nColumns: {list(obs.data.columns)}")

# View statistics
print("\nObservable statistics:")
print(obs.data[['energy', 'magnetization', 'acceptance_rate']].describe())

# %% Plot observable evolution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot energy evolution
obs.data.plot(x='measurement_step', y='energy', ax=axes[0, 0], marker='o', markersize=2)
axes[0, 0].set_title(f'Energy Evolution (T={obs.get_temperature()}, Rank={obs.get_rank()})')
axes[0, 0].set_ylabel('Energy per spin')
axes[0, 0].grid(True, alpha=0.3)

# Plot magnetization evolution
obs.data.plot(x='measurement_step', y='magnetization', ax=axes[0, 1], marker='o', markersize=2, color='orange')
axes[0, 1].set_title('Magnetization Evolution')
axes[0, 1].set_ylabel('Magnetization per spin')
axes[0, 1].grid(True, alpha=0.3)

# Plot acceptance rate
obs.data.plot(x='measurement_step', y='acceptance_rate', ax=axes[1, 0], marker='o', markersize=2, color='green')
axes[1, 0].set_title('Acceptance Rate Evolution')
axes[1, 0].set_ylabel('Acceptance Rate (%)')
axes[1, 0].grid(True, alpha=0.3)

# Plot correlations
corr_cols = [col for col in obs.data.columns if col.startswith('corr_')]
for col in corr_cols:
    obs.data.plot(x='measurement_step', y=col, ax=axes[1, 1], marker='o', markersize=2, label=col)
axes[1, 1].set_title('Spin Correlations')
axes[1, 1].set_ylabel('Correlation')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Compare different temperatures
# Load observables for different temperatures
# obs_high_T = ObservableReader(f"{dump_dir}/observables_rank0_T2.00.dat")
obs_high_T = ObservableReader(f"{dump_dir}/observables_rank0_T1.40.dat")
# obs_low_T = ObservableReader(f"{dump_dir}/observables_rank0_T1.50.dat")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Compare energy
axes[0].plot(obs_high_T.data['measurement_step'], obs_high_T.data['energy'], 
             label=f'T={obs_high_T.get_temperature():.2f}', marker='o', markersize=2)
axes[0].plot(obs_low_T.data['measurement_step'], obs_low_T.data['energy'], 
             label=f'T={obs_low_T.get_temperature():.2f}', marker='o', markersize=2)
axes[0].set_xlabel('Measurement Step')
axes[0].set_ylabel('Energy per spin')
axes[0].set_title('Energy vs Temperature')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Compare magnetization
axes[1].plot(obs_high_T.data['measurement_step'], obs_high_T.data['magnetization'], 
             label=f'T={obs_high_T.get_temperature():.2f}', marker='o', markersize=2)
axes[1].plot(obs_low_T.data['measurement_step'], obs_low_T.data['magnetization'], 
             label=f'T={obs_low_T.get_temperature():.2f}', marker='o', markersize=2)
axes[1].set_xlabel('Measurement Step')
axes[1].set_ylabel('Magnetization per spin')
axes[1].set_title('Magnetization vs Temperature')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Visualize spin configuration (slice through lattice)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Load a configuration
config = ConfigurationReader(f"{dump_dir}/config_rank0_T2.00_meas3000.dat")

# Get spin components
sx = config.get_spin_grid('Fe_moment', component='sx')
sy = config.get_spin_grid('Fe_moment', component='sy')
sz = config.get_spin_grid('Fe_moment', component='sz')

# Plot middle slice (z = L/2)
z_slice = sx.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot sx component
im0 = axes[0].imshow(sx[:, :, z_slice], cmap='RdBu', norm=Normalize(-1, 1))
axes[0].set_title(f'Sx (z={z_slice})')
axes[0].set_xlabel('y')
axes[0].set_ylabel('x')
plt.colorbar(im0, ax=axes[0])

# Plot sy component
im1 = axes[1].imshow(sy[:, :, z_slice], cmap='RdBu', norm=Normalize(-1, 1))
axes[1].set_title(f'Sy (z={z_slice})')
axes[1].set_xlabel('y')
axes[1].set_ylabel('x')
plt.colorbar(im1, ax=axes[1])

# Plot sz component
im2 = axes[2].imshow(sz[:, :, z_slice], cmap='RdBu', norm=Normalize(-1, 1))
axes[2].set_title(f'Sz (z={z_slice})')
axes[2].set_xlabel('y')
axes[2].set_ylabel('x')
plt.colorbar(im2, ax=axes[2])

plt.suptitle(f'Spin Configuration (T={config.get_temperature():.2f}, Step={config.get_measurement_step()})')
plt.tight_layout()
plt.show()
