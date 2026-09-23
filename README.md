# MeroCircuit

Neuromorphic autoencoder implementation using memristive networks and equilibrium propagation principles.

## Project Overview

This project implements a theoretical framework for autonomous learning in physical memristive networks. The system uses equilibrium propagation: learning occurs through local adaptation driven by differences in electrical activity between free and clamped phases, without requiring external backpropagation.

### Key Features

- **Voltage dynamics solver**: Transient relaxation of voltages in resistive/memristive networks
- **Autoencoder topology**: Input → Hidden → Output architecture with penalty coupling
- **Multiple I-V characteristics**: Ohmic, ReLU, sigmoid, and diode models
- **Comprehensive visualization**: Network states, current flows, and animated relaxation dynamics
- **Experimental protocol**: Realistic simulation with fixed exposure times per phase

## Installation

### Requirements

- Python ≥ 3.9
- pip and venv

### Setup

1. **Clone the repository:**
```bash
   git clone https://github.com/VladimirBash/MeroCircuit
   cd MeroCircuit
```

2. **Create and activate virtual environment:**
```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
```

4. **Run tests:**
```bash
   pytest tests/
```

## Project Structure
```
MeroCircuit/
├── network/                 # Voltage dynamics and I-V characteristics
│   ├── dynamics.py         # VoltageDynamics solver (Euler method), Grid/Memristor-based
│   ├── builders.py         # Glue: (adjacency, weights, iv_func) <-> Grid + Memristor array
│   ├── iv_characteristics.py  # Ohmic, ReLU, sigmoid, diode I-V curves
│   └── legacy.py           # Volodya's original R_Network (preserved, unrelated solver)
├── grid/
│   └── grid.py             # Grid: topology + boundary conditions
├── memristor/
│   └── memristor.py        # Memristor: per-edge state, local Q integrator, plasticity hook
├── datasets/               # Bars & Stripes pattern generation
│   ├── bars_stripes.py    # BarsAndStripes class
│   └── generate.py        # Dataset generation script
├── visualization/          # Plotting and animation tools
│   ├── plotting.py        # Static plots for patterns
│   └── dynamics_viz.py    # Network graphs, current flows, animations
├── training/
│   ├── plasticity.py      # SimplePlasticity: explicit two-snapshot contrastive rule
│   ├── rules.py            # Memristor.plast_func rules (default: global_threshold_rule)
│   └── trainer.py          # Trainer: free/clamped cycle + the shared theta
├── tests/                  # Test suite (see DEVELOPMENT.md for current pass/fail status)
│   ├── test_dynamics_basic.py         # Simple circuits (chains, dividers)
│   ├── test_dynamics_autoencoder.py   # Autoencoder topology tests
│   ├── test_network_visualization.py  # Visualization tests
│   ├── test_plasticity_simple.py      # 3-node chain, explicit contrastive rule
│   ├── test_plasticity.py             # 4->3->4 autoencoder, explicit contrastive rule
│   ├── test_with_memristors.py        # Grid+Memristor+Trainer smoke test
│   └── test_smoke_sweep.py            # Whole-simulator sanity sweep (48 checks, ~90s)
└── DEVELOPMENT.md          # Why things look the way they do, and what's still open
```

## Current Implementation Status

See `DEVELOPMENT.md` for the full, current picture (test results, open
problems, and a note on a merge that briefly broke `main` - worth reading
before assuming anything below is up to date).

### ✅ Completed

**Voltage Dynamics Solver** (`network/dynamics.py`)
- Transient relaxation via explicit Euler method, on top of `Grid` (topology
  + boundary conditions) and an array of `Memristor` objects (per-edge state)
- Penalty coupling for autoencoder reconstruction
- Configurable I-V characteristics per edge, and capacitances
- Free phase (β=0) and clamped phase (β>0) support

**Grid & Memristor** (`grid/grid.py`, `memristor/memristor.py`)
- `Grid`: adjacency, clamped nodes/values, capacitances
- `Memristor`: local conductance state, a fast windowed average of a local
  observable `Q`, and a pluggable plasticity rule (`plast_func`) - see
  `training/rules.py` and DEVELOPMENT.md's "Global vs local theta"

**I-V Characteristics** (`network/iv_characteristics.py`)
- Ohmic (linear)
- ReLU with threshold
- Sigmoid (smooth nonlinearity)
- Diode (asymmetric)

**Datasets** (`datasets/`)
- Bars & Stripes pattern generator for N×N grids
- Voltage encoding/decoding
- Train/test splitting
- Pattern type classification

**Visualization** (`visualization/`)
- Time series plots of voltage evolution
- Network graphs with voltage-coded nodes
- Current flow visualization with arrows
- Animated relaxation (free → clamped phase transition)
- Side-by-side phase comparison

**Testing**
- Basic circuit tests (resistor chains, voltage dividers)
- Autoencoder topology with realistic experimental protocol
- Penalty coupling validation
- Convergence tests with different capacitances
- Plasticity: 3-node chain (converges) and 4→3→4 autoencoder (learns, then
  plateaus - see DEVELOPMENT.md, Open Problems)
- Grid+Memristor+Trainer smoke test (`test_with_memristors.py`)

### 🚧 In Development

**Plasticity Rules** (`training/`)
- Explicit two-snapshot contrastive rule (`SimplePlasticity`) - works, plateaus
- Local/online rule driven by a shared global threshold (`training/rules.py`) - new, only smoke-tested at length

**Training Loop** (`training/trainer.py`)
- Single-cycle, single-pattern free/clamped protocol - done
- Multi-cycle, full-dataset loop over Bars & Stripes - not yet wired up
- MSE tracking / weight evolution visualization - not started

### 📋 Planned

- Wire `Trainer` to the full 4×4 Bars & Stripes dataset
- Resolve the 4→3→4 plateau (see DEVELOPMENT.md, Open Problems)
- Load-test at the target ~64→8→64 scale

## Usage Examples

### Generate Bars & Stripes Dataset
```python
from datasets.bars_stripes import BarsAndStripes

# Create dataset
ds = BarsAndStripes(N=4, voltage_on=1.0, voltage_off=0.0)

# Get all patterns
patterns = ds.get_all_patterns()  # Shape: (n_patterns, 4, 4)

# Sample random batch
batch = ds.sample(n_samples=8)

# Split for training
train_patterns, test_patterns = ds.split_train_test(test_fraction=0.2)
```

### Run Voltage Relaxation
```python
from grid.grid import Grid
from network.builders import build_static_memristor_array
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic
import numpy as np

# Simple 3-node chain: 0 -- 1 -- 2
adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
conductances = adjacency.astype(float)

grid = Grid(adjacency, clamped_nodes=np.array([0, 2]),
            clamped_values=np.array([1.0, 0.0]), capacitances=1.0)
memristors = build_static_memristor_array(adjacency, conductances, ohmic)
solver = VoltageDynamics(grid, memristors)

# Relax with boundary conditions (already set on the Grid above)
V_init = np.array([1.0, 0.5, 0.0])
result = solver.relax_transient(V_init, dt=0.01, max_steps=1000, record_history=True)

print(f"Final voltages: {result['V_final']}")
print(f"Converged: {result['converged']} in {result['n_steps']} steps")
```

### Autoencoder with Penalty Coupling
```python
# Build 4→3→4 autoencoder
n_input, n_hidden, n_output = 4, 3, 4
n_total = n_input + n_hidden + n_output

# ... build adjacency, conductances, input_nodes, V_input ...

grid = Grid(adjacency, clamped_nodes=input_nodes, clamped_values=V_input, capacitances=1.0)
memristors = build_static_memristor_array(adjacency, conductances, ohmic)
solver = VoltageDynamics(grid, memristors)

penalty_pairs = [(i, n_input + n_hidden + i) for i in range(n_input)]

# Free phase (β=0)
result_free = solver.relax_transient(V_init, beta=0.0, dt=0.001, max_steps=10000)

# Clamped phase (β>0)
result_clamped = solver.relax_transient(
    result_free['V_final'],
    penalty_pairs=penalty_pairs,
    beta=1.0, g_penalty=1.0,
    dt=0.001, max_steps=10000
)
```

For a trainable (not just static) network, build the `Memristor` array with
`network.builders.build_memristor_array` instead (it maps `w` through a
`g_min`/`g_max` range and takes a real `plast_func`/`obs_func` - see
`training/rules.py` and `training/trainer.py`, or `tests/test_with_memristors.py`
for a complete worked example).

### Visualize Network Dynamics
```python
from visualization.dynamics_viz import plot_voltage_evolution, animate_relaxation

# Plot voltage evolution
fig = plot_voltage_evolution(
    result['V_history'],
    node_groups={'Input': [0, 3], 'Hidden': [1, 2]},
    dt=0.01
)
fig.savefig('voltage_evolution.png')

# Create animation (free → clamped) - animate_relaxation still takes a plain
# conductance matrix + iv_function directly, independent of Grid/Memristor
V_combined = np.vstack([result_free['V_history'], result_clamped['V_history']])
anim = animate_relaxation(
    V_combined, adjacency, conductances, ohmic,
    nodes_to_plot=[0, 4, 8],  # Selected nodes
    phase_transition_frame=len(result_free['V_history']),
    output_file='relaxation.gif'
)
```

## Running Tests

Most test files are plain scripts, not pytest files - run them directly:
```bash
python3 tests/test_smoke_sweep.py       # start here: 48-check sanity sweep, ~90s
python3 tests/test_dynamics_basic.py
python3 tests/test_plasticity.py        # 4->3->4 learning run, ~80s
```

`test_smoke_sweep.py` is the one to run after any change to the solver, the memristor
model, or a plasticity rule. It checks physics invariants (Kirchhoff, analytic agreement,
linearity), that the physical knobs move things the right way, that both plasticity paths
stay bounded, that runs are reproducible, and that the dataset loop works end to end.

`test_bars_stripes.py` is a real pytest file and needs `pytest` installed:
```bash
pytest tests/test_bars_stripes.py -v
```

## Key Parameters

### Solver Configuration
- `dt`: Time step (0.001 for stable clamped phase with high β)
- `tol`: Convergence tolerance (1e-10 for high precision)
- `max_steps`: Maximum iterations per phase
- `divergence_threshold`: Abort and report `result['diverged']` if |V| exceeds this
  (default 1e6; `None` disables)

> **Stability warning.** The solver is explicit Euler, which is only conditionally
> stable. At `beta=100, g_penalty=10` the critical time step is **≈0.002**, so the
> default `dt=0.001` has only about a **2x margin**. If you raise `beta` or
> `g_penalty`, lower `dt` to match, and check `result['diverged']` — see
> DEVELOPMENT.md ("Stability boundary, quantified") for the measured map.

### Physical Parameters
- `beta`: Penalty coupling strength (0 = free, >0 = clamped)
- `g_penalty`: Penalty link conductance
- `capacitances`: Node capacitances (affects relaxation speed, not the fixed point)

### Experimental Protocol
- `exposure_time_free`: Duration of free phase
- `exposure_time_clamped`: Duration of clamped phase
- Both measured in seconds, independent of solver timestep

## Team

- **Andrey Bagrov**
- **Anna Kravchenko**
- **Vladimir Bashmakov**

## License

[To be determined]

## References

[Key papers and theoretical background to be added]