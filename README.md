# MeroCircuit

Simulation of a neuromorphic autoencoder built from memristive networks and trained by
equilibrium-propagation-style local learning: inference is electrical relaxation, learning
is slow local adaptation of conductances driven by the difference between a free and a
weakly clamped regime. See the project proposal (*Fully autonomous neuromorphic element
based on autoencoder learning principles*) for the physical framework.

## Status at a glance

- **Works:** the solver, both plasticity paths, the CLI, and a 64-check regression suite.
  A 4→3→4 network learns a single pattern `[1,0,1,0]` to MSE ≈ 0.03–0.05; thresholded at
  0.5, the reconstruction is bit-perfect.
- **Does not work yet:** learning a *dataset*. On 2×2 Bars & Stripes the current rules do
  not reach the best achievable reconstruction, and for the symmetric I-V curves in the
  repo even the best achievable reconstruction of 3×3 Bars & Stripes is poor.
- **Why, and what to try next:** [`docs/SPEC.md`](docs/SPEC.md) — measured diagnosis
  (threshold loss per hop, nudge-strength regime of the learning rule, representational
  limits of passive symmetric networks) and a prioritized experiment plan.
  [`DEVELOPMENT.md`](DEVELOPMENT.md) has the history and design decisions.

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/VladimirBash/MeroCircuit
cd MeroCircuit
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python3 tests/test_smoke_sweep.py  # ~3 min, should end with "64 passed, 0 failed"
```

## Quick start: the `sim.py` CLI

The fastest way to get a feel for the model is to turn its knobs from the command line.
Every subcommand takes `--help`.

```bash
python3 sim.py info                  # available I-V curves, observables, rules; default wiring
python3 sim.py relax --beta 0        # one free-phase relaxation: voltages, residual, MSE
python3 sim.py train                 # 4->3->4 on [1,0,1,0], contrastive rule, 40 cycles
python3 sim.py sweep --param beta --values 0,1,10,100
python3 sim.py stability             # where the explicit Euler solver blows up
python3 sim.py bench                 # solver cost vs network size
```

Things worth trying:

```bash
# Watch the penalty coupling pull the output toward the input
python3 sim.py sweep --param beta --values 0,0.1,1,10,100 --max-steps 30000

# The two plasticity paths on the same problem (both end around MSE 0.03-0.05)
python3 sim.py train --rule contrastive  --cycles 40
python3 sim.py train --rule global-theta --cycles 25

# Fully relaxed free phase instead of the fixed-exposure protocol
python3 sim.py train --rule global-theta --cycles 25 --relax-max-steps 100000

# Other I-V curve / bigger bottleneck / the real dataset
python3 sim.py train --iv sigmoid --n-hidden 6 --cycles 20
python3 sim.py train --dataset bars-stripes --n-input 4 --cycles 30

# Save the learning curve
python3 sim.py train --cycles 40 --plot mse.png

# Break it on purpose: dt past the stability boundary
python3 sim.py relax --dt 0.05 --beta 100
```

> **"not converged" is usually not an error.** Relaxing for a fixed exposure time is the
> intended protocol, and the free phase typically does not fully settle within it.
> `DIVERGED` is a real failure: lower `--dt` (see *Key parameters*).

> **Leave `--window-pts` alone when changing `--micro-steps` or `--exposure`.** The
> memristor's averaging window must span exactly one phase, otherwise the network stops
> learning while still running and staying finite. `sim.py` ties the two together by
> default and `Trainer` warns if they drift apart. Details: DEVELOPMENT.md, "The window
> must span one phase".

## Project structure

```
MeroCircuit/
├── sim.py                    # CLI: relax / train / sweep / stability / bench / info
├── AGENTS.md                 # Working rules for coding agents (Codex etc.)
├── DEVELOPMENT.md            # History, design decisions, measured findings
├── docs/
│   └── SPEC.md               # Current diagnosis + prioritized research/engineering plan
├── grid/
│   └── grid.py               # Grid: topology, clamped nodes/values, capacitances
├── memristor/
│   └── memristor.py          # Memristor: state w, windowed local observable, plasticity hook
├── network/
│   ├── dynamics.py           # VoltageDynamics: explicit-Euler relaxation, penalty coupling
│   ├── builders.py           # build_autoencoder_topology, Memristor-array builders,
│   │                         #   extract_weights / set_weights
│   ├── iv_characteristics.py # ohmic, relu (dead zone), sigmoid, diode
│   └── legacy.py             # original R_Network solver (kept, not used)
├── training/
│   ├── plasticity.py         # SimplePlasticity: explicit two-snapshot contrastive rule
│   ├── rules.py              # plast_func rules + local observables (dV^2, dV, dV*I)
│   └── trainer.py            # Trainer: free/clamped cycle, shared theta, online path
├── datasets/
│   ├── bars_stripes.py       # BarsAndStripes: N×N patterns, encoding, splits
│   └── generate.py
├── visualization/
│   ├── plotting.py           # pattern plots
│   └── dynamics_viz.py       # voltage evolution, network states, animations
├── experiments/              # informal scripts behind claims in DEVELOPMENT.md
│                             #   (written against the pre-fix solver API - see its README)
└── tests/
    ├── test_smoke_sweep.py           # start here: 64 checks, ~3 min
    ├── test_dynamics_basic.py        # chains, dividers, penalty coupling
    ├── test_dynamics_autoencoder.py  # autoencoder topology, exposure protocol
    ├── test_iv_characteristics.py    # I-V curves
    ├── test_plasticity_simple.py     # 3-node chain, contrastive rule
    ├── test_plasticity.py            # 4->3->4 on [1,0,1,0], contrastive rule (~80 s)
    ├── test_with_memristors.py       # Grid + Memristor + Trainer
    ├── test_network_visualization.py # plots and animation
    ├── test_bars_stripes.py          # dataset (pytest)
    └── test_visualization.py         # pattern plots (needs an interactive matplotlib backend)
```

## What is implemented

**Physics.** Nodes with capacitances relax by explicit Euler under Kirchhoff's current
law; input nodes are clamped; a penalty link `β·g_p·(V_in − V_out)` couples each output to
its input in the clamped phase. Each edge is a `Memristor` with state `w ∈ [0,1]`,
conductance `g = g_min + (g_max − g_min)·w`, and a pluggable I-V curve (ohmic, thresholded
ReLU with dead zone, sigmoid, diode). The solver detects divergence and reports it.

**Learning — two paths on the same substrate.**
- *Explicit contrastive* (`training/plasticity.py`): relax free and clamped, compute
  `ΔQ = Q_clamped − Q_free` per edge from the two snapshots, update all weights at once:
  `dw = −η·ΔQ·(1 − w) − γ·w`.
- *Online* (`training/trainer.py`, `training/rules.py`): each memristor integrates its own
  local observable over a window one phase long and compares it with one slowly adapting,
  network-wide threshold θ; no phase label reaches any element.

Note the sign: the code uses `−η` (the gradient-descent direction under equilibrium
propagation), while eq. (14) of the proposal is written with `+η`. See `docs/SPEC.md`.

**Tooling.** `sim.py` CLI; Bars & Stripes generator; plotting and animation;
`tests/test_smoke_sweep.py` covering physics invariants (Kirchhoff residual, agreement with a
direct Laplacian solve, linearity), parameter responses, stability, that both plasticity
paths actually reduce error, reproducibility, and the CLI.

## Usage from Python

### Bars & Stripes

```python
from datasets.bars_stripes import BarsAndStripes

ds = BarsAndStripes(N=4, voltage_on=1.0, voltage_off=0.0)
patterns = ds.get_all_patterns()            # (n_patterns, 4, 4)
flat = ds.get_all_flattened()               # (n_patterns, 16)
train, test = ds.split_train_test(test_fraction=0.2)
```

### One free and one clamped relaxation of a 4→3→4 autoencoder

```python
import numpy as np
from grid.grid import Grid
from network.builders import build_autoencoder_topology, build_static_memristor_array
from network.dynamics import VoltageDynamics
from network.iv_characteristics import relu_iv

adjacency, w, inp, hid, out, pairs = build_autoencoder_topology(n_input=4, n_hidden=3, seed=42)
pattern = np.array([1.0, 0.0, 1.0, 0.0])
g = 0.01 + (1.0 - 0.01) * w                     # state w in [0,1] -> conductance

grid = Grid(adjacency, clamped_nodes=inp, clamped_values=pattern, capacitances=1.0)
solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, g, relu_iv))

V0 = np.zeros(len(adjacency)); V0[inp] = pattern
free = solver.relax_transient(V0, beta=0.0, dt=0.001, max_steps=10000, record_history=True)
clamped = solver.relax_transient(free['V_final'], penalty_pairs=pairs, beta=100.0,
                                 g_penalty=10.0, dt=0.001, max_steps=10000,
                                 record_history=True)
print("free output:   ", free['V_final'][out].round(3))
print("clamped output:", clamped['V_final'][out].round(3))
```

For a trainable network use `network.builders.build_memristor_array` (maps `w` through
`g_min/g_max` and takes a real `plast_func`/`obs_func`) together with `training.Trainer`;
`tests/test_with_memristors.py` and `sim.py` (`train_online`) are complete worked examples.

### Plots and animation

```python
from visualization.dynamics_viz import plot_voltage_evolution, animate_relaxation

fig = plot_voltage_evolution(free['V_history'],
                             node_groups={'Input': list(inp), 'Hidden': list(hid),
                                          'Output': list(out)},
                             dt=0.001)
fig.savefig('voltage_evolution.png')

# animate_relaxation takes a plain conductance matrix + I-V function
V_all = np.vstack([free['V_history'], clamped['V_history']])
animate_relaxation(V_all, adjacency, g, relu_iv, nodes_to_plot=[0, 4, 8],
                   phase_transition_frame=len(free['V_history']),
                   output_file='relaxation.gif')
```

## Running tests

The test files are plain scripts (except `test_bars_stripes.py`):

```bash
python3 tests/test_smoke_sweep.py    # run after any change to solver, memristor or rules
python3 tests/test_plasticity.py     # 4->3->4 learning run; final MSE must stay 0.050202
python3 tests/test_dynamics_basic.py
pytest tests/test_bars_stripes.py -v
```

`test_plasticity.py` reproducing `Final MSE: 0.050202` exactly is the canary for any
refactor that is supposed to leave the physics unchanged.

## Key parameters

**Solver.** `dt` (0.001), `tol` (1e-10), `max_steps`, `divergence_threshold` (1e6; `None`
disables).

> **Stability.** Explicit Euler is only conditionally stable. At `beta=100, g_penalty=10`
> the critical time step is ≈ 0.002, so the default `dt=0.001` has only a 2× margin. Raise
> `beta` or `g_penalty` → lower `dt`, and check `result['diverged']`.

**Physics.** `beta`, `g_penalty` (penalty strength is their product), `capacitances`
(change relaxation speed, not the fixed point), `g_min`/`g_max` (0.01/1.0), and the I-V
curve. For the thresholded ReLU the transport threshold `V_th = 0.1` costs up to `V_th` of
signal per hop — see `docs/SPEC.md`.

**Learning.** `eta`, `gamma`; contrastive path: `tau_integrate`, `dt_plasticity`; online
path: `exposure` (10), `micro_steps` (200), `window_pts` (= `micro_steps`), `tau_theta`
(40), `relax_max_steps` (defaults to `exposure/dt`). All times are in dimensionless model
units.

## Team

Andrey Bagrov, Anna Kravchenko, Vladimir Bashmakov

## References

- B. Scellier, Y. Bengio, *Equilibrium propagation: bridging the gap between energy-based
  models and backpropagation*, Front. Comput. Neurosci. 11, 24 (2017).
- J. Kendall et al., *Training end-to-end analog neural networks with equilibrium
  propagation*, [arXiv:2006.01981](https://arxiv.org/abs/2006.01981) (2020).
- M. Stern et al., *Supervised learning in physical networks: from machine learning to
  learning machines*, [Phys. Rev. X 11, 021045](https://link.aps.org/doi/10.1103/PhysRevX.11.021045) (2021).
- S. Dillavou et al., *Demonstration of decentralized physics-driven learning*,
  [Phys. Rev. Applied 18, 014040](https://link.aps.org/doi/10.1103/PhysRevApplied.18.014040) (2022).
- S. Dillavou et al., *Machine learning without a processor: emergent learning in a
  nonlinear analog network*, [PNAS (2024)](https://www.pnas.org/doi/10.1073/pnas.2319718121).
- A. Laborieux et al., *Scaling equilibrium propagation to deep ConvNets by drastically
  reducing its gradient estimator bias*, [Front. Neurosci. (2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.633674/full).
- B. Scellier, S. Mishra, *A universal approximation theorem for nonlinear resistive
  networks*, [Phys. Rev. Applied 23, 044009 (2025)](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.23.044009).
- V. R. Anisetti et al., *Frequency propagation: multimechanism learning in nonlinear
  physical networks*, [Neural Computation 36(4) (2024)](https://direct.mit.edu/neco/article/36/4/596/119787/Frequency-Propagation-Multimechanism-Learning-in).
- M. Guzman, S. Ciarella, A. J. Liu, *Unsupervised and probabilistic learning with
  contrastive local learning networks: the Restricted Kirchhoff Machine*,
  [arXiv:2509.15842](https://arxiv.org/abs/2509.15842).

## License

To be determined.
