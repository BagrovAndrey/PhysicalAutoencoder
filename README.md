# MeroCircuit

Simulation of a neuromorphic autoencoder built from memristive networks and trained by
equilibrium-propagation-style local learning: inference is electrical relaxation, learning
is slow local adaptation of conductances driven by the difference between a free and a
weakly clamped regime. See the project proposal (*Fully autonomous neuromorphic element
based on autoencoder learning principles*) for the physical framework.

## Status at a glance

- **Works:** a fast vectorized engine with seven edge elements (ohmic, dead-zone ReLU,
  tanh/sigmoid, sinh, rectifying antiparallel pairs, Shockley diode + filament pairs), two
  relaxation protocols, three learning rules (historical contrastive, clock-free
  global-theta, equilibrium propagation), an oracle (best achievable reconstruction) and a
  hand-built existence proof, all behind one CLI, `sim.py`. A 4→3→4 network learns a single
  pattern `[1,0,1,0]` to MSE ≈ 0.03–0.05, bit-perfect after thresholding at 0.5.
- **Partly works:** learning a *dataset*. Weak-nudge EP reaches the best achievable
  reconstruction on 2×2 Bars & Stripes. Symmetric elements cannot represent 3×3 at all;
  rectifying pairs can (a hand-built network gets 14/14), but local learning from random
  initial states plateaus at ~60% exact patterns.
- **Why, and what to try next:** [`docs/SPEC.md`](docs/SPEC.md) — measured diagnosis
  (threshold loss per hop, nudge-strength regime, representational limits of passive
  symmetric networks, rectifying and Shockley elements, the open question of clock-free
  learning) and a prioritized plan. [`DEVELOPMENT.md`](DEVELOPMENT.md) has the history.

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
python3 tests/test_smoke_sweep.py  # ~3 min, should end with "66 passed, 0 failed"
python3 tests/test_engine.py       # ~30 s, "58 passed, 0 failed"
```

## Quick start: the `sim.py` CLI

Everything runs through one CLI. Every subcommand takes `--help`; `info` lists elements,
rules, protocols and sweepable parameters.

```bash
python3 sim.py info                  # elements, rules, protocols, default wiring
python3 sim.py relax --beta 0        # one free-phase relaxation: voltages, residual, MSE
python3 sim.py train                 # 4->3->4 on [1,0,1,0], contrastive rule, 40 cycles
python3 sim.py oracle                # best reconstruction any conductances can reach
python3 sim.py handbuilt             # hand-wired AND/OR network, 3x3 Bars & Stripes, 14/14
python3 sim.py sweep --param beta --values 0,1,10,100
python3 sim.py stability             # where the explicit Euler solver blows up
python3 sim.py bench                 # solver cost vs network size
```

Things worth trying:

```bash
# The historical rules on one pattern (0.043978 and 0.032565)
python3 sim.py train --rule contrastive  --cycles 40
python3 sim.py train --rule global-theta --cycles 25

# Equilibrium propagation proper: weak nudge, co-content observable, fully relaxed phases
python3 sim.py train --rule ep --element rectpair --dataset bars-stripes --n-input 4 --epochs 60
python3 sim.py train --rule ep --element rectpair --dataset bars-stripes --n-input 9 \
        --n-hidden 6 --epochs 200 --holdout 3 --show-patterns

# Ceilings for different elements
python3 sim.py oracle --element ohmic    --dataset bars-stripes --n-input 9 --n-hidden 5
python3 sim.py oracle --element rectpair --dataset bars-stripes --n-input 9 --n-hidden 6
python3 sim.py oracle --element tanh --steepness 2 --dataset bars-stripes --n-input 4

# Realistic rectifiers
python3 sim.py handbuilt --vf 0.1                            # ideal pair with forward drop
python3 sim.py handbuilt --element shockley --Is 1e-4 --vsig 1
python3 sim.py handbuilt --n-input 16 --g-min 0.001 --pullup 0.1   # 4x4

# Parameter scans (add --train to scan a training run instead of one relaxation)
python3 sim.py sweep --param beta --values 0,0.1,1,10,100 --max-steps 30000
python3 sim.py sweep --param vf --values 0,0.05,0.1 --train --rule ep --element rectpair \
        --dataset bars-stripes --n-input 4 --epochs 30

# Save the learning curve / everything
python3 sim.py train --cycles 40 --plot mse.png --json run.json

# Break it on purpose: dt past the stability boundary
python3 sim.py relax --dt 0.05 --beta 100
```

Main flags: `--element` (`--iv` is an alias) with its parameters `--vth`, `--steepness`,
`--v0`, `--vf`, `--Is`, `--n`; `--vsig` (signal amplitude, V); `--beta` and `--g-penalty`
(only the product matters; default 1000 for the historical rules, 0.1 for `ep`, 1e-4 for
`ep` on `shockley`); `--protocol exposure|equilibrium`; `--rule`; `--cycles` or `--epochs`;
`--holdout k`; `--dataset bars-stripes` with `--n-input N²`; `--n-hidden`; `--arch`
(`plain`, `bias`, `dual`, `dual+bias`).

Every run ends with the **distribution of wrong pixels per pattern** (after thresholding
at 0.5), besides MSE, bit accuracy and margin:

```
    wrong pixels |     0     1     2     3
    patterns     |   71%    0%    0%   29%
                 |    10     0     0     4
```

> **"not converged" is usually not an error** under `--protocol exposure`: relaxing for a
> fixed exposure time is the historical protocol, and the free phase typically does not
> settle within it. A blow-up is a real failure: lower `--dt`.

> **`global-theta`: leave `--window-pts` alone** when changing `--micro-steps` or
> `--exposure`. Measured: the rule learns only with a window of exactly one phase. Why is
> an open question (SPEC 3.5) — do not read it as a device constraint yet.

## Project structure

```
MeroCircuit/
├── sim.py                    # the CLI: relax / train / oracle / handbuilt / sweep /
│                             #   stability / bench / info
├── AGENTS.md                 # Working rules for coding agents (Codex etc.)
├── DEVELOPMENT.md            # History, design decisions, measured findings
├── docs/
│   └── SPEC.md               # Current diagnosis + prioritized research/engineering plan
├── grid/
│   └── grid.py               # Grid: topology, clamped nodes/values, capacitances
├── memristor/
│   └── memristor.py          # Memristor: state w, windowed local observable, plasticity hook
├── network/
│   ├── elements.py           # edge elements: current, slope, dI/dg, d(co-content)/dg
│   ├── equilibrium.py        # vectorized engine: Euler + equilibrium relaxation, exact
│   │                         #   gradients, autoencoder topologies (what sim.py runs on)
│   ├── dynamics.py           # VoltageDynamics: explicit-Euler relaxation, penalty coupling
│   ├── builders.py           # build_autoencoder_topology, Memristor-array builders,
│   │                         #   extract_weights / set_weights
│   ├── iv_characteristics.py # ohmic, relu (dead zone), sigmoid, diode
│   └── legacy.py             # original R_Network solver (kept, not used)
├── training/
│   ├── learning.py           # engine rules (contrastive, global-theta, ep), training
│   │                         #   loop, metrics, oracle, hand-built network
│   ├── plasticity.py         # SimplePlasticity: explicit two-snapshot contrastive rule
│   ├── rules.py              # plast_func rules + local observables (dV^2, dV, dV*I)
│   └── trainer.py            # Trainer: free/clamped cycle, shared theta, online path
├── datasets/
│   ├── bars_stripes.py       # BarsAndStripes: N×N patterns, encoding, splits
│   └── generate.py
├── visualization/
│   ├── plotting.py           # pattern plots
│   └── dynamics_viz.py       # voltage evolution, network states, animations
├── experiments/              # dated studies behind claims in SPEC.md / DEVELOPMENT.md
└── tests/
    ├── test_smoke_sweep.py           # start here: 66 checks, ~3 min
    ├── test_engine.py                # engine vs legacy, derivatives, EP, hand-built (~30 s)
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

**Physics.** Nodes with capacitances relax under Kirchhoff's current law; input nodes are
clamped; a penalty link `β·g_p·(V_in − V_out)` couples each output to its input in the
nudged phase. Each edge has state `w ∈ [0,1]` (two states for rectifying pairs),
conductance `g = g_min + (g_max − g_min)·w`, and an element model (`network/elements.py`).
Two protocols: fixed exposure (explicit Euler for a step budget, detects divergence) and
full equilibrium (Euler warm start + Newton).

**Learning.**
- *Contrastive* (historical): relax free and clamped, `dw = −η·(Q_clamped − Q_free)·(1 − w) − γ·w`
  with `Q = ΔV²`, strong nudge.
- *Global-theta* (historical, clock-free): each memristor averages its own `Q` over a
  window and compares it with one slowly adapting network-wide θ; no phase label reaches
  any element.
- *EP*: `Δw = −α·[∂Φ/∂g(nudged) − ∂Φ/∂g(free)]/β'` with `Φ` the co-content of the element,
  weak nudge. Contrastive and EP need each element to know the phase.

Note the sign: the code uses `−η` (gradient descent under equilibrium propagation), while
eq. (14) of the proposal is written with `+η`. See `docs/SPEC.md`.

**Two engines.** `sim.py` runs on the vectorized engine (`network/equilibrium.py`,
`training/learning.py`). The object engine (`Grid`, `Memristor`, `VoltageDynamics`,
`Trainer`) is kept for the Python API below and the regression canaries;
`tests/test_engine.py` checks that the two agree to machine precision and that the
historical training numbers are reproduced.

## Usage from Python

### Training on the engine (what `sim.py train` does)

```python
import numpy as np
from network.elements import make_element
from network.equilibrium import autoencoder, init_weights
from training.learning import EPRule, Protocol, train, evaluate, metrics
from datasets.bars_stripes import BarsAndStripes

X = BarsAndStripes(N=2).get_all_flattened().astype(float)
net = autoencoder(4, 3, make_element('rectpair'))        # 4->3->4, two branches per edge
w0 = init_weights(net, seed=42)
w, history, err = train(net, X, EPRule(alpha=0.5), Protocol('equilibrium'), w0,
                        cycles=60 * len(X), beta_gp=0.1)
m = metrics(evaluate(net, w, X, 1.0, Protocol('equilibrium')), X)
print(m['mse'], m['exact'], m['wrong_hist'])   # ~0.056 1.0 [1.]
```

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
python3 tests/test_engine.py         # run after any change to the engine or sim.py
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
(change relaxation speed, not the fixed point), `g_min`/`g_max` (0.01/1.0), the element and
the signal amplitude `vsig`. For the thresholded ReLU a floating output relaxed from 0
stops `V_th` short per hop along its path — see `docs/SPEC.md` 3.1. For rectifiers only
`V_f / V_sig` matters; Shockley elements need signals of ~1–3 V and nudges ≪ kT/q.

**Learning.** `eta`, `gamma`; contrastive path: `tau_integrate`, `dt_plasticity`; online
path: `exposure` (10), `micro_steps` (200), `window_pts` (= `micro_steps`), `tau_theta`
(40), `relax_max_steps` (defaults to `exposure/dt`); EP: `alpha` (0.5), `beta·g_p` (0.1),
symmetric nudging, soft bounds. All times are in dimensionless model
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
