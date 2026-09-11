# Development Notes

Working document for restoring context between sessions. README.md describes
*what* the code does; this file describes *why* it looks the way it does and
what is currently broken.

Last updated: 2026-09-11

---

## Current Status

| Component | Status |
|---|---|
| Voltage dynamics solver (`network/dynamics.py`) | Done, tested |
| I-V characteristics (`network/iv_characteristics.py`) | Done, tested |
| Datasets (`datasets/bars_stripes.py`) | Done, tested |
| Visualization (`visualization/dynamics_viz.py`) | Done, tested (free→clamped animation, adaptive frame skip) |
| Plasticity rule (`training/plasticity.py`) | Works, but see "Open Problems" |
| Trainer (`training/trainer.py`) | **Not created.** Drafted but not committed; `test_plasticity.py` runs the training loop inline instead. |
| Training visualization | Not started |
| Integration with Anya (memristor) / Volodya (topology) | Not started; both are delayed |

### Test results

- `tests/test_plasticity_simple.py` (3-node chain, target V[1]=0.68): **passes**, converges to target.
- `tests/test_plasticity.py`, 2→2→2 on pattern [1,0]: MSE 0.26 → 0.05, outputs differentiate correctly.
- `tests/test_plasticity.py`, 4→3→4 on pattern [1,0,1,0]: MSE 0.32 → ~0.044, then **plateaus**.
  Free-phase output stalls around [0.73, 0.13, 0.73, 0.13]; clamped-phase output is exact [1,0,1,0].

---

## Physics Recap

- Node dynamics: `C_i dV_i/dt = Σ_j I_ij(V_i − V_j) + I_penalty_i`, explicit Euler.
- Penalty coupling = virtual resistive edge between input[i] and output[i]:
  `I_penalty = beta * g_penalty * (V_input − V_output)`. `beta=0` free phase, `beta>0` clamped phase.
  `beta * g_penalty` is one effective parameter; kept separate conceptually (device vs. experiment).
- Experimental protocol uses fixed exposure times, not convergence: `max_steps = exposure_time / dt`.
  Non-convergence is a warning, not an error.
- Three time scales, keep them separate:
  - `dt` — solver step (fast electronic relaxation)
  - `tau_integrate` — memristor memory window for EMA of the observable (device property)
  - `exposure_time_free/clamped` — how long each phase is held (experiment setting)
  - `dt_plasticity` — multiplier on weight update per cycle (effectively a learning-rate scale, not a memory time)

---

## Plasticity Rule — Design Decisions and Why

Current rule in `SimplePlasticity.update_weights`:

    delta_Q = Q_clamped − Q_free
    dw = −eta * delta_Q * (1 − w) − gamma * w
    w ← clip(w + dw * dt_plasticity, 0, 1)

with observable `Q_ij = V_j − V_i` (linear, signed) integrated by EMA with time constant `tau_integrate`.

Each of the following was found by debugging, not chosen a priori:

1. **Sign is negative.** With `+eta` the 3-node chain test drove V[1] *away* from
   the target (w[1,2] collapsed to 0). Flipping the sign fixed it.

2. **Observable is linear, not quadratic.** With `Q = (ΔV)²`, two output nodes at 0.5
   being pulled to 1 and to 0 respectively give identical Q=0.25 on their edges, so
   plasticity cannot tell them apart and drives all weights identically. Symmetry is
   never broken. Linear `Q = ΔV` keeps the direction.

3. **Conductance range must be wide.** With `g_min=0.1`, `w∈[0.3,0.7]` the resistance
   spread is only ~2×; all outputs converge to the same value and stay there.
   Working values: `g_min=0.01, g_max=1.0`, initial `w∈[0.05,0.95]` (≈ ×30 spread).
   This was the change that made 2→2→2 learn.

4. **EMA vs. snapshot.** `compute_Q_from_voltages` (snapshot of final state) was added as
   a diagnostic alternative to `integrate_observable_ema`. When phases are long relative to
   `tau_integrate` they agree; snapshot is less physical. Currently `test_plasticity.py` prints
   both. Prefer EMA; snapshot exists for comparison.

5. **Saturation `(1−w)`** is kept because it is physical (finite max conductance).
   Removing it was considered and rejected for now.

6. **Gamma** (decay) was set to 0 during debugging with no visible effect on the plateau.
   Its value is not currently important.

Things that were tried and did **not** help the plateau:
- ReLU I-V instead of ohmic — no change (the underlying issue was the conductance range).
- `tau_integrate` 1 → 5 → 20 — no change once other issues were fixed.
- Adaptive `beta = beta_base * (1 + alpha/(MSE+eps))` — either no effect (alpha=0.05) or NaN blow-up (alpha≥0.1).
- `beta_clamped = 200` — **numerically unstable** at `dt=0.001`; voltages diverge to 1e5, hidden nodes go negative.
  Longer exposure makes it worse. Keep `beta ≤ ~100` with `g_penalty=10` at `dt=0.001`, or reduce `dt`.
- Increasing `eta` — pointless because `eta` and `beta` only enter via the contrast; same behaviour as raising `beta`.

---

## Open Problems

### 1. Plateau in 4→3→4 (vanishing gradient)

Diagnostics at the plateau vs. early cycles:

    early:   delta_Q ∈ [−0.035, 0.48],   dw ∈ [−0.055, 0.036]
    plateau: delta_Q ∈ [−0.007, 0.25],   dw ∈ [−0.002, 0.003]

As MSE drops, free and clamped states get closer, the contrast shrinks, and updates die
out before reconstruction is good. Weights are not stuck at the [0,1] boundaries
(mean≈0.48, std≈0.24, min≈0.07, max≈0.92), so it is not saturation.

**Note:** 4→3→4 is the minimal meaningful test. The target architecture is ~64→8→64
(8× compression), so this must work here first. Do not blame the bottleneck.

Candidate next steps (untested):
- **Momentum** on `dw` (physically: ionic/thermal inertia). Draft: keep `self.velocity`,
  `velocity = momentum*velocity + dw`, `w += velocity*dt_plasticity`. Try `momentum≈0.9`.
- Revisit the observable: current is voltage; could be current `I_ij` or signed power.
- Check whether the linear observable + this rule is actually a gradient of anything
  (equilibrium-propagation consistency) — we have not derived it, only tuned it.

### 2. Beta / dt stability

Explicit Euler with strong penalty is stiff. If larger `beta` is needed, either lower `dt`
or switch the integrator (`_compute_time_derivative` is separated from the integration
scheme for exactly this reason — RK4 or `scipy.integrate` is a drop-in).

### 3. `test_plasticity.py` is messy

It has accumulated a lot of inline diagnostics (EMA vs. manual Q, dw ranges, weight stats,
convergence flags, adaptive beta remnants). Before building `Trainer`, clean it back down to
the config dataclass + loop + a small diagnostic block.

---

## Conventions

- Parameters live in a `@dataclass` config at the top of each test file
  (`SimConfig` in `test_network_visualization.py`, `PlasticityTestConfig` in `test_plasticity.py`).
  No hard-coded `dt` / `max_steps` scattered through a file.
- Research-code style: short docstrings, minimal validation, parameters as function args.
- Tests are plain scripts runnable with `python3 tests/<name>.py`; they save plots to cwd.
- Setup: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install -e .`
  (`pip install -e .` is what makes `from network... / from training...` resolve.)

---

## Next Steps (rough order)

1. Try momentum on the 4→3→4 plateau.
2. Clean up `test_plasticity.py`.
3. Create `training/trainer.py` once single-pattern learning is reliable.
4. Train on the full 4×4 Bars & Stripes dataset.
5. Training visualization (MSE vs. cycle, weight evolution, reconstruction animation).
6. Interface with Anya's memristor model and Volodya's topology code.
