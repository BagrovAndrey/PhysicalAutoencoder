# Development Notes

Working document for restoring context between sessions. README.md describes
*what* the code does; this file describes *why* it looks the way it does and
what is currently broken.

Last updated: 2026-09-23

---

## History: the broken merge (read this first if things don't match what you remember)

Commit `2add7d5` ("Added classes Memristor and Grid, the solver is adapted
for working with the classes", merged as PR #1 / `a80c999`) rewrote
`network/dynamics.py`'s method bodies (`relax_transient`,
`_compute_time_derivative`) to use the new `Grid`/`Memristor` classes, but
did **not** update `__init__` (still took the old `adjacency, iv_function,
capacitances` args and never set `self.grid` / `self.memristors`) or
`_compute_time_derivative`'s own signature (still declared `conductances,
free_nodes, ...` while being called with different arguments). The file was
internally inconsistent from that commit onward: `AttributeError:
'VoltageDynamics' object has no attribute 'grid'` on the very first call.

This went unnoticed because nobody re-ran the test suite after the merge -
the "Current Status" table below (and the plateau numbers in "Test
results") describe the state *before* the merge, not the state that was
actually on `main` afterward. If you're reading this confused because
`test_plasticity.py` didn't behave the way this file says: that's why.

Separately, `tests/test_with_memristors.py` (meant to exercise
`Memristor`/`Grid` directly) called both classes with keyword arguments
that matched neither's real constructor - it was a sketch written ahead of
the classes actually being finalized, never runnable as committed.

**Fixed as of 2026-09-14**: `network/dynamics.py` is now the
self-consistent `Grid`/`Memristor`-based solver (what had been developed in
parallel as `network/dynamics_improved.py`, now retired/merged in). This is
the canonical solver going forward - see "Architecture decisions" below for
what else changed alongside this fix, and "Current Status" for what is
actually verified to run now.

---

## Current Status

| Component | Status |
|---|---|
| Voltage dynamics solver (`network/dynamics.py`) | Fixed 2026-09-14 (see History above). Grid/Memristor-based. All of `tests/test_dynamics_basic.py`, `test_dynamics_autoencoder.py` pass. |
| Grid (`grid/grid.py`), Memristor (`memristor/memristor.py`) | Done. Memristor extended with a `theta` parameter for its plasticity rule (see "Global vs local theta"). |
| I-V characteristics (`network/iv_characteristics.py`) | Done, tested |
| Datasets (`datasets/bars_stripes.py`) | Done, tested |
| Visualization (`visualization/dynamics_viz.py`) | Done, tested (free→clamped animation, adaptive frame skip); ported to Grid/Memristor in `test_network_visualization.py` |
| Plasticity rule, explicit contrast (`training/plasticity.py`, `SimplePlasticity`) | Works, ported to run on top of Grid/Memristor (see `test_plasticity.py`, `test_plasticity_simple.py`). Plateau still open, see "Open Problems". |
| Plasticity rule, local/online (`training/rules.py`, `Memristor.plast_func`) | New. `global_threshold_rule` is the default going forward - see "Global vs local theta". Exercised by `tests/test_with_memristors.py` (smoke test only, see its docstring for scope). |
| Trainer (`training/trainer.py`) | **Created 2026-09-14.** Owns the free/clamped cycle and the one shared `theta`. Not yet wired to the Bars & Stripes dataset - still one-pattern only. |
| `network/builders.py` | New. Glue between (adjacency, weight matrix, iv_function) and Grid/Memristor objects - used throughout the ported tests. |
| Training visualization | Not started |
| Integration with Anya (memristor) / Volodya (topology) | **Done in the sense of "compiles and runs together"** (Grid + Memristor + VoltageDynamics + Trainer all use each other now). Not done in the sense of "the plateau is understood" or "runs on the full 64→8→64 target architecture" - see Open Problems. |

### Test results (2026-09-14, after the fix)

- `tests/test_dynamics_basic.py`, `test_dynamics_autoencoder.py`, `test_network_visualization.py`: pass.
- `tests/test_plasticity_simple.py` (3-node chain, target V[1]=0.68): **passes**, converges to target (95% error reduction over 50 cycles) - numerically identical behaviour to before the merge broke it.
- `tests/test_plasticity.py`, 4→3→4 on pattern [1,0,1,0], quadratic `Q`: MSE 0.247 → 0.050 over 40 cycles (reduced from the historical 300-cycle run for test runtime - see the note in the file). This reproduces the historical plateau trajectory exactly (same numbers to 4 significant figures as the pre-merge run at matching cycle counts): output settles toward `[~0.73, ~0.13, ~0.73, ~0.13]` instead of `[1,0,1,0]`. **Still open, see below.**
- `tests/test_with_memristors.py`: new smoke test (Grid + Memristor + Trainer, `global_threshold_rule`). Verifies the machinery runs, stays numerically finite, and moves weights - not a quality benchmark (short exposure times for test speed; see its docstring). ~~A quick informal run of the *same rule* ... reached MSE≈0.041~~ — **this claim was wrong, corrected 2026-09-23: see "The window must span one phase" below.** That number came from `experiments/bcm_threshold_check.py`, which uses a **local per-edge theta**, not the global one the project implements. The global rule does reach comparable numbers (0.033), but only under a condition nobody had identified at the time, and the sentence as written sent a reader straight into a non-learning configuration.

### Smoke sweep (2026-09-23): `tests/test_smoke_sweep.py`

A harness that asks whether the thing works *as a simulator* — can you build different
shapes, swap I-V curves and plasticity rules, turn the physical knobs, and get finite,
reproducible answers that move the way physics says they should. 48 checks, ~90s, all
passing. It says nothing about whether the plateau is solved; it is the regression net
that should catch the next silent breakage the way nobody caught the merge one.

What it establishes:

- **Physics invariants hold.** Boundary conditions are pinned exactly; Kirchhoff residual
  at converged free nodes is ~1e-10; the relaxed solution matches an independent direct
  Laplacian solve to ~1e-10 for the ohmic case; the ohmic network is linear; the fixed
  point is independent of capacitance while relaxation time scales with it.
- **The knobs do what they should.** Increasing `beta` (0 → 1 → 10 → 100) drives
  reconstruction MSE monotonically 0.249 → 0.0053 → 7e-5 → 0; `g_penalty` does the same;
  `dt` changes cost but not the fixed point; all four I-V characteristics relax to finite
  fixed points. Note `diode_iv` saturates every output to 1.0 — correct rectifying
  behaviour, but it means the diode curve is not usable as-is for autoencoding.
- **Both plasticity paths stay bounded** across `eta`/`gamma`/`tau_theta` sweeps, and the
  rule really is a one-line swap: a custom rule passed as `plast_func` changes behaviour
  with no other edit. Per-edge `theta` is reachable today (two edges can hold different
  values and produce different `dw/dt`), confirming the future-extension point is real
  rather than decorative.
- **Runs are bit-reproducible** for a fixed seed, and different seeds give different
  networks.
- **The dataset path works end to end.** `Trainer` driven over all 6 Bars & Stripes N=2
  patterns for 3 epochs stays finite, and distinct inputs do produce distinct outputs
  (max per-node spread ≈0.06) — small, but not the fully collapsed state. This closes the
  mechanical half of Open Problem #4; what remains there is quality, not wiring.

### `sim.py`: the CLI (2026-09-23)

Everything above was previously only reachable by editing constants inside a test file,
which made "what does this knob do" an edit-run-revert loop. `sim.py` exposes it:
`relax` (one relaxation, with voltages, equilibrium residual and optionally every edge
current), `train` (either plasticity path, optionally over the real dataset, optionally
plotting), `sweep` (one parameter over several values, tabulated), `stability` (the map
above, regenerated for your settings, plus a bisected critical `dt`), `bench` (cost vs
size with extrapolation), and `info`.

Two conventions worth keeping if you extend it. Topology construction now lives in
`network/builders.build_autoencoder_topology` rather than being copy-pasted per test —
the CLI and `test_smoke_sweep.py` share it. Local observables now live alongside the
rules in `training/rules.py` (`quadratic_observable`, `linear_observable`,
`power_observable`), so swapping `Q` is a one-line change exactly like swapping a rule,
and `sim.py --observable` gets them for free.

`test_smoke_sweep.py` section 6 checks the CLI starts, that every subcommand parses,
that a CLI relaxation reproduces the library's number exactly, and that a diverging run
exits non-zero rather than printing garbage.

### The window must span one phase (2026-09-23) — read this before tuning anything

Found by Andrey running `sim.py train --rule global-theta` on the shipped defaults and
getting a flat, dead MSE=0.5. The defaults were wrong, but chasing that turned up a real
and previously unstated physical constraint.

**Each memristor's averaging window must span exactly one phase:**
`window_pts == micro_steps_per_phase`. Measured on 4→3→4 / `[1,0,1,0]`, 30 cycles,
everything else held fixed:

| `window_pts / micro_steps` | final MSE |
|---|---|
| 1.00 | **0.033** (learns) |
| 0.50 | 0.493 (dead) |
| 0.25 | 0.493 (dead) |
| 0.05 | 0.493 (dead) |

Why, mechanically: `V` is held constant within a phase (quasi-static), so every sample
the window sees during a phase is identical. A window *shorter* than the phase therefore
saturates at the current phase's `Q` well before the phase ends, and carries no trace of
the other phase — so `Q_avg − theta` reduces to a purely spatial comparison ("is this
edge busier than the network average?"), which contains no reconstruction error signal.
A window exactly one phase long never fully catches up: it lags, and that lag *is* the
memory of the other phase. The per-edge free/clamped contrast lives entirely in that lag.

This is a statement about the physical element, not a hyperparameter: **the device's
internal relaxation time must be matched to the drive period.** Too fast a memristor
forgets the other phase and cannot learn, no matter how the rest is tuned. Worth putting
in the proposal — it is a concrete, testable constraint on the material, and it arrived
from the simulation rather than from theory.

`Trainer` now warns when the ratio is off (`check_window=False` to silence).

Two further things this exposed:

- **`exposure_time` was overloaded.** It set both the plasticity timescale *and* the
  relaxation budget (`max_steps = exposure/dt`), so the two could not be varied
  independently. `Trainer(relax_max_steps=...)` now separates them.
- **The free phase never actually reaches equilibrium** under the standard protocol
  (0/30 cycles converged at `exposure=10`, which allows 10k steps against the ~40k it
  needs). That is the intended "fixed exposure time" protocol, but it was invisible.
  Letting it fully relax gives MSE 0.052 instead of 0.033 — so the learning is *real*,
  not an artifact of truncation, and 0.052 lands essentially on the explicit-contrastive
  plateau of 0.050. Both paths agree. `sim.py train` now reports the convergence count.

### Stability boundary, quantified (2026-09-23)

Open Problem #2 used to say "explicit Euler is stiff, lower `dt` if you raise `beta`".
The boundary has now been measured, and it is tighter than that wording suggests. At the
repo's standard `beta=100, g_penalty=10`, critical `dt` is **≈0.0020** — the `dt=0.001`
used everywhere has only about a **2x margin**, not the order of magnitude one would want.
Map (relu, `g_penalty=10`):

| `dt` | β=1 | β=10 | β=100 | β=1000 |
|------|-----|------|-------|--------|
| 0.001 | ok | ok | ok | diverges |
| 0.005 | ok | ok | diverges | diverges |
| 0.01 | ok | ok | diverges | diverges |
| 0.05 | ok | diverges | diverges | diverges |

Practical consequence: **anyone sweeping `beta`, `g_penalty` or `dt` will cross this
boundary.** Previously that failure was silent — `V` filled with inf/nan and the caller
got `converged=False`, indistinguishable from "needed more steps". `relax_transient` now
takes `divergence_threshold` (default 1e6) and returns a `'diverged'` flag; pass
`divergence_threshold=None` for the old raw behaviour. This changes no physics: the full
suite, including `test_plasticity.py`, reproduces identical numbers (final MSE 0.050202).

### Cost at the target scale (2026-09-23)

Per-step solver cost measured at fixed step budget: 4→3→4 (11 nodes) ≈39 µs, 9→4→9 (22)
≈89 µs, 16→8→16 (40) ≈258 µs (absolute numbers are machine-dependent — vary by 2-3x
between machines; the *scaling* is the robust part, and `sim.py bench` re-measures it on
yours) — consistent with the O(n²) dense node-pair loop in
`_compute_time_derivative`. Extrapolating to the proposal's **64→8→64 (136 nodes): ~3 ms
per step, ~60 s per free+clamped cycle at 10k steps/phase, so a 300-cycle run is ~5 hours**
of pure Python. That is the real obstacle to working at target scale, and it is not a
physics problem: the inner double loop over `range(n)` is vectorizable into a couple of
dense matrix ops, which should buy one to two orders of magnitude. Worth doing before,
not after, the next round of physics experiments.

### Correction to the "Design Decisions" section below (found 2026-09-14)

The observable actually used in `test_plasticity.py`/`test_plasticity_simple.py` is
`compute_Q_from_voltages` (**quadratic**, `Q = (V_j - V_i)^2`), not the linear
`integrate_observable_ema`/`compute_observable` path described as adopted in
point 2 below (that call is commented out in the training loop). The
quadratic version is what actually produced the plateau numbers on record.
Tried swapping in the literal linear signed `Q = V_j - V_i` on the same
4→3→4 case: it does **not** fix the plateau, it makes reconstruction worse
(MSE rose to ~0.41, outputs collapsed toward 0 instead of tracking the
pattern) - plausibly because `w[i,j]` and `w[j,i]` are independent
(non-symmetrized) in this code, and a signed antisymmetric `Q` interacts
badly with that. Point 2 below is kept for the historical reasoning, but
its conclusion should be treated as unresolved, not adopted.

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

## Architecture decisions (2026-09-14)

Context for these: after the broken-merge fix, the question came up of
*how* to reconcile the explicit two-snapshot contrastive rule
(`SimplePlasticity`, array-level) with Anya's per-edge `Memristor` class,
which only ever sees one continuously-running local average `Q_avg` - it
has no way to know "am I in the free phase or the clamped phase right now".

**Decision: both coexist, not one replacing the other.**
`network/dynamics.py`'s `VoltageDynamics` and the `Memristor` array are now
the single shared electrical substrate. On top of that substrate there are
two independent ways to move a `Memristor`'s `w`:
1. **Explicit/batch** (`training/plasticity.py`, `SimplePlasticity`): compute
   `Q_free`/`Q_clamped` from two voltage snapshots externally, call
   `update_weights`, write the result back with `network.builders.set_weights`.
   This is what reproduces the historical plateau numbers (see Test results).
2. **Local/online** (`training/rules.py`, `Memristor.plast_func`, driven by
   `training/trainer.py`): each edge continuously integrates its own
   windowed `Q_avg` and evolves `w` every micro-step via `mem.step()`.

### Global vs local theta

A `Memristor` driven purely by its own `Q_avg` (option 2 above) cannot
learn at all if `Q` is sign-definite (e.g. the quadratic `(ΔV)^2` used
throughout this project): `dw/dt = -eta * Q_avg * (1-w) - gamma*w` is
always ≤0, so every edge just decays to `g_min` regardless of phase -
verified empirically, not just argued (weights collapsed to `w_mean=0.000`,
MSE got worse than random init). A local rule needs *something* to compare
`Q_avg` against.

Considered:
- **A local clock/phase signal** telling each edge "you are now in the
  clamped phase" - rejected: cheap in principle but still per-element
  information beyond current/voltage, which we'd rather not require.
- **A second, slower local integrator per edge** (BCM-style metaplasticity:
  `theta_ij` = a low-pass of the edge's own `Q_avg`, `dw/dt` driven by
  `Q_avg - theta_ij`). Physically real (see literature below), but is a
  specific composite/dual-timescale device, not a property of a generic
  memristor - a real design commitment, not something free.
- **A single global `theta`, shared by the whole network** (chosen):
  physically like a shared slow field (substrate temperature, a common
  bias rail) that every edge's dynamics already couples to in real
  crossbar hardware (thermal crosstalk between cells is a well-documented,
  usually-parasitic effect - here it's exactly the mechanism we want).
  Needs only one extra scalar for the whole circuit, not one per edge.

**Decision: global `theta` is the actual default** (`training/trainer.py`
owns it, `training/rules.py`'s `global_threshold_rule` is the rule).
**Local per-edge `theta` is kept as a real extension point, not built now**:
`Memristor.plast_func(Q_avg, w, theta)` always takes `theta` as a plain
argument - today `Trainer` always supplies the same shared value to every
edge via `mem.set_theta(...)`; a future version could instead have
`Memristor` maintain its own local low-pass of `Q_avg` and never call
`set_theta` from outside. That only changes where `theta` comes from, not
the rule signature or anything that calls it.

Literature grounding (for when this needs defending, or if a specific
material platform gets chosen later and this needs revisiting): dual-mode
volatile+nonvolatile dynamics coexisting in one device, composite
diffusive+drift memristor synapses for short-term/long-term plasticity, and
direct experimental demonstration of BCM-style metaplasticity in
memristors are all published (Wang et al., *Nature Materials* 2017;
"Implementation of Neuro-Memristive Synapse for Long- and Short-Term
Bio-Synaptic Plasticity", PMC7831501; Ding et al., dual-mode SiO2
memristors, *Advanced Science*; "Emulation of synaptic metaplasticity in
memristors"). We are not tied to a specific material, so this is
background, not a constraint - variant 3 (global, shared) was chosen for
being the more economical assumption, not because the alternative is
implausible.

---

## Plasticity Rule — Design Decisions and Why (historical, pre-merge)

Current rule in `SimplePlasticity.update_weights`:

    delta_Q = Q_clamped − Q_free
    dw = −eta * delta_Q * (1 − w) − gamma * w
    w ← clip(w + dw * dt_plasticity, 0, 1)

with observable `Q_ij` intended to be `V_j − V_i` (linear, signed)
integrated by EMA with time constant `tau_integrate` - **but see the
correction above: the code paths that actually produced the numbers below
use the quadratic snapshot `compute_Q_from_voltages` instead.**

Each of the following was found by debugging, not chosen a priori:

1. **Sign is negative.** With `+eta` the 3-node chain test drove V[1] *away* from
   the target (w[1,2] collapsed to 0). Flipping the sign fixed it.

2. **Observable is linear, not quadratic** *(see correction above - this was the
   intent, but the code that generated the plateau numbers uses quadratic Q,
   and switching to the literal linear Q on the 4→3→4 case made things worse,
   not better. Unresolved.)* With `Q = (ΔV)²`, two output nodes at 0.5
   being pulled to 1 and to 0 respectively give identical Q=0.25 on their edges, so
   plasticity cannot tell them apart and drives all weights identically. Symmetry is
   never broken - in theory. In practice the 4→3→4 case *does* break symmetry
   correctly even with quadratic Q (outputs track the pattern), so this
   reasoning is not the whole story; treat it as a hypothesis, not settled.

3. **Conductance range must be wide.** With `g_min=0.1`, `w∈[0.3,0.7]` the resistance
   spread is only ~2×; all outputs converge to the same value and stay there.
   Working values: `g_min=0.01, g_max=1.0`, initial `w∈[0.05,0.95]` (≈ ×30 spread).
   This was the change that made 2→2→2 learn.

4. **EMA vs. snapshot.** `compute_Q_from_voltages` (snapshot of final state) was added as
   a diagnostic alternative to `integrate_observable_ema`. When phases are long relative to
   `tau_integrate` they agree; snapshot is less physical. Currently `test_plasticity.py` prints
   both. **Correction: `test_plasticity.py`'s actual MSE/training loop uses the snapshot
   (quadratic) path, not EMA - "prints both" refers only to the cycle-10 diagnostic block.**

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
- **(2026-09-14)** Switching the observable to the literal linear signed `Q = V_j - V_i`
  on the 4→3→4 case — makes it worse (MSE 0.25→0.41), see correction above.
- **(2026-09-14)** Driving `w` purely from each edge's own local `Q_avg` with no
  reference to compare against — collapses every weight to `g_min` (no learning
  signal at all, not just a worse plateau). See "Global vs local theta" above.

---

## Open Problems

### 1. Plateau in 4→3→4 (vanishing gradient?)

Diagnostics at the plateau vs. early cycles:

    early:   delta_Q ∈ [−0.035, 0.48],   dw ∈ [−0.055, 0.036]
    plateau: delta_Q ∈ [−0.007, 0.25],   dw ∈ [−0.002, 0.003]

As MSE drops, free and clamped states get closer, the contrast shrinks, and updates die
out before reconstruction is good. Weights are not stuck at the [0,1] boundaries
(mean≈0.48, std≈0.24, min≈0.07, max≈0.92), so it is not saturation.

**(2026-09-23, supersedes a weaker 2026-09-14 note) Independent evidence the plateau is
not an artifact of one rule's plumbing.** The online global-theta rule — structurally
different, no explicit two-snapshot contrast anywhere — was run through the real
`Trainer`, not a prototype:

| path | final MSE |
|---|---|
| explicit contrastive (two snapshots) | 0.050 |
| online global theta, free phase truncated | 0.033 |
| online global theta, free phase fully relaxed | 0.052 |

Done properly (fully relaxed), the two agree to within 4%: **0.052 vs 0.050**. They get
there by qualitatively different trajectories — smooth descent versus
collapse-then-recovery — which makes the agreement more meaningful, not less. Two
independent algorithms hitting the same ceiling is now reasonably strong evidence this is
a real property of this topology / observable / saturation combination, rather than a bug
in either rule. It is still not a proof — both share the same electrical model, the same
quadratic `Q`, and the same `(1-w)` saturation term, so a common cause in any of those
would produce exactly this agreement. The `1/beta` normalization question below remains
the most likely such common cause and is the thing to attack next.

**Note:** 4→3→4 is the minimal meaningful test. The target architecture is ~64→8→64
(8× compression), so this must work here first. Do not blame the bottleneck.

Candidate next steps (untested):
- **Momentum** on `dw` (physically: ionic/thermal inertia). Draft: keep `self.velocity`,
  `velocity = momentum*velocity + dw`, `w += velocity*dt_plasticity`. Try `momentum≈0.9`.
- Revisit the observable: current is voltage; could be current `I_ij` or signed power.
- Check whether the linear observable + this rule is actually a gradient of anything
  (equilibrium-propagation consistency) — we have not derived it, only tuned it.
  Concretely: the standard EP gradient estimator is `(1/beta) * (Q_beta - Q_0)` in the
  beta→0 limit; this rule uses the raw, un-normalized `(Q_clamped - Q_free)`, and `eta`/`beta`
  are known to be degenerate (see "Increasing eta" above) - that degeneracy is a symptom of
  the missing `1/beta` normalization, and is worth deriving properly rather than re-tuning
  around.
- Try the global-theta local rule (`training/rules.py`) through the real `Trainer` for a
  full-length run (the informal prototype run above used a fast vectorized stand-in, not
  the actual `Memristor`/`Trainer` objects) and see if it actually breaks past ~0.04.

### 2. Beta / dt stability

Explicit Euler with strong penalty is stiff. If larger `beta` is needed, either lower `dt`
or switch the integrator (`_compute_time_derivative` is separated from the integration
scheme for exactly this reason — RK4 or `scipy.integrate` is a drop-in).

**Measured 2026-09-23** (see "Stability boundary, quantified" above): critical `dt` ≈0.0020
at `beta=100, g_penalty=10`, so the default `dt=0.001` has only ~2x margin. Divergence is
now detected and reported (`result['diverged']`) instead of silently producing inf/nan,
but the underlying stiffness is unaddressed — switching integrator is still the real fix,
and it becomes more attractive now that the cost extrapolation (above) argues for
vectorizing that same inner loop anyway.

### 3. `test_plasticity.py` is messy

It has accumulated a lot of inline diagnostics (EMA vs. manual Q, dw ranges, weight stats,
convergence flags, adaptive beta remnants). It was ported to Grid/Memristor as-is
(2026-09-14) to keep the fix minimal and reviewable; the cleanup below is still
pending.

### 4. `training/trainer.py` and the Bars & Stripes dataset

It runs the free/clamped protocol for a single pattern (`run_cycle(pattern)`).
**Updated 2026-09-23:** looping it over `datasets.bars_stripes.BarsAndStripes` is now
demonstrated to work (`test_smoke_sweep.py`, section 4e: all 6 N=2 patterns, 3 epochs,
~180 ms/cycle, stays finite, distinct inputs give distinct outputs). So the wiring is no
longer the open part — *quality* is: mean MSE sits near 0.46–0.50 and does not improve
across epochs, and the per-output spread between different patterns is only ~0.06, i.e.
the network is barely discriminating between inputs. Whether that is the same
underlying problem as the 4→3→4 plateau (#1) or a separate multi-pattern interference
issue is untested. A convenience `Trainer.run_epoch(patterns)` would also be worth adding
so every caller stops writing the same loop.

### 5. Scale: ~5 hours per 300-cycle run at 64→8→64

Measured 2026-09-23, see "Cost at the target scale" above. `_compute_time_derivative`
loops over all node pairs in Python; vectorizing it into dense matrix ops is the single
highest-leverage change in the codebase right now, and blocks doing physics at the
proposal's target size.

---

## Conventions

- Parameters live in a `@dataclass` config at the top of each test file
  (`SimConfig` in `test_network_visualization.py`, `PlasticityTestConfig` in `test_plasticity.py`).
  No hard-coded `dt` / `max_steps` scattered through a file.
- Research-code style: short docstrings, minimal validation, parameters as function args.
- Tests are plain scripts runnable with `python3 tests/<name>.py`; they save plots to cwd.
  (`test_bars_stripes.py` is the exception - it's a real pytest file, needs `pytest` installed.)
- Setup: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pip install -e .`
  (`pip install -e .` is what makes `from network... / from training...` resolve without the
  `sys.path.insert` boilerplate at the top of each test file.)
- New convention (2026-09-14): building a `Grid` + `Memristor` array from a plain
  (adjacency, weights, iv_function) description - the shape most tests/experiments start
  from - goes through `network/builders.py` (`build_memristor_array` for plastic edges,
  `build_static_memristor_array` for a fixed, non-learning solver, `extract_weights`/
  `set_weights` to read/write a plain matrix back). Don't hand-roll this per test.

---

## Next Steps (rough order)

*Reordered 2026-09-23 after the smoke sweep.* Vectorizing the solver moved to the top:
every remaining physics question needs long runs, and right now a 300-cycle run at target
scale costs ~5 hours. Making experiments cheap comes before running more of them.

0. **Vectorize `_compute_time_derivative`** (Open Problems #5). Replace the Python
   double loop over node pairs with dense matrix ops. Expect 1–2 orders of magnitude;
   `test_smoke_sweep.py` + `test_plasticity.py` (which reproduces exact numbers) together
   make this safe to do as a pure refactor with a numerical check at the end.
1. ~~Run the global-theta local rule through the real `Trainer`~~ **Done 2026-09-23.**
   It lands on the same ceiling: 0.033 with the truncated relaxation, 0.052 with the
   free phase fully settled, against the explicit-contrastive 0.050. Two structurally
   different update rules reaching the same number is now decent evidence that the
   plateau is a property of the model or the observable, not of either rule — which
   sharpens Open Problems #1 considerably.
2. Derive the EP gradient correspondence properly for this rule (the missing `1/beta`
   normalization question) instead of continuing to hand-tune around the plateau.
3. Try momentum on `dw` as a cheaper thing to test first.
4. Clean up `test_plasticity.py` (Open Problems #3).
5. Investigate multi-pattern quality on Bars & Stripes - the loop now runs, but the
   network barely discriminates between patterns (Open Problems #4). Add
   `Trainer.run_epoch(patterns)` while in there.
6. Training visualization (MSE vs. cycle, weight evolution, reconstruction animation).
7. Load-test at (or scale down gracefully toward) the target ~64→8→64 architecture -
   cost is now measured (~3 ms/step, ~60 s/cycle; see "Cost at the target scale"), but
   nothing has actually been *trained* at that size.
