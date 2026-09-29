# MeroCircuit — status, diagnosis and research plan

*Last updated 2026-09-29. Audience: coding agents (Codex) and the team. Read
[`AGENTS.md`](../AGENTS.md) first for working rules.*

## 0. How to use this document

Sections 1–2 describe the model and the code as it is. Section 3 is a measured diagnosis
of why learning stalls; every number there is reproducible with `sim.py` or with the
scripts in [`experiments/2026-09-24_diagnostics/`](../experiments/2026-09-24_diagnostics/)
(commands in section 7). Section 4 is the plan: ordered work packages (WP), each with
deliverables, acceptance criteria and current status. Section 5 lists physics decisions
that belong to the team, not to an agent — implement those options behind flags with
unchanged defaults, and report; do not choose.

Claims are marked by strength: **measured** (reproducible command), **derived** (an
argument, checked numerically where noted), **hypothesis** (not yet tested). Do not
promote a hypothesis in the proposal or in code comments.

Scope for agents: WP0–WP3 are engineering (done, see status lines). WP4–WP7 are research:
run the experiments, report honestly (including negative results), and do not tune until
a number looks good.

## 1. The model

**Network.** Graph with node potentials `V_i` and adaptive edges. Input nodes are clamped
to the pattern times the signal amplitude `V_sig` (default 1 V). Output nodes are paired
one-to-one with inputs. The default topology is a bipartite autoencoder `n_in → n_h → n_in`
(every input connected to every hidden node, every hidden to every output; no
input–output edges). Optional extra clamped nodes: bias (0 and `V_sig`), complementary
inputs (`V_sig − x`).

**Edges** (`network/elements.py`). State `w ∈ [0,1]`, conductance
`g = g_min + (g_max − g_min)·w`, `g_min = 0.01`, `g_max = 1`. Current `I(g, ΔV)`:

| element | current | notes |
|---|---|---|
| `ohmic` | `g·ΔV` | |
| `relu` | `g·sign(ΔV)·max(|ΔV| − V_th, 0)` | dead zone, `V_th = 0.1` (eq. 15 of the proposal); repo default |
| `tanh` / `sigmoid` | `g·tanh(k·ΔV)` | `sigmoid` = historical name for `k = 10` |
| `sinh` | `g·V0·sinh(ΔV/V0)` | |
| `rectpair` | two antiparallel branches, each `g_b·max(ΔV − V_f, 0)` | independent states per branch; ideal diode, optional forward drop `V_f` |
| `shockley` | two antiparallel branches, each a Shockley diode in series with the filament `g_b`: `ΔV = n·V_T·ln(1 + I/I_s) + I/g_b` | closed form via Wright omega; `V_T = 25.85 mV` |

**Co-content.** For an element with current `I(V)` the co-content is
`Φ(V) = ∫₀^V I(V') dV'` (Millar 1951; the "dual" of the energy stored in a resistor, it is
the quantity a network of nonlinear resistors minimizes). For an ohmic edge `Φ = g·V²/2`
(half the dissipated power); for a dead-zone ReLU `Φ = (g/2)·max(|V| − V_th, 0)²`. The EP
observable of an element is `∂Φ/∂g` at its own voltage — `V²/2` only for ohmic edges.
For the Shockley branch it is `(I/g)²/2`: half the squared voltage across the filament,
not across the whole branch.

**Energy.** Stationary states minimize
`E(V; w, β) = Σ_branches Φ_b(ΔV_b; g_b) + (β·g_p/2)·Σ_k (V_out,k − V_in,k)²`. The second term is
the penalty link; only the product `β' = β·g_p` enters the physics.

**Inference** = relaxation `C·dV/dt = −∂E/∂V` from `V_free = 0` with inputs clamped.
**Free phase** `β = 0`; **nudged (clamped) phase** `β > 0`. Two protocols
(`--protocol`): `exposure` — explicit Euler for a fixed step budget (the historical
"fixed exposure time"; the free phase is usually *not* at equilibrium), and `equilibrium`
— fully relaxed stationary state.

**Learning.** Three rules, all in `training/learning.py` and `sim.py train --rule`:
- `contrastive` (historical, `training/plasticity.py`): `dw = −η·(Q_clamped − Q_free)·(1 − w) − γ·w`
  with `Q = ΔV²` from two snapshots; strong nudge `β' = 1000`.
- `global-theta` (historical online rule, `training/trainer.py`): each memristor averages
  its own `Q` over a window; `dw = −η·(Q_avg − θ)·(1 − w) − γ·w` with one slowly adapting
  network-wide `θ`. No phase label reaches any element.
- `ep` (equilibrium propagation): `Δw = −α·[∂Φ/∂g(nudged) − ∂Φ/∂g(free)]/β'`, weak nudge
  (default `β' = 0.1`), optional symmetric `±β'` and soft bounds.

`contrastive` and `ep` need each element to know which phase it is in (to sign its
observable): an implicit **clock**. Only `global-theta` is clock-free.

**EP, for reference** (derived; Scellier & Bengio 2017). For small `β'`,
`∂L/∂g ≈ (1/β')·[∂Φ/∂g(ΔV^β) − ∂Φ/∂g(ΔV^0)]`. Consequences: the sign of the update is
**negative** (the code uses `−η`; eq. 14 of the proposal has `+η` and should be corrected);
the observable is the co-content derivative of *that* element; the nudge must be *weak*
and the difference divided by `β'`. For elements with sharp curvature the nudge must be
weak on the element's own voltage scale (section 3.6).

## 2. Where the code stands

- **Engine** (`network/elements.py`, `network/equilibrium.py`): vectorized relaxation for
  every element, both protocols, exact gradients. The explicit Euler path reproduces
  `VoltageDynamics.relax_transient` to machine precision (ohmic/ReLU/sigmoid, free and
  `β' = 1000`); `sim.py bench` measures ~17–23 µs per Euler step from 4-3-4 to 25-8-25
  (the object engine: milliseconds per step at 64-8-64).
- **Rules and loop** (`training/learning.py`): the three rules, shuffled epochs,
  held-out split, metrics. The historical numbers are reproduced exactly and 8–9× faster:
  `contrastive`, 40 cycles → 0.043978 (13 s instead of 117 s); `global-theta`, 25 cycles →
  0.032565 (7 s instead of 59 s).
- **One CLI**, `sim.py`: `relax`, `train`, `oracle`, `handbuilt`, `sweep`, `stability`,
  `bench`, `info`. (`experiments/2026-09-24_diagnostics/play.py` was folded into it.)
- **Legacy engine** (Grid + Memristor objects + `VoltageDynamics` + `Trainer`) is kept for
  the library API and the regression canaries; nothing in it changed physically.
- **Tests / canaries:** `tests/test_smoke_sweep.py` → 66 passed (~2–3 min);
  `tests/test_plasticity.py` → `Final MSE: 0.050202`; `tests/test_engine.py` → 58 passed
  (~30 s: legacy agreement, element derivatives, co-content, equilibrium, gradient checks,
  EP alignment, hand-built 14/14, learning).
- **Metrics:** MSE; bit accuracy (threshold 0.5); exact-pattern fraction; margin
  `min |V_out/V_sig − 0.5|`; and the **distribution of wrong pixels per pattern** (on
  Bars & Stripes errors come in whole lines, so the distribution has gaps).
- **Changed behaviour** (2026-09-29): `sim.py relax` now maps `w → g` exactly like
  training does; before it used the raw `w` as conductance, so its printed voltages
  differ slightly from earlier versions.
- **Found, not fixed** in legacy code: `network/iv_characteristics.diode_iv` does not
  conserve current when an edge is evaluated from both ends (the net current summed over
  the network is +4.67 on the 4-3-4 test net), so it was not ported; the `linear`
  observable in `training/rules.py` is not reciprocal for undirected edges
  (`sign(ΔV)` flips with the orientation) and was not ported either.

## 3. Diagnosis

Tools: the engine above; exact gradients of the reconstruction MSE through the
equilibrium (implicit function theorem, finite-difference checked); an **oracle** —
projected Adam with those gradients. The oracle is not physical; it estimates the best
reconstruction *any* conductance setting of an architecture can reach.

### 3.1 The single-pattern plateau is the transport threshold (measured, 2026-09-24)

A hand-wired *ideal* 4→2→4 network for `[1,0,1,0]` (strong edges exactly where needed),
inference from `V = 0` (`threshold_floor.py`):

| V_th | g_min = 0.01: output, MSE | g_min = 1e-4: output, MSE | (2·V_th)² |
|---|---|---|---|
| 0 | [0.96 0.04 0.96 0.04], 0.0014 | [1.0 0.0 1.0 0.0], 0.0000 | 0 |
| 0.05 | [0.87 0.13 0.87 0.13], 0.0172 | [0.9 0.1 0.9 0.1], 0.0101 | 0.01 |
| **0.1** | [0.78 0.23 0.78 0.23], 0.0504 | **[0.8 0.2 0.8 0.2], 0.0401** | **0.04** |
| 0.2 | [0.59 0.41 0.59 0.41], 0.1692 | [0.6 0.4 0.6 0.4], 0.1601 | 0.16 |

Mechanism (derived). Output nodes float: no current leaves them except through the
network. Under gradient flow from `V = 0` a floating node is pulled toward its strongly
connected neighbour only while the drop exceeds `V_th`; inside the dead zone the strong
edge carries no current, so the node stops `V_th` short (then only the weak `g_min`
leaks move it). Along a path of `h` edges from a clamped input to a floating output the
shortfall adds up hop by hop: here `h = 2`, so a `1` arrives as `1 − 2·V_th = 0.8` and a
`0` as `0.2`, MSE `(2·V_th)²`. The cost depends on the **path length** (number of layers),
not on the width, and it applies to floating nodes relaxed from `V = 0`; a node held by
current from both sides (a hidden node between two driven nodes) sits wherever the
balance puts it. The `global-theta` stopping point `[0.779, 0.128, 0.779, 0.128]` (MSE
0.0326) is this floor; the fixed-exposure protocol (0.033) looks better than the fully
relaxed network (0.052) most likely because the leak through `g_min` has time constant
`C/g_min ≈ 100`, much longer than the exposure of 10. The oracle lands at 0.0504: a ceiling,
not a failure to learn.

Side effect: gradient flow parks many edges exactly at `|ΔV| = V_th` (the kink), so the
loss is only piecewise smooth at the operating point; the oracle optimizes a smoothed
surrogate for `relu` and evaluates with the exact curve.

The proposal (section "Operating regimes") asks for inference amplitudes comparable to
`V_th` so that the nonlinearity participates. That is in direct tension with this floor.
Note that the smooth `sigmoid` (`tanh(10 V)`) has no dead zone, which is why the best
single-pattern logs came from it.

### 3.2 The learning rule runs in the wrong regime (measured, 2026-09-24)

Alignment (cosine) between the EP estimate and the true gradient, ohmic 4→3→4 at the repo's
initial weights (`symmetric_elements.py`, E2; `[1,0,1,0]` / 2×2 B&S):

| β·g_p | one-sided EP estimate | symmetric (±β) estimate | repo-style update* vs descent |
|---|---|---|---|
| **1000 (repo)** | **0.32 / 0.34** | — | **0.35 / 0.41** |
| 100 | 0.33 / 0.35 | — | 0.35 / 0.42 |
| 10 | 0.36 / 0.38 | — | 0.37 / 0.44 |
| 1 | 0.60 / 0.61 | 0.09 / 0.04 | 0.55 / 0.60 |
| **0.1** | **0.975 / 0.970** | **0.999 / 0.999** | 0.85 / 0.85 |
| 0.01 | 1.000 / 1.000 | 1.000 / 1.000 | 0.90 / 0.89 |

\*`−ΔQ·(1 − w)` with `Q = ΔV²`: the `(1 − w)` factor alone costs ~10% alignment even at weak
nudge. Symmetric nudging at `β·g_p = 1` fails because the anti-nudge `−β` is a negative
conductance comparable to the network's and distorts the state. The bias grows with `β'`
and depends on the weights: at seed 1, summed over 2×2 B&S, the one-sided cosine is 0.88
at 0.1 and 0.998 at 0.01 (`tests/test_engine.py` uses 0.01).

Learning on 2×2 B&S, 4→3→4, 60 epochs, one pattern per update, fully relaxed phases (E3):

| rule | element | final MSE | exact | ceiling (oracle) |
|---|---|---|---|---|
| repo: `−1.05·ΔV²-contrast·(1−w)`, β·g_p = 1000 | ohmic | 0.087 | 0.67 | 0.0427 |
| repo | ReLU | 0.17 (no learning) | 0.33 | 0.0634 |
| EP: `−α·ΔΦ/β'`, β' = 0.1, α = 0.5 | ohmic | **0.044** | 0.67–1.00 | 0.0427 |
| EP symmetric, β' = 0.1 | ohmic | 0.045 | 0.67–1.00 | 0.0427 |
| EP, β' = 0.1 | sinh | 0.061 | 1.00 | 0.054 |

Weak-nudge EP reaches the ceiling; the repo rule does not. "Exact" oscillates between 0.67
and 1.00 at the ceiling because the ceiling's margin is 0.008.

Not yet tested: the `(1 − w)` factor multiplies depression as well as potentiation, so
depression vanishes near `w = 1`. `--soft-bounds` (potentiation ∝ `1 − w`, depression ∝ `w`)
is implemented for `ep`; no study yet.

### 3.3 Passive networks of symmetric elements cannot represent Bars & Stripes (measured + derived)

Oracle ceilings (`symmetric_elements.py`, E1; best of 2 restarts, 1 for 3×3):

| data | n_h | ohmic MSE / exact | ReLU (V_th=0.1) | sinh (V0=0.25) |
|---|---|---|---|---|
| [1,0,1,0] | 2 | 0.0014 / 1.00 | 0.0504 / 1.00 | 0.016 / 1.00 |
| 2×2 B&S (6) | 2 | 0.084 / 0.67 | — | 0.089 / 0.67 |
| 2×2 B&S | 3 | 0.043 / 1.00, margin 0.008 | 0.063 / 1.00, margin 0.10 | 0.054 / 0.67 |
| 2×2 B&S | 4 | 0.0033 / 1.00 | — | 0.022 / 1.00 |
| 3×3 B&S (14) | 3 | 0.110 / 0.43 | — | 0.119 / 0.29 |
| 3×3 B&S | 5 | 0.067 / 0.14 | — | 0.083 / 0.29 |

Bias nodes and/or complementary inputs did **not** help: 2×2, n_h = 3, ohmic: 0.045 /
0.044 / 0.047 (bias / dual / both); 3×3 dual+bias n_h = 5: 0.102 / 0.43.

Why, for the ohmic case: a passive network obeys the maximum principle, so
`V_out = M·V_in` with `M` non-negative and row-stochastic, and `rank M ≤ n_h`. Exact
reconstruction of 2×2 B&S requires `M` to be the identity on the data span
`{1111, 0011, 0101}`; the only non-negative row-stochastic matrix doing that is `I` itself
(rank 4 > 3). More generally the N×N B&S patterns span `2N − 1` dimensions: 5 for 3×3,
**15 for 8×8**. The proposal's 64→8→64 target is below the linear rank. Symmetric
nonlinearities did not change this picture. Complementary inputs do not help a passive
network because the all-zeros and all-ones patterns force their contributions to cancel.
The universality construction for resistive networks (Scellier & Mishra 2025) needs
paired units *and* bias sources *and* VCVS gain; Kendall et al. (2020) use amplifiers.

### 3.4 Rectifying antiparallel pairs: representable, partly learnable (measured)

Bars & Stripes has AND/OR structure: a pixel is on iff its row OR its column is on, and a
line detector is an AND over its pixels. Passive **rectifying** elements compute max (OR)
and min (AND) natively; symmetric elements only average.

Element (`rectpair`): each edge `{a,b}` is two antiparallel rectifying branches with
independent states. Equal states: a resistor. One branch at `g_min`: a diode — the edge
learns its own rectification direction. The element is *symmetric* (the same device
either way round); the *state* breaks the symmetry. Still passive; still EP-compatible
(each branch's observable is its own co-content). Self-rectifying memristors are an
established device class (see references).

- **Existence proof**: a hand-built 9→6→9 network (3 row-AND + 3 column-AND detectors;
  each output = OR of its row and column detector) reconstructs **all 14** 3×3 patterns —
  MSE 0.062, margin 0.053 at `g_min = 0.01` (pull-up 0.2); MSE 0.025, margin 0.16 at
  `g_min = 1e-4`. By construction it generalizes. `sim.py handbuilt`.
- **Stable basin**: exact gradient descent started from it keeps 100% and improves MSE to
  0.049, margin to 0.098.
- **Random-init optimization does not find it**: oracle, 2×2 n_h=3: 0.043 / 1.00;
  3×3 n_h=6: 0.054 / **0.29**–0.71 (restart-dependent; the current CLI run gives 10/14
  patterns exact, the other 4 with 3 wrong pixels = one whole line); 4×4 n_h=8: 0.074 / 0.57.
- **A local rule learns on it** (weak-nudge EP, β' = 0.1, α = 0.5): 2×2 → 100% exact by
  epoch 60; 3×3, n_h = 6 → 64% exact at epoch 100, plateau 57–64% over 300 epochs.
- **Generalization, preliminary** (3×3, 10 train / 4 held out): oracle 60–80% train /
  25–50% test; local EP 60–70% / 50%. Not meaningful until training reaches 100%.

Summary: rectification removes the *representational* barrier; what remains is
*trainability* — getting from random states into the AND/OR basin.

### 3.5 Clock-free learning: the window effect is real, its explanation is open

**Measured.** The `global-theta` rule learns `[1,0,1,0]` only when each memristor's window
spans exactly one phase (`window_pts / micro_steps` = 1 → MSE 0.033; 0.5, 0.25, 0.05 →
0.493; 2026-09-23). With the one-phase window but **no nudge** (`--beta 0`) it also ends at
0.4925 (2026-09-29): the learning *does* come from the free/clamped contrast, not from the
homeostatic term alone. On 2×2 B&S it has not been shown to learn.

**The old explanation is disputed.** It said: V is constant within a phase, so a shorter
window forgets the other phase, and "the lag of a one-phase window *is* the memory of the
other phase", hence "the device's relaxation time must match the drive period". Against it
(derived, linear response): a phase-blind rule that is linear in `Q_avg − θ`, with weights
that change little over a cycle, integrates over a free + clamped cycle (duration `T` each)

  `Σ dw ∝ −T·(Q_f + Q_c − 2θ) = −(Q_c − Q_f)·T − 2·(Q_f − θ)·T`.

A window with unit DC gain changes *when* the signal arrives but not its integral, so to
first order the window shape drops out. The first term is the contrast; the second is a
homeostatic term that compares the edge's free-phase activity with the network average
and carries no error signal.

**The two do not fit together yet.** The measurement says the window matters decisively;
the argument says it cannot matter to first order. The argument's premise is probably
what fails: with the historical settings (`η = 1.05`, exposure 10, `Q ~ 0.1–1`) a single
phase moves `w` by O(1), and the rule has the nonlinear factor `(1 − w)` and a hard clip
at 0 and 1, so *when* within the cycle the push arrives changes where it lands
(hypothesis). If so, the window effect is a property of fast, strongly nonlinear
plasticity, not a device-timing constraint, and a slow-plasticity version would lose it.
**Do not put the window mechanism in the proposal** until this is settled.

Separately (derived): on datasets the homeostatic term differs per pattern, and with a weak
nudge it is O(1) against an O(β') contrast; a clock-free rule needs *some* local signal
correlated with the phase to extract the sign of the contrast.

Checks (Andrey): (1) slow plasticity — `--eta 0.1` and `--eta 0.01` with proportionally
more `--cycles`, window 200 vs 100: does the window dependence disappear? (2) count how
many weights sit at the clip bounds at the end (`--json`, field `w`) for window 200 vs 100;
(3) the same on `--dataset bars-stripes --n-input 4`.

### 3.6 Realistic elements (measured, 2026-09-29)

- **Forward drop.** Ideal pairs with `V_f > 0` recreate a dead zone. Only `V_f/V_sig`
  matters (the equations are scale-invariant). The hand-built 3×3 network keeps 14/14 up
  to `V_f/V_sig ≈ 0.1` (margin 0.003 at 0.1; comfortable at 0.05).
- **Shockley + filament.** No hard threshold, but a rectification ratio `R` costs a
  turn-on drop ≈ `n·(kT/q)·ln R` (Boltzmann): useful signals are ~1–3 V. Hand-built 3×3
  works at `V_sig = 1`, `I_s = 1e-4`, `n = 1` (14/14, margin 0.006). EP on Shockley
  branches aligns with the gradient only when the nudge shifts voltages by ≪ kT/q:
  cosine −0.04 at `β' = 1e-2`, 0.18 at 1e-3, 0.95 at 1e-4, 1.00 below; symmetric nudging
  0.996 at 1e-3. `sim.py` defaults: 1e-4, or 1e-3 with `--sym`. Open: millivolt nudges vs
  kT/C noise on femtofarad nodes. Local learning on Shockley pairs is slow and untuned
  (2×2, `--sym --alpha 10`: MSE 0.167 → 0.126, 67% exact after 40 epochs; ceiling 0.091).
- **Saturation blocks the nudge.** `tanh(10 V)` at 1 V signals: good ceiling on 2×2
  (0.033) but the local rule does not learn — saturated edges pass no nudge (the analog
  of vanishing gradients). `tanh(2 V)` learns (MSE 0.040 vs ceiling 0.038). The
  saturation scale must be comparable to the signal.
- **On/off ratio scales with fan-in.** Hand-built 4×4 (16-8-16): 73% at `g_min = 0.01`
  whatever the pull-up, 100% at `g_min = 0.001`: summed leakage grows with fan-in.

### 3.7 Open, not yet tested

- **Symmetric nudging needs an active element**: the anti-nudge is a negative conductance.
- **Sign of eq. (14) of the proposal**: see section 1.

## 4. Work packages

### WP0 — Fast, exact solver — **done** (2026-09-29)

Implemented as a separate engine (`network/equilibrium.py`) rather than inside
`VoltageDynamics`: vectorized Euler with identical semantics, `relax_equilibrium`
(Euler warm start + active-set Newton), `loss_and_grad` (implicit function theorem).
Checked in `tests/test_engine.py`. Not done: routing `VoltageDynamics`/`Trainer`
themselves through it (the canaries still run the object engine; `sim.py` does not).

### WP1 — Evaluation harness — **partly done**

Done: metrics incl. wrong-pixel distribution; `--holdout k` (never all-zeros/all-ones);
per-epoch shuffling (`--order`); `--json` output with arguments, history, final metrics
and weights. Open: multi-seed runner (`n_seeds ≥ 3` by default), git hash in the JSON,
experiment configs.

### WP2 — EP-consistent learning rule — **done**, study open

`--rule ep` with `β'` configurable (default 0.1), `∂Φ/∂g` from each element, `--sym`,
`--soft-bounds`. Alignment checked in `tests/test_engine.py` (cos ≥ 0.95 at β' = 0.01 for
ohmic, tanh, rectpair; 1e-4 for shockley). Open study: `β' ∈ {1000, 10, 1, 0.1, 0.01}` ×
{contrastive, ep, ep + soft bounds}, multiple seeds, final MSE / exact / margin vs the
oracle; whether soft bounds reduce weights at the bounds and whether that matters.

### WP3 — Rectifying pairs as an edge element — **done**

`--element rectpair` (optional `--vf`) and `--element shockley`; hand-built 9→6→9 gives
14/14 (`sim.py handbuilt`, `tests/test_engine.py`).

### WP4 — Trainability on rectifying pairs (main research question)

Target: a **local** rule reaches 14/14 exact on 3×3 B&S in ≥ 3 of 5 seeds, then
generalizes to held-out non-trivial patterns. Baseline to beat: 57–64% exact.

Levers — test one at a time against the baseline, 5 seeds each, 300 epochs:
1. Initialization: asymmetric (per edge one branch strong, the other near `g_min`, random
   direction); sparse; small-and-equal.
2. Noise/annealing in `Δw`; learning-rate schedules.
3. Soft bounds.
4. Over-complete hidden layer (n_h = 9, 12) with a mild decay to prune.
5. Curriculum: single-line patterns first.
6. Nudge strength and symmetric nudging (see D3).
7. Competition between hidden units by a *passive* mechanism; document the physics.
8. A nonlinear penalty link that rewards margin rather than MSE.

Report for each lever: exact / MSE / margin / wrong-pixel distributions over seeds, and
whether the learned weights resemble line detectors (see WP7).

### WP5 — Transport threshold and signal scale (study)

Quantify 3.1 against `V_th / V_sig`: floor for 1, 2, 3 hops; inference from mid-rail
(0.5) instead of 0; smooth alternatives (`tanh` with `k·V_sig ~ 1–3`, `sinh`) and their
leakage; forward drop and Shockley turn-on (3.6). Keep the proposal's requirement
(nonlinearity must participate at inference amplitudes) explicit.

### WP6 — Clock-free (online) learning (research)

First settle 3.5: explain the window dependence on one pattern — log the contrast and
homeostatic parts of `Σ dw` per edge separately, and test whether it survives slow
plasticity. Then look for a clock-free mechanism that supplies a phase-correlated local
signal (e.g. the nudge itself modulating something the element can sense; frequency
propagation, Anisetti et al. 2024). Compare with `--rule ep` on 2×2 B&S. Positive result:
within 10% of the EP rule's final MSE, 3 seeds.

### WP7 — Scaling and latent structure (research, after WP4)

- 4×4 B&S (30 patterns): rectifying pairs, n_h = 8; `g_min ≤ 0.001` (3.6).
- Latent analysis: hidden voltages per pattern; do units become row/column detectors?
  Linear readout (bars vs stripes); generative test — clamp hidden nodes, read outputs.
- Compression: a line-sharing code (N shared line units + 2 orientation units) would need
  about N + 2 bottleneck nodes but extra logic layers. Unverified sketch; see D5.

## 5. Decisions reserved for the team

- **D1 Element.** Adopt rectifying antiparallel pairs as the default edge? (The CLI default
  is still `relu`.)
- **D2 Gain.** Allow active elements (amplifiers/VCVS)? Required by the known universality
  construction; changes the "passive substrate" story.
- **D3 Anti-nudge.** Symmetric nudging needs a negative conductance (active). Allowed?
- **D4 Operating point.** `V_th / V_sig`, `V_f / V_sig`; for Shockley elements the signal
  (1–3 V) vs the nudge (≪ kT/q); inference from `V = 0` or mid-rail.
- **D5 Target.** 64→8→64 is below the linear rank (15) and below the 2N = 16 AND/OR
  construction. Keep 8, move to ~10–16, or add depth?
- **D6 Protocol.** Make the fully relaxed equilibrium the default for physics experiments?
  (`--rule ep` already defaults to it; the historical rules keep `exposure` so the canary
  numbers hold.)
- **D7 Proposal.** Correct the sign in eq. (14); name the observable as the co-content;
  drop the window/phase-matching claim unless 3.5 is resolved in its favour.
- **D8 Clock.** Accept an explicit phase signal per element (EP proper), or require
  clock-free learning (WP6)?

## 6. Guardrails (full list in AGENTS.md)

Never change physics defaults silently; a refactor that claims to be physics-neutral must
reproduce `0.050202` (legacy) and `0.043978` / `0.032565` (engine). Tests must assert that
learning reduces error, not only that values stay finite. Check `diverged`. Report
multiple seeds and the wrong-pixel distribution, not only MSE.

## 7. Reproducing section 3

```bash
python3 sim.py train                                   # 2: contrastive, 0.043978
python3 sim.py train --rule global-theta --cycles 25   # 2: 0.032565
python3 sim.py oracle --element relu --n-hidden 2      # 3.1: ceiling on [1,0,1,0]
python3 sim.py oracle --element ohmic --dataset bars-stripes --n-input 9 --n-hidden 5
python3 sim.py handbuilt --show-patterns               # 3.4: 14/14
python3 sim.py oracle --element rectpair --dataset bars-stripes --n-input 9 --n-hidden 6
python3 sim.py train --rule ep --element rectpair --dataset bars-stripes --n-input 9 \
        --n-hidden 6 --epochs 100                      # 3.4: local EP on 3x3
python3 sim.py handbuilt --vf 0.1                      # 3.6: forward drop
python3 sim.py handbuilt --element shockley --vsig 1   # 3.6: Shockley branches
python3 sim.py handbuilt --n-input 16 --g-min 0.001 --pullup 0.1   # 3.6: 4x4
python3 sim.py train --rule ep --element tanh --steepness 2 --dataset bars-stripes --n-input 4 --epochs 40
```

Tables of 3.1–3.4 as published, from `experiments/2026-09-24_diagnostics/`:

```bash
python3 threshold_floor.py                          # 3.1
python3 symmetric_elements.py /tmp/sym.json         # 3.2, 3.3   (~3 min)
python3 rectifying_pairs.py /tmp/rect.json          # 3.4        (~4 min)
python3 rectifying_local_ep_long.py /tmp/long.json  # 3.4, long local-EP runs
python3 shockley_pairs.py                           # 3.6
```

## 8. References

- Millar, *Some general theorems for non-linear systems possessing resistance*, Phil.
  Mag. 42, 1150 (1951) — content and co-content.
- Scellier & Bengio, *Equilibrium propagation*, Front. Comput. Neurosci. 11, 24 (2017).
- Laborieux et al., *Scaling equilibrium propagation to deep ConvNets by drastically
  reducing its gradient estimator bias*, Front. Neurosci. (2021) — symmetric nudging.
- Kendall et al., *Training end-to-end analog neural networks with equilibrium
  propagation*, arXiv:2006.01981 (2020) — diodes, amplifiers, paired inputs.
- Stern et al., *Supervised learning in physical networks: from machine learning to learning
  machines*, Phys. Rev. X 11, 021045 (2021) — coupled learning in resistor networks.
- Dillavou et al., *Demonstration of decentralized physics-driven learning*, Phys. Rev.
  Applied 18, 014040 (2022); *Machine learning without a processor: emergent learning in a
  nonlinear analog network*, PNAS (2024).
- Scellier & Mishra, *A universal approximation theorem for nonlinear resistive networks*,
  Phys. Rev. Applied 23, 044009 (2025) — resistors, diodes, voltage sources, VCVS.
- Anisetti et al., *Frequency propagation: multimechanism learning in nonlinear physical
  networks*, Neural Computation 36(4) (2024) — contrast without phase switching.
- Guzman, Ciarella & Liu, *Unsupervised and probabilistic learning with contrastive local
  learning networks: the Restricted Kirchhoff Machine*, arXiv:2509.15842.
- *Self-rectifying memristors for beyond-CMOS computing: mechanisms, materials, and
  integration prospects*, Nano-Micro Letters (2025),
  https://link.springer.com/article/10.1007/s40820-025-02035-1.
