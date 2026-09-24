# MeroCircuit — status, diagnosis and research plan

*Last updated 2026-09-24. Audience: coding agents (Codex) and the team. Read
[`AGENTS.md`](../AGENTS.md) first for working rules.*

## 0. How to use this document

Sections 1–2 describe the model and the code as it is. Section 3 is a measured diagnosis
of why learning stalls; every number there is reproducible with the scripts in
[`experiments/2026-09-24_diagnostics/`](../experiments/2026-09-24_diagnostics/). Section
4 is the plan: ordered work packages (WP), each with deliverables and acceptance
criteria. Section 5 lists physics decisions that belong to the team, not to an agent —
implement those options behind flags with unchanged defaults, and report; do not choose.

Scope for agents: WP0–WP3 are engineering with clear acceptance tests. WP4–WP7 are
research: run the experiments, report honestly (including negative results), and do not
tune until a number looks good.

## 1. The model

**Network.** Graph with node potentials `V_i` and adaptive edges. Input nodes are clamped
to the pattern (`0`/`1` V). Output nodes are paired one-to-one with inputs. The default
topology is a bipartite autoencoder `n_in → n_h → n_in` (every input connected to every
hidden node, every hidden to every output; no input–output edges).

**Edges.** State `w ∈ [0,1]`, conductance `g = g_min + (g_max − g_min)·w` with
`g_min = 0.01`, `g_max = 1`. Current `I = g·f(ΔV)` with a fixed I-V curve `f`: ohmic,
thresholded ReLU with dead zone `f(x) = sign(x)·max(|x| − V_th, 0)` (`V_th = 0.1`, eq. 15
of the proposal), sigmoid, or diode (`network/iv_characteristics.py`).

**Energy.** With `Φ' = f`, stationary states minimize
`E(V; w, β) = Σ_edges g·Φ(ΔV) + (β·g_p/2)·Σ_k (V_out,k − V_in,k)²`. The second term is the
penalty link; only the product `β·g_p` enters the physics.

**Inference** = relaxation `C·dV/dt = −∂E/∂V` from `V_free = 0` with inputs clamped
(`network/dynamics.py`, explicit Euler). **Free phase** `β = 0`; **clamped phase** `β > 0`.

**Learning.** Two implemented paths on the same substrate:
- *Explicit contrastive* (`training/plasticity.py`): `dw = −η·(Q_clamped − Q_free)·(1 − w) − γ·w`
  with `Q = ΔV²` from two relaxed snapshots.
- *Online* (`training/trainer.py`, `training/rules.py`): each memristor averages its own
  `Q` over a boxcar window exactly one phase long; `dw = −η·(Q_avg − θ)·(1 − w) − γ·w` with one
  slowly adapting network-wide `θ`. No phase label reaches any element. The window must
  span exactly one phase or learning stops (DEVELOPMENT.md, "The window must span one phase").

**Equilibrium propagation (EP), for reference.** For small nudge `β' = β·g_p`, the gradient
of the reconstruction loss is `∂L/∂g ≈ (1/β')·[Φ(ΔV^β) − Φ(ΔV^0)]`, so gradient descent is
`Δg ∝ −(1/β')·[Φ(ΔV^β) − Φ(ΔV^0)]` — the co-content `Φ` of *that* element, evaluated in the
two phases. Three consequences used below: the sign is **negative** (the code uses `−η`;
eq. 14 of the proposal is written with `+η` and should be corrected); the local observable
is the element's co-content (`ΔV²/2` only for ohmic elements); and the nudge must be
*weak* relative to network conductances, with the difference divided by `β'`.

## 2. Where the code stands

- Solver, both learning paths, `sim.py` CLI, divergence detection, Bars & Stripes
  generator, plotting. See README.
- Regression canaries: `python3 tests/test_smoke_sweep.py` → 64 passed (~3 min);
  `python3 tests/test_plasticity.py` → `Final MSE: 0.050202`.
- Single pattern `[1,0,1,0]`, 4→3→4: contrastive rule → MSE 0.044–0.050 (40 cycles); online
  rule → 0.033 with the fixed-exposure protocol, 0.052 with the free phase fully relaxed.
  All bit-correct after thresholding at 0.5.
- Datasets: the loop runs (`sim.py train --dataset bars-stripes`) but no configuration
  learns 2×2 Bars & Stripes to its ceiling, and none reconstructs 3×3.
- Cost: pure-Python O(n²) inner loop; extrapolated to 64→8→64 it is several ms per step
  (3–18 ms measured on different machines/loads, `sim.py bench`), i.e. hours per
  training run. The free phase needs ~40k steps at `dt = 0.001` to converge. Explicit Euler is stable only
  up to `dt ≈ 0.002` at `β·g_p = 1000`.

## 3. Diagnosis (measured 2026-09-24)

Tools (see the diagnostics README): a fast solver (Euler gradient flow + active-set Newton)
that matches `VoltageDynamics` to ~1e-9; exact gradients of the reconstruction MSE through
the equilibrium (implicit function theorem, finite-difference checked to ~1e-9); and an
**oracle** — projected Adam with those exact gradients. The oracle is not physical; it
estimates the best reconstruction *any* conductance setting of an architecture can reach.
Metrics: MSE; bit accuracy (threshold 0.5); **exact** = fraction of patterns with every
pixel right; **margin** = min |V_out − 0.5|.

### 3.1 The single-pattern plateau is the transport threshold

A hand-wired *ideal* 4→2→4 network for `[1,0,1,0]` (strong edges exactly where needed),
inference from `V = 0` (`threshold_floor.py`):

| V_th | g_min = 0.01: output, MSE | g_min = 1e-4: output, MSE | (2·V_th)² |
|---|---|---|---|
| 0 | [0.96 0.04 0.96 0.04], 0.0014 | [1.0 0.0 1.0 0.0], 0.0000 | 0 |
| 0.05 | [0.87 0.13 0.87 0.13], 0.0172 | [0.9 0.1 0.9 0.1], 0.0101 | 0.01 |
| **0.1** | [0.78 0.23 0.78 0.23], 0.0504 | **[0.8 0.2 0.8 0.2], 0.0401** | **0.04** |
| 0.2 | [0.59 0.41 0.59 0.41], 0.1692 | [0.6 0.4 0.6 0.4], 0.1601 | 0.16 |

Each hop costs `V_th`. Inside the dead zone a strong edge carries no current, so a node
drifts wherever the *weakest* leak pushes it until it sits `V_th` away; two hops cost
`2·V_th` on every output. The online rule's stopping point `[0.779, 0.128, 0.779, 0.128]`
(MSE 0.0326) is this floor. The fixed-exposure protocol looks better (0.033) than the
fully relaxed network (0.052), most likely because drift through a leak of conductance
`g_min` has time constant `C/g_min ≈ 100`, much longer than the exposure of 10 — the
truncation stops the drift early rather than doing anything useful. The oracle for this network also lands at 0.0504; it is a ceiling, not a failure to
learn.

Side effect: gradient flow parks many edges exactly at `|ΔV| = V_th` (the kink of `f`),
so the loss is only piecewise smooth at the operating point. Treat gradient-based analysis
of the dead-zone model with care.

The proposal (section "Operating regimes") asks for inference amplitudes comparable to
`V_th` so that the nonlinearity participates. That is in direct tension with this floor.

### 3.2 The learning rule runs in the wrong regime

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
conductance comparable to the network's and distorts the state.

Learning on 2×2 B&S, 4→3→4, 60 epochs, one pattern per update, fully relaxed phases (E3):

| rule | element | final MSE | exact | ceiling (oracle) |
|---|---|---|---|---|
| repo: `−1.05·ΔV²-contrast·(1−w)`, β·g_p = 1000 | ohmic | 0.087 | 0.67 | 0.0427 |
| repo | ReLU | 0.17 (no learning) | 0.33 | 0.0634 |
| EP: `−α·ΔΦ/β'`, β' = 0.1, α = 0.5 | ohmic | **0.044** | 0.67–1.00 | 0.0427 |
| EP symmetric, β' = 0.1 | ohmic | 0.045 | 0.67–1.00 | 0.0427 |
| EP, β' = 0.1 | sinh | 0.061 | 1.00 | 0.054 |

Weak-nudge EP reaches the ceiling; the repo rule does not. "Exact" oscillates between 0.67
and 1.00 at the ceiling because the ceiling's margin is 0.008 — a bit flips back and forth.

Also noted, not yet tested: the `(1 − w)` factor multiplies depression as well as
potentiation, so depression vanishes near `w = 1` and is unbounded (hard-clipped) at
`w = 0`. Weights pile up at the bounds (`w_at_bounds` ≈ 0.5 after the repo rule). Compact
memristor models use window functions vanishing at both ends (potentiation ∝ `1 − w`,
depression ∝ `w`); worth adopting.

### 3.3 Passive networks of symmetric elements cannot represent Bars & Stripes

Oracle ceilings (`symmetric_elements.py`, E1; best of 2 restarts, 1 for 3×3):

| data | n_h | ohmic MSE / exact | ReLU (V_th=0.1) | sinh (V0=0.25) |
|---|---|---|---|---|
| [1,0,1,0] | 2 | 0.0014 / 1.00 | 0.0504 / 1.00 | 0.016 / 1.00 |
| 2×2 B&S (6) | 2 | 0.084 / 0.67 | — | 0.089 / 0.67 |
| 2×2 B&S | 3 | 0.043 / 1.00, margin 0.008 | 0.063 / 1.00, margin 0.10 | 0.054 / 0.67 |
| 2×2 B&S | 4 | 0.0033 / 1.00 | — | 0.022 / 1.00 |
| 3×3 B&S (14) | 3 | 0.110 / 0.43 | — | 0.119 / 0.29 |
| 3×3 B&S | 5 | 0.067 / 0.14 | — | 0.083 / 0.29 |

Adding bias nodes (clamped at 0 and 1) and/or complementary inputs `1 − x` did **not**
help: 2×2, n_h = 3, ohmic: 0.045 / 0.044 / 0.047 (bias / dual / both); 3×3 dual+bias
n_h = 5: 0.102 / 0.43.

Why, for the ohmic case: a passive network obeys the maximum principle, so
`V_out = M·V_in` with `M` non-negative and row-stochastic, and `rank M ≤ n_h`. Exact
reconstruction of 2×2 B&S requires `M` to be the identity on the data span
`{1111, 0011, 0101}`; the only non-negative row-stochastic matrix doing that is `I` itself
(rank 4 > 3). With n_h = 4 the ceiling drops to leakage level (0.0033). More generally the
N×N B&S patterns span `2N − 1` dimensions (rows plus columns, one shared all-ones vector):
5 for 3×3, **15 for 8×8**. The proposal's 64→8→64 target is therefore below the linear rank:
it cannot be reached by anything close to linear, and positivity makes it worse still.
Symmetric nonlinearities (ReLU dead zone, sinh) did not change this picture; sinh with
`V0 = 0.25` additionally amplifies leakage at large drops.

Complementary inputs do not help a passive network because the all-zeros and all-ones
patterns force the complementary contributions to cancel. The universality construction for
resistive networks (Scellier & Mishra 2025) uses paired excitatory/inhibitory units *and*
bias sources *and* voltage-controlled voltage sources (gain) to counter attenuation;
Kendall et al. (2020) likewise use amplifiers. Without gain, signs alone buy little.

### 3.4 Rectifying antiparallel pairs: representable, partly learnable

Bars & Stripes has AND/OR structure: a pixel is on iff its row OR its column is on, and a
line detector is an AND over its pixels. Passive **rectifying** elements compute max
(OR) and min (AND) natively; symmetric elements only average.

Element (`rectifying_pairs.py`): each edge `{a,b}` is two antiparallel rectifying branches
with independent states: `K[a,b]` conducts from `b` into `a` when `V_b > V_a`, `K[b,a]` the
reverse. `K` symmetric is ohmic; one branch at `g_min` is a diode — the edge learns its own
rectification direction. Still passive; still EP-compatible (`E = Σ K·Φ₊(ΔV)`, each
branch's observable is its own co-content). Physically: self-rectifying memristors are an
established device class (see references).

- **Existence proof** (E5): a hand-built 9→6→9 network (3 row-AND + 3 column-AND
  detectors; each output = OR of its row and column detector) reconstructs **all 14**
  3×3 patterns exactly — MSE 0.062, margin 0.053 at `g_min = 0.01` (pull-up 0.2), and MSE
  0.025, margin 0.16 at `g_min = 1e-4` (pull-up 0.02). By construction it generalizes to
  unseen patterns. Compare: symmetric elements, oracle, ≤ 43% exact.
- **It is a stable, low-loss basin.** Exact MSE gradient descent started *from* the
  hand-built network keeps 100% exact and improves MSE to 0.049 and margin to 0.098.
- **But random-init optimization does not find it.** Oracle from random init, rectifying
  pairs: 2×2 n_h=3: 0.043 / 1.00 (the ohmic-like solution); 3×3 n_h=6: 0.054 / **0.29**;
  4×4 n_h=8: 0.074 / 0.57. Lower MSE than some crisp solutions, far worse exactness.
- **A local rule learns on it** (E7, weak-nudge EP with per-branch co-content, β' = 0.1,
  α = 0.5): 2×2 → 100% exact by epoch 60 (MSE 0.044); 3×3, n_h = 6 → 64% exact at epoch
  100 — better than the exact oracle from random init and than any symmetric network — but
  over 300 epochs it plateaus at 57–64% exact, MSE ≈ 0.06.
- **Generalization, preliminary** (3×3, 10 train / 4 held out = all-zeros, all-ones and two
  stripe patterns): oracle 60–80% train / 25–50% test; local EP 60–70% train / 50% test.
  Not meaningful until training reaches 100%.

Summary: rectification removes the *representational* barrier for this dataset; the
barrier that remains is *trainability* — getting from random initial states into the
AND/OR basin.

### 3.5 Observed or suspected, not yet tested

- **Online path on datasets.** The one-phase window that encodes the free/clamped contrast
  also straddles every pattern switch: at the start of pattern k's free phase the window
  still holds pattern k−1's clamped phase. Pattern-to-pattern differences in `Q` are
  typically far larger than a weak-nudge contrast. Separately, for all-zeros and all-ones
  patterns there is no error yet `Q_avg − θ ≠ 0`, so plasticity happens anyway. Hypothesis:
  the online path cannot learn datasets as is. Evidence so far is weak: mean MSE ≈ 0.46–0.50
  on 2×2 B&S with no improvement over 3 epochs, measured with the short-exposure settings
  of `test_smoke_sweep.py` (exposure 1, 20 micro-steps), not with the corrected defaults.
- **Symmetric nudging needs an active element**: the anti-nudge is a negative conductance.
- **Sign of eq. (14) of the proposal**: see section 1.

## 4. Work packages

Order matters: WP0 makes everything else cheap; WP1 makes results comparable.

### WP0 — Fast, exact solver in the library (engineering)

Goal: stop paying ~1 s per relaxation. Two solvers, both behind the current interface.

1. Vectorized explicit Euler with *identical semantics* to `VoltageDynamics.relax_transient`
   (same `dt`, `max_steps`, `tol`, divergence guard, fixed-exposure protocol): replace the
   Python double loop by dense array operations over an `(n, n)` conductance matrix.
2. `solve_equilibrium(...)`: Euler warm start + active-set Newton to the stationary state
   (port from `experiments/2026-09-24_diagnostics/symmetric_elements.py::solve`), for
   the quasi-static limit the proposal actually describes.
3. `loss_and_grad(...)`: reconstruction MSE and its exact gradient w.r.t. `w` via the
   implicit function theorem (port `loss_grad`). This is a diagnostic ("oracle"), never a
   learning rule.

The per-edge `Memristor` objects can stay as the *plasticity* state; the solver should read
a conductance matrix assembled from them (or from a vector of `w`).

Acceptance:
- `tests/test_plasticity.py` still prints `Final MSE: 0.050202` (absolute tolerance 1e-6 on
  every logged cycle) with the vectorized Euler.
- `test_smoke_sweep.py` passes unchanged.
- `solve_equilibrium` agrees with `relax_transient(max_steps=400000, tol=1e-12)` to 1e-8 on
  the 4→3→4 net (seed 42), ohmic and ReLU, free and `β·g_p = 1000`, patterns `[1,0,1,0]`,
  `[0,1,1,0]`, `[1,1,0,0]`.
- New test: gradient check of `loss_and_grad` vs finite differences, ohmic, 2×2 B&S,
  relative error < 1e-6.
- `sim.py bench` shows ≥ 20× speed-up at 16→8→16 for the Euler path.

### WP1 — Evaluation harness (engineering)

- `evaluation.py`: MSE, bit accuracy, exact-pattern fraction, margin; per-pattern table.
- Train/test splits for B&S that **exclude all-zeros and all-ones from the test set**
  (report them separately); fixed seeds; `n_seeds ≥ 3` by default.
- A small runner that takes a config (dict/YAML) and writes a JSON result with config,
  git hash, seeds, per-epoch metrics. Experiments go to `experiments/<date>_<topic>/`
  with a README (see AGENTS.md).
- `sim.py train --dataset` should shuffle pattern order per epoch (currently fixed order).

Acceptance: re-running any config with the same seed gives an identical JSON (except
timings).

### WP2 — EP-consistent learning rule (engineering + short study)

Add a rule, selectable next to the existing ones (defaults unchanged):
`Δw = −α · [Φ(ΔV^β) − Φ(ΔV^0)] / β'` with `β' = β·g_p` configurable (default 0.1), `Φ` =
the co-content of the edge's own I-V curve (provide `Φ` alongside each `f` in
`network/iv_characteristics.py`), optional symmetric variant (±β'), and a soft-bound option
(potentiation ∝ `1 − w`, depression ∝ `w`).

Acceptance:
- Unit test: cosine(EP estimate, `loss_and_grad`) ≥ 0.95 at `β' = 0.1`, ohmic, 4→3→4,
  seed 42, `[1,0,1,0]` and 2×2 B&S.
- On 2×2 B&S, ohmic, n_h = 3: final MSE ≤ 0.046 within 60 epochs for 3/3 seeds.
- Study: sweep `β' ∈ {1000, 10, 1, 0.1, 0.01}` × {repo rule, EP rule, EP + soft bounds};
  table of final MSE / exact / margin vs the oracle ceiling. Report whether soft bounds
  reduce `w_at_bounds` and whether that matters.

### WP3 — Rectifying antiparallel pairs as an edge element (engineering)

Implement the element from 3.4 in the library: a directed conductance matrix `K`, current
`F_i = Σ_j K[i,j]·r(V_j − V_i) − K[j,i]·r(V_i − V_j)`, `r(x) = max(x − V_f, 0)` with optional
forward drop `V_f`; two plastic states per undirected edge; `Φ₊(x) = ½ r(x)²` per branch.
Make it available to the solver(s), the EP rule and `sim.py` (`--edge rectpair`).

Acceptance:
- `K` symmetric with `V_f = 0` reproduces ohmic results to 1e-10.
- Gradient check < 1e-6 (rectifying pairs, `V_f = 0`, 2×2 B&S).
- The hand-built 9→6→9 network (port `handbuilt_w`) reconstructs 14/14 3×3 patterns at
  `g_min = 0.01`, pull-up 0.2.

### WP4 — Trainability on rectifying pairs (main research question)

Target: a **local** rule (WP2 on WP3) reaches 14/14 exact on 3×3 B&S in ≥ 3 of 5 seeds,
then generalizes to held-out non-trivial patterns. Baseline to beat: 57–64% exact.

Candidate levers — test one at a time against the baseline, 5 seeds each, 300 epochs:
1. Initialization: asymmetric (per edge one branch strong, the other near `g_min`, random
   direction); sparse; small-and-equal.
2. Noise/annealing: additive noise in `Δw` with decaying amplitude; learning-rate schedules.
3. Soft bounds (WP2).
4. Over-complete hidden layer (n_h = 9, 12) with a mild decay `γ` to prune.
5. Curriculum: single-line patterns first, then combinations.
6. Nudge strength and symmetric nudging (see D3).
7. Competition between hidden units: any *passive* mechanism that makes detectors
   specialize (e.g. shared-node or resistive lateral coupling); document the physics.
8. A nonlinear penalty link (the proposal allows it) that rewards margin rather than MSE.

Report for each lever: exact / MSE / margin distributions over seeds, and whether the
learned weights resemble line detectors (see WP7). Then run the held-out protocol from WP1.

### WP5 — Transport threshold and signal scale (study)

Quantify the floor of 3.1 against `V_th / V_signal` and propose operating points.
Deliverables: floor vs `V_th` for 1, 2, 3 hops; effect of starting inference at mid-rail
(0.5) instead of 0; smooth monotone alternatives (sinh with `V0 ∈ {0.25, 0.5, 1}`, tanh) and
their leakage penalty; a short recommendation. Keep the proposal's requirement (nonlinearity
must participate at inference amplitudes) explicit in the write-up.

### WP6 — Clock-free (online) learning on datasets (research)

Test the contamination hypothesis in 3.5: decompose `Δw` over an epoch into the part due to
pattern switches and the part due to nudging. Try protocols that keep the "no phase label
per element" constraint: several free/clamped cycles per pattern; a free–clamped–free
sandwich; window/`θ` time constants relative to the pattern dwell time. Compare with the
explicit EP rule from WP2 on 2×2 B&S. Acceptance for a positive result: within 10% of the
WP2 rule's final MSE on 2×2 B&S, 3 seeds.

### WP7 — Scaling and latent structure (research, after WP4)

- 4×4 B&S (30 patterns): rectifying pairs, n_h = 8 (oracle from random init: 57% exact) and
  n_h = 2N = 8 hand-built as reference.
- Latent analysis: hidden voltages per pattern; do units become row/column detectors?
  Linear readout (bars vs stripes) from hidden voltages; generative test — clamp hidden
  nodes, read outputs.
- Compression: the AND/OR construction uses 2N hidden units (16 for 8×8). A line-sharing code
  (N shared line units + 2 orientation units, output = OR(orientation AND line)) would need
  about N + 2 bottleneck nodes, but two logic levels on each side, i.e. extra layers of
  nodes. Unverified sketch; see D5.

## 5. Decisions reserved for the team

- **D1 Element.** Adopt rectifying antiparallel pairs (or another rectifying element) as the
  default edge? Everything in 3.3–3.4 says symmetric passive elements cannot do B&S.
- **D2 Gain.** Allow active elements (amplifiers/VCVS) anywhere? Required by the known
  universality construction; changes the "passive substrate" story.
- **D3 Anti-nudge.** Symmetric nudging needs a negative conductance (active). Allowed?
- **D4 Operating point.** Acceptable `V_th / V_signal`; inference from `V = 0` or mid-rail.
- **D5 Target.** 64→8→64 is below the linear rank (15) and below the 2N = 16 AND/OR
  construction. Keep 8, move to ~10–16, or add depth?
- **D6 Protocol.** Make the fully relaxed (quasi-static) equilibrium the default for physics
  experiments, keeping finite exposure as an explicit option? It changes the canary numbers.
- **D7 Proposal.** Correct the sign in eq. (14) and name the observable as the co-content.

## 6. Guardrails (full list in AGENTS.md)

Never change physics defaults silently; a refactor that claims to be physics-neutral must
reproduce `0.050202`. Tests must assert that learning reduces error, not only that values
stay finite. Keep `window_pts == micro_steps_per_phase` for the online path. Check
`result['diverged']`. Report multiple seeds and all four metrics.

## 7. Reproducing section 3

From `experiments/2026-09-24_diagnostics/`:

```bash
python3 threshold_floor.py                          # 3.1
python3 symmetric_elements.py /tmp/sym.json         # 3.2, 3.3   (~3 min)
python3 rectifying_pairs.py /tmp/rect.json          # 3.4        (~4 min)
python3 rectifying_local_ep_long.py /tmp/long.json  # 3.4, long local-EP runs
```

## 8. References

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
  learning networks: the Restricted Kirchhoff Machine*, arXiv:2509.15842 — unsupervised
  learning in resistor networks; differential conductances for signed couplings.
- *Self-rectifying memristors for beyond-CMOS computing: mechanisms, materials, and
  integration prospects*, Nano-Micro Letters (2025),
  https://link.springer.com/article/10.1007/s40820-025-02035-1.
