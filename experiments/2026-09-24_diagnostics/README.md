# 2026-09-24 diagnostics: learning failure or physics limit?

Scripts behind the diagnosis in [`docs/SPEC.md`](../../docs/SPEC.md), section 3. They are
**research code, not library code**: each uses its own small, fast solver instead of
`network/dynamics.py`, so it can run hundreds of optimizations in minutes.

**The solver matches the repo's physics.** Euler gradient flow from `V_free = 0` (the same
dynamics as `VoltageDynamics`), followed by active-set Newton polishing. Cross-checked
against `VoltageDynamics.relax_transient` (fully relaxed) on the 4→3→4 network with the
repo's initial weights, for ohmic and thresholded-ReLU edges, free and clamped
(β·g_p = 1000): maximum voltage difference ≈ 1e-9.

**Gradients are exact.** `loss_grad` differentiates the reconstruction MSE through the
equilibrium by the implicit function theorem; checked against finite differences
(relative error ~1e-9).

**"Oracle" ≠ learning rule.** The oracle is projected Adam on `w ∈ [0,1]` using those
exact gradients. It is not physical; it estimates the best reconstruction that *any*
conductance setting of a given architecture can reach, so that a learning rule can be
judged against a ceiling instead of against zero.

## Scripts

| script | what it answers | runtime (2 cores) |
|---|---|---|
| `threshold_floor.py` | Why the single-pattern plateau is MSE ≈ 0.03–0.05: the ReLU transport threshold costs V_th per hop. | ~30 s |
| `symmetric_elements.py OUT.json` | E1: ceilings for ohmic / ReLU (V_th=0.1) / sinh edges on `[1,0,1,0]`, 2×2 and 3×3 B&S, with bias nodes and complementary inputs. E2: alignment of the EP estimate with the true gradient vs nudge strength. E3: repo rule vs weak-nudge EP on 2×2 B&S. | ~3 min |
| `rectifying_pairs.py OUT.json` | Edges as antiparallel pairs of rectifying memristors. E4 ceilings, E5 hand-built AND/OR network (existence proof), E6 held-out generalization, E7 local EP on 2×2 and 3×3 B&S. | ~4 min |
| `rectifying_local_ep_long.py OUT.json` | Local EP on rectifying pairs, 3×3 B&S, 300 epochs; all 14 patterns, and 10 train / 4 held out. | ~2–5 min |

Run from this directory, e.g. `python3 symmetric_elements.py /tmp/sym.json`. Committed
results: `results_symmetric.json`, `results_rectifying.json`, `results_long_ep.json`
(raw weights stripped).

## Conventions inside these scripts

- `w ∈ [0,1]`, `g = g_min + (g_max − g_min)·w`, `g_min = 0.01`, `g_max = 1`, as in the repo.
- Inference always starts from `V_free = 0`, as in the repo. With a dead-zone I-V this
  matters: the relaxed state depends on where the flow started.
- Metrics: MSE; bit accuracy after thresholding at 0.5; exact-pattern fraction (all
  pixels right); margin = min |V_out − 0.5| (how close the nearest bit is to flipping).
- `beta_gp` is the product β·g_p (the only combination that enters the physics).
- Bars & Stripes are generated here directly (`bars_stripes(N)`, sorted); patterns 0 and
  last are all-zeros and all-ones.

## Caveat

For the dead-zone ReLU, gradient flow parks many edges exactly at |ΔV| = V_th, where the
I-V curve has a kink, so the loss is only piecewise smooth at the operating point. The
gradient check passes with this solver, but an earlier implementation with a different
warm-start trajectory failed it. ReLU ceilings are therefore obtained by optimizing a
smoothed surrogate and evaluating with the exact curve; treat them as estimates.
