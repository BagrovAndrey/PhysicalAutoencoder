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
| `shockley_pairs.py` | The hand-built 3×3 network with realistic branches: Shockley diode in series with the programmable filament. No hard threshold; shows the trade-off rectification ratio R ↔ turn-on drop ≈ n·(kT/q)·ln R, and which signal amplitudes work. | ~3 min |
| `rectifying_local_ep_long.py OUT.json` | Local EP on rectifying pairs, 3×3 B&S, 300 epochs; all 14 patterns, and 10 train / 4 held out. | ~2–5 min |

Run from this directory, e.g. `python3 symmetric_elements.py /tmp/sym.json`. Committed
results: `results_symmetric.json`, `results_rectifying.json`, `results_long_ep.json`
(raw weights stripped).

## Playground: `play.py`

Command-line access to all of the above, so network size, element and parameters are
flags rather than constants. Run from this directory; every mode takes `--help`.

```bash
# existence proof: hand-built AND/OR network for N x N Bars & Stripes
python3 play.py handbuilt --data bs3 --show               # 14/14, prints every reconstruction
python3 play.py handbuilt --data bs3 --vf 0.1             # ideal pairs with a forward drop
python3 play.py handbuilt --data bs3 --element shockley --Is 1e-4 --n 1 --vsig 1
python3 play.py handbuilt --data bs4 --pullup 0.1 --gmin 0.001

# ceiling: the best any conductance setting can reach (exact gradients, not physical)
python3 play.py oracle --data bs3 --n-hidden 6 --element rectpair
python3 play.py oracle --data bs2 --n-hidden 3 --element tanh --steepness 2
python3 play.py oracle --data bs2 --n-hidden 3 --element ohmic --arch dual+bias

# the physical local rule: weak-nudge EP, each branch learns from its own co-content
python3 play.py train --data bs2 --n-hidden 3 --epochs 60
python3 play.py train --data bs3 --n-hidden 6 --epochs 200 --holdout 3 --show
python3 play.py train --data bs2 --n-hidden 3 --element tanh --steepness 2 --epochs 40
```

Elements: `rectpair` (default; ideal antiparallel rectifying pair, optional `--vf`),
`shockley` (each branch = Shockley diode in series with the filament; `--Is`, `--n`,
`--vsig`), `ohmic`, `relu` (`--vth`), `tanh` (`--steepness`), `sinh` (`--v0`). All six
work in `oracle` and `train`; `handbuilt` takes `rectpair` or `shockley`.
Data: `bs2`…`bs5` or an explicit pattern such as `1,0,1,0`. `--holdout k` holds out k
non-trivial patterns (never all-zeros/all-ones) as a test set.

Things found while building it (2026-09-29):
- **4x4 needs a better on/off ratio.** The hand-built 16-8-16 network reaches only 73%
  at `g_min = 0.01` whatever the pull-up, and 100% at `g_min = 0.001`: each node has twice
  the fan-in of the 3x3 case, so the summed leakage grows with network size. The required
  `g_max/g_min` scales with fan-in.
- **Saturation blocks local learning.** With `tanh(10 V)` edges the oracle ceiling on 2x2
  is good (0.033) but the local rule does not learn at any learning rate tried: at 1 V
  signals most edges are saturated and the nudge cannot propagate through them. With
  `tanh(2 V)` the same rule reaches 100% (MSE 0.040 vs ceiling 0.038). The saturation
  scale must be comparable to the signal, the physical counterpart of vanishing gradients.
- **Shockley branches need a much weaker nudge.** The EP observable of a branch is
  `dPhi/dg = (I/g)^2/2`, half the squared voltage across its filament (derived and checked
  against numerical integration; the oracle gradient passes a finite-difference check to
  2e-8). But the one-sided EP estimate only aligns with the true gradient once the nudge
  shifts node voltages by << kT/q: cosine -0.04 at beta*g_p = 1e-2 (60 mV shifts), 0.18
  at 1e-3, 0.95 at 1e-4 (1 mV), 1.00 below. Symmetric nudging already gives 0.996 at 1e-3
  (10 mV). Defaults in `play.py train --element shockley`: 1e-4, or 1e-3 with `--sym`.
  The ideal pair only has curvature at its kink, which is why beta = 0.1 works there.
  Open question: millivolt nudges are comparable to kT/C noise on femtofarad nodes.
- **Local learning on Shockley pairs is slow and untuned.** 2x2, 1 V, `--sym --alpha 10`:
  MSE 0.167 -> 0.126, 67% exact after 40 epochs (oracle ceiling 0.091, 100% exact with
  zero margin). At 3 V it gets worse with the same settings. ~2 s per epoch.

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
