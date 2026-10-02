# Track A: EP alignment at forward-drop kinks

## Question and fixed protocol

**hypothesis:** the cosine between one-sided EP and the library's implicit MSE gradient falls as rectpair forward drop grows. The requested refutation criterion is cosine >0.9 at every vf; both seed minima and group means will be reported, including the vf=0 control.

Architecture 4–3–4, plain rectpair, all six 2×2 Bars & Stripes patterns, seeds 0–4; vf={0,0.02,0.05,0.1}, probe beta*g_penalty={0.1,0.01,0.001}, epochs={0,30,60}. Training stays at beta*g_penalty=0.1, alpha=0.5, shuffled pattern order, equilibrium protocol, hard bounds. This is one 60-epoch trajectory per (vf,seed), not a separate training run per probe. Snapshots after 180 and 360 updates are copied by train's callback, preserving the same shuffled trajectory. Initialization matches sim.py (uniform w in [0.1,0.9]); g_min=0.01, g_max=1, V_sig=1.

## Commands

```bash
python3 experiments/2026-10-02_ep_alignment/alignment.py --workers 4 \
  --json experiments/2026-10-02_ep_alignment/results/results.json
```

Run locally. Prints mean ± sample standard deviation across seeds. The command saves full results under the ignored results/ folder; summary.json is the committed compact record. Without --json the script writes results.json beside itself, which should be kept out of tracked artifacts. Only library builders, elements, equilibrium, loss_and_grad, EPRule and train are used for the physics and learning; library code is unchanged.

## Observable and validity

For each fixed snapshot, average EPRule's raw estimator [dPhi_dg(nudged)-dPhi_dg(free)]/(beta*g_penalty) over six patterns, and compare its direction with net.loss_and_grad(w,X). The cosine is between gradient estimates, not between a weight update and a gradient: the negative descent sign, alpha and clipping are excluded. Weights are checked unchanged after each diagnostic. For gradient norms and relative error, convert the raw estimate to the normalized MSE gradient w.r.t. w by the positive factor 2*(g_max-g_min)/(n_output*V_sig^2); this does not change cosine.

All diagnostic free/nudged states must converge and not diverge; the checked free MSE must agree with loss_and_grad. Failed trajectories are retained with errors and make the complete-grid criterion invalid, never silently discarded. Zero-norm vectors yield an undefined cosine, not an artificial perfect alignment. JSON retains gradients, estimates, norms, weights, reconstruction metrics, per-pattern kink/active-set counts, configuration and source hashes.

## Results

**measured (2026-10-02):** Local execution took 276.3 s with four workers. All 20 training trajectories and 180 probe evaluations completed, with 36 groups of five seed cosines. All diagnostic free/nudged phases converged without divergence; maximum recorded residual 9.977e-13. No training free-phase non-convergence occurred in 7,200 updates. Gradient norms were nonzero (minimum across snapshots 0.003725); no cosine was undefined. This does not certify unrecorded nudged training-phase convergence.

Library revision: `0a2fb40aaf6eeac770495948bd678844e22f983f`; script and library source SHA-256 hashes are retained in JSON. The diagnostic ran before its experiment commit; content hashes identify its exact sources. Full vectors/weights live in `results/results.json` (ignored by git); compact per-seed results, metrics, configuration and hashes are committed as `summary.json`. Library code and training settings are unchanged.

**measured:** Cosine of dataset-mean EP and implicit MSE gradients, mean ± sample standard deviation across five seeds:

| epoch | beta*g_penalty | vf=0 | vf=0.02 | vf=0.05 | vf=0.1 |
|---|---|---|---|---|---|
| 0 | 0.1 | 0.9379 ± 0.0387 | 0.7373 ± 0.0614 | 0.4832 ± 0.2236 | -0.0269 ± 0.4741 |
| 0 | 0.01 | 0.9984 ± 0.0020 | 0.8780 ± 0.2423 | 0.4777 ± 0.3640 | -0.0833 ± 0.4169 |
| 0 | 0.001 | 1.0000 ± 0.0000 | 0.8948 ± 0.2350 | 0.5075 ± 0.3376 | -0.1021 ± 0.3930 |
| 30 | 0.1 | 0.9200 ± 0.0929 | 0.2312 ± 0.4370 | 0.0808 ± 0.2516 | 0.0352 ± 0.0427 |
| 30 | 0.01 | 0.9904 ± 0.0127 | 0.5473 ± 0.4681 | 0.0907 ± 0.4388 | -0.2023 ± 0.2060 |
| 30 | 0.001 | 1.0000 ± 0.0000 | 0.6602 ± 0.4592 | 0.2355 ± 0.6547 | -0.3906 ± 0.2506 |
| 60 | 0.1 | 0.9828 ± 0.0204 | 0.5462 ± 0.4566 | 0.3469 ± 0.4556 | 0.1395 ± 0.1866 |
| 60 | 0.01 | 0.9991 ± 0.0017 | 0.8084 ± 0.1732 | 0.3909 ± 0.4904 | 0.1380 ± 0.2756 |
| 60 | 0.001 | 1.0000 ± 0.0000 | 0.9714 ± 0.0477 | 0.4358 ± 0.6348 | 0.1828 ± 0.6217 |

### Hypothesis and criterion

**derived:** Neither all group means nor all per-seed cosines stay >0.9. The declared refutation criterion is not met: 26 of 36 group means are ≤0.9. At epoch 60 and probe=0.001, means across vf are 1.0000, 0.9714, 0.4358 and 0.1828. The vf=0.02 mean there exceeds 0.9, but its minimum seed cosine is 0.8885, so the stricter seed-level criterion still fails.

**measured / derived:** Each of the nine fixed (epoch,probe) combinations has a strictly decreasing sample mean across vf=0,0.02,0.05,0.1. These observations support the proposed trend on this grid, not for every seed or device. At vf=0, probes 0.01 and 0.001 are well aligned at every epoch, while probe 0.1 has some seeds below 0.9 even in the control. At vf=0.1, epoch 30 and probe=0.001, all five cosines are negative (mean -0.3906 ± 0.2506). Decreasing nudge therefore does not universally restore alignment.

**measured:** Every seed at initial vf=0.05 and 0.1 has non-trivial-pattern branch drops within 1e-8 of the kink: respectively 4–8 and 12–16 branch occurrences across the four non-trivial patterns. All vf=0 controls have none at any snapshot. Several learned vf=0.02 snapshots also contain kink branches. Nonzero norms and tiny residuals do not make a selected-active-set derivative valid across a kink.

**hypothesis, still open:** This drop may involve EP bias, active-set changes or failure of the nonsmooth/singular implicit-gradient reference to describe the selected relaxation state. No finite-difference validation of that reference at kinks was requested or added. Measured disagreement with net.loss_and_grad does not prove that EP points against the true physical descent direction. Aggregate cosine does not determine each single-pattern or clipped update's direction.

### Reconstruction during the fixed trajectories

**measured:** Mean ± sample standard deviation; wrong-pixel counts pool 30 patterns (six × five seeds) per row. All 20 final endpoints agree with matching prior fine-grid or pilot runs to floating-point precision (maximum absolute weight difference 9.88e-15, MSE difference 4.44e-16). This independently checks training settings and snapshot capture.

| epoch | vf | MSE | bit accuracy | exact fraction | margin | wrong-pixel counts k=0..4 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.1707 ± 0.0019 | 0.6583 ± 0.0186 | 0.3333 ± 0.0000 | 0.0195 ± 0.0166 | 10, 0, 19, 1, 0 |
| 0 | 0.02 | 0.1708 ± 0.0018 | 0.6667 ± 0.0295 | 0.3333 ± 0.0000 | 0.0212 ± 0.0184 | 10, 1, 18, 1, 0 |
| 0 | 0.05 | 0.1716 ± 0.0022 | 0.6667 ± 0.0000 | 0.3333 ± 0.0000 | 0.0247 ± 0.0204 | 10, 0, 20, 0, 0 |
| 0 | 0.1 | 0.1776 ± 0.0036 | 0.6667 ± 0.0000 | 0.3333 ± 0.0000 | 0.0287 ± 0.0263 | 10, 0, 20, 0, 0 |
| 30 | 0 | 0.1100 ± 0.0290 | 0.8917 ± 0.0864 | 0.6667 ± 0.2357 | 0.0043 ± 0.0041 | 20, 8, 1, 1, 0 |
| 30 | 0.02 | 0.1538 ± 0.0258 | 0.7167 ± 0.0745 | 0.3667 ± 0.0745 | 0.0152 ± 0.0153 | 11, 4, 15, 0, 0 |
| 30 | 0.05 | 0.1683 ± 0.0088 | 0.6750 ± 0.0456 | 0.3333 ± 0.0000 | 0.0371 ± 0.0343 | 10, 3, 16, 0, 1 |
| 30 | 0.1 | 0.1790 ± 0.0028 | 0.6667 ± 0.0000 | 0.3333 ± 0.0000 | 0.0614 ± 0.0368 | 10, 0, 20, 0, 0 |
| 60 | 0 | 0.0540 ± 0.0072 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0380 ± 0.0288 | 30, 0, 0, 0, 0 |
| 60 | 0.02 | 0.1220 ± 0.0412 | 0.8083 ± 0.1271 | 0.5667 ± 0.2236 | 0.0116 ± 0.0097 | 17, 5, 6, 2, 0 |
| 60 | 0.05 | 0.1624 ± 0.0198 | 0.7167 ± 0.0543 | 0.4333 ± 0.1491 | 0.0190 ± 0.0186 | 13, 1, 15, 1, 0 |
| 60 | 0.1 | 0.1788 ± 0.0153 | 0.6917 ± 0.0373 | 0.3333 ± 0.0000 | 0.0456 ± 0.0555 | 10, 3, 17, 0, 0 |

## Validation

Independently compared the estimator with EPRule.update before clipping on a copied initial state; agreement at floating-point precision. Recomputed all 180 saved cosines from the saved vectors; checked record counts, seed coverage, source hashes, endpoints and aggregation. Weights remain unchanged during diagnostics. No library refactor or new learning rule; the full canary suite was not rerun.


## Caveats

The oracle is the library implicit-function derivative on the solver's selected active set, not an independent ground-truth gradient at a kink. Rectpair has slope zero at DeltaV=vf; near those boundaries the loss may not be differentiable, the Jacobian may be singular, and least-squares derivatives can depend on the selected branch. Small beta may change active sets. Kink counts are diagnostics, not a proof that a low cosine is a physical learning failure.

This is cosine of the dataset-mean gradients. Pattern-specific errors can cancel, and a high aggregate cosine does not prove alignment for every single-pattern update. Trivial all-zero/all-one patterns are included in the training dataset. Reconstruction metrics are reported per snapshot with no held-out interpretation. At small gradient norms, cosine is sensitive to residual solver error; norms and relative errors are saved. The measured criterion applies only to the declared seeds, weights and nudges; a failure to refute does not by itself prove monotonic dependence on vf or a causal explanation for stalled training.
