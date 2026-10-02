# Track A: beta_at_vf

## Question

At vf=0.05, does stronger nudging recover learning or impair it?

This is part of track A's broader device comparison: representation, local trainability and operating requirements, including tolerance to non-idealities.

## Fixed design

`study.json` fixes 20 runs before any results: the listed grid times seeds 0–4. Shared physics: 2×2 Bars & Stripes (all six patterns), 4–3–4, rectpair, plain architecture, V_sig=1, g_min=0.01, g_max=1. No code or default changes.

EP uses the pilot settings: 60 epochs, 360 updates, equilibrium phases, alpha=0.5, one-sided nudge, hard bounds, shuffled patterns. Initial weights and shuffle use seeds 0–4. No held-out split.

Explicit g_penalty=10 and beta={0.01,0.03,0.1,0.3} give beta*g_penalty={0.1,0.3,1,3}. **hypothesis:** the dead zone hides a weak nudge, while a stronger nudge biases the gradient estimate. Reconstruction metrics alone cannot establish gradient alignment; no alignment measurement is added to this study.

## Commands

```bash
python3 experiments/run.py plan experiments/2026-10-02_beta_at_vf/study.json
experiments/cluster.sh submit A experiments/2026-10-02_beta_at_vf/study.json
experiments/cluster.sh status A experiments/2026-10-02_beta_at_vf/study.json
experiments/cluster.sh log A experiments/2026-10-02_beta_at_vf/study.json
experiments/cluster.sh fetch A experiments/2026-10-02_beta_at_vf/study.json
```

Submit studies sequentially: oracle_vf, vf_fine, beta_at_vf. Check runtime/core-hour estimates and failures before proceeding; no submission exceeds 500 runs or 20 estimated core-hours. Commit and push the study before submission, as required by cluster.sh. Raw results remain in LOCAL_RESULTS, outside git.

## Results and run state

**measured:** Submitted job `11776403` on 2026-10-02; snapshot `0bc1afb475ddd11465767a0565f873c14dffef0b`. All 20 planned records fetched: 20 successful, 0 failed, none missing. Every successful final evaluation reports `eval_converged=1.0`; all reported scalar metrics are finite.

**derived:** Sum of recorded run wall times: 1796.34 s = 0.499 core-hours of single-process elapsed work (not SLURM allocation billing). Mean 89.82 s per run.

**measured:** `report.py` output below; means ± sample standard deviations across the five seed entries. `perfect` counts entries with 6/6 exact patterns. Raw records: `LOCAL_RESULTS/A/2026-10-02_beta_at_vf/results.jsonl`; fetched aggregate is committed as `summary.json`.

| beta | n | mse | bits | exact | margin | wrong | perfect |
|---|---|---|---|---|---|---|---|
| 0.01 | 5 | 0.1624 ± 0.0198 | 0.717 ± 0.054 | 0.43 ± 0.15 | 0.019 ± 0.019 | 1.13 ± 0.22 | 0/5 |
| 0.03 | 5 | 0.1758 ± 0.0306 | 0.717 ± 0.075 | 0.47 ± 0.14 | 0.040 ± 0.045 | 1.13 ± 0.30 | 0/5 |
| 0.1 | 5 | 0.1853 ± 0.0282 | 0.708 ± 0.072 | 0.47 ± 0.14 | 0.062 ± 0.072 | 1.17 ± 0.29 | 0/5 |
| 0.3 | 5 | 0.1771 ± 0.0177 | 0.683 ± 0.037 | 0.40 ± 0.09 | 0.058 ± 0.059 | 1.27 ± 0.15 | 0/5 |

Wrong pixels per pattern, pooled over seeds (share of patterns with k wrong):
  beta=0.01:  0: 43%  1: 3%  2: 50%  3: 3%
  beta=0.03:  0: 47%  2: 50%  4: 3%
  beta=0.1:  0: 47%  2: 47%  3: 3%  4: 3%
  beta=0.3:  0: 40%  1: 7%  2: 47%  4: 7%

### Per-seed reconstruction

**measured:** Wrong-pixel vectors use pattern order `0000`, `0011`, `0101`, `1010`, `1100`, `1111`. Threshold is 0.5; margins refer to V_out/V_sig. These are training-set evaluations; no held-out claim.

| beta | seed | MSE | exact fraction | wrong-pixel vector |
|---|---|---|---|---|
| 0.01 | 0 | 0.157666 | 0.333333 | 0 2 2 2 1 0 |
| 0.01 | 1 | 0.146735 | 0.500000 | 0 0 2 2 2 0 |
| 0.01 | 2 | 0.142590 | 0.666667 | 0 0 2 0 3 0 |
| 0.01 | 3 | 0.175303 | 0.333333 | 0 2 2 2 2 0 |
| 0.01 | 4 | 0.189519 | 0.333333 | 0 2 2 2 2 0 |
| 0.03 | 0 | 0.131536 | 0.666667 | 0 0 2 2 0 0 |
| 0.03 | 1 | 0.168694 | 0.333333 | 0 2 2 2 2 0 |
| 0.03 | 2 | 0.180971 | 0.500000 | 0 0 2 2 4 0 |
| 0.03 | 3 | 0.180879 | 0.500000 | 0 2 2 0 2 0 |
| 0.03 | 4 | 0.216797 | 0.333333 | 0 2 2 2 2 0 |
| 0.1 | 0 | 0.152245 | 0.666667 | 0 0 2 2 0 0 |
| 0.1 | 1 | 0.171298 | 0.333333 | 0 2 2 2 2 0 |
| 0.1 | 2 | 0.190761 | 0.500000 | 0 0 2 2 4 0 |
| 0.1 | 3 | 0.183532 | 0.500000 | 0 2 3 0 2 0 |
| 0.1 | 4 | 0.228459 | 0.333333 | 0 2 2 2 2 0 |
| 0.3 | 0 | 0.154947 | 0.333333 | 0 2 1 2 1 0 |
| 0.3 | 1 | 0.171802 | 0.333333 | 0 2 2 2 2 0 |
| 0.3 | 2 | 0.178989 | 0.500000 | 0 0 2 2 4 0 |
| 0.3 | 3 | 0.175806 | 0.500000 | 0 2 4 0 2 0 |
| 0.3 | 4 | 0.204119 | 0.333333 | 0 2 2 2 2 0 |

## Interpretation

**measured:** The `beta` column in report.py is the CLI parameter; with fixed g_penalty=10 its effective nudge is ten times larger. At products 0.1, 0.3, 1 and 3, mean MSE is 0.1624 ± 0.0198, 0.1758 ± 0.0306, 0.1853 ± 0.0282 and 0.1771 ± 0.0177. No setting yields a perfect seed (0/5 at every point). Mean exact fractions are 0.4333, 0.4667, 0.4667 and 0.4000.

**measured / derived:** Matched-architecture comparison with the vf=0.05 oracle (attained mean MSE 0.063518 ± 0.006626, exact 0.8667 ± 0.0745, one perfect entry):

| CLI beta | beta*g_penalty | mean EP MSE | mean MSE minus oracle | mean exact | perfect seeds |
|---|---|---|---|---|---|
| 0.01 | 0.1 | 0.162362 | 0.098845 | 0.433333 | 0/5 |
| 0.03 | 0.3 | 0.175775 | 0.112258 | 0.466667 | 0/5 |
| 0.1 | 1 | 0.185259 | 0.121741 | 0.466667 | 0/5 |
| 0.3 | 3 | 0.177133 | 0.113615 | 0.400000 | 0/5 |

**derived:** Raising the nudge within this fixed grid does not rescue exact reconstruction and does not close the large MSE gap to the oracle. All larger nudges have worse sample-mean MSE than 0.1. The small improvement in mean exact fraction at 0.3 and 1 coexists with worse MSE; there is no monotone deterioration of every metric, and five seeds do not establish a universal optimum or statistical significance.

**measured:** The five effective-nudge=0.1 runs exactly reproduce the corresponding vf_fine baseline in final weights and every saved final metric. All 20 training histories have 360 updates; no failed runs were discarded or hyperparameters changed after observing outcomes.

**hypothesis assessment:** The proposed dead-zone/gradient-bias mechanism remains unverified. This study tests its possible training consequence, and does not support a simple rescue by stronger nudge at vf=0.05 under these settings. It did not measure per-edge activation changes or alignment with `loss_and_grad`, so the worse MSE cannot by itself demonstrate gradient-estimator bias. The finite training budget and clipping also remain possible contributors. No follow-up gradient or device-model experiments were silently added.


## Caveats

Only five seeds and a finite training/optimization budget. Report MSE, bit accuracy, exact-pattern fraction, normalized margin and wrong-pixel distributions. A large absolute margin may belong to confidently wrong predictions. Check final evaluation convergence; successful process exit alone is not a convergence certificate. These are training-set results including the two trivial patterns, not generalization tests. Mark interpretation as **measured**, **derived** or **hypothesis**; do not infer an unmeasured continuous tolerance threshold from the discrete grid.

Final convergence is checked from saved evaluation flags; intermediate relaxation states and all of their convergence flags are not archived in these records. For training runs the rule aborts on free/nudged divergence, but a successful exit and final convergence do not certify convergence of every intermediate gradient-estimation phase. No claim about unrecorded solver trajectories is made.
