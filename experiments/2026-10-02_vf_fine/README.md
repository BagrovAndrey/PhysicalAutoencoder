# Track A: vf_fine

## Question

At which tested forward drop does local EP lose exact reconstruction after 60 epochs?

This is part of track A's broader device comparison: representation, local trainability and operating requirements, including tolerance to non-idealities.

## Fixed design

`study.json` fixes 25 runs before any results: the listed grid times seeds 0–4. Shared physics: 2×2 Bars & Stripes (all six patterns), 4–3–4, rectpair, plain architecture, V_sig=1, g_min=0.01, g_max=1. No code or default changes.

EP uses the pilot settings: 60 epochs, 360 updates, equilibrium phases, alpha=0.5, one-sided nudge, hard bounds, shuffled patterns. Initial weights and shuffle use seeds 0–4. No held-out split.

For training, unchanged default beta*g_penalty=0.1. Compare overlapping vf points with the pilot; no parameter selection after observing results.

## Commands

```bash
python3 experiments/run.py plan experiments/2026-10-02_vf_fine/study.json
experiments/cluster.sh submit A experiments/2026-10-02_vf_fine/study.json
experiments/cluster.sh status A experiments/2026-10-02_vf_fine/study.json
experiments/cluster.sh log A experiments/2026-10-02_vf_fine/study.json
experiments/cluster.sh fetch A experiments/2026-10-02_vf_fine/study.json
```

Submit studies sequentially: oracle_vf, vf_fine, beta_at_vf. Check runtime/core-hour estimates and failures before proceeding; no submission exceeds 500 runs or 20 estimated core-hours. Commit and push the study before submission, as required by cluster.sh. Raw results remain in LOCAL_RESULTS, outside git.

## Results and run state

**measured:** Submitted job `11775928` on 2026-10-02; snapshot `0bc1afb475ddd11465767a0565f873c14dffef0b`. All 25 planned records fetched: 25 successful, 0 failed, none missing. Every successful final evaluation reports `eval_converged=1.0`; all reported scalar metrics are finite.

**derived:** Sum of recorded run wall times: 1975.41 s = 0.549 core-hours of single-process elapsed work (not SLURM allocation billing). Mean 79.02 s per run.

**measured:** `report.py` output below; means ± sample standard deviations across the five seed entries. `perfect` counts entries with 6/6 exact patterns. Raw records: `LOCAL_RESULTS/A/2026-10-02_vf_fine/results.jsonl`; fetched aggregate is committed as `summary.json`.

| vf | n | mse | bits | exact | margin | wrong | perfect |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.0540 ± 0.0072 | 1.000 | 1.00 | 0.038 ± 0.029 | 0.00 | 5/5 |
| 0.01 | 5 | 0.0814 ± 0.0293 | 0.950 ± 0.054 | 0.87 ± 0.14 | 0.034 ± 0.032 | 0.20 ± 0.22 | 2/5 |
| 0.02 | 5 | 0.1220 ± 0.0412 | 0.808 ± 0.127 | 0.57 ± 0.22 | 0.012 ± 0.010 | 0.77 ± 0.51 | 0/5 |
| 0.03 | 5 | 0.1361 ± 0.0481 | 0.817 ± 0.124 | 0.63 ± 0.25 | 0.019 ± 0.020 | 0.73 ± 0.49 | 1/5 |
| 0.05 | 5 | 0.1624 ± 0.0198 | 0.717 ± 0.054 | 0.43 ± 0.15 | 0.019 ± 0.019 | 1.13 ± 0.22 | 0/5 |

Wrong pixels per pattern, pooled over seeds (share of patterns with k wrong):
  vf=0:  0: 100%
  vf=0.01:  0: 87%  1: 7%  2: 7%
  vf=0.02:  0: 57%  1: 17%  2: 20%  3: 7%
  vf=0.03:  0: 63%  1: 3%  2: 30%  3: 3%
  vf=0.05:  0: 43%  1: 3%  2: 50%  3: 3%

### Per-seed reconstruction

**measured:** Wrong-pixel vectors use pattern order `0000`, `0011`, `0101`, `1010`, `1100`, `1111`. Threshold is 0.5; margins refer to V_out/V_sig. These are training-set evaluations; no held-out claim.

| vf | seed | MSE | exact fraction | wrong-pixel vector |
|---|---|---|---|---|
| 0 | 0 | 0.048626 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 1 | 0.047563 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 2 | 0.053391 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 3 | 0.065703 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 4 | 0.054897 | 1.000000 | 0 0 0 0 0 0 |
| 0.01 | 0 | 0.050276 | 1.000000 | 0 0 0 0 0 0 |
| 0.01 | 1 | 0.051639 | 1.000000 | 0 0 0 0 0 0 |
| 0.01 | 2 | 0.116352 | 0.666667 | 0 0 2 0 1 0 |
| 0.01 | 3 | 0.091381 | 0.833333 | 0 0 0 0 1 0 |
| 0.01 | 4 | 0.097584 | 0.833333 | 0 0 0 2 0 0 |
| 0.02 | 0 | 0.107957 | 0.666667 | 0 1 1 0 0 0 |
| 0.02 | 1 | 0.058105 | 0.833333 | 0 0 1 0 0 0 |
| 0.02 | 2 | 0.130563 | 0.666667 | 0 0 3 0 2 0 |
| 0.02 | 3 | 0.161200 | 0.333333 | 0 3 1 2 2 0 |
| 0.02 | 4 | 0.152150 | 0.333333 | 0 2 2 2 1 0 |
| 0.03 | 0 | 0.138540 | 0.666667 | 0 0 3 1 0 0 |
| 0.03 | 1 | 0.057881 | 1.000000 | 0 0 0 0 0 0 |
| 0.03 | 2 | 0.133067 | 0.666667 | 0 0 2 0 2 0 |
| 0.03 | 3 | 0.170844 | 0.333333 | 0 2 2 2 2 0 |
| 0.03 | 4 | 0.180022 | 0.500000 | 0 2 2 2 0 0 |
| 0.05 | 0 | 0.157666 | 0.333333 | 0 2 2 2 1 0 |
| 0.05 | 1 | 0.146735 | 0.500000 | 0 0 2 2 2 0 |
| 0.05 | 2 | 0.142590 | 0.666667 | 0 0 2 0 3 0 |
| 0.05 | 3 | 0.175303 | 0.333333 | 0 2 2 2 2 0 |
| 0.05 | 4 | 0.189519 | 0.333333 | 0 2 2 2 2 0 |

## Interpretation

**measured:** All five seeds are perfect at vf=0. At the first nonzero tested point vf=0.01, only 2/5 are perfect and mean MSE is 0.0814 ± 0.0293 (versus 0.0540 ± 0.0072 at zero). Perfect-seed counts at vf=0.02, 0.03 and 0.05 are 0/5, 1/5 and 0/5; mean MSE rises to 0.1220, 0.1361 and 0.1624 respectively.

**derived:** Reliable 5/5 exact learning under this 60-epoch protocol is already lost at vf/V_sig=0.01 among the sampled points. This is not a measured universal critical vf: the exact-pattern and perfect-seed curves are not monotone, the grid is finite and only five seeds were tested. In particular the success at 0.03 must not be omitted to claim a hard cutoff at 0.02. Imperfect final reconstruction also does not by itself mean zero learning.

**measured:** The ten runs at vf=0 and 0.05 reproduce the earlier pilot exactly: maximal absolute difference in final weights and MSE is zero, and all wrong-pixel vectors match. This checks comparability across the pilot and current snapshots without using those outcomes to change the grid.

**measured / derived:** Same-architecture oracle references from `../2026-10-02_oracle_vf/`: at vf=0, MSE 0.04267 and mean exact=1.0 versus EP MSE 0.05402, exact=1.0; at vf=0.05, oracle MSE 0.06352, exact=0.8667 versus EP MSE 0.16244, exact=0.4333. An oracle entry is fully exact at vf=0.05, establishing binary representability there and a large EP trainability gap. No oracle was requested at vf=0.01, 0.02 or 0.03, so their errors cannot be partitioned into representation and optimization effects from this study alone.

**hypothesis:** Dead-zone effects on local training remain a candidate mechanism. The beta_at_vf study tests whether increasing the nudge rescues final reconstruction at fixed vf=0.05; these data alone do not measure gradient alignment or device noise tolerance.


## Caveats

Only five seeds and a finite training/optimization budget. Report MSE, bit accuracy, exact-pattern fraction, normalized margin and wrong-pixel distributions. A large absolute margin may belong to confidently wrong predictions. Check final evaluation convergence; successful process exit alone is not a convergence certificate. These are training-set results including the two trivial patterns, not generalization tests. Mark interpretation as **measured**, **derived** or **hypothesis**; do not infer an unmeasured continuous tolerance threshold from the discrete grid.
