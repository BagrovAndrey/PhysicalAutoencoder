# Track A: oracle_vf

## Question

Does the same-architecture oracle degrade with the forward-drop training pilot, or does optimization retain better reconstruction?

This is part of track A's broader device comparison: representation, local trainability and operating requirements, including tolerance to non-idealities.

## Fixed design

`study.json` fixes 15 runs before any results: the listed grid times seeds 0–4. Shared physics: 2×2 Bars & Stripes (all six patterns), 4–3–4, rectpair, plain architecture, V_sig=1, g_min=0.01, g_max=1. No code or default changes.

Oracle uses unchanged CLI defaults: 450 Adam iterations, learning rate 0.03, two restarts (seed s and s+1). Adjacent seed entries therefore share a restart; spread is descriptive, not five independent restart pairs. Oracle performance is an attained optimization reference, not a proven global bound.

Compare against `../2026-10-01_pilot/README.md` at matching vf, architecture and conductance bounds. Good oracle reconstruction alongside poor EP performance supports a trainability limitation. Poor oracle reconstruction alone does not prove a representation limit because optimization may miss a better basin.

## Commands

```bash
python3 experiments/run.py plan experiments/2026-10-02_oracle_vf/study.json
experiments/cluster.sh submit A experiments/2026-10-02_oracle_vf/study.json
experiments/cluster.sh status A experiments/2026-10-02_oracle_vf/study.json
experiments/cluster.sh log A experiments/2026-10-02_oracle_vf/study.json
experiments/cluster.sh fetch A experiments/2026-10-02_oracle_vf/study.json
```

Submit studies sequentially: oracle_vf, vf_fine, beta_at_vf. Check runtime/core-hour estimates and failures before proceeding; no submission exceeds 500 runs or 20 estimated core-hours. Commit and push the study before submission, as required by cluster.sh. Raw results remain in LOCAL_RESULTS, outside git.

## Results and run state

**measured:** Submitted job `11775898` on 2026-10-02; snapshot `0bc1afb475ddd11465767a0565f873c14dffef0b`. All 15 planned records fetched: 15 successful, 0 failed, none missing. Every successful final evaluation reports `eval_converged=1.0`; all reported scalar metrics are finite.

**derived:** Sum of recorded run wall times: 974.36 s = 0.271 core-hours of single-process elapsed work (not SLURM allocation billing). Mean 64.96 s per run.

**measured:** `report.py` output below; means ± sample standard deviations across the five seed entries. `perfect` counts entries with 6/6 exact patterns. Raw records: `LOCAL_RESULTS/A/2026-10-02_oracle_vf/results.jsonl`; fetched aggregate is committed as `summary.json`.

| vf | n | mse | bits | exact | margin | wrong | perfect |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.0427 ± 0.0000 | 1.000 | 1.00 | 0.008 ± 0.000 | 0.00 | 5/5 |
| 0.05 | 5 | 0.0635 ± 0.0066 | 0.933 ± 0.037 | 0.87 ± 0.07 | 0.024 ± 0.017 | 0.27 ± 0.15 | 1/5 |
| 0.1 | 5 | 0.0860 ± 0.0180 | 0.933 ± 0.070 | 0.87 ± 0.14 | 0.097 ± 0.003 | 0.27 ± 0.28 | 2/5 |

Wrong pixels per pattern, pooled over seeds (share of patterns with k wrong):
  vf=0:  0: 100%
  vf=0.05:  0: 87%  2: 13%
  vf=0.1:  0: 87%  2: 13%

### Per-seed reconstruction

**measured:** Wrong-pixel vectors use pattern order `0000`, `0011`, `0101`, `1010`, `1100`, `1111`. Threshold is 0.5; margins refer to V_out/V_sig. These are training-set evaluations; no held-out claim.

| vf | seed | MSE | exact fraction | wrong-pixel vector |
|---|---|---|---|---|
| 0 | 0 | 0.042670 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 1 | 0.042670 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 2 | 0.042670 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 3 | 0.042670 | 1.000000 | 0 0 0 0 0 0 |
| 0 | 4 | 0.042670 | 1.000000 | 0 0 0 0 0 0 |
| 0.05 | 0 | 0.062312 | 0.833333 | 0 2 0 0 0 0 |
| 0.05 | 1 | 0.062312 | 0.833333 | 0 2 0 0 0 0 |
| 0.05 | 2 | 0.069656 | 0.833333 | 0 0 2 0 0 0 |
| 0.05 | 3 | 0.069656 | 0.833333 | 0 0 2 0 0 0 |
| 0.05 | 4 | 0.053652 | 1.000000 | 0 0 0 0 0 0 |
| 0.1 | 0 | 0.091319 | 0.833333 | 0 2 0 0 0 0 |
| 0.1 | 1 | 0.110630 | 0.666667 | 0 0 2 2 0 0 |
| 0.1 | 2 | 0.068287 | 1.000000 | 0 0 0 0 0 0 |
| 0.1 | 3 | 0.068287 | 1.000000 | 0 0 0 0 0 0 |
| 0.1 | 4 | 0.091310 | 0.833333 | 0 0 0 2 0 0 |

## Interpretation

**measured:** Oracle mean MSE rises from 0.04267 at vf=0 to 0.06352 at 0.05 and 0.08597 at 0.1. It nevertheless reaches 6/6 exact patterns at each tested forward drop: selected oracle entry seed 4 at vf=0.05, and entries seeds 2 and 3 at vf=0.1. The latter two share the same successful internal restart (initialization seed 3), so they are not two independent successes.

**measured / derived:** Comparison with the pilot (mean metrics below, difference derived from the two means; spread for each study is reported in its own table):

| vf | oracle MSE | pilot EP MSE | EP minus oracle | oracle exact | EP exact |
|---|---|---|---|---|---|
| 0 | 0.042670 | 0.054036 | 0.011366 | 1.000000 | 1.000000 |
| 0.05 | 0.063518 | 0.162362 | 0.098845 | 0.866667 | 0.433333 |
| 0.1 | 0.085966 | 0.178793 | 0.092827 | 0.866667 | 0.333333 |

**derived:** The tested 4–3–4 architecture can represent all six patterns after thresholding at both nonzero drops: the successful weights are constructive examples. The EP-to-oracle MSE gap is about 0.099 at vf=0.05 and 0.093 at vf=0.1, versus 0.011 at vf=0. A large trainability gap therefore remains; the pilot collapse is not solely a loss of binary representability.

**caveat:** The attained oracle MSE also worsens with forward drop, but this finite optimizer does not prove a global analog-error floor or quantify a purely representational contribution. Loss kinks, local basins, finite iterations and MSE-versus-exact tradeoffs remain. No statement that stronger vf monotonically improves exact accuracy follows from the observed 1/5 versus 2/5 perfect entries.

**hypothesis:** A forward-drop dead zone may weaken the EP training signal; the fixed vf_fine and beta_at_vf studies test outcomes, not the mechanism directly. This bears on operating point D4 without selecting a default.


## Caveats

Only five seeds and a finite training/optimization budget. Report MSE, bit accuracy, exact-pattern fraction, normalized margin and wrong-pixel distributions. A large absolute margin may belong to confidently wrong predictions. Check final evaluation convergence; successful process exit alone is not a convergence certificate. These are training-set results including the two trivial patterns, not generalization tests. Mark interpretation as **measured**, **derived** or **hypothesis**; do not infer an unmeasured continuous tolerance threshold from the discrete grid.
