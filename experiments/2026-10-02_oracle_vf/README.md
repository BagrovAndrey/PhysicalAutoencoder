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

Prepared; not submitted. No measured findings yet. After fetch, record the snapshot hash, success/failure counts, report.py table (mean ± sample standard deviation), perfect-seed count and pooled wrong-pixel distribution in this file; commit summary.json. Retain convergence warnings and failed runs.

## Caveats

Only five seeds and a finite training/optimization budget. Report MSE, bit accuracy, exact-pattern fraction, normalized margin and wrong-pixel distributions. A large absolute margin may belong to confidently wrong predictions. Check final evaluation convergence; successful process exit alone is not a convergence certificate. These are training-set results including the two trivial patterns, not generalization tests. Mark interpretation as **measured**, **derived** or **hypothesis**; do not infer an unmeasured continuous tolerance threshold from the discrete grid.
