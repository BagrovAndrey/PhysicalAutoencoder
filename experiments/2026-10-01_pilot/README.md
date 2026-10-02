# Track A: forward-drop pilot

## Question

Does local equilibrium-propagation learning on rectifying pairs tolerate a
forward drop on 2×2 Bars & Stripes? Compare `vf = 0, 0.05, 0.1` on the same
4–3–4 architecture, with 60 epochs and seeds 0–4. The `vf=0` row is the ideal-pair
baseline. All other settings use the existing defaults; see `study.json`.

## Commands

From the repository root on branch `track-a-device`:

```bash
python3 experiments/run.py plan experiments/2026-10-01_pilot/study.json
experiments/cluster.sh check --compute
experiments/cluster.sh submit A experiments/2026-10-01_pilot/study.json
experiments/cluster.sh status A experiments/2026-10-01_pilot/study.json
experiments/cluster.sh log A experiments/2026-10-01_pilot/study.json
experiments/cluster.sh fetch A experiments/2026-10-01_pilot/study.json
```

`fetch` saves raw records and the report summary under
`LOCAL_RESULTS/A/2026-10-01_pilot/`. The fetched
`summary.json` is copied verbatim into this folder. The table below is the
output of `report.py` (mean ± sample standard deviation over seeds).
Raw records stay outside git.

## Run state (2026-09-30)

**measured:** Cluster checks passed on login node `cn82` and compute node `cn24`.
Submitted SLURM array job `11732123` (`mero-A-2026-10-01_pilot`), two tasks with
eight CPUs each, at most two tasks running, 30 minutes per task.
Snapshot: `dd7d74377ae29e2510124746f7c97cd414fb64fb`.

**measured:** Final status: 15 successful, 0 failed, 0 remaining; queue empty.
Mean run time: 84.0 seconds. All records use the snapshot above, have finite
final metrics, and contain 360 training cycles (60 epochs × 6 patterns).

**derived:** Sum of run wall times: 1260.6 seconds = 0.350 core-hours of
single-process elapsed work; this is not the SLURM allocation billing total.
The training path checks free/nudged `diverged` flags and returns nonzero on
an aborted phase; all 15 records have successful exit status. Final evaluation
does not separately retain solver convergence/divergence flags in these records.

## Report table and interpretation

**measured:** Final training-set metrics from `report.py`; `wrong` is the mean
number of incorrect pixels per pattern, and `perfect` counts seeds with every
pattern reconstructed exactly. Margins use normalized output voltages; here
`V_sig=1 V`. Bit decisions use threshold 0.5.

| vf | n | mse | bits | exact | margin | wrong | perfect |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.0540 ± 0.0072 | 1.000 | 1.00 | 0.038 ± 0.029 | 0.00 | 5/5 |
| 0.05 | 5 | 0.1624 ± 0.0198 | 0.717 ± 0.054 | 0.43 ± 0.15 | 0.019 ± 0.019 | 1.13 ± 0.22 | 0/5 |
| 0.1 | 5 | 0.1788 ± 0.0153 | 0.692 ± 0.037 | 0.33 | 0.046 ± 0.056 | 1.23 ± 0.15 | 0/5 |

**measured:** All five ideal-pair seeds reconstruct 6/6 patterns. Neither
nonzero-drop point produces a perfect seed at epoch 60. Exact-pattern fraction
is 0.433 ± 0.149 at `vf=0.05` and 0.333 ± 0 at `vf=0.1`.

**derived:** Mean MSE is about 3.00× and 3.31× the ideal-pair baseline at
`vf=0.05` and `vf=0.1`, respectively. These points do not preserve the
ideal-pair reconstruction performance under the tested training settings.
The larger mean margin at `vf=0.1` does not imply greater accuracy: confidently
wrong pixels also contribute to the absolute distance from 0.5.

**hypothesis:** The forward-drop dead zone may impair the learning signal.
This pilot does not isolate that mechanism or distinguish trainability from
representation; that requires the same-architecture oracle comparison.
The result bears on WP5's transport-threshold study without deciding D1–D8.

### Wrong-pixel distributions

**measured:** Counts of patterns with exactly k incorrect pixels, separately
for every seed. Each row contains six training patterns, including all-zeros
and all-ones; these are not held-out-test results.

| vf | seed | k=0 | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|---|
| 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| 0 | 1 | 6 | 0 | 0 | 0 | 0 |
| 0 | 2 | 6 | 0 | 0 | 0 | 0 |
| 0 | 3 | 6 | 0 | 0 | 0 | 0 |
| 0 | 4 | 6 | 0 | 0 | 0 | 0 |
| 0.05 | 0 | 2 | 1 | 3 | 0 | 0 |
| 0.05 | 1 | 3 | 0 | 3 | 0 | 0 |
| 0.05 | 2 | 4 | 0 | 1 | 1 | 0 |
| 0.05 | 3 | 2 | 0 | 4 | 0 | 0 |
| 0.05 | 4 | 2 | 0 | 4 | 0 | 0 |
| 0.1 | 0 | 2 | 0 | 4 | 0 | 0 |
| 0.1 | 1 | 2 | 2 | 2 | 0 | 0 |
| 0.1 | 2 | 2 | 1 | 3 | 0 | 0 |
| 0.1 | 3 | 2 | 0 | 4 | 0 | 0 |
| 0.1 | 4 | 2 | 0 | 4 | 0 | 0 |

## Caveats

- This coarse pilot covers three forward drops, not an acceptable-parameter
  window or a tuned optimum.
- Five seeds provide a limited estimate of spread. No tuning or additional
  parameter points were selected after seeing these results.
- No failed training runs were reported. Final evaluation solver flags are
  not saved separately; successful training is not a separate convergence
  certificate for every final evaluation.
- There is no held-out evaluation and no same-architecture oracle run in this
  study. The ideal-pair baseline measures degradation relative to `vf=0`, but
  cannot distinguish a representational limitation from a learning limitation.
- The existing folder name is retained as requested; the actual execution date
  from `date +%F` is 2026-09-30. No new study folder was created.

## Exact-pattern identity at vf=0.1 (checked 2026-10-02)

**measured (stored pilot records):** For every seed, the only two exact patterns are the trivial all-zeros (`0000`) and all-ones (`1111`). The four non-trivial patterns each contain errors. Read `result.final.train.wrong` from `LOCAL_RESULTS/A/2026-10-01_pilot/results.jsonl` (snapshot `dd7d74377ae29e2510124746f7c97cd414fb64fb`); no training or relaxation was rerun.

The recorded evaluation uses the lexicographic dataset order from `BarsAndStripes(2).get_all_flattened()`: `0000`, `0011`, `0101`, `1010`, `1100`, `1111`. The vector below gives wrong-pixel counts in that order; zero identifies an exact pattern.

| seed | wrong-pixel vector | exact patterns |
|---|---|---|
| 0 | 0 2 2 2 2 0 | `0000`, `1111` |
| 1 | 0 1 2 1 2 0 | `0000`, `1111` |
| 2 | 0 2 2 1 2 0 | `0000`, `1111` |
| 3 | 0 2 2 2 2 0 | `0000`, `1111` |
| 4 | 0 2 2 2 2 0 | `0000`, `1111` |

**derived:** At vf=0.1 the pilot's 2/6 exact fraction contains no exact reconstruction of a non-trivial bar or stripe (0/4 for every seed). This verifies the pattern-identity hypothesis in SPEC §3.6; it does not identify the cause or prove a representation limit.
