# experiments/

Exploration scripts - **not** part of the maintained test suite (see `../tests/`). They
record the empirical basis for claims in `../DEVELOPMENT.md` and `../docs/SPEC.md`. New
studies go in dated subfolders, `<YYYY-MM-DD>_<topic>/`, each with its own README and
results JSON (see `../AGENTS.md`).

## Running studies (batch runner, cluster)

Grids of `sim.py` runs are described by a JSON study file and executed by `run.py`
(parallel, resumable, one result file per run); `report.py` aggregates seeds;
`cluster.sh` submits to SLURM over ssh from your own machine. See `../AGENTS.md`
("Working on the cluster"), the header of `cluster.sh`, and the first assignment,
[`TRACK_A.md`](TRACK_A.md). Small pilot studies to test the chain: `2026-10-01_pilot/`.

```bash
python3 experiments/run.py plan   experiments/2026-10-01_pilot/study.json    # what would run
python3 experiments/run.py run    experiments/2026-10-01_pilot/study.json --out /tmp/pilot   # here, all cores
python3 experiments/run.py pack   experiments/2026-10-01_pilot/study.json --out /tmp/pilot
python3 experiments/report.py /tmp/pilot/results.jsonl
```

## `2026-09-24_diagnostics/` — current

Why learning stalls: the transport-threshold floor, the nudge-strength regime of the
learning rule, the representational ceiling of passive symmetric networks, rectifying
memristor pairs and Shockley elements. Self-contained (own fast solver, cross-checked
against `network/dynamics.py` to ~1e-9) and runnable as is. Backs `docs/SPEC.md` §3. For
interactive exploration use `sim.py` (its README maps the old `play.py` commands).

## Top-level scripts (2026-09-14) — historical

Written and run against `network/dynamics.py`'s **old** array/`iv_function` API, before
the 2026-09-14 fix that made it Grid/Memristor-based (see `DEVELOPMENT.md`, "History: the
broken merge"). Kept as-is, unrun since the fix, as the record of the numbers cited in
`DEVELOPMENT.md`. To run one again, check out the old `network/dynamics.py` (commit
`d456d0f`) or port its solver setup to `Grid` + `network.builders.build_memristor_array`.

- `plateau_baseline.py` - reproduces the 4→3→4 plateau with the explicit contrastive rule
  (quadratic Q) and checks whether a literal linear Q (as an earlier draft of
  DEVELOPMENT.md claimed) helps. It doesn't — reconstruction gets worse.
- `online_rule_check.py` - a fully local, clock-free online rule driven by raw `Q_avg` with
  no reference value. Every weight collapses to `g_min`: a sign-definite observable can only
  push weights one way, so a two-sided signal needs something to compare against.
- `bcm_threshold_check.py` - the BCM-style fix: a per-edge, slowly adapting theta. Recovers
  to MSE ≈ 0.041; the project chose a single *global* theta instead (DEVELOPMENT.md,
  "Global vs local theta").
