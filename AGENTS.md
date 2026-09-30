# AGENTS.md — working rules for coding agents

MeroCircuit simulates a memristive-network autoencoder trained by local,
equilibrium-propagation-style learning. **What to work on and why: `docs/SPEC.md`.**
History and design decisions: `DEVELOPMENT.md`. Usage: `README.md`.

## Setup and checks

```bash
pip install -r requirements.txt && pip install -e .
python3 tests/test_smoke_sweep.py   # must end "66 passed, 0 failed" (~3 min)
python3 tests/test_plasticity.py    # must print "Final MSE: 0.050202"
python3 tests/test_engine.py        # must end "58 passed, 0 failed" (~30 s)
python3 tests/test_runner.py        # must end "33 passed, 0 failed" (~10 s)
python3 sim.py train                # must end at MSE 0.043978
python3 sim.py train --rule global-theta --cycles 25   # must end at MSE 0.032565
```

Two engines: `sim.py` runs on the vectorized engine (`network/elements.py`,
`network/equilibrium.py`, `training/learning.py`); the object engine (`Grid`, `Memristor`,
`VoltageDynamics`, `Trainer`) is kept for the Python API and the legacy canaries. New
physics goes into the vectorized engine and `sim.py` (one CLI; do not add parallel
playground scripts). A new element needs `current_and_slope`, `dI_dg` and `dPhi_dg`, and
entries in `tests/test_engine.py` section 2.

Test files are plain scripts (`python3 tests/<file>.py`), except
`tests/test_bars_stripes.py` (pytest). `tests/test_visualization.py` needs an interactive
matplotlib backend and may fail headless; that is not a regression.

## Hard rules

1. **Do not change physics defaults silently.** That includes I-V curves, `g_min/g_max`,
   `V_th`, `beta`, `g_penalty`, `dt`, exposure/window settings, the default element and
protocol of each rule, the plasticity rule and its
   sign. New behaviour goes behind a flag or a new function with the old default intact.
2. **Physics-neutral refactors must reproduce the canaries**: `Final MSE: 0.050202` (every
   logged cycle within 1e-6 absolute), `0.043978` / `0.032565` from `sim.py train`, the full
   smoke sweep and `tests/test_engine.py`.
3. **Decisions D1–D8 in `docs/SPEC.md` §5 belong to the team.** Implement options, measure,
   report; do not pick one by changing a default.
4. **A test that only checks "finite and bounded" is not a learning test.** Any test of a
   learning rule must assert that reconstruction error goes down. (A non-learning default
   once shipped under 60 green checks for exactly this reason.)
5. **Online path (`global-theta`):** keep `window_pts == micro_steps_per_phase` (`Trainer`
   warns otherwise); other ratios were measured not to learn. The mechanism is disputed
   (`docs/SPEC.md` 3.5): do not describe it as a device constraint.
6. **Solver stability:** explicit Euler diverges above `dt ≈ 0.002` at `β·g_p = 1000`.
   Check `diverged`; never report numbers from a diverged run.
7. **EP needs a weak nudge on the element's own voltage scale**: `β·g_p ≈ 0.1` for
   ideal elements, `≤ 1e-4` (or `1e-3` symmetric) for Shockley branches (shifts ≪ kT/q).
   Check alignment with `loss_and_grad` before trusting a new element.
8. **Report honestly.** Negative results are results. Never tune until a number looks
   good and report only that number.

## Working on the cluster

Runs go through `experiments/cluster.sh` (local machine -> ssh -> SLURM); read its header.
The agent lives on the local machine and does not need anything installed on the cluster.

- **One track = one branch = one git worktree** (`git worktree add ../mero-A -b track-a-...`).
  The cluster script takes the track label as an argument, so parallel tracks never share a
  directory, a job name or a results folder.
- **Only `cluster.sh`.** Do not run ad-hoc `ssh` commands. It does: `check`, `submit`,
  `status`, `log`, `fetch`, `cancel`. Nothing heavy runs on the login node; the agent only
  submits, waits and reads.
- **A study is a committed, pushed JSON file** (`experiments/<date>_<topic>/study.json`,
  format in `experiments/run.py`). `submit` runs a snapshot of that commit, so queued jobs
  never see later edits and every result carries its commit hash. Never submit from a dirty
  tree; `--refresh-code` replaces a snapshot only when no job of the study is queued.
- **Start small.** `run.py plan STUDY` shows the runs; submit one seed or a coarse grid first;
  check `status` (it estimates core-hours) before the full grid. Resubmitting the same study
  runs only what is missing or failed.
- **Results stay out of git and out of the home directory** (5 GB quota): they live under
  `REMOTE_RUNS` on the cluster and in `LOCAL_RESULTS` locally. Commit only `study.json`,
  `README.md` and `summary.json`.
- **Waiting is not work.** After `submit`, write down the state (which job, which study, what
  comes next) in the study README or your notes, and stop; collect with `fetch` later.
- A failed run is recorded, not fatal; look at `status` and `log` before resubmitting, and
  report failures in the README instead of silently dropping them.
- The imperfect-device parameters, the model and the library are changed by dedicated patches
  reviewed by the maintainer, not from an experiment branch.

## Reporting experiments

- Put each study in `experiments/<YYYY-MM-DD>_<topic>/` with a `README.md` (question,
  how to run, findings, caveats) and a JSON of results. Include config, seeds and the git
  hash. Do not commit PNG/GIF outputs.
- Always report MSE, bit accuracy (threshold 0.5), exact-pattern fraction, margin
  (min |V_out − 0.5|) and the distribution of wrong pixels per pattern (`sim.py` prints
  it; `--json` saves it). Use ≥ 3 seeds; give spread, not just the best seed.
- For Bars & Stripes held-out tests, exclude the all-zeros and all-ones patterns from the
  test set and report them separately.
- Compare a learning rule against the oracle ceiling for the same architecture
  (`docs/SPEC.md` §3), not only against the initial error.
- When a finding changes what we believe, add a dated entry to `DEVELOPMENT.md` and update
  the relevant section of `docs/SPEC.md`.

## Code conventions

- Library code lives in the package directories (`network/`, `memristor/`, `grid/`,
  `training/`, `datasets/`, `visualization/`); experiment scripts do not get imported by
  library code.
- Build networks via `network.equilibrium.autoencoder` (engine) or `network/builders.py`
  (object engine); do not hand-roll wiring.
- Research-code style: short docstrings that say *why*, parameters as function arguments,
  no hidden globals in library code.
- Units are dimensionless model units, except that voltages are volts where an element
  has a physical voltage scale (`shockley`: `V_T = 25.85 mV`; `--vsig`).
- Commit messages: imperative subject; body explains the reason and quotes the numbers
  that justify the change.
