# AGENTS.md — working rules for coding agents

MeroCircuit simulates a memristive-network autoencoder trained by local,
equilibrium-propagation-style learning. **What to work on and why: `docs/SPEC.md`.**
History and design decisions: `DEVELOPMENT.md`. Usage: `README.md`.

## Setup and checks

```bash
pip install -r requirements.txt && pip install -e .
python3 tests/test_smoke_sweep.py   # must end "64 passed, 0 failed" (~3 min)
python3 tests/test_plasticity.py    # must print "Final MSE: 0.050202"
```

Test files are plain scripts (`python3 tests/<file>.py`), except
`tests/test_bars_stripes.py` (pytest). `tests/test_visualization.py` needs an interactive
matplotlib backend and may fail headless; that is not a regression.

## Hard rules

1. **Do not change physics defaults silently.** That includes I-V curves, `g_min/g_max`,
   `V_th`, `beta`, `g_penalty`, `dt`, exposure/window settings, the plasticity rule and its
   sign. New behaviour goes behind a flag or a new function with the old default intact.
2. **Physics-neutral refactors must reproduce the canaries**: `Final MSE: 0.050202` (every
   logged cycle within 1e-6 absolute) and the full smoke sweep.
3. **Decisions D1–D7 in `docs/SPEC.md` §5 belong to the team.** Implement options, measure,
   report; do not pick one by changing a default.
4. **A test that only checks "finite and bounded" is not a learning test.** Any test of a
   learning rule must assert that reconstruction error goes down. (A non-learning default
   once shipped under 60 green checks for exactly this reason.)
5. **Online path:** each memristor's `window_pts` must equal `micro_steps_per_phase`
   (`Trainer` warns otherwise). Changing the ratio silently stops learning.
6. **Solver stability:** explicit Euler diverges above `dt ≈ 0.002` at `β·g_p = 1000`.
   Check `result['diverged']`; never report numbers from a diverged run.
7. **Report honestly.** Negative results are results. Never tune until a number looks
   good and report only that number.

## Reporting experiments

- Put each study in `experiments/<YYYY-MM-DD>_<topic>/` with a `README.md` (question,
  how to run, findings, caveats) and a JSON of results. Include config, seeds and the git
  hash. Do not commit PNG/GIF outputs.
- Always report all four metrics: MSE, bit accuracy (threshold 0.5), exact-pattern fraction,
  margin (min |V_out − 0.5|). Use ≥ 3 seeds; give spread, not just the best seed.
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
- Build `Grid` + memristor arrays via `network/builders.py`; do not hand-roll wiring.
- Research-code style: short docstrings that say *why*, parameters as function arguments,
  no hidden globals in library code.
- Units are dimensionless model units throughout.
- Commit messages: imperative subject; body explains the reason and quotes the numbers
  that justify the change.
