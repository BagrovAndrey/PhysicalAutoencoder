# experiments/

Informal, one-off exploration scripts - **not** part of the maintained test
suite (see `../tests/` for that, and `../DEVELOPMENT.md` for current
pass/fail status). They exist to record the empirical basis for design
decisions written up in `DEVELOPMENT.md`, in particular the "Correction to
the 'Design Decisions' section" and "Global vs local theta" sections.

All three were written and run against `network/dynamics.py`'s **old**
array/`iv_function` API, before the 2026-09-14 fix that made it
Grid/Memristor-based (see `DEVELOPMENT.md`, "History: the broken merge").
They are kept as-is, unrun since the fix, as a record of the exact numbers
now cited in `DEVELOPMENT.md`. To run one again, either check out the old
`network/dynamics.py` from git history (commit `d456d0f`), or port its
solver setup to `Grid` + `network.builders.build_memristor_array`, the same
way `tests/test_plasticity.py` was ported.

- `plateau_baseline.py` - reproduces the 4→3→4 plateau with the explicit
  contrastive rule (quadratic Q), and checks whether switching to a literal
  linear Q (as an earlier draft of DEVELOPMENT.md claimed) helps. It doesn't
  - it makes the reconstruction worse.
- `online_rule_check.py` - tests a fully local, clock-free online rule
  driven by raw `Q_avg` with no reference value. Every weight collapses to
  `g_min` - a sign-definite quadratic observable can only ever push weights
  one way, so a two-sided (potentiate/depress) signal needs *something* to
  compare against.
- `bcm_threshold_check.py` - tests a BCM-style fix: a per-edge, purely
  local, slowly-adapting theta. It works (recovers to MSE ≈ 0.041), but
  the project went with a single *global* theta instead (see
  `DEVELOPMENT.md`, "Global vs local theta") - this script is kept as the
  record of why the local-per-edge alternative was considered and what it
  would take to revisit it later.
