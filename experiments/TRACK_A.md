# Track A — realistic memristor properties

Assignment for a coding agent. Working rules: [`AGENTS.md`](../AGENTS.md) (read it, in
particular "Working on the cluster"); background: [`docs/SPEC.md`](../docs/SPEC.md) §3.4, §3.6
and WP5. Branch: `track-a-device`, worktree `../mero-A`.

## Question

Which non-idealities of a self-rectifying memristor pair can the AND/OR solution and the
local learning rule tolerate? The output is a **window of acceptable device parameters**,
not a single best setting, so that Vladimir and Anya can compare it with datasheets and
measurements of real devices.

## Scope

Phase 1 — parameters that already exist in `sim.py` (no library changes needed):

| knob | flag | what it stands for |
|---|---|---|
| forward drop | `--vf` (with `--vsig`) | turn-on voltage of the diode; only `V_f / V_sig` matters |
| on/off ratio | `--g-min` (with `--g-max`) | leakage of the "off" branch |
| Shockley diode | `--element shockley --Is --n --vsig` | rectification ratio vs turn-on drop |
| signal amplitude | `--vsig` | with Shockley: the operating point on the exponential |
| nudge strength | `--beta` (product `beta*g_penalty`) | EP needs it weak on the element's voltage scale |

Phase 2 — after the device-imperfection patch (device-to-device spread, finite number of
conductance levels, write noise, asymmetric potentiation/depression) lands on `main`. Do
not implement these yourself; ask.

Questions to answer, in this order (each: one study folder, see below):

1. **Hand-built network** (`handbuilt`, deterministic, seconds per run): for which
   `(V_f/V_sig, g_min)` and for which `(I_s, n, V_sig)` does the 3×3 AND/OR network still
   reconstruct 14/14, and how does the margin shrink? Then 4×4 (`--n-input 16`).
2. **Local EP training** (`train --rule ep`, 5 seeds): how much of the ideal-pair
   performance survives the same parameters on 2×2 (60 epochs) and 3×3 (`--n-hidden 6`,
   200–300 epochs)? Compare with the ideal pair run in the same study (`vf=0`).
3. **Where the ceiling moves**: for the parameter points where training fails, is the
   oracle (`oracle`, same flags) also bad (representation) or fine (trainability)?

## Deliverables

For every study, a folder `experiments/<YYYY-MM-DD>_<topic>/` containing

- `study.json` (or several `*.json`) — the grid; committed and pushed **before** submitting;
- `README.md` — question, how to run (the `cluster.sh` commands), the table from
  `report.py`, a short reading of it, caveats. Mark each claim **measured** / **derived** /
  **hypothesis**. No PNG/GIF files;
- `summary.json` — from `cluster.sh fetch`.

At the end, a dated entry in `DEVELOPMENT.md` and, if a finding changes what we believe, an
edit to the relevant part of `docs/SPEC.md` (§3.6 or WP5) — as a separate commit.

## Limits (ask before exceeding)

- One submission: at most 500 runs and about 20 core-hours (`run.py status` prints an
  estimate after the first runs). Start every new study with 1 seed or a coarse grid.
- Do not edit `network/`, `training/` or `sim.py` on this branch. Need a new flag or
  element? Write down what and why, and stop.
- Do not change any default. Do not decide D1–D8 of SPEC §5; report which decision the data
  bears on.
- At least 3 seeds for anything with a learning rule (5 for the headline table); report the
  spread and the wrong-pixel distribution, not the best seed. Never tune until a number
  looks good.

## Start here

`experiments/2026-10-01_pilot/` contains two small studies for checking the whole chain
(`handbuilt.json`: 12 runs, `study.json`: 15 runs). Run them first:

```bash
experiments/cluster.sh check --compute
experiments/cluster.sh submit A experiments/2026-10-01_pilot/handbuilt.json
experiments/cluster.sh status A experiments/2026-10-01_pilot/handbuilt.json
experiments/cluster.sh fetch  A experiments/2026-10-01_pilot/handbuilt.json
```
