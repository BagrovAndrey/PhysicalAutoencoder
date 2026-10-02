# Journal, track A

## State (rewrite, do not append)
- Branch / last commit: `track-a-device`; final documentation/journal commit at HEAD, parent `453d6f0` (alignment script/report). Local diagnostic used library revision `0a2fb40`.
- In the queue: none for previous studies; beta_at_vf job 11776403 verified complete (20/20). Check: `experiments/cluster.sh status A experiments/2026-10-02_beta_at_vf/study.json`.
- Main open question: whether disagreement at forward-drop kinks reflects EP error or limitations of the selected-active-set implicit derivative; numbers in ep_alignment README.
- Next step: user pushes the diagnostic/report commit, then supplies the next task. No scheduled work; user pushes manually.

## 2026-10-02 (current session)
Done:        Created ep_alignment/alignment.py and README; ran 20 trajectories locally (fixed training nudge, three probes on epochs 0/30/60), saved compact summary.json and ignored full results/results.json in report commit 453d6f0. Updated DEVELOPMENT and SPEC §3.6. Corrected stale branch state and consolidated prior session notes under current journal rules.
Learned:     measured / derived: cosine versus the library gradient decreases with vf in the declared comparisons; refutation criterion not met. Full table, spread, norms and kink caveats in `2026-10-02_ep_alignment/README.md`.
Open:        Implicit oracle at nonsmooth/singular points is not an independently validated physical gradient. The alignment trend does not establish the cause of stalled learning; no finite-difference test at kinks was requested.
Next:        User push and next research task; no additional runs scheduled.
Do not redo: All 180 probes valid and converged; source hashes, aggregation, unchanged diagnostic weights and training endpoints checked. Library/defaults unchanged. Raw vectors/weights remain in ignored results/; statistics are in summary.json. No cluster job needed.

## 2026-10-02 (previous session)
Done:        Completed oracle_vf, vf_fine, beta_at_vf on snapshot 0bc1afb: jobs 11775898, 11775928, 11776403. Report commits 90549a7, 0d4e986, aa64626; findings/journal 4a448c3. User pushed results.
Learned:     measured / derived: fully exact oracle examples exist at nonzero vf; fine-grid exact success is not monotone and stronger nudges do not rescue EP in the fixed grid. Numbers/spread in respective READMEs, summary in SPEC §3.6.
Open:        Dead-zone masking versus gradient bias not directly measured; finite oracle is not a global bound and adjacent oracle seeds share internal restarts.
Next:        New local alignment diagnostic, current session above.
Do not redo: All 60 records fetched and verified; final evaluations converged and overlapping baselines match exactly. Raw records remain in LOCAL_RESULTS. Pilot vf=0.1 exact-pattern identities are in its README. User pushes manually; no authentication/configuration work needed.
