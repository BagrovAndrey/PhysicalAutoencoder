# Journal, track A

## State (rewrite, do not append)
- Branch / last commit: `track-a-device`; studies published at `0bc1afb`; oracle report commit follows it.
- In the queue: none; oracle_vf job 11775898 completed 15/15 with no failures, fetched. Check: `experiments/cluster.sh status A experiments/2026-10-02_oracle_vf/study.json`.
- Main open question: which devices suit the autoencoder; next quantify EP forward-drop tolerance and nudge dependence with the fixed grids.
- Next step: submit vf_fine from published 0bc1afb, then fetch/report/commit; beta_at_vf last. User performs pushes, now including the journal with other changes.

## 2026-10-02 (oracle_vf collected)
Done:        Submitted job 11775898, fetched all 15 records from 0bc1afb; wrote oracle_vf README and summary.json, including pilot comparison and convergence checks.
Learned:     measured / derived: constructive binary-representation examples exist at both nonzero vf values; attained MSE and EP gap are documented in `2026-10-02_oracle_vf/README.md`. No global-optimality claim.
Open:        Adjacent oracle entries share internal restarts; finite oracle failure is not a proof of impossibility. Nudge mechanism remains a hypothesis.
Next:        vf_fine, then beta_at_vf, unchanged parameter grids.
Do not redo: Oracle 15/15 successful, all final evaluations converged; raw records stay in LOCAL_RESULTS, aggregate committed.

## 2026-10-02 (resumed execution)
Done:        User pushed 0bc1afb; verified origin matches. User now allows journal in every combined push; user remains responsible for pushes. All studies will use published 0bc1afb; result commits remain local until final push.
Next:        Execute the three studies sequentially and report completion as requested.

## 2026-10-02 (push clarification and access failure)
Done:        User clarified: studies/reports may be pushed as needed, only journal at session end. Removed journal from the unpublished preparation commit (6292be0 replaced by 0bc1afb), preserving its contents locally until this final journal commit. Rechecked pilot status: 15/15 successful, queue empty.
Learned:     No new physics measurements. Prepared studies and pilot-identity report are ready; code/defaults unchanged.
Open:        HTTPS push failed: no Username/authentication available. SSH push to the same repository failed: Host key verification failed. No new commits reached origin; no new studies submitted. Asked user to configure GitHub access; did not bypass host verification.
Next:        With access restored, push track-a-device and submit oracle_vf first. Follow AGENTS.md: record submitted job and stop rather than wait; fetch/report/commit in collection session, then next study.
Do not redo: Grids and pilot identities already checked. Preserve oracle restart-overlap caveat and fixed beta*g_penalty products; push journal only at session end.

## 2026-10-02 (study preparation)
Done:        Prepared `2026-10-02_oracle_vf`, `2026-10-02_vf_fine`, `2026-10-02_beta_at_vf`, each with study.json and README; plans verified (15, 25, 20 runs). Checked stored pilot pattern identities and documented them in pilot README, DEVELOPMENT and SPEC §3.6.
Learned:     measured (stored pilot records): per-seed identities at vf=0.1 are in `2026-10-01_pilot/README.md`; no new simulation measurements.
Open:        Push timing subsequently clarified (entry above); publication now depends on GitHub authentication. Oracle defaults use overlapping restart pairs across adjacent seeds; disclose this in report.
Next:        Start oracle_vf once push timing is resolved. Never adapt the predefined grids to results. Check runtime/core-hours and solver warnings before subsequent studies.
Do not redo: Pattern-identity check used stored wrong vectors and confirmed dataset code identical to snapshot dd7d74377; no training replay needed.

## 2026-10-02 (orientation)
Done:        Read rules, SPEC, core code and track assignment. Initialized missing journal; attempted commit was blocked by worktree metadata permissions and escalation was interrupted, so no orientation commit exists.
Learned:     measured (existing pilot): findings and caveats remain in `2026-10-01_pilot/README.md` and SPEC §3.6.
Open:        Oracle comparison absent. Tests not rerun during orientation; no code changes.
Next:        Continue the user's three requested studies.
Do not redo: Both old pilots completed on dd7d74377: training 15/15, hand-built 12/12, no failures or queued tasks at orientation check. Do not resubmit or rename their folder.
