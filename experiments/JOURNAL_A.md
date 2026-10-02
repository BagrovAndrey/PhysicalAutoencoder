# Journal, track A

## State (rewrite, do not append)
- Branch / last commit: `track-a-device`; final findings/journal commit at HEAD, parent `aa64626`. Earlier report commits: `90549a7` (oracle), `0d4e986` (fine grid). Published simulation snapshot: `0bc1afb`.
- In the queue: none, verified 2026-10-02. Completed/fetched jobs: oracle_vf 11775898 (15/15), vf_fine 11775928 (25/25), beta_at_vf 11776403 (20/20), no failures. Check: `experiments/cluster.sh status A experiments/2026-10-02_oracle_vf/study.json` (repeat for vf_fine and beta_at_vf).
- Main open question: which devices best suit the autoencoder, including representation, local trainability, operating scales and non-idealities. Gradient-alignment explanation for the forward-drop failure remains unmeasured.
- Next step: user pushes the four local report/findings commits, including journal, then supplies the next research task. The current three-study prompt is complete; no further runs scheduled.

## 2026-10-02 (three studies completed)
Done:        Collected all 60 planned runs from 0bc1afb; study folders `2026-10-02_oracle_vf`, `2026-10-02_vf_fine`, `2026-10-02_beta_at_vf` contain fixed study.json, README and fetched summary.json. Updated DEVELOPMENT and SPEC §3.6 separately from experiment reports. Pilot README identifies exact patterns at vf=0.1 per seed.
Learned:     measured / derived: same-architecture oracle constructs fully exact solutions at nonzero drops; EP has a substantial trainability gap. Details and limits in oracle_vf README.
Learned:     measured / derived: fine-grid exact success is not monotone; stronger nudges in the fixed grid do not rescue exact reconstruction. Numbers and spread in respective study READMEs.
Open:        Dead-zone masking and strong-nudge gradient bias remain hypotheses; these studies did not measure alignment. Oracle is an attained reference, not a proven global bound; adjacent seed entries share internal restarts.
Next:        Final user push and next task. User handles pushes; journal now travels with all other changes on each push. Workspace actions and cluster submissions are authorized without repeated confirmation.
Do not redo: All final evaluations converged; all 60 records, metrics, hashes, summaries and seed/config coverage verified. Overlapping pilot and nudge-baseline runs match weights and metrics exactly. Code/defaults unchanged; no code tests needed for documentation-only changes. Raw records remain in LOCAL_RESULTS, not git.

## 2026-10-02 (execution)
Done:        Ran studies sequentially; fetched/reported/committed each before submitting the next. Temporarily detached at published 0bc1afb for later submissions, then returned to track-a-device with reports intact; avoided intermediate pushes.
Learned:     No additional physics claims beyond the study READMEs.
Open:        None blocking the requested work. Interrupted beta_at_vf submission was checked: nothing had been submitted; then job 11776403 was submitted once.
Next:        All requested work subsequently completed (entry above).
Do not redo: Parameters fixed in advance; no post-result tuning, no extra simulations, no library changes.

## 2026-10-02 (preparation and push clarification)
Done:        Prepared all three grids and checked stored pilot wrong-pixel vectors. Pilot clarification commit 950dda9; study preparation 0bc1afb (replaces unpublished 6292be0 after separating the journal). User published 0bc1afb.
Learned:     measured: pilot vf=0.1 exact patterns are recorded per seed in `2026-10-01_pilot/README.md`; dataset ordering unchanged from dd7d74377.
Open:        Agent HTTPS push lacked authentication; SSH failed host-key verification. User chose to perform pushes. Later changed preference to include journal with every push.
Next:        Superseded by completed execution above.
Do not redo: Do not repeat authentication work or bypass host verification; user handles push. No need to separate journal commits for publication anymore.

## 2026-10-02 (orientation)
Done:        Read rules, SPEC, core code and track assignment; initialized missing journal. Old pilot statuses verified: training 15/15, hand-built 12/12 complete on dd7d74377.
Learned:     measured (existing pilot): findings and caveats in `2026-10-01_pilot/README.md` and SPEC §3.6.
Open:        Historical oracle question resolved constructively in the new study; device-choice scope is broader than non-ideality tolerance.
Next:        Await next task after publishing results.
Do not redo: Do not resubmit or rename completed pilot folders. Orientation commit attempt was interrupted; no separate orientation commit exists.
