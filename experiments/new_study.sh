#!/usr/bin/env bash
# experiments/new_study.sh TOPIC [COMMAND]
#
# Creates experiments/<today>_<TOPIC>/ with a skeleton study.json and README.md.
# The date comes from the machine this runs on (`date +%F`) - never type it by hand,
# and do not rename the folder after submitting: results on the cluster are stored
# under the folder name.
#
#   experiments/new_study.sh vf_fine            # a `train` study
#   experiments/new_study.sh oracle_vf oracle   # train | oracle | handbuilt
set -euo pipefail

[ "$#" -ge 1 ] || { sed -n '3,12p' "$0"; exit 1; }
topic=$1
command=${2:-train}
[[ "$topic" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "TOPIC: letters, digits, - and _ only" >&2; exit 1; }
case "$command" in train|oracle|handbuilt) ;; *) echo "COMMAND must be train, oracle or handbuilt" >&2; exit 1;; esac

root="$(git rev-parse --show-toplevel)"
dir="$root/experiments/$(date +%F)_${topic}"
[ ! -e "$dir" ] || { echo "$dir already exists" >&2; exit 1; }
mkdir -p "$dir"

case "$command" in
  train)     base='{"rule": "ep", "element": "rectpair", "dataset": "bars-stripes", "n-input": 4, "n-hidden": 3, "epochs": 60, "protocol": "equilibrium"}'
             grid='{"vf": [0, 0.05]}'; seeds='"seeds": 5,';;
  oracle)    base='{"element": "rectpair", "dataset": "bars-stripes", "n-input": 4, "n-hidden": 3}'
             grid='{"vf": [0, 0.05]}'; seeds='"seeds": 5,';;
  handbuilt) base='{"element": "rectpair"}'
             grid='{"vf": [0, 0.05]}'; seeds='';;
esac

cat > "$dir/study.json" <<JSON
{
  "question": "TODO: one sentence, what this study asks",
  "command": "$command",
  "base": $base,
  "grid": $grid,
  $seeds
  "slurm": {"partition": "tcm", "time": "01:00:00", "cpus": 8, "max_parallel": 2}
}
JSON

cat > "$dir/README.md" <<MD
# TODO: title

## Question

TODO

## Commands

\`\`\`bash
python3 experiments/run.py plan experiments/$(basename "$dir")/study.json
experiments/cluster.sh submit A experiments/$(basename "$dir")/study.json   # track label: A, B, ...
experiments/cluster.sh status A experiments/$(basename "$dir")/study.json
experiments/cluster.sh fetch  A experiments/$(basename "$dir")/study.json
\`\`\`

## Results

TODO: table from report.py; state the commit hash of the runs.

## Reading

TODO: each claim marked **measured** / **derived** / **hypothesis**.

## Caveats

TODO
MD

echo "created $dir"
echo "next: edit study.json, commit, push, then cluster.sh submit"
