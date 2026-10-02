#!/usr/bin/env bash
# experiments/cluster.sh
#
# The only way an agent (or you) talks to the cluster. Runs on your LOCAL machine, reaches
# the cluster over ssh, and does exactly these things:
#
#   cluster.sh check   [--compute]                   can we reach it, is python/SLURM/disk OK
#   cluster.sh submit  TRACK STUDY [--refresh-code] [--dry-run]
#   cluster.sh status  TRACK STUDY                   done / failed / queued
#   cluster.sh log     TRACK STUDY                   tail of the newest job log
#   cluster.sh fetch   TRACK STUDY                   results.jsonl -> LOCAL_RESULTS, then report
#   cluster.sh cancel  TRACK STUDY
#
# TRACK  a short label for the line of work (A, B, ...): keeps parallel tracks apart.
# STUDY  path of a study JSON inside this repo, e.g. experiments/2026-10-01_device/study.json.
#        It must be committed and pushed: the cluster runs a SNAPSHOT of that commit
#        (git archive), so jobs waiting in the queue never see later edits, and every
#        result carries the commit it was computed with.
#
# Layout on the cluster (REMOTE_RUNS):
#   .repo.git/                     mirror of the GitHub repo (only git objects)
#   <TRACK>/<study>/code/          snapshot of the commit, read-only in spirit
#   <TRACK>/<study>/results/       one JSON per run;  results.jsonl after `fetch`
#   <TRACK>/<study>/logs/ submit.sh jobs.txt
#
# Settings: experiments/cluster.conf, overridden by ~/.config/mero/cluster.conf
# (or $MERO_CLUSTER_CONF). ssh details (host name, user, jump host, ControlMaster) belong
# in ~/.ssh/config under the alias CLUSTER_HOST.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
CLUSTER_HOST=mero
REMOTE_RUNS=
LOCAL_RESULTS="$HOME/mero-results"
PARTITION=
PYTHON=python3
REPO_URL="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
# shellcheck disable=SC1090
[ -f "$ROOT/experiments/cluster.conf" ] && . "$ROOT/experiments/cluster.conf"
USER_CONF="${MERO_CLUSTER_CONF:-$HOME/.config/mero/cluster.conf}"
# shellcheck disable=SC1090
[ -f "$USER_CONF" ] && . "$USER_CONF"

die() { echo "cluster.sh: $*" >&2; exit 1; }
[ -n "$REMOTE_RUNS" ] || die "REMOTE_RUNS is not set (experiments/cluster.conf)"

remote() {  # remote SCRIPT-ON-STDIN ARGS... : run a bash script on the cluster, args as $1 $2 ...
  local q; q="$(printf '%q ' "$@")"        # ssh joins argv into one string: quote each argument
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$CLUSTER_HOST" "bash -s -- $q"
}

need_args() { [ "$#" -ge 2 ] || die "usage: cluster.sh CMD TRACK STUDY"; }
check_track() { [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || die "TRACK must be letters, digits, - or _"; }

study_name() {  # experiments/2026-10-01_device/study.json -> 2026-10-01_device
  local dir stem
  dir="$(basename "$(dirname "$1")")"; stem="$(basename "$1" .json)"
  if [ "$stem" = study ]; then echo "$dir"; else echo "${dir}_${stem}"; fi
}

RUN_DIR=; JOB=; STUDY_REL=
resolve() {  # resolve TRACK STUDYFILE -> RUN_DIR JOB STUDY_REL
  check_track "$1"
  local f abs
  abs="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
  case "$abs" in "$ROOT"/*) STUDY_REL="${abs#"$ROOT"/}";; *) die "study file must be inside the repository";; esac
  local name; name="$(study_name "$STUDY_REL")"
  RUN_DIR="$REMOTE_RUNS/$1/$name"
  JOB="mero-$1-$name"
}

cmd_check() {
  local compute=0; [ "${1:-}" = "--compute" ] && compute=1
  echo "== ssh $CLUSTER_HOST"
  remote "$REMOTE_RUNS" "$PARTITION" "$PYTHON" "$compute" <<'EOF'
runs=$1; part=$2; py=$3; compute=$4
echo "host: $(hostname)"
$py -c "import sys, numpy, scipy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)" \
  || echo "!! python check failed ($py)"
command -v git >/dev/null && echo "git: $(git --version)" || echo "!! git missing"
command -v sbatch >/dev/null && echo "sbatch: $(command -v sbatch)" || echo "!! sbatch not in PATH here"
mkdir -p "$runs" && touch "$runs/.w" && rm "$runs/.w" && echo "storage: $runs writable" || echo "!! cannot write $runs"
df -h "$runs" | tail -1
if [ -n "$part" ] && command -v sinfo >/dev/null; then
  sinfo -h -p "$part" -o "partition $part: %l limit, %c cpus, %m MB, %D nodes" | head -2
fi
if [ "$compute" = 1 ]; then
  echo "-- on a compute node:"
  # --immediate: give up after 60 s if no node is free instead of waiting in the queue
  srun --immediate=60 ${part:+-p "$part"} -t 2 -c 1 bash -c "hostname; $py -c 'import numpy, scipy; print(\"numpy/scipy ok\")'; ls -ld '$runs'" \
    || echo "!! no compute node within 60 s (partition busy?) - this says nothing about the setup"
fi
EOF
}

cmd_submit() {
  need_args "$@"; local track=$1 study=$2; shift 2
  local refresh=0 dry=0
  for a in "$@"; do case "$a" in --refresh-code) refresh=1;; --dry-run) dry=1;; *) die "unknown option $a";; esac; done
  resolve "$track" "$study"
  cd "$ROOT"
  git ls-files --error-unmatch "$STUDY_REL" >/dev/null 2>&1 || die "$STUDY_REL is not tracked: commit it first"
  git diff --quiet HEAD -- "$STUDY_REL" || die "$STUDY_REL has uncommitted changes: commit first"
  local hash; hash="$(git rev-parse HEAD)"
  git fetch -q origin
  [ -n "$(git branch -r --contains "$hash" 2>/dev/null)" ] || die "commit ${hash:0:9} is not on origin: git push first"
  [ -n "$REPO_URL" ] || die "REPO_URL unknown (no origin remote)"
  echo "submitting $STUDY_REL from commit ${hash:0:9} as track $track -> $CLUSTER_HOST:$RUN_DIR"
  remote "$REMOTE_RUNS" "$RUN_DIR" "$REPO_URL" "$hash" "$STUDY_REL" "$JOB" "$PARTITION" "$PYTHON" "$refresh" "$dry" <<'EOF'
set -euo pipefail
runs=$1; run=$2; url=$3; hash=$4; study=$5; job=$6; part=$7; py=$8; refresh=$9; dry=${10}
mkdir -p "$runs" "$run/logs"
if [ ! -d "$runs/.repo.git" ]; then git clone -q --mirror "$url" "$runs/.repo.git"; fi
git --git-dir="$runs/.repo.git" fetch -q --prune origin
git --git-dir="$runs/.repo.git" cat-file -e "$hash^{commit}" || { echo "commit $hash not visible on the cluster"; exit 1; }
if [ -f "$run/code/COMMIT" ] && [ "$(cat "$run/code/COMMIT")" != "$hash" ] && [ "$refresh" != 1 ]; then
  echo "NOTE: keeping the existing snapshot $(cut -c1-9 "$run/code/COMMIT") (you are on ${hash:0:9});"
  echo "      pass --refresh-code to replace it (only when no job of this study is queued)."
else
  if [ "$refresh" = 1 ] && [ -d "$run/code" ]; then
    if squeue -h -u "$USER" --name="$job" 2>/dev/null | grep -q .; then echo "jobs of $job are still queued: cancel first"; exit 1; fi
    rm -rf "$run/code"
  fi
  if [ ! -d "$run/code" ]; then
    mkdir -p "$run/code"
    git --git-dir="$runs/.repo.git" archive "$hash" | tar -x -C "$run/code"
    echo "$hash" > "$run/code/COMMIT"
  fi
fi
cd "$run"
opts=(--out "$run" --backend slurm --job-name "$job")
[ -n "$part" ] && opts+=(--partition "$part")
[ "$dry" = 1 ] && opts+=(--dry-run)
$py code/experiments/run.py run "code/$study" "${opts[@]}"
EOF
}

cmd_status() {
  need_args "$@"; resolve "$1" "$2"
  remote "$RUN_DIR" "$STUDY_REL" "$JOB" "$PYTHON" <<'EOF'
run=$1; study=$2; job=$3; py=$4
[ -d "$run/code" ] || { echo "nothing submitted yet for this study"; exit 0; }
echo "snapshot: $(cut -c1-9 "$run/code/COMMIT")"
$py "$run/code/experiments/run.py" status "$run/code/$study" --out "$run"
echo "queue:"
squeue -h -u "$USER" --name="$job" -o "  %i %T %M %R" 2>/dev/null | head -20 || true
EOF
}

cmd_log() {
  need_args "$@"; resolve "$1" "$2"
  remote "$RUN_DIR" <<'EOF'
f=$(ls -t "$1"/logs/*.out 2>/dev/null | head -1)
[ -n "$f" ] && { echo "== $f"; tail -n 30 "$f"; } || echo "no logs yet"
EOF
}

cmd_fetch() {
  need_args "$@"; resolve "$1" "$2"
  remote "$RUN_DIR" "$STUDY_REL" "$PYTHON" <<'EOF' >&2
run=$1; study=$2; py=$3
$py "$run/code/experiments/run.py" pack "$run/code/$study" --out "$run"
EOF
  local dest="$LOCAL_RESULTS/$1/$(study_name "$STUDY_REL")"
  mkdir -p "$dest"
  ssh -o BatchMode=yes "$CLUSTER_HOST" "cat '$RUN_DIR/results.jsonl'" > "$dest/results.jsonl.part"
  mv "$dest/results.jsonl.part" "$dest/results.jsonl"
  echo "fetched -> $dest/results.jsonl"
  python3 "$ROOT/experiments/report.py" "$dest/results.jsonl" --json "$dest/summary.json"
}

cmd_cancel() {
  need_args "$@"; resolve "$1" "$2"
  remote "$JOB" <<'EOF'
scancel -u "$USER" --name="$1" && echo "cancelled jobs named $1"
EOF
}

[ "$#" -ge 1 ] || { sed -n '3,22p' "$0"; exit 1; }
cmd=$1; shift
case "$cmd" in
  check) cmd_check "$@";;
  submit) cmd_submit "$@";;
  status) cmd_status "$@";;
  log) cmd_log "$@";;
  fetch) cmd_fetch "$@";;
  cancel) cmd_cancel "$@";;
  -h|--help|help) sed -n '3,22p' "$0";;
  *) die "unknown command $cmd";;
esac
