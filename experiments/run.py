#!/usr/bin/env python3
# experiments/run.py

"""
Batch runner: a study file (JSON) describes a grid of `sim.py` runs; this expands it
into independent runs, executes them in parallel, and stores one JSON per run.

    python3 experiments/run.py plan   STUDY.json                  # what would run
    python3 experiments/run.py run    STUDY.json --out DIR        # run here, N processes
    python3 experiments/run.py run    STUDY.json --out DIR --backend slurm
    python3 experiments/run.py status STUDY.json --out DIR
    python3 experiments/run.py pack   STUDY.json --out DIR        # -> DIR/results.jsonl

Study file:
    {
      "question": "free text: what is this study asking",
      "command":  "train",                       # sim.py subcommand: train | oracle | handbuilt
      "base":     {"rule": "ep", "element": "rectpair", "dataset": "bars-stripes",
                   "n-input": 4, "n-hidden": 3, "epochs": 60},     # fixed sim.py flags
      "grid":     {"vf": [0, 0.05, 0.1], "alpha": [0.25, 0.5]},   # cartesian product
      "seeds":    [0, 1, 2],                     # or an integer n = seeds 0..n-1; sets --seed
      "slurm":    {"partition": "tcm", "time": "04:00:00", "cpus": 16, "mem": "16G",
                   "max_parallel": 10, "max_tasks": 50},
      "timeout_s": 3600                          # per run, optional
    }
Keys are `sim.py` flags without the dashes (`n-hidden` or `n_hidden`, `Is`, ...); `true`
gives a bare flag, `false`/`null` omits it. `grid` may instead be a list of dicts (explicit
points, not a product). Runs are `sim.py <command> --flag=value ... --json FILE`, so every
result is reproducible by hand from the `argv` stored in it.

Resumable: a run's id is a hash of (command, flags, seed); a run whose result file says
"ok" is skipped, everything else (missing, failed, corrupt) is run again. The runner uses
the `sim.py` next to it, so a code snapshot brings its own `sim.py`.
"""

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESERVED = {'json', 'plot', 'log-lines', 'help', 'h'}
HISTORY_POINTS = 200


# ---------------------------------------------------------------- study -> runs
def _norm(key):
    return key.replace('_', '-')


def load_study(path):
    study = json.loads(Path(path).read_text())
    if study.get('command') not in ('train', 'oracle', 'handbuilt'):
        raise SystemExit("study: 'command' must be train, oracle or handbuilt")
    study.setdefault('base', {})
    study.setdefault('grid', {})
    study.setdefault('name', Path(path).resolve().parent.name)
    return study


def _seeds(study):
    s = study.get('seeds')
    if s is None:
        return [None]
    return list(range(s)) if isinstance(s, int) else list(s)


def expand(study):
    """All runs of a study, in a fixed order: [{run_id, command, params, argv}]."""
    grid = study['grid']
    if isinstance(grid, dict):
        keys = list(grid)
        points = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    else:
        points = [dict(p) for p in grid]
    runs, seen = [], set()
    for point in points:
        for seed in _seeds(study):
            params = {_norm(k): v for k, v in {**study['base'], **point}.items()}
            if seed is not None:
                params['seed'] = seed
            bad = RESERVED & set(params)
            if bad:
                raise SystemExit(f"study: the runner controls {sorted(bad)}; remove from base/grid")
            canon = json.dumps({'command': study['command'], 'params': params}, sort_keys=True)
            run_id = hashlib.sha1(canon.encode()).hexdigest()[:12]
            if run_id in seen:
                raise SystemExit(f"study: duplicate run {params}")
            seen.add(run_id)
            runs.append(dict(run_id=run_id, command=study['command'], params=params,
                             argv=to_argv(study['command'], params)))
    return runs


def to_argv(command, params):
    argv = [command]
    for k, v in params.items():
        if v is True:
            argv.append(f'--{k}')
        elif v is False or v is None:
            continue
        else:
            argv.append(f'--{k}={v}')
    return argv


# ---------------------------------------------------------------- results on disk
def result_path(out, run_id):
    return Path(out) / 'results' / f'{run_id}.json'


def read_record(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def is_done(out, run_id):
    rec = read_record(result_path(out, run_id))
    return bool(rec) and rec.get('status') == 'ok'


def code_version():
    f = REPO / 'COMMIT'
    if f.exists():
        return f.read_text().strip()
    try:
        h = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'],
                               capture_output=True, text=True).stdout.strip()
        return h + ('+dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _thin(history):
    if not history or len(history) <= HISTORY_POINTS:
        return history
    step = len(history) / HISTORY_POINTS
    return [history[int(i * step)] for i in range(HISTORY_POINTS)]


def _tail(text, n=20):
    return '\n'.join(text.strip().splitlines()[-n:])


def run_one(run, out, timeout, version):
    out = Path(out)
    tmp = out / 'tmp' / f"{run['run_id']}.json"
    cmd = [sys.executable, str(REPO / 'sim.py')] + run['argv'] + ['--json', str(tmp)]
    if run['command'] == 'train':
        cmd.append('--log-lines=0')
    env = dict(os.environ)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[v] = '1'                       # many small runs in parallel: no BLAS oversubscription
    rec = dict(run_id=run['run_id'], command=run['command'], params=run['params'],
               argv=run['argv'], git=version, host=platform.node(),
               started=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        rec['seconds'] = round(time.perf_counter() - t0, 2)
        if p.returncode == 0 and tmp.exists():
            d = json.loads(tmp.read_text())
            hist = d.pop('history', None)
            rec.update(status='ok', result=dict(final=d['final'], w=d['w']))
            if hist is not None:
                rec['result']['history'] = _thin(hist)
                rec['result']['history_len'] = len(hist)
            if 'restart_mse' in d:
                rec['result']['restart_mse'] = d['restart_mse']
        else:
            rec.update(status='failed', returncode=p.returncode,
                       stderr_tail=_tail(p.stderr), stdout_tail=_tail(p.stdout))
    except subprocess.TimeoutExpired:
        rec.update(status='failed', returncode=None, stderr_tail=f'timeout after {timeout}s',
                   seconds=round(time.perf_counter() - t0, 2))
    except (OSError, ValueError) as e:
        rec.update(status='failed', returncode=None, stderr_tail=repr(e),
                   seconds=round(time.perf_counter() - t0, 2))
    finally:
        if tmp.exists():
            tmp.unlink()
    dest = result_path(out, run['run_id'])
    part = dest.with_suffix('.part')
    part.write_text(json.dumps(rec))
    os.replace(part, dest)                 # atomic: a result file is complete or absent
    return rec


# ---------------------------------------------------------------- commands
def _shard(runs, spec):
    if not spec:
        return runs
    i, n = (int(x) for x in spec.split('/'))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {spec}: need 0 <= i < n")
    return runs[i::n]


def _prepare(out):
    for sub in ('results', 'logs', 'tmp'):
        (Path(out) / sub).mkdir(parents=True, exist_ok=True)


def cmd_plan(args):
    study = load_study(args.study)
    runs = expand(study)
    print(f"study {study['name']!r}: {len(runs)} runs of `sim.py {study['command']}`")
    if study.get('question'):
        print(f"question: {study['question']}")
    for r in runs[:args.show]:
        print(f"  {r['run_id']}  sim.py {' '.join(r['argv'])}")
    if len(runs) > args.show:
        print(f"  ... {len(runs) - args.show} more")
    return 0


def cmd_run(args):
    study = load_study(args.study)
    runs = expand(study)
    out = Path(args.out or Path('runs') / study['name']).resolve()
    _prepare(out)
    mine = _shard(runs, args.shard)
    pending = [r for r in mine if not is_done(out, r['run_id'])]
    print(f"study {study['name']!r}: {len(runs)} runs, {len(mine)} in this shard, "
          f"{len(mine) - len(pending)} already done, {len(pending)} to run  -> {out}", flush=True)
    if args.dry_run and args.backend == 'local':
        for r in pending[:args.show]:
            print(f"  would run: sim.py {' '.join(r['argv'])}")
        return 0
    if not pending:
        return 0
    if args.backend == 'slurm':
        return _submit_slurm(study, runs, out, args)

    version = code_version()
    workers = args.workers or os.cpu_count() or 1
    timeout = study.get('timeout_s')
    failed = 0
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, r, out, timeout, version): r for r in pending}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            failed += rec['status'] != 'ok'
            print(f"[{k}/{len(pending)}] {rec['run_id']} {rec['status']:6s} {rec['seconds']:7.1f}s  "
                  f"{json.dumps(rec['params'], sort_keys=True)}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s, {failed} failed of {len(pending)}")
    return 1 if failed else 0


def _submit_slurm(study, runs, out, args):
    cfg = dict(partition=None, time='04:00:00', cpus=16, mem='16G', max_parallel=10, max_tasks=50)
    cfg.update(study.get('slurm', {}))
    if args.partition:
        cfg['partition'] = args.partition
    pending = [r for r in runs if not is_done(out, r['run_id'])]
    n_tasks = min(cfg['max_tasks'], max(1, math.ceil(len(pending) / cfg['cpus'])))
    job = args.job_name or f"mero-{study['name']}"
    lines = ['#!/bin/bash', f'#SBATCH --job-name={job}']
    if cfg['partition']:
        lines.append(f"#SBATCH --partition={cfg['partition']}")
    lines += [f"#SBATCH --time={cfg['time']}", '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
              f"#SBATCH --cpus-per-task={cfg['cpus']}", f"#SBATCH --mem={cfg['mem']}",
              f"#SBATCH --array=0-{n_tasks - 1}%{cfg['max_parallel']}",
              f"#SBATCH --output={out}/logs/%A_%a.out",
              'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
              f'cd "{out}"',
              f'"{sys.executable}" "{Path(__file__).resolve()}" run "{Path(args.study).resolve()}" '
              f'--out "{out}" --backend local --workers {cfg["cpus"]} '
              f'--shard "${{SLURM_ARRAY_TASK_ID}}/{n_tasks}"']
    script = '\n'.join(lines) + '\n'
    (out / 'submit.sh').write_text(script)
    print(f"{len(pending)} runs -> array of {n_tasks} tasks x {cfg['cpus']} cores, "
          f"at most {cfg['max_parallel']} at once, {cfg['time']} each")
    if args.dry_run:
        print('--- submit.sh ---\n' + script)
        return 0
    try:
        p = subprocess.run(['sbatch', '--parsable', str(out / 'submit.sh')], capture_output=True,
                           text=True)
    except FileNotFoundError:
        raise SystemExit("sbatch not found: run this on the cluster, or use --backend local")
    if p.returncode != 0:
        raise SystemExit(f"sbatch failed: {p.stderr.strip()}")
    jobid = p.stdout.strip().split(';')[0]
    with open(out / 'jobs.txt', 'a') as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {jobid} "
                f"{job} {len(pending)} runs\n")
    print(f"submitted job {jobid} ({job})")
    return 0


def cmd_status(args):
    study = load_study(args.study)
    runs = expand(study)
    out = Path(args.out or Path('runs') / study['name']).resolve()
    ok, failed, secs = 0, [], []
    for r in runs:
        rec = read_record(result_path(out, r['run_id']))
        if rec and rec.get('status') == 'ok':
            ok += 1
            secs.append(rec.get('seconds', 0))
        elif rec:
            failed.append(rec)
    pending = len(runs) - ok - len(failed)
    print(f"study {study['name']!r} in {out}")
    print(f"  total {len(runs)}   ok {ok}   failed {len(failed)}   not run {pending}")
    if secs:
        mean = sum(secs) / len(secs)
        print(f"  mean {mean:.1f}s per run; remaining {len(runs) - ok} runs ~ "
              f"{(len(runs) - ok) * mean / 3600:.2f} core-hours")
    for rec in failed[:5]:
        last = (rec.get('stderr_tail') or '').splitlines()[-1:] or ['']
        print(f"  FAILED {rec['run_id']} {json.dumps(rec['params'], sort_keys=True)}: {last[0]}")
    if len(failed) > 5:
        print(f"  ... {len(failed) - 5} more failed")
    return 0


def cmd_pack(args):
    study = load_study(args.study)
    out = Path(args.out or Path('runs') / study['name']).resolve()
    files = sorted((out / 'results').glob('*.json'))
    dest = out / 'results.jsonl'
    n = 0
    with open(dest.with_suffix('.tmp'), 'w') as f:
        for p in files:
            rec = read_record(p)
            if rec:
                f.write(json.dumps(rec) + '\n')
                n += 1
    os.replace(dest.with_suffix('.tmp'), dest)
    print(f"{n} records -> {dest}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('plan', cmd_plan), ('run', cmd_run), ('status', cmd_status),
                     ('pack', cmd_pack)):
        p = sub.add_parser(name)
        p.add_argument('study', help='study JSON file')
        if name != 'plan':
            p.add_argument('--out', help='results directory (default runs/<study name>)')
        p.set_defaults(func=fn)
        if name in ('plan', 'run'):
            p.add_argument('--show', type=int, default=8, help='how many runs to list')
        if name == 'run':
            p.add_argument('--backend', choices=('local', 'slurm'), default='local')
            p.add_argument('--workers', type=int, help='parallel processes (default: all cores)')
            p.add_argument('--shard', metavar='I/N', help='only runs I, I+N, I+2N, ... (used by SLURM)')
            p.add_argument('--partition', help='slurm partition (overrides the study file)')
            p.add_argument('--job-name', help='slurm job name')
            p.add_argument('--dry-run', action='store_true', help='show what would happen')
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
