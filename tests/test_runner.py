# tests/test_runner.py

"""
Batch runner and cluster script (experiments/run.py, report.py, cluster.sh).

  1. Study expansion: run count, stable ids, flag conversion, reserved flags refused.
  2. Local execution: results appear, a second call runs nothing, a deleted result is
     recomputed, a failing run is recorded without stopping the others and retried later,
     shards partition the runs.
  3. Report: grouping by varying parameters, seeds aggregated, failures counted.
  4. SLURM script: array size, limits, shard argument (dry run, nothing submitted).
  5. cluster.sh end to end against a FAKE cluster (ssh and sbatch replaced by local shims,
     a bare git repository as origin): snapshot of the pushed commit, submit, resume,
     status, fetch + report, refusal to submit an unpushed commit.

Run: python3 tests/test_runner.py   (about 40 s)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'experiments'))

import report as report_mod  # noqa: E402
import run as run_mod        # noqa: E402

PASSED, FAILED = 0, 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"    [ok]   {name}  {detail}")
    else:
        FAILED += 1
        print(f"    [FAIL] {name}  {detail}")


STUDY = {
    "command": "train",
    "base": {"rule": "ep", "element": "rectpair", "dataset": "bars-stripes",
             "n_input": 4, "n_hidden": 3, "epochs": 2},
    "grid": {"vf": [0, 0.05]},
    "seeds": 2,
    "slurm": {"partition": "tcm", "time": "00:30:00", "cpus": 2, "max_parallel": 3},
}


def cli(*args, **kw):
    return subprocess.run([sys.executable, str(REPO / 'experiments' / 'run.py'), *map(str, args)],
                          capture_output=True, text=True, **kw)


def test_expansion():
    print("\n[1] Study expansion")
    d = Path(tempfile.mkdtemp())
    (d / 'study.json').write_text(json.dumps(STUDY))
    study = run_mod.load_study(d / 'study.json')
    runs = run_mod.expand(study)
    check("2 grid values x 2 seeds = 4 runs", len(runs) == 4)
    check("run ids are unique", len({r['run_id'] for r in runs}) == 4)
    check("run ids are stable between calls",
          [r['run_id'] for r in runs] == [r['run_id'] for r in run_mod.expand(study)])
    argv = runs[0]['argv']
    check("underscores become dashes, values use --flag=value",
          '--n-input=4' in argv and '--seed=0' in argv and argv[0] == 'train', ' '.join(argv))
    argv2 = run_mod.to_argv('train', {'sym': True, 'soft-bounds': False, 'Is': 1e-4})
    check("true is a bare flag, false is dropped, case is kept",
          argv2 == ['train', '--sym', '--Is=0.0001'], str(argv2))
    bad = dict(STUDY, base={**STUDY['base'], 'json': 'x.json'})
    (d / 'bad.json').write_text(json.dumps(bad))
    try:
        run_mod.expand(run_mod.load_study(d / 'bad.json'))
        check("reserved flag --json is refused", False)
    except SystemExit:
        check("reserved flag --json is refused", True)
    pts = dict(STUDY, grid=[{"vf": 0}, {"vf": 0.1, "alpha": 0.25}], seeds=None)
    check("explicit points work (no product)", len(run_mod.expand(pts)) == 2)
    shutil.rmtree(d)


def test_local_and_report():
    print("\n[2] Local execution, resume, failures")
    d = Path(tempfile.mkdtemp())
    (d / 'study.json').write_text(json.dumps(STUDY))
    out = d / 'out'
    r = cli('run', d / 'study.json', '--out', out, '--workers', 4)
    files = sorted((out / 'results').glob('*.json'))
    check("run exits 0 and writes 4 result files", r.returncode == 0 and len(files) == 4, r.stderr[-200:])
    rec = json.loads(files[0].read_text())
    check("record has status, argv, git version, final metrics and weights",
          rec['status'] == 'ok' and rec['argv'][0] == 'train' and rec['git']
          and 'train' in rec['result']['final'] and rec['result']['w'])
    check("history is stored (thinned)", len(rec['result']['history']) >= 1)
    check("final evaluation convergence is recorded",
          rec['result']['final']['train'].get('eval_converged') in (0.0, 1.0)
          or 0 <= rec['result']['final']['train'].get('eval_converged', -1) <= 1)

    r2 = cli('run', d / 'study.json', '--out', out)
    check("a second call has nothing to do", '0 to run' in r2.stdout, r2.stdout.strip().splitlines()[0])
    files[0].unlink()
    r3 = cli('run', d / 'study.json', '--out', out)
    check("a deleted result is recomputed, others are kept", '1 to run' in r3.stdout
          and len(list((out / 'results').glob('*.json'))) == 4)

    # a failing run does not stop the rest and is retried on the next call
    broken = dict(STUDY, grid={"element": ["rectpair", "no-such-element"]}, seeds=[0])
    (d / 'broken.json').write_text(json.dumps(broken))
    outb = d / 'outb'
    rb = cli('run', d / 'broken.json', '--out', outb, '--workers', 2)
    st = cli('status', d / 'broken.json', '--out', outb).stdout
    check("failure: exit 1, recorded as failed, the good run is ok",
          rb.returncode == 1 and 'ok 1' in st and 'failed 1' in st, st.strip().replace('\n', ' | ')[:150])
    rb2 = cli('run', d / 'broken.json', '--out', outb)
    check("failed runs are retried, ok ones skipped", '1 to run' in rb2.stdout)

    # shards partition the runs
    ids = set()
    for i in range(3):
        outs = d / f'shard{i}'
        rs = cli('run', d / 'study.json', '--out', outs, '--shard', f'{i}/3', '--dry-run', '--show', 99)
        ids |= {ln.split('would run: ')[1] for ln in rs.stdout.splitlines() if 'would run' in ln}
    check("3 shards together cover all 4 runs exactly once", len(ids) == 4)

    print("\n[3] Report")
    cli('pack', d / 'study.json', '--out', out)
    recs = report_mod.load(out / 'results.jsonl')
    s = report_mod.summarize(recs)
    check("grouped by the varying parameter only", s['keys'] == ['vf'] and len(s['rows']) == 2, str(s['keys']))
    check("seeds are aggregated (n = 2 per row)", all(row['n'] == 2 for row in s['rows']))
    check("metrics present", all(row['mse'][0] is not None for row in s['rows']))
    txt = report_mod.table(s)
    check("markdown table renders", txt.startswith('| vf | n | mse'))
    check("wrong-pixel distribution is pooled and printed",
          'Wrong pixels per pattern' in txt and all(abs(sum(r['wrong_dist'].values()) - 1) < 1e-9
                                                     for r in s['rows']))
    sb = report_mod.summarize(report_mod.load(outb))
    check("failures are counted, not tabulated", len(sb['failed']) == 1 and sb['ok'] == 1)
    check("results from a directory of files load too", len(report_mod.load(out)) >= 1)

    print("\n[4] SLURM script (dry run)")
    o = cli('run', d / 'study.json', '--out', d / 'sl', '--backend', 'slurm', '--dry-run').stdout
    check("array size = ceil(4 runs / 2 cpus) = 2 tasks, at most 3 at once", '--array=0-1%3' in o, '')
    check("partition, time and shard argument are set",
          '--partition=tcm' in o and '--time=00:30:00' in o and '--shard "${SLURM_ARRAY_TASK_ID}/2"' in o)
    shutil.rmtree(d)


def test_new_study():
    print("\n[4b] new_study.sh")
    d = Path(tempfile.mkdtemp())
    (d / 'experiments').mkdir()
    shutil.copy(REPO / 'experiments' / 'new_study.sh', d / 'experiments')
    subprocess.run(['git', 'init', '-q'], cwd=d)
    today = subprocess.run(['date', '+%F'], capture_output=True, text=True).stdout.strip()
    r = subprocess.run(['bash', 'experiments/new_study.sh', 'demo', 'oracle'], cwd=d, capture_output=True, text=True)
    folder = d / 'experiments' / f'{today}_demo'
    check("folder is named with today's date from `date +%F`", r.returncode == 0 and folder.is_dir(), r.stderr[-100:])
    runs = run_mod.expand(run_mod.load_study(folder / 'study.json'))
    check("the skeleton study parses and expands", len(runs) == 10 and runs[0]['argv'][0] == 'oracle')
    r2 = subprocess.run(['bash', 'experiments/new_study.sh', 'demo'], cwd=d, capture_output=True, text=True)
    check("an existing study is not overwritten", r2.returncode != 0)
    r3 = subprocess.run(['bash', 'experiments/new_study.sh', 'bad name'], cwd=d, capture_output=True, text=True)
    check("a bad topic is refused", r3.returncode != 0)
    shutil.rmtree(d)


def test_cluster_flow():
    print("\n[5] cluster.sh against a fake cluster")
    tmp = Path(tempfile.mkdtemp())
    bin_ = tmp / 'bin'
    bin_.mkdir()
    (bin_ / 'ssh').write_text('#!/bin/bash\nwhile [ "$1" = "-o" ]; do shift 2; done\nshift\nexec bash -c "$*"\n')
    (bin_ / 'sbatch').write_text(
        '#!/bin/bash\nscript="${@: -1}"\n'
        'read lo hi <<< "$(grep -- "--array=" "$script" | sed "s/.*--array=\\([0-9]*\\)-\\([0-9]*\\).*/\\1 \\2/")"\n'
        'for i in $(seq $lo $hi); do SLURM_ARRAY_TASK_ID=$i bash "$script" > "$(dirname "$script")/logs/99_$i.out" 2>&1; done\n'
        'echo 99\n')
    for n in ('squeue', 'scancel'):
        (bin_ / n).write_text('#!/bin/bash\nexit 0\n')
    for p in bin_.iterdir():
        p.chmod(0o755)

    def sh(*cmd, cwd=None, env=None):
        return subprocess.run(list(map(str, cmd)), cwd=cwd, capture_output=True, text=True, env=env)

    git = ['git', '-c', 'user.name=t', '-c', 'user.email=t@t']
    origin, work = tmp / 'origin.git', tmp / 'work'
    sh('git', 'init', '-q', '--bare', origin)
    work.mkdir()
    files = sh('git', 'ls-files', '--cached', '--others', '--exclude-standard', cwd=REPO).stdout.split()
    for f in files:                                  # the working tree, including uncommitted edits
        if (REPO / f).is_file():
            (work / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / f, work / f)
    sdir = work / 'experiments' / '2099-01-01_fake'
    sdir.mkdir(parents=True)
    (sdir / 'study.json').write_text(json.dumps(STUDY))
    sh('git', 'init', '-q', cwd=work)
    sh('git', 'remote', 'add', 'origin', origin, cwd=work)
    sh(*git, 'add', '-A', cwd=work)
    sh(*git, 'commit', '-qm', 'snapshot', cwd=work)
    sh('git', 'push', '-q', 'origin', 'HEAD:refs/heads/main', cwd=work)

    conf = tmp / 'conf'
    conf.write_text(f'REMOTE_RUNS={tmp}/cluster\nLOCAL_RESULTS={tmp}/results\nPARTITION=tcm\n')
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", MERO_CLUSTER_CONF=str(conf))
    cs = lambda *a: sh('bash', work / 'experiments' / 'cluster.sh', *a, cwd=work, env=env)  # noqa: E731
    study = 'experiments/2099-01-01_fake/study.json'

    r = cs('check')
    check("check reports python, numpy and writable storage",
          'numpy' in r.stdout and 'writable' in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-150:])
    r = cs('submit', 'A', study)
    check("submit: snapshot + array job", r.returncode == 0 and 'submitted job' in r.stdout, r.stdout[-200:] + r.stderr[-200:])
    rundir = tmp / 'cluster' / 'A' / '2099-01-01_fake'
    check("code snapshot carries the commit hash",
          (rundir / 'code' / 'COMMIT').exists() and len(list((rundir / 'results').glob('*.json'))) == 4)
    r = cs('status', 'A', study)
    check("status: 4 ok", 'ok 4' in r.stdout, r.stdout.strip().replace('\n', ' | ')[:120])
    r = cs('submit', 'A', study)
    check("resubmit: nothing to run", '0 to run' in r.stdout)
    r = cs('fetch', 'A', study)
    check("fetch: results.jsonl arrives and the report is printed",
          (tmp / 'results' / 'A' / '2099-01-01_fake' / 'results.jsonl').exists() and '| vf | n |' in r.stdout,
          r.stderr[-150:])
    commit = json.loads(next((rundir / 'results').glob('*.json')).read_text())['git']
    head = sh('git', 'rev-parse', 'HEAD', cwd=work).stdout.strip()
    check("every result records the snapshot commit", commit == head, commit[:9])
    r = cs('submit', 'B', study, '--dry-run')
    check("tracks are separate directories", (tmp / 'cluster' / 'B' / '2099-01-01_fake' / 'code').exists())
    sh(*git, 'commit', '-q', '--allow-empty', '-m', 'unpushed', cwd=work)
    r = cs('submit', 'A', study)
    check("an unpushed commit is refused", r.returncode != 0 and 'git push' in r.stderr, r.stderr.strip()[-100:])
    (sdir / 'study.json').write_text(json.dumps(dict(STUDY, seeds=3)))
    r = cs('submit', 'A', study)
    check("an uncommitted study file is refused", r.returncode != 0 and 'commit' in r.stderr, r.stderr.strip()[-100:])
    shutil.rmtree(tmp)


if __name__ == '__main__':
    t0 = time.perf_counter()
    test_expansion()
    test_local_and_report()
    test_new_study()
    test_cluster_flow()
    print(f"\n{PASSED} passed, {FAILED} failed   ({time.perf_counter() - t0:.1f}s)")
    sys.exit(1 if FAILED else 0)
