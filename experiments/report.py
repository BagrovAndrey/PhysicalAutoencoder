#!/usr/bin/env python3
# experiments/report.py

"""
Summarize the results of a study: one row per parameter combination, seeds aggregated.

    python3 experiments/report.py PATH [--by vf,alpha] [--json summary.json]

PATH is a results.jsonl (from `run.py pack` / `cluster.sh fetch`), or a run directory
containing results/*.json. Only runs with status "ok" enter the table; failures are
counted and listed. Columns (mean +- std over seeds; `n` = number of seeds):

    mse, bits, exact, margin       final metrics on the training patterns
    wrong                          mean number of wrong pixels per pattern
    perfect                        seeds with every training pattern exact (k/n)
    test exact, test wrong         same on held-out patterns, if the study had --holdout

Rows are grouped by every parameter that varies in the study (except the seed); --by
picks the columns explicitly. Report the spread, not the best seed (AGENTS.md).
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    p = Path(path)
    if p.is_dir():
        if (p / 'results.jsonl').exists():
            p = p / 'results.jsonl'
        else:
            recs = []
            for f in sorted((p / 'results').glob('*.json')):
                try:
                    recs.append(json.loads(f.read_text()))
                except ValueError:
                    pass
            return recs
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def metric_values(final):
    """Scalars from one metrics dict as written by sim.py."""
    if not final:
        return {}
    wrong = final.get('wrong') or []
    return dict(mse=final['mse'], bits=final['bits'], exact=final['exact'],
                margin=final['margin'], wrong=(sum(wrong) / len(wrong)) if wrong else float('nan'))


def mean_std(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


def fmt(m, s, digits=3):
    if m is None:
        return '-'
    return f"{m:.{digits}f}" if s is None or s == 0 else f"{m:.{digits}f} ± {s:.{digits}f}"


def summarize(recs, by=None):
    ok = [r for r in recs if r.get('status') == 'ok']
    failed = [r for r in recs if r.get('status') != 'ok']
    if not ok:
        return dict(rows=[], keys=[], ok=0, failed=failed, gits=set())
    names = sorted({k for r in ok for k in r['params'] if k != 'seed'})
    varying = [k for k in names if len({json.dumps(r['params'].get(k)) for r in ok}) > 1]
    keys = by if by else varying
    groups = defaultdict(list)
    for r in ok:
        groups[tuple(json.dumps(r['params'].get(k)) for k in keys)].append(r)

    rows = []
    for gk, rs in groups.items():
        row = dict(params={k: json.loads(v) for k, v in zip(keys, gk)}, n=len(rs))
        train = [metric_values(r['result']['final'].get('train')) for r in rs]
        for name in ('mse', 'bits', 'exact', 'margin', 'wrong'):
            row[name] = mean_std([t.get(name) for t in train])
        row['perfect'] = sum(1 for t in train if t.get('exact') == 1.0)
        if any('test' in r['result']['final'] for r in rs):
            test = [metric_values(r['result']['final'].get('test')) for r in rs]
            row['test_exact'] = mean_std([t.get('exact') for t in test])
            row['test_wrong'] = mean_std([t.get('wrong') for t in test])
        rows.append(row)
    rows.sort(key=lambda r: [(v is None, v if isinstance(v, (int, float)) else str(v))
                             for v in r['params'].values()])
    return dict(rows=rows, keys=keys, ok=len(ok), failed=failed,
                gits={r.get('git') for r in ok})


def table(summary):
    keys, rows = summary['keys'], summary['rows']
    has_test = any('test_exact' in r for r in rows)
    head = keys + ['n', 'mse', 'bits', 'exact', 'margin', 'wrong', 'perfect']
    if has_test:
        head += ['test exact', 'test wrong']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] * len(head)) + '|']
    for r in rows:
        cells = [str(r['params'][k]) for k in keys] + [str(r['n']),
                 fmt(*r['mse'], 4), fmt(*r['bits'], 3), fmt(*r['exact'], 2),
                 fmt(*r['margin'], 3), fmt(*r['wrong'], 2), f"{r['perfect']}/{r['n']}"]
        if has_test:
            cells += [fmt(*r.get('test_exact', (None, None)), 2),
                      fmt(*r.get('test_wrong', (None, None)), 2)]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--by', help='comma-separated parameters to group by (default: all that vary)')
    ap.add_argument('--json', metavar='FILE', help='write the summary as JSON')
    args = ap.parse_args()

    recs = load(args.path)
    s = summarize(recs, args.by.split(',') if args.by else None)
    n_fail = len(s['failed'])
    print(f"{len(recs)} runs: {s['ok']} ok, {n_fail} failed")
    if len(s['gits']) > 1:
        print(f"WARNING: results come from {len(s['gits'])} different code versions: "
              f"{sorted(map(str, s['gits']))}")
    if any(str(g).endswith('+dirty') for g in s['gits']):
        print("WARNING: some runs used uncommitted code (+dirty)")
    for r in s['failed'][:5]:
        last = (r.get('stderr_tail') or '').splitlines()[-1:] or ['']
        print(f"FAILED {r['run_id']} {json.dumps(r['params'], sort_keys=True)}: {last[0]}")
    if s['rows']:
        print()
        print(table(s))
    if args.json:
        out = dict(ok=s['ok'], failed=n_fail, keys=s['keys'], code=sorted(map(str, s['gits'])),
                   rows=[{**r, 'params': r['params']} for r in s['rows']])
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nsummary written to {args.json}")
    return 1 if n_fail and not s['ok'] else 0


if __name__ == '__main__':
    sys.exit(main())
