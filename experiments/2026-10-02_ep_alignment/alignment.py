#!/usr/bin/env python3
"""Probe EP direction on identical training snapshots without moving their weights."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Small solves need one BLAS thread per independent seed, not nested thread pools.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from datasets.bars_stripes import BarsAndStripes
from network.elements import make_element
from network.equilibrium import autoencoder, init_weights
from training.learning import EPRule, Protocol, metrics, train


def checked(result, label):
    if result['diverged'] or not result['converged']:
        raise RuntimeError(f'{label}: diverged={result["diverged"]}, '
                           f'converged={result["converged"]}, residual={result.get("residual")}')
    if not np.all(np.isfinite(result['V_final'])):
        raise RuntimeError(f'{label}: non-finite voltages')
    return result['V_final']


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return None
    return float(np.clip(np.dot(a / na, b / nb), -1, 1))


def diagnostic(net, w, X, protocol, betas, vsig):
    before = w.copy()
    g = net.conductance(w)
    free = [protocol.relax(net, g, net.boundary(x, vsig)) for x in X]
    voltages = [checked(r, f'free pattern {p}') for p, r in enumerate(free)]
    loss, grad = net.loss_and_grad(w, X, vsig=vsig)
    m = metrics(np.array([V[net.outputs] / vsig for V in voltages]), X)
    if not np.all(np.isfinite(grad)) or not np.isfinite(loss):
        raise RuntimeError('non-finite library gradient')
    if not np.isclose(loss, m['mse'], rtol=1e-9, atol=1e-12):
        raise RuntimeError('reference loss and checked free states disagree')
    records = []
    for beta in betas:
        estimates, patterns = [], []
        for p, (x, Vf) in enumerate(zip(X, voltages)):
            plus = protocol.relax(net, g, net.boundary(x, vsig), V0=Vf, beta_gp=beta)
            Vp = checked(plus, f'nudged pattern {p}, beta={beta}')
            Df, Dp = net.drops(Vf), net.drops(Vp)
            # The unmodified one-sided estimator in EPRule; no alpha or clipping.
            est = (net.element.dPhi_dg(g, Dp) - net.element.dPhi_dg(g, Df)) / beta
            estimates.append(est)
            _, sf = net.element.current_and_slope(g, Df)
            _, sp = net.element.current_and_slope(g, Dp)
            patterns.append(dict(pattern=''.join(str(int(v)) for v in x),
                                 free_residual=free[p]['residual'], nudged_residual=plus['residual'],
                                 free_kink_branches=int(np.sum(np.abs(Df-net.element.vf) <= 1e-8)),
                                 changed_active_branches=int(np.sum((sf > 0) != (sp > 0)))))
        raw = np.mean(estimates, axis=0)
        # Energy uses 1/2 sum of voltage errors; library loss is normalized MSE
        # with gradient w.r.t. w, whereas dPhi_dg is a derivative w.r.t. g.
        scaled = raw * 2 * (net.g_max-net.g_min) / (len(net.outputs)*vsig**2)
        records.append(dict(beta_gp=beta, cosine=cosine(raw, grad),
                            ep_norm=float(np.linalg.norm(scaled)),
                            oracle_norm=float(np.linalg.norm(grad)),
                            relative_gradient_error=(float(np.linalg.norm(scaled-grad)/np.linalg.norm(grad))
                                                     if np.linalg.norm(grad) > 0 else None),
                            raw_ep_estimate=raw.tolist(), mse_scaled_ep_estimate=scaled.tolist(),
                            patterns=patterns))
    if not np.array_equal(before, w):
        raise RuntimeError('diagnostic modified weights')
    return dict(metrics={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()},
                gradient=grad.tolist(), weights=w.tolist(), alignment=records)


def run_seed(vf, seed, config):
    t0 = time.perf_counter()
    try:
        X = BarsAndStripes(2).get_all_flattened().astype(float)
        net = autoencoder(4, 3, make_element('rectpair', vf=vf), g_min=0.01, g_max=1.0)
        protocol = Protocol('equilibrium')
        initial = init_weights(net, seed, w_min=0.1, w_max=0.9)
        snapshots = {0:initial.copy()}
        unconverged = []
        def capture(cycle, x, mse, info, w):
            if not info['free_converged']:
                unconverged.append(cycle+1)
            if cycle+1 in (30*len(X), 60*len(X)):
                snapshots[(cycle+1)//len(X)] = w.copy()
        _, history, error = train(net, X, EPRule(alpha=0.5), protocol, initial,
                                  cycles=60*len(X), beta_gp=config['training_beta_gp'],
                                  vsig=1.0, order='shuffle', seed=seed, on_cycle=capture)
        if error:
            raise RuntimeError(error)
        results = []
        for epoch, w in sorted(snapshots.items()):
            results.append(dict(epoch=epoch, **diagnostic(net,w,X,protocol,config['probe_beta_gp'],1.0)))
        return dict(vf=vf, seed=seed, status='ok', seconds=time.perf_counter()-t0,
                    unconverged_training_free_cycles=unconverged, history_len=len(history), snapshots=results)
    except Exception as exc:
        return dict(vf=vf, seed=seed, status='failed', seconds=time.perf_counter()-t0,
                    error=f'{type(exc).__name__}: {exc}')


def summarize(jobs, config):
    rows = []
    for epoch in (0,30,60):
        for beta in config['probe_beta_gp']:
            for vf in config['vf']:
                entries = [(job['seed'], a) for job in jobs if job['status']=='ok' and job['vf']==vf
                           for snap in job['snapshots'] if snap['epoch']==epoch
                           for a in snap['alignment'] if a['beta_gp']==beta]
                vals = [a['cosine'] for _,a in entries if a['cosine'] is not None]
                rows.append(dict(epoch=epoch,beta_gp=beta,vf=vf,n=len(vals),
                                 mean=float(np.mean(vals)) if vals else None,
                                 std=float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                                 minimum=min(vals) if vals else None, maximum=max(vals) if vals else None,
                                 seeds_below_or_equal_09=[seed for seed,a in entries if a['cosine'] is not None and a['cosine']<=0.9],
                                 undefined_seeds=[seed for seed,a in entries if a['cosine'] is None]))
    return rows


def table(rows):
    out=['| epoch | beta*g_penalty | vf | cosine mean ± sample std | min | n |',
         '|---|---|---|---|---|---|']
    for r in rows:
        value=f'{r["mean"]:.6f} ± {r["std"]:.6f}' if r['mean'] is not None and r['std'] is not None else 'undefined'
        minimum=f'{r["minimum"]:.6f}' if r['minimum'] is not None else 'undefined'
        out.append(f'| {r["epoch"]} | {r["beta_gp"]:g} | {r["vf"]:g} | {value} | {minimum} | {r["n"]} |')
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,default=Path(__file__).with_name('results.json'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--training-beta',type=float,default=0.1,
                        help='fixed training beta*g_penalty; probes do not alter trajectories')
    args=parser.parse_args()
    if args.workers < 1 or args.training_beta <= 0:
        parser.error('workers and training-beta must be positive')
    config=dict(vf=[0,0.02,0.05,0.1],seeds=list(range(5)),probe_beta_gp=[0.1,0.01,0.001],
                training_beta_gp=args.training_beta,epochs=[0,30,60],n_input=4,n_hidden=3,
                element='rectpair',dataset='2x2 Bars & Stripes (all six)',g_min=0.01,g_max=1.0,
                vsig=1.0,alpha=0.5,w_min=0.1,w_max=0.9,order='shuffle',protocol='equilibrium',
                symmetric=False,soft_bounds=False,kink_tolerance=1e-8)
    t0=time.perf_counter()
    jobs=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_seed,vf,seed,config) for vf in config['vf'] for seed in config['seeds']]
        for f in as_completed(futures):
            job=f.result();jobs.append(job)
            print(f'[{len(jobs)}/20] vf={job["vf"]} seed={job["seed"]} {job["status"]} '
                  f'{job["seconds"]:.1f}s'+(f' {job["error"]}' if 'error' in job else ''),flush=True)
    jobs.sort(key=lambda j:(j['vf'],j['seed']))
    rows=summarize(jobs,config)
    failed=[j for j in jobs if j['status']!='ok']
    valid=len(failed)==0 and all(r['n']==5 for r in rows)
    payload=dict(config=config,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 library_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [ROOT/'network/elements.py',ROOT/'network/equilibrium.py',ROOT/'training/learning.py',ROOT/'datasets/bars_stripes.py']},
                 created_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-t0,
                 jobs=jobs,rows=rows,valid=valid,
                 criterion=dict(threshold=0.9,all_seed_cosines_above_09=valid and all(r['minimum']>0.9 for r in rows),
                                all_group_means_above_09=valid and all(r['mean']>0.9 for r in rows)))
    args.json.parent.mkdir(parents=True,exist_ok=True)
    args.json.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('\n'+table(rows));print('\nCriterion:',json.dumps(payload['criterion']))
    print('JSON:',args.json)
    return 0 if valid else 1

if __name__=='__main__':
    raise SystemExit(main())
