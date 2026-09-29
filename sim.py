#!/usr/bin/env python3

"""
Command-line interface to the MeroCircuit simulator (engine: network/elements.py,
network/equilibrium.py, training/learning.py).

    python3 sim.py relax                  one relaxation of the initial network
    python3 sim.py train                  a learning run
    python3 sim.py oracle                 best reconstruction any conductances can reach
    python3 sim.py handbuilt              hand-wired AND/OR network for Bars & Stripes
    python3 sim.py sweep --param beta     one knob, several values
    python3 sim.py stability              where explicit Euler blows up
    python3 sim.py bench                  cost vs network size
    python3 sim.py info                   what's in the box

Every subcommand takes --help. Defaults reproduce the repo's historical runs:
a bare `python3 sim.py train` is the 4-3-4 contrastive run on [1,0,1,0].
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from network.elements import ELEMENTS, make_element
from network.equilibrium import autoencoder, init_weights
from training.learning import (ContrastiveRule, EPRule, GlobalThetaRule, Protocol,
                               evaluate, handbuilt_bars_stripes, holdout_split, metrics,
                               oracle, train)

RULES = ('contrastive', 'global-theta', 'ep')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_pattern(spec, n_input, seed=0):
    """--pattern accepts: alternating | ones | zeros | random | 1,0,1,0"""
    if spec == 'alternating':
        return np.array([1.0 if k % 2 == 0 else 0.0 for k in range(n_input)])
    if spec == 'ones':
        return np.ones(n_input)
    if spec == 'zeros':
        return np.zeros(n_input)
    if spec == 'random':
        return np.random.default_rng(seed).integers(0, 2, n_input).astype(float)
    values = [float(x) for x in spec.split(',')]
    if len(values) != n_input:
        raise SystemExit(f"--pattern has {len(values)} values but n_input is {n_input}")
    return np.array(values)


def load_patterns(args):
    if getattr(args, 'dataset', None) == 'bars-stripes':
        from datasets.bars_stripes import BarsAndStripes
        N = int(round(args.n_input ** 0.5))
        if N * N != args.n_input:
            raise SystemExit(f"--dataset bars-stripes needs a square --n-input (4, 9, 16, ...), "
                             f"got {args.n_input}")
        return np.asarray(BarsAndStripes(N=N).get_all_flattened(), dtype=float), N
    return parse_pattern(args.pattern, args.n_input, args.seed)[None, :], None


def element_from(args):
    return make_element(args.element, vth=args.vth, steepness=args.steepness, v0=args.v0,
                        vf=args.vf, Is=args.Is, n=args.n)


def make_net(args, n_hidden=None):
    net = autoencoder(args.n_input, n_hidden or args.n_hidden, element_from(args),
                      arch=args.arch, g_min=args.g_min, g_max=args.g_max,
                      capacitance=args.capacitance)
    return net


def fmt(arr, p=4):
    return np.array2string(np.asarray(arr), precision=p, suppress_small=True,
                           max_line_width=200)


def status(res):
    if res.get('diverged'):
        return "DIVERGED"
    return "converged" if res['converged'] else "not converged"


def default_beta(args, rule=None):
    """Only beta * g_penalty enters the physics. Historical rules use a strong nudge
    (1000); EP needs a weak one (0.1), and much weaker for Shockley diodes, whose
    nudge must move voltages by << kT/q (1e-4, or 1e-3 with symmetric nudging)."""
    if args.beta is not None:
        return args.beta
    if rule == 'ep':
        if args.element == 'shockley':
            return (1e-3 if getattr(args, 'sym', False) else 1e-4) / args.g_penalty
        return 0.1 / args.g_penalty
    return 100.0


def dist_line(m):
    return " ".join(f"{k}:{f * 100:.0f}%" for k, f in enumerate(m['wrong_hist']))


def print_distribution(title, m, n_input):
    P = len(m['wrong'])
    counts = np.bincount(m['wrong'], minlength=n_input + 1)
    ks = range(int(m['wrong'].max()) + 1)
    print(f"\n  {title}, {P} pattern{'s' if P > 1 else ''} (pixels thresholded at 0.5):")
    print("    wrong pixels | " + " ".join(f"{k:>5}" for k in ks))
    print("    patterns     | " + " ".join(f"{counts[k] / P * 100:>4.0f}%" for k in ks))
    print("                 | " + " ".join(f"{counts[k]:>5}" for k in ks))
    print(f"    mean wrong pixels per pattern: {m['wrong'].mean():.2f} of {n_input}"
          f"   (bit accuracy {m['bits'] * 100:.1f}%, MSE {m['mse']:.4f}, margin {m['margin']:+.3f})")


def show_patterns(outs, X, N):
    for x, o in zip(X, outs):
        k = int(np.sum((o > 0.5) != (x > 0.5)))
        if N and N > 1:
            xi, oi = x.reshape(N, N), o.reshape(N, N)
            for r in range(N):
                tag = f"wrong: {k}" if r == 0 else ""
                print("      " + " ".join(str(int(v)) for v in xi[r]) + "   ->   "
                      + " ".join(f"{v:4.2f}" for v in oi[r]) + "   " + tag)
            print()
        else:
            print(f"    {x.astype(int)} -> {np.round(o, 2)}   wrong: {k}")


def header(args, net, extra=""):
    n_hid = len(net.groups['hidden'])
    print(f"network : {args.n_input}-{n_hid}-{args.n_input}"
          + (f" ({net.arch})" if net.arch != 'plain' else "")
          + f", {net.n} nodes, {len(net.edges)} edges, {net.n_branches} branches")
    print(f"element : {net.element.describe()}   g in [{args.g_min}, {args.g_max}]   "
          f"signal {args.vsig} V" + extra)


# ---------------------------------------------------------------------------
# relax
# ---------------------------------------------------------------------------

def cmd_relax(args):
    net = make_net(args)
    X, _ = load_patterns(args)
    x = X[0]
    beta = default_beta(args)
    nudge = beta * args.g_penalty
    w = init_weights(net, args.seed, args.w_min, args.w_max)
    g = net.conductance(w)
    header(args, net)
    print(f"pattern : {fmt(x)}")
    proto = Protocol(args.protocol or 'exposure', dt=args.dt, max_steps=args.max_steps, tol=args.tol,
                     divergence_threshold=None if args.no_divergence_guard else args.divergence_threshold)
    print(f"protocol: {proto.describe()}   nudge beta*g_p = {nudge:g}")
    print()
    t0 = time.perf_counter()
    res = proto.relax(net, g, net.boundary(x, args.vsig), beta_gp=nudge)
    elapsed = time.perf_counter() - t0
    V = res['V_final']
    steps = f" in {res['n_steps']} steps" if 'n_steps' in res else ""
    print(f"result  : {status(res)}{steps} ({elapsed * 1000:.0f} ms)")
    if res.get('diverged'):
        print()
        print("  The solver blew up. Explicit Euler is only conditionally stable;")
        print(f"  at beta*g_penalty={nudge:g} try a smaller --dt")
        print("  (critical dt is ~0.002 at beta*g_penalty=1000). Run")
        print("  `python3 sim.py stability` to see the map.")
        return 1
    if not res['converged'] and proto.kind == 'exposure':
        print("           (not an error: relaxing for a fixed exposure time is the")
        print("            intended protocol - raise --max-steps or use --protocol equilibrium)")
    print()
    for name in ('input', 'complement', 'bias', 'hidden', 'output'):
        idx = net.groups.get(name, [])
        if len(idx):
            print(f"  {name:<10} {fmt(V[idx])}")
    print()
    out = V[net.outputs] / args.vsig
    print(f"  reconstruction MSE : {float(np.mean((out - x) ** 2)):.6f}")
    F = net.forces(V, g, nudge)
    print(f"  max |dV/dt|        : {float(np.abs(F[net.free] / net.C[net.free]).max()):.3e}  (~0 means settled)")
    F0 = net.forces(V, g, 0.0)
    imb = float(np.abs(F0[net.free]).max())
    if nudge:
        print(f"  element current imbalance : {imb:.3e}  (the penalty injection - this is what drives learning)")
    else:
        print(f"  max |net current|  : {imb:.3e}  (Kirchhoff residual, ~0 at equilibrium)")
    if args.show_currents:
        I = net.currents(V, g)
        print("\n  branch currents (from -> into : I):")
        for b in range(net.n_branches):
            print(f"    {net.bj[b]:3d} -> {net.bi[b]:<3d} : {I[b]:+.5f}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def make_rule(args):
    if args.rule == 'contrastive':
        return ContrastiveRule(eta=args.eta, gamma=args.gamma, dt_plasticity=args.dt_plasticity)
    if args.rule == 'global-theta':
        return GlobalThetaRule(eta=args.eta, gamma=args.gamma, sign=args.sign,
                               tau_theta=args.tau_theta, exposure=args.exposure,
                               micro_steps=args.micro_steps, window_pts=args.window_pts,
                               observable=args.observable)
    return EPRule(alpha=args.alpha, symmetric=args.sym, soft_bounds=args.soft_bounds)


def train_protocol(args):
    kind = args.protocol or ('equilibrium' if args.rule == 'ep' else 'exposure')
    if args.rule == 'global-theta':
        steps = args.relax_max_steps or int(round(args.exposure / args.dt))
    else:
        steps = args.max_steps
    return Protocol(kind, dt=args.dt, max_steps=steps, tol=args.tol,
                    divergence_threshold=None if args.no_divergence_guard else args.divergence_threshold)


def cmd_train(args):
    net = make_net(args)
    X, N = load_patterns(args)
    train_set, test_set = holdout_split(X, args.holdout, args.seed) if len(X) > 1 else (X, None)
    rule = make_rule(args)
    proto = train_protocol(args)
    beta = default_beta(args, args.rule)
    nudge = beta * args.g_penalty
    P = len(train_set)
    cycles = args.epochs * P if args.epochs else args.cycles
    w0 = init_weights(net, args.seed, args.w_min, args.w_max)

    header(args, net)
    print(f"rule    : {rule.describe()}")
    print(f"protocol: {proto.describe()}   nudge beta*g_p = {nudge:g}")
    print(f"patterns: {P} train" + (f" / {len(test_set)} held out (non-trivial)" if test_set is not None else "")
          + ("  (Bars & Stripes)" if N else f"  {fmt(X[0])}")
          + (f", {args.order} order" if P > 1 else ""))
    print(f"cycles  : {cycles}" + (f" ({cycles / P:g} epochs)" if P > 1 else ""))
    print()

    every = max(1, cycles // args.log_lines) if args.log_lines else cycles + 1
    n_conv = [0]

    def on_cycle(c, x, mse, info, w):
        n_conv[0] += int(bool(info.get('free_converged')))
        if c == 0 or (c + 1) % every == 0 or c == cycles - 1:
            line = (f"  cycle {c + 1:>5}/{cycles}  MSE={mse:.6f}  "
                    f"out={fmt(info['V_free'][net.outputs] / args.vsig, 3)}")
            if 'theta' in info:
                line += f"  theta={info['theta']:.5f}"
            print(line + f"  w=[{w.min():.3f},{w.max():.3f}]", flush=True)

    t0 = time.perf_counter()
    w, history, err = train(net, train_set, rule, proto, w0, cycles, nudge, vsig=args.vsig,
                            order=args.order, seed=args.seed, on_cycle=on_cycle)
    elapsed = time.perf_counter() - t0
    if err:
        print(f"\n  ABORTED: {err}. Try a smaller --dt, or a smaller --beta / --eta.")
        return 1

    print(f"\n  {cycles} cycles in {elapsed:.1f}s ({elapsed / cycles * 1000:.0f} ms/cycle)")
    if proto.kind == 'exposure':
        print(f"  free phase reached equilibrium in {n_conv[0]}/{cycles} cycles"
              + ("  (fixed exposure cuts it off; --protocol equilibrium relaxes fully)"
                 if n_conv[0] == 0 else ""))
    if P > 1:
        n_full = len(history) // P
        curve = ([float(np.mean(history[e * P:(e + 1) * P])) for e in range(n_full)]
                 if n_full else [float(np.mean(history))])
        unit = 'epoch'
        if len(curve) <= 40:
            print(f"  mean MSE per epoch (during training): {[round(c, 4) for c in curve]}")
    else:
        curve, unit = history, 'cycle'
    first, last = curve[0], curve[-1]
    change = f"(improvement: {(first - last) / first * 100:+.1f}%)" if first > 0 else ""
    print(f"  {'mean ' if P > 1 else ''}MSE {first:.6f} -> {last:.6f}   {change}")

    # Final evaluation: fresh free relaxation of every pattern with the final weights.
    outs = evaluate(net, w, train_set, args.vsig, proto)
    m = metrics(outs, train_set)
    print_distribution("final reconstruction (fresh free relaxation)", m, args.n_input)
    if args.show_patterns:
        show_patterns(outs, train_set, N)
    result = dict(train=m)
    if test_set is not None:
        outs_t = evaluate(net, w, test_set, args.vsig, proto)
        mt = metrics(outs_t, test_set)
        print_distribution("held-out patterns", mt, args.n_input)
        if args.show_patterns:
            show_patterns(outs_t, test_set, N)
        result['test'] = mt

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(curve) + 1), curve, linewidth=2)
        ax.set_xlabel(unit)
        ax.set_ylabel('reconstruction MSE (free phase, during training)')
        ax.set_title(f"{args.n_input}-{args.n_hidden}-{args.n_input}, {args.element}, rule={args.rule}")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"\n  plot saved to {args.plot}")
    if args.json:
        dump = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                for k, v in result.items()}
        json.dump(dict(args={k: v for k, v in vars(args).items() if k != 'func'}, history=history, final=dump, w=w.tolist()),
                  open(args.json, 'w'), indent=1)
        print(f"  saved {args.json}")
    return 0


# ---------------------------------------------------------------------------
# oracle and handbuilt
# ---------------------------------------------------------------------------

def cmd_oracle(args):
    net = make_net(args)
    X, N = load_patterns(args)
    header(args, net)
    print("oracle  : projected Adam with exact gradients at the free-phase equilibrium")
    print("          (not a physical rule - the ceiling a rule should be judged against)")
    proto = Protocol('equilibrium')
    t0 = time.perf_counter()
    best = None
    for r in range(args.restarts):
        w0 = init_weights(net, args.seed + r, args.w_min, args.w_max)
        w = oracle(net, X, w0, vsig=args.vsig, iters=args.iters, lr=args.lr)
        outs = evaluate(net, w, X, args.vsig, proto)
        m = metrics(outs, X)
        print(f"  restart {r + 1}/{args.restarts}: MSE={m['mse']:.4f}  wrong px [{dist_line(m)}]", flush=True)
        if best is None or m['mse'] < best[0]['mse']:
            best = (m, outs)
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    print_distribution("best restart", best[0], args.n_input)
    if args.show_patterns:
        show_patterns(best[1], X, N)
    return 0


def cmd_handbuilt(args):
    X, N = load_patterns(args)
    if not N or len(X) < 2:
        raise SystemExit("handbuilt needs --dataset bars-stripes and a square --n-input")
    if args.element not in ('rectpair', 'shockley'):
        raise SystemExit("handbuilt needs a rectifying element: --element rectpair or shockley")
    net = make_net(args, n_hidden=2 * N)
    header(args, net)
    print(f"weights : hand-built AND/OR (N={N} row + {N} column detectors), pull-up {args.pullup}")
    w = handbuilt_bars_stripes(net, N, args.pullup)
    t0 = time.perf_counter()
    outs = evaluate(net, w, X, args.vsig, Protocol('equilibrium'))
    m = metrics(outs, X)
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    print_distribution("reconstruction", m, args.n_input)
    if args.show_patterns:
        show_patterns(outs, X, N)
    return 0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

SWEEPABLE = ('beta', 'g_penalty', 'dt', 'capacitance', 'eta', 'gamma', 'alpha', 'tau_theta',
             'n_hidden', 'window_pts', 'exposure', 'w_min', 'seed', 'vsig', 'vth', 'vf',
             'steepness', 'v0', 'Is', 'g_min')
INT_PARAMS = ('n_hidden', 'window_pts', 'seed')


def cmd_sweep(args):
    if args.param not in SWEEPABLE:
        raise SystemExit(f"--param must be one of: {', '.join(SWEEPABLE)}")
    cast = int if args.param in INT_PARAMS else float
    values = [cast(v) for v in args.values.split(',')]
    mode = 'train' if args.train else 'relax'
    print(f"sweeping {args.param} over {values}   (mode: {mode}, element: {args.element})\n")
    if mode == 'relax':
        head = f"{args.param:>12} | {'status':<13} | {'steps':>7} | {'MSE':>10} | output"
    else:
        head = f"{args.param:>12} | {'MSE start':>10} | {'MSE end':>10} | {'exact':>6} | wrong px (final)"
    print(head); print("-" * len(head))
    for v in values:
        setattr(args, args.param, v)
        net = make_net(args)
        X, _ = load_patterns(args)
        if mode == 'relax':
            nudge = default_beta(args) * args.g_penalty
            proto = Protocol(args.protocol or 'exposure', dt=args.dt, max_steps=args.max_steps, tol=args.tol)
            w = init_weights(net, args.seed, args.w_min, args.w_max)
            res = proto.relax(net, net.conductance(w), net.boundary(X[0], args.vsig), beta_gp=nudge)
            if res.get('diverged'):
                print(f"{v:>12} | {'DIVERGED':<13} | {res.get('n_steps', 0):>7} | {'-':>10} | -")
                continue
            out = res['V_final'][net.outputs] / args.vsig
            print(f"{v:>12} | {status(res):<13} | {res.get('n_steps', 0):>7} | "
                  f"{float(np.mean((out - X[0]) ** 2)):>10.6f} | {fmt(out, 3)}")
        else:
            proto = train_protocol(args)
            nudge = default_beta(args, args.rule) * args.g_penalty
            w0 = init_weights(net, args.seed, args.w_min, args.w_max)
            cycles = args.epochs * len(X) if args.epochs else args.cycles
            w, hist, err = train(net, X, make_rule(args), proto, w0, cycles, nudge,
                                 vsig=args.vsig, order=args.order, seed=args.seed)
            if err:
                print(f"{v:>12} | ABORTED: {err}")
                continue
            m = metrics(evaluate(net, w, X, args.vsig, proto), X)
            print(f"{v:>12} | {hist[0]:>10.6f} | {hist[-1]:>10.6f} | {m['exact'] * 100:>5.0f}% | {dist_line(m)}")
    return 0


# ---------------------------------------------------------------------------
# stability and bench
# ---------------------------------------------------------------------------

def cmd_stability(args):
    dts = [float(x) for x in args.dts.split(',')]
    betas = [float(x) for x in args.betas.split(',')]
    net = make_net(args)
    X, _ = load_patterns(args)
    g = net.conductance(init_weights(net, args.seed, args.w_min, args.w_max))
    bv = net.boundary(X[0], args.vsig)
    print(f"stability map: {net.element.describe()}, g_penalty={args.g_penalty}, "
          f"{args.n_input}-{args.n_hidden}-{args.n_input}, max_steps={args.max_steps}\n")
    print(f"{'dt':>10} | " + " | ".join(f"beta={b:<9g}" for b in betas))
    print("-" * (12 + 15 * len(betas)))

    def run(dt, beta):
        return net.relax_euler(g, bv, beta_gp=beta * args.g_penalty, dt=dt,
                               max_steps=args.max_steps, tol=args.tol)
    for dt in dts:
        cells = []
        for b in betas:
            r = run(dt, b)
            cells.append("DIVERGES" if r['diverged'] else (f"ok {r['n_steps']}" if r['converged'] else "slow"))
        print(f"{dt:>10g} | " + " | ".join(f"{c:<14}" for c in cells))
    print("\n  ok N     = converged in N steps")
    print("  slow     = finite but not converged within max_steps")
    print("  DIVERGES = blew up (this is what --divergence-threshold catches)")
    beta = default_beta(args)
    lo, hi = 1e-5, 1.0
    for _ in range(args.bisect_iters):
        mid = 0.5 * (lo + hi)
        if run(mid, beta)['diverged']:
            hi = mid
        else:
            lo = mid
    print(f"\n  critical dt at beta={beta:g}, g_penalty={args.g_penalty}: ~{lo:.5f}")
    print(f"  the default dt=0.001 has ~{lo / 0.001:.1f}x margin there")
    return 0


def cmd_bench(args):
    sizes = [tuple(int(v) for v in s.split('-')) for s in args.sizes.split(',')]
    print(f"cost per relaxation, {make_element(args.element).describe()}\n")
    print(f"{'topology':>12} | {'nodes':>5} | {'branches':>8} | {'Euler us/step':>13} | "
          f"{'10k-step phase':>14} | {'equilibrium':>11}")
    print("-" * 80)
    res = []
    for n_in, n_h in sizes:
        args.n_input, args.n_hidden = n_in, n_h
        net = make_net(args)
        x = np.array([1.0 if k % 2 == 0 else 0.0 for k in range(n_in)])
        g = net.conductance(init_weights(net, args.seed, args.w_min, args.w_max))
        bv = net.boundary(x, args.vsig)
        t0 = time.perf_counter()
        net.relax_euler(g, bv, dt=args.dt, max_steps=args.steps, tol=0.0)
        us = (time.perf_counter() - t0) / args.steps * 1e6
        t0 = time.perf_counter()
        net.relax_equilibrium(g, bv)
        eq = time.perf_counter() - t0
        res.append((net.n, us))
        print(f"{f'{n_in}-{n_h}-{n_in}':>12} | {net.n:>5} | {net.n_branches:>8} | {us:>13.1f} | "
              f"{us * 1e-2:>12.2f} s | {eq * 1000:>8.0f} ms")
    if len(res) >= 2:
        n_t = 2 * args.target_input + args.target_hidden
        (n0, t0_), (n1, t1_) = res[0], res[-1]
        slope = np.log(t1_ / t0_) / np.log(n1 / n0) if n1 != n0 else 2.0
        est = t1_ * (n_t / n1) ** slope
        print(f"\n  Euler cost grows ~ nodes^{slope:.1f}; extrapolated "
              f"{args.target_input}-{args.target_hidden}-{args.target_input} ({n_t} nodes): "
              f"~{est:.0f} us/step, ~{est * 2e4 * 1e-6:.0f} s per free+clamped cycle at 10k steps/phase")
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args):
    print("Elements (--element):")
    print("  ohmic      I = g V")
    print("  relu       I = g sign(V) max(|V| - V_th, 0): dead zone, costs up to V_th per hop (--vth)")
    print("  sigmoid    I = g tanh(10 V), historical name; same as tanh with --steepness 10")
    print("  tanh       I = g tanh(k V) (--steepness k); saturated edges block the nudge")
    print("  sinh       I = g V0 sinh(V/V0) (--v0)")
    print("  rectpair   two antiparallel ideal rectifying branches, independent states (--vf)")
    print("  shockley   two antiparallel branches, Shockley diode + filament (--Is, --n)")
    print("\nRules (sim.py train --rule):")
    print("  contrastive   explicit two-snapshot update, dV^2, strong nudge (historical)")
    print("  global-theta  clock-free: windowed Q vs one shared slow theta (historical)")
    print("  ep            equilibrium propagation proper: weak nudge, co-content observable")
    print("                (contrastive and ep need each element to know the phase - an implicit clock)")
    print("\nProtocols (--protocol):")
    print("  exposure      explicit Euler for a fixed step budget (default for contrastive/global-theta)")
    print("  equilibrium   fully relaxed stationary state (default for ep; used by oracle/handbuilt)")
    print("\nNudge: only beta*g_penalty matters. Defaults: 1000 for the historical rules,")
    print("0.1 for ep, 1e-4 (1e-3 with --sym) for ep on shockley.")
    print(f"\nSweepable (sim.py sweep --param): {', '.join(SWEEPABLE)}")
    net = make_net(args)
    print(f"\nDefault network: {args.n_input}-{args.n_hidden}-{args.n_input}, {net.n} nodes, "
          f"{len(net.edges)} edges")
    for name in ('input', 'hidden', 'output'):
        print(f"  {name:<7} nodes: {[int(v) for v in net.groups[name]]}")
    print(f"  penalty pairs: {[(int(a), int(b)) for a, b in zip(net.pp_in, net.pp_out)]}")
    return 0


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------

def add_common(p, steps_default=10000):
    g = p.add_argument_group('network and data')
    g.add_argument('--n-input', type=int, default=4, help='input nodes (default 4)')
    g.add_argument('--n-hidden', type=int, default=3, help='hidden nodes (default 3)')
    g.add_argument('--arch', default='plain', choices=('plain', 'bias', 'dual', 'dual+bias'),
                   help='extra clamped nodes: bias (0 V and V_sig) and/or complementary inputs')
    g.add_argument('--seed', type=int, default=42, help='RNG seed for initial weights')
    g.add_argument('--w-min', type=float, default=0.1, help='min initial state')
    g.add_argument('--w-max', type=float, default=0.9, help='max initial state')
    g.add_argument('--pattern', default='alternating',
                   help='single pattern: alternating | ones | zeros | random | "1,0,1,0"')
    g.add_argument('--dataset', choices=('bars-stripes',), default=None,
                   help='use the whole Bars & Stripes set (N x N with N^2 = --n-input)')
    g = p.add_argument_group('physics')
    g.add_argument('--element', '--iv', dest='element', choices=ELEMENTS, default='relu',
                   help='edge element (default relu)')
    g.add_argument('--vth', type=float, default=0.1, help='relu: threshold, V')
    g.add_argument('--steepness', type=float, default=10.0, help='tanh/sigmoid: k in tanh(kV)')
    g.add_argument('--v0', type=float, default=0.25, help='sinh: V0')
    g.add_argument('--vf', type=float, default=0.0, help='rectpair: forward drop, V')
    g.add_argument('--Is', type=float, default=1e-4, help='shockley: saturation current (g_max*1V units)')
    g.add_argument('--n', type=float, default=1.0, help='shockley: ideality factor')
    g.add_argument('--vsig', type=float, default=1.0, help='signal amplitude, V (a 1 pixel = vsig)')
    g.add_argument('--beta', type=float, default=None,
                   help='penalty strength; only beta*g_penalty matters (default: 100, or weak for ep)')
    g.add_argument('--g-penalty', type=float, default=10.0, help='penalty link conductance')
    g.add_argument('--capacitance', type=float, default=1.0, help='node capacitance')
    g.add_argument('--g-min', type=float, default=0.01, help='conductance at w=0')
    g.add_argument('--g-max', type=float, default=1.0, help='conductance at w=1')
    g = p.add_argument_group('solver')
    g.add_argument('--protocol', choices=('exposure', 'equilibrium'), default=None,
                   help='fixed-exposure Euler or fully relaxed equilibrium (default depends on command/rule)')
    g.add_argument('--dt', type=float, default=0.001, help='Euler step; too large diverges')
    g.add_argument('--tol', type=float, default=1e-10, help='convergence tolerance')
    g.add_argument('--max-steps', type=int, default=steps_default, help='Euler steps per phase')
    g.add_argument('--divergence-threshold', type=float, default=1e6)
    g.add_argument('--no-divergence-guard', action='store_true')


def add_learning(p):
    g = p.add_argument_group('learning')
    g.add_argument('--rule', choices=RULES, default='contrastive', help='plasticity rule (default contrastive)')
    g.add_argument('--cycles', type=int, default=40, help='pattern presentations (default 40)')
    g.add_argument('--epochs', type=int, default=None, help='alternative to --cycles: passes over the data')
    g.add_argument('--order', choices=('shuffle', 'fixed'), default='shuffle',
                   help='pattern order within an epoch')
    g.add_argument('--holdout', type=int, default=0, help='hold out k non-trivial patterns as a test set')
    g.add_argument('--eta', type=float, default=1.05, help='contrastive/global-theta: learning rate')
    g.add_argument('--gamma', type=float, default=0.001, help='contrastive/global-theta: decay')
    g.add_argument('--dt-plasticity', type=float, default=1.0, help='contrastive: update step')
    g.add_argument('--alpha', type=float, default=0.5, help='ep: learning rate')
    g.add_argument('--sym', action='store_true', help='ep: symmetric +beta/-beta nudging')
    g.add_argument('--soft-bounds', action='store_true',
                   help='ep: potentiation ~ (1-w), depression ~ w')
    g.add_argument('--tau-theta', type=float, default=40.0, help='global-theta: shared threshold time constant')
    g.add_argument('--exposure', type=float, default=10.0, help='global-theta: phase duration')
    g.add_argument('--micro-steps', type=int, default=200, help='global-theta: plasticity steps per phase')
    g.add_argument('--window-pts', type=int, default=None, help='global-theta: averaging window (default = micro-steps)')
    g.add_argument('--relax-max-steps', type=int, default=None,
                   help='global-theta: Euler steps per phase (default exposure/dt)')
    g.add_argument('--sign', type=float, default=-1.0, help='global-theta: sign convention')
    g.add_argument('--observable', choices=('quadratic', 'power'), default='quadratic',
                   help='global-theta: local observable')


def main():
    parser = argparse.ArgumentParser(prog='sim.py', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('relax', help='one relaxation of the initial network')
    add_common(p)
    p.add_argument('--show-currents', action='store_true', help='print every branch current')
    p.set_defaults(func=cmd_relax)

    p = sub.add_parser('train', help='a learning run')
    add_common(p); add_learning(p)
    p.add_argument('--log-lines', type=int, default=20, help='progress lines (0 = none)')
    p.add_argument('--show-patterns', action='store_true', help='print every final reconstruction')
    p.add_argument('--plot', metavar='FILE', help='save the learning curve')
    p.add_argument('--json', metavar='FILE', help='save history, final metrics and weights')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('oracle', help='best reconstruction any conductances can reach')
    add_common(p)
    p.add_argument('--restarts', type=int, default=2)
    p.add_argument('--iters', type=int, default=450)
    p.add_argument('--lr', type=float, default=0.03)
    p.add_argument('--show-patterns', action='store_true')
    p.set_defaults(func=cmd_oracle, seed=100)

    p = sub.add_parser('handbuilt', help='hand-wired AND/OR network for Bars & Stripes')
    add_common(p)
    p.add_argument('--pullup', type=float, default=0.2, help='weak pull-up of the AND detectors')
    p.add_argument('--show-patterns', action='store_true')
    p.set_defaults(func=cmd_handbuilt, element='rectpair', dataset='bars-stripes', n_input=9)

    p = sub.add_parser('sweep', help='vary one parameter and tabulate the effect')
    add_common(p); add_learning(p)
    p.add_argument('--param', required=True, help=f"one of: {', '.join(SWEEPABLE)}")
    p.add_argument('--values', required=True, help='comma-separated, e.g. 0,1,10,100')
    p.add_argument('--train', action='store_true', help='a learning run per value instead of one relaxation')
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser('stability', help='map where explicit Euler blows up')
    add_common(p, steps_default=3000)
    p.add_argument('--dts', default='0.001,0.005,0.01,0.05,0.1')
    p.add_argument('--betas', default='1,10,100,1000')
    p.add_argument('--bisect-iters', type=int, default=20)
    p.set_defaults(func=cmd_stability)

    p = sub.add_parser('bench', help='solver cost vs network size')
    add_common(p)
    p.add_argument('--sizes', default='4-3,9-4,16-8,25-8', help='comma list of INPUT-HIDDEN')
    p.add_argument('--steps', type=int, default=500, help='Euler steps to time per size')
    p.add_argument('--target-input', type=int, default=64)
    p.add_argument('--target-hidden', type=int, default=8)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser('info', help='available elements, rules, protocols, default wiring')
    add_common(p)
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
