#!/usr/bin/env python3

"""
Command-line playground for the MeroCircuit simulator.

Everything the test files do with hardcoded constants, exposed as flags, so
you can turn a knob and immediately see what it does to the physics.

    python3 sim.py relax                      # one electrical relaxation
    python3 sim.py train                      # a learning run
    python3 sim.py sweep --param beta         # one knob, several values
    python3 sim.py stability                  # where explicit Euler blows up
    python3 sim.py bench                      # cost vs network size
    python3 sim.py info                       # what's in the box

Every subcommand takes --help. Defaults match the repo's test suite, so a
bare `python3 sim.py train` reproduces the documented 4->3->4 run.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from grid.grid import Grid
from network.builders import (build_autoencoder_topology, build_memristor_array,
                              build_static_memristor_array, extract_weights, set_weights)
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic, relu_iv, sigmoid_iv, diode_iv
from training.plasticity import SimplePlasticity, compute_Q_from_voltages
from training.rules import (global_threshold_rule, make_rule, quadratic_observable,
                            linear_observable, power_observable)
from training.trainer import Trainer

IV_FUNCS = {'ohmic': ohmic, 'relu': relu_iv, 'sigmoid': sigmoid_iv, 'diode': diode_iv}
OBSERVABLES = {'quadratic': quadratic_observable, 'linear': linear_observable,
               'power': power_observable}


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


def make_network(args, plastic=False, plast_func=None, obs_func=None):
    """Build Grid + memristor array from the common topology flags."""
    adjacency, weights, inp, hid, out, pairs = build_autoencoder_topology(
        args.n_input, args.n_hidden, seed=args.seed, w_min=args.w_min, w_max=args.w_max)
    pattern = parse_pattern(args.pattern, args.n_input, seed=args.seed)
    grid = Grid(adjacency, inp, pattern.copy(), capacitances=args.capacitance)
    iv = IV_FUNCS[args.iv]

    if plastic:
        # window_pts defaults to one full phase, and dt_local follows from
        # the exposure time - these are not independent knobs, see
        # Trainer._check_window_matches_phase.
        window_pts = args.window_pts if args.window_pts else args.micro_steps
        dt_local = args.exposure / args.micro_steps
        memristors = build_memristor_array(
            adjacency, weights, iv_func=iv,
            obs_func=obs_func or quadratic_observable,
            plast_func=plast_func or (lambda Q, w, th: 0.0),
            window_pts=window_pts, dt_local=dt_local,
            g_min=args.g_min, g_max=args.g_max)
    else:
        memristors = build_static_memristor_array(adjacency, weights, iv)

    return dict(adjacency=adjacency, weights=weights, grid=grid, memristors=memristors,
                input_nodes=inp, hidden_nodes=hid, output_nodes=out,
                penalty_pairs=pairs, pattern=pattern,
                solver=VoltageDynamics(grid, memristors))


def fmt(arr, p=4):
    return np.array2string(np.asarray(arr), precision=p, suppress_small=True,
                           max_line_width=200)


def status(result):
    if result.get('diverged'):
        return "DIVERGED"
    return "converged" if result['converged'] else "not converged"


# ---------------------------------------------------------------------------
# relax
# ---------------------------------------------------------------------------

def cmd_relax(args):
    net = make_network(args)
    solver, grid = net['solver'], net['grid']
    pattern, out_nodes = net['pattern'], net['output_nodes']

    V_init = np.zeros(grid.n_nodes)
    V_init[net['input_nodes']] = pattern

    print(f"network : {args.n_input}-{args.n_hidden}-{args.n_input} "
          f"({grid.n_nodes} nodes, {int(net['adjacency'].sum())} directed edges)")
    print(f"I-V     : {args.iv}     capacitance: {args.capacitance}")
    print(f"pattern : {fmt(pattern)}")
    print(f"solver  : dt={args.dt}  tol={args.tol}  max_steps={args.max_steps}  "
          f"beta={args.beta}  g_penalty={args.g_penalty}")
    print()

    t0 = time.perf_counter()
    res = solver.relax_transient(
        V_init, penalty_pairs=net['penalty_pairs'] if args.beta > 0 else None,
        beta=args.beta, g_penalty=args.g_penalty,
        dt=args.dt, max_steps=args.max_steps, tol=args.tol,
        divergence_threshold=None if args.no_divergence_guard else args.divergence_threshold)
    elapsed = time.perf_counter() - t0

    V = res['V_final']
    print(f"result  : {status(res)} in {res['n_steps']} steps ({elapsed*1000:.0f} ms)")
    if res.get('diverged'):
        print()
        print("  The solver blew up. Explicit Euler is only conditionally stable;")
        print(f"  at beta={args.beta}, g_penalty={args.g_penalty} try a smaller --dt")
        print("  (critical dt is ~0.002 at beta=100/g_penalty=10). Run")
        print("  `python3 sim.py stability` to see the map.")
        return 1

    if not res['converged']:
        print("           (not an error: relaxing for a fixed exposure time is the")
        print("            intended protocol - raise --max-steps to reach equilibrium)")
    print()
    print(f"  input   {fmt(V[net['input_nodes']])}")
    print(f"  hidden  {fmt(V[net['hidden_nodes']])}")
    print(f"  output  {fmt(V[out_nodes])}")
    print()
    mse = float(np.mean((V[out_nodes] - pattern) ** 2))
    print(f"  reconstruction MSE : {mse:.6f}")

    # True equilibrium residual: dV/dt at the free nodes, penalty included.
    free = np.setdiff1d(np.arange(grid.n_nodes), grid.clamped_nodes)
    dV_dt = solver._compute_time_derivative(
        V, net['penalty_pairs'] if args.beta > 0 else None, args.beta, args.g_penalty)
    print(f"  max |dV/dt|        : {float(np.abs(dV_dt[free]).max()):.3e}  "
          f"(~0 means settled)")

    # With beta>0 the penalty links inject current, so the memristor currents
    # alone are deliberately NOT balanced - that imbalance is the drive signal.
    I = solver.compute_currents(V)
    mem_only = float(np.abs(I[free].sum(axis=1)).max())
    label = "memristor current imbalance" if args.beta > 0 else "max |net current|"
    note = ("(the penalty injection - this is what drives learning)"
            if args.beta > 0 else "(Kirchhoff residual, ~0 at equilibrium)")
    print(f"  {label:<18} : {mem_only:.3e}  {note}")

    if args.show_currents:
        print("\n  edge currents (from -> to : I):")
        for i in range(grid.n_nodes):
            for j in range(grid.n_nodes):
                if net['adjacency'][i, j] and i < j:
                    print(f"    {j:3d} -> {i:<3d} : {I[i, j]:+.5f}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train_contrastive(args, net, patterns, log):
    """Explicit two-snapshot contrastive rule (training/plasticity.py)."""
    solver, grid, adjacency = net['solver'], net['grid'], net['adjacency']
    memristors, out_nodes = net['memristors'], net['output_nodes']
    weights = net['weights'].copy()
    plasticity = SimplePlasticity(eta=args.eta, gamma=args.gamma,
                                  tau_integrate=args.tau_integrate)

    for cycle in range(args.cycles):
        pattern = patterns[cycle % len(patterns)]
        set_weights(memristors, adjacency, weights)
        grid.clamped_values = pattern.copy()
        V_init = np.zeros(grid.n_nodes)
        V_init[net['input_nodes']] = pattern

        free = solver.relax_transient(V_init, beta=0.0, dt=args.dt,
                                      max_steps=args.max_steps, tol=args.tol)
        if free.get('diverged'):
            return None, "free phase diverged"
        Q_free = compute_Q_from_voltages(free['V_final'], adjacency)

        clamped = solver.relax_transient(
            free['V_final'], penalty_pairs=net['penalty_pairs'],
            beta=args.beta, g_penalty=args.g_penalty,
            dt=args.dt, max_steps=args.max_steps, tol=args.tol)
        if clamped.get('diverged'):
            return None, "clamped phase diverged"
        Q_clamped = compute_Q_from_voltages(clamped['V_final'], adjacency)

        weights = plasticity.update_weights(weights, Q_free, Q_clamped,
                                            adjacency, dt_plasticity=args.dt_plasticity)
        mse = float(np.mean((free['V_final'][out_nodes] - pattern) ** 2))
        log(cycle, mse, free['V_final'][out_nodes], None, weights[adjacency])

    return weights, None


def train_online(args, net, patterns, log):
    """Online path: local plast_func + the one shared theta (Trainer)."""
    trainer = Trainer(
        net['grid'], net['memristors'], net['penalty_pairs'],
        beta_free=0.0, beta_clamped=args.beta, g_penalty=args.g_penalty,
        exposure_time_free=args.exposure, exposure_time_clamped=args.exposure,
        micro_steps_per_phase=args.micro_steps, dt=args.dt, tol=args.tol,
        tau_theta=args.tau_theta, relax_max_steps=args.relax_max_steps)

    n_free_converged = 0
    for cycle in range(args.cycles):
        pattern = patterns[cycle % len(patterns)]
        res = trainer.run_cycle(pattern)
        n_free_converged += int(res['free_converged'])
        w = extract_weights(net['memristors'], net['adjacency'])[net['adjacency']]
        if not np.all(np.isfinite(w)):
            return None, "weights went non-finite"
        log(cycle, res['mse_free'], res['V_free'][net['output_nodes']], res['theta'], w)

    print(f"\n  free phase reached equilibrium in {n_free_converged}/{args.cycles} cycles")
    if n_free_converged == 0:
        print("    (relaxation is cut off by the exposure-time budget - that is the")
        print("     intended protocol, but raise --relax-max-steps to see the")
        print("     difference between it and a fully settled network)")
    return extract_weights(net['memristors'], net['adjacency']), None


def cmd_train(args):
    obs = OBSERVABLES[args.observable]
    if args.rule == 'contrastive':
        net = make_network(args, plastic=True, obs_func=obs)
    else:
        rule = make_rule(global_threshold_rule, eta=args.eta, gamma=args.gamma,
                         sign=args.sign)
        net = make_network(args, plastic=True, plast_func=rule, obs_func=obs)

    # patterns: one fixed vector, or the whole Bars & Stripes set
    if args.dataset == 'bars-stripes':
        from datasets.bars_stripes import BarsAndStripes
        N = int(round(args.n_input ** 0.5))
        if N * N != args.n_input:
            raise SystemExit(f"--dataset bars-stripes needs a square n_input "
                             f"(4, 9, 16...), got {args.n_input}")
        patterns = list(BarsAndStripes(N=N).get_all_flattened())
    else:
        patterns = [net['pattern']]

    print(f"network : {args.n_input}-{args.n_hidden}-{args.n_input}   rule: {args.rule}"
          f"   observable: {args.observable}")
    print(f"learning: eta={args.eta} gamma={args.gamma}"
          + (f" tau_theta={args.tau_theta} exposure={args.exposure}"
             if args.rule != 'contrastive' else f" tau_integrate={args.tau_integrate}"))
    print(f"physics : beta={args.beta} g_penalty={args.g_penalty} dt={args.dt} iv={args.iv}")
    print(f"patterns: {len(patterns)} "
          f"({'Bars & Stripes' if args.dataset else fmt(patterns[0])})")
    print(f"cycles  : {args.cycles}")
    print()

    history = []
    every = max(1, args.cycles // args.log_lines) if args.log_lines else 1

    def log(cycle, mse, out, theta, w):
        history.append(mse)
        if cycle == 0 or (cycle + 1) % every == 0 or cycle == args.cycles - 1:
            line = f"  cycle {cycle+1:>4}/{args.cycles}  MSE={mse:.6f}  out={fmt(out, 3)}"
            if theta is not None:
                line += f"  theta={theta:.5f}"
            line += f"  w=[{w.min():.3f},{w.max():.3f}]"
            print(line)

    t0 = time.perf_counter()
    runner = train_contrastive if args.rule == 'contrastive' else train_online
    weights, err = runner(args, net, patterns, log)
    elapsed = time.perf_counter() - t0

    if err:
        print(f"\n  ABORTED: {err}")
        print("  Try a smaller --dt, or a smaller --beta / --eta.")
        return 1

    print()
    print(f"  {args.cycles} cycles in {elapsed:.1f}s ({elapsed/args.cycles*1000:.0f} ms/cycle)")
    change = (history[0] - history[-1]) / history[0] * 100
    print(f"  MSE {history[0]:.6f} -> {history[-1]:.6f}"
          f"   (improvement: {change:+.1f}%{'' if change >= 0 else '  <-- got worse'})")
    print(f"  best MSE seen: {min(history):.6f} at cycle {int(np.argmin(history))+1}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, len(history) + 1), history, linewidth=2)
        ax.set_xlabel('cycle'); ax.set_ylabel('reconstruction MSE (free phase)')
        ax.set_title(f"{args.n_input}-{args.n_hidden}-{args.n_input}, rule={args.rule}, "
                     f"eta={args.eta}, beta={args.beta}")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"  plot saved to {args.plot}")
    return 0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

SWEEPABLE = ('beta', 'g_penalty', 'dt', 'capacitance', 'eta', 'gamma',
             'tau_theta', 'n_hidden', 'window_pts', 'exposure', 'w_min', 'seed')


def cmd_sweep(args):
    if args.param not in SWEEPABLE:
        raise SystemExit(f"--param must be one of: {', '.join(SWEEPABLE)}")

    raw = [v.strip() for v in args.values.split(',')]
    cast = int if args.param in ('n_hidden', 'window_pts', 'seed') else float
    values = [cast(v) for v in raw]

    mode = 'train' if args.train else 'relax'
    print(f"sweeping {args.param} over {values}   (mode: {mode})")
    print()

    if mode == 'relax':
        header = f"{args.param:>12} | {'status':<12} | {'steps':>7} | {'MSE':>10} | {'output':<28}"
    else:
        header = (f"{args.param:>12} | {'MSE start':>10} | {'MSE end':>10} | "
                  f"{'best':>10} | {'change':>8}")
    print(header)
    print("-" * len(header))

    for value in values:
        setattr(args, args.param, value)
        if mode == 'relax':
            net = make_network(args)
            V_init = np.zeros(net['grid'].n_nodes)
            V_init[net['input_nodes']] = net['pattern']
            res = net['solver'].relax_transient(
                V_init, penalty_pairs=net['penalty_pairs'] if args.beta > 0 else None,
                beta=args.beta, g_penalty=args.g_penalty, dt=args.dt,
                max_steps=args.max_steps, tol=args.tol)
            if res.get('diverged'):
                print(f"{value:>12} | {'DIVERGED':<12} | {res['n_steps']:>7} | "
                      f"{'-':>10} | {'-':<28}")
                continue
            V = res['V_final']
            mse = float(np.mean((V[net['output_nodes']] - net['pattern']) ** 2))
            print(f"{value:>12} | {status(res):<12} | {res['n_steps']:>7} | "
                  f"{mse:>10.6f} | {fmt(V[net['output_nodes']], 3):<28}")
        else:
            obs = OBSERVABLES[args.observable]
            if args.rule == 'contrastive':
                net = make_network(args, plastic=True, obs_func=obs)
            else:
                net = make_network(args, plastic=True, obs_func=obs,
                                   plast_func=make_rule(global_threshold_rule,
                                                        eta=args.eta, gamma=args.gamma,
                                                        sign=args.sign))
            history = []
            runner = train_contrastive if args.rule == 'contrastive' else train_online
            _, err = runner(args, net, [net['pattern']],
                            lambda c, m, o, t, w: history.append(m))
            if err:
                print(f"{value:>12} | {'ABORTED: ' + err:<44}")
                continue
            change = (history[0] - history[-1]) / history[0] * 100
            print(f"{value:>12} | {history[0]:>10.6f} | {history[-1]:>10.6f} | "
                  f"{min(history):>10.6f} | {change:>+7.1f}%")
    return 0


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------

def cmd_stability(args):
    dts = [float(x) for x in args.dts.split(',')]
    betas = [float(x) for x in args.betas.split(',')]

    print(f"stability map: {args.iv}, g_penalty={args.g_penalty}, "
          f"{args.n_input}-{args.n_hidden}-{args.n_input}, max_steps={args.max_steps}")
    print()
    print(f"{'dt':>10} | " + " | ".join(f"beta={b:<9g}" for b in betas))
    print("-" * (12 + 15 * len(betas)))

    def run(dt, beta):
        net = make_network(args)
        V_init = np.zeros(net['grid'].n_nodes)
        V_init[net['input_nodes']] = net['pattern']
        return net['solver'].relax_transient(
            V_init, penalty_pairs=net['penalty_pairs'], beta=beta,
            g_penalty=args.g_penalty, dt=dt, max_steps=args.max_steps, tol=args.tol)

    for dt in dts:
        cells = []
        for beta in betas:
            r = run(dt, beta)
            cells.append("DIVERGES" if r.get('diverged')
                         else (f"ok {r['n_steps']}" if r['converged'] else "slow"))
        print(f"{dt:>10g} | " + " | ".join(f"{c:<14}" for c in cells))

    print()
    print("  ok N     = converged in N steps")
    print("  slow     = finite but not converged within max_steps")
    print("  DIVERGES = blew up (this is what --divergence-threshold catches)")

    # Bisect the critical dt at the requested beta.
    print()
    beta = args.beta
    lo, hi = 1e-5, 1.0
    for _ in range(args.bisect_iters):
        mid = 0.5 * (lo + hi)
        if not run(mid, beta).get('diverged'):
            lo = mid
        else:
            hi = mid
    print(f"  critical dt at beta={beta}, g_penalty={args.g_penalty}: ~{lo:.5f}")
    print(f"  the repo default dt=0.001 has ~{lo/0.001:.1f}x margin there")
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def cmd_bench(args):
    sizes = []
    for spec in args.sizes.split(','):
        a, b = spec.split('-')
        sizes.append((int(a), int(b)))

    print(f"per-step solver cost ({args.iv}, {args.steps} steps each)")
    print()
    print(f"{'topology':>14} | {'nodes':>6} | {'edges':>6} | {'us/step':>9} | {'s/10k steps':>12}")
    print("-" * 62)

    results = []
    for n_in, n_hid in sizes:
        args.n_input, args.n_hidden = n_in, n_hid
        args.pattern = 'alternating'
        net = make_network(args)
        V_init = np.zeros(net['grid'].n_nodes)
        V_init[net['input_nodes']] = net['pattern']
        t0 = time.perf_counter()
        net['solver'].relax_transient(V_init, dt=args.dt, max_steps=args.steps, tol=0.0)
        elapsed = time.perf_counter() - t0
        n_nodes = net['grid'].n_nodes
        per_step = elapsed / args.steps * 1e6
        results.append((n_nodes, per_step))
        print(f"{f'{n_in}-{n_hid}-{n_in}':>14} | {n_nodes:>6} | "
              f"{int(net['adjacency'].sum()):>6} | {per_step:>9.1f} | "
              f"{per_step*1e-6*10000:>12.2f}")

    if len(results) >= 2:
        (n0, t0_us), (n1, t1_us) = results[0], results[-1]
        print()
        print(f"  scaling: {t1_us/t0_us:.1f}x cost for {n1/n0:.1f}x nodes "
              f"(n^2 would predict {(n1/n0)**2:.1f}x)")
        n_t = args.target_input * 2 + args.target_hidden
        est = t1_us * (n_t / n1) ** 2
        cycle_s = est * 1e-6 * 2 * args.target_steps
        print(f"  extrapolated {args.target_input}-{args.target_hidden}-{args.target_input} "
              f"({n_t} nodes): ~{est/1000:.1f} ms/step")
        print(f"    ~{cycle_s:.0f} s per free+clamped cycle at {args.target_steps} steps/phase")
        print(f"    ~{cycle_s*300/3600:.1f} h for a 300-cycle run")
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args):
    print("I-V characteristics (--iv):")
    for name in IV_FUNCS:
        print(f"  {name:<10} {IV_FUNCS[name].__doc__.strip().splitlines()[0]}")
    print("\nLocal observables (--observable):")
    for name, fn in OBSERVABLES.items():
        print(f"  {name:<10} {fn.__doc__.strip().splitlines()[0]}")
    print("\nPlasticity rules (--rule):")
    print("  contrastive    explicit two-snapshot update (training/plasticity.py),")
    print("                 computes (Q_clamped - Q_free) and rewrites all weights at once")
    print("  global-theta   online local rule (training/rules.py) driven through Trainer,")
    print("                 each edge integrates its own Q against one shared theta")
    print("\nSweepable parameters (sim.py sweep --param):")
    print("  " + ", ".join(SWEEPABLE))
    print("\nStability: explicit Euler, critical dt ~0.002 at beta=100/g_penalty=10.")
    print("Run `python3 sim.py stability` to map it for your settings.")

    net = make_network(args)
    print(f"\nDefault network: {args.n_input}-{args.n_hidden}-{args.n_input}, "
          f"{net['grid'].n_nodes} nodes, {int(net['adjacency'].sum())} directed edges")
    print(f"  input nodes  : {[int(x) for x in net['input_nodes']]}")
    print(f"  hidden nodes : {[int(x) for x in net['hidden_nodes']]}")
    print(f"  output nodes : {[int(x) for x in net['output_nodes']]}")
    print(f"  penalty pairs: {net['penalty_pairs']}")
    return 0


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------

def add_topology_args(p):
    g = p.add_argument_group('topology')
    g.add_argument('--n-input', type=int, default=4, help='input nodes (default: 4)')
    g.add_argument('--n-hidden', type=int, default=3, help='hidden nodes (default: 3)')
    g.add_argument('--seed', type=int, default=42, help='RNG seed for initial weights')
    g.add_argument('--w-min', type=float, default=0.1, help='min initial weight')
    g.add_argument('--w-max', type=float, default=0.9, help='max initial weight')
    g.add_argument('--pattern', default='alternating',
                   help='alternating | ones | zeros | random | "1,0,1,0"')


def add_physics_args(p):
    g = p.add_argument_group('physics')
    g.add_argument('--iv', choices=list(IV_FUNCS), default='relu',
                   help='I-V characteristic (default: relu)')
    g.add_argument('--beta', type=float, default=100.0,
                   help='penalty coupling; 0 = free phase (default: 100)')
    g.add_argument('--g-penalty', type=float, default=10.0,
                   help='penalty link conductance (default: 10)')
    g.add_argument('--capacitance', type=float, default=1.0,
                   help='node capacitance; changes speed, not the fixed point')
    g.add_argument('--g-min', type=float, default=0.01, help='conductance at w=0')
    g.add_argument('--g-max', type=float, default=1.0, help='conductance at w=1')


def add_solver_args(p):
    g = p.add_argument_group('solver')
    g.add_argument('--dt', type=float, default=0.001,
                   help='time step; too large diverges (default: 0.001)')
    g.add_argument('--tol', type=float, default=1e-10, help='convergence tolerance')
    g.add_argument('--max-steps', type=int, default=10000, help='max steps per phase')
    g.add_argument('--divergence-threshold', type=float, default=1e6,
                   help='abort if |V| exceeds this (default: 1e6)')
    g.add_argument('--no-divergence-guard', action='store_true',
                   help='disable the guard and let it produce inf/nan')


def add_learning_args(p):
    g = p.add_argument_group('learning')
    g.add_argument('--rule', choices=('contrastive', 'global-theta'),
                   default='contrastive', help='which plasticity path (default: contrastive)')
    g.add_argument('--observable', choices=list(OBSERVABLES), default='quadratic',
                   help='local observable Q (default: quadratic)')
    g.add_argument('--cycles', type=int, default=40, help='learning cycles (default: 40)')
    g.add_argument('--eta', type=float, default=1.05, help='learning rate')
    g.add_argument('--gamma', type=float, default=0.001, help='weight decay')
    g.add_argument('--sign', type=float, default=-1.0,
                   help='sign convention for global-theta rule')
    g.add_argument('--tau-integrate', type=float, default=20.0,
                   help='contrastive: Q integration time constant')
    g.add_argument('--dt-plasticity', type=float, default=1.0,
                   help='contrastive: weight-update step size')
    g.add_argument('--tau-theta', type=float, default=40.0,
                   help='global-theta: time constant of the shared threshold')
    g.add_argument('--exposure', type=float, default=10.0,
                   help='global-theta: exposure time per phase (default: 10; '
                        'shorter cuts the relaxation off far from equilibrium '
                        'and the network stops learning)')
    g.add_argument('--micro-steps', type=int, default=200,
                   help='global-theta: plasticity micro-steps per phase (default: 200; '
                        'below ~100 learning degrades)')
    g.add_argument('--window-pts', type=int, default=None,
                   help='memristor averaging window, in samples. Defaults to '
                        '--micro-steps, i.e. exactly one phase. Changing this '
                        'ratio away from 1.0 stops the network learning - see '
                        'DEVELOPMENT.md, "The window must span one phase".')
    g.add_argument('--relax-max-steps', type=int, default=None,
                   help='relaxation step budget; defaults to exposure/dt')


def main():
    parser = argparse.ArgumentParser(
        prog='sim.py', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('relax', help='run one electrical relaxation and inspect it')
    add_topology_args(p); add_physics_args(p); add_solver_args(p)
    p.add_argument('--show-currents', action='store_true', help='print every edge current')
    p.set_defaults(func=cmd_relax)

    p = sub.add_parser('train', help='run a learning loop')
    add_topology_args(p); add_physics_args(p); add_solver_args(p); add_learning_args(p)
    p.add_argument('--dataset', choices=('bars-stripes',), default=None,
                   help='cycle through a real dataset instead of one fixed pattern')
    p.add_argument('--log-lines', type=int, default=20, help='how many progress lines')
    p.add_argument('--plot', metavar='FILE', help='save an MSE-vs-cycle plot')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('sweep', help='vary one parameter and tabulate the effect')
    add_topology_args(p); add_physics_args(p); add_solver_args(p); add_learning_args(p)
    p.add_argument('--param', required=True, help=f"one of: {', '.join(SWEEPABLE)}")
    p.add_argument('--values', required=True, help='comma-separated, e.g. 0,1,10,100')
    p.add_argument('--train', action='store_true',
                   help='run a learning loop per value instead of one relaxation')
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser('stability', help='map where explicit Euler blows up')
    add_topology_args(p); add_physics_args(p); add_solver_args(p)
    p.add_argument('--dts', default='0.001,0.005,0.01,0.05,0.1')
    p.add_argument('--betas', default='1,10,100,1000')
    p.add_argument('--bisect-iters', type=int, default=20)
    p.set_defaults(func=cmd_stability, max_steps=3000)

    p = sub.add_parser('bench', help='measure solver cost vs network size')
    add_topology_args(p); add_physics_args(p); add_solver_args(p)
    p.add_argument('--sizes', default='4-3,9-4,16-8', help='comma list of INPUT-HIDDEN')
    p.add_argument('--steps', type=int, default=300, help='steps to time per size')
    p.add_argument('--target-input', type=int, default=64)
    p.add_argument('--target-hidden', type=int, default=8)
    p.add_argument('--target-steps', type=int, default=10000)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser('info', help='list available options and the default network')
    add_topology_args(p); add_physics_args(p); add_solver_args(p)
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
