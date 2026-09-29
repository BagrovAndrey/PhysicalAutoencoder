#!/usr/bin/env python3
"""
Playground for the 2026-09-24 diagnostic models (research code, not the library;
the library integration is WP3 in docs/SPEC.md). Three modes:

  handbuilt  the AND/OR network for N x N Bars & Stripes (2N hidden units),
             with ideal rectifying pairs or Shockley-diode branches
  oracle     best reconstruction ANY conductance setting can reach
             (exact-gradient Adam; not a physical rule)
  train      the physical local rule: weak-nudge EP, each edge/branch learns
             from its own co-content

Examples:
  python3 play.py handbuilt --data bs3 --show
  python3 play.py handbuilt --data bs4 --element shockley --Is 1e-4 --vsig 1
  python3 play.py oracle --data bs2 --n-hidden 3 --element tanh
  python3 play.py train --data bs3 --n-hidden 6 --epochs 100 --holdout 3

Data: bs2 | bs3 | bs4 | bs5 (Bars & Stripes N x N) or an explicit pattern "1,0,1,0".
Every mode takes --help. Voltages in volts (signal 1 V unless --vsig).
"""
import argparse
import json
import time

import numpy as np

import symmetric_elements as S
import rectifying_pairs as R
import shockley_pairs as SH

SYMMETRIC = ('ohmic', 'relu', 'tanh', 'sinh')


# ------------------------------------------------------------------ helpers
def load_data(spec):
    if spec.startswith('bs') and spec[2:].isdigit():
        return S.bars_stripes(int(spec[2:])), int(spec[2:])
    x = np.array([[float(c) for c in spec.split(',')]])
    n = int(round(np.sqrt(x.shape[1])))
    return x, (n if n * n == x.shape[1] else None)


def tanh_iv(k):
    return (lambda x: np.tanh(k * x), lambda x: k / np.cosh(k * x) ** 2,
            lambda x: np.log(np.cosh(k * x)) / k)


def sym_iv(a):
    if a.element == 'tanh':
        return tanh_iv(a.steepness)
    return S.make_iv(a.element, vth=a.vth, v0=a.v0)


def metrics(outs, X):
    return dict(mse=float(np.mean((outs - X) ** 2)),
                bits=float(np.mean((outs > 0.5) == (X > 0.5))),
                exact=float(np.mean(np.all((outs > 0.5) == (X > 0.5), axis=1))),
                margin=float(np.min(np.abs(outs - 0.5))))


def line(m, prefix=''):
    return (f"{prefix}MSE={m['mse']:.4f}  bits={m['bits']:.3f}  "
            f"exact={m['exact'] * 100:.0f}%  margin={m['margin']:+.3f}")


def show(outs, X, N):
    for x, o in zip(X, outs):
        ok = 'ok ' if np.all((o > 0.5) == (x > 0.5)) else 'BAD'
        if N:
            xi, oi = x.reshape(N, N), o.reshape(N, N)
            rows = [' '.join(str(int(v)) for v in xi[r]) + '   ->   ' +
                    ' '.join(f"{v:4.2f}" for v in oi[r]) for r in range(N)]
            print(f"  [{ok}] " + rows[0])
            for rr in rows[1:]:
                print("        " + rr)
        else:
            print(f"  [{ok}] {x.astype(int)} -> {np.round(o, 3)}")


def rect_outs(net, w, X, vf):
    K = R.Kmat(net, w)
    return np.array([R.solve(net, K, x, vf, euler_steps=20000)[net['out']] for x in X])


def sym_outs(net, w, X, iv):
    G = S.Gmat(net, w)
    return np.array([S.solve(net, G, x, iv, euler_steps=20000)[net['out']] for x in X])


def holdout_split(X, k, seed):
    if k <= 0:
        return X, None
    trivial = np.all(X == X[:, :1], axis=1)             # all-zeros / all-ones
    idx = np.where(~trivial)[0]
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.delete(X, test, axis=0), X[test]


def set_gmin(g):
    S.G_MIN = R.G_MIN = g


# ------------------------------------------------------------------ modes
def cmd_handbuilt(a):
    X, N = load_data(a.data)
    if not N or len(X) < 2:
        raise SystemExit("handbuilt needs a Bars & Stripes dataset: --data bs2|bs3|bs4|bs5")
    set_gmin(a.gmin)
    net = R.make_net(N * N, 2 * N)
    w = R.handbuilt_w(net, N, a.pullup, a.gmin)
    print(f"hand-built {N*N}-{2*N}-{N*N} AND/OR network, {len(X)} patterns, "
          f"g_min={a.gmin}, pull-up={a.pullup}, element={a.element}")
    t0 = time.perf_counter()
    if a.element == 'rectpair':
        print(f"  ideal rectifying pairs, forward drop V_f={a.vf} V")
        outs = rect_outs(net, w, X, a.vf)
    else:
        print(f"  Shockley branches: I_s={a.Is:g}, n={a.n}, signal={a.vsig} V")
        K = np.zeros((net['n'], net['n'])); i, j = net['br'].T
        K[i, j] = a.gmin + (1 - a.gmin) * w
        outs = np.array([SH.solve(net, K, x * a.vsig, a.Is, a.n)[net['out']]
                         for x in X]) / a.vsig
    print(line(metrics(outs, X), '  ') + f"   ({time.perf_counter() - t0:.1f}s)")
    if a.show:
        show(outs, X, N)


def cmd_oracle(a):
    X, N = load_data(a.data)
    set_gmin(a.gmin)
    t0 = time.perf_counter()
    best = None
    for r in range(a.restarts):
        rng = np.random.default_rng(a.seed + r)
        if a.element == 'rectpair':
            net = R.make_net(X.shape[1], a.n_hidden)
            w0 = rng.uniform(0.1, 0.9, len(net['br']))
            _, w = R.adam(net, X, a.vf, w0, iters=a.iters)
            outs = rect_outs(net, w, X, a.vf)
        else:
            net = S.make_arch(X.shape[1], a.n_hidden, a.arch)
            w0 = rng.uniform(0.1, 0.9, len(net['edges']))
            if a.element == 'relu':      # dead zone: optimize a smoothed surrogate
                w = w0
                for s, it in ((0.05, a.iters // 3), (0.02, a.iters // 3), (0.008, a.iters // 3)):
                    _, w = S.adam(net, X, S.make_iv('srelu', vth=a.vth, s=s), w, it, lr=0.02)
            else:
                _, w = S.adam(net, X, sym_iv(a), w0, a.iters)
            outs = sym_outs(net, w, X, sym_iv(a))
        m = metrics(outs, X)
        print(f"  restart {r + 1}/{a.restarts}: " + line(m))
        if best is None or m['mse'] < best[0]['mse']:
            best = (m, outs)
    print(f"oracle ceiling, {a.element}, {X.shape[1]}-{a.n_hidden}-{X.shape[1]}"
          + (f", arch={a.arch}" if a.element != 'rectpair' else '') + ":")
    print(line(best[0], '  ') + f"   ({time.perf_counter() - t0:.1f}s)")
    if a.show:
        show(best[1], X, N)


def cmd_train(a):
    X, N = load_data(a.data)
    set_gmin(a.gmin)
    train, test = holdout_split(X, a.holdout, a.seed)
    rng = np.random.default_rng(a.seed)
    rect = a.element == 'rectpair'
    if rect:
        net = R.make_net(X.shape[1], a.n_hidden)
        i, j = net['br'].T
        w = rng.uniform(0.1, 0.9, len(net['br']))
        Phi = lambda d: 0.5 * np.maximum(d - a.vf, 0.0) ** 2
        relax = lambda K, x, V0=None, b=0.0: R.solve(net, K, x, a.vf, V0=V0, beta_gp=b, euler_steps=4000)
        mat = lambda w: R.Kmat(net, w)
        outs_of = lambda w, Xs: rect_outs(net, w, Xs, a.vf)
        dQ = lambda V1, V2: Phi(V1[j] - V1[i]) - Phi(V2[j] - V2[i])
    else:
        iv = sym_iv(a); Phi = iv[2]
        net = S.make_arch(X.shape[1], a.n_hidden, 'plain')
        ea, eb = net['edges'].T
        w = rng.uniform(0.1, 0.9, len(net['edges']))
        relax = lambda K, x, V0=None, b=0.0: S.solve(net, K, x, iv, V0=V0, beta_gp=b, euler_steps=4000)
        mat = lambda w: S.Gmat(net, w)
        outs_of = lambda w, Xs: sym_outs(net, w, Xs, iv)
        dQ = lambda V1, V2: Phi(V1[ea] - V1[eb]) - Phi(V2[ea] - V2[eb])
    print(f"local weak-nudge EP: {a.element}, {X.shape[1]}-{a.n_hidden}-{X.shape[1]}, "
          f"beta*g_p={a.beta}, alpha={a.alpha}, {'symmetric +-beta, ' if a.sym else ''}"
          f"g_min={a.gmin}, {len(train)} train"
          + (f" / {len(test)} held out (non-trivial)" if test is not None else "") + " patterns")
    hist, t0 = [], time.perf_counter()
    for ep in range(1, a.epochs + 1):
        for p in rng.permutation(len(train)):
            x = train[p]; K = mat(w)
            V0 = relax(K, x)
            Vp = relax(K, x, V0, a.beta)
            if a.sym:
                Vm = relax(K, x, V0, -a.beta)
                w = np.clip(w - a.alpha * dQ(Vp, Vm) / (2 * a.beta), 0, 1)
            else:
                w = np.clip(w - a.alpha * dQ(Vp, V0) / a.beta, 0, 1)
        if ep == 1 or ep % a.eval_every == 0 or ep == a.epochs:
            mt = metrics(outs_of(w, train), train)
            row = dict(epoch=ep, train=mt)
            msg = f"  epoch {ep:4d}  " + line(mt, 'train ')
            if test is not None:
                ms = metrics(outs_of(w, test), test); row['test'] = ms
                msg += f"  | test exact={ms['exact'] * 100:.0f}%"
            print(msg + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
            hist.append(row)
    if a.show:
        print("\nfinal reconstructions (train):"); show(outs_of(w, train), train, N)
        if test is not None:
            print("held out:"); show(outs_of(w, test), test, N)
    if a.json:
        json.dump(dict(args=vars(a), hist=hist, w=w.tolist()), open(a.json, 'w'), indent=1)
        print(f"saved {a.json}")


# ------------------------------------------------------------------ CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(q, element_choices, default_element):
        q.add_argument('--data', default='bs3', help='bs2|bs3|bs4|bs5 or "1,0,1,0" (default bs3)')
        q.add_argument('--element', choices=element_choices, default=default_element)
        q.add_argument('--gmin', type=float, default=0.01, help='conductance at w=0 (g_max=1)')
        q.add_argument('--vf', type=float, default=0.0, help='rectpair: forward drop, V')
        q.add_argument('--vth', type=float, default=0.1, help='relu: dead-zone threshold, V')
        q.add_argument('--steepness', type=float, default=10.0, help='tanh: I = g*tanh(k V)')
        q.add_argument('--v0', type=float, default=0.25, help='sinh: I = g*V0*sinh(V/V0)')
        q.add_argument('--show', action='store_true', help='print every reconstruction')

    q = sub.add_parser('handbuilt', help='AND/OR network for N x N B&S (existence proof)')
    common(q, ('rectpair', 'shockley'), 'rectpair')
    q.add_argument('--pullup', type=float, default=0.2, help='weak pull-up conductance of AND detectors')
    q.add_argument('--Is', type=float, default=1e-4, help='shockley: saturation current (units of g_max*1V)')
    q.add_argument('--n', type=float, default=1.0, help='shockley: ideality factor')
    q.add_argument('--vsig', type=float, default=1.0, help='shockley: signal amplitude, V')
    q.set_defaults(func=cmd_handbuilt)

    q = sub.add_parser('oracle', help='best any conductances can do (exact gradients; not physical)')
    common(q, SYMMETRIC + ('rectpair',), 'rectpair')
    q.add_argument('--n-hidden', type=int, default=6)
    q.add_argument('--arch', default='plain', choices=('plain', 'bias', 'dual', 'dual+bias'),
                   help='symmetric elements only: bias nodes / complementary inputs')
    q.add_argument('--restarts', type=int, default=2)
    q.add_argument('--iters', type=int, default=450)
    q.add_argument('--seed', type=int, default=100)
    q.set_defaults(func=cmd_oracle)

    q = sub.add_parser('train', help='physical local rule: weak-nudge EP on co-content')
    common(q, SYMMETRIC + ('rectpair',), 'rectpair')
    q.add_argument('--n-hidden', type=int, default=6)
    q.add_argument('--epochs', type=int, default=100)
    q.add_argument('--beta', type=float, default=0.1, help='nudge strength beta*g_p (weak!)')
    q.add_argument('--alpha', type=float, default=0.5, help='learning rate')
    q.add_argument('--sym', action='store_true', help='symmetric nudging (+beta/-beta)')
    q.add_argument('--holdout', type=int, default=0, help='hold out k non-trivial patterns as a test set')
    q.add_argument('--eval-every', type=int, default=10)
    q.add_argument('--seed', type=int, default=1)
    q.add_argument('--json', help='save history and final weights')
    q.set_defaults(func=cmd_train)

    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
