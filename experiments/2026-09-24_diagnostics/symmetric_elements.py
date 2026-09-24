"""
Second round of diagnostics (builds on oracle.py findings):

E1  representational ceiling (exact-gradient "oracle", NOT a physical rule) on
    single pattern / 2x2 B&S / 3x3 B&S, for three transport curves:
      ohmic                          - linear, exact gradients
      relu  (V_th=0.1, dead zone)    - optimized on a smoothed surrogate,
                                       EVALUATED with the exact dead-zone curve
      sinh  (I = g*V0*sinh(V/V0))    - smooth, strictly monotone, no dead zone
    and four architectures: plain / +bias nodes / +complementary inputs / both.
E2  EP gradient estimate vs true gradient (ohmic, where EP theory is clean):
    nudge strength, symmetric nudging, and the repo's actual update direction.
E3  can a physical local rule reach the ceiling on 2x2 B&S?
      repo rule   : dw = -1.05 (dV_c^2 - dV_f^2)(1-w) - 0.001 w,  beta*g_p = 1000
      EP-proper   : dw = -a (Phi(dV_b) - Phi(dV_0)) / beta',       beta' = 0.1
      EP-symmetric: dw = -a (Phi(dV_+b) - Phi(dV_-b)) / (2 beta'), beta' = 0.1
"""
import sys, time, json
from multiprocessing import Pool
import numpy as np
from scipy.special import expit

G_MIN, G_MAX = 0.01, 1.0


# ------------------------------------------------------------------ I-V curves
def make_iv(kind, vth=0.1, s=0.02, v0=0.25):
    if kind == 'ohmic':
        return (lambda x: x, lambda x: np.ones_like(x), lambda x: 0.5 * x * x)
    if kind == 'relu':
        f = lambda x: np.sign(x) * np.maximum(np.abs(x) - vth, 0.0)
        fp = lambda x: (np.abs(x) > vth).astype(float)
        Phi = lambda x: 0.5 * np.maximum(np.abs(x) - vth, 0.0) ** 2
        return f, fp, Phi
    if kind == 'srelu':     # smooth, odd surrogate of the dead-zone relu
        sp = lambda z: s * np.logaddexp(0.0, z / s)
        f = lambda x: sp(x - vth) - sp(-x - vth)
        fp = lambda x: expit((x - vth) / s) + expit((-x - vth) / s)
        return f, fp, None
    if kind == 'sinh':
        return (lambda x: v0 * np.sinh(x / v0), lambda x: np.cosh(x / v0),
                lambda x: v0 * v0 * (np.cosh(x / v0) - 1.0))
    raise ValueError(kind)


# ------------------------------------------------------------------ data / nets
def bars_stripes(N):
    pats = set()
    for bits in range(2 ** N):
        row = np.array([(bits >> k) & 1 for k in range(N)], float)
        pats.add(tuple(np.repeat(row[:, None], N, 1).ravel()))
        pats.add(tuple(np.repeat(row[None, :], N, 0).ravel()))
    return np.array(sorted(pats))


def make_arch(n_in, n_h, arch):
    idx = 0
    inp = np.arange(n_in); idx += n_in
    comp = np.arange(idx, idx + n_in) if 'dual' in arch else np.array([], int); idx += len(comp)
    bias = np.arange(idx, idx + 2) if 'bias' in arch else np.array([], int); idx += len(bias)
    hid = np.arange(idx, idx + n_h); idx += n_h
    out = np.arange(idx, idx + n_in); idx += n_in
    edges = [(a, h) for a in np.concatenate([inp, comp]) for h in hid]
    edges += [(h, o) for h in hid for o in out]
    edges += [(b, t) for b in bias for t in np.concatenate([hid, out])]
    clamped = np.concatenate([inp, comp, bias]).astype(int)
    return dict(n=idx, inp=inp, comp=comp, bias=bias, hid=hid, out=out,
                edges=np.array(edges), clamped=clamped,
                free=np.setdiff1d(np.arange(idx), clamped))


def boundary(net, x):
    v = [x]
    if len(net['comp']): v.append(1 - x)
    if len(net['bias']): v.append(np.array([0.0, 1.0]))
    return np.concatenate(v)


def Gmat(net, w):
    G = np.zeros((net['n'], net['n']))
    a, b = net['edges'].T
    g = G_MIN + (G_MAX - G_MIN) * w
    G[a, b] = g; G[b, a] = g
    return G


def solve(net, G, x, iv, V0=None, beta_gp=0.0, euler_steps=3000):
    f, fp, _ = iv
    free, out, inp = net['free'], net['out'], net['inp']
    V = np.zeros(net['n']) if V0 is None else V0.copy()
    V[net['clamped']] = boundary(net, x)

    def F_of(V):
        F = (G * f(V[None, :] - V[:, None])).sum(1)
        if beta_gp:
            F[out] += beta_gp * (V[inp] - V[out])
        return F

    for s in range(euler_steps):
        D = V[None, :] - V[:, None]
        dt = 0.9 / (2 * (G * fp(D)).sum(1).max() + abs(beta_gp) + 1e-12)
        F = (G * f(D)).sum(1)
        if beta_gp:
            F[out] += beta_gp * (V[inp] - V[out])
        V[free] += min(dt, 0.5) * F[free]
        if s % 50 == 0 and np.abs(F[free]).max() < 1e-10:
            break
    for _ in range(80):
        r = F_of(V)[free]
        nr = np.abs(r).max()
        if nr < 1e-12:
            break
        W = G * fp(V[None, :] - V[:, None])
        J = W - np.diag(W.sum(1))
        if beta_gp:
            J[out, out] -= beta_gp
        step = -np.linalg.lstsq(J[np.ix_(free, free)], r, rcond=1e-10)[0]
        t, moved = 1.0, False
        for _ in range(30):
            Vn = V.copy(); Vn[free] += t * step
            if np.abs(F_of(Vn)[free]).max() < nr:
                V, moved = Vn, True
                break
            t *= 0.5
        if not moved:
            break
    return V


def loss_grad(net, w, X, iv, Vcache=None):
    f, fp, _ = iv
    G = Gmat(net, w)
    free, out = net['free'], net['out']
    a, b = net['edges'].T
    P, no = len(X), len(out)
    L, grad = 0.0, np.zeros(len(a))
    for p, x in enumerate(X):
        V0 = None if Vcache is None else Vcache[p]
        V = solve(net, G, x, iv, V0=V0, euler_steps=3000 if V0 is None else 200)
        if Vcache is not None:
            Vcache[p] = V
        err = V[out] - x
        L += (err ** 2).sum() / (P * no)
        dLdV = np.zeros(net['n']); dLdV[out] = 2 * err / (P * no)
        W = G * fp(V[None, :] - V[:, None])
        J = W - np.diag(W.sum(1))
        lam = np.zeros(net['n'])
        lam[free] = np.linalg.lstsq(J[np.ix_(free, free)].T, dLdV[free], rcond=1e-10)[0]
        fab = f(V[b] - V[a])
        grad += -(lam[a] - lam[b]) * fab * (G_MAX - G_MIN)
    return L, grad


def adam(net, X, iv, w0, iters, lr=0.03):
    w = w0.copy(); m = np.zeros_like(w); v = np.zeros_like(w)
    Vc = [None] * len(X); best = (np.inf, w.copy())
    for t in range(1, iters + 1):
        L, g = loss_grad(net, w, X, iv, Vc)
        if L < best[0]:
            best = (L, w.copy())
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = np.clip(w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0, 1)
    return best


def evaluate(net, w, X, iv):
    G = Gmat(net, w)
    outs = np.array([solve(net, G, x, iv, euler_steps=20000)[net['out']] for x in X])
    return dict(mse=float(np.mean((outs - X) ** 2)),
                bits=float(np.mean((outs > 0.5) == (X > 0.5))),
                exact=float(np.mean(np.all((outs > 0.5) == (X > 0.5), axis=1))),
                margin=float(np.min(np.abs(outs - 0.5))),
                outs=np.round(outs, 3).tolist())


# ------------------------------------------------------------------ E1
def e1_job(cfg):
    name, X, n_h, kind, arch, restarts = cfg
    net = make_arch(X.shape[1], n_h, arch)
    best = None
    for r in range(restarts):
        w = np.random.default_rng(100 + r).uniform(0.1, 0.9, len(net['edges']))
        if kind == 'relu':
            for s, it in ((0.05, 150), (0.02, 150), (0.008, 150)):
                _, w = adam(net, X, make_iv('srelu', s=s), w, it, lr=0.02)
        else:
            _, w = adam(net, X, make_iv(kind), w, 400)
        ev = evaluate(net, w, X, make_iv(kind))
        if best is None or ev['mse'] < best['mse']:
            best = ev
    best.update(name=name, n_h=n_h, iv=kind, arch=arch, P=len(X), n_in=X.shape[1])
    if len(X) > 2:
        best.pop('outs')
    return best


# ------------------------------------------------------------------ E2
def repo_init(net):
    rng = np.random.default_rng(42)
    return np.array([rng.uniform(0.1, 0.9) for _ in net['edges']])


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def e2():
    iv = make_iv('ohmic'); _, _, Phi = iv
    out = {}
    for name, X in (('single', np.array([[1.0, 0, 1, 0]])), ('BS2', bars_stripes(2))):
        net = make_arch(4, 3, 'plain'); w = repo_init(net)
        a, b = net['edges'].T
        _, g = loss_grad(net, w, X, iv)
        G = Gmat(net, w)
        rows = []
        for bg in (1000.0, 100.0, 10.0, 1.0, 0.1, 0.01):
            one = np.zeros(len(a)); sym = np.zeros(len(a)); repo = np.zeros(len(a))
            for x in X:
                V0 = solve(net, G, x, iv)
                Vp = solve(net, G, x, iv, V0=V0, beta_gp=bg)
                Vm = solve(net, G, x, iv, V0=V0, beta_gp=-bg) if bg <= 1.0 else None
                one += (Phi(Vp[a] - Vp[b]) - Phi(V0[a] - V0[b])) / bg
                if Vm is not None:
                    sym += (Phi(Vp[a] - Vp[b]) - Phi(Vm[a] - Vm[b])) / (2 * bg)
                repo += -((Vp[a] - Vp[b]) ** 2 - (V0[a] - V0[b]) ** 2) * (1 - w)
            rows.append(dict(beta_gp=bg, cos_onesided=cos(one, g),
                             cos_symmetric=cos(sym, g) if bg <= 1.0 else None,
                             cos_repo_update_vs_descent=cos(repo, -g),
                             norm_ratio_onesided=float(np.linalg.norm(one) / np.linalg.norm(g) * (2 / (len(X) * 4)) ** -1)))
        out[name] = rows
    return out


# ------------------------------------------------------------------ E3
def e3_job(cfg):
    rule, kind, alpha, epochs = cfg
    X = bars_stripes(2)
    net = make_arch(4, 3, 'plain'); w = repo_init(net)
    iv = make_iv(kind); _, _, Phi = iv
    a, b = net['edges'].T
    rng = np.random.default_rng(0)
    hist = []
    for ep in range(epochs):
        for p in rng.permutation(len(X)):
            x = X[p]; G = Gmat(net, w)
            V0 = solve(net, G, x, iv, euler_steps=4000)
            if rule == 'repo':
                Vc = solve(net, G, x, iv, V0=V0, beta_gp=1000.0, euler_steps=4000)
                dQ = (Vc[a] - Vc[b]) ** 2 - (V0[a] - V0[b]) ** 2
                w = np.clip(w + (-1.05 * dQ * (1 - w) - 0.001 * w), 0, 1)
            elif rule == 'ep':
                Vb = solve(net, G, x, iv, V0=V0, beta_gp=0.1, euler_steps=4000)
                w = np.clip(w - alpha * (Phi(Vb[a] - Vb[b]) - Phi(V0[a] - V0[b])) / 0.1, 0, 1)
            elif rule == 'ep_sym':
                Vp = solve(net, G, x, iv, V0=V0, beta_gp=0.1, euler_steps=4000)
                Vm = solve(net, G, x, iv, V0=V0, beta_gp=-0.1, euler_steps=4000)
                w = np.clip(w - alpha * (Phi(Vp[a] - Vp[b]) - Phi(Vm[a] - Vm[b])) / 0.2, 0, 1)
        if ep % 10 == 9 or ep == 0:
            ev = evaluate(net, w, X, iv)
            hist.append((ep + 1, ev['mse'], ev['bits'], ev['exact']))
    return dict(rule=rule, iv=kind, alpha=alpha, hist=hist,
                w_at_bounds=float(np.mean((w < 1e-3) | (w > 1 - 1e-3))))


if __name__ == '__main__':
    out_path = sys.argv[1]
    t0 = time.perf_counter()
    X1 = np.array([[1.0, 0.0, 1.0, 0.0]])
    BS2, BS3 = bars_stripes(2), bars_stripes(3)

    e1 = []
    for kind in ('ohmic', 'relu', 'sinh'):
        e1.append(('single', X1, 2, kind, 'plain', 2))
        for arch in ('plain', 'bias', 'dual', 'dual+bias'):
            e1.append(('BS2', BS2, 3, kind, arch, 2))
    for kind in ('ohmic', 'sinh'):
        for n_h in (2, 4, 6):
            e1.append(('BS2', BS2, n_h, kind, 'plain', 2))
        for n_h in (3, 5):
            for arch in ('plain', 'dual+bias'):
                e1.append(('BS3', BS3, n_h, kind, arch, 1))
    e3 = [('repo', 'ohmic', None, 60), ('repo', 'relu', None, 60),
          ('ep', 'ohmic', 0.5, 60), ('ep', 'ohmic', 2.0, 60),
          ('ep_sym', 'ohmic', 2.0, 60), ('ep', 'sinh', 0.5, 60)]

    log = {}
    with Pool(2) as pool:
        r2 = pool.apply_async(e2)
        r3 = pool.map_async(e3_job, e3)
        res = []
        for r in pool.imap_unordered(e1_job, e1):
            res.append(r)
            print(f"[{time.perf_counter()-t0:5.0f}s] E1 {r['name']:<6} {r['arch']:<10} {r['iv']:<6} "
                  f"n_h={r['n_h']}  MSE={r['mse']:.4f} bits={r['bits']:.3f} exact={r['exact']:.2f} "
                  f"margin={r['margin']:+.3f}", flush=True)
        log['E1'] = res
        log['E2'] = r2.get()
        print(f"[{time.perf_counter()-t0:5.0f}s] E2 done", flush=True)
        log['E3'] = r3.get()
        print(f"[{time.perf_counter()-t0:5.0f}s] E3 done", flush=True)
    log['elapsed_s'] = time.perf_counter() - t0
    json.dump(log, open(out_path, 'w'), indent=1)
    print("ALL DONE", flush=True)
