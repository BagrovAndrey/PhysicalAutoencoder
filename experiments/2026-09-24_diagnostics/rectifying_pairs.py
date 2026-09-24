"""
Rectifying antiparallel pairs: each edge {a,b} carries TWO adaptive branches,
  K[a,b]: conducts current b -> a when V_b > V_a (+ optional forward drop vth)
  K[b,a]: conducts current a -> b when V_a > V_b
Current into node i:  F_i = sum_j K[i,j] r(V_j - V_i) - K[j,i] r(V_i - V_j),
r(x) = max(x - vth, 0). K symmetric -> ohmic (vth=0). Still passive
(max principle holds), but can compute min/max, i.e. AND/OR - which is
exactly the structure of Bars & Stripes. EP-compatible: E = sum K Phi(dV),
Phi(x) = 0.5 r(x)^2, local observable per branch = its own co-content.

E4  oracle ceiling for rect-pairs (vth = 0 and 0.05) on 1010 / BS2 / BS3 / BS4
E5  hand-built 2N-hidden diode-logic network for BS3 (existence proof)
E6  generalization: train on a subset of BS3, test on held-out patterns
E7  can proper local EP (weak nudge) learn it? BS2 and BS3
"""
import sys, time, json
from multiprocessing import Pool
import numpy as np
from symmetric_elements import bars_stripes

G_MIN, G_MAX = 0.01, 1.0


def make_net(n_in, n_h):
    inp = np.arange(n_in); hid = np.arange(n_in, n_in + n_h)
    out = np.arange(n_in + n_h, 2 * n_in + n_h); n = 2 * n_in + n_h
    und = [(a, h) for a in inp for h in hid] + [(h, o) for h in hid for o in out]
    br = []                       # directed branches (i, j): K[i, j]
    for a, b in und:
        br.append((a, b)); br.append((b, a))
    return dict(n=n, inp=inp, hid=hid, out=out, br=np.array(br),
                free=np.setdiff1d(np.arange(n), inp))


def Kmat(net, w):
    K = np.zeros((net['n'], net['n']))
    i, j = net['br'].T
    K[i, j] = G_MIN + (G_MAX - G_MIN) * w
    return K


def r(x, vth):  return np.maximum(x - vth, 0.0)
def H(x, vth):  return (x > vth).astype(float)


def F_of(K, V, vth, net, beta_gp=0.0, x=None):
    D = V[None, :] - V[:, None]            # D[i,j] = V_j - V_i
    F = (K * r(D, vth)).sum(1) - (K.T * r(-D, vth)).sum(1)
    if beta_gp:
        F[net['out']] += beta_gp * (x - V[net['out']])
    return F


def Wmat(K, V, vth):
    D = V[None, :] - V[:, None]
    return K * H(D, vth) + K.T * H(-D, vth)


def solve(net, K, x, vth, V0=None, beta_gp=0.0, euler_steps=3000):
    free = net['free']
    V = np.zeros(net['n']) if V0 is None else V0.copy()
    V[net['inp']] = x
    dt = 0.9 / (2 * (K + K.T).sum(1).max() + abs(beta_gp))
    for s in range(euler_steps):
        F = F_of(K, V, vth, net, beta_gp, x)
        V[free] += dt * F[free]
        if s % 50 == 0 and np.abs(F[free]).max() < 1e-10:
            break
    for _ in range(80):
        res = F_of(K, V, vth, net, beta_gp, x)[free]
        nr = np.abs(res).max()
        if nr < 1e-12:
            break
        W = Wmat(K, V, vth)
        J = W - np.diag(W.sum(1))
        if beta_gp:
            J[net['out'], net['out']] -= beta_gp
        step = -np.linalg.lstsq(J[np.ix_(free, free)], res, rcond=1e-10)[0]
        t, moved = 1.0, False
        for _ in range(30):
            Vn = V.copy(); Vn[free] += t * step
            if np.abs(F_of(K, Vn, vth, net, beta_gp, x)[free]).max() < nr:
                V, moved = Vn, True
                break
            t *= 0.5
        if not moved:
            break
    return V


def loss_grad(net, w, X, vth, Vc=None):
    K = Kmat(net, w); free, out = net['free'], net['out']
    i, j = net['br'].T
    P, no = len(X), len(out)
    L, g = 0.0, np.zeros(len(i))
    for p, x in enumerate(X):
        V = solve(net, K, x, vth, V0=None if Vc is None else Vc[p],
                  euler_steps=3000 if Vc is None or Vc[p] is None else 200)
        if Vc is not None:
            Vc[p] = V
        err = V[out] - x
        L += (err ** 2).sum() / (P * no)
        dLdV = np.zeros(net['n']); dLdV[out] = 2 * err / (P * no)
        W = Wmat(K, V, vth); J = W - np.diag(W.sum(1))
        lam = np.zeros(net['n'])
        lam[free] = np.linalg.lstsq(J[np.ix_(free, free)].T, dLdV[free], rcond=1e-10)[0]
        # K[i,j] adds +r(V_j - V_i) to F_i and -r(V_j - V_i) to F_j
        g += -(lam[i] - lam[j]) * r(V[j] - V[i], vth) * (G_MAX - G_MIN)
    return L, g


def adam(net, X, vth, w0, iters=500, lr=0.03):
    w = w0.copy(); m = np.zeros_like(w); v = np.zeros_like(w)
    Vc = [None] * len(X); best = (np.inf, w.copy())
    for t in range(1, iters + 1):
        L, g = loss_grad(net, w, X, vth, Vc)
        if L < best[0]:
            best = (L, w.copy())
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = np.clip(w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0, 1)
    return best


def evaluate(net, w, X, vth):
    K = Kmat(net, w)
    outs = np.array([solve(net, K, x, vth, euler_steps=20000)[net['out']] for x in X])
    return dict(mse=float(np.mean((outs - X) ** 2)),
                bits=float(np.mean((outs > 0.5) == (X > 0.5))),
                exact=float(np.mean(np.all((outs > 0.5) == (X > 0.5), axis=1))),
                margin=float(np.min(np.abs(outs - 0.5))))


def gradcheck():
    X = bars_stripes(2); net = make_net(4, 3)
    w = np.random.default_rng(7).uniform(0.2, 0.8, len(net['br']))
    _, g = loss_grad(net, w, X, 0.0)
    num = np.zeros_like(w); h = 1e-6
    for k in range(len(w)):
        wp = w.copy(); wp[k] += h; wm = w.copy(); wm[k] -= h
        num[k] = (loss_grad(net, wp, X, 0.0)[0] - loss_grad(net, wm, X, 0.0)[0]) / (2 * h)
    return float(np.max(np.abs(g - num)) / np.max(np.abs(num)))


# ---------------------------------------------------------------- jobs
def e4_job(cfg):
    name, N, n_h, vth, restarts = cfg
    X = np.array([[1.0, 0, 1, 0]]) if N == 0 else bars_stripes(N)
    net = make_net(X.shape[1], n_h)
    best = None
    for rs in range(restarts):
        w0 = np.random.default_rng(200 + rs).uniform(0.1, 0.9, len(net['br']))
        _, w = adam(net, X, vth, w0)
        ev = evaluate(net, w, X, vth)
        if best is None or ev['mse'] < best['mse']:
            best = ev
    best.update(name=name, n_h=n_h, vth=vth, P=len(X), n_in=X.shape[1])
    return best


def handbuilt_w(net, N, pullup, gmin):
    """
    Hidden = N row detectors + N column detectors, each an AND (= min) of its
    line's pixels; output pixel (r, c) = OR (= max) of detectors row r, col c.
      AND: strong branch K[input, h] (conducts h -> input when V_h > V_in) pulls
           h down to its lowest member; weak K[h, input] pulls it up when all high.
      OR : strong branch K[out, h] (conducts h -> out when V_h > V_out).
    """
    n_in = N * N
    inp, hid, out = set(net['inp']), set(net['hid']), set(net['out'])

    def members(hk):
        return {hk * N + c for c in range(N)} if hk < N else {r * N + (hk - N) for r in range(N)}

    w = np.zeros(len(net['br']))
    for k, (i, j) in enumerate(net['br']):
        if i in inp and j in hid and i in members(j - n_in):
            w[k] = 1.0                                            # AND pull-down
        elif i in hid and j in inp and j in members(i - n_in):
            w[k] = (pullup - gmin) / (G_MAX - gmin)               # AND weak pull-up
        elif i in out and j in hid and (i - n_in - 2 * N) in members(j - n_in):
            w[k] = 1.0                                            # OR pull-up
    return w


def e5_handbuilt(N=3):
    """
    Existence proof that a PASSIVE rectifying network can represent B&S exactly,
    for a grid of pull-up strengths at realistic (0.01) and near-ideal (1e-4)
    leakage. Then exact MSE-gradient descent STARTED FROM the hand-built network
    (g_min=0.01, pull-up 0.2): does minimizing MSE keep the crisp solution, or
    trade it for a blurrier lower-MSE one?
    """
    global G_MIN
    X = bars_stripes(N)
    net = make_net(N * N, 2 * N)
    results = {}
    for gmin in (0.01, 1e-4):
        for pullup in (0.02, 0.05, 0.2):
            G_MIN = gmin
            results[f"g_min={gmin},pullup={pullup}"] = evaluate(
                net, handbuilt_w(net, N, pullup, gmin), X, 0.0)
    G_MIN = 0.01
    w = handbuilt_w(net, N, 0.2, 0.01)
    m = np.zeros_like(w); v = np.zeros_like(w); Vc = [None] * len(X)
    descent = {}
    for t in range(1, 301):
        _, g = loss_grad(net, w, X, 0.0, Vc)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w = np.clip(w - 0.01 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0, 1)
        if t in (25, 100, 300):
            descent[f"step {t}"] = evaluate(net, w, X, 0.0)
    results['mse_descent_from_handbuilt'] = descent
    return results


def e6_job(cfg):
    N, n_h, held_idx, seed = cfg
    X = bars_stripes(N)
    test = X[held_idx]; train = np.delete(X, held_idx, axis=0)
    net = make_net(X.shape[1], n_h)
    w0 = np.random.default_rng(seed).uniform(0.1, 0.9, len(net['br']))
    _, w = adam(net, train, 0.0, w0, iters=600)
    return dict(N=N, n_h=n_h, held=[int(h) for h in held_idx], seed=seed,
                train=evaluate(net, w, train, 0.0), test=evaluate(net, w, test, 0.0))


def e7_job(cfg):
    N, n_h, alpha, epochs, beta = cfg
    X = bars_stripes(N); net = make_net(X.shape[1], n_h)
    i, j = net['br'].T
    w = np.random.default_rng(1).uniform(0.1, 0.9, len(net['br']))
    Phi = lambda d: 0.5 * np.maximum(d, 0.0) ** 2
    rng = np.random.default_rng(0); hist = []
    for ep in range(epochs):
        for p in rng.permutation(len(X)):
            x = X[p]; K = Kmat(net, w)
            V0 = solve(net, K, x, 0.0, euler_steps=4000)
            Vb = solve(net, K, x, 0.0, V0=V0, beta_gp=beta, euler_steps=4000)
            dQ = Phi(Vb[j] - Vb[i]) - Phi(V0[j] - V0[i])     # each branch's own co-content
            w = np.clip(w - alpha * dQ / beta, 0, 1)
        if ep == 0 or ep % 20 == 19:
            ev = evaluate(net, w, X, 0.0)
            hist.append((ep + 1, ev['mse'], ev['bits'], ev['exact'], ev['margin']))
    return dict(N=N, n_h=n_h, alpha=alpha, beta=beta, hist=hist)


if __name__ == '__main__':
    t0 = time.perf_counter()
    log = dict(gradcheck=gradcheck())
    print(f"gradcheck {log['gradcheck']:.2e}", flush=True)
    log['E5_handbuilt_BS3'] = e5_handbuilt(3)
    print("E5 hand-built BS3:", log['E5_handbuilt_BS3'], flush=True)

    e4 = [('single', 0, 2, 0.0, 2), ('single', 0, 2, 0.05, 2)]
    for vth in (0.0, 0.05):
        e4 += [('BS2', 2, 2, vth, 2), ('BS2', 2, 3, vth, 2),
               ('BS3', 3, 3, vth, 2), ('BS3', 3, 5, vth, 2), ('BS3', 3, 6, vth, 2)]
    e4 += [('BS4', 4, 8, 0.0, 1)]
    rng = np.random.default_rng(5)
    e6 = [(3, 6, np.sort(rng.choice(14, 4, replace=False)), s) for s in range(3)]
    e7 = [(2, 3, 0.5, 100, 0.1), (3, 6, 0.5, 100, 0.1)]

    with Pool(2) as pool:
        a6 = pool.map_async(e6_job, e6)
        a7 = pool.map_async(e7_job, e7)
        res = []
        for rr in pool.imap_unordered(e4_job, e4):
            res.append(rr)
            print(f"[{time.perf_counter()-t0:5.0f}s] E4 {rr['name']:<6} n_h={rr['n_h']} vth={rr['vth']:<4} "
                  f"MSE={rr['mse']:.4f} bits={rr['bits']:.3f} exact={rr['exact']:.2f} "
                  f"margin={rr['margin']:+.3f}", flush=True)
        log['E4'] = res
        log['E6'] = a6.get()
        for e in log['E6']:
            print(f"[{time.perf_counter()-t0:5.0f}s] E6 seed={e['seed']} held={e['held']} "
                  f"train exact={e['train']['exact']:.2f} MSE={e['train']['mse']:.4f} | "
                  f"TEST exact={e['test']['exact']:.2f} MSE={e['test']['mse']:.4f}", flush=True)
        log['E7'] = a7.get()
        for e in log['E7']:
            print(f"[{time.perf_counter()-t0:5.0f}s] E7 BS{e['N']} n_h={e['n_h']}: " +
                  "  ".join(f"ep{h[0]}:MSE={h[1]:.3f},exact={h[3]:.2f}" for h in e['hist']), flush=True)
    log['elapsed_s'] = time.perf_counter() - t0
    json.dump(log, open(sys.argv[1], 'w'), indent=1)
    print("ALL DONE", flush=True)
