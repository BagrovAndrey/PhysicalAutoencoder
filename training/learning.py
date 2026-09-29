# training/learning.py

"""
Learning on the vectorized engine (network/equilibrium.py): the plasticity
rules, the training loop, evaluation metrics, the exact-gradient oracle and
the hand-built Bars & Stripes network. sim.py is a thin CLI over this module.

Rules (all act on the branch states w in [0, 1]):

  ContrastiveRule   the repo's explicit two-snapshot rule (training/plasticity.py):
                    dw = -eta (Q_c - Q_f)(1 - w) - gamma w,  Q = dV^2,  strong nudge.
  EPRule            equilibrium propagation proper: weak nudge, and each branch
                    uses its own co-content derivative dPhi/dg as observable:
                    dw = -alpha [dPhi/dg(nudged) - dPhi/dg(free)] / beta'.
                    Optional symmetric (+beta'/-beta') nudging and soft bounds.
  GlobalThetaRule   the clock-free online path (training/trainer.py): each branch
                    averages its own Q over a window, compares it with ONE shared
                    slow threshold theta; no phase label reaches any element.

Contrastive and EP rules take two snapshots and subtract them, i.e. each element
must know which phase it is in (an implicit clock). Only GlobalThetaRule is
clock-free. Keep that in mind when comparing numbers.
"""

import copy

import numpy as np

from network.elements import SmoothThresholdReLU, ThresholdReLU


# ---------------------------------------------------------------- protocol
class Protocol:
    """How a phase is relaxed: 'exposure' = explicit Euler for a fixed step
    budget from the previous state (the repo's historical protocol, usually not
    converged); 'equilibrium' = the stationary state (quasi-static limit)."""

    def __init__(self, kind='exposure', dt=0.001, max_steps=10000, tol=1e-10,
                 divergence_threshold=1e6):
        if kind not in ('exposure', 'equilibrium'):
            raise ValueError(kind)
        self.kind, self.dt, self.max_steps, self.tol = kind, dt, max_steps, tol
        self.divergence_threshold = divergence_threshold

    def relax(self, net, g, bvals, V0=None, beta_gp=0.0):
        if self.kind == 'exposure':
            return net.relax_euler(g, bvals, V0=V0, beta_gp=beta_gp, dt=self.dt,
                                   max_steps=self.max_steps, tol=self.tol,
                                   divergence_threshold=self.divergence_threshold)
        return net.relax_equilibrium(g, bvals, V0=V0, beta_gp=beta_gp)

    def describe(self):
        return (f"exposure (Euler, dt={self.dt}, {self.max_steps} steps/phase)"
                if self.kind == 'exposure' else "equilibrium (fully relaxed)")


def _phase(net, w, x, vsig, protocol, beta_gp, V0=None):
    return protocol.relax(net, net.conductance(w), net.boundary(x, vsig), V0=V0, beta_gp=beta_gp)


# ---------------------------------------------------------------- rules
class ContrastiveRule:
    name = 'contrastive'

    def __init__(self, eta=1.05, gamma=0.001, dt_plasticity=1.0):
        self.eta, self.gamma, self.dt_plasticity = eta, gamma, dt_plasticity

    def reset(self, net):
        pass

    def describe(self):
        return f"contrastive (eta={self.eta}, gamma={self.gamma}, Q = dV^2)"

    def update(self, net, w, x, vsig, protocol, beta_gp):
        free = _phase(net, w, x, vsig, protocol, 0.0)
        if free['diverged']:
            return w, dict(error='free phase diverged')
        cl = _phase(net, w, x, vsig, protocol, beta_gp, V0=free['V_final'])
        if cl['diverged']:
            return w, dict(error='clamped phase diverged')
        dQ = net.drops(cl['V_final']) ** 2 - net.drops(free['V_final']) ** 2
        dw = -self.eta * dQ * (1 - w) - self.gamma * w
        w = np.clip(w + dw * self.dt_plasticity, 0, 1)
        return w, dict(V_free=free['V_final'], free_converged=free['converged'])


class EPRule:
    name = 'ep'

    def __init__(self, alpha=0.5, symmetric=False, soft_bounds=False):
        self.alpha, self.symmetric, self.soft_bounds = alpha, symmetric, soft_bounds

    def reset(self, net):
        pass

    def describe(self):
        return (f"ep (alpha={self.alpha}, observable = dPhi/dg"
                + (", symmetric +-beta" if self.symmetric else "")
                + (", soft bounds" if self.soft_bounds else "") + ")")

    def update(self, net, w, x, vsig, protocol, beta_gp):
        g = net.conductance(w)
        el = net.element
        free = _phase(net, w, x, vsig, protocol, 0.0)
        if free['diverged']:
            return w, dict(error='free phase diverged')
        Vf = free['V_final']
        plus = _phase(net, w, x, vsig, protocol, beta_gp, V0=Vf)
        if plus['diverged']:
            return w, dict(error='nudged phase diverged')
        if self.symmetric:
            minus = _phase(net, w, x, vsig, protocol, -beta_gp, V0=Vf)
            if minus['diverged']:
                return w, dict(error='anti-nudged phase diverged')
            est = (el.dPhi_dg(g, net.drops(plus['V_final']))
                   - el.dPhi_dg(g, net.drops(minus['V_final']))) / (2 * beta_gp)
        else:
            est = (el.dPhi_dg(g, net.drops(plus['V_final']))
                   - el.dPhi_dg(g, net.drops(Vf))) / beta_gp
        step = -self.alpha * est
        if self.soft_bounds:                  # potentiation ~ (1 - w), depression ~ w
            step = np.where(step > 0, step * (1 - w), step * w)
        return np.clip(w + step, 0, 1), dict(V_free=Vf, free_converged=free['converged'])


class GlobalThetaRule:
    """Vectorized port of training/trainer.py (same micro-step order). The
    observable is 'quadratic' (dV^2) or 'power' (dV * I); the historical
    'linear' observable is not supported - for a reciprocal edge its sign
    depends on an arbitrary orientation (see DEVELOPMENT.md)."""
    name = 'global-theta'

    def __init__(self, eta=1.05, gamma=0.001, sign=-1.0, tau_theta=40.0, exposure=10.0,
                 micro_steps=200, window_pts=None, observable='quadratic'):
        if observable not in ('quadratic', 'power'):
            raise ValueError("global-theta observable must be 'quadratic' or 'power'")
        self.eta, self.gamma, self.sign = eta, gamma, sign
        self.tau_theta, self.exposure, self.micro_steps = tau_theta, exposure, micro_steps
        self.window_pts = window_pts or micro_steps
        self.observable = observable

    def describe(self):
        return (f"global-theta (eta={self.eta}, gamma={self.gamma}, tau_theta={self.tau_theta}, "
                f"exposure={self.exposure}, micro_steps={self.micro_steps}, "
                f"window={self.window_pts}, Q={self.observable})")

    def reset(self, net):
        self.hist = np.zeros((net.n_branches, self.window_pts))
        self.running = np.zeros(net.n_branches)
        self.idx = 0
        self.theta = 0.0

    def _Q(self, net, w, D):
        if self.observable == 'quadratic':
            return D * D
        return D * net.element.current(net.conductance(w), D)

    def _evolve(self, net, w, V):
        dt = self.exposure / self.micro_steps
        a = dt / self.tau_theta
        D = net.drops(V)
        for _ in range(self.micro_steps):
            Q = self._Q(net, w, D)
            self.theta = (1 - a) * self.theta + a * float(Q.mean())
            self.running += Q - self.hist[:, self.idx]
            self.hist[:, self.idx] = Q
            self.idx = (self.idx + 1) % self.window_pts
            q_avg = self.running / self.window_pts
            dwdt = self.sign * self.eta * (q_avg - self.theta) * (1 - w) - self.gamma * w
            w = np.clip(w + dt * dwdt, 0, 1)
        return w

    def update(self, net, w, x, vsig, protocol, beta_gp):
        free = _phase(net, w, x, vsig, protocol, 0.0)
        if free['diverged']:
            return w, dict(error='free phase diverged')
        Vf = free['V_final']
        w = self._evolve(net, w, Vf)
        cl = _phase(net, w, x, vsig, protocol, beta_gp, V0=Vf)
        if cl['diverged']:
            return w, dict(error='clamped phase diverged')
        w = self._evolve(net, w, cl['V_final'])
        return w, dict(V_free=Vf, free_converged=free['converged'], theta=self.theta)


# ---------------------------------------------------------------- evaluation
def evaluate(net, w, X, vsig, protocol):
    """Fresh free-phase relaxation (from V=0) of every pattern; outputs in units
    of the signal amplitude."""
    g = net.conductance(w)
    return np.array([protocol.relax(net, g, net.boundary(x, vsig))['V_final'][net.outputs]
                     for x in X]) / vsig


def metrics(outs, X):
    """MSE, bit accuracy, exact-pattern fraction, margin, and the distribution of
    the number of wrong pixels per pattern (threshold 0.5)."""
    X = np.asarray(X)
    wrong = np.sum((outs > 0.5) != (X > 0.5), axis=1)
    hist = np.bincount(wrong, minlength=X.shape[1] + 1) / len(X)
    return dict(mse=float(np.mean((outs - X) ** 2)),
                bits=float(np.mean((outs > 0.5) == (X > 0.5))),
                exact=float(np.mean(wrong == 0)),
                margin=float(np.min(np.abs(outs - 0.5))),
                wrong=wrong, wrong_hist=hist[:int(wrong.max()) + 1])


def holdout_split(X, k, seed):
    """Hold out k non-trivial patterns (never all-zeros/all-ones) as a test set."""
    X = np.asarray(X)
    if k <= 0:
        return X, None
    trivial = np.all(X == X[:, :1], axis=1)
    idx = np.where(~trivial)[0]
    test = np.sort(np.random.default_rng(seed).choice(idx, size=min(k, len(idx)), replace=False))
    return np.delete(X, test, axis=0), X[test]


# ---------------------------------------------------------------- training loop
def train(net, patterns, rule, protocol, w0, cycles, beta_gp, vsig=1.0, order='shuffle',
          seed=0, on_cycle=None):
    """
    Present patterns one per cycle (free + nudged phase, then the rule's update).
    order='shuffle': a fresh permutation every epoch; 'fixed': cyclic order.
    on_cycle(cycle, pattern, mse_free, info, w) is called after each cycle with the
    free-phase error measured BEFORE that cycle's update.
    Returns (w, history of free-phase MSE per cycle, error message or None).
    """
    rng = np.random.default_rng(seed)
    rule.reset(net)
    w = w0.copy()
    P = len(patterns)
    history, perm = [], None
    for c in range(cycles):
        if order == 'fixed':
            p = c % P
        else:
            if c % P == 0:
                perm = rng.permutation(P)
            p = perm[c % P]
        x = patterns[p]
        w, info = rule.update(net, w, x, vsig, protocol, beta_gp)
        if 'error' in info:
            return w, history, info['error']
        if not np.all(np.isfinite(w)):
            return w, history, 'weights went non-finite'
        mse = float(np.mean((info['V_free'][net.outputs] / vsig - x) ** 2))
        history.append(mse)
        if on_cycle:
            on_cycle(c, x, mse, info, w)
    return w, history, None


# ---------------------------------------------------------------- oracle
def oracle(net, X, w0, vsig=1.0, iters=450, lr=0.03):
    """
    Best reconstruction any conductance setting can reach, by projected Adam with
    exact gradients at the free-phase equilibrium. Not a physical learning rule -
    it is the ceiling a rule should be judged against. For the dead-zone ReLU the
    kinks make gradients one-sided, so it optimizes a smoothed surrogate first.
    """
    def adam(model, w, n):
        m = np.zeros_like(w); v = np.zeros_like(w)
        cache = [None] * len(X)
        best = (np.inf, w.copy())
        for t in range(1, n + 1):
            L, gr = model.loss_and_grad(w, X, vsig, cache)
            if L < best[0]:
                best = (L, w.copy())
            m = 0.9 * m + 0.1 * gr
            v = 0.999 * v + 0.001 * gr * gr
            w = np.clip(w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0, 1)
        return best[1]

    if isinstance(net.element, ThresholdReLU):
        w = w0.copy()
        for s in (0.05, 0.02, 0.008):
            model = copy.copy(net)
            model.element = SmoothThresholdReLU(net.element.vth, s)
            w = adam(model, w, max(1, iters // 3))
        return w
    return adam(net, w0.copy(), iters)


# ---------------------------------------------------------------- hand-built network
def handbuilt_bars_stripes(net, N, pullup=0.2):
    """
    Existence proof that a passive RECTIFYING network represents N x N Bars &
    Stripes exactly. Hidden = N row detectors + N column detectors, each an AND
    (= min) of its line's pixels; output pixel (r, c) = OR (= max) of detectors
    row r and column c. Requires a two-branch element, n_input = N^2,
    n_hidden = 2N, arch='plain'.
      AND pull-down: branch input <- hidden (conducts when V_h > V_in), strong
      AND pull-up:   branch hidden <- input, conductance = pullup
      OR:            branch output <- hidden, strong
    """
    if net.element.branches_per_edge != 2:
        raise ValueError("handbuilt needs a rectifying element (rectpair or shockley)")
    n_in = N * N
    if len(net.inputs) != n_in or len(net.groups['hidden']) != 2 * N or getattr(net, 'arch', 'plain') != 'plain':
        raise ValueError(f"handbuilt needs n_input={n_in}, n_hidden={2 * N}, arch=plain")
    inp, hid, out = set(net.inputs), set(net.groups['hidden']), set(net.outputs)
    h0, o0 = int(min(hid)), int(min(out))

    def members(hk):
        return {hk * N + c for c in range(N)} if hk < N else {r * N + (hk - N) for r in range(N)}

    w = np.zeros(net.n_branches)
    for k, (i, j) in enumerate(zip(net.bi, net.bj)):
        if i in inp and j in hid and i in members(j - h0):
            w[k] = 1.0
        elif i in hid and j in inp and j in members(i - h0):
            w[k] = (pullup - net.g_min) / (net.g_max - net.g_min)
        elif i in out and j in hid and (i - o0) in members(j - h0):
            w[k] = 1.0
    return w
