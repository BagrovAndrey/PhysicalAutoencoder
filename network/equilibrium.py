# network/equilibrium.py

"""
Vectorized network engine: relaxation and exact gradients for any element in
network/elements.py. This is what sim.py runs on.

Two relaxation protocols:
  relax_euler        explicit Euler with EXACTLY the semantics of
                     network.dynamics.VoltageDynamics.relax_transient (same dt,
                     step budget, tolerance, divergence guard) - the repo's
                     "fixed exposure" protocol, ~10x faster than the object loop.
  relax_equilibrium  the quasi-static limit the proposal describes: Euler
                     gradient flow as a warm start, then active-set Newton to the
                     stationary state.

The legacy engine (Grid + Memristor objects + VoltageDynamics + Trainer) is kept
for the library API and the regression tests; tests/test_engine.py checks that
both engines agree.
"""

import numpy as np


class Network:
    """
    Nodes 0..n-1; undirected edges expanded into branches by the element
    (1 or 2 per edge). Branch b carries current from bj[b] into bi[b].
    Clamped nodes get their values from boundary(x, vsig), where x is the input
    pattern (entries 0/1) and vsig the signal amplitude in volts.
    """

    def __init__(self, n_nodes, edges, element, clamped, boundary, inputs, outputs,
                 penalty_pairs, g_min=0.01, g_max=1.0, capacitance=1.0, groups=None):
        self.n = n_nodes
        self.edges = np.asarray(edges, dtype=int)
        self.element = element
        self.clamped = np.asarray(clamped, dtype=int)
        self.free = np.setdiff1d(np.arange(n_nodes), self.clamped)
        self.boundary = boundary
        self.inputs = np.asarray(inputs, dtype=int)
        self.outputs = np.asarray(outputs, dtype=int)
        self.pp_in = np.array([a for a, _ in penalty_pairs], dtype=int)
        self.pp_out = np.array([b for _, b in penalty_pairs], dtype=int)
        self.g_min, self.g_max = g_min, g_max
        self.C = np.full(n_nodes, float(capacitance)) if np.isscalar(capacitance) \
            else np.asarray(capacitance, dtype=float)
        self.groups = groups or {}
        a, b = self.edges[:, 0], self.edges[:, 1]
        if element.branches_per_edge == 1:
            self.bi, self.bj = a.copy(), b.copy()
        else:                                   # (a<-b), (b<-a) for every edge
            self.bi = np.stack([a, b], 1).ravel()
            self.bj = np.stack([b, a], 1).ravel()
        self.n_branches = len(self.bi)

    # ------------------------------------------------------------ physics
    def conductance(self, w):
        return self.g_min + (self.g_max - self.g_min) * w

    def drops(self, V):
        return V[self.bj] - V[self.bi]

    def forces(self, V, g, beta_gp=0.0):
        """Net current into every node (Kirchhoff residual), penalty included."""
        I = self.element.current(g, self.drops(V))
        F = (np.bincount(self.bi, weights=I, minlength=self.n)
             - np.bincount(self.bj, weights=I, minlength=self.n))
        if beta_gp and len(self.pp_out):
            p = beta_gp * (V[self.pp_in] - V[self.pp_out])
            np.add.at(F, self.pp_out, p)
            np.add.at(F, self.pp_in, -p)
        return F

    def forces_and_jacobian(self, V, g, beta_gp=0.0):
        I, s = self.element.current_and_slope(g, self.drops(V))
        n = self.n
        F = np.bincount(self.bi, weights=I, minlength=n) - np.bincount(self.bj, weights=I, minlength=n)
        J = np.zeros(n * n)
        bi, bj = self.bi, self.bj
        np.add.at(J, bi * n + bj, s)
        np.add.at(J, bi * n + bi, -s)
        np.add.at(J, bj * n + bi, s)
        np.add.at(J, bj * n + bj, -s)
        J = J.reshape(n, n)
        if beta_gp and len(self.pp_out):
            pi, po = self.pp_in, self.pp_out
            p = beta_gp * (V[pi] - V[po])
            np.add.at(F, po, p)
            np.add.at(F, pi, -p)
            np.add.at(J, (po, po), -beta_gp)
            np.add.at(J, (po, pi), beta_gp)
            np.add.at(J, (pi, pi), -beta_gp)
            np.add.at(J, (pi, po), beta_gp)
        return F, J

    def currents(self, V, g):
        """Branch currents (from bj into bi) - for inspection."""
        return self.element.current(g, self.drops(V))

    def initial_state(self, bvals):
        V = np.zeros(self.n)
        V[self.clamped] = bvals
        return V

    # ------------------------------------------------------------ protocols
    def relax_euler(self, g, bvals, V0=None, beta_gp=0.0, dt=0.001, max_steps=10000,
                    tol=1e-10, divergence_threshold=1e6, record_history=False):
        """Explicit Euler, same semantics as VoltageDynamics.relax_transient."""
        V = self.initial_state(bvals) if V0 is None else V0.copy()
        V[self.clamped] = bvals
        free = self.free
        hist = [V.copy()] if record_history else None

        def done(converged, diverged, n_steps):
            out = dict(V_final=V, converged=converged, diverged=diverged, n_steps=n_steps)
            if record_history:
                out['V_history'] = np.array(hist)
            return out

        if len(free) == 0:
            return done(True, False, 0)
        with np.errstate(over='ignore', invalid='ignore'):
            for step in range(max_steps):
                dV = dt * self.forces(V, g, beta_gp)[free] / self.C[free]
                V[free] += dV
                max_change = np.max(np.abs(dV))
                if record_history:
                    hist.append(V.copy())
                if divergence_threshold is not None and (
                        not np.isfinite(max_change)
                        or np.max(np.abs(V[free])) > divergence_threshold):
                    return done(False, True, step + 1)
                if max_change < tol:
                    return done(True, False, step + 1)
        return done(False, False, max_steps)

    def relax_equilibrium(self, g, bvals, V0=None, beta_gp=0.0, euler_steps=3000,
                          tol=1e-12):
        """Stationary state: Euler gradient flow from V0 (default: all free nodes
        at 0, like the fixed-exposure protocol) then active-set Newton."""
        V = self.initial_state(bvals) if V0 is None else V0.copy()
        V[self.clamped] = bvals
        free = self.free
        if len(free) == 0:
            return dict(V_final=V, converged=True, diverged=False, residual=0.0)
        with np.errstate(over='ignore', invalid='ignore'):
            for s in range(euler_steps):
                F, J = self.forces_and_jacobian(V, g, beta_gp)
                r = F[free] / self.C[free]
                if s % 25 == 0 and np.abs(r).max() < tol:
                    break
                stiff = np.max(np.abs(np.diag(J))[free] / self.C[free]) + 1e-12
                V[free] += min(0.9 / (2 * stiff), 0.5) * r
            for _ in range(100):
                F, J = self.forces_and_jacobian(V, g, beta_gp)
                r = F[free]
                nr = np.abs(r).max()
                if nr < tol:
                    break
                step = -np.linalg.lstsq(J[np.ix_(free, free)], r, rcond=1e-12)[0]
                t, moved = 1.0, False
                for _ in range(40):
                    Vn = V.copy()
                    Vn[free] += t * step
                    if np.abs(self.forces(Vn, g, beta_gp)[free]).max() < nr:
                        V, moved = Vn, True
                        break
                    t *= 0.5
                if not moved:
                    break
        res = float(np.abs(self.forces(V, g, beta_gp)[free]).max())
        finite = bool(np.all(np.isfinite(V)))
        return dict(V_final=V, converged=finite and res < 1e-8, diverged=not finite,
                    residual=res)

    # ------------------------------------------------------------ gradients
    def loss_and_grad(self, w, X, vsig=1.0, V_cache=None, warm_steps=200):
        """
        Reconstruction MSE of normalized outputs (V_out / vsig vs pattern) at the
        free-phase EQUILIBRIUM, and its exact gradient w.r.t. w (implicit function
        theorem). A diagnostic ("oracle"), never a physical learning rule.
        """
        g = self.conductance(w)
        free, out = self.free, self.outputs
        P, no = len(X), len(out)
        L, grad = 0.0, np.zeros(self.n_branches)
        for p, x in enumerate(X):
            warm = V_cache is not None and V_cache[p] is not None
            res = self.relax_equilibrium(g, self.boundary(x, vsig),
                                         V0=V_cache[p] if warm else None,
                                         euler_steps=warm_steps if warm else 3000)
            V = res['V_final']
            if V_cache is not None:
                V_cache[p] = V
            err = V[out] / vsig - x
            L += float((err ** 2).sum()) / (P * no)
            dLdV = np.zeros(self.n)
            dLdV[out] = 2 * err / (P * no) / vsig
            _, J = self.forces_and_jacobian(V, g)
            lam = np.zeros(self.n)
            lam[free] = np.linalg.lstsq(J[np.ix_(free, free)].T, dLdV[free], rcond=1e-12)[0]
            dIdg = self.element.dI_dg(g, self.drops(V))
            grad += -(lam[self.bi] - lam[self.bj]) * dIdg * (self.g_max - self.g_min)
        return L, grad


# ---------------------------------------------------------------- topologies
def autoencoder(n_in, n_hidden, element, arch='plain', g_min=0.01, g_max=1.0,
                capacitance=1.0):
    """
    Bipartite input <-> hidden <-> output autoencoder, node numbering and edge
    order identical to network.builders.build_autoencoder_topology for
    arch='plain'. Optional extra clamped nodes:
      'bias'  two bias nodes held at 0 and 1 V, wired to every hidden and output node
      'dual'  complementary inputs (1 - x), wired to every hidden node
    """
    idx = 0
    inp = np.arange(n_in); idx += n_in
    comp = np.arange(idx, idx + n_in) if 'dual' in arch else np.array([], int); idx += len(comp)
    bias = np.arange(idx, idx + 2) if 'bias' in arch else np.array([], int); idx += len(bias)
    hid = np.arange(idx, idx + n_hidden); idx += n_hidden
    out = np.arange(idx, idx + n_in); idx += n_in
    edges = [(a, h) for a in inp for h in hid]
    edges += [(a, h) for a in comp for h in hid]
    edges += [(h, o) for h in hid for o in out]
    edges += [(b, t) for b in bias for t in np.concatenate([hid, out])]
    clamped = np.concatenate([inp, comp, bias]).astype(int)

    def boundary(x, vsig=1.0, _dual=len(comp) > 0, _bias=len(bias) > 0):
        """Clamped-node voltages for pattern x (entries 0/1) at signal amplitude vsig."""
        x = np.asarray(x, dtype=float)
        vals = [x * vsig]
        if _dual:
            vals.append((1.0 - x) * vsig)
        if _bias:
            vals.append(np.array([0.0, vsig]))
        return np.concatenate(vals)

    net = Network(idx, edges, element, clamped, boundary, inp, out,
                  [(int(inp[k]), int(out[k])) for k in range(n_in)],
                  g_min=g_min, g_max=g_max, capacitance=capacitance,
                  groups=dict(input=inp, complement=comp, bias=bias, hidden=hid, output=out))
    net.arch = arch
    return net


def init_weights(net, seed=42, w_min=0.1, w_max=0.9):
    """Uniform initial states, one per branch, drawn in branch order. For one-branch
    elements this reproduces build_autoencoder_topology(seed) exactly."""
    return np.random.default_rng(seed).uniform(w_min, w_max, net.n_branches)
