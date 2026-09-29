# tests/test_engine.py

"""
Checks for the vectorized engine that sim.py runs on:
network/elements.py, network/equilibrium.py, training/learning.py.

  1. Agreement with the legacy object engine (Grid + Memristor + VoltageDynamics):
     same initial weights, same Euler trajectory, free and nudged.
  2. Elements: current conservation, slopes, dI/dg and dPhi/dg vs finite differences.
  3. Equilibrium solver: Kirchhoff residual ~0, agrees with long Euler.
  4. Exact gradient (oracle) vs finite differences; EP estimate aligned with it
     in the weak-nudge limit.
  5. Hand-built rectifying network reconstructs 3x3 Bars & Stripes exactly.
  6. Learning: the historical contrastive rule reproduces the legacy Trainer
     number; EP on 2x2 Bars & Stripes reduces the error.

Run: python3 tests/test_engine.py   (about a minute)
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from grid.grid import Grid
from network.builders import build_autoencoder_topology, build_static_memristor_array
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic, relu_iv, sigmoid_iv
from network.elements import make_element, ELEMENTS
from network.equilibrium import autoencoder, init_weights
from training.learning import (ContrastiveRule, EPRule, Protocol, evaluate, metrics,
                               train, handbuilt_bars_stripes)
from datasets.bars_stripes import BarsAndStripes

PASSED, FAILED = 0, 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"    [ok]   {name}  {detail}")
    else:
        FAILED += 1
        print(f"    [FAIL] {name}  {detail}")


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def test_legacy_agreement():
    print("\n[1] Agreement with the legacy engine")
    adj, W, inp, hid, out, pp = build_autoencoder_topology(4, 3, seed=42)
    for name, iv in (('ohmic', ohmic), ('relu', relu_iv), ('sigmoid', sigmoid_iv)):
        net = autoencoder(4, 3, make_element(name))
        w = init_weights(net, 42)
        check(f"{name}: initial weights identical",
              np.allclose(w, [W[a, b] for a, b in net.edges]))
        g = net.conductance(w)
        G = np.zeros_like(W)
        for (a, b), gi in zip(net.edges, g):
            G[a, b] = G[b, a] = gi
        x = np.array([1.0, 0.0, 1.0, 0.0])
        for beta_gp in (0.0, 1000.0):
            grid = Grid(adj, inp, x.copy(), capacitances=1.0)
            solver = VoltageDynamics(grid, build_static_memristor_array(adj, G, iv))
            V0 = np.zeros(len(adj)); V0[inp] = x
            ref = solver.relax_transient(V0, penalty_pairs=pp, beta=beta_gp / 10, g_penalty=10.0,
                                         dt=0.001, max_steps=3000, tol=1e-10)
            new = net.relax_euler(g, net.boundary(x), beta_gp=beta_gp, dt=0.001,
                                  max_steps=3000, tol=1e-10)
            err = np.abs(ref['V_final'] - new['V_final']).max()
            check(f"{name}, beta*g_p={beta_gp:g}: Euler matches VoltageDynamics", err < 1e-12,
                  f"max diff {err:.1e}")


def test_elements():
    print("\n[2] Elements")
    rng = np.random.default_rng(0)
    g = rng.uniform(0.05, 1.0, 200)
    V = rng.uniform(-1.5, 1.5, 200)
    h = 1e-6
    for name in ELEMENTS:
        el = make_element(name)
        I, s = el.current_and_slope(g, V)
        fd_s = (el.current(g, V + h) - el.current(g, V - h)) / (2 * h)
        fd_g = (el.current(g + h, V) - el.current(g - h, V)) / (2 * h)
        # away from kinks for the non-smooth elements
        mask = np.ones_like(V, bool)
        if not el.smooth:
            mask = np.min(np.abs(np.abs(V)[:, None] - np.array([0.0, 0.1])), axis=1) > 1e-3
        check(f"{name}: dI/dV matches finite differences",
              np.allclose(s[mask], fd_s[mask], rtol=1e-4, atol=1e-6))
        check(f"{name}: dI/dg matches finite differences",
              np.allclose(el.dI_dg(g, V)[mask], fd_g[mask], rtol=1e-4, atol=1e-6))
        # dPhi/dg = d/dg integral_0^V I dV'  (trapezoid on a fine grid)
        vv = np.linspace(0, 1, 4001)[:, None] * V[None, :50]
        def phi(gg):
            I_ = el.current(np.broadcast_to(gg[:50], vv.shape), vv)
            return np.sum(0.5 * (I_[1:] + I_[:-1]) * np.diff(vv, axis=0), axis=0)
        fd_phi = (phi(g + 1e-4) - phi(g - 1e-4)) / 2e-4
        check(f"{name}: dPhi/dg is d(co-content)/dg",
              np.allclose(el.dPhi_dg(g[:50], V[:50]), fd_phi, rtol=1e-3, atol=1e-6))
        net = autoencoder(4, 3, el)
        Vn = rng.uniform(-1, 1, net.n)
        check(f"{name}: currents conserve charge",
              abs(net.forces(Vn, net.conductance(init_weights(net))).sum()) < 1e-12)


def test_equilibrium():
    print("\n[3] Equilibrium solver")
    x = np.array([1.0, 0.0, 1.0, 0.0])
    for name in ('ohmic', 'relu', 'tanh', 'rectpair', 'shockley'):
        net = autoencoder(4, 3, make_element(name))
        g = net.conductance(init_weights(net))
        eq = net.relax_equilibrium(g, net.boundary(x), beta_gp=10.0)
        check(f"{name}: equilibrium converges", eq['converged'], f"residual {eq['residual']:.1e}")
        eu = net.relax_euler(g, net.boundary(x), beta_gp=10.0, dt=0.01, max_steps=400000, tol=1e-13)
        d = np.abs(eu['V_final'] - eq['V_final']).max()
        check(f"{name}: agrees with long Euler", d < 1e-6, f"max diff {d:.1e}")


def test_gradients():
    print("\n[4] Exact gradient and the EP estimate")
    X = BarsAndStripes(N=2).get_all_flattened().astype(float)
    for name, vsig, bgp in (('ohmic', 1.0, 0.01), ('tanh', 1.0, 0.01), ('rectpair', 1.0, 0.01),
                            ('shockley', 2.0, 1e-4)):
        el = make_element(name, steepness=2.0)
        net = autoencoder(4, 3, el)
        w = init_weights(net, 1)
        L, gr = net.loss_and_grad(w, X, vsig)
        d = np.random.default_rng(3).standard_normal(len(w)) * 1e-5
        fd = (net.loss_and_grad(w + d, X, vsig)[0] - net.loss_and_grad(w - d, X, vsig)[0]) / 2
        check(f"{name}: implicit-function gradient vs finite differences",
              abs(fd - gr @ d) < 1e-3 * abs(fd) + 1e-14, f"rel err {abs(fd - gr @ d) / abs(fd):.1e}")
        # EP: (1/beta') [dPhi/dg(nudged) - dPhi/dg(free)], summed over patterns
        g = net.conductance(w)
        est = np.zeros_like(w)
        for x in X:
            f = net.relax_equilibrium(g, net.boundary(x, vsig))['V_final']
            p = net.relax_equilibrium(g, net.boundary(x, vsig), V0=f, beta_gp=bgp)['V_final']
            est += (el.dPhi_dg(g, net.drops(p)) - el.dPhi_dg(g, net.drops(f))) / bgp
        c = cos(est, gr)
        check(f"{name}: EP estimate aligned with the gradient (beta*g_p={bgp:g})", c > 0.95,
              f"cos {c:.3f}")


def test_handbuilt():
    print("\n[5] Hand-built rectifying network")
    X = BarsAndStripes(N=3).get_all_flattened().astype(float)
    net = autoencoder(9, 6, make_element('rectpair'))
    w = handbuilt_bars_stripes(net, 3)
    m = metrics(evaluate(net, w, X, 1.0, Protocol('equilibrium')), X)
    check("rectpair reconstructs all 14 3x3 Bars & Stripes patterns", m['exact'] == 1.0,
          f"exact {m['exact']:.2f}")


def test_learning():
    print("\n[6] Learning")
    net = autoencoder(4, 3, make_element('relu'))
    x = np.array([[1.0, 0.0, 1.0, 0.0]])
    w, hist, err = train(net, x, ContrastiveRule(), Protocol('exposure'), init_weights(net), 40,
                         beta_gp=1000.0, order='fixed')
    mse = hist[-1]            # free-phase MSE of the last cycle, as the Trainer reports it
    check("contrastive, 40 cycles: legacy Trainer number 0.043978", err is None and abs(mse - 0.043978) < 1e-6,
          f"{mse:.6f}")

    X = BarsAndStripes(N=2).get_all_flattened().astype(float)
    net = autoencoder(4, 4, make_element('rectpair'))
    w0 = init_weights(net)
    prot = Protocol('equilibrium')
    before = metrics(evaluate(net, w0, X, 1.0, prot), X)['mse']
    w, _, err = train(net, X, EPRule(0.5), prot, w0, 60 * len(X), beta_gp=0.1)
    after = metrics(evaluate(net, w, X, 1.0, prot), X)['mse']
    check("EP on rectifying pairs lowers the 2x2 Bars & Stripes error", err is None and after < 0.5 * before,
          f"{before:.4f} -> {after:.4f}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    test_legacy_agreement()
    test_elements()
    test_equilibrium()
    test_gradients()
    test_handbuilt()
    test_learning()
    print(f"\n{PASSED} passed, {FAILED} failed   ({time.perf_counter() - t0:.1f}s)")
    sys.exit(1 if FAILED else 0)
