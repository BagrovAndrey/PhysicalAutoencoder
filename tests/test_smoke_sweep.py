# tests/test_smoke_sweep.py

"""
Smoke / sanity harness for the simulator as a whole.

This is deliberately NOT a physics-quality benchmark (it says nothing about
whether the plateau is resolved). It answers a narrower question: does the
machinery actually work as a simulator - can you build networks of different
shapes, swap I-V characteristics and plasticity rules, turn the physical
knobs, and get results that are finite, reproducible, and move in the
direction physics says they should?

Grouped as:
  1. Physics invariants   - boundary conditions, Kirchhoff, analytic solution
  2. Solver knobs         - dt, capacitance, beta, g_penalty, I-V curve
  3. Scale                - 4-3-4 up to 16-8-16, with cost extrapolation
  4. Plasticity paths     - explicit contrastive + online global-theta
  5. Reproducibility      - same seed, same numbers
  6. CLI                  - sim.py starts, parses, and agrees with the library

Run with:  python3 tests/test_smoke_sweep.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from grid.grid import Grid
from network.builders import (build_autoencoder_topology, build_memristor_array,
                              build_static_memristor_array, extract_weights, set_weights)
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic, relu_iv, sigmoid_iv, diode_iv
from training.plasticity import SimplePlasticity, compute_Q_from_voltages
from training.rules import global_threshold_rule, make_rule
from training.trainer import Trainer


PASS, FAIL = [], []


def check(name, condition, detail=""):
    """Record a single assertion without aborting the whole run, so one
    failure doesn't hide the state of everything after it."""
    if condition:
        PASS.append(name)
        print(f"    [ok]   {name}" + (f"  {detail}" if detail else ""))
    else:
        FAIL.append((name, detail))
        print(f"    [FAIL] {name}" + (f"  {detail}" if detail else ""))


def quadratic_obs(V_drop, I):
    """Matches training.plasticity.compute_Q_from_voltages (Q = dV^2)."""
    return V_drop ** 2


def build_autoencoder(n_input, n_hidden, seed=42, w_min=0.1, w_max=0.9):
    """Thin wrapper over the library helper, dropping hidden_nodes (which
    most checks here don't need) to keep call sites short."""
    adjacency, weights, inp, _hid, out, pairs = build_autoencoder_topology(
        n_input, n_hidden, seed=seed, w_min=w_min, w_max=w_max)
    return adjacency, weights, inp, out, pairs


def solve_ohmic_analytically(adjacency, conductances, clamped_nodes, clamped_values):
    """
    Steady state of a linear (ohmic) resistor network, solved directly as a
    Laplacian boundary-value problem - an independent reference for what the
    transient solver should relax to.
    """
    n = len(adjacency)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adjacency[i, j]:
                L[i, j] -= conductances[i, j]
                L[i, i] += conductances[i, j]

    free = np.setdiff1d(np.arange(n), clamped_nodes)
    V = np.zeros(n)
    V[clamped_nodes] = clamped_values
    if len(free):
        rhs = -L[np.ix_(free, clamped_nodes)] @ clamped_values
        V[free] = np.linalg.solve(L[np.ix_(free, free)], rhs)
    return V


# ----------------------------------------------------------------------
# 1. Physics invariants
# ----------------------------------------------------------------------

def test_physics_invariants():
    print("\n[1] Physics invariants")

    # --- 1a. Boundary conditions are held exactly -----------------------
    adjacency, weights, input_nodes, output_nodes, _ = build_autoencoder(4, 3)
    clamped_values = np.array([1.0, 0.0, 1.0, 0.0])
    grid = Grid(adjacency, input_nodes, clamped_values.copy(), capacitances=1.0)
    memristors = build_static_memristor_array(adjacency, weights, ohmic)
    solver = VoltageDynamics(grid, memristors)

    V_init = np.zeros(len(adjacency))
    V_init[input_nodes] = clamped_values
    res = solver.relax_transient(V_init, dt=0.01, max_steps=20000, tol=1e-12)

    check("clamped nodes stay pinned to their values",
          np.allclose(res['V_final'][input_nodes], clamped_values, atol=1e-14))
    check("free relaxation converges", res['converged'], f"steps={res['n_steps']}")

    # --- 1b. Kirchhoff current law at the fixed point -------------------
    I = solver.compute_currents(res['V_final'])
    free_nodes = np.setdiff1d(np.arange(len(adjacency)), input_nodes)
    residual = np.abs(I[free_nodes].sum(axis=1)).max()
    check("Kirchhoff: net current into every free node ~ 0",
          residual < 1e-6, f"max|sum I| = {residual:.2e}")

    # --- 1c. Agreement with an independent analytic solve ---------------
    V_exact = solve_ohmic_analytically(adjacency, weights, input_nodes, clamped_values)
    err = np.abs(res['V_final'] - V_exact).max()
    check("matches direct Laplacian solve for the ohmic case",
          err < 1e-6, f"max|dV| = {err:.2e}")

    # --- 1d. Linearity of the ohmic network ------------------------------
    grid.clamped_values = clamped_values * 2.0
    V_init2 = np.zeros(len(adjacency))
    V_init2[input_nodes] = clamped_values * 2.0
    res2 = solver.relax_transient(V_init2, dt=0.01, max_steps=20000, tol=1e-12)
    lin_err = np.abs(res2['V_final'] - 2.0 * res['V_final']).max()
    check("ohmic network is linear (2x boundary -> 2x solution)",
          lin_err < 1e-6, f"max|dV| = {lin_err:.2e}")

    # --- 1e. Fixed point is independent of capacitance -------------------
    fixed_points, step_counts = [], []
    for C in (0.25, 1.0, 4.0):
        g = Grid(adjacency, input_nodes, clamped_values.copy(), capacitances=C)
        s = VoltageDynamics(g, build_static_memristor_array(adjacency, weights, ohmic))
        r = s.relax_transient(V_init.copy(), dt=0.002, max_steps=200000, tol=1e-12)
        fixed_points.append(r['V_final'])
        step_counts.append(r['n_steps'])
    spread = max(np.abs(fp - fixed_points[0]).max() for fp in fixed_points)
    check("capacitance changes relaxation speed, not the fixed point",
          spread < 1e-6, f"max spread = {spread:.2e}, steps = {step_counts}")
    check("larger capacitance means slower relaxation",
          step_counts[0] < step_counts[1] < step_counts[2], f"steps = {step_counts}")


# ----------------------------------------------------------------------
# 2. Solver knobs
# ----------------------------------------------------------------------

def test_solver_knobs():
    print("\n[2] Solver knobs")

    adjacency, weights, input_nodes, output_nodes, penalty_pairs = build_autoencoder(4, 3)
    clamped_values = np.array([1.0, 0.0, 1.0, 0.0])
    V_init = np.zeros(len(adjacency))
    V_init[input_nodes] = clamped_values

    # --- 2a. Every I-V characteristic runs and stays finite -------------
    print("  I-V characteristics:")
    for label, iv in (("ohmic", ohmic), ("relu", relu_iv),
                      ("sigmoid", sigmoid_iv), ("diode", diode_iv)):
        grid = Grid(adjacency, input_nodes, clamped_values.copy(), capacitances=1.0)
        solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, weights, iv))
        r = solver.relax_transient(V_init.copy(), dt=0.001, max_steps=50000, tol=1e-10)
        V = r['V_final']
        ok = np.all(np.isfinite(V)) and r['converged']
        check(f"{label:<8} relaxes to a finite fixed point", ok,
              f"converged={r['converged']} steps={r['n_steps']} "
              f"V_out={np.round(V[output_nodes], 3)}")

    # --- 2b. dt does not change the fixed point (until it destabilises) --
    print("  time step:")
    ref = None
    for dt in (0.0005, 0.001, 0.005):
        grid = Grid(adjacency, input_nodes, clamped_values.copy(), capacitances=1.0)
        solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, weights, ohmic))
        r = solver.relax_transient(V_init.copy(), dt=dt, max_steps=int(200 / dt), tol=1e-12)
        if ref is None:
            ref = r['V_final']
            check(f"dt={dt} converges", r['converged'], f"steps={r['n_steps']}")
        else:
            err = np.abs(r['V_final'] - ref).max()
            check(f"dt={dt} reaches the same fixed point", err < 1e-5,
                  f"max|dV| vs dt=0.0005: {err:.2e}, steps={r['n_steps']}")

    # --- 2c. beta monotonically pulls output toward input ---------------
    print("  penalty coupling:")
    recon_err = []
    for beta in (0.0, 1.0, 10.0, 100.0):
        grid = Grid(adjacency, input_nodes, clamped_values.copy(), capacitances=1.0)
        solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, weights, ohmic))
        r = solver.relax_transient(V_init.copy(), penalty_pairs=penalty_pairs,
                                   beta=beta, g_penalty=10.0,
                                   dt=0.001, max_steps=50000, tol=1e-10)
        err = float(np.mean((r['V_final'][output_nodes] - clamped_values) ** 2))
        recon_err.append(err)
        check(f"beta={beta:<6} stays finite", np.all(np.isfinite(r['V_final'])),
              f"MSE(out, in) = {err:.5f}")
    check("stronger beta drives output closer to input",
          all(recon_err[i] > recon_err[i + 1] for i in range(len(recon_err) - 1)),
          f"MSE sequence = {[round(e, 5) for e in recon_err]}")

    # --- 2d. g_penalty has the same qualitative effect -------------------
    print("  penalty conductance:")
    gp_err = []
    for g_pen in (0.1, 1.0, 10.0):
        grid = Grid(adjacency, input_nodes, clamped_values.copy(), capacitances=1.0)
        solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, weights, ohmic))
        r = solver.relax_transient(V_init.copy(), penalty_pairs=penalty_pairs,
                                   beta=1.0, g_penalty=g_pen,
                                   dt=0.001, max_steps=50000, tol=1e-10)
        gp_err.append(float(np.mean((r['V_final'][output_nodes] - clamped_values) ** 2)))
    check("stronger g_penalty drives output closer to input",
          all(gp_err[i] > gp_err[i + 1] for i in range(len(gp_err) - 1)),
          f"MSE sequence = {[round(e, 5) for e in gp_err]}")


def test_stability_boundary():
    """
    Explicit Euler is only conditionally stable. Anyone sweeping parameters
    will cross that boundary, so the solver must SAY it diverged rather than
    quietly returning inf/nan. Also maps where the boundary actually is.
    """
    print("\n[2e] Stability boundary and divergence reporting")

    adjacency, weights, input_nodes, output_nodes, penalty_pairs = build_autoencoder(4, 3)
    clamped_values = np.array([1.0, 0.0, 1.0, 0.0])
    V_init = np.zeros(len(adjacency))
    V_init[input_nodes] = clamped_values

    def run(dt, beta, g_pen=10.0, max_steps=3000, **kw):
        grid = Grid(adjacency, input_nodes, clamped_values.copy(), capacitances=1.0)
        solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, weights, relu_iv))
        return solver.relax_transient(V_init.copy(), penalty_pairs=penalty_pairs,
                                      beta=beta, g_penalty=g_pen,
                                      dt=dt, max_steps=max_steps, tol=1e-10, **kw)

    # A configuration known to be past the boundary must be flagged.
    bad = run(dt=0.05, beta=100.0)
    check("an unstable configuration is reported as diverged",
          bad['diverged'] and not bad['converged'],
          f"diverged={bad['diverged']} after {bad['n_steps']} steps")

    # ... and a good one must not be.
    good = run(dt=0.001, beta=100.0, max_steps=50000)
    check("a stable configuration is not falsely flagged",
          good['converged'] and not good['diverged'],
          f"steps={good['n_steps']}")

    # The guard must be defeatable for anyone who wants the raw behaviour.
    raw = run(dt=0.05, beta=100.0, divergence_threshold=None)
    check("divergence guard can be disabled",
          not raw['diverged'] and not np.all(np.isfinite(raw['V_final'])))

    # Map the boundary: bisect critical dt at the repo's default stiffness.
    lo, hi = 0.001, 0.2
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if not run(dt=mid, beta=100.0)['diverged']:
            lo = mid
        else:
            hi = mid
    check("dt=0.001 default sits below the stability boundary", lo > 0.001,
          f"critical dt ~ {lo:.4f} at beta=100/g_penalty=10 (only ~{lo/0.001:.1f}x margin)")

    print("    [info] stability map (relu, g_penalty=10, 3000 steps):")
    print("    " + f"{'dt':>8} | " + " | ".join(f"beta={b:<6g}" for b in (1, 10, 100, 1000)))
    for dt in (0.001, 0.005, 0.01, 0.05):
        cells = []
        for beta in (1.0, 10.0, 100.0, 1000.0):
            r = run(dt=dt, beta=beta)
            cells.append("DIVERGE" if r['diverged'] else ("ok" if r['converged'] else "slow"))
        print("    " + f"{dt:>8} | " + " | ".join(f"{c:<11}" for c in cells))


# ----------------------------------------------------------------------
# 3. Scale
# ----------------------------------------------------------------------

def test_scale():
    print("\n[3] Scale and cost")

    timings = []
    for n_input, n_hidden in ((4, 3), (9, 4), (16, 8)):
        adjacency, weights, input_nodes, output_nodes, _ = build_autoencoder(n_input, n_hidden)
        n_total = len(adjacency)
        pattern = np.array([1.0 if k % 2 == 0 else 0.0 for k in range(n_input)])
        grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
        solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, weights, ohmic))

        V_init = np.zeros(n_total)
        V_init[input_nodes] = pattern

        # Fixed step budget so this measures cost per step, not convergence.
        n_steps = 300
        t0 = time.perf_counter()
        r = solver.relax_transient(V_init, dt=0.001, max_steps=n_steps, tol=0.0)
        elapsed = time.perf_counter() - t0

        n_edges = int(adjacency.sum())
        per_step_us = elapsed / n_steps * 1e6
        timings.append((n_total, n_edges, per_step_us))
        check(f"{n_input}-{n_hidden}-{n_input} ({n_total} nodes) runs and stays finite",
              np.all(np.isfinite(r['V_final'])),
              f"{per_step_us:.0f} us/step, {n_edges} edges")

    # Cost should track node count quadratically (the solver's inner loop is
    # over all node pairs) - worth knowing before scaling to 64-8-64.
    n0, _, t0_us = timings[0]
    n2, _, t2_us = timings[-1]
    observed = t2_us / t0_us
    quadratic = (n2 / n0) ** 2
    check("per-step cost grows roughly as nodes^2 (dense pair loop)",
          0.3 * quadratic < observed < 3.0 * quadratic,
          f"observed {observed:.1f}x vs n^2 prediction {quadratic:.1f}x")

    # Extrapolate to the target size named in the proposal.
    n_target = 64 + 8 + 64
    est_us = t2_us * (n_target / n2) ** 2
    est_cycle_s = est_us * 1e-6 * 20000  # ~2 relaxations x 10k steps
    print(f"    [info] extrapolated 64-8-64: ~{est_us/1000:.1f} ms/step, "
          f"~{est_cycle_s:.0f} s per free+clamped cycle at 10k steps/phase")
    print(f"    [info] -> a 300-cycle run would take ~{est_cycle_s*300/3600:.1f} h "
          f"in pure Python; vectorising the inner loop is the obvious lever")


# ----------------------------------------------------------------------
# 4. Plasticity paths
# ----------------------------------------------------------------------

def test_explicit_contrastive_path():
    print("\n[4a] Explicit contrastive path (SimplePlasticity)")

    adjacency, weights0, input_nodes, output_nodes, penalty_pairs = build_autoencoder(4, 3)
    pattern = np.array([1.0, 0.0, 1.0, 0.0])

    for eta, gamma in ((1.05, 0.001), (0.3, 0.001), (1.05, 0.05)):
        weights = weights0.copy()
        grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
        memristors = build_memristor_array(
            adjacency, weights, iv_func=relu_iv,
            obs_func=quadratic_obs, plast_func=lambda Q, w, th: 0.0,
            window_pts=1, dt_local=1.0, g_min=0.01, g_max=1.0)
        solver = VoltageDynamics(grid, memristors)
        plasticity = SimplePlasticity(eta=eta, gamma=gamma, tau_integrate=20.0)

        V_init = np.zeros(len(adjacency))
        V_init[input_nodes] = pattern
        mse_first = mse_last = None

        for cycle in range(12):
            set_weights(memristors, adjacency, weights)
            grid.clamped_values = pattern.copy()

            free = solver.relax_transient(V_init.copy(), beta=0.0,
                                          dt=0.001, max_steps=10000, tol=1e-10)
            Q_free = compute_Q_from_voltages(free['V_final'], adjacency)

            clamped = solver.relax_transient(free['V_final'], penalty_pairs=penalty_pairs,
                                             beta=100.0, g_penalty=10.0,
                                             dt=0.001, max_steps=10000, tol=1e-10)
            Q_clamped = compute_Q_from_voltages(clamped['V_final'], adjacency)

            weights = plasticity.update_weights(weights, Q_free, Q_clamped,
                                                adjacency, dt_plasticity=1.0)
            mse = float(np.mean((free['V_final'][output_nodes] - pattern) ** 2))
            if cycle == 0:
                mse_first = mse
            mse_last = mse

        w_edges = weights[adjacency]
        check(f"eta={eta}, gamma={gamma}: weights stay in [0,1] and finite",
              np.all(np.isfinite(w_edges)) and w_edges.min() >= -1e-12 and w_edges.max() <= 1 + 1e-12,
              f"w range [{w_edges.min():.3f}, {w_edges.max():.3f}], "
              f"MSE {mse_first:.4f} -> {mse_last:.4f}")


def test_online_theta_path():
    print("\n[4b] Online path (Trainer + global theta)")

    adjacency, weights, input_nodes, output_nodes, penalty_pairs = build_autoencoder(4, 3)
    pattern = np.array([1.0, 0.0, 1.0, 0.0])

    for tau_theta, eta in ((40.0, 1.05), (200.0, 1.05), (40.0, 0.2)):
        grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
        memristors = build_memristor_array(
            adjacency, weights, iv_func=relu_iv,
            obs_func=quadratic_obs,
            plast_func=make_rule(global_threshold_rule, eta=eta, gamma=0.001),
            window_pts=20, dt_local=0.05, g_min=0.01, g_max=1.0)

        trainer = Trainer(grid, memristors, penalty_pairs,
                          beta_free=0.0, beta_clamped=100.0, g_penalty=10.0,
                          exposure_time_free=1.0, exposure_time_clamped=1.0,
                          micro_steps_per_phase=20, dt=0.001, tol=1e-10,
                          tau_theta=tau_theta)

        thetas, mses = [], []
        for _ in range(5):
            out = trainer.run_cycle(pattern)
            thetas.append(out['theta'])
            mses.append(out['mse_free'])

        w_edges = extract_weights(memristors, adjacency)[adjacency]
        finite = np.all(np.isfinite(w_edges)) and np.all(np.isfinite(thetas))
        bounded = w_edges.min() >= -1e-12 and w_edges.max() <= 1 + 1e-12
        check(f"tau_theta={tau_theta}, eta={eta}: finite and bounded",
              finite and bounded,
              f"theta {thetas[0]:.4f} -> {thetas[-1]:.4f}, "
              f"w range [{w_edges.min():.3f}, {w_edges.max():.3f}]")

    # theta must actually respond to its own time constant, otherwise the
    # shared-field knob is decorative.
    def final_theta(tau):
        g = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
        m = build_memristor_array(adjacency, weights, iv_func=relu_iv,
                                  obs_func=quadratic_obs,
                                  plast_func=make_rule(global_threshold_rule),
                                  window_pts=20, dt_local=0.05)
        t = Trainer(g, m, penalty_pairs, beta_clamped=100.0, g_penalty=10.0,
                    exposure_time_free=1.0, exposure_time_clamped=1.0,
                    micro_steps_per_phase=20, dt=0.001, tau_theta=tau)
        for _ in range(3):
            t.run_cycle(pattern)
        return t.theta

    fast, slow = final_theta(20.0), final_theta(400.0)
    check("faster tau_theta tracks the observable more quickly",
          fast > slow, f"theta(tau=20) = {fast:.5f} > theta(tau=400) = {slow:.5f}")


def test_paths_actually_learn():
    """
    The gap that let a completely non-learning default configuration ship
    with 60 green checks: everything above asserts "finite and bounded",
    which a dead network satisfies perfectly. These assert the thing we
    actually care about - that reconstruction error goes DOWN.
    """
    print("\n[4f] Both plasticity paths actually learn")

    adjacency, weights, input_nodes, output_nodes, penalty_pairs = build_autoencoder(4, 3)
    pattern = np.array([1.0, 0.0, 1.0, 0.0])

    # --- online / global theta -----------------------------------------
    exposure, micro = 10.0, 200
    grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
    memristors = build_memristor_array(
        adjacency, weights, iv_func=relu_iv, obs_func=quadratic_obs,
        plast_func=make_rule(global_threshold_rule, eta=1.05, gamma=0.001),
        window_pts=micro, dt_local=exposure / micro)
    trainer = Trainer(grid, memristors, penalty_pairs, beta_clamped=100.0,
                      g_penalty=10.0, exposure_time_free=exposure,
                      exposure_time_clamped=exposure, micro_steps_per_phase=micro,
                      dt=0.001, tau_theta=40.0)
    mses = [trainer.run_cycle(pattern)['mse_free'] for _ in range(8)]
    check("global-theta path reduces reconstruction error",
          mses[-1] < 0.12 and mses[-1] < 0.6 * mses[0],
          f"MSE {mses[0]:.5f} -> {mses[-1]:.5f}")

    # --- explicit contrastive -------------------------------------------
    w = weights.copy()
    grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
    mem = build_memristor_array(adjacency, w, iv_func=relu_iv, obs_func=quadratic_obs,
                                plast_func=lambda Q, ww, th: 0.0, window_pts=1, dt_local=1.0)
    solver = VoltageDynamics(grid, mem)
    plasticity = SimplePlasticity(eta=1.05, gamma=0.001, tau_integrate=20.0)
    V_init = np.zeros(len(adjacency)); V_init[input_nodes] = pattern
    first = last = None
    for cycle in range(12):
        set_weights(mem, adjacency, w)
        free = solver.relax_transient(V_init.copy(), beta=0.0, dt=0.001,
                                      max_steps=10000, tol=1e-10)
        clamped = solver.relax_transient(free['V_final'], penalty_pairs=penalty_pairs,
                                         beta=100.0, g_penalty=10.0, dt=0.001,
                                         max_steps=10000, tol=1e-10)
        w = plasticity.update_weights(
            w, compute_Q_from_voltages(free['V_final'], adjacency),
            compute_Q_from_voltages(clamped['V_final'], adjacency), adjacency, 1.0)
        mse = float(np.mean((free['V_final'][output_nodes] - pattern) ** 2))
        first = mse if cycle == 0 else first
        last = mse
    check("contrastive path reduces reconstruction error",
          last < 0.12 and last < 0.6 * first, f"MSE {first:.5f} -> {last:.5f}")


def test_window_phase_guard():
    """The window/phase ratio decides whether learning happens at all, and
    a mismatch is otherwise silent - so it must warn."""
    print("\n[4g] Window-vs-phase mismatch is reported")

    import warnings as _w
    adjacency, weights, input_nodes, _, penalty_pairs = build_autoencoder(4, 3)
    pattern = np.array([1.0, 0.0, 1.0, 0.0])

    def build(window_pts):
        grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
        mem = build_memristor_array(
            adjacency, weights, iv_func=relu_iv, obs_func=quadratic_obs,
            plast_func=make_rule(global_threshold_rule),
            window_pts=window_pts, dt_local=0.05)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            Trainer(grid, mem, penalty_pairs, micro_steps_per_phase=200)
        return [x for x in caught if issubclass(x.category, RuntimeWarning)]

    check("mismatched window warns", len(build(100)) == 1,
          build(100)[0].message.args[0][:60] + "..." if build(100) else "no warning")
    check("matched window is silent", len(build(200)) == 0)


def test_rule_swapping():
    print("\n[4c] Swapping the plasticity rule (the one-line requirement)")

    adjacency, weights, input_nodes, output_nodes, penalty_pairs = build_autoencoder(4, 3)
    pattern = np.array([1.0, 0.0, 1.0, 0.0])

    # A completely different rule, defined here and passed in unchanged -
    # no edits anywhere else in the codebase.
    def decay_only_rule(Q_avg, w, theta, gamma=0.05):
        return -gamma * w

    def run_with(plast_func):
        grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
        memristors = build_memristor_array(
            adjacency, weights, iv_func=relu_iv, obs_func=quadratic_obs,
            plast_func=plast_func, window_pts=20, dt_local=0.05)
        trainer = Trainer(grid, memristors, penalty_pairs, beta_clamped=100.0,
                          g_penalty=10.0, exposure_time_free=1.0,
                          exposure_time_clamped=1.0, micro_steps_per_phase=20,
                          dt=0.001, tau_theta=40.0)
        for _ in range(3):
            trainer.run_cycle(pattern)
        return extract_weights(memristors, adjacency)[adjacency]

    w_default = run_with(make_rule(global_threshold_rule))
    w_decay = run_with(decay_only_rule)
    w_frozen = run_with(lambda Q, w, th: 0.0)

    check("custom rule is accepted with no other code change",
          np.all(np.isfinite(w_decay)))
    check("decay-only rule shrinks every weight",
          np.all(w_decay < weights[adjacency]),
          f"mean w: {weights[adjacency].mean():.4f} -> {w_decay.mean():.4f}")
    check("zero rule leaves weights untouched",
          np.allclose(w_frozen, weights[adjacency], atol=1e-12))
    check("default rule behaves differently from decay-only",
          not np.allclose(w_default, w_decay, atol=1e-6),
          f"mean w default {w_default.mean():.4f} vs decay {w_decay.mean():.4f}")


def test_theta_extension_point():
    print("\n[4d] Per-edge theta is reachable (future-proofing check)")

    adjacency, weights, input_nodes, _, _ = build_autoencoder(4, 3)
    memristors = build_memristor_array(
        adjacency, weights, iv_func=relu_iv, obs_func=quadratic_obs,
        plast_func=make_rule(global_threshold_rule), window_pts=5, dt_local=0.05)

    edges = [(i, j) for i in range(len(adjacency)) for j in range(len(adjacency))
             if adjacency[i, j]][:2]
    (a, b), (c, d) = edges
    for (i, j) in edges:
        for _ in range(5):
            memristors[i, j].step(0.4)

    memristors[a, b].set_theta(0.0)
    memristors[c, d].set_theta(0.5)
    check("two edges can hold different theta values simultaneously",
          memristors[a, b].theta != memristors[c, d].theta)
    check("a per-edge theta actually changes that edge's dw/dt",
          not np.isclose(memristors[a, b].dw_dt(), memristors[c, d].dw_dt()),
          f"dw/dt: {memristors[a, b].dw_dt():.5f} vs {memristors[c, d].dw_dt():.5f}")


# ----------------------------------------------------------------------
# 5. Reproducibility
# ----------------------------------------------------------------------

def test_dataset_integration():
    """
    The end-to-end question: can the machinery be driven by the real
    dataset, presenting many different patterns in sequence, rather than one
    hand-written input vector?
    """
    print("\n[4e] Driving the Trainer from the Bars & Stripes dataset")

    from datasets.bars_stripes import BarsAndStripes

    ds = BarsAndStripes(N=2, voltage_on=1.0, voltage_off=0.0)
    patterns = ds.get_all_flattened()
    n_input = patterns.shape[1]

    adjacency, weights, input_nodes, output_nodes, penalty_pairs = build_autoencoder(
        n_input, 3, seed=11)
    grid = Grid(adjacency, input_nodes, patterns[0].copy(), capacitances=1.0)
    memristors = build_memristor_array(
        adjacency, weights, iv_func=relu_iv, obs_func=quadratic_obs,
        plast_func=make_rule(global_threshold_rule, eta=1.05, gamma=0.001),
        window_pts=20, dt_local=0.05)
    trainer = Trainer(grid, memristors, penalty_pairs, beta_clamped=100.0,
                      g_penalty=10.0, exposure_time_free=1.0,
                      exposure_time_clamped=1.0, micro_steps_per_phase=20,
                      dt=0.001, tau_theta=40.0)

    t0 = time.perf_counter()
    epoch_mse = []
    for epoch in range(3):
        mses = [trainer.run_cycle(p)['mse_free'] for p in patterns]
        epoch_mse.append(float(np.mean(mses)))
    elapsed = time.perf_counter() - t0

    n_cycles = 3 * len(patterns)
    check(f"presents all {len(patterns)} dataset patterns over 3 epochs",
          np.all(np.isfinite(epoch_mse)),
          f"{n_cycles} cycles in {elapsed:.1f}s ({elapsed/n_cycles*1000:.0f} ms/cycle), "
          f"mean MSE per epoch = {[round(m, 4) for m in epoch_mse]}")

    w = extract_weights(memristors, adjacency)[adjacency]
    check("weights and theta stay finite across a multi-pattern run",
          np.all(np.isfinite(w)) and np.isfinite(trainer.theta),
          f"w range [{w.min():.3f}, {w.max():.3f}], theta={trainer.theta:.5f}")

    # Does the network respond at all differently to different inputs? If
    # every pattern produced the same output the run would be finite and
    # "passing" while being physically vacuous.
    outs = np.array([trainer.run_cycle(p)['V_free'][output_nodes] for p in patterns])
    spread = float((outs.max(axis=0) - outs.min(axis=0)).max())
    check("distinct input patterns produce distinct outputs",
          spread > 1e-3,
          f"max per-node spread across patterns = {spread:.5f}")


def test_reproducibility():
    print("\n[5] Reproducibility")

    def run():
        adjacency, weights, input_nodes, output_nodes, penalty_pairs = build_autoencoder(4, 3, seed=7)
        pattern = np.array([1.0, 0.0, 1.0, 0.0])
        grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
        memristors = build_memristor_array(
            adjacency, weights, iv_func=relu_iv, obs_func=quadratic_obs,
            plast_func=make_rule(global_threshold_rule), window_pts=20, dt_local=0.05)
        trainer = Trainer(grid, memristors, penalty_pairs, beta_clamped=100.0,
                          g_penalty=10.0, exposure_time_free=1.0,
                          exposure_time_clamped=1.0, micro_steps_per_phase=20,
                          dt=0.001, tau_theta=40.0)
        mses = [trainer.run_cycle(pattern)['mse_free'] for _ in range(4)]
        return np.array(mses), extract_weights(memristors, adjacency), trainer.theta

    m1, w1, t1 = run()
    m2, w2, t2 = run()
    check("identical config reproduces bit-identical MSE trace",
          np.array_equal(m1, m2), f"{np.round(m1, 6)}")
    check("identical config reproduces bit-identical weights", np.array_equal(w1, w2))
    check("identical config reproduces bit-identical theta", t1 == t2, f"theta = {t1:.8f}")

    # Different seeds must actually produce different networks, otherwise
    # the seed is being ignored somewhere.
    _, w_a, _, _, _ = build_autoencoder(4, 3, seed=1)
    _, w_b, _, _, _ = build_autoencoder(4, 3, seed=2)
    check("different seeds give different initial weights",
          not np.allclose(w_a, w_b))


def test_cli():
    """The CLI is the intended way to poke at this by hand, so it has to at
    least start, parse flags, and agree with the library it wraps."""
    print("\n[6] Command-line interface (sim.py)")

    import subprocess
    repo = Path(__file__).parent.parent
    sim = repo / "sim.py"

    check("sim.py exists", sim.exists())

    def run(*argv, timeout=300):
        return subprocess.run([sys.executable, str(sim), *argv],
                              capture_output=True, text=True, timeout=timeout, cwd=repo)

    r = run("--help")
    check("sim.py --help works", r.returncode == 0 and "relax" in r.stdout)

    for cmd in ("relax", "train", "sweep", "stability", "bench", "info"):
        r = run(cmd, "--help")
        check(f"subcommand '{cmd}' has help", r.returncode == 0)

    # A free-phase relaxation from the CLI must match the library directly.
    r = run("relax", "--beta", "0", "--iv", "ohmic", "--max-steps", "50000")
    check("sim.py relax runs", r.returncode == 0, r.stderr.strip()[:120])

    adjacency, weights, input_nodes, output_nodes, _ = build_autoencoder(4, 3)
    pattern = np.array([1.0, 0.0, 1.0, 0.0])
    grid = Grid(adjacency, input_nodes, pattern.copy(), capacitances=1.0)
    solver = VoltageDynamics(grid, build_static_memristor_array(adjacency, weights, ohmic))
    V_init = np.zeros(len(adjacency)); V_init[input_nodes] = pattern
    expected = solver.relax_transient(V_init, dt=0.001, max_steps=50000, tol=1e-10)
    expected_mse = float(np.mean((expected['V_final'][output_nodes] - pattern) ** 2))

    printed = None
    for line in r.stdout.splitlines():
        if "reconstruction MSE" in line:
            printed = float(line.split(":")[1])
    check("CLI reconstruction MSE matches the library",
          printed is not None and abs(printed - expected_mse) < 1e-6,
          f"CLI {printed} vs library {expected_mse:.6f}")

    # The CLI must surface divergence rather than printing garbage.
    r = run("relax", "--dt", "0.05", "--beta", "100")
    check("CLI reports divergence with a non-zero exit code",
          r.returncode == 1 and "blew up" in r.stdout)

    r = run("sweep", "--param", "beta", "--values", "0,1,10", "--max-steps", "5000")
    check("sim.py sweep runs and tabulates", r.returncode == 0 and "beta" in r.stdout)


if __name__ == "__main__":
    print("=" * 72)
    print("MeroCircuit smoke / sanity sweep")
    print("=" * 72)

    t_start = time.perf_counter()

    test_physics_invariants()
    test_solver_knobs()
    test_stability_boundary()
    test_scale()
    test_explicit_contrastive_path()
    test_online_theta_path()
    test_paths_actually_learn()
    test_window_phase_guard()
    test_rule_swapping()
    test_theta_extension_point()
    test_dataset_integration()
    test_reproducibility()
    test_cli()

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 72)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({elapsed:.1f}s)")
    if FAIL:
        print("\nFailures:")
        for name, detail in FAIL:
            print(f"  - {name}  {detail}")
    print("=" * 72)

    sys.exit(1 if FAIL else 0)
