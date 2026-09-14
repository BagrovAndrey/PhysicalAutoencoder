"""
Scratch experiment (not part of the repo). Tests the fully local,
clock-free plasticity scheme: no phase label is ever given to an edge,
only a continuously-running windowed/EMA average of a local observable Q,
exactly like Memristor.step()/Q_avg() in memristor/memristor.py.

We reuse the same 4->3->4 / [1,0,1,0] setup and quadratic Q as the
established baseline (scratch_plateau_check.py), and the same electrical
relaxation (quasi-static: relax voltages to steady state per phase, then
let the local integrators evolve while V is held at that steady value for
the duration of the exposure time - this matches the tau_el << tau_plastic
assumption already in the proposal, and is exactly what
tests/test_with_memristors.py does).

The only thing that changes vs. scratch_plateau_check.py is *how* the
weight update is computed: instead of one bulk update per cycle driven by
the explicit contrast (Q_clamped - Q_free), each edge continuously runs

    Q_avg <- (1-alpha)*Q_avg + alpha*Q_instantaneous      (EMA, alpha=dt_micro/tau_integrate)
    dw/dt = -eta*Q_avg*(1-w) - gamma*w
    w     <- clip(w + dw/dt * dt_micro, 0, 1)

at every micro-step of BOTH phases, with no reference to which phase it is
and no stored "previous phase" value anywhere.

NOTE: written and run against network/dynamics.py's OLD array/iv_function
API, before the 2026-09-14 fix that made it Grid/Memristor-based (see
DEVELOPMENT.md, "History: the broken merge"). The finding it records
(pure Q_avg with no reference collapses every weight to g_min) is now
also implemented properly as the live online path in memristor/memristor.py
+ training/rules.py + training/trainer.py, exercised by
tests/test_with_memristors.py. Kept as-is as a record of the original
exploration; to run it again as a standalone script, either restore the
old API from git history (commit d456d0f), or port it to Grid +
network.builders.build_memristor_array the same way tests/test_plasticity.py
was ported.
"""
import sys
import numpy as np

from network.dynamics import VoltageDynamics
from network.iv_characteristics import relu_iv
from training.plasticity import weights_to_conductances

n_input, n_hidden, n_output = 4, 3, 4
eta, gamma, tau_integrate = 1.05, 0.001, 20.0
n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 60
exposure_time_free = exposure_time_clamped = 10.0
beta_free, beta_clamped, g_penalty = 0.0, 100.0, 10.0
dt, tol = 0.001, 1e-10
w_min, w_max, g_min, g_max = 0.1, 0.9, 0.01, 1.0
random_seed = 42

micro_steps_per_phase = 200
dt_micro_free = exposure_time_free / micro_steps_per_phase
dt_micro_clamped = exposure_time_clamped / micro_steps_per_phase
alpha_free = dt_micro_free / tau_integrate
alpha_clamped = dt_micro_clamped / tau_integrate

n_total = n_input + n_hidden + n_output
input_nodes = np.arange(n_input)
hidden_nodes = np.arange(n_input, n_input + n_hidden)
output_nodes = np.arange(n_input + n_hidden, n_total)

adjacency = np.zeros((n_total, n_total), dtype=bool)
for i in input_nodes:
    for h in hidden_nodes:
        adjacency[i, h] = adjacency[h, i] = True
for h in hidden_nodes:
    for o in output_nodes:
        adjacency[h, o] = adjacency[o, h] = True

solver = VoltageDynamics(adjacency, relu_iv, capacitances=1.0)
penalty_pairs = [(i, n_input + n_hidden + i) for i in range(n_input)]

rng = np.random.default_rng(random_seed)
weights = rng.uniform(w_min, w_max, size=(n_total, n_total)) * adjacency
Q_avg = np.zeros((n_total, n_total))  # continuous, never reset, no phase memory

pattern = np.array([1.0, 0.0, 1.0, 0.0])
max_steps_free = int(exposure_time_free / dt)
max_steps_clamped = int(exposure_time_clamped / dt)


def evolve_online(V_const, alpha, dt_micro, w, Q_avg):
    """Advance the continuous local integrators for one phase's exposure
    time, using the (quasi-static) constant relaxed voltage V_const."""
    V_drop = V_const[None, :] - V_const[:, None]  # V[j]-V[i] at [i,j]
    Q_instant = (V_drop.T) ** 2 * adjacency  # Q_ij = (V_j - V_i)^2, matches compute_Q_from_voltages
    for _ in range(micro_steps_per_phase):
        Q_avg = (1 - alpha) * Q_avg + alpha * Q_instant
        dw = (-eta * Q_avg * (1 - w) - gamma * w) * adjacency
        w = np.clip(w + dw * dt_micro, 0, 1)
    return w, Q_avg


mse_free_history = []
print(f"=== ONLINE windowed rule (no phase label), n_cycles={n_cycles} ===")

for cycle in range(n_cycles):
    conductances = weights_to_conductances(weights, g_min, g_max)
    V_init = np.zeros(n_total)
    V_init[input_nodes] = pattern

    result_free = solver.relax_transient(
        conductances, V_init,
        clamped_nodes=input_nodes, clamped_values=pattern,
        penalty_pairs=penalty_pairs, beta=beta_free,
        dt=dt, tol=tol, max_steps=max_steps_free, record_history=False,
    )
    V_free_final = result_free['V_final']
    V_output_free = V_free_final[output_nodes]
    mse_free = np.mean((pattern - V_output_free) ** 2)
    mse_free_history.append(mse_free)

    weights, Q_avg = evolve_online(V_free_final, alpha_free, dt_micro_free, weights, Q_avg)

    conductances = weights_to_conductances(weights, g_min, g_max)
    result_clamped = solver.relax_transient(
        conductances, V_free_final,
        clamped_nodes=input_nodes, clamped_values=pattern,
        penalty_pairs=penalty_pairs, beta=beta_clamped, g_penalty=g_penalty,
        dt=dt, tol=tol, max_steps=max_steps_clamped, record_history=False,
    )
    V_clamped_final = result_clamped['V_final']
    V_output_clamped = V_clamped_final[output_nodes]

    weights, Q_avg = evolve_online(V_clamped_final, alpha_clamped, dt_micro_clamped, weights, Q_avg)

    if (cycle + 1) % 10 == 0 or cycle == 0:
        print(f"  cycle {cycle+1:4d}: MSE_free={mse_free:.5f}  V_out_free={np.round(V_output_free,3)}  V_out_clamped={np.round(V_output_clamped,3)}  w_mean={weights[adjacency].mean():.3f}")

print(f"Initial MSE: {mse_free_history[0]:.5f}  Final MSE: {mse_free_history[-1]:.5f}")
