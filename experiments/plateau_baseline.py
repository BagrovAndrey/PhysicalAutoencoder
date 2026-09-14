"""
Informal experiment (not part of the maintained test suite - see
tests/test_plasticity.py for the maintained, currently-passing regression
test that reproduces the same numbers): reproduce the 4->3->4 plateau from
DEVELOPMENT.md, then check whether swapping the quadratic snapshot
observable (compute_Q_from_voltages, Q=(dV)^2) for the linear one (Q=dV,
as DEVELOPMENT.md's design notes claim is current) changes the outcome.

NOTE: written and run against network/dynamics.py's OLD array/iv_function
API, before the 2026-09-14 fix that made it Grid/Memristor-based (see
DEVELOPMENT.md, "History: the broken merge"). Kept as-is for a record of
the exact numbers cited in DEVELOPMENT.md; to run it again, either restore
that old API from git history (commit d456d0f), or port the solver setup
below to Grid + network.builders.build_memristor_array the same way
tests/test_plasticity.py was ported.
"""
import sys
import numpy as np

from network.dynamics import VoltageDynamics
from network.iv_characteristics import relu_iv
from training.plasticity import SimplePlasticity, weights_to_conductances, compute_Q_from_voltages

# ---- config, matching tests/test_plasticity.py CONFIG ----
n_input, n_hidden, n_output = 4, 3, 4
eta, gamma, tau_integrate = 1.05, 0.001, 20.0
n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 60
exposure_time_free = exposure_time_clamped = 10.0
beta_free, beta_clamped, g_penalty = 0.0, 100.0, 10.0
dt, tol, dt_plasticity = 0.001, 1e-10, 1.0
w_min, w_max, g_min, g_max = 0.1, 0.9, 0.01, 1.0
random_seed = 42

USE_LINEAR = "--linear" in sys.argv

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
plasticity = SimplePlasticity(eta=eta, gamma=gamma, tau_integrate=tau_integrate)

penalty_pairs = [(i, n_input + n_hidden + i) for i in range(n_input)]

rng = np.random.default_rng(random_seed)
weights = rng.uniform(w_min, w_max, size=(n_total, n_total)) * adjacency

pattern = np.array([1.0, 0.0, 1.0, 0.0])

max_steps_free = int(exposure_time_free / dt)
max_steps_clamped = int(exposure_time_clamped / dt)


def compute_Q_linear(V, adjacency):
    n = len(V)
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adjacency[i, j]:
                Q[i, j] = V[j] - V[i]
    return Q


mse_free_history = []

print(f"=== {'LINEAR' if USE_LINEAR else 'QUADRATIC'} observable, n_cycles={n_cycles} ===")

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

    if USE_LINEAR:
        Q_free = compute_Q_linear(result_free['V_final'], adjacency)
    else:
        Q_free = compute_Q_from_voltages(result_free['V_final'], adjacency)

    V_output_free = result_free['V_final'][output_nodes]
    mse_free = np.mean((pattern - V_output_free) ** 2)
    mse_free_history.append(mse_free)

    result_clamped = solver.relax_transient(
        conductances, result_free['V_final'],
        clamped_nodes=input_nodes, clamped_values=pattern,
        penalty_pairs=penalty_pairs, beta=beta_clamped, g_penalty=g_penalty,
        dt=dt, tol=tol, max_steps=max_steps_clamped, record_history=False,
    )

    if USE_LINEAR:
        Q_clamped = compute_Q_linear(result_clamped['V_final'], adjacency)
    else:
        Q_clamped = compute_Q_from_voltages(result_clamped['V_final'], adjacency)

    weights = plasticity.update_weights(weights, Q_free, Q_clamped, adjacency, dt_plasticity)

    if (cycle + 1) % 10 == 0 or cycle == 0:
        V_output_clamped = result_clamped['V_final'][output_nodes]
        print(f"  cycle {cycle+1:4d}: MSE_free={mse_free:.5f}  V_out_free={np.round(V_output_free,3)}  V_out_clamped={np.round(V_output_clamped,3)}")

print(f"Initial MSE: {mse_free_history[0]:.5f}  Final MSE: {mse_free_history[-1]:.5f}")
