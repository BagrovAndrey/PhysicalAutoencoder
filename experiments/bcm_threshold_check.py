"""
Scratch experiment. The pure online rule (scratch_online_rule_check.py)
collapsed every weight to g_min - a quadratic, sign-definite Q_avg can only
ever push w in one direction, so with zero clock/phase-label there is no
way to get a two-sided (potentiate here, depress there) signal.

This tries the standard local fix for that class of problem (BCM-style
plasticity, a la Bienenstock-Cooper-Munro): keep Volodya's fast local
window Q_avg exactly as is, but let each edge ALSO keep one more purely
local, slowly-adapting state variable theta_ij = a much slower low-pass
average of Q_avg itself (time constant >> one full free+clamped cycle).
Drive plasticity by (Q_avg - theta) instead of raw Q_avg. No external
clock, no phase label, no cross-edge communication - theta is derived
from the SAME local signal each edge already has, just filtered slower.

Because a perfectly symmetric free/clamped split makes the positive
excursion (during clamped, if Qc>Qf) exactly cancel the negative
excursion (during free) to first order, this also tests asymmetric
exposure times as the symmetry-breaking knob.

NOTE: written and run against network/dynamics.py's OLD array/iv_function
API, before the 2026-09-14 fix that made it Grid/Memristor-based (see
DEVELOPMENT.md, "History: the broken merge"). The architecture decision
that came out of this exploration (a single GLOBAL theta, not a per-edge
local one as tried here) is what's actually implemented in
memristor/memristor.py + training/rules.py + training/trainer.py - see
DEVELOPMENT.md, "Global vs local theta", for why. Kept as-is as a record
of the original exploration; to run it again as a standalone script,
either restore the old API from git history (commit d456d0f), or port it
to Grid + network.builders.build_memristor_array the same way
tests/test_plasticity.py was ported.
"""
import sys
import numpy as np

from network.dynamics import VoltageDynamics
from network.iv_characteristics import relu_iv
from training.plasticity import weights_to_conductances

n_input, n_hidden, n_output = 4, 3, 4
eta, gamma = 1.05, 0.001
tau_integrate = 20.0       # fast window (matches CONFIG.tau_integrate)
tau_baseline = 400.0       # slow threshold, >> one full cycle
n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 40
exposure_time_free = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
exposure_time_clamped = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
sign = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0
beta_free, beta_clamped, g_penalty = 0.0, 100.0, 10.0
dt, tol = 0.001, 1e-10
w_min, w_max, g_min, g_max = 0.1, 0.9, 0.01, 1.0
random_seed = 42

micro_steps_per_phase = 200
dt_micro_free = exposure_time_free / micro_steps_per_phase
dt_micro_clamped = exposure_time_clamped / micro_steps_per_phase

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
Q_avg = np.zeros((n_total, n_total))
theta = np.zeros((n_total, n_total))

pattern = np.array([1.0, 0.0, 1.0, 0.0])
max_steps_free = int(exposure_time_free / dt)
max_steps_clamped = int(exposure_time_clamped / dt)


def evolve_online(V_const, dt_micro, w, Q_avg, theta):
    V_drop = (V_const[None, :] - V_const[:, None]).T
    Q_instant = (V_drop ** 2) * adjacency
    alpha_fast = dt_micro / tau_integrate
    alpha_slow = dt_micro / tau_baseline
    for _ in range(micro_steps_per_phase):
        Q_avg = (1 - alpha_fast) * Q_avg + alpha_fast * Q_instant
        theta = (1 - alpha_slow) * theta + alpha_slow * Q_avg
        dw = (sign * eta * (Q_avg - theta) * (1 - w) - gamma * w) * adjacency
        w = np.clip(w + dw * dt_micro, 0, 1)
    return w, Q_avg, theta


mse_free_history = []
print(f"=== BCM-style local threshold rule, sign={sign}, T_free={exposure_time_free}, T_clamped={exposure_time_clamped}, n_cycles={n_cycles} ===")

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

    weights, Q_avg, theta = evolve_online(V_free_final, dt_micro_free, weights, Q_avg, theta)

    conductances = weights_to_conductances(weights, g_min, g_max)
    result_clamped = solver.relax_transient(
        conductances, V_free_final,
        clamped_nodes=input_nodes, clamped_values=pattern,
        penalty_pairs=penalty_pairs, beta=beta_clamped, g_penalty=g_penalty,
        dt=dt, tol=tol, max_steps=max_steps_clamped, record_history=False,
    )
    V_clamped_final = result_clamped['V_final']
    V_output_clamped = V_clamped_final[output_nodes]

    weights, Q_avg, theta = evolve_online(V_clamped_final, dt_micro_clamped, weights, Q_avg, theta)

    if (cycle + 1) % 5 == 0 or cycle == 0:
        print(f"  cycle {cycle+1:4d}: MSE_free={mse_free:.5f}  V_out_free={np.round(V_output_free,3)}  w_mean={weights[adjacency].mean():.3f}")

print(f"Initial MSE: {mse_free_history[0]:.5f}  Final MSE: {mse_free_history[-1]:.5f}")
