# tests/test_with_memristors.py

"""
Smoke test for the Grid + Memristor + Trainer pipeline (the "new
architecture" discussed in DEVELOPMENT.md).

This replaces the previous version of this file, which called Memristor()
and VoltageDynamics() with keyword arguments that matched neither class as
actually implemented (a sketch written ahead of the real signatures, never
runnable - see DEVELOPMENT.md, "History: the broken merge").

Scope: this is a *correctness* smoke test (does the machinery run, stay
finite, and actually move the weights), not a reproduction of the 4->3->4
plateau documented in DEVELOPMENT.md - that took ~90 long (10s exposure)
cycles even in the fast, vectorized prototype used to characterize it.
Running the same duration through per-edge Memristor objects (one Python
object and method call per edge per solver step) is considerably slower;
for that kind of longer empirical run, increase exposure_time_free/clamped,
micro_steps_per_phase and n_cycles below and be prepared for it to take
minutes rather than seconds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from grid.grid import Grid
from network.builders import build_memristor_array, extract_weights
from network.iv_characteristics import relu_iv
from training.rules import make_rule, global_threshold_rule
from training.trainer import Trainer


def quadratic_observable(V_drop, I):
    """Q = (V_drop)^2 - matches compute_Q_from_voltages in
    training/plasticity.py, i.e. the observable that actually produced the
    documented 4->3->4 plateau (not the linear one described as adopted in
    DEVELOPMENT.md's design notes - see "History: the broken merge" for why
    those two disagree)."""
    return V_drop ** 2


def build_autoencoder(n_input=4, n_hidden=3, n_output=4, seed=42,
                       w_min=0.1, w_max=0.9):
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

    rng = np.random.default_rng(seed)
    weights = rng.uniform(w_min, w_max, size=(n_total, n_total)) * adjacency

    penalty_pairs = [(i, n_input + n_hidden + i) for i in range(n_input)]

    return adjacency, weights, input_nodes, output_nodes, penalty_pairs, n_total


def test_smoke_training_runs_and_moves_weights():
    print("=== Smoke test: Grid + Memristor + Trainer, 4->3->4 ===")

    adjacency, weights, input_nodes, output_nodes, penalty_pairs, n_total = build_autoencoder()
    pattern = np.array([1.0, 0.0, 1.0, 0.0])

    grid = Grid(
        adjacency=adjacency,
        clamped_nodes=input_nodes,
        clamped_values=pattern.copy(),
        capacitances=1.0,
    )

    plast_func = make_rule(global_threshold_rule, eta=1.05, gamma=0.001, sign=-1.0)

    memristors = build_memristor_array(
        adjacency, weights,
        iv_func=relu_iv,
        obs_func=quadratic_observable,
        plast_func=plast_func,
        window_pts=20,
        dt_local=0.05,
        g_min=0.01, g_max=1.0,
    )

    trainer = Trainer(
        grid, memristors, penalty_pairs,
        beta_free=0.0, beta_clamped=100.0, g_penalty=10.0,
        exposure_time_free=1.0, exposure_time_clamped=1.0,
        micro_steps_per_phase=20,
        # dt=0.001 matches the stability bound noted in DEVELOPMENT.md for
        # beta_clamped=100/g_penalty=10 (explicit Euler with strong penalty
        # coupling is stiff - a larger dt here diverges, see "Open Problems
        # #2: Beta / dt stability").
        dt=0.001, tol=1e-10,
        tau_theta=40.0,
    )

    w_before = extract_weights(memristors, adjacency).copy()

    mse_history = []
    for cycle in range(5):
        result = trainer.run_cycle(pattern)
        mse_history.append(result['mse_free'])
        print(f"  cycle {cycle + 1}: MSE_free={result['mse_free']:.5f}  "
              f"theta={result['theta']:.5f}  "
              f"free_converged={result['free_converged']}  clamped_converged={result['clamped_converged']}")

    w_after = extract_weights(memristors, adjacency)

    assert np.all(np.isfinite(mse_history)), "MSE became non-finite (NaN/inf) during training"
    assert np.all(np.isfinite(w_after[adjacency])), "Weights became non-finite during training"
    assert not np.allclose(w_before, w_after), "Weights did not move at all - plasticity loop is not wired up"
    assert np.all((w_after[adjacency] >= 0.0) & (w_after[adjacency] <= 1.0)), "Weights left [0, 1]"

    print(f"MSE history: {[f'{m:.4f}' for m in mse_history]}")
    print("✓ Smoke test passed (machinery runs, stays finite, weights move)\n")


if __name__ == "__main__":
    test_smoke_training_runs_and_moves_weights()
