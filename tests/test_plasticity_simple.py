# tests/test_plasticity_simple.py

"""
Minimal plasticity test: 3-node chain.

Setup: 0 -- 1 -- 2
Clamp: V[0]=0, V[2]=1
Goal: Train V[1] to reach target (e.g., 0.68)

This requires plasticity to make R01 ≠ R12.

Uses the explicit contrastive rule (SimplePlasticity, batch update from two
voltage snapshots) on top of the Grid + Memristor substrate: weights are
computed the old way each cycle and written back into the Memristor array
with network.builders.set_weights, rather than driven online through each
Memristor's own plast_func/theta (that online path is exercised separately
in tests/test_with_memristors.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid.grid import Grid
from network.builders import build_memristor_array, extract_weights, set_weights
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic
from training.plasticity import SimplePlasticity, weights_to_conductances, compute_Q_from_voltages


def test_three_node_chain():
    """
    Simplest possible test: 3-node chain.

    Free phase: V[1] settles based on current weights
    Clamped phase: Penalty pulls V[1] toward target
    Update: Weights should adjust to make V[1] closer to target in free phase
    """
    print("\n=== Test: 3-Node Chain Plasticity ===")
    print("Topology: 0 -- 1 -- 2")
    print("Boundary: V[0]=0, V[2]=1")
    print("Target: V[1]=0.68\n")

    # Topology
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=bool)

    # Initialize with equal weights
    weights = np.array([
        [0, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0]
    ], dtype=float)

    g_min, g_max = 0.1, 1.0

    # Solver and plasticity. The Memristor array's own plast_func/obs_func
    # are unused here - SimplePlasticity computes weight updates externally
    # from voltage snapshots, and we write them back with set_weights.
    clamped_nodes = np.array([0, 2])
    clamped_values = np.array([0.0, 1.0])
    grid = Grid(adjacency, clamped_nodes, clamped_values.copy(), capacitances=1.0)
    memristors = build_memristor_array(
        adjacency, weights, iv_func=ohmic,
        obs_func=lambda V, I: 0.0, plast_func=lambda Q, w, theta: 0.0,
        window_pts=1, dt_local=1.0, g_min=g_min, g_max=g_max,
    )
    solver = VoltageDynamics(grid, memristors)
    plasticity = SimplePlasticity(eta=0.1, gamma=0.001, tau_integrate=5.0)

    target_V1 = 0.68

    # Training parameters
    n_cycles = 50
    dt = 0.001
    max_steps = 10000

    # History
    V1_free_history = []
    V1_clamped_history = []

    print("Training...")
    for cycle in range(n_cycles):
        # === FREE PHASE ===
        grid.clamped_nodes = clamped_nodes
        grid.clamped_values = clamped_values
        V_init = np.array([0.0, 0.5, 1.0])
        result_free = solver.relax_transient(
            V_init, dt=dt, max_steps=max_steps, tol=1e-10
        )

        V1_free = result_free['V_final'][1]
        V1_free_history.append(V1_free)

        Q_free = compute_Q_from_voltages(result_free['V_final'], adjacency)

        # === CLAMPED PHASE ===
        # Manually push V[1] toward target by temporarily clamping it
        # (this simulates the penalty coupling effect)
        V_init_clamped = result_free['V_final'].copy()
        V_init_clamped[1] = target_V1  # Start closer to target

        grid.clamped_nodes = np.array([0, 1, 2])  # Clamp middle node too
        grid.clamped_values = np.array([0.0, target_V1, 1.0])
        result_clamped = solver.relax_transient(
            V_init_clamped, dt=dt, max_steps=max_steps, tol=1e-10
        )

        V1_clamped = result_clamped['V_final'][1]
        V1_clamped_history.append(V1_clamped)

        Q_clamped = compute_Q_from_voltages(result_clamped['V_final'], adjacency)

        # === WEIGHT UPDATE ===
        weights_old = extract_weights(memristors, adjacency)
        weights = plasticity.update_weights(weights_old, Q_free, Q_clamped, adjacency, dt_plasticity=1.0)
        set_weights(memristors, adjacency, weights)

        if (cycle + 1) % 10 == 0:
            dw = weights - weights_old
            error = abs(V1_free - target_V1)
            print(f"  Cycle {cycle+1}/{n_cycles}:")
            print(f"    V[1] free: {V1_free:.4f} (target: {target_V1}, error: {error:.4f})")
            print(f"    Weights: w[0,1]={weights[0,1]:.4f}, w[1,2]={weights[1,2]:.4f}")
            print(f"    dw: {dw[0,1]:.6f}, {dw[1,2]:.6f}")

    # Final result
    final_error = abs(V1_free_history[-1] - target_V1)
    initial_error = abs(V1_free_history[0] - target_V1)

    print(f"\nInitial error: {initial_error:.4f}")
    print(f"Final error: {final_error:.4f}")
    print(f"Improvement: {(initial_error - final_error)/initial_error*100:.1f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    cycles = np.arange(1, n_cycles + 1)
    ax1.plot(cycles, V1_free_history, 'b-', linewidth=2, label='V[1] free')
    ax1.axhline(target_V1, color='r', linestyle='--', linewidth=2, label='Target')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('V[1]')
    ax1.set_title('Node 1 Voltage Evolution')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(cycles, np.abs(np.array(V1_free_history) - target_V1), 'g-', linewidth=2)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Absolute Error vs Target')
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('test_plasticity_simple.png', dpi=100)

    # Assertions
    assert final_error < initial_error, "Error did not decrease!"
    assert final_error < 0.1, f"Final error too large: {final_error:.4f}"

    print("✓ Test passed!")
    print("✓ Plot saved to test_plasticity_simple.png\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Simple 3-Node Chain Plasticity Test")
    print("=" * 70)

    test_three_node_chain()

    print("=" * 70)
    print("✓ Test completed!")
    print("=" * 70)
