# tests/test_dynamics_basic.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid.grid import Grid
from network.builders import build_static_memristor_array
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic, relu_iv


def make_solver(adjacency, conductances, iv_func, clamped_nodes, clamped_values, capacitances=None):
    """Assemble Grid + static Memristor array + VoltageDynamics, matching
    what used to be a single `VoltageDynamics(adjacency, iv_func,
    capacitances)` call in the pre-Grid/Memristor API."""
    adjacency = np.asarray(adjacency, dtype=bool)
    grid = Grid(adjacency, np.asarray(clamped_nodes), np.asarray(clamped_values, dtype=float),
                capacitances=capacitances)
    memristors = build_static_memristor_array(adjacency, conductances, iv_func)
    return VoltageDynamics(grid, memristors)


def test_single_resistor():
    """Test: single resistor between two nodes."""
    print("\n=== Test: Single Resistor ===")

    # Topology: 0 -- 1
    adjacency = np.array([[0, 1], [1, 0]])
    conductances = np.array([[0, 1], [1, 0]], dtype=float)

    V_init = np.array([1.0, 0.0])
    clamped = np.array([0, 1])
    clamped_vals = np.array([1.0, 0.0])

    solver = make_solver(adjacency, conductances, ohmic, clamped, clamped_vals, capacitances=1.0)
    result = solver.relax_transient(V_init)

    print(f"V_final: {result['V_final']}")
    print(f"Converged: {result['converged']}, Steps: {result['n_steps']}")

    # Should stay at boundary values
    assert np.allclose(result['V_final'], [1.0, 0.0])
    print("✓ Passed\n")


def test_resistor_chain():
    """Test: 3-node resistor chain, find middle voltage."""
    print("=== Test: Resistor Chain ===")
    print("Topology: 0 -- 1 -- 2")
    print("Boundary: V[0]=1, V[2]=0")
    print("Expected: V[1]=0.5\n")

    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    conductances = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=float)

    V_init = np.array([1.0, 0.5, 0.0])
    clamped = np.array([0, 2])
    clamped_vals = np.array([1.0, 0.0])

    solver = make_solver(adjacency, conductances, ohmic, clamped, clamped_vals, capacitances=1.0)
    result = solver.relax_transient(V_init, dt=0.01, tol=1e-6, record_history=True)

    print(f"V_final: {result['V_final']}")
    print(f"Converged: {result['converged']}, Steps: {result['n_steps']}")
    print(f"V[1] = {result['V_final'][1]:.6f} (expected 0.5)")

    assert result['converged']
    assert np.abs(result['V_final'][1] - 0.5) < 1e-3

    # Plot relaxation
    V_history = result['V_history']
    time = np.arange(len(V_history)) * 0.01

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, V_history[:, 0], 'b-', label='V[0] (clamped)', linewidth=2)
    ax.plot(time, V_history[:, 1], 'r-', label='V[1] (free)', linewidth=2)
    ax.plot(time, V_history[:, 2], 'g-', label='V[2] (clamped)', linewidth=2)
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    ax.set_title('Resistor Chain Relaxation')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_resistor_chain.png', dpi=100)

    print("✓ Passed")
    print("✓ Plot saved to test_resistor_chain.png\n")


def test_voltage_divider():
    """Test: voltage divider with different resistances."""
    print("=== Test: Voltage Divider ===")
    print("Topology: 0 --[R1]-- 1 --[R2]-- 2")
    print("R1=1, R2=2")
    print("Boundary: V[0]=1, V[2]=0")
    print("Expected: V[1] = R2/(R1+R2) = 2/3 ≈ 0.667\n")

    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    # Different conductances: g1=1, g2=0.5 (R1=1, R2=2)
    conductances = np.array([
        [0, 1.0, 0],
        [1.0, 0, 0.5],
        [0, 0.5, 0]
    ], dtype=float)

    V_init = np.array([1.0, 0.5, 0.0])
    clamped = np.array([0, 2])
    clamped_vals = np.array([1.0, 0.0])

    solver = make_solver(adjacency, conductances, ohmic, clamped, clamped_vals)
    result = solver.relax_transient(V_init)

    expected = 2.0 / 3.0
    print(f"V[1] = {result['V_final'][1]:.6f} (expected {expected:.6f})")

    assert np.abs(result['V_final'][1] - expected) < 1e-3
    print("✓ Passed\n")


def test_penalty_coupling():
    """Test: autoencoder penalty coupling pulls output toward input."""
    print("=== Test: Penalty Coupling ===")
    print("Setup: input node 0, output node 1 (no direct connection)")
    print("Input clamped at V=1.0")
    print("Free phase (β=0): output stays at initial V=0")
    print("Clamped phase (β>0): penalty pulls output toward input\n")

    # No direct connection between nodes
    adjacency = np.array([[0, 0], [0, 0]])
    conductances = np.zeros((2, 2))

    V_init = np.array([1.0, 0.0])
    clamped = np.array([0])
    clamped_vals = np.array([1.0])
    penalty_pairs = [(0, 1)]

    solver = make_solver(adjacency, conductances, ohmic, clamped, clamped_vals)

    # Free phase (β=0): output should stay ~0
    result_free = solver.relax_transient(
        V_init, penalty_pairs=penalty_pairs, beta=0.0, dt=0.01, max_steps=100
    )

    print(f"Free phase (β=0): V_output = {result_free['V_final'][1]:.6f}")
    assert np.abs(result_free['V_final'][1]) < 1e-3

    # Clamped phase (β=1): output pulled toward input
    result_clamped = solver.relax_transient(
        V_init, penalty_pairs=penalty_pairs, beta=1.0, g_penalty=1.0,
        dt=0.01, max_steps=1000, record_history=True
    )

    print(f"Clamped phase (β=1.0): V_output = {result_clamped['V_final'][1]:.6f}")
    assert result_clamped['V_final'][1] > 0.9  # Should be close to 1.0

    # Plot evolution
    V_history = result_clamped['V_history']
    time = np.arange(len(V_history)) * 0.01

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, V_history[:, 0], 'b-', label='V_input (clamped)', linewidth=2)
    ax.plot(time, V_history[:, 1], 'r-', label='V_output (penalty-driven)', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    ax.set_title('Penalty Coupling: Output Pulled Toward Input')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_penalty_coupling.png', dpi=100)

    print("✓ Passed")
    print("✓ Plot saved to test_penalty_coupling.png\n")


def test_nonlinear_relu():
    """Test: ReLU nonlinearity with threshold."""
    print("=== Test: ReLU I-V Characteristic ===")
    print("Topology: 0 -- 1 with ReLU(V_th=0.2)")
    print("Small voltage (0.1): no current flow")
    print("Large voltage (1.0): current flows\n")

    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    conductances = np.ones_like(adjacency, dtype=float)
    relu_thresholded = lambda V, g: relu_iv(V, g, V_th=0.2)

    # Test 1: Small voltage difference
    V_init_small = np.array([0.1, 0.05, 0.0])
    solver_small = make_solver(adjacency, conductances, relu_thresholded,
                                np.array([0, 2]), np.array([0.1, 0.0]))
    result_small = solver_small.relax_transient(V_init_small)

    print(f"Small voltage: V[1] = {result_small['V_final'][1]:.6f}")

    # Test 2: Large voltage difference
    V_init_large = np.array([1.0, 0.5, 0.0])
    solver_large = make_solver(adjacency, conductances, relu_thresholded,
                                np.array([0, 2]), np.array([1.0, 0.0]))
    result_large = solver_large.relax_transient(V_init_large)

    print(f"Large voltage: V[1] = {result_large['V_final'][1]:.6f}")
    print("✓ Passed\n")


def test_convergence_speed():
    """Test: convergence with different capacitances."""
    print("=== Test: Convergence Speed ===")
    print("Same circuit, different capacitances")
    print("Small C → fast convergence")
    print("Large C → slow convergence\n")

    adjacency = np.array([[0, 1], [1, 0]])
    conductances = np.array([[0, 1], [1, 0]], dtype=float)

    V_init = np.array([1.0, 0.5])
    clamped = np.array([0])
    clamped_vals = np.array([1.0])

    # Small capacitance
    solver_fast = make_solver(adjacency, conductances, ohmic, clamped, clamped_vals, capacitances=0.1)
    result_fast = solver_fast.relax_transient(V_init)

    # Large capacitance
    solver_slow = make_solver(adjacency, conductances, ohmic, clamped, clamped_vals, capacitances=10.0)
    result_slow = solver_slow.relax_transient(V_init)

    print(f"C=0.1: {result_fast['n_steps']} steps")
    print(f"C=10.0: {result_slow['n_steps']} steps")

    assert result_fast['n_steps'] < result_slow['n_steps']
    print("✓ Passed\n")


if __name__ == "__main__":
    """Run all tests with visual output."""
    print("=" * 60)
    print("Testing VoltageDynamics Solver")
    print("=" * 60)

    test_single_resistor()
    test_resistor_chain()
    test_voltage_divider()
    test_penalty_coupling()
    test_nonlinear_relu()
    test_convergence_speed()

    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - test_resistor_chain.png")
    print("  - test_penalty_coupling.png")
