# tests/test_network_visualization.py

"""
Test visualization functions for voltage dynamics.

Creates example plots demonstrating all visualization capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from dataclasses import dataclass

from grid.grid import Grid
from network.builders import build_static_memristor_array
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic
from visualization.dynamics_viz import (
    plot_voltage_evolution,
    animate_relaxation,
    _layered_layout
)


# ============================================================
# SHARED CONFIGURATION
# ============================================================

@dataclass
class SimConfig:
    """Shared configuration for all visualization tests."""
    # Solver parameters
    dt: float = 0.001
    tol: float = 1e-10
    max_steps_free: int = 10000
    max_steps_clamped: int = 10000
    
    # Physical parameters
    beta_free: float = 0.0
    beta_clamped: float = 1.0
    g_penalty: float = 1.0
    
    # Network architecture
    n_input: int = 4
    n_hidden: int = 3
    n_output: int = 4
    
    # Reproducibility
    random_seed: int = 42
    
    # Animation
    animation_fps: int = 20
    animation_max_frames: int = 200

CONFIG = SimConfig()


# ============================================================
# TEST FUNCTIONS
# ============================================================

def test_voltage_evolution_plot():
    """Test time series plotting."""
    print("\n=== Test: Voltage Evolution Plot ===")
    
    # Simple 3-node chain relaxation
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    conductances = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)

    grid = Grid(adjacency, np.array([0, 2]), np.array([1.0, 0.0]), capacitances=1.0)
    memristors = build_static_memristor_array(adjacency, conductances, ohmic)
    solver = VoltageDynamics(grid, memristors)

    V_init = np.array([1.0, 0.5, 0.0])
    result = solver.relax_transient(
        V_init,
        dt=CONFIG.dt * 10,  # Coarser for simple test
        record_history=True
    )
    
    # Plot with groups
    node_groups = {
        'Clamped': [0, 2],
        'Free': [1]
    }
    
    fig = plot_voltage_evolution(
        result['V_history'],
        node_groups=node_groups,
        dt=CONFIG.dt * 10,
        title='3-Node Chain Relaxation'
    )
    
    fig.savefig('test_viz_voltage_evolution.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved test_viz_voltage_evolution.png\n")
    plt.close(fig)


def test_network_states():
    """Test network visualization in free and clamped phases."""
    print("=== Test: Network States (Free vs Clamped) ===")
    
    # Build autoencoder
    n_total = CONFIG.n_input + CONFIG.n_hidden + CONFIG.n_output
    
    input_nodes = np.arange(CONFIG.n_input)
    hidden_nodes = np.arange(CONFIG.n_input, CONFIG.n_input + CONFIG.n_hidden)
    output_nodes = np.arange(CONFIG.n_input + CONFIG.n_hidden, n_total)
    
    adjacency = np.zeros((n_total, n_total), dtype=bool)
    for i in input_nodes:
        for h in hidden_nodes:
            adjacency[i, h] = adjacency[h, i] = True
    for h in hidden_nodes:
        for o in output_nodes:
            adjacency[h, o] = adjacency[o, h] = True
    
    rng = np.random.default_rng(CONFIG.random_seed)
    conductances = adjacency.astype(float) * rng.uniform(0.5, 1.5, (n_total, n_total))
    conductances = (conductances + conductances.T) / 2

    V_input = np.array([1.0, 0.0, 1.0, 0.0])
    grid = Grid(adjacency, input_nodes, V_input.copy())
    memristors = build_static_memristor_array(adjacency, conductances, ohmic)
    solver = VoltageDynamics(grid, memristors)
    V_init = np.zeros(n_total)
    V_init[input_nodes] = V_input
    V_init[hidden_nodes] = rng.uniform(0, 1, size=CONFIG.n_hidden)
    V_init[output_nodes] = rng.uniform(0, 1, size=CONFIG.n_output)

    node_groups = {
        'Input': list(input_nodes),
        'Hidden': list(hidden_nodes),
        'Output': list(output_nodes)
    }

    penalty_pairs = [(i, CONFIG.n_input + CONFIG.n_hidden + i) for i in range(CONFIG.n_input)]

    # === FREE PHASE ===
    print("  Solving free phase...")
    result_free = solver.relax_transient(
        V_init,
        dt=CONFIG.dt,
        max_steps=CONFIG.max_steps_free,
        tol=CONFIG.tol
    )

    # === CLAMPED PHASE ===
    print("  Solving clamped phase...")
    result_clamped = solver.relax_transient(
        result_free['V_final'],
        penalty_pairs=penalty_pairs,
        beta=CONFIG.beta_clamped,
        g_penalty=CONFIG.g_penalty,
        dt=CONFIG.dt,
        max_steps=CONFIG.max_steps_clamped,
        tol=CONFIG.tol
    )
    
    # === VISUALIZATION: Side-by-side comparison ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Shared layout
    G = nx.from_numpy_array(adjacency)
    pos = _layered_layout(node_groups, n_total)
    
    vmin = min(result_free['V_final'].min(), result_clamped['V_final'].min())
    vmax = max(result_free['V_final'].max(), result_clamped['V_final'].max())
    
    for idx, (result, title, beta_val) in enumerate([
        (result_free, 'Free Phase', CONFIG.beta_free),
        (result_clamped, 'Clamped Phase', CONFIG.beta_clamped)
    ]):
        ax = axes[idx]
        V = result['V_final']
        
        # Compute currents
        currents = np.zeros((n_total, n_total))
        for i in range(n_total):
            for j in range(n_total):
                if adjacency[i, j]:
                    V_drop = V[j] - V[i]
                    currents[i, j] = ohmic(V_drop, conductances[i, j])
        
        max_current = np.abs(currents).max()
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=V, node_size=500,
                              cmap='coolwarm', vmin=vmin, vmax=vmax, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
        
        # Draw current arrows
        threshold = 0.01
        for i in range(n_total):
            for j in range(i + 1, n_total):
                if adjacency[i, j]:
                    I_ij = currents[i, j]
                    
                    if np.abs(I_ij) > threshold * max_current:
                        if I_ij > 0:
                            start, end = i, j
                        else:
                            start, end = j, i
                        
                        x_start, y_start = pos[start]
                        x_end, y_end = pos[end]
                        
                        width = np.abs(I_ij) / max_current * 5
                        alpha_val = min(np.abs(I_ij) / max_current * 2, 1.0)
                        
                        arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                              arrowstyle='->', mutation_scale=20,
                                              linewidth=width, alpha=alpha_val,
                                              color='black', zorder=1)
                        ax.add_patch(arrow)
        
        # Labels
        labels = {i: f'{i}\n{V[i]:.2f}V' for i in range(n_total)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'{title} (β={beta_val})', fontsize=14)
        ax.axis('off')
    
    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.92)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Voltage (V)', fontsize=11)
    fig.savefig('test_viz_network_states.png', dpi=100, bbox_inches='tight')
    print("  ✓ Saved test_viz_network_states.png\n")
    plt.close(fig)


def test_animation():
    """Test relaxation animation showing free and clamped phases."""
    print("=== Test: Relaxation Animation ===")
    
    # Build autoencoder
    n_total = CONFIG.n_input + CONFIG.n_hidden + CONFIG.n_output
    
    input_nodes = np.arange(CONFIG.n_input)
    hidden_nodes = np.arange(CONFIG.n_input, CONFIG.n_input + CONFIG.n_hidden)
    output_nodes = np.arange(CONFIG.n_input + CONFIG.n_hidden, n_total)
    
    adjacency = np.zeros((n_total, n_total), dtype=bool)
    for i in input_nodes:
        for h in hidden_nodes:
            adjacency[i, h] = adjacency[h, i] = True
    for h in hidden_nodes:
        for o in output_nodes:
            adjacency[h, o] = adjacency[o, h] = True
    
    rng = np.random.default_rng(CONFIG.random_seed)
    conductances = adjacency.astype(float) * rng.uniform(0.5, 1.5, (n_total, n_total))
    conductances = (conductances + conductances.T) / 2

    V_input = np.array([1.0, 0.0, 1.0, 0.0])
    grid = Grid(adjacency, input_nodes, V_input.copy(), capacitances=1.0)
    memristors = build_static_memristor_array(adjacency, conductances, ohmic)
    solver = VoltageDynamics(grid, memristors)

    V_init = np.zeros(n_total)
    V_init[input_nodes] = V_input
    V_init[hidden_nodes] = rng.uniform(0, 1, size=CONFIG.n_hidden)
    V_init[output_nodes] = rng.uniform(0, 1, size=CONFIG.n_output)

    penalty_pairs = [(i, CONFIG.n_input + CONFIG.n_hidden + i) for i in range(CONFIG.n_input)]

    # === FREE PHASE ===
    print("  Simulating free phase...")
    result_free = solver.relax_transient(
        V_init,
        dt=CONFIG.dt,
        max_steps=CONFIG.max_steps_free,
        tol=CONFIG.tol,
        record_history=True
    )

    # === CLAMPED PHASE ===
    print("  Simulating clamped phase...")
    result_clamped = solver.relax_transient(
        result_free['V_final'],
        penalty_pairs=penalty_pairs,
        beta=CONFIG.beta_clamped,
        g_penalty=CONFIG.g_penalty,
        dt=CONFIG.dt,
        max_steps=CONFIG.max_steps_clamped,
        tol=CONFIG.tol,
        record_history=True
    )
    
    # Combine histories
    V_history_combined = np.vstack([
        result_free['V_history'],
        result_clamped['V_history']
    ])
    
    phase_transition_frame = len(result_free['V_history'])
    
    node_groups = {
        'Input': list(input_nodes),
        'Hidden': list(hidden_nodes),
        'Output': list(output_nodes)
    }
    
    # Plot selected nodes: first input, first hidden, first output
    nodes_to_plot = [0, CONFIG.n_input, CONFIG.n_input + CONFIG.n_hidden]
    
    print(f"  Creating animation for {n_total} nodes...")
    print(f"  Free phase: {len(result_free['V_history'])} frames")
    print(f"  Clamped phase: {len(result_clamped['V_history'])} frames")
    print(f"  Plotting voltage evolution for nodes: {nodes_to_plot}")
    
    anim = animate_relaxation(
        V_history_combined, adjacency, conductances, ohmic,
        node_groups=node_groups,
        nodes_to_plot=nodes_to_plot,
        phase_transition_frame=phase_transition_frame,
        dt=CONFIG.dt,
        max_frames=CONFIG.animation_max_frames,
        output_file='test_viz_relaxation.gif',
        fps=CONFIG.animation_fps
    )
    
    print("  ✓ Saved test_viz_relaxation.gif\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Visualization Functions")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  dt={CONFIG.dt}, tol={CONFIG.tol}")
    print(f"  max_steps: free={CONFIG.max_steps_free}, clamped={CONFIG.max_steps_clamped}")
    print(f"  beta_clamped={CONFIG.beta_clamped}, g_penalty={CONFIG.g_penalty}")
    print(f"  Network: {CONFIG.n_input}→{CONFIG.n_hidden}→{CONFIG.n_output}")
    print()
    
    test_voltage_evolution_plot()
    test_network_states()
    test_animation()
    
    print("=" * 70)
    print("✓ All visualization tests passed!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - test_viz_voltage_evolution.png")
    print("  - test_viz_network_states.png")
    print("  - test_viz_relaxation.gif")
