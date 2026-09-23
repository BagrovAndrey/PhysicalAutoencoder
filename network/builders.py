# network/builders.py

"""
Small convenience helpers for assembling a Grid + array of Memristor
objects from the more familiar (adjacency, weight matrix, iv_function)
representation used throughout datasets/tests. Not part of the physical
model - just wiring, so tests and experiments don't all reimplement it.
"""

import numpy as np

from memristor.memristor import Memristor


def build_memristor_array(adjacency: np.ndarray,
                           weights: np.ndarray,
                           iv_func,
                           obs_func,
                           plast_func,
                           window_pts: int,
                           dt_local: float,
                           g_min: float = 0.01,
                           g_max: float = 1.0) -> np.ndarray:
    """
    Build an (n, n) object array of Memristor instances, one per edge where
    adjacency is True, sharing the same iv_func/obs_func/plast_func but each
    with its own initial weight taken from `weights`.

    Args:
        adjacency: (n, n) boolean connectivity.
        weights: (n, n) initial weights in [0, 1], used only where adjacency
            is True.
        iv_func: callable f(V_drop, g) -> current (e.g. from
            network/iv_characteristics.py).
        obs_func: callable obs(V_drop, I) -> local observable Q.
        plast_func: callable rule(Q_avg, w, theta) -> dw/dt (see
            training/rules.py).
        window_pts: length of each memristor's local averaging window.
        dt_local: local integration time step used by Memristor.step.
        g_min, g_max: conductance range mapped from w in [0, 1].

    Returns:
        (n, n) object array; entries are Memristor instances on edges,
        None elsewhere.
    """
    n = len(adjacency)
    memristors = np.empty((n, n), dtype=object)

    def cur_func(V, w, _iv=iv_func, _gmin=g_min, _gmax=g_max):
        g = _gmin + (_gmax - _gmin) * w
        return _iv(V, g)

    for i in range(n):
        for j in range(n):
            if adjacency[i, j]:
                memristors[i, j] = Memristor(
                    cur_func=cur_func,
                    plast_func=plast_func,
                    obs_func=obs_func,
                    window_pts=window_pts,
                    dt=dt_local,
                    w0=float(weights[i, j]),
                )

    return memristors


def build_static_memristor_array(adjacency: np.ndarray,
                                  conductances: np.ndarray,
                                  iv_func) -> np.ndarray:
    """
    Build an (n, n) object array of Memristor instances that never evolve
    (no plasticity) - for tests/experiments that only exercise the
    electrical solver, not learning.

    `conductances[i, j]` is used directly as that edge's Memristor state
    `w` (not mapped through a g_min/g_max range), and `cur_func` is
    `iv_func` unchanged, so `mem.current(V_drop) == iv_func(V_drop,
    conductances[i, j])` - i.e. this reproduces the old array/iv_function
    VoltageDynamics API's behaviour exactly, just packaged as Memristor
    objects. `mem.step()` is never expected to be called on these.
    """
    n = len(adjacency)
    memristors = np.empty((n, n), dtype=object)
    zero_rule = lambda Q_avg, w, theta: 0.0
    zero_obs = lambda V, I: 0.0
    for i in range(n):
        for j in range(n):
            if adjacency[i, j]:
                memristors[i, j] = Memristor(
                    cur_func=iv_func,
                    plast_func=zero_rule,
                    obs_func=zero_obs,
                    window_pts=1,
                    dt=1.0,
                    w0=float(conductances[i, j]),
                )
    return memristors


def build_autoencoder_topology(n_input: int, n_hidden: int, seed: int = 42,
                                w_min: float = 0.1, w_max: float = 0.9):
    """
    Standard input <-> hidden <-> output autoencoder wiring: every input node
    connects to every hidden node, every hidden node to every output node,
    no input-output edges (the reconstruction coupling is the penalty link,
    not a wire). Symmetric: adjacency[i,j] == adjacency[j,i].

    Args:
        n_input: number of input nodes (n_output is the same).
        n_hidden: number of hidden/bottleneck nodes.
        seed: RNG seed for the initial weights.
        w_min, w_max: uniform range for initial weights.

    Returns:
        (adjacency, weights, input_nodes, hidden_nodes, output_nodes,
         penalty_pairs) where penalty_pairs pairs each input node with its
         corresponding output node, as relax_transient expects.
    """
    n_output = n_input
    n_total = n_input + n_hidden + n_output
    rng = np.random.default_rng(seed)

    adjacency = np.zeros((n_total, n_total), dtype=bool)
    weights = np.zeros((n_total, n_total))

    input_nodes = np.arange(n_input)
    hidden_nodes = np.arange(n_input, n_input + n_hidden)
    output_nodes = np.arange(n_input + n_hidden, n_total)

    for group_a, group_b in ((input_nodes, hidden_nodes), (hidden_nodes, output_nodes)):
        for i in group_a:
            for j in group_b:
                w = rng.uniform(w_min, w_max)
                adjacency[i, j] = adjacency[j, i] = True
                weights[i, j] = weights[j, i] = w

    penalty_pairs = [(int(input_nodes[k]), int(output_nodes[k])) for k in range(n_input)]
    return adjacency, weights, input_nodes, hidden_nodes, output_nodes, penalty_pairs


def extract_weights(memristors: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Read the current w of every edge back into a plain (n, n) array
    (0.0 where adjacency is False / no memristor)."""
    n = len(adjacency)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adjacency[i, j] and memristors[i, j] is not None:
                w[i, j] = memristors[i, j].w
    return w


def set_weights(memristors: np.ndarray, adjacency: np.ndarray, weights: np.ndarray) -> None:
    """Write a plain (n, n) weight array back into each edge's Memristor.w
    in place. Used by the batch/explicit-contrast update path (see
    training/plasticity.py's SimplePlasticity), which computes a whole new
    weight matrix at once rather than evolving each Memristor's own
    window/plast_func online."""
    n = len(adjacency)
    for i in range(n):
        for j in range(n):
            if adjacency[i, j] and memristors[i, j] is not None:
                memristors[i, j].w = float(weights[i, j])
