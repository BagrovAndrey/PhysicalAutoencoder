# network/dynamics.py

"""
Voltage relaxation solver for memristive networks.

This is the canonical solver going forward. It operates on a `Grid`
(topology + boundary conditions, see grid/grid.py) and an array of
`Memristor` objects (per-edge state + local dynamics, see
memristor/memristor.py).

History note: an earlier, array/iv_function-based version of this module
(independent of the Grid/Memristor classes) existed before the
"Memristor_Class" merge (PR #1, commit 2add7d5). That merge rewrote this
file's method bodies to use Grid/Memristor but left __init__ and
_compute_time_derivative on the old signatures, so the module was
internally inconsistent from that commit onward (self.grid / self.memristors
referenced but never set; _compute_time_derivative called with a different
arity than it declared). See DEVELOPMENT.md, "History: the broken merge".
This file replaces that broken state outright with the self-consistent
Grid/Memristor version that had been developed in parallel as
network/dynamics_improved.py (now retired/merged into this file).
"""

import numpy as np
from typing import Optional, Tuple, List


class VoltageDynamics:
    """
    Solver for voltage relaxation in memristive networks.

    Implements transient dynamics:
        C_i * dV_i/dt = sum_j I_ij + I_penalty

    IMPORTANT:
    - Memristors are treated as STATIC during this relaxation (quasi-static
      assumption: tau_el << tau_plastic, per the proposal). Their internal
      state (conductance, local integrators) evolves externally, between
      calls to relax_transient - see training/trainer.py.
    """

    def __init__(self, grid, memristors: np.ndarray):
        """
        Args:
            grid: a Grid instance (topology + boundary conditions).
            memristors: (n_nodes, n_nodes) object array; memristors[i, j] is
                a Memristor instance for edge (i, j) where grid.adjacency[i, j]
                is True, and None elsewhere.
        """
        self.grid = grid
        self.memristors = memristors
        self.n_nodes = grid.n_nodes

    def relax_transient(self,
                         V_init: np.ndarray,
                         penalty_pairs: Optional[List[Tuple[int, int]]] = None,
                         beta: float = 0.0,
                         g_penalty: float = 1.0,
                         max_steps: int = 1000,
                         tol: float = 1e-6,
                         dt: float = 0.01,
                         record_history: bool = False,
                         divergence_threshold: Optional[float] = 1e6) -> dict:
        """
        Relax voltages to steady state via explicit Euler transient dynamics.

        Args:
            V_init: (n_nodes,) initial voltages.
            penalty_pairs: list of (input_idx, output_idx) for autoencoder
                reconstruction coupling. None / beta=0 -> free phase.
            beta: penalty coupling strength (0 = free phase, >0 = clamped).
            g_penalty: conductance of the penalty links.
            dt, max_steps, tol: solver parameters.
            record_history: if True, keep the full V trajectory.
            divergence_threshold: abort if |V| exceeds this (or goes
                non-finite), reporting 'diverged': True. Explicit Euler is
                only conditionally stable and blows up rather than relaxing
                once dt is too large for the stiffness set by
                beta * g_penalty - the practical boundary sits near
                dt ~ 0.002 for beta=100, g_penalty=10, i.e. only ~2x above
                the dt=0.001 used throughout the repo. Pass None to disable.

        Returns:
            dict with 'V_final', 'converged', 'diverged', 'n_steps', and
            optionally 'V_history'. Note 'converged' False means either
            "needed more steps" or "blew up" - check 'diverged' to tell
            those apart.
        """
        V = V_init.copy()
        V[self.grid.clamped_nodes] = self.grid.clamped_values

        free_nodes = np.setdiff1d(np.arange(self.n_nodes), self.grid.clamped_nodes)

        # Early exit if every node is clamped - nothing to relax, and
        # np.max on the empty free_nodes selection below would raise.
        if len(free_nodes) == 0:
            result = {'V_final': V, 'converged': True, 'diverged': False, 'n_steps': 0}
            if record_history:
                result['V_history'] = np.array([V])
            return result

        if record_history:
            V_history = [V.copy()]

        def _finish(converged, diverged, n_steps):
            result = {'V_final': V, 'converged': converged,
                      'diverged': diverged, 'n_steps': n_steps}
            if record_history:
                result['V_history'] = np.array(V_history)
            return result

        for step in range(max_steps):
            with np.errstate(over='ignore', invalid='ignore'):
                dV_dt = self._compute_time_derivative(V, penalty_pairs, beta, g_penalty)
                V[free_nodes] += dt * dV_dt[free_nodes]
                max_change = np.max(np.abs(dt * dV_dt[free_nodes]))

            if record_history:
                V_history.append(V.copy())

            # Divergence guard. Without it an unstable run is silent: V just
            # fills with inf/nan and the caller sees converged=False, which
            # is indistinguishable from "needed more steps". That matters
            # most exactly when someone is sweeping beta/g_penalty/dt.
            if divergence_threshold is not None:
                if (not np.isfinite(max_change)
                        or np.max(np.abs(V[free_nodes])) > divergence_threshold):
                    return _finish(False, True, step + 1)

            if max_change < tol:
                return _finish(True, False, step + 1)

        return _finish(False, False, max_steps)

    def _compute_time_derivative(self, V, penalty_pairs, beta, g_penalty):
        dV_dt = np.zeros(self.n_nodes)

        clamped_mask = np.zeros(self.n_nodes, dtype=bool)
        clamped_mask[self.grid.clamped_nodes] = True

        for i in range(self.n_nodes):
            if clamped_mask[i]:
                continue
            I_neighbors = 0.0

            for j in range(self.n_nodes):
                if self.grid.adjacency[i, j]:
                    mem = self.memristors[i, j]
                    if mem is not None:
                        V_drop = V[j] - V[i]
                        I_neighbors += mem.current(V_drop)

            I_penalty = 0.0
            if penalty_pairs is not None and beta > 0:
                I_penalty = self._compute_penalty_current(i, V, penalty_pairs, beta, g_penalty)

            dV_dt[i] = (I_neighbors + I_penalty) / self.grid.C[i]

        return dV_dt

    def _compute_penalty_current(self, node_idx, V, penalty_pairs, beta, g_penalty):
        """Penalty current for autoencoder input<->output coupling."""
        I_pen = 0.0
        for in_node, out_node in penalty_pairs:
            if node_idx == out_node:
                V_drop = V[in_node] - V[out_node]
                I_pen += beta * g_penalty * V_drop
            elif node_idx == in_node:
                V_drop = V[in_node] - V[out_node]
                I_pen -= beta * g_penalty * V_drop
        return I_pen

    def compute_currents(self, V: np.ndarray) -> np.ndarray:
        """Return I[i, j] = current flowing j -> i for every edge."""
        I = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                mem = self.memristors[i, j]
                if mem is not None:
                    V_drop = V[j] - V[i]
                    I[i, j] = mem.current(V_drop)
        return I
