# training/trainer.py

"""
Couples the electrical solver (Grid + Memristor + VoltageDynamics) to a
free/clamped equilibrium-propagation-style learning protocol.

This is also where the ONE global, network-wide slow threshold `theta`
lives (see memristor/memristor.py and training/rules.py for why it exists):
Trainer maintains it as a single scalar, a stand-in for a shared physical
slow field (e.g. substrate temperature or a common bias line) that every
edge's plasticity rule reads the same value from - not a per-edge state.
Per-edge (local) theta remains a possible future extension (see the
Memristor docstring); it would only change where each memristor's `theta`
attribute gets set, not this class's public interface.

This was the missing "training/trainer.py" flagged in DEVELOPMENT.md's
roadmap - previously test_plasticity.py ran an equivalent loop inline, and
tests/test_with_memristors.py sketched a Memristor-based loop that did not
match either the Memristor or VoltageDynamics constructors that ended up
being implemented.
"""

import numpy as np

from network.dynamics import VoltageDynamics


class Trainer:
    """
    Orchestrates free/clamped learning cycles for a Grid + Memristor
    network built with network.builders.build_memristor_array.
    """

    def __init__(self, grid, memristors: np.ndarray, penalty_pairs,
                 beta_free: float = 0.0, beta_clamped: float = 1.0, g_penalty: float = 1.0,
                 exposure_time_free: float = 10.0, exposure_time_clamped: float = 10.0,
                 micro_steps_per_phase: int = 200,
                 dt: float = 1e-3, tol: float = 1e-10,
                 tau_theta: float = 400.0):
        self.grid = grid
        self.memristors = memristors
        self.penalty_pairs = penalty_pairs

        self.beta_free = beta_free
        self.beta_clamped = beta_clamped
        self.g_penalty = g_penalty

        self.exposure_time_free = exposure_time_free
        self.exposure_time_clamped = exposure_time_clamped
        self.micro_steps_per_phase = micro_steps_per_phase

        self.dt = dt
        self.tol = tol
        self.tau_theta = tau_theta

        self.solver = VoltageDynamics(grid, memristors)

        # The one shared, slowly-adapting reference value (see module
        # docstring). Starts at 0.0 and is updated in _evolve_phase.
        self.theta = 0.0

    def _mean_Q_instant(self, V: np.ndarray) -> float:
        """Network-wide average of the instantaneous local observable
        across all edges - what the shared theta slowly tracks."""
        adjacency = self.grid.adjacency
        n = len(V)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(n):
                mem = self.memristors[i, j]
                if adjacency[i, j] and mem is not None:
                    V_drop = V[j] - V[i]
                    total += mem.obs(V_drop, mem.current(V_drop))
                    count += 1
        return total / count if count else 0.0

    def _evolve_phase(self, V_const: np.ndarray, exposure_time: float):
        """
        Advance every memristor's local integrator, and the shared theta,
        for one phase's exposure time. V is held fixed at its relaxed value
        for the duration (quasi-static assumption tau_el << tau_plastic,
        already used throughout the model) rather than re-solving the
        electrical relaxation at every micro-step.
        """
        if self.micro_steps_per_phase <= 0:
            return
        dt_micro = exposure_time / self.micro_steps_per_phase
        alpha_theta = dt_micro / self.tau_theta

        adjacency = self.grid.adjacency
        n = len(V_const)

        for _ in range(self.micro_steps_per_phase):
            mean_Q = self._mean_Q_instant(V_const)
            self.theta = (1 - alpha_theta) * self.theta + alpha_theta * mean_Q

            for i in range(n):
                for j in range(n):
                    mem = self.memristors[i, j]
                    if adjacency[i, j] and mem is not None:
                        mem.set_theta(self.theta)
                        V_drop = V_const[j] - V_const[i]
                        mem.step(V_drop)

    def run_cycle(self, pattern: np.ndarray) -> dict:
        """
        One free + clamped learning cycle for a given input pattern.
        Mutates grid.clamped_values in place, so consecutive calls can
        present different patterns from the same dataset.

        Returns a dict with the free/clamped final voltages and outputs,
        MSE against the pattern, and the current theta - enough for a
        training loop to log progress without re-deriving it.
        """
        pattern = np.asarray(pattern, dtype=float)
        self.grid.clamped_values = pattern

        V_init = np.zeros(self.grid.n_nodes)
        V_init[self.grid.clamped_nodes] = pattern

        free = self.solver.relax_transient(
            V_init,
            beta=self.beta_free,
            dt=self.dt, tol=self.tol,
            max_steps=int(self.exposure_time_free / self.dt),
        )
        V_free = free['V_final']
        self._evolve_phase(V_free, self.exposure_time_free)

        clamped = self.solver.relax_transient(
            V_free,
            penalty_pairs=self.penalty_pairs,
            beta=self.beta_clamped, g_penalty=self.g_penalty,
            dt=self.dt, tol=self.tol,
            max_steps=int(self.exposure_time_clamped / self.dt),
        )
        V_clamped = clamped['V_final']
        self._evolve_phase(V_clamped, self.exposure_time_clamped)

        output_nodes = np.array([out for _, out in self.penalty_pairs])
        mse_free = float(np.mean((pattern - V_free[output_nodes]) ** 2))

        return {
            'V_free': V_free,
            'V_clamped': V_clamped,
            'mse_free': mse_free,
            'theta': self.theta,
            'free_converged': free['converged'],
            'clamped_converged': clamped['converged'],
        }
