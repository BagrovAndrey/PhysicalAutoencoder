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

import warnings
from typing import Optional

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
                 tau_theta: float = 400.0,
                 relax_max_steps: Optional[int] = None,
                 check_window: bool = True):
        """
        Args (beyond the obvious):
            relax_max_steps: step budget for each electrical relaxation.
                Defaults to exposure_time/dt, which is the "fixed exposure
                time" protocol - but note that couples two independent
                things: how long the network is driven (plasticity
                timescale) and how far the solver is allowed to relax.
                Set this explicitly to vary one without the other.
            check_window: warn if each memristor's averaging window does not
                span exactly one phase. See the warning text for why that
                matters - it is the difference between learning and not.
        """
        self.grid = grid
        self.memristors = memristors
        self.penalty_pairs = penalty_pairs
        self.relax_max_steps = relax_max_steps

        if check_window:
            self._check_window_matches_phase(micro_steps_per_phase)

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

    def _check_window_matches_phase(self, micro_steps_per_phase: int) -> None:
        """
        Each Memristor averages Q over a boxcar window of `window_pts`
        samples, and one phase is `micro_steps_per_phase` samples long.
        That ratio is not a free tuning knob - it decides whether learning
        happens at all (measured 2026-09-23, 4->3->4 / [1,0,1,0]):

            window_pts / micro_steps = 1.0   ->  MSE 0.242 -> 0.033
            window_pts / micro_steps = 0.5   ->  MSE 0.242 -> 0.493 (no learning)

        Why: V is held constant within a phase, so a window shorter than the
        phase saturates at the current phase's Q and carries no memory of
        the other phase. The lag of a window exactly one phase long is what
        encodes the per-edge free/clamped contrast that plasticity needs.
        Physically this says the device's internal relaxation time must be
        matched to the drive period - it is a constraint on the element, not
        a hyperparameter.
        """
        windows = {mem.window_pts for mem in self.memristors.ravel() if mem is not None}
        if not windows:
            return
        bad = {w for w in windows if w != micro_steps_per_phase}
        if bad:
            warnings.warn(
                f"memristor window_pts {sorted(bad)} != micro_steps_per_phase "
                f"({micro_steps_per_phase}). The averaging window should span "
                f"exactly one phase, otherwise the free/clamped contrast is lost "
                f"and the network will not learn (it will still run and stay "
                f"finite, which is what makes this easy to miss). "
                f"Pass check_window=False to silence.",
                RuntimeWarning, stacklevel=3)

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
            max_steps=self.relax_max_steps
            if self.relax_max_steps is not None
            else int(self.exposure_time_free / self.dt),
        )
        V_free = free['V_final']
        self._evolve_phase(V_free, self.exposure_time_free)

        clamped = self.solver.relax_transient(
            V_free,
            penalty_pairs=self.penalty_pairs,
            beta=self.beta_clamped, g_penalty=self.g_penalty,
            dt=self.dt, tol=self.tol,
            max_steps=self.relax_max_steps
            if self.relax_max_steps is not None
            else int(self.exposure_time_clamped / self.dt),
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
