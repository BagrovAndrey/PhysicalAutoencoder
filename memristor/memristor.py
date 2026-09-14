import numpy as np
from typing import Optional
import random

class Memristor:
    """
    A single memristive edge: local conductance state `w`, a fast local
    window-average integrator of an observable `Q`, and a plasticity rule
    `plast_func` that turns that average into dw/dt.

    theta: `plast_func` is called as `plast_func(Q_avg, w, theta)`. `theta`
    is a reference value the local Q_avg is compared against - without it,
    a sign-definite Q (e.g. (dV)^2) can only ever push w in one direction
    (see DEVELOPMENT.md, "Global vs local theta"). Today `theta` is always
    set from the outside (see Memristor.theta / set_theta below) by a single
    network-wide owner (training/trainer.py) - a stand-in for a shared
    physical slow field (e.g. substrate temperature or a common bias rail),
    not something this class computes for itself.

    This is a deliberate extension point, not an oversight: a future
    version could have each Memristor track its own local low-pass average
    of its own Q_avg instead of taking theta from outside. That only
    requires changing where `self.theta` gets its value (e.g. updating it
    inside `step`) - the `plast_func(Q_avg, w, theta)` call signature, and
    every rule written against it (training/rules.py), stays the same.
    """

    def __init__(self, cur_func, plast_func, obs_func, window_pts, dt, w0: Optional[float] = None, Q_history: Optional[np.ndarray] = None, theta: float = 0.0):

        self.dt = dt
        self.window_pts = window_pts
        self.theta = theta  # see class docstring: externally-supplied by default


        # Setup weights
        
        if w0 is None:
            self.w = random.random()
        else:
            self.w = w0
        
        # history setup

        if window_pts <= 0:
            raise ValueError("window_pts must be positive")

        if Q_history is None:
            self.Q_history = np.zeros(self.window_pts)
        elif np.isscalar(Q_history):
            self.Q_history = np.full(self.window_pts, Q_history)
        else:
            if len(Q_history) == self.window_pts:
                self.Q_history = np.array(Q_history)
            else:
                raise ValueError("Wrong history record dimension")
        
        
        
        self.cur = cur_func
        self.plast = plast_func
        self.obs = obs_func

        self.running_sum = np.sum(self.Q_history)
        self.idx = 0


    def current(self, V):
        return self.cur(V, self.w)

    

    def update_window(self, Q):
        old = self.Q_history[self.idx]
        
        # update running sum
        self.running_sum += Q - old
        
        # overwrite oldest value
        self.Q_history[self.idx] = Q

        # move pointer
        self.idx = (self.idx + 1) % self.window_pts

    
    def Q_avg(self):
        return self.running_sum / self.window_pts

    def set_theta(self, theta: float):
        """Set the reference value plast_func compares Q_avg against.
        Called by the network-wide owner (training/trainer.py) each step."""
        self.theta = theta

    def dw_dt(self):
        Q_av = self.Q_avg()
        return self.plast(Q_av, self.w, self.theta)

    def step(self, V):
        Q = self.obs(V, self.current(V))

        # update memory
        self.update_window(Q)

        # evolve state
        self.w += self.dt * self.dw_dt()

        # enforce bounds
        self.w = np.clip(self.w, 0.0, 1.0)