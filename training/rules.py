# training/rules.py

"""
Plasticity rules for Memristor.plast_func.

Rule signature (matches Memristor.dw_dt -> self.plast(Q_avg, w, theta)):

    rule(Q_avg, w, theta) -> dw/dt

`theta` is whatever reference value the memristor's `theta` attribute
currently holds - see memristor/memristor.py's class docstring for why a
reference is needed at all (a sign-definite Q_avg can only ever push w one
way; without something to compare it to there is no way to potentiate here
and depress there). Today `theta` is always the single, network-wide value
maintained by training/trainer.py (Trainer.theta) - realized physically as
one shared slow field, not per-edge state (see DEVELOPMENT.md, "Global vs
local theta" for the reasoning and the literature this is grounded in).

To use a different rule, pass a different callable as `plast_func` when
constructing a Memristor (or bind hyperparameters with `make_rule` below) -
no other code needs to change.
"""

import functools


def global_threshold_rule(Q_avg: float, w: float, theta: float,
                           eta: float = 1.05, gamma: float = 0.001,
                           sign: float = -1.0) -> float:
    """
    Default rule. BCM-style: plasticity is driven by how far the local,
    fast-windowed observable Q_avg sits from the slow reference theta,
    rather than by Q_avg's raw (sign-definite) magnitude.

        dw/dt = sign * eta * (Q_avg - theta) * (1 - w) - gamma * w

    `sign` is an empirically-tuned convention (see DEVELOPMENT.md,
    "Plasticity Rule - Design Decisions and Why", point 1) - it has not
    been derived from first principles, only tuned so growth moves w in
    the direction that improves reconstruction on the cases tried so far.
    `(1 - w)` is the saturation term (finite conductance range); `gamma`
    is spontaneous decay.
    """
    return sign * eta * (Q_avg - theta) * (1 - w) - gamma * w


def make_rule(rule_fn=global_threshold_rule, **hyperparams):
    """
    Bind hyperparameters (eta, gamma, sign, ...) to a rule function,
    returning a plain rule(Q_avg, w, theta) callable suitable for
    Memristor(plast_func=...).

    Example:
        plast_func = make_rule(global_threshold_rule, eta=0.5, gamma=0.0)
    """
    return functools.partial(rule_fn, **hyperparams)


# --- Local observables (Memristor obs_func) ---------------------------------
#
# obs(V_drop, I) -> Q, the instantaneous local quantity each memristor
# integrates over its own window. Which one is "right" is an open question
# (see DEVELOPMENT.md, Open Problems #1) - they are collected here so
# swapping between them is a one-line change, same as swapping rules.

def quadratic_observable(V_drop: float, I: float) -> float:
    """
    Q = (dV)^2. The default, and what the explicit contrastive path uses
    (training.plasticity.compute_Q_from_voltages), so the online and batch
    paths integrate the same quantity and stay comparable.

    Note it is sign-definite, which is exactly why a reference value
    (theta) is needed at all - see the module docstring above.
    """
    return V_drop ** 2


def linear_observable(V_drop: float, I: float) -> float:
    """
    Q = dV. Kept because DEVELOPMENT.md's design notes long claimed this was
    the observable actually in use. It is not, and switching to it
    measurably worsens reconstruction - see experiments/plateau_baseline.py.
    """
    return V_drop


def power_observable(V_drop: float, I: float) -> float:
    """
    Q = dV * I, the locally dissipated power - arguably the most physical
    choice, and sign-definite for a passive element. Listed as a candidate
    in DEVELOPMENT.md's Open Problems but not yet evaluated.
    """
    return V_drop * I
