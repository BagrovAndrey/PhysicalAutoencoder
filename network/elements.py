# network/elements.py

"""
Edge elements for the vectorized engine (network/equilibrium.py).

Every element is described per BRANCH. A branch b carries current from node
bj into node bi as a function of its conductance state g and the voltage
V = V[bj] - V[bi]; the same current leaves bj, so charge is conserved by
construction.

- Symmetric (odd) elements use ONE branch per undirected edge: ohmic,
  thresholded ReLU, tanh ("sigmoid"), sinh.
- Rectifying elements use TWO antiparallel branches per edge with
  independent states, each conducting only forward: the ideal rectifying
  pair (optional forward drop) and the Shockley pair (diode in series with
  the programmable filament). With equal states an ideal pair is a resistor;
  with unequal states the edge learns its own rectification direction.

Each element provides, vectorized over branches:
    current(g, V)            I
    current_and_slope(g, V)  (I, dI/dV)  - Jacobian of the Kirchhoff equations
    dI_dg(g, V)              sensitivity to the state - exact gradients (oracle)
    dPhi_dg(g, V)            the EP local observable: d(co-content)/dg

Co-content: Phi(V) = integral_0^V I dV'. Stationary states minimize the sum of
co-contents (plus the penalty term); by equilibrium propagation the gradient
of the loss w.r.t. g is (1/beta) * [dPhi/dg(nudged) - dPhi/dg(free)]. For an
ohmic edge dPhi/dg = V^2/2; for other elements it is NOT a function of V^2.

The old network/iv_characteristics.diode_iv is deliberately not ported: used
from both ends of an edge it does not conserve current (see DEVELOPMENT.md).
"""

import numpy as np
from scipy.special import wrightomega


class Element:
    name = 'element'
    branches_per_edge = 1
    smooth = True                  # False: kinks where gradients are one-sided

    def current(self, g, V):
        return self.current_and_slope(g, V)[0]

    def current_and_slope(self, g, V):
        raise NotImplementedError

    def dI_dg(self, g, V):
        raise NotImplementedError

    def dPhi_dg(self, g, V):
        raise NotImplementedError

    def describe(self):
        return self.name


class Ohmic(Element):
    name = 'ohmic'

    def current_and_slope(self, g, V):
        return g * V, g * np.ones_like(V)

    def dI_dg(self, g, V):
        return V

    def dPhi_dg(self, g, V):
        return 0.5 * V * V


class ThresholdReLU(Element):
    """I = g * sign(V) * max(|V| - V_th, 0): dead zone of width +-V_th (eq. 15 of the
    proposal). Every hop between a current-carrying node and a floating output
    costs up to V_th of signal."""
    name = 'relu'
    smooth = False

    def __init__(self, vth=0.1):
        self.vth = vth

    def current_and_slope(self, g, V):
        m = np.maximum(np.abs(V) - self.vth, 0.0)
        return g * m * np.sign(V), g * (np.abs(V) > self.vth)

    def dI_dg(self, g, V):
        return np.maximum(np.abs(V) - self.vth, 0.0) * np.sign(V)

    def dPhi_dg(self, g, V):
        return 0.5 * np.maximum(np.abs(V) - self.vth, 0.0) ** 2

    def describe(self):
        return f"relu (V_th={self.vth})"


class SmoothThresholdReLU(Element):
    """Softplus surrogate of ThresholdReLU, used only by the oracle to optimize
    across the kinks; results are always evaluated with the exact element."""
    name = 'relu-smooth'

    def __init__(self, vth=0.1, s=0.02):
        self.vth, self.s = vth, s

    def _sp(self, z):
        return self.s * np.logaddexp(0.0, z / self.s)

    def _sig(self, z):
        return 0.5 * (1.0 + np.tanh(0.5 * z / self.s))

    def current_and_slope(self, g, V):
        f = self._sp(V - self.vth) - self._sp(-V - self.vth)
        return g * f, g * (self._sig(V - self.vth) + self._sig(-V - self.vth))

    def dI_dg(self, g, V):
        return self._sp(V - self.vth) - self._sp(-V - self.vth)


def _logcosh(x):
    ax = np.abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0)


class Tanh(Element):
    """I = g * tanh(k V). The repo's historical 'sigmoid' is k = 10. Saturates for
    |V| >> 1/k: saturated edges pass no nudge (the analog of vanishing gradients),
    so 1/k should be comparable to the signal amplitude."""
    name = 'tanh'

    def __init__(self, k=10.0):
        self.k = k

    def current_and_slope(self, g, V):
        t = np.tanh(self.k * V)
        return g * t, g * self.k * (1.0 - t * t)

    def dI_dg(self, g, V):
        return np.tanh(self.k * V)

    def dPhi_dg(self, g, V):
        return _logcosh(self.k * V) / self.k

    def describe(self):
        return f"tanh (k={self.k})"


class Sinh(Element):
    """I = g * V0 * sinh(V / V0): smooth, strictly monotone, superlinear."""
    name = 'sinh'

    def __init__(self, v0=0.25):
        self.v0 = v0

    def current_and_slope(self, g, V):
        x = V / self.v0
        return g * self.v0 * np.sinh(x), g * np.cosh(x)

    def dI_dg(self, g, V):
        return self.v0 * np.sinh(V / self.v0)

    def dPhi_dg(self, g, V):
        return self.v0 ** 2 * (np.cosh(V / self.v0) - 1.0)

    def describe(self):
        return f"sinh (V0={self.v0})"


class RectifyingPair(Element):
    """Two antiparallel ideal rectifying branches per edge, each
    I = g * max(V - V_f, 0). With V_f > 0 a dead zone of +-V_f reappears."""
    name = 'rectpair'
    branches_per_edge = 2
    smooth = False

    def __init__(self, vf=0.0):
        self.vf = vf

    def current_and_slope(self, g, V):
        r = np.maximum(V - self.vf, 0.0)
        return g * r, g * (V > self.vf)

    def dI_dg(self, g, V):
        return np.maximum(V - self.vf, 0.0)

    def dPhi_dg(self, g, V):
        return 0.5 * np.maximum(V - self.vf, 0.0) ** 2

    def describe(self):
        return f"rectpair (V_f={self.vf})"


class ShockleyPair(Element):
    """
    Two antiparallel branches per edge, each a Shockley diode in series with the
    programmable filament conductance g. For branch voltage V:
        V = n V_T ln(1 + I/I_s) + I/g
    solved in closed form with the Wright omega function (W(e^x) = omega(x)).
    No hard threshold; rectification ratio R costs a turn-on drop ~ n V_T ln R.
    EP observable: dPhi/dg = (I/g)^2 / 2, half the squared voltage across the
    filament. Nudges must shift voltages by << kT/q for EP to track the gradient.
    Voltages in volts (V_T = 25.85 mV at 300 K).
    """
    name = 'shockley'
    branches_per_edge = 2
    VT = 0.02585

    def __init__(self, Is=1e-4, n=1.0):
        self.Is, self.n = Is, n

    def current_and_slope(self, g, V):
        nvt = self.n * self.VT
        x = np.log(self.Is / (nvt * g)) + (V + self.Is / g) / nvt
        I = nvt * g * np.real(wrightomega(x)) - self.Is
        slope = 1.0 / (nvt / np.maximum(I + self.Is, 1e-300) + 1.0 / g)
        return I, slope

    def dI_dg(self, g, V):
        I, slope = self.current_and_slope(g, V)
        return I / g ** 2 * slope

    def dPhi_dg(self, g, V):
        I, _ = self.current_and_slope(g, V)
        return 0.5 * (I / g) ** 2

    def describe(self):
        return f"shockley (I_s={self.Is:g}, n={self.n})"


ELEMENTS = ('ohmic', 'relu', 'sigmoid', 'tanh', 'sinh', 'rectpair', 'shockley')


def make_element(name, vth=0.1, steepness=10.0, v0=0.25, vf=0.0, Is=1e-4, n=1.0):
    """Factory used by sim.py. 'sigmoid' is the historical name of tanh(10 V)."""
    if name == 'ohmic':
        return Ohmic()
    if name == 'relu':
        return ThresholdReLU(vth)
    if name in ('sigmoid', 'tanh'):
        return Tanh(steepness)
    if name == 'sinh':
        return Sinh(v0)
    if name == 'rectpair':
        return RectifyingPair(vf)
    if name == 'shockley':
        return ShockleyPair(Is, n)
    raise ValueError(f"unknown element {name!r}; choose from {ELEMENTS}")
