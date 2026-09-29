"""
Realistic rectifying branch: Shockley diode in series with a programmable
filament conductance g. Branch current for total branch voltage V (V>0 forward):
    V = n*V_T*ln(1 + I/I_s) + I/g
closed form via the Wright omega function  (W(e^x) = omega(x)):
    I = n*V_T*g * omega( ln(I_s/(n*V_T*g)) + (V + I_s/g)/(n*V_T) ) - I_s
No hard threshold: near V=0 the branch is linear; strong rectification only for
|V| >> n*V_T; reverse current saturates at -I_s whatever the filament state.
Voltages in volts (V_T = 25.9 mV at 300 K); conductances in the model's units.
"""
import numpy as np
from scipy.special import wrightomega
from rectifying_pairs import make_net, handbuilt_w
from symmetric_elements import bars_stripes

VT = 0.02585


def branch(g, V, Is, n):
    nvt = n * VT
    x = np.log(Is / (nvt * g)) + (V + Is / g) / nvt
    I = nvt * g * np.real(wrightomega(x)) - Is
    dIdV = 1.0 / (nvt / np.maximum(I + Is, 1e-300) + 1.0 / g)   # deep reverse: -> 0
    return I, dIdV


def solve(net, K, x, Is, n, steps=3000):
    free = net['free']; V = np.zeros(net['n']); V[net['inp']] = x
    def F_J(V):
        D = V[None, :] - V[:, None]               # V_j - V_i
        Kg = np.where(K > 0, K, 1.0)
        I1, d1 = branch(Kg, D, Is, n)             # K[i,j]: conducts j -> i
        I2, d2 = branch(Kg.T, -D, Is, n)          # K[j,i]: conducts i -> j
        mask = (K > 0).astype(float)
        F = (mask * I1).sum(1) - (mask.T * I2).sum(1)
        W = mask * d1 + mask.T * d2
        return F, W
    dt = 0.9 / (2 * (K + K.T).sum(1).max())
    for s in range(steps):
        F, _ = F_J(V); V[free] += dt * F[free]
        if s % 50 == 0 and np.abs(F[free]).max() < 1e-12: break
    for _ in range(100):
        F, W = F_J(V); r = F[free]; nr = np.abs(r).max()
        if nr < 1e-13: break
        J = W - np.diag(W.sum(1))
        step = -np.linalg.lstsq(J[np.ix_(free, free)], r, rcond=1e-12)[0]
        t = 1.0
        for _ in range(40):
            Vn = V.copy(); Vn[free] += t * step
            if np.abs(F_J(Vn)[0][free]).max() < nr: V = Vn; break
            t *= 0.5
        else: break
    return V


def evaluate(net, K, X, Vsig, Is, n):
    outs = np.array([solve(net, K, x * Vsig, Is, n)[net['out']] for x in X]) / Vsig
    return (float(np.mean((outs - X) ** 2)),
            float(np.mean(np.all((outs > 0.5) == (X > 0.5), axis=1))),
            float(np.min(np.abs(outs - 0.5))))


if __name__ == '__main__':
    X = bars_stripes(3); net = make_net(9, 6)
    print("hand-built 9-6-9 on 3x3 B&S, each branch = filament g (0.01..1) + Shockley diode")
    print("best over pull-up in {0.05, 0.2, 0.5}; exact = fraction of 14 patterns fully right\n")
    print(f"{'n':>4} {'I_s/g_max':>10} {'~V_on*':>7} |" +
          "".join(f"  V_sig={v:<4}   " for v in (0.3, 1.0, 3.0)))
    print("-" * 70)
    for n in (1.0, 1.5):
        for Is in (1e-2, 1e-4, 1e-6, 1e-8):
            von = n * VT * np.log(0.5 / Is)
            row = f"{n:>4} {Is:>10.0e} {von:>6.2f}V |"
            for Vsig in (0.3, 1.0, 3.0):
                best = None
                for pu in (0.05, 0.2, 0.5):
                    w = handbuilt_w(net, 3, pu, 0.01)
                    K = np.zeros((net['n'], net['n']))
                    i, j = net['br'].T; K[i, j] = 0.01 + 0.99 * w
                    ev = evaluate(net, K, X, Vsig, Is, n)
                    if best is None or (ev[1], ev[2]) > (best[1], best[2]): best = ev
                row += f"  {best[1]*100:>3.0f}% m={best[2]:+.2f} "
            print(row)
    print("\n*V_on = n*V_T*ln(0.5*g_max/I_s): drop at which a strong branch carries a typical current")
