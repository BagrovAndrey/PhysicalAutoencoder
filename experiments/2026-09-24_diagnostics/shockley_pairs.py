"""
Realistic rectifying branch: Shockley diode in series with a programmable
filament conductance g. Branch current for total branch voltage V (V>0 forward):
    V = n*V_T*ln(1 + I/I_s) + I/g
closed form via the Wright omega function  (W(e^x) = omega(x)):
    I = n*V_T*g * omega( ln(I_s/(n*V_T*g)) + (V + I_s/g)/(n*V_T) ) - I_s
No hard threshold: near V=0 the branch is linear; strong rectification only for
|V| >> n*V_T; reverse current saturates at -I_s whatever the filament state.
Voltages in volts (V_T = 25.9 mV at 300 K); conductances in the model's units.

Learning-relevant derivatives (at fixed branch voltage V):
    dI/dg      = (I/g^2) * dI/dV
    dPhi/dg    = (I/g)^2 / 2      Phi = co-content of the branch = I*V - content(I)
The EP observable of a branch is therefore half the squared voltage across its
FILAMENT (I/g), not across the whole element. For an ideal pair (no diode) it
reduces to max(V,0)^2/2, as used in rectifying_pairs.py.
"""
import numpy as np
from scipy.special import wrightomega
import rectifying_pairs as R
from rectifying_pairs import make_net, handbuilt_w
from symmetric_elements import bars_stripes

VT = 0.02585


def branch(g, V, Is, n):
    nvt = n * VT
    x = np.log(Is / (nvt * g)) + (V + Is / g) / nvt
    I = nvt * g * np.real(wrightomega(x)) - Is
    dIdV = 1.0 / (nvt / np.maximum(I + Is, 1e-300) + 1.0 / g)   # deep reverse: -> 0
    return I, dIdV


def branch_dg(g, V, Is, n):
    I, dIdV = branch(g, V, Is, n)
    return I / g ** 2 * dIdV


def cocontent_dg(g, V, Is, n):
    I, _ = branch(g, V, Is, n)
    return 0.5 * (I / g) ** 2


def F_J(net, K, V, Is, n, x=None, beta_gp=0.0):
    D = V[None, :] - V[:, None]                   # V_j - V_i
    Kg = np.where(K > 0, K, 1.0)
    I1, d1 = branch(Kg, D, Is, n)                 # K[i,j]: conducts j -> i
    I2, d2 = branch(Kg.T, -D, Is, n)              # K[j,i]: conducts i -> j
    mask = (K > 0).astype(float)
    F = (mask * I1).sum(1) - (mask.T * I2).sum(1)
    W = mask * d1 + mask.T * d2
    if beta_gp:
        F[net['out']] += beta_gp * (x - V[net['out']])
    return F, W


def solve(net, K, x, Is, n, V0=None, beta_gp=0.0, steps=3000):
    free, out = net['free'], net['out']
    V = np.zeros(net['n']) if V0 is None else V0.copy()
    V[net['inp']] = x
    dt = 0.9 / (2 * (K + K.T).sum(1).max() + abs(beta_gp))
    for s in range(steps):
        F, _ = F_J(net, K, V, Is, n, x, beta_gp); V[free] += dt * F[free]
        if s % 50 == 0 and np.abs(F[free]).max() < 1e-12: break
    for _ in range(100):
        F, W = F_J(net, K, V, Is, n, x, beta_gp); r = F[free]; nr = np.abs(r).max()
        if nr < 1e-13: break
        J = W - np.diag(W.sum(1))
        if beta_gp:
            J[out, out] -= beta_gp
        step = -np.linalg.lstsq(J[np.ix_(free, free)], r, rcond=1e-12)[0]
        t = 1.0
        for _ in range(40):
            Vn = V.copy(); Vn[free] += t * step
            if np.abs(F_J(net, K, Vn, Is, n, x, beta_gp)[0][free]).max() < nr: V = Vn; break
            t *= 0.5
        else: break
    return V


def loss_grad(net, w, X, vsig, Is, n, Vc=None):
    """MSE of normalized outputs (V_out / vsig) and its exact gradient w.r.t. w."""
    K = R.Kmat(net, w); free, out = net['free'], net['out']
    i, j = net['br'].T
    P, no = len(X), len(out)
    L, g = 0.0, np.zeros(len(i))
    for p, x in enumerate(X):
        warm = Vc is not None and Vc[p] is not None
        V = solve(net, K, x * vsig, Is, n, V0=Vc[p] if warm else None, steps=200 if warm else 3000)
        if Vc is not None:
            Vc[p] = V
        err = V[out] / vsig - x
        L += (err ** 2).sum() / (P * no)
        dLdV = np.zeros(net['n']); dLdV[out] = 2 * err / (P * no) / vsig
        _, W = F_J(net, K, V, Is, n)
        J = W - np.diag(W.sum(1))
        lam = np.zeros(net['n'])
        lam[free] = np.linalg.lstsq(J[np.ix_(free, free)].T, dLdV[free], rcond=1e-12)[0]
        g += -(lam[i] - lam[j]) * branch_dg(K[i, j], V[j] - V[i], Is, n) * (R.G_MAX - R.G_MIN)
    return L, g


def adam(net, X, vsig, Is, n, w0, iters=450, lr=0.03):
    w = w0.copy(); m = np.zeros_like(w); v = np.zeros_like(w)
    Vc = [None] * len(X); best = (np.inf, w.copy())
    for t in range(1, iters + 1):
        L, gr = loss_grad(net, w, X, vsig, Is, n, Vc)
        if L < best[0]:
            best = (L, w.copy())
        m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr * gr
        w = np.clip(w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), 0, 1)
    return best


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
