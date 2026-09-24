"""
Why the single-pattern plateau sits at MSE ~0.03-0.05: the transport threshold.

(a) Hand-wired IDEAL 4-2-4 network for [1,0,1,0] (strong edges exactly where
    they should be), fresh inference from V_free = 0, for several V_th and two
    leak levels. With leaks ~gone the error is exactly 2*V_th per output.
(b) Mechanism: inside the dead zone a strong edge carries no current, so a
    node drifts wherever the WEAKEST leak pushes it, until it is V_th away.
    Gradient flow systematically parks edges exactly at |dV| = V_th (the
    kink), so the loss is only piecewise smooth at the operating point.
    With THIS solver the implicit-function gradient still matches finite
    differences (printed below); an earlier implementation with a different
    warm-start trajectory did not. Treat gradient-based analysis of the
    dead-zone model with care - a caveat, not a result.

Run: python3 threshold_floor.py
"""
import numpy as np
import symmetric_elements as O


def ideal_4_2_4():
    net = O.make_arch(4, 2, 'plain')          # in 0-3, hidden 4-5, out 6-9
    w = np.zeros(len(net['edges']))
    for k, (u, v) in enumerate(net['edges']):
        h, o = (v, u) if v in (4, 5) else (u, v)
        idx = o if o < 4 else o - 6
        w[k] = 1.0 if (h == 4) == (idx in (0, 2)) else 0.0
    return net, w


if __name__ == '__main__':
    x = np.array([[1.0, 0.0, 1.0, 0.0]])
    net, w = ideal_4_2_4()
    print("(a) ideal 4-2-4 wiring for [1,0,1,0], inference from V=0")
    for gmin in (0.01, 1e-4):
        O.G_MIN = gmin
        for vth in (0.0, 0.05, 0.1, 0.2):
            ev = O.evaluate(net, w, x, O.make_iv('relu', vth=vth))
            print(f"   g_min={gmin:<7} V_th={vth:<5} out={ev['outs'][0]}  MSE={ev['mse']:.4f}"
                  f"   (2*V_th)^2={(2 * vth) ** 2:.4f}")
    O.G_MIN = 0.01

    print("\n(b) gradient check (implicit-function vs finite differences), 2x2 B&S, random w")
    X = O.bars_stripes(2)
    net = O.make_arch(4, 3, 'plain')
    w = np.random.default_rng(5).uniform(0.2, 0.8, len(net['edges']))
    for vth in (1e-9, 0.02, 0.1):
        iv = O.make_iv('relu', vth=vth)
        _, g = O.loss_grad(net, w, X, iv)
        num = np.zeros_like(w); h = 1e-6
        for k in range(len(w)):
            wp = w.copy(); wp[k] += h; wm = w.copy(); wm[k] -= h
            num[k] = (O.loss_grad(net, wp, X, iv)[0] - O.loss_grad(net, wm, X, iv)[0]) / (2 * h)
        print(f"   V_th={vth:<6} relative error {np.max(np.abs(g - num)) / np.max(np.abs(num)):.1e}")

    print("\n   where the relaxed state sits (V_th=0.1):")
    iv = O.make_iv('relu', vth=0.1); G = O.Gmat(net, w); a, b = net['edges'].T
    for xx in X:
        V = O.solve(net, G, xx, iv, euler_steps=20000)
        d = np.abs(V[a] - V[b])
        dead = [int(i) for i in net['free'] if not np.any((d > 0.1)[(a == i) | (b == i)])]
        print(f"   x={xx.astype(int)}  edges parked at |dV|=V_th: {int(np.sum(np.abs(d - 0.1) < 1e-3)):2d}"
              f"   free nodes with no conducting edge: {dead}   out={np.round(V[net['out']], 3)}")
