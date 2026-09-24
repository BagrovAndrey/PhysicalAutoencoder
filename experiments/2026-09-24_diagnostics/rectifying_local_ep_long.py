"""Long local-EP runs on rectifying pairs, 3x3 B&S, n_h=6:
   (a) all 14 patterns, 300 epochs;  (b) 10 train / 4 held out, 300 epochs."""
import sys, json, time
from multiprocessing import Pool
import numpy as np
import rectifying_pairs as O


def run(cfg):
    held, epochs = cfg
    X = O.bars_stripes(3)
    test = X[held] if held else None
    train = np.delete(X, held, axis=0) if held else X
    net = O.make_net(9, 6); i, j = net['br'].T
    w = np.random.default_rng(1).uniform(0.1, 0.9, len(net['br']))
    Phi = lambda d: 0.5 * np.maximum(d, 0.0) ** 2
    rng = np.random.default_rng(0); hist = []
    for ep in range(epochs):
        for p in rng.permutation(len(train)):
            x = train[p]; K = O.Kmat(net, w)
            V0 = O.solve(net, K, x, 0.0, euler_steps=4000)
            Vb = O.solve(net, K, x, 0.0, V0=V0, beta_gp=0.1, euler_steps=4000)
            w = np.clip(w - 0.5 * (Phi(Vb[j] - Vb[i]) - Phi(V0[j] - V0[i])) / 0.1, 0, 1)
        if ep % 25 == 24:
            row = dict(epoch=ep + 1, train=O.evaluate(net, w, train, 0.0))
            if test is not None:
                row['test'] = O.evaluate(net, w, test, 0.0)
            hist.append(row)
            print(json.dumps(dict(held=held, **{k: (v if k == 'epoch' else
                  {m: round(v[m], 4) for m in ('mse', 'exact', 'bits', 'margin')})
                  for k, v in row.items()})), flush=True)
    return dict(held=held, hist=hist, w=w.tolist())


if __name__ == '__main__':
    t0 = time.perf_counter()
    with Pool(2) as pool:
        res = pool.map(run, [([], 300), ([0, 3, 11, 13], 300)])
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(f"ALL DONE {time.perf_counter()-t0:.0f}s", flush=True)
