"""Checks model.backward() against finite differences on random weights."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import MicroOpus
from train import CONFIG, init_weights

rng = np.random.default_rng(0)
cfg = dict(CONFIG, V=12)
model = MicroOpus(W=init_weights(cfg, 12, rng), cfg=cfg, vocab=[str(i) for i in range(12)])
x = rng.integers(0, 12, (3, 7))
y = rng.integers(0, 12, (3, 7))
tr = {}
model.forward(x, tr)
G = model.backward(tr, y)
worst = 0
for name, w in model.W.items():
    for _ in range(4):
        idx = tuple(rng.integers(0, s) for s in w.shape)
        old = w[idx]
        w[idx] = old + 1e-5
        up = model.cross_entropy(model.forward(x), y)
        w[idx] = old - 1e-5
        dn = model.cross_entropy(model.forward(x), y)
        w[idx] = old
        num = (up - dn) / 2e-5
        worst = max(worst, abs(num - G[name][idx]) / max(1e-6, abs(num) + abs(G[name][idx])))
print(f"worst relative gradient error: {worst:.2e}")
sys.exit(0 if worst < 1e-4 else 1)
