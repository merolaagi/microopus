"""
Trains Micro Opus from random noise and writes weights/model.json.

This is the only place weights ever change. model.py only reads them.
Uses the tiny `autograd` package so the whole thing runs on numpy.
"""
import json, random
from pathlib import Path
import autograd.numpy as np
from autograd import grad
import numpy as onp

random.seed(7); onp.random.seed(7)

# ---------- corpus: a tiny world with long-range dependencies ----------
subj = {
    "cat":  ["sat", "slept"],
    "dog":  ["ran", "slept"],
    "bird": ["flew", "sang"],
    "fish": ["swam"],
    "kid":  ["ran", "sat", "sang"],
}
place = {
    "sat":   ["on", ["mat", "bed", "chair"]],
    "slept": ["on", ["mat", "bed"]],
    "ran":   ["to", ["park", "house", "store"]],
    "flew":  ["over", ["tree", "house"]],
    "sang":  ["in", ["tree", "park"]],
    "swam":  ["in", ["pond", "lake"]],
}
adjs = ["big", "small", "happy"]

def sentence():
    s = random.choice(list(subj))
    v = random.choice(subj[s])
    p, objs = place[v]
    o = random.choice(objs)
    words = ["the"]
    if random.random() < 0.35:
        words.append(random.choice(adjs))
    words += [s, v, p, "the", o, "."]
    return words

vocab = ["<s>"]
for w in ["the", ".", *adjs, *subj, *place, "on", "to", "over", "in"]:
    if w not in vocab: vocab.append(w)
for v in place.values():
    for o in v[1]:
        if o not in vocab: vocab.append(o)
stoi = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

T = 16
def sample_seq():
    toks = ["<s>"]
    while len(toks) < T + 1:
        toks += sentence()
    return [stoi[w] for w in toks[:T + 1]]

data = onp.array([sample_seq() for _ in range(512)])

# ---------- model ----------
D, H, L, F = 16, 2, 2, 64
DH = D // H

def init():
    r = lambda *s: onp.random.randn(*s) * 0.08
    P = {"W_E": r(V, D), "W_P": r(T, D), "g_f": onp.ones(D), "W_U": r(D, V)}
    for l in range(L):
        P[f"L{l}.g1"] = onp.ones(D)
        P[f"L{l}.W_Q"] = r(D, D)
        P[f"L{l}.W_K"] = r(D, D)
        P[f"L{l}.W_V"] = r(D, D)
        P[f"L{l}.W_O"] = r(D, D)
        P[f"L{l}.g2"] = onp.ones(D)
        P[f"L{l}.W_up"] = r(D, F)
        P[f"L{l}.W_down"] = r(F, D) * 0.5
    return P

def rms(x, g):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * g

def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

mask = onp.triu(onp.ones((T, T)), 1) * -1e9

def forward(P, idx):
    B, t = idx.shape
    x = P["W_E"][idx] + P["W_P"][:t]
    for l in range(L):
        h = rms(x, P[f"L{l}.g1"])
        q = np.reshape(h @ P[f"L{l}.W_Q"], (B, t, H, DH))
        k = np.reshape(h @ P[f"L{l}.W_K"], (B, t, H, DH))
        v = np.reshape(h @ P[f"L{l}.W_V"], (B, t, H, DH))
        s = np.einsum("bihd,bjhd->bhij", q, k) / np.sqrt(DH) + mask[:t, :t]
        s = s - np.max(s, axis=-1, keepdims=True)
        a = np.exp(s); a = a / np.sum(a, axis=-1, keepdims=True)
        o = np.reshape(np.einsum("bhij,bjhd->bihd", a, v), (B, t, D))
        x = x + o @ P[f"L{l}.W_O"]
        h = rms(x, P[f"L{l}.g2"])
        x = x + gelu(h @ P[f"L{l}.W_up"]) @ P[f"L{l}.W_down"]
    return rms(x, P["g_f"]) @ P["W_U"]

def loss(P, batch):
    logits = forward(P, batch[:, :-1])
    y = batch[:, 1:]
    m = np.max(logits, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(logits - m), axis=-1)) + m[..., 0]
    tgt = np.sum(logits * onp.eye(V)[y], axis=-1)
    return np.mean(lse - tgt)

P = init()
g = grad(loss)
m = {k: onp.zeros_like(v) for k, v in P.items()}
s2 = {k: onp.zeros_like(v) for k, v in P.items()}
lr, b1, b2 = 0.01, 0.9, 0.99
steps = 2500
hist = []
for step in range(1, steps + 1):
    batch = data[onp.random.choice(len(data), 64)]
    G = g(P, batch)
    cur = lr * min(1, step / 100) * (0.1 + 0.9 * (1 - step / steps))
    for k in P:
        m[k] = b1 * m[k] + (1 - b1) * G[k]
        s2[k] = b2 * s2[k] + (1 - b2) * G[k] ** 2
        P[k] = P[k] - cur * (m[k] / (1 - b1 ** step)) / (onp.sqrt(s2[k] / (1 - b2 ** step)) + 1e-8)
    if step % 100 == 0 or step == 1:
        lv = float(loss(P, data[:128]))
        hist.append([step, round(lv, 4)])
        print(step, round(lv, 4))

# ---------- export ----------
R = lambda a: onp.round(onp.asarray(a), 4).tolist()
test_ids = onp.array([[stoi[w] for w in "<s> the bird flew over the".split()]])
out = {
    "config": {"V": V, "T": T, "D": D, "H": H, "L": L, "F": F},
    "vocab": vocab,
    "loss_curve": hist,
    "weights": {k: R(v) for k, v in P.items()},
}
Pr = {k: onp.array(v) for k, v in out["weights"].items()}
out["test"] = {"ids": test_ids[0].tolist(), "logits_last": R(forward(Pr, test_ids)[0, -1])}
out_path = Path(__file__).parent / "weights" / "model.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out, separators=(",", ":")))
probs = onp.exp(out["test"]["logits_last"]); probs /= probs.sum()
print("after 'the bird flew over the':", [(vocab[i], round(probs[i], 3)) for i in onp.argsort(-probs)[:4]])
print("params:", sum(onp.size(v) for v in P.values()))
