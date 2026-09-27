"""
Micro Opus: a 7,440-weight decoder-only transformer.

A language model is two things:
  1. a dictionary of named weight matrices, loaded once and never changed here
  2. this forward() function, which pushes activations through them
"""
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).parent / "weights" / "model.json"


def rmsnorm(x, g):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * g


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def gelu(z):
    return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def causal_mask(T):
    return np.triu(np.full((T, T), -np.inf), k=1)


class MicroOpus:
    def __init__(self, path=WEIGHTS):
        blob = json.loads(Path(path).read_text())
        self.cfg = blob["config"]
        self.vocab = blob["vocab"]
        self.W = {name: np.array(t) for name, t in blob["weights"].items()}

    def tokenize(self, text):
        return [0] + [self.vocab.index(w) for w in text.split()]

    def forward(self, ids, tr=None):
        W, cfg = self.W, self.cfg
        T, D, H = len(ids), cfg["D"], cfg["H"]
        dh = D // H
        tr = {} if tr is None else tr

        emb = W["W_E"][ids]
        pos = W["W_P"][:T]
        x = emb + pos
        tr.update(emb=emb, pos=pos, x0=x, layers=[])

        for l in range(cfg["L"]):
            P = lambda name: W[f"L{l}.{name}"]
            xin = x
            h1 = rmsnorm(x, P("g1"))
            q = h1 @ P("W_Q")
            k = h1 @ P("W_K")
            v = h1 @ P("W_V")
            qh, kh, vh = (t.reshape(T, H, dh).transpose(1, 0, 2) for t in (q, k, v))
            scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(dh) + causal_mask(T)
            att = softmax(scores)
            mix = (att @ vh).transpose(1, 0, 2).reshape(T, D)
            attn_out = mix @ P("W_O")
            x = x + attn_out
            xmid = x
            h2 = rmsnorm(x, P("g2"))
            up = h2 @ P("W_up")
            act = gelu(up)
            down = act @ P("W_down")
            x = x + down
            tr["layers"].append(dict(
                xin=xin, h1=h1, q=q, k=k, v=v, scores=scores, att=att, mix=mix,
                attnOut=attn_out, xmid=xmid, h2=h2, up=up, act=act, down=down, xout=x))

        hf = rmsnorm(x, W["g_f"])
        logits = hf @ W["W_U"]
        probs = softmax(logits)
        tr.update(hf=hf, logits=logits, probs=probs)
        return probs


if __name__ == "__main__":
    m = MicroOpus()
    p = m.forward(m.tokenize("the bird flew over the"))[-1]
    for i in np.argsort(-p)[:3]:
        print(f"{m.vocab[i]:>6}  {p[i]:.3f}")
