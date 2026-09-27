"""Runs the browser engine (engine.js, via node) against model.py: forward, backward, and KV-cache step."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import MicroOpus

for name in ("words", "names"):
    m = MicroOpus(name)
    blob = json.loads((ROOT / "weights" / f"{name}.json").read_text())
    x, y = blob["eval"]["x"][0], blob["eval"]["y"][0]
    tr = {}
    m.forward(np.array([x]), tr)
    G = m.backward(tr, np.array([y]))
    js = subprocess.run(["node", "-e", f"""
const C = require('{ROOT}/frontend/engine.js'); const b = require('{ROOT}/weights/{name}.json');
const x = {json.dumps(x)}, y = {json.dumps(y)};
const tr = C.forward(b.weights, b.config, x); const G = {{}}; for (const k in b.weights) G[k] = C.like(b.weights[k]);
C.backward(b.weights, b.config, tr, y, G, 1 / x.length);
const cache = Array.from({{length: b.config.L}}, () => ({{k: [], v: []}})); let p;
x.forEach((t, i) => p = C.step(b.weights, b.config, t, i, cache).probs);
console.log(JSON.stringify({{probs: tr.probs, G, last: p}}));
"""], capture_output=True, text=True, check=True)
    J = json.loads(js.stdout)
    fwd = np.abs(np.array(J["probs"]) - tr["probs"][0]).max()
    bwd = max(np.abs(np.array(J["G"][k]) - G[k]).max() for k in G)
    kv = np.abs(np.array(J["last"]) - tr["probs"][0][-1]).max()
    print(f"{name}: forward {fwd:.1e}  backward {bwd:.1e}  kv-step {kv:.1e}")
    assert max(fwd, bwd, kv) < 1e-9
print("engine.js matches model.py")
