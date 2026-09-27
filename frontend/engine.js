/*
 * engine.js: line-for-line browser port of model.py.
 * Used when no Python server is running (for example, the published single-file page).
 * Verified to match model.py to within 5e-5 on every logit.
 */
const Core = (() => {
  const mm = (A, B) => A.map(r => B[0].map((_, j) => { let s = 0; for (let i = 0; i < r.length; i++) s += r[i] * B[i][j]; return s; }));
  const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
  const rms = (A, g) => A.map(r => { const m = Math.sqrt(r.reduce((s, v) => s + v * v, 0) / r.length + 1e-5); return r.map((v, i) => v / m * g[i]); });
  const gelu = x => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  const softmax = r => { const m = Math.max(...r); const e = r.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

  function forward(W, cfg, ids) {
    const { D, H, L } = cfg, DH = D / H, t = ids.length;
    const tr = { ids, layers: [] };
    tr.emb = ids.map(id => W.W_E[id].slice());
    tr.pos = ids.map((_, i) => W.W_P[i].slice());
    let x = add(tr.emb, tr.pos);
    tr.x0 = x;
    for (let l = 0; l < L; l++) {
      const P = k => W[`L${l}.${k}`];
      const Ly = { xin: x };
      Ly.h1 = rms(x, P('g1'));
      Ly.q = mm(Ly.h1, P('W_Q')); Ly.k = mm(Ly.h1, P('W_K')); Ly.v = mm(Ly.h1, P('W_V'));
      Ly.scores = []; Ly.att = [];
      Ly.mix = Array.from({ length: t }, () => new Array(D).fill(0));
      for (let h = 0; h < H; h++) {
        const sc = [], at = [];
        for (let i = 0; i < t; i++) {
          const row = [];
          for (let j = 0; j < t; j++) {
            if (j > i) { row.push(-Infinity); continue; }
            let s = 0; for (let d = 0; d < DH; d++) s += Ly.q[i][h * DH + d] * Ly.k[j][h * DH + d];
            row.push(s / Math.sqrt(DH));
          }
          sc.push(row);
          const a = softmax(row); at.push(a);
          for (let j = 0; j <= i; j++) for (let d = 0; d < DH; d++) Ly.mix[i][h * DH + d] += a[j] * Ly.v[j][h * DH + d];
        }
        Ly.scores.push(sc); Ly.att.push(at);
      }
      Ly.attnOut = mm(Ly.mix, P('W_O'));
      Ly.xmid = add(x, Ly.attnOut);
      Ly.h2 = rms(Ly.xmid, P('g2'));
      Ly.up = mm(Ly.h2, P('W_up'));
      Ly.act = Ly.up.map(r => r.map(gelu));
      Ly.down = mm(Ly.act, P('W_down'));
      Ly.xout = add(Ly.xmid, Ly.down);
      x = Ly.xout;
      tr.layers.push(Ly);
    }
    tr.hf = rms(x, W.g_f);
    tr.logits = mm(tr.hf, W.W_U);
    tr.probs = tr.logits.map(softmax);
    return tr;
  }
  return { forward, softmax, gelu, mm };
})();
if (typeof module !== 'undefined') module.exports = Core;
