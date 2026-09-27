/*
 * engine.js: line-for-line browser port of model.py and the Adam step in train.py.
 * Used for training in the browser, for surgery and generation, and for the
 * forward pass whenever no Python server is running. Checked against model.py
 * (forward, backward and step) by tools/check_engine.py.
 */
const Core = (() => {
  const zeros = (r, c) => c == null ? new Array(r).fill(0) : Array.from({ length: r }, () => new Array(c).fill(0));
  const like = t => Array.isArray(t[0]) ? zeros(t.length, t[0].length) : zeros(t.length);
  const mm = (A, B) => A.map(r => { const o = new Array(B[0].length).fill(0); for (let i = 0; i < r.length; i++) { const a = r[i]; if (a === 0) continue; const Bi = B[i]; for (let j = 0; j < o.length; j++) o[j] += a * Bi[j]; } return o; });
  const mmT = (A, B) => A.map(r => B.map(b => { let s = 0; for (let j = 0; j < r.length; j++) s += r[j] * b[j]; return s; }));
  const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
  const accAtB = (G, A, B) => { for (let t = 0; t < A.length; t++) { const a = A[t], b = B[t]; for (let i = 0; i < a.length; i++) { const ai = a[i]; if (ai === 0) continue; const Gi = G[i]; for (let j = 0; j < b.length; j++) Gi[j] += ai * b[j]; } } };
  const rms = (A, g) => A.map(r => { const m = Math.sqrt(r.reduce((s, v) => s + v * v, 0) / r.length + 1e-5); return r.map((v, i) => v / m * g[i]); });
  const gelu = x => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  const geluGrad = z => { const c = 0.7978845608, t = Math.tanh(c * (z + 0.044715 * z * z * z)); return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * c * (1 + 3 * 0.044715 * z * z); };
  const softmax = r => { const m = Math.max(...r); const e = r.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

  function forward(W, cfg, ids) {
    const { D, H, L } = cfg, DH = D / H, t = ids.length;
    const tr = { ids, layers: [] };
    tr.emb = ids.map(id => W.W_E[id].slice());
    tr.pos = ids.map((_, i) => W.W_P[i].slice());
    let x = add(tr.emb, tr.pos);
    tr.x0 = x;
    for (let l = 0; l < L; l++) {
      const P = k => W[`L${l}.${k}`], Ly = { xin: x };
      Ly.h1 = rms(x, P('g1'));
      Ly.q = mm(Ly.h1, P('W_Q')); Ly.k = mm(Ly.h1, P('W_K')); Ly.v = mm(Ly.h1, P('W_V'));
      Ly.scores = []; Ly.att = []; Ly.mix = zeros(t, D);
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

  function rmsBack(dy, X, g, dg) {
    return X.map((x, t) => {
      const r = 1 / Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length + 1e-5);
      const dxhat = dy[t].map((v, i) => v * g[i]);
      const m = dxhat.reduce((s, v, i) => s + v * x[i], 0) / x.length;
      for (let i = 0; i < x.length; i++) dg[i] += dy[t][i] * x[i] * r;
      return dxhat.map((v, i) => r * v - x[i] * r * r * r * m);
    });
  }

  // backward(): adds d(loss)/d(weight) into G. scale = 1 / (tokens in the whole batch)
  function backward(W, cfg, tr, targets, G, scale) {
    const { D, H, L } = cfg, DH = D / H, t = tr.ids.length;
    const dlogits = tr.probs.map((r, i) => r.map((p, j) => (p - (j === targets[i] ? 1 : 0)) * scale));
    accAtB(G.W_U, tr.hf, dlogits);
    let dx = rmsBack(mmT(dlogits, W.W_U), tr.layers[L - 1].xout, W.g_f, G.g_f);
    for (let l = L - 1; l >= 0; l--) {
      const Ly = tr.layers[l], P = k => W[`L${l}.${k}`], GP = k => G[`L${l}.${k}`];
      accAtB(GP('W_down'), Ly.act, dx);
      const dup = mmT(dx, P('W_down')).map((r, i) => r.map((v, j) => v * geluGrad(Ly.up[i][j])));
      accAtB(GP('W_up'), Ly.h2, dup);
      dx = add(dx, rmsBack(mmT(dup, P('W_up')), Ly.xmid, P('g2'), GP('g2')));
      accAtB(GP('W_O'), Ly.mix, dx);
      const dmix = mmT(dx, P('W_O'));
      const dq = zeros(t, D), dk = zeros(t, D), dv = zeros(t, D);
      for (let h = 0; h < H; h++) {
        const o = h * DH;
        for (let i = 0; i < t; i++) {
          const a = Ly.att[h][i], datt = new Array(i + 1).fill(0);
          let s = 0;
          for (let j = 0; j <= i; j++) {
            let d = 0; for (let e = 0; e < DH; e++) d += dmix[i][o + e] * Ly.v[j][o + e];
            datt[j] = d; s += d * a[j];
            for (let e = 0; e < DH; e++) dv[j][o + e] += a[j] * dmix[i][o + e];
          }
          for (let j = 0; j <= i; j++) {
            const ds = a[j] * (datt[j] - s) / Math.sqrt(DH);
            for (let e = 0; e < DH; e++) { dq[i][o + e] += ds * Ly.k[j][o + e]; dk[j][o + e] += ds * Ly.q[i][o + e]; }
          }
        }
      }
      accAtB(GP('W_Q'), Ly.h1, dq); accAtB(GP('W_K'), Ly.h1, dk); accAtB(GP('W_V'), Ly.h1, dv);
      const dh1 = add(add(mmT(dq, P('W_Q')), mmT(dk, P('W_K'))), mmT(dv, P('W_V')));
      dx = add(dx, rmsBack(dh1, Ly.xin, P('g1'), GP('g1')));
    }
    for (let i = 0; i < t; i++) for (let d = 0; d < D; d++) { G.W_P[i][d] += dx[i][d]; G.W_E[tr.ids[i]][d] += dx[i][d]; }
    return G;
  }

  // step(): one new token through the model, appending its key and value to the cache
  function step(W, cfg, token, pos, cache) {
    const { D, H, L } = cfg, DH = D / H;
    let x = W.W_E[token].map((v, i) => v + W.W_P[pos][i]);
    const atts = [];
    for (let l = 0; l < L; l++) {
      const P = k => W[`L${l}.${k}`];
      const h1 = rms([x], P('g1'));
      cache[l].k.push(mm(h1, P('W_K'))[0]); cache[l].v.push(mm(h1, P('W_V'))[0]);
      const q = mm(h1, P('W_Q'))[0], mix = new Array(D).fill(0), la = [];
      for (let h = 0; h < H; h++) {
        const a = softmax(cache[l].k.map(k => { let s = 0; for (let d = 0; d < DH; d++) s += q[h * DH + d] * k[h * DH + d]; return s / Math.sqrt(DH); }));
        a.forEach((w, j) => { for (let d = 0; d < DH; d++) mix[h * DH + d] += w * cache[l].v[j][h * DH + d]; });
        la.push(a);
      }
      atts.push(la);
      x = x.map((v, i) => v + mm([mix], P('W_O'))[0][i]);
      const up = mm(rms([x], P('g2')), P('W_up'))[0].map(gelu);
      const dn = mm([up], P('W_down'))[0];
      x = x.map((v, i) => v + dn[i]);
    }
    return { probs: softmax(mm(rms([x], W.g_f), W.W_U)[0]), att: atts };
  }

  function initWeights(cfg, rand) {
    const { D, F, T, V, L } = cfg;
    const gauss = () => { const u = rand() || 1e-12, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const r = (a, b, s = 0.08) => Array.from({ length: a }, () => Array.from({ length: b }, () => gauss() * s));
    const W = { W_E: r(V, D), W_P: r(T, D), g_f: new Array(D).fill(1), W_U: r(D, V) };
    for (let l = 0; l < L; l++) Object.assign(W, { [`L${l}.g1`]: new Array(D).fill(1), [`L${l}.W_Q`]: r(D, D), [`L${l}.W_K`]: r(D, D), [`L${l}.W_V`]: r(D, D), [`L${l}.W_O`]: r(D, D), [`L${l}.g2`]: new Array(D).fill(1), [`L${l}.W_up`]: r(D, F), [`L${l}.W_down`]: r(F, D, 0.04) });
    return W;
  }

  // one full training step: forward + loss + backward over a batch, then Adam
  function trainStep(W, cfg, xs, ys, opt, lr) {
    const G = {}; for (const k in W) G[k] = like(W[k]);
    const N = xs.length * xs[0].length; let loss = 0;
    xs.forEach((x, b) => { const tr = forward(W, cfg, x); ys[b].forEach((y, i) => loss -= Math.log(tr.probs[i][y] + 1e-12)); backward(W, cfg, tr, ys[b], G, 1 / N); });
    opt.t = (opt.t || 0) + 1; const b1 = 0.9, b2 = 0.99, c1 = 1 - b1 ** opt.t, c2 = 1 - b2 ** opt.t;
    const upd = {};
    for (const k in W) {
      if (!opt.m) { opt.m = {}; opt.v = {}; }
      if (!opt.m[k]) { opt.m[k] = like(W[k]); opt.v[k] = like(W[k]); }
      const two = Array.isArray(W[k][0]);
      const rows = two ? W[k] : [W[k]], gr = two ? G[k] : [G[k]], mr = two ? opt.m[k] : [opt.m[k]], vr = two ? opt.v[k] : [opt.v[k]];
      let s = 0;
      for (let i = 0; i < rows.length; i++) for (let j = 0; j < rows[i].length; j++) {
        const g = gr[i][j];
        mr[i][j] = b1 * mr[i][j] + (1 - b1) * g; vr[i][j] = b2 * vr[i][j] + (1 - b2) * g * g;
        const d = lr * (mr[i][j] / c1) / (Math.sqrt(vr[i][j] / c2) + 1e-8);
        rows[i][j] -= d; s += d * d;
      }
      upd[k] = Math.sqrt(s);
    }
    return { loss: loss / N, G, upd };
  }

  function loss(W, cfg, xs, ys) {
    let s = 0, n = 0;
    xs.forEach((x, b) => { const p = forward(W, cfg, x).probs; ys[b].forEach((y, i) => { s -= Math.log(p[i][y] + 1e-12); n++; }); });
    return s / n;
  }

  return { forward, backward, step, trainStep, initWeights, loss, softmax, gelu, mm, zeros, like };
})();
if (typeof module !== 'undefined') module.exports = Core;
