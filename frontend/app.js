(() => {
const MODEL = window.MODEL, SOURCES = window.SOURCES;
const { config: CFG, vocab: VOCAB } = MODEL;
const W0 = MODEL.weights;
const { D, H, L, F, T, V } = CFG;
const DH = D / H;
const SVGNS = 'http://www.w3.org/2000/svg';
const ix = w => VOCAB.indexOf(w);
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString('en-US');
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const store = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };

const S = {
  ids: [0, ...'the bird flew over the'.split(' ').map(ix)],
  focus: -1, section: 'workbench', step: 0, playing: false, timer: null, sel: 'L0.W_Q', codeTab: 'model.py',
  gen: { ids: null }, surg: { heads: new Set(), mlps: new Set(), noiseT: 'L0.W_up', sigma: 0, seed: 1 }
};
let TR = null, ENGINE = 'browser';
const focusPos = () => S.focus < 0 || S.focus >= S.ids.length ? S.ids.length - 1 : S.focus;

/* ================= engine: Python server if present, else browser port ================= */
async function detectServer() {
  try {
    const ctl = new AbortController(); const tm = setTimeout(() => ctl.abort(), 1500);
    const r = await fetch('api/health', { signal: ctl.signal }); clearTimeout(tm);
    const j = r.ok ? await r.json() : null;
    if (j && j.engine === 'model.py') {
      ENGINE = 'python';
      for (const name of ['model.py', 'train.py', 'server.py', 'frontend/engine.js']) {
        const s = await fetch('api/source/' + name); if (s.ok) SOURCES[name.replace('frontend/', '')] = await s.text();
      }
    }
  } catch (e) { ENGINE = 'browser'; }
  const b = $('engine');
  b.classList.toggle('py', ENGINE === 'python');
  b.querySelector('span').textContent = ENGINE === 'python' ? 'Numbers from model.py on the server' : 'Numbers from engine.js in your browser';
  b.title = ENGINE === 'python' ? 'Every forward pass on the Workbench is computed by the Python file in the code window.' : 'No Python server found, so the line-for-line browser port runs instead. Start server.py to use Python.';
}
async function getTrace(ids) {
  if (ENGINE === 'python') {
    try {
      const r = await fetch('api/trace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids }) });
      if (r.ok) { const t = await r.json(); t.layers.forEach(Ly => Ly.scores = Ly.scores.map(h => h.map(row => row.map(v => v === null ? -Infinity : v)))); return t; }
    } catch (e) {}
  }
  return Core.forward(W0, CFG, ids);
}

/* ================= shared drawing helpers ================= */
const maxAbs = M => { let m = 1e-9; (Array.isArray(M[0]) ? M.flat() : M).forEach(v => { if (isFinite(v) && Math.abs(v) > m) m = Math.abs(v); }); return m; };
const shape = t => Array.isArray(t[0]) ? [t.length, t[0].length] : [t.length];
const count = t => shape(t).reduce((a, b) => a * b, 1);
const TIPS = {}; let tipSeq = 0;

function heat(M, o = {}) {
  const M2 = Array.isArray(M[0]) ? M : [M];
  const R = M2.length, C = M2[0].length, c = o.cell || 14;
  const lw = o.rowLabels ? (o.lw || 48) : 0, th = o.colLabels ? (o.th || 44) : 0;
  const sc = o.scale || maxAbs(M2), gap = c > 6 ? 1 : 0;
  const id = 'h' + (tipSeq++);
  TIPS[id] = { M: M2, name: o.name || '', rowLabels: o.rowLabels, colLabels: o.colLabels, fmtTip: o.fmtTip };
  const Wd = lw + C * c, Ht = th + R * c;
  let s = `<svg class="hm" data-tip="${id}" viewBox="0 0 ${Wd} ${Ht}" width="${Wd}" role="img" aria-label="${esc(o.name || 'matrix')} heatmap">`;
  s += `<rect class="bg grid" x="${lw}" y="${th}" width="${C * c}" height="${R * c}"/>`;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const v = M2[i][j]; if (!isFinite(v)) continue;
    const a = Math.min(1, Math.abs(v) / sc); if (a < 0.02) continue;
    s += `<rect class="${v >= 0 ? 'p' : 'n'}" x="${lw + j * c}" y="${th + i * c}" width="${c - gap}" height="${c - gap}" fill-opacity="${a.toFixed(2)}"/>`;
  }
  if (o.rowLabels) o.rowLabels.forEach((t, i) => s += `<text x="${lw - 5}" y="${th + i * c + c * 0.72}" text-anchor="end" style="font-size:${Math.min(11, c * 0.85)}px">${esc(t)}</text>`);
  if (o.colLabels) o.colLabels.forEach((t, j) => s += `<text transform="translate(${lw + j * c + c * 0.7},${th - 4}) rotate(-60)" style="font-size:${Math.min(11, c * 0.85)}px">${esc(t)}</text>`);
  (o.hiRows || []).forEach(i => s += `<rect class="hiRow" x="${lw - 1}" y="${th + i * c - 1}" width="${C * c + 1}" height="${c + 1}"/>`);
  if (o.focusRow != null) s += `<rect class="hiFocus" x="${lw - 2}" y="${th + o.focusRow * c - 2}" width="${C * c + 3}" height="${c + 3}"/>`;
  if (o.frame) s += `<rect class="${o.frame}" x="${lw - 3}" y="${th - 3}" width="${C * c + 5}" height="${R * c + 5}" rx="3"/>`;
  return s + '</svg>';
}
document.addEventListener('pointermove', e => {
  const tip = $('tip'), svg = e.target.closest && e.target.closest('svg[data-tip]');
  if (!svg) { tip.style.display = 'none'; return; }
  const Tt = TIPS[svg.dataset.tip], bg = svg.querySelector('rect.grid'); if (!Tt || !bg) return;
  const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  const R = Tt.M.length, C = Tt.M[0].length, x0 = +bg.getAttribute('x'), y0 = +bg.getAttribute('y');
  const j = Math.floor((p.x - x0) / (+bg.getAttribute('width') / C)), i = Math.floor((p.y - y0) / (+bg.getAttribute('height') / R));
  if (i < 0 || j < 0 || i >= R || j >= C) { tip.style.display = 'none'; return; }
  const v = Tt.M[i][j];
  let txt = Tt.fmtTip ? Tt.fmtTip(i, j, v) : `${Tt.name}[${i}][${j}] = ${isFinite(v) ? v.toFixed(4) : 'masked'}`;
  if (!Tt.fmtTip && Tt.rowLabels) txt += `   row: ${Tt.rowLabels[i]}`;
  if (!Tt.fmtTip && Tt.colLabels) txt += `   col: ${Tt.colLabels[j]}`;
  tip.textContent = txt; tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = (e.clientY + 16) + 'px';
});
function bars(probs, k = 6, dim = false) {
  return probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([p, i]) =>
    `<div class="bar ${dim ? 'dim' : ''}"><span>${esc(VOCAB[i])}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span class="v">${(p * 100).toFixed(1)}%</span></div>`).join('');
}

/* ================= navigation, theme, prompt ================= */
const SECTIONS = [['workbench', 'Workbench'], ['overview', 'Overview'], ['anatomy', 'Anatomy'], ['attention', 'Attention'], ['generation', 'Generation'], ['surgery', 'Surgery'], ['scale', 'Scale']];
$('nav').innerHTML = SECTIONS.map(([k, t]) => `<li><button data-s="${k}">${t}</button></li>`).join('');
$('nav').addEventListener('click', e => { const b = e.target.closest('button'); if (b) go(b.dataset.s); });
function go(s) {
  S.section = s; stopPlay();
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('on', v.id === 'v-' + s));
  document.querySelectorAll('#nav button').forEach(b => b.setAttribute('aria-current', b.dataset.s === s));
  store.set('mo-section', s); render(); window.scrollTo(0, 0); sizeShell();
}
$('themeBtn').onclick = () => {
  const r = document.documentElement;
  const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme = dark ? 'light' : 'dark'; store.set('mo-theme', r.dataset.theme);
};
if (store.get('mo-theme')) document.documentElement.dataset.theme = store.get('mo-theme');
function sizeShell() { document.documentElement.style.setProperty('--hdr', ($('hdr').offsetHeight + (document.querySelector('.transport')?.offsetHeight || 0)) + 'px'); }
addEventListener('resize', sizeShell);

const GROUPS = [['Glue', ['the', '.']], ['Describe', ['big', 'small', 'happy']], ['Who', ['cat', 'dog', 'bird', 'fish', 'kid']], ['Did', ['sat', 'slept', 'ran', 'flew', 'sang', 'swam']], ['Where', ['on', 'to', 'over', 'in']], ['What', ['mat', 'bed', 'chair', 'park', 'house', 'store', 'tree', 'pond', 'lake']]];
$('palette').innerHTML = GROUPS.map(([g, ws]) => `<span class="grp">${g}</span>` + ws.map(w => `<button data-w="${w}">${w}</button>`).join('')).join('');
$('palette').addEventListener('click', e => { const b = e.target.closest('button[data-w]'); if (!b || S.ids.length >= T) return; S.ids.push(ix(b.dataset.w)); S.focus = -1; changed(); });
$('palBtn').onclick = () => { const p = $('palette'); p.classList.toggle('open'); $('palBtn').textContent = p.classList.contains('open') ? 'Hide words' : 'Add words'; sizeShell(); };
$('backBtn').onclick = () => { if (S.ids.length > 2) { S.ids.pop(); S.focus = -1; changed(); } };
$('preset').onchange = e => { if (!e.target.value) return; S.ids = [0, ...e.target.value.split(' ').map(ix)]; S.focus = -1; e.target.value = ''; changed(); };
$('chips').addEventListener('click', e => { const c = e.target.closest('.chip'); if (!c) return; const i = +c.dataset.i; S.focus = i === S.ids.length - 1 ? -1 : i; renderChips(); if (S.section === 'workbench') { renderScope(); renderLocals(); } });
function renderChips() {
  const f = focusPos();
  $('chips').innerHTML = S.ids.map((id, i) => `<button class="chip ${i === f ? 'focus' : ''}" data-i="${i}" title="Follow this token">${esc(VOCAB[id])}<sub>${id}</sub></button>`).join('') +
    (S.ids.length >= T ? '<span class="small muted">Context window full (16 tokens)</span>' : '');
}
async function changed() { stopPlay(); TR = await getTrace(S.ids); S.gen.ids = null; renderChips(); updateGraphLabels(); render(); }

/* ================= steps ================= */
const STEPS = [
  { k: 'tok', t: 'Tokenize', w: [] }, { k: 'emb', t: 'Embedding lookup', w: ['W_E'] }, { k: 'pos', t: 'Add position', w: ['W_P'] }
];
for (let l = 0; l < L; l++) {
  const P = n => `L${l}.${n}`;
  STEPS.push({ k: 'n1', l, t: 'Normalize', w: [P('g1')] }, { k: 'q', l, t: 'Make queries', w: [P('W_Q')] }, { k: 'kk', l, t: 'Make keys', w: [P('W_K')] },
    { k: 'v', l, t: 'Make values', w: [P('W_V')] }, { k: 'sc', l, t: 'Attention scores', w: [] }, { k: 'mix', l, t: 'Blend values', w: [] },
    { k: 'o', l, t: 'Project and add back', w: [P('W_O')] }, { k: 'n2', l, t: 'Normalize', w: [P('g2')] },
    { k: 'up', l, t: 'MLP expand + GELU', w: [P('W_up')] }, { k: 'down', l, t: 'MLP shrink and add back', w: [P('W_down')] });
}
STEPS.push({ k: 'nf', t: 'Final normalize', w: ['g_f'] }, { k: 'un', t: 'Score every word', w: ['W_U'] }, { k: 'sm', t: 'Softmax and pick', w: [] });
const stepOf = (k, l) => STEPS.findIndex(s => s.k === k && (l == null || s.l === l));
$('scrub').max = STEPS.length;

/* ================= graph ================= */
const G = { N: [], E: [], F: [], slots: {}, h: 0 };
function buildGraph() {
  const NX = 196, NW = 196, LANE = 434, RX = 8, RW = 128, GW = 460;
  let y = 44;
  const node = (id, step, label, o = {}) => {
    const n = { id, step, label, sub: o.sub || '', x: o.x ?? NX, y: o.y ?? y, w: o.w ?? NW, h: o.h ?? 36, kind: o.kind || 'op', wt: o.wt || [] };
    n.cx = n.x + n.w / 2; n.cy = n.y + n.h / 2; G.N.push(n);
    if (o.y == null) y += n.h + (o.gap ?? 22);
    return n;
  };
  const edge = (a, b, kind = 'main') => G.E.push({ a, b, kind });
  const blocks = [];
  const inp = node('in', stepOf('tok'), 'Input tokens', { sub: ' ' });
  const emb = node('emb', stepOf('emb'), 'Embedding lookup', { wt: ['W_E'] });
  const pos = node('pos', stepOf('pos'), 'Add position', { wt: ['W_P'] });
  edge(inp, emb); edge(emb, pos);
  let prev = pos;
  for (let l = 0; l < L; l++) {
    const P = n => `L${l}.${n}`, b0 = y - 6; y += 16;
    const n1 = node('n1' + l, stepOf('n1', l), 'RMSNorm', { wt: [P('g1')] });
    const ry = y;
    const qn = node('q' + l, stepOf('q', l), 'Q', { x: NX, y: ry, w: 60, wt: [P('W_Q')] });
    const kn = node('k' + l, stepOf('kk', l), 'K', { x: NX + 68, y: ry, w: 60, wt: [P('W_K')] });
    const vn = node('v' + l, stepOf('v', l), 'V', { x: NX + 136, y: ry, w: 60, wt: [P('W_V')] });
    y += 36 + 22;
    const sc = node('sc' + l, stepOf('sc', l), 'Scores + softmax', { w: 128 });
    const mix = node('mix' + l, stepOf('mix', l), 'Blend values');
    const o = node('o' + l, stepOf('o', l), 'Linear out', { wt: [P('W_O')], gap: 16 });
    const a1 = node('a1' + l, stepOf('o', l), '+', { kind: 'add', x: NX + NW / 2 - 14, w: 28, h: 28, gap: 18 });
    const n2 = node('n2' + l, stepOf('n2', l), 'RMSNorm', { wt: [P('g2')] });
    const up = node('up' + l, stepOf('up', l), 'Linear up', { wt: [P('W_up')], gap: 16 });
    const ge = node('ge' + l, stepOf('up', l), 'GELU', { w: 100, x: NX + 48, h: 28, gap: 16 });
    const dn = node('dn' + l, stepOf('down', l), 'Linear down', { wt: [P('W_down')], gap: 16 });
    const a2 = node('a2' + l, stepOf('down', l), '+', { kind: 'add', x: NX + NW / 2 - 14, w: 28, h: 28 });
    edge(prev, n1); edge(n1, qn); edge(n1, kn); edge(n1, vn); edge(qn, sc); edge(kn, sc); edge(sc, mix); edge(vn, mix);
    edge(mix, o); edge(o, a1); edge(prev, a1, 'res'); edge(a1, n2); edge(n2, up); edge(up, ge); edge(ge, dn); edge(dn, a2); edge(a1, a2, 'res');
    blocks.push([l, b0, y + 4]); y += 30; prev = a2;
  }
  const nf = node('nf', stepOf('nf'), 'RMSNorm', { wt: ['g_f'] });
  const un = node('un', stepOf('un'), 'Linear unembed', { wt: ['W_U'] });
  const sm = node('sm', stepOf('sm'), 'Softmax', { gap: 18 });
  const out = node('out', stepOf('sm'), 'Next word', { sub: ' ' });
  edge(prev, nf); edge(nf, un); edge(un, sm); edge(sm, out);
  G.h = y + 10;

  const svg = $('graph');
  svg.setAttribute('viewBox', `0 0 ${GW} ${G.h}`);
  let s = `<rect class="rack" x="${RX - 4}" y="10" width="${RW + 8}" height="${G.h - 20}" rx="8"/><text class="rackT" x="${RX + 2}" y="28">Weight memory</text>`;
  blocks.forEach(([l, a, b]) => s += `<rect class="blk" x="${NX - 22}" y="${a}" width="${LANE - NX + 40}" height="${b - a}" rx="10"/><text class="blkT" x="${NX - 14}" y="${a + 15}">Block ${l + 1}: same wiring, its own weights</text>`);
  s += '<g id="gEdges">';
  const path = (a, b, kind) => {
    const ab = a.kind === 'add' ? a.y + a.h : a.y + a.h, bt = b.y;
    if (kind === 'res') {
      const sx = a.x + a.w, sy = a.cy, ex = b.x + b.w;
      return `M${sx} ${sy} H${LANE - 8} Q${LANE} ${sy} ${LANE} ${sy + 8} V${b.cy - 8} Q${LANE} ${b.cy} ${LANE - 8} ${b.cy} H${ex}`;
    }
    if (Math.abs(a.cx - b.cx) < 1) return `M${a.cx} ${ab} V${bt}`;
    const tx = b.id.startsWith('mix') && a.id.startsWith('v') ? a.cx : b.cx;
    const dy = (bt - ab) / 2;
    return `M${a.cx} ${ab} C${a.cx} ${ab + dy} ${tx} ${bt - dy} ${tx} ${bt}`;
  };
  G.E.forEach((e, i) => { e.i = i; s += `<path class="ed ${e.kind === 'res' ? 'res' : ''}" id="ed${i}" d="${path(e.a, e.b, e.kind)}"/>`; });
  s += '</g><g id="gFlights"></g><g id="gSlots">';
  // weight slots, aligned with the node that uses them
  const qkvOff = { 0: -34, 1: -11, 2: 12 };
  G.N.filter(n => n.wt.length).forEach(n => {
    n.wt.forEach(w => {
      const isRow = /^[qkv]\d$/.test(n.id), hh = isRow ? 21 : 26;
      const sy = isRow ? n.cy + qkvOff[{ q: 0, k: 1, v: 2 }[n.id[0]]] : n.cy - 13;
      const sh = shape(W0[w]).join('×');
      G.slots[w] = { x: RX, y: sy, h: hh, node: n.id };
      s += `<g class="slot" id="sl-${w.replace('.', '_')}" data-w="${w}"><rect x="${RX}" y="${sy}" width="${RW}" height="${hh}" rx="5"/><text x="${RX + 7}" y="${sy + hh / 2 + 3.5}">${w}</text><text x="${RX + RW - 7}" y="${sy + hh / 2 + 3.5}" text-anchor="end">${sh}</text></g>`;
      const fx = RX + RW, fy = sy + hh / 2;
      const d = isRow && n.id[0] !== 'q'
        ? `M${fx} ${fy} C${fx + 40} ${fy} ${n.cx} ${n.y - 42} ${n.cx} ${n.y}`
        : `M${fx} ${fy} C${fx + 30} ${fy} ${n.x - 30} ${n.cy} ${n.x} ${n.cy}`;
      G.F.push({ w, node: n.id, d });
    });
  });
  s += '</g><g id="gNodes">';
  G.N.forEach(n => {
    const par = n.wt.length ? 'param' : '';
    if (n.kind === 'add') s += `<g class="gn" id="gn-${n.id}" data-step="${n.step}" tabindex="0"><circle cx="${n.cx}" cy="${n.cy}" r="14"/><text x="${n.cx}" y="${n.cy + 5}" text-anchor="middle" style="font-size:17px">+</text></g>`;
    else {
      const wl = n.wt.length ? `<text class="wsub" x="${n.x + n.w - 8}" y="${n.cy + 4}" text-anchor="end">${n.w > 90 ? shape(W0[n.wt[0]]).join('×') : ''}</text>` : '';
      s += `<g class="gn ${par}" id="gn-${n.id}" data-step="${n.step}" tabindex="0"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="8"/><text x="${n.w > 90 ? n.x + 10 : n.cx}" y="${n.cy + (n.sub !== '' ? -2 : 4)}" ${n.w > 90 ? '' : 'text-anchor="middle"'}>${esc(n.label)}</text>${n.sub !== '' ? `<text class="sub" id="sub-${n.id}" x="${n.x + 10}" y="${n.cy + 11}"></text>` : ''}${wl}</g>`;
    }
  });
  s += `</g><g id="gParticles"></g><text class="blkT" transform="translate(${LANE + 12},${G.N.find(n => n.id === 'n10').cy + 40}) rotate(90)">residual stream</text>`;
  svg.innerHTML = s;
  G.E.forEach(e => e.el = $('ed' + e.i));
  const fg = $('gFlights');
  G.F.forEach((f, i) => { const p = document.createElementNS(SVGNS, 'path'); p.setAttribute('d', f.d); p.setAttribute('class', 'fl'); p.id = 'fl' + i; fg.appendChild(p); f.el = p; f.slotEl = $('sl-' + f.w.replace('.', '_')); });
  G.N.forEach(n => n.el = $('gn-' + n.id));
  svg.addEventListener('click', e => {
    const g = e.target.closest('.gn'); if (g) { stopPlay(); setStep(+g.dataset.step); return; }
    const sl = e.target.closest('.slot'); if (sl) { S.sel = sl.dataset.w; go('anatomy'); }
  });
  svg.addEventListener('keydown', e => { const g = e.target.closest('.gn'); if (g && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); stopPlay(); setStep(+g.dataset.step); } });
}
function updateGraphLabels() {
  const si = $('sub-in'), so = $('sub-out');
  if (si) si.textContent = `ids ${S.ids.join(', ')}`;
  if (so && TR) { const p = TR.probs.at(-1), b = p.indexOf(Math.max(...p)); so.textContent = `"${VOCAB[b]}" at ${(p[b] * 100).toFixed(0)}%`; }
}
let animId = 0;
function fly(pathEl, cls, dur, delay = 0) {
  const my = animId;
  return new Promise(res => {
    if (reduce || !pathEl) { res(); return; }
    const len = pathEl.getTotalLength(), c = document.createElementNS(SVGNS, 'circle');
    c.setAttribute('r', cls === 'wp' ? 5 : 6); c.setAttribute('class', cls); c.style.opacity = 0;
    $('gParticles').appendChild(c);
    const t0 = performance.now() + delay;
    const f = now => {
      if (my !== animId) { c.remove(); res(); return; }
      const k = Math.max(0, Math.min(1, (now - t0) / dur)), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      const p = pathEl.getPointAtLength(e * len);
      c.setAttribute('cx', p.x); c.setAttribute('cy', p.y); c.style.opacity = now < t0 ? 0 : 1;
      if (k < 1) requestAnimationFrame(f); else { c.remove(); res(); }
    };
    requestAnimationFrame(f);
  });
}
const stepDur = () => +$('speed').value;
const particleDur = () => Math.min(700, stepDur() * 0.3);
async function animateGraph(i) {
  const my = ++animId;
  $('gParticles').innerHTML = '';
  G.N.forEach(n => { n.el.classList.toggle('cur', n.step === i); n.el.classList.toggle('done', n.step < i); n.el.classList.remove('hot'); });
  G.E.forEach(e => e.el.classList.toggle('done', e.b.step < i || (e.b.step === i && reduce)));
  G.F.forEach(f => { f.el.classList.remove('live'); f.slotEl.classList.remove('live'); f.slotEl.classList.toggle('used', G.N.find(n => n.id === f.node).step < i); });
  const nodes = G.N.filter(n => n.step === i);
  if (nodes[0]) scrollGraphTo(nodes[0]);
  const dur = particleDur();
  for (const n of nodes) {
    if (my !== animId) return;
    const ins = G.E.filter(e => e.b === n), ws = G.F.filter(f => f.node === n.id);
    ws.forEach(f => { f.el.classList.add('live'); f.slotEl.classList.add('live'); });
    await Promise.all([...ins.map(e => fly(e.el, 'ap', dur)), ...ws.map(f => fly(f.el, 'wp', dur, dur * 0.12))]);
    if (my !== animId) return;
    ins.forEach(e => e.el.classList.add('done'));
    n.el.classList.add('hot');
  }
}
function scrollGraphTo(n) {
  const box = $('gscroll'), svg = $('graph');
  const scale = svg.getBoundingClientRect().width / 460;
  const target = svg.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop + n.cy * scale - box.clientHeight / 2;
  box.scrollTo({ top: Math.max(0, target), behavior: reduce ? 'auto' : 'smooth' });
}

/* ================= code window ================= */
const CODE_FILES = [
  ['model.py', 'The forward pass. This is the code the animation follows, line by line.'],
  ['train.py', 'How the weights were made: 2,500 steps of gradient descent on a toy corpus. The only code that ever changes a weight.'],
  ['server.py', 'The web app backend. Runs model.py for every forward pass the Workbench shows.'],
  ['engine.js', 'Line-for-line browser port of model.py, used when no Python server is running.']
];
const RMS = ['def rmsnorm', 'return x / np.sqrt'], SMX = ['def softmax', 'z = z - np.max', 'e = np.exp(z)', 'return e / np.sum'];
const LAYER = ['for l in range(cfg["L"])', 'P = lambda name'];
const CODE = {
  tok: { p: ['return [0] + [self.vocab.index(w)'], s: ['def tokenize'] },
  emb: { p: ['emb = W["W_E"][ids]'] },
  pos: { p: ['pos = W["W_P"][:T]', 'x = emb + pos'] },
  n1: { p: ['h1 = rmsnorm(x, P("g1"))'], s: RMS },
  q: { p: ['q = h1 @ P("W_Q")'] }, kk: { p: ['k = h1 @ P("W_K")'] }, v: { p: ['v = h1 @ P("W_V")'] },
  sc: { p: ['qh, kh, vh =', 'scores = qh @', 'att = softmax(scores)'], s: ['def causal_mask', 'return np.triu', ...SMX] },
  mix: { p: ['mix = (att @ vh)'] },
  o: { p: ['attn_out = mix @ P("W_O")', 'x = x + attn_out'] },
  n2: { p: ['h2 = rmsnorm(x, P("g2"))'], s: RMS },
  up: { p: ['up = h2 @ P("W_up")', 'act = gelu(up)'], s: ['def gelu', 'return 0.5 * z'] },
  down: { p: ['down = act @ P("W_down")', 'x = x + down'] },
  nf: { p: ['hf = rmsnorm(x, W["g_f"])'], s: RMS },
  un: { p: ['logits = hf @ W["W_U"]'] },
  sm: { p: ['probs = softmax(logits)', 'return probs'], s: SMX }
};
function annFor(s) {
  const t = S.ids.length, sh = (a, b) => `(${a}, ${b})`;
  const p = TR.probs.at(-1), b = p.indexOf(Math.max(...p));
  return {
    tok: [`ids = [${S.ids.join(', ')}]`], emb: [`emb ${sh(t, D)}: rows [${S.ids.join(', ')}] of W_E`],
    pos: [`pos ${sh(t, D)}: rows 0..${t - 1} of W_P`, `x ${sh(t, D)}`], n1: [`h1 ${sh(t, D)}`],
    q: [`${sh(t, D)} @ ${sh(D, D)} → q ${sh(t, D)}`], kk: [`${sh(t, D)} @ ${sh(D, D)} → k ${sh(t, D)}`], v: [`${sh(t, D)} @ ${sh(D, D)} → v ${sh(t, D)}`],
    sc: [`each (${H}, ${t}, ${DH})`, `scores (${H}, ${t}, ${t})`, 'att: every row sums to 1'], mix: [`mix ${sh(t, D)}`],
    o: [`${sh(t, D)} @ ${sh(D, D)}`, 'residual add'], n2: [`h2 ${sh(t, D)}`],
    up: [`${sh(t, D)} @ ${sh(D, F)} → ${sh(t, F)}`, `act ${sh(t, F)}`], down: [`${sh(t, F)} @ ${sh(F, D)} → ${sh(t, D)}`, 'residual add'],
    nf: [`hf ${sh(t, D)}`], un: [`${sh(t, D)} @ ${sh(D, V)} → logits ${sh(t, V)}`], sm: [`next word: "${VOCAB[b]}" ${(p[b] * 100).toFixed(0)}%`, '']
  }[s.k] || [];
}
function hlPy(line, st) {
  let h = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  if (st.doc || /^\s*"""/.test(line)) {
    const toggles = (line.match(/"""/g) || []).length;
    const out = `<span class="c">${h}</span>`;
    if (toggles % 2 === 1) st.doc = !st.doc;
    return out;
  }
  const re = /(#.*$|\/\/.*$|\/\*.*|^\s*\*.*)|(W\["[^"]*"\]|P\("[^"]*"\)|W\.W_[A-Za-z]+|W\[`[^`]*`\])|("[^"]*"|'[^']*'|`[^`]*`)|\b(def|class|return|for|in|import|from|lambda|if|else|not|is|None|and|or|as|with|raise|const|let|function|new|typeof|continue)\b|\b(\d+\.?\d*(?:e-?\d+)?)\b/g;
  return h.replace(re, (m, c, w, s, k, n) => c ? `<span class="c">${c}</span>` : w ? `<span class="wref">${w}</span>` : s ? `<span class="s">${s}</span>` : k ? `<span class="k">${k}</span>` : `<span class="num">${n}</span>`);
}
let codeLines = [];
function renderCodeTabs() {
  $('codeTabs').innerHTML = CODE_FILES.map(([n]) => `<button role="tab" aria-selected="${n === S.codeTab}" data-f="${n}">${n}</button>`).join('');
  $('codeNote').textContent = CODE_FILES.find(f => f[0] === S.codeTab)[1];
  const src = SOURCES[S.codeTab] || '', st = { doc: false };
  codeLines = src.split('\n');
  $('codeBody').innerHTML = codeLines.map((ln, i) => `<div class="ln" data-n="${i}"><span class="no">${i + 1}</span><span class="src">${hlPy(ln, st) || ' '}<span class="ann"></span></span></div>`).join('');
  highlightCode();
}
$('codeTabs').addEventListener('click', e => { const b = e.target.closest('button[data-f]'); if (b) { S.codeTab = b.dataset.f; renderCodeTabs(); } });
function findLine(sub) { return codeLines.findIndex(l => l.includes(sub)); }
function highlightCode() {
  const body = $('codeBody');
  body.querySelectorAll('.ln').forEach(el => { el.classList.remove('hl', 'hl2', 'lyr'); el.querySelector('.ann').textContent = ''; });
  if (S.codeTab !== 'model.py' || !TR) return;
  const s = STEPS[S.step], map = CODE[s.k], anns = annFor(s);
  let first = null;
  map.p.forEach((sub, j) => {
    const i = findLine(sub); if (i < 0) return;
    const el = body.children[i]; el.classList.add('hl'); if (anns[j]) el.querySelector('.ann').textContent = '# ' + anns[j];
    if (first === null) first = el;
  });
  (map.s || []).forEach(sub => { const i = findLine(sub); if (i >= 0) body.children[i].classList.add('hl2'); });
  if (s.l != null) LAYER.forEach((sub, j) => { const i = findLine(sub); if (i >= 0) { body.children[i].classList.add('lyr'); if (j === 0) body.children[i].querySelector('.ann').textContent = `# l = ${s.l}  (block ${s.l + 1})`; } });
  if (first) body.scrollTo({ top: first.offsetTop - body.clientHeight * 0.4, behavior: reduce ? 'auto' : 'smooth' });
}
function renderLocals() {
  if (!TR) return;
  const s = STEPS[S.step], f = focusPos(), l = s.l, Ly = l != null ? TR.layers[l] : null;
  const A = (nm, M, note) => ['ac', nm, shape(M).join(', '), Array.isArray(M[0]) ? M[f] : M, note];
  const Wt = w => ['wt', w, shape(W0[w]).join(', '), W0[w], 'weight'];
  const rows = {
    tok: [['ac', 'ids', `${S.ids.length}`, null, `[${S.ids.join(', ')}]`]],
    emb: [Wt('W_E'), A('emb', TR.emb)], pos: [A('emb', TR.emb), Wt('W_P'), A('x', TR.x0)],
    nf: [A('x', TR.layers.at(-1).xout), Wt('g_f'), A('hf', TR.hf)], un: [A('hf', TR.hf), Wt('W_U'), A('logits', TR.logits)], sm: [A('logits', TR.logits), A('probs', TR.probs)]
  }[s.k] || ({
    n1: () => [A('x', Ly.xin), Wt(`L${l}.g1`), A('h1', Ly.h1)], q: () => [A('h1', Ly.h1), Wt(`L${l}.W_Q`), A('q', Ly.q)],
    kk: () => [A('h1', Ly.h1), Wt(`L${l}.W_K`), A('k', Ly.k)], v: () => [A('h1', Ly.h1), Wt(`L${l}.W_V`), A('v', Ly.v)],
    sc: () => [A('q', Ly.q), A('k', Ly.k), ['ac', 'att[head 1]', `${H}, ${S.ids.length}, ${S.ids.length}`, Ly.att[0][f], 'row for this token'], ['ac', 'att[head 2]', '', Ly.att[1][f], '']],
    mix: () => [A('v', Ly.v), A('mix', Ly.mix)], o: () => [A('mix', Ly.mix), Wt(`L${l}.W_O`), A('attn_out', Ly.attnOut), A('x', Ly.xmid)],
    n2: () => [A('x', Ly.xmid), Wt(`L${l}.g2`), A('h2', Ly.h2)], up: () => [A('h2', Ly.h2), Wt(`L${l}.W_up`), A('up', Ly.up), A('act', Ly.act)],
    down: () => [A('act', Ly.act), Wt(`L${l}.W_down`), A('down', Ly.down), A('x', Ly.xout)]
  }[s.k] || (() => []))();
  $('locals').innerHTML = `<h4>Variables at this line (activations shown for "${esc(VOCAB[S.ids[f]])}", weights shown whole)</h4>` + rows.map(([kind, nm, sh, M, note]) =>
    `<div class="lv ${kind}"><span class="nm" title="${esc(nm)}">${esc(nm)}</span><span class="sh">(${sh})</span><span>${M ? heat(M.map ? M : [M], { name: nm, cell: kind === 'wt' ? (shape(M)[1] > 30 ? 2 : 3) : (M.length > 30 ? 3 : 7), scale: kind === 'ac' && nm.startsWith('att') ? 1 : undefined, frame: null }) : `<span class="muted">${esc(note)}</span>`}</span></div>`).join('');
}

/* ================= stepping ================= */
function setStep(i, animate = true) {
  S.step = Math.max(0, Math.min(STEPS.length - 1, i));
  const s = STEPS[S.step];
  $('stepNo').textContent = `${S.step + 1} / ${STEPS.length}`;
  $('stepName').textContent = s.t + (s.l != null ? ` (block ${s.l + 1})` : '');
  $('scrub').value = S.step + 1;
  if (animate) animateGraph(S.step); else { animId++; G.N.forEach(n => { n.el.classList.toggle('cur', n.step === S.step); n.el.classList.toggle('done', n.step < S.step); }); }
  renderScope(); highlightCode(); renderLocals();
}
$('prevBtn').onclick = () => { stopPlay(); setStep(S.step - 1); };
$('nextBtn').onclick = () => { stopPlay(); setStep(S.step + 1); };
$('playBtn').onclick = () => S.playing ? stopPlay() : startPlay();
$('replayBtn').onclick = () => { stopPlay(); S.step = -1; startPlay(); };
$('scrub').oninput = e => { stopPlay(); setStep(+e.target.value - 1); };
function startPlay() {
  if (S.step >= STEPS.length - 1) S.step = -1;
  S.playing = true; $('playBtn').textContent = 'Pause';
  const tick = () => { if (!S.playing) return; if (S.step >= STEPS.length - 1) { stopPlay(); return; } setStep(S.step + 1); S.timer = setTimeout(tick, stepDur()); };
  tick();
}
function stopPlay() { S.playing = false; clearTimeout(S.timer); $('playBtn').textContent = 'Play'; }
document.addEventListener('keydown', e => {
  if (S.section !== 'workbench' || e.target.closest('input,select,textarea')) return;
  if (e.key === 'ArrowRight') { stopPlay(); setStep(S.step + 1); }
  if (e.key === 'ArrowLeft') { stopPlay(); setStep(S.step - 1); }
  if (e.key === ' ' && !e.target.closest('button')) { e.preventDefault(); S.playing ? stopPlay() : startPlay(); }
});

/* ================= microscope ================= */
let sweepRAF = null;
function matmulSVG(x, Wm, out, o) {
  const n = Wm.length, m = Wm[0].length;
  const c = Math.max(8, Math.min(24, Math.floor(640 / (m + 3)), Math.floor(640 / (n + 3))));
  const gx = c * 2 + 30, gy = o.colLabels ? 52 : 26, lh = 22;
  const wsc = maxAbs(Wm), xsc = maxAbs(x), osc = maxAbs(out);
  const Wd = gx + m * c + 10, Ht = gy + n * c + 18 + c + lh;
  const id = 'h' + (tipSeq++); TIPS[id] = { M: Wm, name: o.wname };
  let s = `<svg class="hm mm" data-tip="${id}" viewBox="0 0 ${Wd} ${Ht}" width="${Wd}" role="img" aria-label="${o.xname} times ${o.wname} gives ${o.oname}">`;
  s += `<text class="al" x="0" y="${gy - 8}">${o.xname}</text><text class="wl" x="${gx}" y="${o.colLabels ? 14 : gy - 8}">${o.wname}  (${n}×${m}, fixed)</text>`;
  for (let i = 0; i < n; i++) { const v = x[i], a = Math.min(1, Math.abs(v) / xsc); s += `<rect class="bg" x="0" y="${gy + i * c}" width="${c * 2 - 1}" height="${c - 1}"/><rect class="${v >= 0 ? 'p' : 'n'}" x="0" y="${gy + i * c}" width="${c * 2 - 1}" height="${c - 1}" fill-opacity="${a.toFixed(2)}"/>`; }
  s += `<rect class="aframe" x="-2" y="${gy - 2}" width="${c * 2 + 3}" height="${n * c + 3}" rx="2"/><text x="${c * 2 + 6}" y="${gy + n * c / 2}" style="font-size:16px">×</text>`;
  s += `<g class="wgrid"><rect class="bg grid" x="${gx}" y="${gy}" width="${m * c}" height="${n * c}"/>`;
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const v = Wm[i][j], a = Math.min(1, Math.abs(v) / wsc); if (a > .02) s += `<rect class="${v >= 0 ? 'p' : 'n'}" x="${gx + j * c}" y="${gy + i * c}" width="${c - 1}" height="${c - 1}" fill-opacity="${a.toFixed(2)}"/>`; }
  s += `</g><rect class="wframe" x="${gx - 3}" y="${gy - 3}" width="${m * c + 5}" height="${n * c + 5}" rx="3"/>`;
  if (o.colLabels) o.colLabels.forEach((t, j) => s += `<text transform="translate(${gx + j * c + c * .7},${gy - 4}) rotate(-60)" style="font-size:${Math.min(11, c * .8)}px">${esc(t)}</text>`);
  if (o.headSplit) { const hx = gx + DH * c, hy = gy + DH * c; s += o.headSplit === 'col' ? `<line x1="${hx}" x2="${hx}" y1="${gy - 6}" y2="${gy + n * c + 6}" stroke="var(--ink)" stroke-dasharray="3 3"/>` : `<line x1="${gx - 6}" x2="${gx + m * c + 6}" y1="${hy}" y2="${hy}" stroke="var(--ink)" stroke-dasharray="3 3"/>`; }
  const oy = gy + n * c + 16;
  s += `<text class="al" x="0" y="${oy + c * .8}">=</text>`;
  for (let j = 0; j < m; j++) { const v = out[j], a = Math.min(1, Math.abs(v) / osc); s += `<g class="oc"><rect class="bg" x="${gx + j * c}" y="${oy}" width="${c - 1}" height="${c - 1}"/><rect class="${v >= 0 ? 'p' : 'n'}" x="${gx + j * c}" y="${oy}" width="${c - 1}" height="${c - 1}" fill-opacity="${a.toFixed(2)}"/></g>`; }
  s += `<rect class="aframe" x="${gx - 2}" y="${oy - 2}" width="${m * c + 3}" height="${c + 3}" rx="2"/><text class="al" x="${gx}" y="${oy + c + 16}">${o.oname}</text>`;
  s += `<rect class="sweep" x="${gx}" y="${gy - 1}" width="${c}" height="${n * c + 2}" style="display:none"/><rect class="cellsel" x="0" y="${gy - 1}" width="${c}" height="${n * c + 2}" style="display:none"/>`;
  return { s: s + '</svg>', c, gx, m };
}
function mountMatmul(el, x, Wm, out, o) {
  cancelAnimationFrame(sweepRAF);
  const Gm = matmulSVG(x, Wm, out, o);
  el.innerHTML = `<div class="canvasWrap">${Gm.s}</div><div class="readout" id="ro">Select any column of ${o.wname} to see its dot product.</div>`;
  const svg = el.querySelector('svg.mm'), cells = [...svg.querySelectorAll('g.oc')];
  const sweep = svg.querySelector('.sweep'), colsel = svg.querySelector('.cellsel'), wg = svg.querySelector('.wgrid');
  svg.addEventListener('click', e => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const j = Math.floor((pt.matrixTransform(svg.getScreenCTM().inverse()).x - Gm.gx) / Gm.c);
    if (j < 0 || j >= Gm.m) return;
    const terms = x.map((xi, i) => [xi * Wm[i][j], i]).sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
    colsel.style.display = ''; colsel.setAttribute('x', Gm.gx + j * Gm.c);
    $('ro').textContent = `${o.oname}[${j}]${o.colLabels ? ` ("${o.colLabels[j]}")` : ''} = Σᵢ ${o.xname}[i] × ${o.wname}[i][${j}]   (${x.length} multiplies, then add)\n= ${out[j].toFixed(3)}.  Biggest terms: ` + terms.slice(0, 3).map(([, i]) => `${x[i].toFixed(2)}×${Wm[i][j].toFixed(2)} (row ${i})`).join(', ');
  });
  if (reduce) return;
  // the weight matrix is dim until its particles arrive in the graph, then the multiply sweeps across it
  const arrive = particleDur() * 1.1;
  wg.style.opacity = .25; wg.style.transition = 'opacity .3s';
  cells.forEach(c => c.style.opacity = 0.08);
  const t0 = performance.now() + arrive, dur = Math.min(1500, 450 + Gm.m * 16);
  setTimeout(() => { wg.style.opacity = 1; sweep.style.display = ''; }, arrive);
  const frame = now => {
    if (now < t0) { sweepRAF = requestAnimationFrame(frame); return; }
    const f = Math.min(1, (now - t0) / dur), j = Math.min(Gm.m - 1, Math.floor(f * Gm.m));
    sweep.setAttribute('x', Gm.gx + j * Gm.c);
    for (let k = 0; k <= j; k++) cells[k].style.opacity = 1;
    if (f < 1) sweepRAF = requestAnimationFrame(frame); else sweep.style.display = 'none';
  };
  sweepRAF = requestAnimationFrame(frame);
}
function vecRow(v, name, cls, o = {}) {
  return `<div style="margin:8px 0"><div class="small"><span class="${cls}">${name}</span> <span class="muted">${o.note || ''}</span></div>${heat([v], { name, cell: o.cell || 20, scale: o.scale, frame: cls === 'w' ? 'wframe' : 'aframe' })}</div>`;
}
function streamFor(k, l) {
  if (k === 'tok') return null; if (k === 'emb') return TR.emb; if (k === 'pos') return TR.x0;
  if (l != null) { const Ly = TR.layers[l]; if (['n1', 'q', 'kk', 'v', 'sc', 'mix'].includes(k)) return Ly.xin; if (['o', 'n2', 'up'].includes(k)) return Ly.xmid; return Ly.xout; }
  return TR.layers.at(-1).xout;
}
function streamHTML() {
  const s = STEPS[S.step], X = streamFor(s.k, s.l); if (!X) return '';
  return `<div class="stream"><h3>The residual stream right now</h3>${heat(X, { name: 'x', cell: 14, rowLabels: S.ids.map((id, i) => `${i}:${VOCAB[id]}`), lw: 64, focusRow: focusPos(), frame: 'aframe' })}
    <p class="cap">One 16-number vector per token. Every block reads from this stream and adds its results back into it.</p></div>`;
}
function renderScope() {
  if (!TR) return;
  const s = STEPS[S.step], f = focusPos(), tokW = VOCAB[S.ids[f]];
  const l = s.l, Ly = l != null ? TR.layers[l] : null, P = n => W0[`L${l}.${n}`];
  const labels = S.ids.map((id, i) => `${i}:${VOCAB[id]}`);
  let head = `<h2>${S.step + 1}. ${s.t}${l != null ? ` (block ${l + 1})` : ''}</h2><p class="small muted">Following "<b>${esc(tokW)}</b>" at position ${f}. All ${S.ids.length} tokens pass through this step together; one is shown up close. Select a word in the input bar to follow another.</p>`;
  const where = (txt, none) => `<div class="where ${none ? 'none' : ''}">${txt}</div>`;
  let body = '', after = '';
  const el = $('scope');
  const mm = (x, Wm, out, o) => { el.innerHTML = head + '<div id="mmHost"></div>' + after + streamHTML(); mountMatmul($('mmHost'), x, Wm, out, o); };
  switch (s.k) {
    case 'tok':
      body = `<p>Text is split into tokens and each token becomes its id in the vocabulary. This model knows 30 whole words. Real models use 100,000 to 200,000 sub-word pieces.</p>
        ${where('No weights yet. The tokenizer is a fixed lookup table built before training, not a learned matrix.', true)}
        <div class="chips" style="margin:12px 0">${S.ids.map((id, i) => `<span class="chip ${i === f ? 'focus' : ''}">${esc(VOCAB[id])}<sub>id ${id}</sub></span>`).join('')}</div>`;
      break;
    case 'emb':
      body = `<p>The first weights arrive. Each token id selects one row of <span class="wtag">W_E</span>, and that 16-number row becomes the token's vector. It's a memory read, equivalent to multiplying a one-hot vector by the matrix.</p>
        ${where(`<span class="w">W_E</span> (30×16) enters. Outlined rows are the ones your input selects; the thick outline is "${esc(tokW)}".`)}
        <div class="canvasWrap">${heat(W0.W_E, { name: 'W_E', cell: 16, rowLabels: VOCAB, hiRows: S.ids, focusRow: S.ids[f], frame: 'wframe' })}</div>
        ${vecRow(TR.emb[f], `vector for "${tokW}"`, 'a', { note: '= that row, copied out' })}`;
      break;
    case 'pos':
      body = `<p>Attention alone can't see word order, so <span class="wtag">W_P</span> gives each position its own vector, added to the word vector.</p>
        ${where(`<span class="w">W_P</span> (16×16) enters: row ${f} is added to "${esc(tokW)}". Llama and Kimi use rotary positions instead, a fixed rotation with no weights.`)}
        <div class="canvasWrap">${heat(W0.W_P, { name: 'W_P', cell: 16, rowLabels: [...Array(T).keys()].map(i => 'pos ' + i), hiRows: S.ids.map((_, i) => i), focusRow: f, frame: 'wframe' })}</div>
        ${vecRow(TR.emb[f], 'word vector', 'a')}${vecRow(TR.pos[f], `+ W_P row ${f}`, 'w')}${vecRow(TR.x0[f], '= residual stream x', 'a')}`;
      break;
    case 'n1': case 'n2': case 'nf': {
      const xin = s.k === 'n1' ? Ly.xin : s.k === 'n2' ? Ly.xmid : TR.layers.at(-1).xout;
      const g = s.k === 'nf' ? W0.g_f : P(s.k === 'n1' ? 'g1' : 'g2'), out = s.k === 'n1' ? Ly.h1 : s.k === 'n2' ? Ly.h2 : TR.hf;
      const r = Math.sqrt(xin[f].reduce((a, b) => a + b * b, 0) / D);
      body = `<p>RMSNorm divides the vector by its overall size so values stay stable as blocks stack, then multiplies each position by a learned gain.</p>
        <div class="eq">h = x / ${r.toFixed(3)} × g</div>
        ${where(`<span class="w">${s.w[0]}</span> (16 numbers) enters as an element-by-element multiply.`)}
        ${vecRow(xin[f], 'x (residual stream)', 'a')}${vecRow(g, s.w[0] + ' (gain, fixed)', 'w')}${vecRow(out[f], 'h (normalized)', 'a')}`;
      break;
    }
    case 'q': case 'kk': case 'v': {
      const nm = { q: ['W_Q', 'q', 'query: what this token is looking for'], kk: ['W_K', 'k', 'key: what this token offers'], v: ['W_V', 'v', 'value: what gets copied if another token attends here'] }[s.k];
      head += `<p>The normalized vector is multiplied by <span class="wtag">${nm[0]}</span> to make a ${nm[2]}. Columns 0 to 7 feed head 1, 8 to 15 feed head 2.</p><div class="eq">${nm[1]} = h1 @ ${nm[0]}   (256 multiply-adds per token)</div>` + where(`<span class="w">L${l}.${nm[0]}</span> streams in from memory, then each output number is one column's dot product.`);
      mm(Ly.h1[f], P(nm[0]), Ly[nm[1]][f], { xname: 'h1', wname: `L${l}.${nm[0]}`, oname: nm[1], headSplit: 'col' });
      return;
    }
    case 'sc': {
      let hs = '';
      for (let h = 0; h < H; h++) {
        const row = Ly.scores[h][f], a = Ly.att[h][f];
        hs += `<div><h3>Head ${h + 1}</h3><table class="mixtab"><tr><td class="muted small">looks at</td><td class="n">q·k/√8</td><td class="n">softmax</td><td></td></tr>` +
          S.ids.slice(0, f + 1).map((id, j) => `<tr><td>${esc(VOCAB[id])}</td><td class="n">${row[j].toFixed(2)}</td><td class="n">${(a[j] * 100).toFixed(0)}%</td><td><div style="width:${Math.max(1, a[j] * 120)}px;height:10px;background:var(--act);border-radius:2px"></div></td></tr>`).join('') + '</table>' +
          `<div style="margin-top:10px">${heat(Ly.att[h], { name: 'attention', cell: 15, rowLabels: labels, lw: 60, hiRows: [f], scale: 1, fmtTip: (i, j, v) => isFinite(v) && j <= i ? `"${VOCAB[S.ids[i]]}" → "${VOCAB[S.ids[j]]}": ${(v * 100).toFixed(1)}%` : 'masked (future)' })}</div></div>`;
      }
      body = `<p>Each head compares this token's query with every earlier token's key, scales by √8, then softmax turns scores into percentages.</p>
        ${where('No weights enter here. The pattern comes purely from activations (q and k), so it changes with every input.', true)}<div class="attrow">${hs}</div>`;
      break;
    }
    case 'mix': {
      let hs = '';
      for (let h = 0; h < H; h++) {
        const a = Ly.att[h][f], vsl = S.ids.slice(0, f + 1).map((_, j) => Ly.v[j].slice(h * DH, h * DH + DH)), sc = maxAbs(vsl);
        hs += `<div><h3>Head ${h + 1}</h3><table class="mixtab">` + S.ids.slice(0, f + 1).map((id, j) => `<tr><td class="n">${(a[j] * 100).toFixed(0)}% ×</td><td>${esc(VOCAB[id])}</td><td>${heat([vsl[j]], { name: 'v', cell: 11, scale: sc })}</td></tr>`).join('') +
          `<tr><td class="n">sum =</td><td></td><td>${heat([Ly.mix[f].slice(h * DH, h * DH + DH)], { name: 'blend', cell: 11, scale: sc, frame: 'aframe' })}</td></tr></table></div>`;
      }
      body = `<p>Each head takes a weighted average of earlier tokens' value vectors, using the percentages from the previous step. This is how information moves between words.</p>
        ${where('No weights here either: activations times activations.', true)}<div class="attrow">${hs}</div>${vecRow(Ly.mix[f], 'both heads side by side', 'a', { cell: 18 })}`;
      break;
    }
    case 'o':
      head += `<p><span class="wtag">W_O</span> merges both heads, and the result is <b>added</b> to the residual stream rather than replacing it.</p><div class="eq">x = x + mix @ W_O</div>` + where(`<span class="w">L${l}.W_O</span> enters. Rows above the dashed line read head 1, rows below read head 2.`);
      after = vecRow(Ly.xin[f], 'x before', 'a') + vecRow(Ly.attnOut[f], '+ attention output', 'a') + vecRow(Ly.xmid[f], '= x after', 'a');
      mm(Ly.mix[f], P('W_O'), Ly.attnOut[f], { xname: 'mix', wname: `L${l}.W_O`, oname: 'attn_out', headSplit: 'row' });
      return;
    case 'up':
      head += `<p>The MLP widens 16 numbers to 64. Each column of <span class="wtag">W_up</span> is a detector; GELU switches off detectors that scored negative. Most weights, and much stored knowledge, live in the MLP.</p><div class="eq">act = gelu(h2 @ W_up)   (1,024 multiply-adds)</div>` + where(`<span class="w">L${l}.W_up</span> (16×64) enters.`);
      after = vecRow(Ly.up[f], 'before GELU', 'a', { cell: 9 }) + vecRow(Ly.act[f], 'after GELU', 'a', { cell: 9 });
      mm(Ly.h2[f], P('W_up'), Ly.up[f], { xname: 'h2', wname: `L${l}.W_up`, oname: 'up' });
      return;
    case 'down':
      head += `<p><span class="wtag">W_down</span> maps the 64 detector outputs back to 16 numbers and adds them to the stream. Row k is what detector k writes when it fires.</p><div class="eq">x = x + act @ W_down</div>` + where(`<span class="w">L${l}.W_down</span> (64×16) enters.`);
      after = vecRow(Ly.xmid[f], 'x before', 'a') + vecRow(Ly.down[f], '+ MLP output', 'a') + vecRow(Ly.xout[f], '= x after', 'a');
      mm(Ly.act[f], P('W_down'), Ly.down[f], { xname: 'act', wname: `L${l}.W_down`, oname: 'down' });
      return;
    case 'un':
      head += `<p>The last weights. The final vector is dotted with each column of <span class="wtag">W_U</span>, one per vocabulary word, giving 30 scores (logits).</p>` + where(`<span class="w">W_U</span> (16×30) enters. In Llama 3 this matrix is 16,384 × 128,256.`);
      mm(TR.hf[f], W0.W_U, TR.logits[f], { xname: 'hf', wname: 'W_U', oname: 'logits', colLabels: VOCAB });
      return;
    case 'sm':
      body = `<p>Softmax turns the 30 scores into probabilities. One word is picked, appended to the input, and the whole pass runs again from step 1 with the same weights.</p>
        ${where('No weights. The pass is over: every weight in the model was used once for this token.', true)}
        <div class="bars" style="margin-top:12px">${bars(TR.probs[f], 8)}</div>
        <p class="cap">${f === S.ids.length - 1 ? 'This is the prediction for the next word.' : `This is what the model predicted after "${esc(tokW)}".`}</p>`;
      break;
  }
  el.innerHTML = head + body + streamHTML();
}

/* ================= overview, anatomy ================= */
function renderOverview() { $('ovBars').innerHTML = bars(TR.probs.at(-1), 6); $('ovCtx').textContent = `after "${S.ids.slice(1).map(i => VOCAB[i]).join(' ')}"`; }
const ORGANS = [['Input: words into vectors', ['W_E', 'W_P']],
  ['Block 1', ['L0.g1', 'L0.W_Q', 'L0.W_K', 'L0.W_V', 'L0.W_O', 'L0.g2', 'L0.W_up', 'L0.W_down']],
  ['Block 2', ['L1.g1', 'L1.W_Q', 'L1.W_K', 'L1.W_V', 'L1.W_O', 'L1.g2', 'L1.W_up', 'L1.W_down']],
  ['Output: vector into word scores', ['g_f', 'W_U']]];
const ROLE = {
  W_E: ['Token embedding', 'One row per vocabulary word. Row i is the starting vector for word i. Looked up, not multiplied.', VOCAB, null],
  W_P: ['Position embedding', 'One row per position 0 to 15, added to the word vector so the model knows word order.', [...Array(T).keys()].map(i => 'pos ' + i), null],
  g1: ['Norm gain (before attention)', 'After the vector is rescaled to a standard size, each of its 16 numbers is multiplied by the matching gain.', null, null],
  W_Q: ['Query projection', 'Turns each token vector into a query. Columns 0 to 7 feed head 1, 8 to 15 feed head 2.', null, null],
  W_K: ['Key projection', 'Turns each token vector into a key. Queries and keys are compared to decide attention.', null, null],
  W_V: ['Value projection', 'Turns each token vector into a value: the information copied when attended to.', null, null],
  W_O: ['Output projection', 'Merges both heads back into the residual stream. Rows 0 to 7 come from head 1, 8 to 15 from head 2.', null, null],
  g2: ['Norm gain (before MLP)', 'Same role as the first gain, in front of the MLP.', null, null],
  W_up: ['MLP up-projection', 'Expands 16 numbers to 64. Each column is a detector for a pattern in the vector.', null, null],
  W_down: ['MLP down-projection', 'Maps 64 detector outputs back to 16 numbers. Row k is what detector k writes.', null, null],
  g_f: ['Final norm gain', 'Rescales the final vector before scoring.', null, null],
  W_U: ['Unembedding', 'One column per vocabulary word. The final vector dotted with column i is word i\'s score.', null, VOCAB]
};
const roleOf = k => ROLE[k.replace(/^L\d\./, '')];
function renderAnatomy() {
  const total = Object.values(W0).reduce((s, t) => s + count(t), 0);
  let h = '<thead><tr><th>Tensor</th><th>Shape</th><th style="text-align:right">Numbers</th><th style="width:70px">Share</th></tr></thead><tbody>';
  ORGANS.forEach(([title, ks]) => {
    h += `<tr class="organ"><td colspan="2">${title}</td><td class="n">${fmt(ks.reduce((s, k) => s + count(W0[k]), 0))}</td><td></td></tr>`;
    ks.forEach(k => { const n = count(W0[k]); h += `<tr class="row ${S.sel === k ? 'sel' : ''}" data-k="${k}" tabindex="0"><td><span class="wtag">${k}</span></td><td class="mono">${shape(W0[k]).join(' × ')}</td><td class="n">${fmt(n)}</td><td><div class="pb" style="width:${Math.max(2, n / total * 260)}px"></div></td></tr>`; });
  });
  $('ftable').innerHTML = h + `<tr class="organ"><td colspan="2">Total</td><td class="n">${fmt(total)}</td><td></td></tr></tbody>`;
  const k = S.sel, t = W0[k], r = roleOf(k), sh = shape(t), M = sh.length === 1 ? [t] : t;
  const cell = sh.length === 1 ? 22 : Math.max(6, Math.min(18, Math.floor(520 / M[0].length)));
  const flat = M.flat(), mean = flat.reduce((a, b) => a + b, 0) / flat.length, sd = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
  $('inspector').innerHTML = `<h3><span class="wtag">${k}</span> ${r[0]}</h3><p class="small">${r[1]}</p>
    <p class="small muted">Shape ${sh.join(' × ')}, ${fmt(flat.length)} numbers, mean ${mean.toFixed(3)}, spread ${sd.toFixed(3)}, largest |value| ${maxAbs(M).toFixed(3)}. Hover any cell for its exact value.</p>
    <div class="scroll">${heat(M, { name: k, cell, rowLabels: r[2] || (sh.length === 1 ? ['gain'] : null), colLabels: r[3], lw: 52, frame: 'wframe' })}</div>`;
}
$('ftable').addEventListener('click', e => { const r = e.target.closest('tr.row'); if (r) { S.sel = r.dataset.k; renderAnatomy(); } });
$('ftable').addEventListener('keydown', e => { const r = e.target.closest('tr.row'); if (r && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); S.sel = r.dataset.k; renderAnatomy(); } });
function renderLoss() {
  const d = MODEL.loss_curve, w = 460, h = 180, p = 34, xm = d.at(-1)[0], ym = Math.ceil(d[0][1]);
  const X = s => p + (s / xm) * (w - p - 10), Y = v => 10 + (1 - v / ym) * (h - p - 10);
  let s = `<svg class="hm" viewBox="0 0 ${w} ${h}" width="${w}" role="img" aria-label="Training loss falling from ${d[0][1]} to ${d.at(-1)[1]}">`;
  for (let v = 0; v <= ym; v++) s += `<line x1="${p}" x2="${w - 10}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${p - 6}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`;
  s += `<polyline fill="none" stroke="var(--weight)" stroke-width="2.5" points="${d.map(([a, b]) => X(a) + ',' + Y(b)).join(' ')}"/>`;
  [0, 1000, 2000].forEach(v => s += `<text x="${X(v)}" y="${h - 12}" text-anchor="middle">${v}</text>`);
  s += `<text x="${w - 10}" y="${h - 12}" text-anchor="end">training step</text><text x="${p + 6}" y="20" class="lab">loss ${d[0][1].toFixed(2)} → ${d.at(-1)[1].toFixed(2)}</text></svg>`;
  $('lossChart').innerHTML = s + '<p class="cap">The loss can\'t reach zero: after "the cat" both "sat" and "slept" are genuinely possible, so the best answer is a 50/50 split.</p>';
}

/* ================= attention ================= */
function renderAttention() {
  let h = ''; const labels = S.ids.map(i => VOCAB[i]);
  for (let l = 0; l < L; l++) {
    h += `<div class="panel"><h3>Block ${l + 1}</h3><div class="attrow">`;
    for (let k = 0; k < H; k++) {
      const A = TR.layers[l].att[k];
      const peaks = A.map((r, i) => { let b = 0; r.forEach((v, j) => { if (j <= i && v > r[b]) b = j; }); return [i, b, r[b]]; }).filter(([i, b]) => i > 0 && b !== i).slice(-3);
      h += `<div><p class="small"><b>Head ${k + 1}</b></p>${heat(A, { name: 'att', cell: 22, rowLabels: labels, colLabels: labels, lw: 54, th: 46, scale: 1, fmtTip: (i, j, v) => j <= i ? `"${labels[i]}" (${i}) → "${labels[j]}" (${j}): ${(v * 100).toFixed(1)}%` : 'masked: cannot look ahead' })}
        <p class="cap">${peaks.map(([i, b, v]) => `"${esc(labels[i])}" looks mostly at "${esc(labels[b])}" (${(v * 100).toFixed(0)}%)`).join('. ')}${peaks.length ? '.' : ''}</p></div>`;
    }
    h += '</div></div>';
  }
  $('attnGrid').innerHTML = h;
}

/* ================= generation ================= */
function sample(p, temp) {
  if (temp <= 0.001) return p.indexOf(Math.max(...p));
  const q = Core.softmax(p.map(v => Math.log(Math.max(v, 1e-12)) / temp)); let r = Math.random(), a = 0;
  for (let i = 0; i < q.length; i++) { a += q[i]; if (r <= a) return i; }
  return q.length - 1;
}
function genInit() { S.gen = { ids: S.ids.slice(), start: S.ids.length, log: [], passes: 0, last: null, dropped: 0 }; }
function genStep() {
  const g = S.gen, p = Core.forward(W0, CFG, g.ids).probs.at(-1), pick = sample(p, +$('temp').value);
  g.passes++; g.last = p;
  g.log.unshift(`<div><span class="mono">pass ${g.passes}</span><span>read ${g.ids.length} tokens, picked <b>${esc(VOCAB[pick])}</b> at ${(p[pick] * 100).toFixed(0)}%</span></div>`);
  g.ids.push(pick);
  if (g.ids.length > T) { g.ids.shift(); g.start--; g.dropped++; }
}
function renderGen() {
  if (!S.gen.ids) genInit();
  const g = S.gen;
  $('genText').innerHTML = (g.dropped ? '<span class="muted">… </span>' : '') + g.ids.map((id, i) => i === 0 && id === 0 ? '' : `<span class="${i >= g.start ? 'new' + (i === g.ids.length - 1 && g.passes ? ' fresh' : '') : ''}">${esc(VOCAB[id])}</span>`).join(' ');
  $('genBars').innerHTML = g.last ? bars(g.last, 6) : '<p class="muted small">Press "Write one word" to run a forward pass.</p>';
  $('genLog').innerHTML = (g.dropped ? '<div class="muted small">The 16-token context window is full, so the oldest token drops off each pass.</div>' : '') + (g.log.join('') || '<p class="muted small">No passes yet.</p>');
  $('genCounter').innerHTML = `<div><b>${g.passes}</b>forward passes</div><div><b>${fmt(g.passes * 7440)}</b>weight uses (about 7,440 per pass)</div><div><b>0</b>weights changed</div><div><b>~${fmt(g.passes * 2 * 7440)}</b>FLOPs spent on weights</div>`;
  const t = +$('temp').value;
  $('genTempNote').textContent = t === 0 ? 'Temperature 0 always takes the top word.' : `Temperature ${t} samples from these odds${t > 1 ? ', flattened, so unlikely words show up more' : ''}.`;
}
$('gen1').onclick = () => { genStep(); renderGen(); };
$('gen8').onclick = () => { let n = 0; const run = () => { genStep(); renderGen(); if (++n < 8) setTimeout(run, reduce ? 0 : 260); }; run(); };
$('genReset').onclick = () => { genInit(); renderGen(); };
$('temp').oninput = e => { $('tempV').textContent = e.target.value; renderGen(); };

/* ================= surgery ================= */
const PROBES = [['the cat', ['sat', 'slept']], ['the fish', ['swam']], ['the dog ran', ['to']], ['the bird flew over the', ['tree', 'house']], ['the fish swam in the', ['pond', 'lake']], ['the kid sat on the', ['mat', 'bed', 'chair']], ['the big dog ran to the', ['park', 'house', 'store']], ['the cat slept on the', ['mat', 'bed']]];
$('headBoxes').innerHTML = [0, 1].flatMap(l => [0, 1].map(h => `<label><input type="checkbox" data-h="${l}-${h}"> Block ${l + 1}, head ${h + 1}</label>`)).join('');
$('mlpBoxes').innerHTML = [0, 1].map(l => `<label><input type="checkbox" data-m="${l}"> Block ${l + 1} MLP</label>`).join('');
$('noiseT').innerHTML = Object.keys(W0).filter(k => shape(W0[k]).length === 2).map(k => `<option ${k === S.surg.noiseT ? 'selected' : ''}>${k}</option>`).join('');
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function patched() {
  const Wp = {}; for (const k in W0) Wp[k] = shape(W0[k]).length === 2 ? W0[k].map(r => r.slice()) : W0[k].slice();
  S.surg.heads.forEach(s => { const [l, h] = s.split('-').map(Number); for (let i = h * DH; i < h * DH + DH; i++) Wp[`L${l}.W_O`][i] = Wp[`L${l}.W_O`][i].map(() => 0); });
  S.surg.mlps.forEach(l => { Wp[`L${l}.W_down`] = Wp[`L${l}.W_down`].map(r => r.map(() => 0)); });
  if (S.surg.sigma > 0) {
    const k = S.surg.noiseT, fl = W0[k].flat(), mu = fl.reduce((a, b) => a + b, 0) / fl.length, sd = Math.sqrt(fl.reduce((a, b) => a + (b - mu) ** 2, 0) / fl.length), r = rng(S.surg.seed * 9973);
    for (const row of Wp[k]) for (let j = 0; j < row.length; j++) row[j] += Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()) * sd * S.surg.sigma;
  }
  return Wp;
}
function renderSurg() {
  const Wp = patched();
  $('sBase').innerHTML = bars(TR.probs.at(-1), 5, true);
  $('sPatch').innerHTML = bars(Core.forward(Wp, CFG, S.ids).probs.at(-1), 5);
  let p0 = 0, p1 = 0;
  const top = p => VOCAB[p.indexOf(Math.max(...p))];
  const rows = PROBES.map(([txt, ok]) => {
    const ids = [0, ...txt.split(' ').map(ix)], a = top(Core.forward(W0, CFG, ids).probs.at(-1)), b = top(Core.forward(Wp, CFG, ids).probs.at(-1));
    const oa = ok.includes(a), ob = ok.includes(b); p0 += oa; p1 += ob;
    return `<tr><td>${txt} …</td><td>${a} <span class="${oa ? 'ok' : 'bad'}">${oa ? 'pass' : 'fail'}</span></td><td>${b} <span class="${ob ? 'ok' : 'bad'}">${ob ? 'pass' : 'fail'}</span></td></tr>`;
  });
  $('probes').innerHTML = `<tr><th>Probe</th><th>Original (${p0}/${PROBES.length})</th><th>After surgery (${p1}/${PROBES.length})</th></tr>` + rows.join('');
}
$('v-surgery').addEventListener('change', e => {
  const t = e.target;
  if (t.dataset.h) t.checked ? S.surg.heads.add(t.dataset.h) : S.surg.heads.delete(t.dataset.h);
  if (t.dataset.m) t.checked ? S.surg.mlps.add(+t.dataset.m) : S.surg.mlps.delete(+t.dataset.m);
  if (t.id === 'noiseT') S.surg.noiseT = t.value;
  renderSurg();
});
$('sigma').oninput = e => { S.surg.sigma = +e.target.value; $('sigmaV').textContent = e.target.value; renderSurg(); };
$('reseed').onclick = () => { S.surg.seed++; renderSurg(); };
$('heal').onclick = () => { S.surg.heads.clear(); S.surg.mlps.clear(); S.surg.sigma = 0; $('sigma').value = 0; $('sigmaV').textContent = '0'; document.querySelectorAll('#v-surgery input[type=checkbox]').forEach(c => c.checked = false); renderSurg(); };
$('paramEq').innerHTML = `12 × ${L} × ${D}² = ${fmt(12 * L * D * D)}  +  embeddings ${V}×${D} + ${D}×${V} + ${T}×${D} = ${fmt(2 * V * D + T * D)}  +  norm gains ${fmt(5 * D)}  =  <b>${fmt(12 * L * D * D + 2 * V * D + T * D + 5 * D)}</b>`;

/* ================= boot ================= */
function render() {
  if (!TR) return;
  for (const k in TIPS) delete TIPS[k];
  ({ workbench: () => { setStep(S.step, false); }, overview: renderOverview, anatomy: renderAnatomy, attention: renderAttention, generation: renderGen, surgery: renderSurg, scale: () => {} })[S.section]();
}
(async () => {
  buildGraph(); renderLoss();
  TR = Core.forward(W0, CFG, S.ids);
  renderChips(); updateGraphLabels(); renderCodeTabs();
  const start = store.get('mo-section');
  go(SECTIONS.some(s => s[0] === start) ? start : 'workbench');
  await detectServer();
  if (ENGINE === 'python') { TR = await getTrace(S.ids); renderCodeTabs(); render(); }
})();
})();
