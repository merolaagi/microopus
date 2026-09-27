(() => {
const MODELS = window.MODELS, SOURCES = window.SOURCES;
const SVGNS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString('en-US');
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const store = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };
const clone = W => { const o = {}; for (const k in W) o[k] = Array.isArray(W[k][0]) ? W[k].map(r => r.slice()) : W[k].slice(); return o; };

/* ================= model registry ================= */
let MODEL, CFG, VOCAB, W0, D, H, L, F, T, V, DH;
const CUSTOM = {};
const PRESETS = {
  words: ['the bird flew over the', 'the cat', 'the fish swam in the', 'the big dog ran to the', 'the happy kid sang', 'the cat sat on the mat . the'],
  names: ['em', 'ja', 'mar', 'chr', 'q', 'ol', 'kay']
};
const DEFAULT_PROMPT = { words: 'the bird flew over the', names: 'mar' };
const isChars = () => !!CFG.chars;
const ix = w => VOCAB.indexOf(w);
const encode = text => [0, ...(isChars() ? text.split('') : text.split(' ')).map(ix).filter(i => i >= 0)];
const tokText = id => VOCAB[id];
const joinToks = ids => ids.map(i => i === 0 ? (isChars() ? ' / ' : ' <s> ') : VOCAB[i] + (isChars() ? '' : ' ')).join('').replace(/\s+/g, ' ').trim();
const paramCount = W => Object.values(W).reduce((s, t) => s + (Array.isArray(t[0]) ? t.length * t[0].length : t.length), 0);

function useModel(name) {
  MODEL = MODELS[name]; CFG = MODEL.config; VOCAB = MODEL.vocab;
  W0 = CUSTOM[name] || MODEL.weights;
  ({ D, H, L, F, T, V } = CFG); DH = D / H;
  S.model = name;
  document.querySelectorAll('.pcount').forEach(e => e.textContent = fmt(paramCount(W0)));
}

const S = {
  model: 'words', ids: [], focus: -1, section: 'workbench', step: 0, playing: false, timer: null, sel: 'L0.W_Q',
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
  updateBadge();
}
function updateBadge() {
  const b = $('engine'), py = ENGINE === 'python' && !CUSTOM[S.model];
  b.classList.toggle('py', py);
  b.querySelector('span').textContent = CUSTOM[S.model] ? 'Your trained weights, in the browser' : py ? 'Numbers from model.py on the server' : 'Numbers from engine.js in your browser';
}
async function getTrace(ids) {
  if (ENGINE === 'python' && !CUSTOM[S.model]) {
    try {
      const r = await fetch('api/trace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids, model: S.model }) });
      if (r.ok) { const t = await r.json(); t.layers.forEach(Ly => Ly.scores = Ly.scores.map(h => h.map(row => row.map(v => v === null ? -Infinity : v)))); return t; }
    } catch (e) {}
  }
  return Core.forward(W0, CFG, ids);
}

/* ================= drawing helpers ================= */
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
  const pc = o.posClass || 'p', nc = o.negClass || 'n';
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const v = M2[i][j]; if (!isFinite(v)) continue;
    const a = Math.min(1, Math.abs(v) / sc); if (a < 0.02) continue;
    s += `<rect class="${v >= 0 ? pc : nc}" x="${lw + j * c}" y="${th + i * c}" width="${c - gap}" height="${c - gap}" fill-opacity="${a.toFixed(2)}"/>`;
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
function lineChart(series, o = {}) {
  const w = o.w || 460, h = o.h || 170, p = 34;
  const all = series.flatMap(s => s.pts);
  if (!all.length) return '<p class="muted small">No data yet.</p>';
  const xm = Math.max(1, ...all.map(q => q[0])), ym = Math.max(0.5, Math.ceil(Math.max(...all.map(q => q[1])) * 2) / 2);
  const X = s => p + (s / xm) * (w - p - 10), Y = v => 10 + (1 - v / ym) * (h - p - 10);
  let s = `<svg class="hm" viewBox="0 0 ${w} ${h}" width="${w}" role="img" aria-label="${esc(o.aria || 'loss chart')}">`;
  for (let v = 0; v <= ym + 1e-9; v += ym > 2 ? 1 : 0.5) s += `<line x1="${p}" x2="${w - 10}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${p - 6}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`;
  series.forEach(se => s += `<polyline fill="none" stroke="${se.color}" stroke-width="2.2" ${se.dash ? 'stroke-dasharray="5 4"' : ''} points="${se.pts.map(([a, b]) => X(a).toFixed(1) + ',' + Y(b).toFixed(1)).join(' ')}"/>`);
  s += `<text x="${w - 10}" y="${h - 12}" text-anchor="end">step ${fmt(xm)}</text>`;
  let lx = p + 6; series.forEach(se => { s += `<line x1="${lx}" x2="${lx + 16}" y1="18" y2="18" stroke="${se.color}" stroke-width="2.5" ${se.dash ? 'stroke-dasharray="4 3"' : ''}/><text x="${lx + 20}" y="21" class="lab" style="font-size:11px">${esc(se.name)}</text>`; lx += 30 + se.name.length * 6.2; });
  return s + '</svg>';
}

/* ================= navigation, theme, model switch, prompt ================= */
const SECTIONS = [['workbench', 'Workbench'], ['training', 'Training'], ['overview', 'Overview'], ['anatomy', 'Anatomy'], ['attention', 'Attention'], ['generation', 'Generation'], ['surgery', 'Surgery'], ['scale', 'Scale']];
$('nav').innerHTML = SECTIONS.map(([k, t]) => `<li><button data-s="${k}">${t}</button></li>`).join('');
$('nav').addEventListener('click', e => { const b = e.target.closest('button'); if (b) go(b.dataset.s); });
function go(s) {
  S.section = s; stopPlay(); if (s !== 'training') TRN.stop();
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
function sizeShell() { document.documentElement.style.setProperty('--hdr', ($('hdr').offsetHeight + (document.querySelector('.view.on .transport')?.offsetHeight || 0)) + 'px'); }
addEventListener('resize', sizeShell);
$('modelSel').onchange = e => switchModel(e.target.value);
async function switchModel(name) {
  stopPlay(); TRN.stop();
  useModel(name); store.set('mo-model', name); $('modelSel').value = name;
  S.ids = encode(DEFAULT_PROMPT[name]); S.focus = -1; S.sel = 'L0.W_Q';
  S.surg.heads.clear(); S.surg.mlps.clear(); S.surg.sigma = 0; $('sigma').value = 0; $('sigmaV').textContent = '0';
  document.querySelectorAll('#v-surgery input[type=checkbox]').forEach(c => c.checked = false);
  buildPalette(); fillNoiseSelect();
  WB.graph.build(); TRN.graph.build(); TRN.reset();
  updateBadge();
  await changed();
}
function buildPalette() {
  const groups = isChars() ? [['Letters', VOCAB.slice(1)]] : [['Glue', ['the', '.']], ['Describe', ['big', 'small', 'happy']], ['Who', ['cat', 'dog', 'bird', 'fish', 'kid']], ['Did', ['sat', 'slept', 'ran', 'flew', 'sang', 'swam']], ['Where', ['on', 'to', 'over', 'in']], ['What', ['mat', 'bed', 'chair', 'park', 'house', 'store', 'tree', 'pond', 'lake']]];
  $('palette').innerHTML = groups.map(([g, ws]) => `<span class="grp">${g}</span>` + ws.map(w => `<button data-w="${w}">${w}</button>`).join('')).join('') + '<span class="grp">End</span><button data-w="<s>">&lt;s&gt;</button>';
  $('preset').innerHTML = '<option value="">Examples</option>' + PRESETS[S.model].map(p => `<option>${p}</option>`).join('');
}
$('palette').addEventListener('click', e => { const b = e.target.closest('button[data-w]'); if (!b || S.ids.length >= T) return; S.ids.push(ix(b.dataset.w)); S.focus = -1; changed(); });
$('palBtn').onclick = () => { const p = $('palette'); p.classList.toggle('open'); $('palBtn').textContent = p.classList.contains('open') ? 'Hide tokens' : 'Add tokens'; sizeShell(); };
$('backBtn').onclick = () => { if (S.ids.length > 2) { S.ids.pop(); S.focus = -1; changed(); } };
$('preset').onchange = e => { if (!e.target.value) return; S.ids = encode(e.target.value); S.focus = -1; e.target.value = ''; changed(); };
$('chips').addEventListener('click', e => { const c = e.target.closest('.chip'); if (!c) return; const i = +c.dataset.i; S.focus = i === S.ids.length - 1 ? -1 : i; renderChips(); if (S.section === 'workbench') { renderScope(); renderLocals(); } });
function renderChips() {
  const f = focusPos();
  $('chips').innerHTML = S.ids.map((id, i) => `<button class="chip ${i === f ? 'focus' : ''}" data-i="${i}" title="Follow this token">${esc(VOCAB[id])}<sub>${id}</sub></button>`).join('') +
    (S.ids.length >= T ? '<span class="small muted">Context window full (16 tokens)</span>' : '');
}
async function changed() { stopPlay(); TR = await getTrace(S.ids); S.gen.ids = null; renderChips(); WB.graph.labels(); render(); }

/* ================= graph factory (used by Workbench and Training) ================= */
function createGraph(svgId, scrollId, pfx) {
  const g = { N: [], E: [], F: [], anim: 0 };
  const el = x => document.getElementById(pfx + x);
  g.build = () => {
    g.N = []; g.E = []; g.F = [];
    const NX = 196, NW = 196, LANE = 434, RX = 8, RW = 128, GW = 460;
    let y = 44;
    const node = (id, step, label, o = {}) => {
      const n = { id, step, label, sub: o.sub || '', x: o.x ?? NX, y: o.y ?? y, w: o.w ?? NW, h: o.h ?? 36, kind: o.kind || 'op', wt: o.wt || [] };
      n.cx = n.x + n.w / 2; n.cy = n.y + n.h / 2; g.N.push(n);
      if (o.y == null) y += n.h + (o.gap ?? 22);
      return n;
    };
    const edge = (a, b, kind = 'main') => g.E.push({ a, b, kind });
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
      y += 58;
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
    const out = node('out', stepOf('sm'), 'Next token', { sub: ' ' });
    edge(prev, nf); edge(nf, un); edge(un, sm); edge(sm, out);
    g.h = y + 10;
    const svg = $(svgId);
    svg.classList.add('gsvg');
    svg.setAttribute('viewBox', `0 0 ${GW} ${g.h}`);
    let s = `<rect class="rack" x="${RX - 4}" y="10" width="${RW + 8}" height="${g.h - 20}" rx="8"/><text class="rackT" x="${RX + 2}" y="28">Weight memory</text>`;
    blocks.forEach(([l, a, b]) => s += `<rect class="blk" x="${NX - 22}" y="${a}" width="${LANE - NX + 40}" height="${b - a}" rx="10"/><text class="blkT" x="${NX - 14}" y="${a + 15}">Block ${l + 1}: same wiring, its own weights</text>`);
    const path = (a, b, kind) => {
      const ab = a.y + a.h, bt = b.y;
      if (kind === 'res') return `M${a.x + a.w} ${a.cy} H${LANE - 8} Q${LANE} ${a.cy} ${LANE} ${a.cy + 8} V${b.cy - 8} Q${LANE} ${b.cy} ${LANE - 8} ${b.cy} H${b.x + b.w}`;
      if (Math.abs(a.cx - b.cx) < 1) return `M${a.cx} ${ab} V${bt}`;
      const tx = b.id.startsWith('mix') && a.id.startsWith('v') ? a.cx : b.cx, dy = (bt - ab) / 2;
      return `M${a.cx} ${ab} C${a.cx} ${ab + dy} ${tx} ${bt - dy} ${tx} ${bt}`;
    };
    s += '<g>';
    g.E.forEach((e, i) => { e.i = i; s += `<path class="ed ${e.kind === 'res' ? 'res' : ''}" id="${pfx}ed${i}" d="${path(e.a, e.b, e.kind)}"/>`; });
    s += `</g><g id="${pfx}fl"></g><g>`;
    const qkvOff = { q: -34, k: -11, v: 12 };
    g.N.filter(n => n.wt.length).forEach(n => n.wt.forEach(w => {
      const isRow = /^[qkv]\d$/.test(n.id), hh = isRow ? 21 : 26;
      const sy = isRow ? n.cy + qkvOff[n.id[0]] : n.cy - 13, sid = pfx + 'sl-' + w.replace('.', '_');
      s += `<g class="slot" id="${sid}" data-w="${w}" tabindex="0"><rect class="sr" x="${RX}" y="${sy}" width="${RW}" height="${hh}" rx="5"/><rect class="gl" x="${RX}" y="${sy}" width="${RW}" height="${hh}" rx="5"/><text x="${RX + 7}" y="${sy + hh / 2 + 3.5}">${w}</text><text x="${RX + RW - 7}" y="${sy + hh / 2 + 3.5}" text-anchor="end">${shape(W0[w]).join('×')}</text></g>`;
      const fx = RX + RW, fy = sy + hh / 2;
      const d = isRow && n.id[0] !== 'q' ? `M${fx} ${fy} C${fx + 40} ${fy} ${n.cx} ${n.y - 42} ${n.cx} ${n.y}` : `M${fx} ${fy} C${fx + 30} ${fy} ${n.x - 30} ${n.cy} ${n.x} ${n.cy}`;
      g.F.push({ w, node: n.id, d });
    }));
    s += '</g><g>';
    g.N.forEach(n => {
      if (n.kind === 'add') s += `<g class="gn" id="${pfx}gn-${n.id}" data-step="${n.step}" tabindex="0"><circle cx="${n.cx}" cy="${n.cy}" r="14"/><text x="${n.cx}" y="${n.cy + 5}" text-anchor="middle" style="font-size:17px">+</text></g>`;
      else {
        const wide = n.w > 90, wl = n.wt.length && wide ? `<text class="wsub" x="${n.x + n.w - 8}" y="${n.cy + 4}" text-anchor="end">${shape(W0[n.wt[0]]).join('×')}</text>` : '';
        s += `<g class="gn ${n.wt.length ? 'param' : ''}" id="${pfx}gn-${n.id}" data-step="${n.step}" ${n.wt.length ? `data-tensor="${n.wt[0]}"` : ''} tabindex="0"><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="8"/><text x="${wide ? n.x + 10 : n.cx}" y="${n.cy + (n.sub ? -2 : 4)}" ${wide ? '' : 'text-anchor="middle"'}>${esc(n.label)}</text>${n.sub ? `<text class="sub" id="${pfx}sub-${n.id}" x="${n.x + 10}" y="${n.cy + 11}"></text>` : ''}${wl}</g>`;
      }
    });
    s += `</g><g id="${pfx}pt"></g><text class="blkT" transform="translate(${LANE + 12},${g.N.find(n => n.id === 'n10').cy + 40}) rotate(90)">residual stream</text>`;
    svg.innerHTML = s;
    g.E.forEach(e => e.el = el('ed' + e.i));
    g.F.forEach((f, i) => { const p = document.createElementNS(SVGNS, 'path'); p.setAttribute('d', f.d); p.setAttribute('class', 'fl'); el('fl').appendChild(p); f.el = p; f.slotEl = el('sl-' + f.w.replace('.', '_')); });
    g.N.forEach(n => n.el = el('gn-' + n.id));
    g.labels();
  };
  g.labels = () => {
    const si = el('sub-in'), so = el('sub-out');
    if (si) si.textContent = `ids ${S.ids.join(', ')}`;
    if (so && TR) { const p = TR.probs.at(-1), b = p.indexOf(Math.max(...p)); so.textContent = `"${VOCAB[b]}" at ${(p[b] * 100).toFixed(0)}%`; }
  };
  g.fly = (pathEl, cls, dur, delay = 0, reverse = false) => {
    const my = g.anim;
    return new Promise(res => {
      if (reduce || !pathEl) { res(); return; }
      const len = pathEl.getTotalLength(), c = document.createElementNS(SVGNS, 'circle');
      c.setAttribute('r', 5.5); c.setAttribute('class', cls); c.style.opacity = 0; el('pt').appendChild(c);
      const t0 = performance.now() + delay;
      const f = now => {
        if (my !== g.anim) { c.remove(); res(); return; }
        const k = Math.max(0, Math.min(1, (now - t0) / dur)), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
        const p = pathEl.getPointAtLength((reverse ? 1 - e : e) * len);
        c.setAttribute('cx', p.x); c.setAttribute('cy', p.y); c.style.opacity = now < t0 ? 0 : 1;
        if (k < 1) requestAnimationFrame(f); else { c.remove(); res(); }
      };
      requestAnimationFrame(f);
    });
  };
  g.clear = () => {
    g.anim++; el('pt').innerHTML = '';
    g.N.forEach(n => n.el.classList.remove('cur', 'hot', 'done', 'bcur'));
    g.E.forEach(e => e.el.classList.remove('done'));
    g.F.forEach(f => { f.el.classList.remove('live', 'glive'); f.slotEl.classList.remove('live', 'glive', 'used', 'upd'); });
  };
  g.forward = async (i, dur) => {
    const my = ++g.anim; el('pt').innerHTML = '';
    g.N.forEach(n => { n.el.classList.toggle('cur', n.step === i); n.el.classList.toggle('done', n.step < i); n.el.classList.remove('hot', 'bcur'); });
    g.E.forEach(e => e.el.classList.toggle('done', e.b.step < i));
    g.F.forEach(f => { f.el.classList.remove('live', 'glive'); f.slotEl.classList.remove('live', 'glive', 'upd'); f.slotEl.classList.toggle('used', g.N.find(n => n.id === f.node).step < i); });
    const nodes = g.N.filter(n => n.step === i);
    if (nodes[0]) g.scrollTo(nodes[0]);
    for (const n of nodes) {
      if (my !== g.anim) return;
      const ins = g.E.filter(e => e.b === n), ws = g.F.filter(f => f.node === n.id);
      ws.forEach(f => { f.el.classList.add('live'); f.slotEl.classList.add('live'); });
      await Promise.all([...ins.map(e => g.fly(e.el, 'ap', dur)), ...ws.map(f => g.fly(f.el, 'wp', dur, dur * 0.12))]);
      if (my !== g.anim) return;
      ins.forEach(e => e.el.classList.add('done')); n.el.classList.add('hot');
    }
  };
  g.backward = async (nodeIds, dur) => {
    const my = ++g.anim; el('pt').innerHTML = '';
    g.N.forEach(n => n.el.classList.remove('cur', 'hot', 'bcur'));
    g.F.forEach(f => { f.el.classList.remove('live', 'glive'); f.slotEl.classList.remove('live', 'upd'); });
    const nodes = nodeIds.map(id => g.N.find(n => n.id === id)).filter(Boolean);
    if (nodes[0]) g.scrollTo(nodes[0]);
    for (const n of nodes) {
      if (my !== g.anim) return;
      n.el.classList.add('bcur');
      const outs = g.E.filter(e => e.a === n);
      await Promise.all(outs.map(e => g.fly(e.el, 'gp', dur, 0, true)));
      const ws = g.F.filter(f => f.node === n.id);
      ws.forEach(f => f.el.classList.add('glive'));
      await Promise.all(ws.map(f => g.fly(f.el, 'gp', dur, 0, true)));
      ws.forEach(f => f.slotEl.classList.add('glive'));
    }
  };
  g.flashUpdate = () => { g.F.forEach(f => { f.slotEl.classList.remove('glive'); f.slotEl.classList.add('upd'); }); setTimeout(() => g.F.forEach(f => f.slotEl.classList.remove('upd')), 900); };
  g.glow = levels => g.F.forEach(f => { const r = f.slotEl.querySelector('.gl'); if (r) r.style.opacity = (0.05 + 0.5 * (levels[f.w] || 0)).toFixed(2); });
  g.scrollTo = n => {
    const box = $(scrollId), svg = $(svgId);
    const scale = svg.getBoundingClientRect().width / 460;
    const target = svg.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop + n.cy * scale - box.clientHeight / 2;
    box.scrollTo({ top: Math.max(0, target), behavior: reduce ? 'auto' : 'smooth' });
  };
  $(svgId).addEventListener('click', e => {
    const n = e.target.closest('.gn'), sl = e.target.closest('.slot');
    if (n && g.onNode) g.onNode(+n.dataset.step);
    if (sl && g.onSlot) g.onSlot(sl.dataset.w);
  });
  return g;
}

/* ================= code view factory ================= */
const CODE_NOTES = {
  'model.py': 'The model: forward, backward and step. The highlight follows the animation line by line.',
  'train.py': 'The training loop: batch, forward, loss, backward, Adam update. The only code that ever changes a weight.',
  'server.py': 'The web app backend. Runs model.py for every forward pass the Workbench shows.',
  'engine.js': 'Line-for-line browser port of model.py, used for training in the browser and when no Python server is running.'
};
function createCodeView(rootId, files) {
  const root = $(rootId), tabs = root.querySelector('.codetabs'), note = root.querySelector('.codenote'), body = root.querySelector('.codebody');
  const cv = { file: files[0], lines: [], last: null };
  function hl(line, st) {
    let h = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    if (st.doc || /^\s*"""/.test(line)) { const n = (line.match(/"""/g) || []).length; if (n % 2 === 1) st.doc = !st.doc; return `<span class="c">${h}</span>`; }
    const re = /(#.*$|\/\/.*$|\/\*.*|^\s*\*.*)|(W\["[^"]*"\]|P\("[^"]*"\)|W\.W_[A-Za-z]+|model\.W\[k\])|(G\["[^"]*"\]|G\[f"[^"]*"\]|grads)|("[^"]*"|'[^']*'|`[^`]*`)|\b(def|class|return|for|in|import|from|lambda|if|else|not|is|None|and|or|as|with|raise|const|let|function|new|continue)\b|\b(\d+\.?\d*(?:e-?\d+)?)\b/g;
    return h.replace(re, (m, c, w, gr, s, k, n) => c ? `<span class="c">${c}</span>` : w ? `<span class="wref">${w}</span>` : gr ? `<span class="g">${gr}</span>` : s ? `<span class="s">${s}</span>` : k ? `<span class="k">${k}</span>` : `<span class="num">${n}</span>`);
  }
  cv.render = () => {
    tabs.innerHTML = files.map(n => `<button role="tab" aria-selected="${n === cv.file}" data-f="${n}">${n}</button>`).join('');
    note.textContent = CODE_NOTES[cv.file];
    const st = { doc: false }; cv.lines = (SOURCES[cv.file] || '').split('\n');
    body.innerHTML = cv.lines.map((ln, i) => `<div class="ln"><span class="no">${i + 1}</span><span class="src">${hl(ln, st) || ' '}<span class="ann"></span></span></div>`).join('');
    if (cv.last) cv.highlight(...cv.last);
  };
  tabs.addEventListener('click', e => { const b = e.target.closest('button[data-f]'); if (b) { cv.file = b.dataset.f; cv.render(); } });
  const find = (sub, after = 0) => { for (let i = after; i < cv.lines.length; i++) if (cv.lines[i].includes(sub)) return i; return cv.lines.findIndex(l => l.includes(sub)); };
  // highlight(file, primary[], secondary[], annotations[], layer, anchor)
  cv.highlight = (file, prim = [], sec = [], anns = [], layerLine = null, anchor = null) => {
    cv.last = [file, prim, sec, anns, layerLine, anchor];
    if (file && file !== cv.file) { cv.file = file; cv.render(); return; }
    body.querySelectorAll('.ln').forEach(el => { el.classList.remove('hl', 'hl2', 'lyr'); el.querySelector('.ann').textContent = ''; });
    const start = anchor ? Math.max(0, find(anchor)) : 0;
    let first = null;
    prim.forEach((sub, j) => { const i = find(sub, start); if (i < 0) return; const el = body.children[i]; el.classList.add('hl'); if (anns[j]) el.querySelector('.ann').textContent = '# ' + anns[j]; if (!first) first = el; });
    sec.forEach(sub => { const i = find(sub); if (i >= 0) body.children[i].classList.add('hl2'); });
    if (layerLine) { const i = find(layerLine[0], start); if (i >= 0) { body.children[i].classList.add('lyr'); body.children[i].querySelector('.ann').textContent = '# ' + layerLine[1]; } }
    if (first) body.scrollTo({ top: first.offsetTop - body.clientHeight * 0.35, behavior: reduce ? 'auto' : 'smooth' });
  };
  return cv;
}

/* ================= workbench steps ================= */
const STEPS = [{ k: 'tok', t: 'Tokenize', w: [] }, { k: 'emb', t: 'Embedding lookup', w: ['W_E'] }, { k: 'pos', t: 'Add position', w: ['W_P'] }];
for (let l = 0; l < 2; l++) {
  const P = n => `L${l}.${n}`;
  STEPS.push({ k: 'n1', l, t: 'Normalize', w: [P('g1')] }, { k: 'q', l, t: 'Make queries', w: [P('W_Q')] }, { k: 'kk', l, t: 'Make keys', w: [P('W_K')] },
    { k: 'v', l, t: 'Make values', w: [P('W_V')] }, { k: 'sc', l, t: 'Attention scores', w: [] }, { k: 'mix', l, t: 'Blend values', w: [] },
    { k: 'o', l, t: 'Project and add back', w: [P('W_O')] }, { k: 'n2', l, t: 'Normalize', w: [P('g2')] },
    { k: 'up', l, t: 'MLP expand + GELU', w: [P('W_up')] }, { k: 'down', l, t: 'MLP shrink and add back', w: [P('W_down')] });
}
STEPS.push({ k: 'nf', t: 'Final normalize', w: ['g_f'] }, { k: 'un', t: 'Score every token', w: ['W_U'] }, { k: 'sm', t: 'Softmax and pick', w: [] });
const stepOf = (k, l) => STEPS.findIndex(s => s.k === k && (l == null || s.l === l));
$('scrub').max = STEPS.length;
const RMS = ['def rmsnorm', 'return x / np.sqrt'], SMX = ['def softmax', 'z = z - np.max', 'e = np.exp(z)', 'return e / np.sum'];
const FWD_CODE = {
  tok: { p: ['return [0] + [self.vocab.index(p)'], s: ['def tokenize'] },
  emb: { p: ['emb = W["W_E"][ids]'] }, pos: { p: ['pos = W["W_P"][:T]', 'x = emb + pos'] },
  n1: { p: ['h1 = rmsnorm(x, P("g1"))'], s: RMS },
  q: { p: ['q = h1 @ P("W_Q")'] }, kk: { p: ['k = h1 @ P("W_K")'] }, v: { p: ['v = h1 @ P("W_V")'] },
  sc: { p: ['qh, kh, vh = split_heads', 'scores = qh @', 'att = softmax(scores)'], s: ['def causal_mask', 'return np.triu', ...SMX] },
  mix: { p: ['mix = merge_heads(att @ vh)'] }, o: { p: ['attn_out = mix @ P("W_O")', 'x = x + attn_out'] },
  n2: { p: ['h2 = rmsnorm(x, P("g2"))'], s: RMS }, up: { p: ['up = h2 @ P("W_up")', 'act = gelu(up)'], s: ['def gelu', 'return 0.5 * z'] },
  down: { p: ['down = act @ P("W_down")', 'x = x + down'] }, nf: { p: ['hf = rmsnorm(x, W["g_f"])'], s: RMS },
  un: { p: ['logits = hf @ W["W_U"]'] }, sm: { p: ['probs = softmax(logits)', 'return probs'], s: SMX }
};
function annFor(s) {
  const t = S.ids.length, sh = (a, b) => `(${a}, ${b})`, p = TR.probs.at(-1), b = p.indexOf(Math.max(...p));
  return {
    tok: [`ids = [${S.ids.join(', ')}]`], emb: [`emb ${sh(t, D)}: rows [${S.ids.join(', ')}] of W_E`],
    pos: [`pos ${sh(t, D)}: rows 0..${t - 1} of W_P`, `x ${sh(t, D)}`], n1: [`h1 ${sh(t, D)}`],
    q: [`${sh(t, D)} @ ${sh(D, D)} → q ${sh(t, D)}`], kk: [`${sh(t, D)} @ ${sh(D, D)} → k ${sh(t, D)}`], v: [`${sh(t, D)} @ ${sh(D, D)} → v ${sh(t, D)}`],
    sc: [`each (${H}, ${t}, ${DH})`, `scores (${H}, ${t}, ${t})`, 'att: every row sums to 1'], mix: [`mix ${sh(t, D)}`],
    o: [`${sh(t, D)} @ ${sh(D, D)}`, 'residual add'], n2: [`h2 ${sh(t, D)}`],
    up: [`${sh(t, D)} @ ${sh(D, F)} → ${sh(t, F)}`, `act ${sh(t, F)}`], down: [`${sh(t, F)} @ ${sh(F, D)} → ${sh(t, D)}`, 'residual add'],
    nf: [`hf ${sh(t, D)}`], un: [`${sh(t, D)} @ ${sh(D, V)} → logits ${sh(t, V)}`], sm: [`next token: "${VOCAB[b]}" ${(p[b] * 100).toFixed(0)}%`, '']
  }[s.k] || [];
}
const WB = { graph: createGraph('graph', 'gscroll', 'w'), code: createCodeView('wbCode', ['model.py', 'train.py', 'server.py', 'engine.js']) };
WB.graph.onNode = i => { stopPlay(); setStep(i); };
WB.graph.onSlot = w => { S.sel = w; go('anatomy'); };
function highlightWbCode() {
  if (!TR) return;
  const s = STEPS[S.step], m = FWD_CODE[s.k];
  WB.code.highlight(null, m.p, m.s || [], annFor(s), s.l != null ? ['for l in range(cfg["L"])', `l = ${s.l}  (block ${s.l + 1})`] : null, s.k === 'tok' ? 'def tokenize' : 'def forward');
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
    sc: () => [A('q', Ly.q), A('k', Ly.k), ['ac', 'att[head 1]', `${H}, ${S.ids.length}, ${S.ids.length}`, Ly.att[0][f], ''], ['ac', 'att[head 2]', '', Ly.att[1][f], '']],
    mix: () => [A('v', Ly.v), A('mix', Ly.mix)], o: () => [A('mix', Ly.mix), Wt(`L${l}.W_O`), A('attn_out', Ly.attnOut), A('x', Ly.xmid)],
    n2: () => [A('x', Ly.xmid), Wt(`L${l}.g2`), A('h2', Ly.h2)], up: () => [A('h2', Ly.h2), Wt(`L${l}.W_up`), A('up', Ly.up), A('act', Ly.act)],
    down: () => [A('act', Ly.act), Wt(`L${l}.W_down`), A('down', Ly.down), A('x', Ly.xout)]
  }[s.k] || (() => []))();
  $('locals').innerHTML = `<h4>Variables at this line (activations for "${esc(VOCAB[S.ids[f]])}", weights shown whole)</h4>` + rows.map(([kind, nm, sh, M, note]) =>
    `<div class="lv ${kind}"><span class="nm" title="${esc(nm)}">${esc(nm)}</span><span class="sh">(${sh})</span><span>${M ? heat(Array.isArray(M[0]) ? M : [M], { name: nm, cell: kind === 'wt' ? (shape(M)[1] > 30 ? 2 : 3) : (M.length > 30 ? 3 : 7), scale: nm.startsWith('att') ? 1 : undefined }) : `<span class="muted">${esc(note)}</span>`}</span></div>`).join('');
}
const stepDur = () => +$('speed').value;
const particleDur = () => Math.min(700, stepDur() * 0.3);
function setStep(i, animate = true) {
  S.step = Math.max(0, Math.min(STEPS.length - 1, i));
  const s = STEPS[S.step];
  $('stepNo').textContent = `${S.step + 1} / ${STEPS.length}`;
  $('stepName').textContent = s.t + (s.l != null ? ` (block ${s.l + 1})` : '');
  $('scrub').value = S.step + 1;
  if (animate) WB.graph.forward(S.step, particleDur());
  else { WB.graph.anim++; WB.graph.N.forEach(n => { n.el.classList.toggle('cur', n.step === S.step); n.el.classList.toggle('done', n.step < S.step); }); }
  renderScope(); highlightWbCode(); renderLocals();
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
      body = `<p>Text is split into tokens and each token becomes its id in the vocabulary. ${isChars() ? `This model reads one letter at a time: ${V} tokens, a to z plus &lt;s&gt; for the start and end of a name.` : `This model knows ${V} whole words.`} Real models use 100,000 to 200,000 sub-word pieces.</p>
        ${where('No weights yet. The tokenizer is a fixed lookup table built before training, not a learned matrix.', true)}
        <div class="chips" style="margin:12px 0">${S.ids.map((id, i) => `<span class="chip ${i === f ? 'focus' : ''}">${esc(VOCAB[id])}<sub>id ${id}</sub></span>`).join('')}</div>`;
      break;
    case 'emb':
      body = `<p>The first weights arrive. Each token id selects one row of <span class="wtag">W_E</span>, and that 16-number row becomes the token's vector. It's a memory read, equivalent to multiplying a one-hot vector by the matrix.</p>
        ${where(`<span class="w">W_E</span> (${V}×16) enters. Outlined rows are the ones your input selects; the thick outline is "${esc(tokW)}".`)}
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
      head += `<p>The last weights. The final vector is dotted with each column of <span class="wtag">W_U</span>, one per vocabulary token, giving ${V} scores (logits).</p>` + where(`<span class="w">W_U</span> (16×${V}) enters. In Llama 3 this matrix is 16,384 × 128,256.`);
      mm(TR.hf[f], W0.W_U, TR.logits[f], { xname: 'hf', wname: 'W_U', oname: 'logits', colLabels: VOCAB });
      return;
    case 'sm':
      body = `<p>Softmax turns the ${V} scores into probabilities. One token is picked, appended to the input, and the whole pass runs again from step 1 with the same weights.</p>
        ${where('No weights. The pass is over: every weight in the model was used once for this token.', true)}
        <div class="bars" style="margin-top:12px">${bars(TR.probs[f], 8)}</div>
        <p class="cap">${f === S.ids.length - 1 ? 'This is the prediction for the next token.' : `This is what the model predicted after "${esc(tokW)}".`}</p>`;
      break;
  }
  el.innerHTML = head + body + streamHTML();
}



/* ================= training ================= */
const TRN = (() => {
  const t = { graph: createGraph('tgraph', 'tgscroll', 't'), code: createCodeView('trCode', ['train.py', 'model.py']), running: false, walking: false, raf: 0 };
  let W, opt, step, curve, seed, rand, lastG, lastUpd, prevSel, samples, phase = -1, walkInfo = null, tokensSeen;
  const BATCH = 8, WALK_MS = 1700;
  t.sel = 'L0.W_Q';
  t.graph.onSlot = w => { t.sel = w; t.renderCenter(); };
  const mkRand = s => () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  t.reset = () => {
    t.stop(); seed = 1 + Math.floor(Math.random() * 1e6); rand = mkRand(seed);
    W = Core.initWeights(CFG, rand); opt = {}; step = 0; curve = []; lastG = null; lastUpd = null; prevSel = null; samples = []; phase = -1; walkInfo = null; tokensSeen = 0;
    curve.push([0, evalLoss(), evalLoss()]);
    if (t.graph.N.length) t.graph.clear();
    t.renderCenter(); t.status(); t.code.highlight('train.py', ['def train('], [], []);
  };
  const lrAt = s => 0.01 * Math.min(1, s / 50);
  function batch() {
    const c = MODEL.corpus, xs = [], ys = [];
    for (let k = 0; k < BATCH; k++) { const st = Math.floor(rand() * (c.length - T - 1)); xs.push(c.slice(st, st + T)); ys.push(c.slice(st + 1, st + T + 1)); }
    return [xs, ys];
  }
  function evalLoss() { const e = MODEL.eval; return Core.loss(W, CFG, e.x.slice(0, 12), e.y.slice(0, 12)); }
  function sample(n) {
    const out = [];
    for (let k = 0; k < n; k++) {
      const cache = Array.from({ length: L }, () => ({ k: [], v: [] })); let tok = 0, s = [];
      for (let pos = 0; pos < T; pos++) {
        const p = Core.step(W, CFG, tok, pos, cache).probs.map(v => Math.pow(v, 1 / 0.8)); const z = p.reduce((a, b) => a + b, 0);
        let r = rand() * z, i = 0; while (i < p.length - 1 && (r -= p[i]) > 0) i++;
        tok = i; if (tok === 0) { if (isChars()) break; s.push('.'); if (s.length > 4) break; continue; }
        s.push(VOCAB[tok]);
      }
      out.push(isChars() ? s.join('') : s.join(' '));
    }
    return out;
  }
  function doStep() {
    const [xs, ys] = batch();
    prevSel = W[t.sel] && (Array.isArray(W[t.sel][0]) ? W[t.sel].map(r => r.slice()) : W[t.sel].slice());
    step++;
    const r = Core.trainStep(W, CFG, xs, ys, opt, lrAt(step));
    lastG = r.G; lastUpd = r.upd; tokensSeen += BATCH * T;
    return r;
  }
  t.status = () => { $('trStat').textContent = `step ${fmt(step)}  loss ${curve.length ? curve.at(-1)[1].toFixed(3) : '-'}`; };
  function glowFromGrads() {
    if (!lastG) return;
    const norms = {}; let mx = 1e-12;
    for (const k in lastG) { const f = Array.isArray(lastG[k][0]) ? lastG[k].flat() : lastG[k]; norms[k] = Math.sqrt(f.reduce((s, v) => s + v * v, 0) / f.length); mx = Math.max(mx, norms[k]); }
    for (const k in norms) norms[k] = Math.sqrt(norms[k] / mx);
    t.graph.glow(norms);
  }
  t.run = () => {
    if (t.walking) return;
    t.running = true; $('trRun').textContent = 'Pause'; t.graph.clear();
    t.code.highlight('train.py', ['xb, yb = batch(', 'probs = model.forward(xb, tr)', 'loss = model.cross_entropy', 'grads = model.backward(tr, yb)', 'model.W[k] -= lr_t'], [], ['', '', '', '', 'every weight, every step'], null, 'for step in range');
    let lastEval = 0;
    const loop = () => {
      if (!t.running) return;
      const k = +$('trSpeed').value; let r;
      for (let i = 0; i < k; i++) r = doStep();
      if (step - lastEval >= 25 || step < 25) { curve.push([step, r.loss, evalLoss()]); lastEval = step; }
      else curve.push([step, r.loss, curve.at(-1)[2]]);
      if (curve.length > 600) curve = curve.filter((_, i) => i % 2 === 0 || i > curve.length - 50);
      if (step % 150 < k) samples = sample(6);
      glowFromGrads(); t.renderCenter(); t.status();
      t.raf = requestAnimationFrame(loop);
    };
    loop();
  };
  t.stop = () => { t.running = false; cancelAnimationFrame(t.raf); $('trRun').textContent = 'Train'; };

  /* ----- one step, walked through slowly ----- */
  const BWD = [
    { id: 'bU', nodes: ['sm', 'un'], w: ['W_U'], t: 'Gradient reaches the unembedding', p: ['dlogits = tr["probs"].copy()', 'dlogits /= targets.size', 'G["W_U"] = sum_rows(tr["hf"], dlogits)', 'dhf = dlogits @ W["W_U"].T'], txt: 'The error signal starts at the output: predicted probability minus 1 for the correct token, plus the probability of every wrong one. Multiplying it back through <span class="wtag">W_U</span> gives W_U\'s gradient and passes the signal down.' },
    { id: 'bgf', nodes: ['nf'], w: ['g_f'], t: 'Through the final norm', p: ['dx, G["g_f"] = rmsnorm_backward'], s: ['def rmsnorm_backward'], txt: 'The norm\'s gain gets its gradient and the signal passes through the normalization.' }
  ];
  for (let l = 1; l >= 0; l--) BWD.push(
    { id: 'bd' + l, l, nodes: ['a2' + l, 'dn' + l], w: [`L${l}.W_down`], t: `Block ${l + 1}: MLP shrink`, p: ['G[f"L{l}.W_down"] = sum_rows', 'dact = dx @ P("W_down").T'], txt: 'The residual add copies the gradient straight through, and a branch goes into the MLP. <span class="wtag">W_down</span> receives its gradient.' },
    { id: 'bu' + l, l, nodes: ['ge' + l, 'up' + l], w: [`L${l}.W_up`], t: `Block ${l + 1}: MLP expand`, p: ['dup = dact * gelu_backward', 'G[f"L{l}.W_up"] = sum_rows', 'dh2 = dup @ P("W_up").T'], s: ['def gelu_backward'], txt: 'Back through GELU (its slope), then <span class="wtag">W_up</span> gets its gradient.' },
    { id: 'bn' + l, l, nodes: ['n2' + l, 'a1' + l], w: [`L${l}.g2`], t: `Block ${l + 1}: norm before MLP`, p: ['dxm, G[f"L{l}.g2"] = rmsnorm_backward', 'dx = dx + dxm'], txt: 'The MLP branch\'s gradient rejoins the residual stream.' },
    { id: 'bo' + l, l, nodes: ['o' + l], w: [`L${l}.W_O`], t: `Block ${l + 1}: attention output`, p: ['G[f"L{l}.W_O"] = sum_rows', 'dmix = split_heads(dx @ P("W_O").T, H)'], txt: '<span class="wtag">W_O</span> receives its gradient; the signal splits back into the two heads.' },
    { id: 'ba' + l, l, nodes: ['mix' + l, 'sc' + l], w: [], t: `Block ${l + 1}: through attention`, p: ['datt = dmix @ vh.swapaxes', 'dvh = att.swapaxes', 'dscores = att * (datt', 'dq, dk, dv = merge_heads'], txt: 'Back through the blend and the softmax. No weights here, but this is how blame flows to <i>earlier tokens</i>: every token that was attended to receives gradient.' },
    { id: 'bq' + l, l, nodes: ['q' + l, 'k' + l, 'v' + l], w: [`L${l}.W_Q`, `L${l}.W_K`, `L${l}.W_V`], t: `Block ${l + 1}: Q, K, V`, p: ['G[f"L{l}.W_Q"] = sum_rows', 'G[f"L{l}.W_K"] = sum_rows', 'G[f"L{l}.W_V"] = sum_rows', 'dh1 = dq @ P("W_Q").T'], txt: 'All three projections receive gradients at once.' },
    { id: 'bg' + l, l, nodes: ['n1' + l], w: [`L${l}.g1`], t: `Block ${l + 1}: norm before attention`, p: ['dxi, G[f"L{l}.g1"] = rmsnorm_backward', 'dx = dx + dxi'], txt: 'The attention branch rejoins the residual stream. On to the block below.' });
  BWD.push({ id: 'be', nodes: ['pos', 'emb'], w: ['W_P', 'W_E'], t: 'Embeddings', p: ['G["W_P"][:T] = dx', 'np.add.at(G["W_E"]'], txt: 'Finally the rows of <span class="wtag">W_P</span> and <span class="wtag">W_E</span> that were used this batch get gradients. Rows for tokens not in the batch get zero.' });
  const PHASES = ['Batch', 'Forward', 'Loss', 'Backward', 'Update'];
  let wsteps = [];
  t.walk = async () => {
    if (t.walking) return;
    t.stop(); t.walking = true; $('trWalk').disabled = true; $('trRun').disabled = true;
    const [xs, ys] = batch();
    const before = clone(W), Wn = clone(W), optn = JSON.parse(JSON.stringify(opt));
    const r = Core.trainStep(Wn, CFG, xs, ys, optn, lrAt(step + 1));
    walkInfo = { xs, ys, loss: r.loss, G: r.G, before, after: Wn };
    const dur = Math.min(700, WALK_MS * 0.3), pause = ms => new Promise(res => setTimeout(res, reduce ? 0 : ms));
    wsteps = [
      { ph: 0, t: 'Pick a batch', f: 'train.py', p: ['xb, yb = batch(stream'], txt: `${BATCH} random windows of ${T} tokens from the training text. The targets are the same windows shifted by one: every position has to predict the next token.` },
      { ph: 1, t: 'Forward pass', f: 'train.py', p: ['probs = model.forward(xb, tr)'], txt: 'Exactly the forward pass from the Workbench, run on all 8 windows. The current weights fly in; nothing changes yet.', fwd: true },
      { ph: 2, t: 'Measure the loss', f: 'train.py', p: ['loss = model.cross_entropy(probs, yb)'], s: ['p = np.take_along_axis', 'return float(-np.mean(np.log'], txt: `Average of −log(probability given to the correct next token) = <b>${r.loss.toFixed(3)}</b>. A perfect model scores 0; random guessing scores about ${Math.log(V).toFixed(2)}.` },
      ...BWD.map(b => ({ ...b, ph: 3, f: 'model.py', bwd: true })),
      { ph: 4, t: 'Adam update: the weights change', f: 'train.py', p: ['m[k] = b1 * m[k]', 'v[k] = b2 * v[k]', 'model.W[k] -= lr_t'], txt: 'Every weight moves a small step against its gradient. Adam scales each step by that weight\'s recent gradient history. This single line is the only place in the whole project where a weight changes.', upd: true }
    ];
    for (let i = 0; i < wsteps.length; i++) {
      phase = i; const ws = wsteps[i];
      if (ws.w && ws.w.length) t.sel = ws.w[0];
      t.code.highlight(ws.f, ws.p, ws.s || [], [], ws.l != null ? ['for l in reversed(range(cfg["L"]))', `l = ${ws.l}  (block ${ws.l + 1})`] : null, ws.bwd ? 'def backward' : ws.f === 'train.py' ? 'for step in range' : null);
      t.renderCenter();
      if (ws.fwd) { t.graph.clear(); for (let s = 0; s < STEPS.length; s++) { await t.graph.forward(s, 120); } }
      else if (ws.bwd) { await t.graph.backward(ws.nodes, dur); await pause(dur * 0.6); }
      else if (ws.upd) {
        W = Wn; opt = optn; step++; tokensSeen += BATCH * T; lastG = r.G; lastUpd = r.upd;
        curve.push([step, r.loss, evalLoss()]);
        t.graph.flashUpdate(); t.renderCenter(); t.status(); await pause(WALK_MS * 1.2);
      } else await pause(WALK_MS * 0.8);
      if (!t.walking) break;
    }
    phase = -1; t.walking = false; $('trWalk').disabled = false; $('trRun').disabled = false; glowFromGrads(); t.renderCenter();
  };

  t.renderCenter = () => {
    if (S.section !== 'training' || !W) return;
    for (const k in TIPS) if (k.startsWith('t')) delete TIPS[k];
    const ws = phase >= 0 ? wsteps[phase] : null;
    const ph = ws ? ws.ph : -1;
    let h = `<h2>${ws ? esc(ws.t) : t.running ? 'Training' : step ? 'Paused' : 'Random weights, before any training'}</h2>`;
    h += `<div class="phases">${PHASES.map((p, i) => `<span class="${i === ph ? 'on' : i < ph ? 'done' : ''}">${i + 1}. ${p}</span>`).join('')}</div>`;
    if (ws) h += `<div class="where ${ws.bwd ? '' : 'none'}" style="${ws.bwd ? 'border-left-color:var(--grad)' : ''}">${ws.txt}</div>`;
    else h += `<p class="small muted">${step ? `Trained ${fmt(step)} steps on ${fmt(tokensSeen)} tokens.` : 'Every weight starts as small random noise. Press "Walk through one step" to watch a single training step in slow motion, or "Train" to run thousands of them.'} Weights in memory glow <span class="g">violet</span> by how hard their gradient is pushing them.</p>`;
    if (ws && ws.ph === 0 && walkInfo) h += `<div class="panel" style="margin:10px 0"><div class="small muted">window 1 of ${BATCH}</div><div class="mono">input:  ${esc(walkInfo.xs[0].map(i => VOCAB[i]).join(isChars() ? ' ' : ' '))}</div><div class="mono">target: ${esc(walkInfo.ys[0].map(i => VOCAB[i]).join(' '))}</div></div>`;
    h += `<div style="margin:12px 0">${lineChart([{ name: 'training loss', color: 'var(--weight)', pts: curve.map(c => [c[0], c[1]]) }, { name: 'held-out loss', color: 'var(--act)', dash: true, pts: curve.map(c => [c[0], c[2]]) }], { aria: 'loss over training steps' })}</div>`;
    const Wsel = W[t.sel], G = ws && ws.bwd && walkInfo ? walkInfo.G[t.sel] : lastG && lastG[t.sel];
    let dW = null;
    if (ws && ws.upd && walkInfo) dW = diff(walkInfo.after[t.sel], walkInfo.before[t.sel]);
    else if (!ws && prevSel && lastG) dW = diff(Wsel, prevSel);
    const opts = Object.keys(W).map(k => `<option ${k === t.sel ? 'selected' : ''}>${k}</option>`).join('');
    const M = x => Array.isArray(x[0]) ? x : [x];
    const cell = Array.isArray(Wsel[0]) ? Math.max(3, Math.min(10, Math.floor(200 / Wsel[0].length))) : 10;
    h += `<div class="panel"><h3 style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">Inside <select class="tsel" id="trSel" aria-label="Weight to inspect">${opts}</select><span class="small muted" style="font-weight:400">or select a slot in the memory rack</span></h3>
      <div class="trio"><div><h4 class="w">Weight</h4>${heat(M(Wsel), { name: t.sel, cell, frame: 'wframe' })}</div>
      <div><h4 class="g">Gradient</h4>${G ? heat(M(G), { name: 'grad ' + t.sel, cell, posClass: 'p', negClass: 'n' }) : '<p class="small muted">Appears after a backward pass.</p>'}</div>
      <div><h4>Change this step</h4>${dW ? heat(M(dW), { name: 'Δ ' + t.sel, cell }) : '<p class="small muted">Appears after an update.</p>'}</div></div>
      <p class="cap">The change is roughly the negative of the gradient: where the gradient is red the weight went down, where it's blue the weight went up.</p></div>`;
    if (samples.length || step) h += `<div class="panel" style="margin-top:14px"><h3>What it writes right now</h3><div class="samples">${(samples.length ? samples : sample(6)).map(s => `<span>${esc(s || '(empty)')}</span>`).join('')}</div><p class="cap">Sampled from the current weights. Watch these go from noise to ${isChars() ? 'plausible names' : 'grammatical sentences'}.</p></div>`;
    $('trCenter').innerHTML = h;
    const sel = $('trSel'); if (sel) sel.onchange = e => { t.sel = e.target.value; t.renderCenter(); };
  };
  const diff = (a, b) => Array.isArray(a[0]) ? a.map((r, i) => r.map((v, j) => v - b[i][j])) : a.map((v, i) => v - b[i]);
  t.weights = () => W;
  $('trRun').onclick = () => t.running ? t.stop() : t.run();
  $('trWalk').onclick = () => t.walk();
  $('trReset').onclick = () => t.reset();
  $('trUse').onclick = async () => {
    if (!step) return;
    CUSTOM[S.model] = clone(W); useModel(S.model); updateBadge(); WB.graph.build();
    await changed();
    $('trUse').textContent = 'Now used everywhere';
    setTimeout(() => $('trUse').textContent = 'Use these weights in the app', 2200);
  };
  return t;
})();

/* ================= overview, anatomy ================= */
function renderOverview() { $('ovBars').innerHTML = bars(TR.probs.at(-1), 6); $('ovCtx').textContent = `after "${joinToks(S.ids.slice(1))}"`; }
const ORGANS = () => [['Input: tokens into vectors', ['W_E', 'W_P']],
  ['Block 1', ['L0.g1', 'L0.W_Q', 'L0.W_K', 'L0.W_V', 'L0.W_O', 'L0.g2', 'L0.W_up', 'L0.W_down']],
  ['Block 2', ['L1.g1', 'L1.W_Q', 'L1.W_K', 'L1.W_V', 'L1.W_O', 'L1.g2', 'L1.W_up', 'L1.W_down']],
  ['Output: vector into token scores', ['g_f', 'W_U']]];
const ROLE = {
  W_E: ['Token embedding', 'One row per vocabulary token. Row i is the starting vector for token i. Looked up, not multiplied.', 'vocab', null],
  W_P: ['Position embedding', 'One row per position 0 to 15, added to the token vector so the model knows order.', 'pos', null],
  g1: ['Norm gain (before attention)', 'After the vector is rescaled to a standard size, each of its 16 numbers is multiplied by the matching gain.', null, null],
  W_Q: ['Query projection', 'Turns each token vector into a query. Columns 0 to 7 feed head 1, 8 to 15 feed head 2.', null, null],
  W_K: ['Key projection', 'Turns each token vector into a key. Queries and keys are compared to decide attention.', null, null],
  W_V: ['Value projection', 'Turns each token vector into a value: the information copied when attended to.', null, null],
  W_O: ['Output projection', 'Merges both heads back into the residual stream. Rows 0 to 7 come from head 1, 8 to 15 from head 2.', null, null],
  g2: ['Norm gain (before MLP)', 'Same role as the first gain, in front of the MLP.', null, null],
  W_up: ['MLP up-projection', 'Expands 16 numbers to 64. Each column is a detector for a pattern in the vector.', null, null],
  W_down: ['MLP down-projection', 'Maps 64 detector outputs back to 16 numbers. Row k is what detector k writes.', null, null],
  g_f: ['Final norm gain', 'Rescales the final vector before scoring.', null, null],
  W_U: ['Unembedding', 'One column per vocabulary token. The final vector dotted with column i is token i\'s score.', null, 'vocab']
};
const lab = x => x === 'vocab' ? VOCAB : x === 'pos' ? [...Array(T).keys()].map(i => 'pos ' + i) : null;
const roleOf = k => { const r = ROLE[k.replace(/^L\d\./, '')]; return [r[0], r[1], lab(r[2]), lab(r[3])]; };
function renderAnatomy() {
  const total = paramCount(W0);
  let h = '<thead><tr><th>Tensor</th><th>Shape</th><th style="text-align:right">Numbers</th><th style="width:70px">Share</th></tr></thead><tbody>';
  ORGANS().forEach(([title, ks]) => {
    h += `<tr class="organ"><td colspan="2">${title}</td><td class="n">${fmt(ks.reduce((s, k) => s + count(W0[k]), 0))}</td><td></td></tr>`;
    ks.forEach(k => { const n = count(W0[k]); h += `<tr class="row ${S.sel === k ? 'sel' : ''}" data-k="${k}" tabindex="0"><td><span class="wtag">${k}</span></td><td class="mono">${shape(W0[k]).join(' × ')}</td><td class="n">${fmt(n)}</td><td><div class="pb" style="width:${Math.max(2, n / total * 260)}px"></div></td></tr>`; });
  });
  $('ftable').innerHTML = h + `<tr class="organ"><td colspan="2">Total</td><td class="n">${fmt(total)}</td><td></td></tr></tbody>`;
  const k = S.sel, t = W0[k], r = roleOf(k), sh = shape(t), M = sh.length === 1 ? [t] : t;
  const cell = sh.length === 1 ? 22 : Math.max(6, Math.min(18, Math.floor(520 / M[0].length)));
  const flat = M.flat(), mean = flat.reduce((a, b) => a + b, 0) / flat.length, sd = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
  $('inspector').innerHTML = `<div class="tguide">${TG.body(k) || ''}</div>
    <p class="small muted" style="margin-top:12px">Shape ${sh.join(' × ')}, mean ${mean.toFixed(3)}, spread ${sd.toFixed(3)}, largest |value| ${maxAbs(M).toFixed(3)}. Hover any cell for its exact value.</p>
    <div class="scroll">${heat(M, { name: k, cell, rowLabels: r[2] || (sh.length === 1 ? ['gain'] : null), colLabels: r[3], lw: 52, frame: 'wframe' })}</div>`;
  const c = MODEL.loss_curve;
  $('lossChart').innerHTML = lineChart([{ name: 'training loss', color: 'var(--weight)', pts: c.map(q => [q[0], q[1]]) }, { name: 'held-out loss', color: 'var(--act)', dash: true, pts: c.map(q => [q[0], q[2]]) }], { aria: 'shipped model training loss' }) +
    `<p class="cap">${isChars() ? 'Names are genuinely unpredictable (after "ma" many letters are fine), so the loss levels off far above zero.' : 'The loss can\'t reach zero: after "the cat" both "sat" and "slept" are genuinely possible, so the best answer is a 50/50 split.'} Watch this happen live in the Training tab.</p>`;
}
$('ftable').addEventListener('click', e => { const r = e.target.closest('tr.row'); if (r) { S.sel = r.dataset.k; renderAnatomy(); } });
$('ftable').addEventListener('keydown', e => { const r = e.target.closest('tr.row'); if (r && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); S.sel = r.dataset.k; renderAnatomy(); } });

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

/* ================= generation with a KV cache ================= */
function sampleIdx(p, temp) {
  if (temp <= 0.001) return p.indexOf(Math.max(...p));
  const q = Core.softmax(p.map(v => Math.log(Math.max(v, 1e-12)) / temp)); let r = Math.random(), a = 0;
  for (let i = 0; i < q.length; i++) { a += q[i]; if (r <= a) return i; }
  return q.length - 1;
}
function prefill(ids) { const cache = Array.from({ length: L }, () => ({ k: [], v: [] })); let res; ids.forEach((t, i) => res = Core.step(W0, CFG, t, i, cache)); return { cache, res }; }
function genInit() {
  const { cache, res } = prefill(S.ids);
  S.gen = { ids: S.ids.slice(), start: S.ids.length, log: [], passes: 0, cache, res, last: res.probs, dropped: 0, prefilled: S.ids.length, rebuilt: 0 };
}
function genStep() {
  const g = S.gen, pick = sampleIdx(g.res.probs, +$('temp').value), p = g.res.probs;
  g.ids.push(pick);
  let note = '';
  if (g.ids.length > T) { g.ids.shift(); g.start--; g.dropped++; const pf = prefill(g.ids.slice(0, -1)); g.cache = pf.cache; g.rebuilt++; note = ' (window slid: cache rebuilt)'; }
  g.res = Core.step(W0, CFG, pick, g.ids.length - 1, g.cache);
  g.passes++; g.last = g.res.probs;
  g.log.unshift(`<div><span class="mono">pass ${g.passes}</span><span>picked <b>${esc(VOCAB[pick])}</b> at ${(p[pick] * 100).toFixed(0)}%, computed 1 new cache row, reused ${g.ids.length - 1}${note}</span></div>`);
}
function renderGen() {
  if (!S.gen.ids) genInit();
  const g = S.gen, sep = isChars() ? '' : ' ';
  $('genText').innerHTML = (g.dropped ? '<span class="muted">… </span>' : '') + g.ids.map((id, i) => {
    if (i === 0 && id === 0) return '';
    const t = id === 0 ? (isChars() ? ' / ' : ' &lt;s&gt; ') : esc(VOCAB[id]);
    return `<span class="${i >= g.start ? 'new' + (i === g.ids.length - 1 && g.passes ? ' fresh' : '') : ''}">${t}</span>`;
  }).join(sep);
  $('genBars').innerHTML = bars(g.last, 6);
  $('genLog').innerHTML = (g.log.join('') || '<p class="muted small">No passes yet. The input was pushed through once to fill the cache.</p>');
  const pc = paramCount(W0);
  $('genCounter').innerHTML = `<div><b>${g.passes}</b>tokens written</div><div><b>${g.ids.length}</b>rows in each cache</div><div><b>${fmt(g.passes * pc)}</b>weight uses (about ${fmt(pc)} per pass)</div><div><b>0</b>weights changed</div>`;
  const t = +$('temp').value;
  $('genTempNote').textContent = t === 0 ? 'Temperature 0 always takes the top token.' : `Temperature ${t} samples from these odds${t > 1 ? ', flattened, so unlikely tokens show up more' : ''}.`;
  const labels = g.ids.map((id, i) => `${i}:${VOCAB[id]}`), last = g.ids.length - 1;
  let kv = '<div class="kvgrid">';
  for (let l = 0; l < L; l++) {
    const att = g.res.att[l];
    kv += `<div><h4>Block ${l + 1}</h4><div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start">
      <div><div class="small a">keys</div>${heat(g.cache[l].k, { name: `K cache block ${l + 1}`, cell: 9, rowLabels: labels, lw: 58, focusRow: last, frame: 'aframe' })}</div>
      <div><div class="small a">values</div>${heat(g.cache[l].v, { name: `V cache block ${l + 1}`, cell: 9, focusRow: last, frame: 'aframe' })}</div>
      <div><div class="small muted">newest token's attention</div>${att.map((a, h) => `<div class="small">head ${h + 1}</div>` + a.map((w, j) => `<div style="display:flex;gap:6px;align-items:center;font-size:12px"><span style="width:52px;overflow:hidden">${esc(VOCAB[g.ids[j]])}</span><div style="width:${Math.max(1, w * 90)}px;height:8px;background:var(--act);border-radius:2px"></div></div>`).join('')).join('')}</div>
    </div></div>`;
  }
  $('kvView').innerHTML = kv + '</div>' + `<p class="cap">Without the cache, pass ${g.passes + 1} would recompute keys and values for all ${g.ids.length} tokens in every block. With it, only 1 new row per block. The weights are still read in full each pass, which is why generation speed is usually limited by memory bandwidth, not arithmetic.</p>`;
}
$('gen1').onclick = () => { genStep(); renderGen(); };
$('gen8').onclick = () => { let n = 0; const run = () => { genStep(); renderGen(); if (++n < 8) setTimeout(run, reduce ? 0 : 260); }; run(); };
$('genReset').onclick = () => { genInit(); renderGen(); };
$('temp').oninput = e => { $('tempV').textContent = e.target.value; renderGen(); };

/* ================= surgery ================= */
const PROBES = [['the cat', ['sat', 'slept']], ['the fish', ['swam']], ['the dog ran', ['to']], ['the bird flew over the', ['tree', 'house']], ['the fish swam in the', ['pond', 'lake']], ['the kid sat on the', ['mat', 'bed', 'chair']], ['the big dog ran to the', ['park', 'house', 'store']], ['the cat slept on the', ['mat', 'bed']]];
$('headBoxes').innerHTML = [0, 1].flatMap(l => [0, 1].map(h => `<label><input type="checkbox" data-h="${l}-${h}"> Block ${l + 1}, head ${h + 1}</label>`)).join('');
$('mlpBoxes').innerHTML = [0, 1].map(l => `<label><input type="checkbox" data-m="${l}"> Block ${l + 1} MLP</label>`).join('');
function fillNoiseSelect() { $('noiseT').innerHTML = Object.keys(W0).filter(k => shape(W0[k]).length === 2).map(k => `<option ${k === S.surg.noiseT ? 'selected' : ''}>${k}</option>`).join(''); }
function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function patched() {
  const Wp = clone(W0);
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
  if (!isChars()) {
    let p0 = 0, p1 = 0;
    const top = p => VOCAB[p.indexOf(Math.max(...p))];
    const rows = PROBES.map(([txt, ok]) => {
      const ids = encode(txt), a = top(Core.forward(W0, CFG, ids).probs.at(-1)), b = top(Core.forward(Wp, CFG, ids).probs.at(-1));
      const oa = ok.includes(a), ob = ok.includes(b); p0 += oa; p1 += ob;
      return `<tr><td>${txt} …</td><td>${a} <span class="${oa ? 'ok' : 'bad'}">${oa ? 'pass' : 'fail'}</span></td><td>${b} <span class="${ob ? 'ok' : 'bad'}">${ob ? 'pass' : 'fail'}</span></td></tr>`;
    });
    $('probePanel').innerHTML = `<h3>Ability check</h3><p class="small muted">Each probe has a set of correct next words from the training world. A probe passes when the top prediction is one of them.</p><table class="probe"><tr><th>Probe</th><th>Original (${p0}/${PROBES.length})</th><th>After surgery (${p1}/${PROBES.length})</th></tr>${rows.join('')}</table>`;
  } else {
    const e = MODEL.eval, x = e.x.slice(0, 16), y = e.y.slice(0, 16);
    const l0 = Core.loss(W0, CFG, x, y), l1 = Core.loss(Wp, CFG, x, y);
    const names = W => { const r = rng(7), out = []; for (let k = 0; k < 8; k++) { const c = Array.from({ length: L }, () => ({ k: [], v: [] })); let t = 0, s = ''; for (let pos = 0; pos < T; pos++) { const p = Core.step(W, CFG, t, pos, c).probs; let u = r(), i = 0; while (i < p.length - 1 && (u -= p[i]) > 0) i++; t = i; if (!t) break; s += VOCAB[t]; } out.push(s || '(empty)'); } return out; };
    $('probePanel').innerHTML = `<h3>Ability check</h3><p class="small muted">Loss on names the model never saw in training (lower is better; random guessing is ${Math.log(V).toFixed(2)}), and names sampled with the same random seed before and after.</p>
      <table class="probe"><tr><th></th><th>Original</th><th>After surgery</th></tr><tr><td>Held-out loss</td><td>${l0.toFixed(3)}</td><td><span class="${l1 > l0 + 0.05 ? 'bad' : 'ok'}">${l1.toFixed(3)}</span></td></tr>
      <tr><td>Sampled names</td><td class="mono">${names(W0).join(', ')}</td><td class="mono">${names(Wp).join(', ')}</td></tr></table>`;
  }
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
function renderScale() { $('paramEq').innerHTML = `12 × ${L} × ${D}² = ${fmt(12 * L * D * D)}  +  embeddings ${V}×${D} + ${D}×${V} + ${T}×${D} = ${fmt(2 * V * D + T * D)}  +  norm gains ${fmt(5 * D)}  =  <b>${fmt(12 * L * D * D + 2 * V * D + T * D + 5 * D)}</b>`; }

/* ================= tensor guide: plain-English + math popover for every weight ================= */
const TG = (() => {
  const card = document.createElement('div');
  card.id = 'tcard'; card.setAttribute('role', 'tooltip'); document.body.appendChild(card);
  let showT = 0, hideT = 0, curEl = null;
  const n3 = v => (v >= 0 ? '' : '−') + Math.abs(v).toFixed(3);
  const vec4 = a => '[' + Array.from(a).slice(0, 4).map(n3).join(', ') + ', …]';
  const M = (s) => `<div class="math">${s}</div>`;
  const sub = (a, b) => `<i>${a}</i><sub>${b}</sub>`;
  const curLayer = () => (S.section === 'workbench' && STEPS[S.step] && STEPS[S.step].l != null) ? STEPS[S.step].l : 0;

  function resolve(el) {
    if (el.dataset.tensor) return el.dataset.tensor;
    if (el.classList.contains('slot')) return el.dataset.w;
    if (el.matches('tr.row')) return el.dataset.k;
    const txt = el.textContent;
    const m = txt.match(/(L\d\.)?(W_[A-Za-z]+|g_f|g1|g2)/);
    if (!m) return null;
    if (m[1]) return m[1] + m[2];
    return ['W_E', 'W_P', 'W_U', 'g_f'].includes(m[2]) ? m[2] : `L${curLayer()}.${m[2]}`;
  }

  function body(name) {
    const role = name.replace(/^L\d\./, ''), l = +(name.match(/^L(\d)/) || [0, 0])[1];
    const W = W0[name]; if (!W || !TR) return null;
    const f = focusPos(), id = S.ids[f], tok = esc(VOCAB[id]), Ly = TR.layers[l];
    const shp = shape(W).join(' × '), n = count(W), pct = (n / paramCount(W0) * 100).toFixed(1);
    const blk = name.startsWith('L') ? ` in block ${l + 1}` : '';
    const eg = s => `<div class="eg"><b>With your input</b> (following "${tok}", position ${f}): ${s}</div>`;
    const dot = (x, Wm, j) => { const t = x.map((xi, i) => [xi * Wm[i][j], i]); return t.reduce((s, [v]) => s + v, 0); };
    const firstTerms = (x, Wm, j, nm) => `${n3(x[0])}×${n3(Wm[0][j])} + ${n3(x[1])}×${n3(Wm[1][j])} + … (${x.length} terms) = <b>${n3(dot(x, Wm, j))}</b>`;
    const C = {
      W_E: {
        t: 'Token embedding table',
        plain: `A lookup table with one row per token in the vocabulary (${V} rows). Each row is the model's starting description of that token, written as ${D} numbers. Tokens that behave alike in the training text end up with similar rows.`,
        like: 'A dictionary where every word\'s definition is written as 16 numbers instead of words.',
        math: M(`${sub('x', 't')} = <i>W</i><sub>E</sub>[ id<sub>t</sub> ]`) + '<p>Pick the row whose number is the token\'s id. No multiplication happens; it is a pure lookup (the same as multiplying a one-hot vector by the table).</p>',
        dims: `rows = tokens, columns = ${D} features. No column has a fixed meaning; training decides what each one ends up tracking.`,
        ex: () => eg(`"${tok}" has id ${id}, so its vector is row ${id}: ${vec4(W[id])}`)
      },
      W_P: {
        t: 'Position table',
        plain: 'One row per position in the context window (0 to 15). Attention by itself treats the input like a bag of tokens, so this row is added to say "I am the 3rd token", "I am the 4th token", and so on.',
        like: 'Seat numbers in a theatre: the same person means something different in row 1 than in row 15.',
        math: M(`${sub('x', 't')} = <i>W</i><sub>E</sub>[ id<sub>t</sub> ] + <i>W</i><sub>P</sub>[ <i>t</i> ]`) + '<p>Add, entry by entry, the row for the token\'s position.</p>',
        dims: `rows = positions 0..${T - 1}, columns = the same ${D} features as W_E, so the two rows can be added.`,
        ex: () => eg(`row ${f} is ${vec4(W[f])}; added to the token row it gives ${vec4(TR.x0[f])}`)
      },
      norm: {
        t: `Normalization gain${blk}`,
        plain: `Before ${role === 'g1' ? 'attention' : role === 'g2' ? 'the MLP' : 'scoring the vocabulary'}, the vector is shrunk or stretched so its typical size is 1. That keeps numbers from growing out of control as blocks stack up. Then each of the ${D} positions is multiplied by its own learned "volume knob": these ${D} numbers.`,
        like: 'Setting a microphone to a standard level, then adjusting each frequency band on an equalizer.',
        math: M(`rms(<i>x</i>) = √( (1/${D}) Σ<sub>i</sub> ${sub('x', 'i')}² + ε )`) + M(`${sub('h', 'i')} = ${sub('x', 'i')} ÷ rms(<i>x</i>) × ${sub('g', 'i')}`) + '<p>ε = 0.00001 only prevents dividing by zero.</p>',
        dims: `a single row of ${D} gains, one per feature. They start at 1.0.`,
        ex: () => {
          const x = role === 'g1' ? Ly.xin[f] : role === 'g2' ? Ly.xmid[f] : TR.layers.at(-1).xout[f];
          const r = Math.sqrt(x.reduce((s, v) => s + v * v, 0) / D + 1e-5);
          return eg(`rms(x) = ${n3(r)}, so h<sub>0</sub> = ${n3(x[0])} ÷ ${n3(r)} × ${n3(W[0])} = <b>${n3(x[0] / r * W[0])}</b>`);
        }
      },
      W_Q: {
        t: `Query projection${blk}`,
        plain: 'Turns each token\'s vector into a question: "what kind of earlier token am I looking for?" Columns 0 to 7 make the question for head 1, columns 8 to 15 for head 2.',
        like: 'Typing a search query. W_K writes the labels on every page; attention matches query against labels.',
        math: M(`<i>q</i> = <i>h</i> · <i>W</i><sub>Q</sub>, &nbsp; ${sub('q', 'j')} = Σ<sub>i</sub> ${sub('h', 'i')} <i>W</i><sub>Q</sub>[<i>i</i>][<i>j</i>]`) + M(`score(<i>i</i>→<i>j</i>) = ( ${sub('q', 'i')} · ${sub('k', 'j')} ) ÷ √<span style="text-decoration:overline">${DH}</span>`),
        dims: `${D} × ${D}: row i = input feature, column j = one query number. Each query number is a weighted sum of all ${D} inputs, so the matrix does ${D * D} multiply-adds per token.`,
        ex: () => eg(`q<sub>0</sub> = ${firstTerms(Ly.h1[f], W, 0)}`)
      },
      W_K: {
        t: `Key projection${blk}`,
        plain: 'Turns each token\'s vector into a label: "here is what I contain." A token\'s query is compared with the key of every earlier token; big matches get attention.',
        like: 'The label on a file folder. The query decides which folders to open.',
        math: M(`<i>k</i> = <i>h</i> · <i>W</i><sub>K</sub>`) + M(`${sub('a', 'ij')} = softmax<sub><i>j</i></sub>( ${sub('q', 'i')} · ${sub('k', 'j')} ÷ √<span style="text-decoration:overline">${DH}</span> ) &nbsp; for <i>j</i> ≤ <i>i</i>`) + '<p>Softmax turns the scores into percentages that add up to 100%. Future tokens are masked out.</p>',
        dims: `${D} × ${D}, split into two ${DH}-column halves for the two heads.`,
        ex: () => { const a = Ly.att[0][f], j = a.indexOf(Math.max(...a)); return eg(`head 1 gives "${esc(VOCAB[S.ids[j]])}" the biggest match: score ${n3(Ly.scores[0][f][j])}, which softmax turns into <b>${(a[j] * 100).toFixed(0)}%</b>`); }
      },
      W_V: {
        t: `Value projection${blk}`,
        plain: 'Turns each token\'s vector into the message it hands over when another token pays attention to it. Keys decide who gets attention; values are what actually gets copied.',
        like: 'The contents inside the folder, as opposed to the label on it.',
        math: M(`<i>v</i> = <i>h</i> · <i>W</i><sub>V</sub>`) + M(`${sub('blend', 'i')} = Σ<sub><i>j</i></sub> ${sub('a', 'ij')} ${sub('v', 'j')}`) + '<p>A weighted average of earlier tokens\' values, weighted by attention.</p>',
        dims: `${D} × ${D}; columns 0 to 7 feed head 1, 8 to 15 feed head 2.`,
        ex: () => eg(`v<sub>0</sub> for "${tok}" = ${firstTerms(Ly.h1[f], W, 0)}`)
      },
      W_O: {
        t: `Attention output projection${blk}`,
        plain: 'Takes the two heads\' blended results (8 + 8 numbers) and mixes them back into one 16-number update, which is then added to the token\'s running vector.',
        like: 'Two reporters each bring notes; the editor combines them into one paragraph and appends it to the story.',
        math: M(`Δ<i>x</i> = blend · <i>W</i><sub>O</sub>`) + M(`<i>x</i> ← <i>x</i> + Δ<i>x</i>`) + '<p>Adding (not replacing) is the residual connection: information from earlier steps is kept.</p>',
        dims: `${D} × ${D}: rows 0 to 7 read head 1's output, rows 8 to 15 read head 2's.`,
        ex: () => eg(`Δx<sub>0</sub> = ${firstTerms(Ly.mix[f], W, 0)}`)
      },
      W_up: {
        t: `MLP expand${blk}`,
        plain: `Widens the vector from ${D} to ${F} numbers. Each of the ${F} columns is a pattern detector: its output is large when the vector looks like that column. GELU then switches off detectors that scored negative.`,
        like: `${F} specialists each reading the same memo and raising a hand, strongly or weakly, if it matches their specialty.`,
        math: M(`${sub('u', 'k')} = Σ<sub>i</sub> ${sub('h', 'i')} <i>W</i><sub>up</sub>[<i>i</i>][<i>k</i>]`) + M(`${sub('a', 'k')} = GELU(${sub('u', 'k')}) = ½ ${sub('u', 'k')} ( 1 + tanh( 0.798 ( ${sub('u', 'k')} + 0.0447 ${sub('u', 'k')}³ ) ) )`) + '<p>GELU is close to "keep positives, zero out negatives", with a smooth bend near zero.</p>',
        dims: `${D} × ${F}: column k is detector k's template.`,
        ex: () => { const a = Ly.act[f], k = a.indexOf(Math.max(...a)); return eg(`the strongest detector is #${k}: u = ${n3(Ly.up[f][k])}, after GELU a = <b>${n3(a[k])}</b>. ${a.filter(v => v > 0.05).length} of ${F} detectors fire.`); }
      },
      W_down: {
        t: `MLP shrink${blk}`,
        plain: `Maps the ${F} detector outputs back to ${D} numbers and adds them to the token's vector. Row k is what detector k writes whenever it fires. Most of what a model "knows" is stored in these two MLP matrices.`,
        like: 'Each specialist who raised a hand adds their own sentence to the memo.',
        math: M(`Δ<i>x</i> = Σ<sub>k</sub> ${sub('a', 'k')} <i>W</i><sub>down</sub>[<i>k</i>]`) + M(`<i>x</i> ← <i>x</i> + Δ<i>x</i>`),
        dims: `${F} × ${D}: row k = what detector k contributes.`,
        ex: () => { const a = Ly.act[f], k = a.indexOf(Math.max(...a)); return eg(`detector #${k} fired at ${n3(a[k])}, so it adds ${n3(a[k])} × row ${k} = ${vec4(W[k].map(v => v * a[k]))}`); }
      },
      W_U: {
        t: 'Unembedding (output scores)',
        plain: `One column per vocabulary token. The final vector is compared with every column; the better it matches, the higher that token's score. Softmax turns the ${V} scores into probabilities for the next token.`,
        like: 'A panel of judges, one per possible next token, each scoring how well the final vector fits them.',
        math: M(`${sub('z', 'w')} = Σ<sub>i</sub> ${sub('h', 'i')} <i>W</i><sub>U</sub>[<i>i</i>][<i>w</i>]`) + M(`${sub('p', 'w')} = e<sup>${sub('z', 'w')}</sup> ÷ Σ<sub><i>u</i></sub> e<sup>${sub('z', 'u')}</sup>`),
        dims: `${D} × ${V}: column w belongs to token w.`,
        ex: () => { const p = TR.probs[f], w = p.indexOf(Math.max(...p)); return eg(`the top score is "${esc(VOCAB[w])}": z = ${n3(TR.logits[f][w])}, probability <b>${(p[w] * 100).toFixed(1)}%</b>`); }
      }
    };
    const c = C[role] || C.norm;
    return `<div class="tc-h"><span class="wtag">${esc(name)}</span><b>${c.t}</b></div>
      <p>${c.plain}</p><p class="like">Think of it as: ${c.like}</p>
      <div class="tc-sec">The math</div>${c.math}
      <div class="tc-sec">Shape ${shp} (${fmt(n)} numbers, ${pct}% of the model)</div><p>${c.dims}</p>
      ${c.ex()}
      <p class="learn">These numbers never change while the model runs. Training set them, starting from random noise, by nudging each entry against its gradient: <span class="math-in"><i>W</i> ← <i>W</i> − η · ∂<i>L</i>/∂<i>W</i></span> (Adam adds per-weight step sizes).</p>`;
  }

  function place(el) {
    const r = el.getBoundingClientRect(), cw = Math.min(440, innerWidth - 16);
    card.style.width = cw + 'px';
    let left = Math.max(8, Math.min(r.left, innerWidth - cw - 8));
    card.style.left = left + 'px';
    const h = card.offsetHeight;
    const below = r.bottom + 8, above = r.top - h - 8;
    card.style.top = (below + h < innerHeight - 8 || above < 8 ? Math.min(below, Math.max(8, innerHeight - h - 8)) : above) + 'px';
  }
  function show(el) {
    const name = resolve(el); if (!name) return;
    const html = body(name); if (!html) return;
    curEl = el; card.innerHTML = html; card.style.display = 'block'; place(el);
    $('tip').style.display = 'none';
  }
  const hide = () => { card.style.display = 'none'; curEl = null; };
  const SEL = '.wtag, .slot, .gn.param, .wref, .lv.wt .nm, [data-tensor]';
  document.addEventListener('pointerover', e => {
    if (e.pointerType === 'touch') return;
    if (e.target.closest('#tcard')) { clearTimeout(hideT); return; }
    const el0 = e.target.closest(SEL), el = el0 && !el0.closest('.tguide, #ftable') ? el0 : null;
    clearTimeout(showT);
    if (el) { clearTimeout(hideT); if (el !== curEl) showT = setTimeout(() => show(el), 260); }
    else if (curEl) { clearTimeout(hideT); hideT = setTimeout(hide, 220); }
  });
  document.addEventListener('focusin', e => { const el = e.target.closest(SEL); if (el && el.matches(':focus-visible')) show(el); });
  document.addEventListener('focusout', () => { hideT = setTimeout(hide, 150); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
  document.addEventListener('pointerdown', e => {
    if (e.pointerType !== 'touch') return;
    if (e.target.closest('#tcard')) return;
    const el = e.target.closest('.wtag, .wref, .lv.wt .nm');
    if (el) { el === curEl ? hide() : show(el); } else hide();
  });
  return { body, hide };
})();

/* ================= boot ================= */
function render() {
  if (!TR) return;
  for (const k in TIPS) delete TIPS[k];
  ({ workbench: () => setStep(S.step, false), training: () => { TRN.renderCenter(); TRN.code.render(); }, overview: renderOverview, anatomy: renderAnatomy, attention: renderAttention, generation: renderGen, surgery: renderSurg, scale: renderScale })[S.section]();
}
(async () => {
  const m = store.get('mo-model');
  useModel(MODELS[m] ? m : 'words'); $('modelSel').value = S.model;
  S.ids = encode(DEFAULT_PROMPT[S.model]);
  buildPalette(); fillNoiseSelect();
  WB.graph.build(); TRN.graph.build(); TRN.reset();
  TR = Core.forward(W0, CFG, S.ids);
  renderChips(); WB.graph.labels(); WB.code.render(); TRN.code.render();
  const start = store.get('mo-section');
  go(SECTIONS.some(s => s[0] === start) ? start : 'workbench');
  await detectServer();
  WB.code.render(); TRN.code.render();
  if (ENGINE === 'python') { TR = await getTrace(S.ids); render(); }
})();
})();
