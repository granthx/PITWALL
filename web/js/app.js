/* PIT WALL console — vanilla JS, no build step.
   Fetches the local API, draws SVG charts client-side, turns the wheel. */
"use strict";

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

const DEG = 25;                 // sim_max_angle_deg
const WHEEL_GAIN = 3.1;         // visual: wheel turns further than the lock
const state = {
  status: null,
  samples: [],
  current: null,               // {path} of last predicted frame
  salOn: false,
  last: null,                  // last /api/predict payload
  runs: [],
  run: null,
};

/* ───────────────────────── helpers ───────────────────────── */

function fmt(x, d = 2) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d); }

const tip = document.createElement("div");
tip.className = "curve-tip";
document.body.appendChild(tip);
function showTip(x, y, html) {
  tip.innerHTML = html;
  tip.style.display = "block";
  const w = tip.offsetWidth;
  tip.style.left = Math.min(window.innerWidth - w - 12, x + 14) + "px";
  tip.style.top = (y + 14) + "px";
}
function hideTip() { tip.style.display = "none"; }

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

/* ───────────────────────── views ───────────────────────── */

$$(".view-btn").forEach(b => b.addEventListener("click", () => {
  $$(".view-btn").forEach(x => x.classList.toggle("active", x === b));
  $$(".view").forEach(v => v.classList.remove("active"));
  $("#view-" + b.dataset.view).classList.add("active");
  hideTip();
}));

/* ───────────────────────── status / model ───────────────────────── */

async function loadStatus() {
  try {
    const s = await jget("/api/status");
    state.status = s;
    $("#chip-device").textContent = s.device;
    $("#chip-params").textContent = s.model.params.toLocaleString() + " params";
    $("#sub-model").textContent = s.model.path.split("/").pop() + ` (${s.model.arch})`;
    $("#sub-data").textContent = `${s.data.dir} · ${s.data.rows} rows`;
    $("#m-ckpt").textContent = s.model.path;
    $("#m-input").textContent = s.model.input;
    $("#m-params").textContent = s.model.params.toLocaleString();
    $("#m-mse").textContent = s.model.val_mse != null ? s.model.val_mse.toFixed(5) : "—";
    $("#m-device").textContent = s.device;
    $("#model-arch").textContent = s.model.arch;
    if (s.model.epoch) $("#m-mse").title = "checkpoint epoch " + s.model.epoch;
    if (s.eval) {
      $("#m-mae").textContent = fmt(s.eval.mae_deg, 2) + "°";
      renderEval(s.eval);
    } else {
      $("#m-mae").textContent = "—";
    }
  } catch (e) {
    console.error("status failed", e);
    $("#chip-device").textContent = "api down";
  }
}

function renderEval(ev) {
  const rows = [
    ["frames evaluated", ev.n],
    ["MAE", `${ev.mae.toFixed(4)} <em style="color:var(--teal)">${fmt(ev.mae_deg, 2)}°</em>`],
    ["RMSE", `${ev.rmse.toFixed(4)} (${fmt(ev.rmse_deg, 2)}°)`],
    ["max |error|", `${ev.max_abs_err.toFixed(3)} (${fmt(ev.max_abs_err_deg, 1)}°)`],
    ["bias", `${ev.bias.toFixed(4)} (${fmt(ev.bias_deg, 2)}°)`],
    ["ckpt val mse", ev.val_mse_ckpt != null ? ev.val_mse_ckpt.toFixed(5) : "—"],
  ];
  $("#eval-meta").textContent = ev.data;
  $("#eval-table").innerHTML =
    `<table class="mtable">` +
    rows.map(r => `<tr><td>${esc(r[0])}</td><td>${r[1]}</td></tr>`).join("") +
    `</table>`;
}

/* ───────────────────────── steering wheel + tape ───────────────────────── */

function buildWheel() {
  const g = $("#wheel-ticks");
  const NS = "http://www.w3.org/2000/svg";
  const cx = 130, cy = 130, r1 = 108, r2 = 118;
  for (let d = -DEG; d <= DEG; d += 5) {
    const a = (d - 90) * Math.PI / 180;
    const maj = d % 25 === 0;
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", cx + Math.cos(a) * r1); l.setAttribute("y1", cy + Math.sin(a) * r1);
    l.setAttribute("x2", cx + Math.cos(a) * (maj ? r2 + 3 : r2));
    l.setAttribute("y2", cy + Math.sin(a) * (maj ? r2 + 3 : r2));
    if (maj) l.classList.add("maj");
    g.appendChild(l);
  }
  const t1 = document.createElementNS(NS, "text");
  t1.setAttribute("x", cx - 121); t1.setAttribute("y", cy + 22); t1.textContent = "−25°";
  const t2 = document.createElementNS(NS, "text");
  t2.setAttribute("x", cx + 121); t2.setAttribute("y", cy + 22); t2.textContent = "+25°";
  g.appendChild(t1); g.appendChild(t2);

  const tape = $("#tape-ticks");
  for (let d = -DEG; d <= DEG; d += 5) {
    const i = document.createElement("i");
    i.style.left = (50 + (d / DEG) * 47) + "%";
    if (d === 0) i.className = "zero";
    if (Math.abs(d) === DEG) i.className = "lim";
    tape.appendChild(i);
  }
  for (const d of [-25, 0, 25]) {
    const em = document.createElement("em");
    em.style.left = (50 + (d / DEG) * 47) + "%";
    em.textContent = (d > 0 ? "+" : "") + d + "°";
    tape.appendChild(em);
  }
}

function renderSteer(p) {
  const deg = p.deg;
  $("#deg-big").textContent = fmt(deg, 1);
  $("#wheel-rot").style.transform = `rotate(${(deg * WHEEL_GAIN).toFixed(2)}deg)`;
  $("#tape-marker").style.left = (50 + (deg / DEG) * 47) + "%";
  $("#val-norm").textContent = fmt(p.steer_norm, 3);
  $("#val-throttle").textContent = (p.throttle >= 0 ? "+" : "−") + Math.abs(p.throttle).toFixed(2);
  $("#val-lat").textContent = p.infer_ms.toFixed(1) + " ms";

  const hu = $("#val-human"), v = $("#verdict");
  if (p.human_deg == null) {
    hu.textContent = "—";
    hu.className = "v dim";
    v.className = "verdict";
    v.textContent = "uploaded frame · no human label to compare";
  } else {
    hu.textContent = fmt(p.human_deg, 1) + "°";
    hu.className = "v";
    const diff = Math.abs(p.deg - p.human_deg);
    if (diff < 3) { v.className = "verdict good"; v.textContent = `Δ ${diff.toFixed(1)}° vs human · inside tolerance`; }
    else if (diff < 8) { v.className = "verdict warn"; v.textContent = `Δ ${diff.toFixed(1)}° vs human · getting sloppy`; }
    else { v.className = "verdict bad"; v.textContent = `Δ ${diff.toFixed(1)}° vs human · would leave the line`; }
  }
}

/* ───────────────────────── predict ───────────────────────── */

function setBusy(b) { $("#stage").classList.toggle("busy", b); }

async function predict(payload, label) {
  setBusy(true);
  try {
    const p = await fetch("/api/predict", { method: "POST", ...payload })
      .then(r => { if (!r.ok) throw new Error("predict " + r.status); return r.json(); });
    state.last = p;
    $("#frame-img").src = state.salOn ? p.saliency : p.input;
    $("#stage").dataset.mode = state.salOn ? "saliency" : "raw";
    $("#stage-mode").textContent = state.salOn ? "saliency overlay" : "raw frame";
    $("#stage-lat").textContent = p.infer_ms.toFixed(1) + " ms";
    $("#frame-meta").textContent = label;
    renderSteer(p);
    renderMaps(p.maps);
  } catch (e) {
    console.error(e);
    $("#frame-meta").textContent = "prediction failed — is the model loaded?";
  } finally {
    setBusy(false);
  }
}

function predictPath(path, name) {
  state.current = path;
  const body = new URLSearchParams({ path });
  predict({ body, headers: { "Content-Type": "application/x-www-form-urlencoded" } },
          `${name || path} · y=${state.samples.find(s => s.path === path)?.angle ?? "?"}`);
}

function renderMaps(maps) {
  $("#maps-strip").innerHTML = maps.map(m => `
    <div class="maps-cell">
      <img src="${m.mean}" alt="conv${m.layer} mean activation">
      <div class="maps-cap"><span>conv${m.layer}</span><b>${esc(m.role)}</b></div>
      <div class="maps-chan">${m.chans.map(c => `<img src="${c}" alt="channel">`).join("")}</div>
    </div>`).join("");
}

function toggleSal() {
  state.salOn = !state.salOn;
  const b = $("#btn-sal");
  b.classList.toggle("on", state.salOn);
  b.innerHTML = `saliency <b>${state.salOn ? "on" : "off"}</b>`;
  if (state.last) {
    $("#frame-img").src = state.salOn ? state.last.saliency : state.last.input;
    $("#stage").dataset.mode = state.salOn ? "saliency" : "raw";
    $("#stage-mode").textContent = state.salOn ? "saliency overlay" : "raw frame";
  }
}

/* ───────────────────────── samples / filmstrip ───────────────────────── */

function filmCell(s, sel) {
  const d = document.createElement("div");
  d.className = "fcell" + (sel ? " sel" : "");
  d.innerHTML = `
    <img loading="lazy" src="/api/frame?path=${encodeURIComponent(s.path)}" alt="frame">
    <div class="fbar"><i style="left:${50 + (s.angle / 1) * 47}%"></i></div>
    <span class="fname">#${s.i} · ${fmt(s.angle * DEG, 1)}°</span>`;
  d.addEventListener("click", () => {
    $$(".fcell").forEach(c => c.classList.remove("sel"));
    d.classList.add("sel");
    predictPath(s.path);
  });
  return d;
}

async function loadSamples() {
  try {
    state.samples = await jget("/api/samples?k=14");
    const fs = $("#filmstrip");
    fs.innerHTML = "";
    state.samples.forEach(s => fs.appendChild(filmCell(s)));
    // start on a frame with an actual turn in it
    const pick = state.samples.find(s => Math.abs(s.angle) > 0.3) || state.samples[0];
    if (pick) predictPath(pick.path);
  } catch (e) { console.error("samples failed", e); }
}

/* upload + drag-drop */
function wireUpload() {
  const fi = $("#file-input");
  $("#btn-upload").addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => { if (fi.files[0]) uploadFile(fi.files[0]); fi.value = ""; });
  const stage = $("#stage");
  ["dragenter", "dragover"].forEach(ev => stage.addEventListener(ev, e => {
    e.preventDefault(); stage.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach(ev => stage.addEventListener(ev, e => {
    e.preventDefault(); if (ev === "drop" || !stage.contains(e.relatedTarget)) stage.classList.remove("dragging");
  }));
  stage.addEventListener("drop", e => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) uploadFile(f);
  });
  stage.addEventListener("click", () => fi.click());
  $("#btn-sal").addEventListener("click", toggleSal);
  $("#btn-random").addEventListener("click", () => {
    if (!state.samples.length) return;
    const s = state.samples[Math.floor(Math.random() * state.samples.length)];
    predictPath(s.path);
  });
}

function uploadFile(f) {
  const fd = new FormData();
  fd.append("file", f);
  predict({ body: fd }, f.name + " · upload");
}

/* ───────────────────────── data view ───────────────────────── */

async function loadHist() {
  try {
    const h = await jget("/api/histogram");
    $("#data-stats").innerHTML = [
      ["raw rows", h.n_raw.toLocaleString()],
      ["after balancing", h.n_balanced.toLocaleString(), "hot"],
      ["near-zero share", (h.zero_share * 100).toFixed(1) + "%"],
      ["cameras / row", "×3"],
      ["side-camera correction", "±" + (state.status?.data.correction ?? 0.2)],
    ].map(s => `<div class="stat ${s[2] || ""}"><span class="k">${s[0]}</span><span class="v">${s[1]}</span></div>`).join("");
    drawHist(h);
  } catch (e) { console.error("hist failed", e); }
}

function drawHist(h) {
  const W = 980, H = 250, pad = { l: 16, r: 16, t: 14, b: 26 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const n = h.raw.length;
  const max = Math.max(...h.raw, ...h.balanced, 1);
  const bw = iw / n;
  let bars = "";
  for (let i = 0; i < n; i++) {
    const x = pad.l + i * bw;
    const hr = (h.raw[i] / max) * ih, hb = (h.balanced[i] / max) * ih;
    const cx0 = h.edges[i], cx1 = h.edges[i + 1];
    const mid = ((cx0 + cx1) / 2 * DEG).toFixed(1);
    bars += `
      <g class="hgroup" data-raw="${h.raw[i]}" data-bal="${h.balanced[i]}" data-lo="${cx0.toFixed(2)}" data-hi="${cx1.toFixed(2)}">
        <rect x="${(x + 1).toFixed(1)}" y="${(pad.t + ih - hr).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${hr.toFixed(1)}"
              fill="rgba(110,168,255,.28)" stroke="rgba(110,168,255,.6)" stroke-width=".6"/>
        <rect x="${(x + 1).toFixed(1)}" y="${(pad.t + ih - hb).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${hb.toFixed(1)}"
              fill="rgba(62,230,193,.42)" stroke="rgba(62,230,193,.85)" stroke-width=".6"/>
        <rect class="hhit" x="${x.toFixed(1)}" y="${pad.t}" width="${bw.toFixed(1)}" height="${ih}" fill="transparent"/>
      </g>`;
    if (i % 5 === 0) {
      bars += `<line x1="${x.toFixed(1)}" y1="${pad.t + ih}" x2="${x.toFixed(1)}" y2="${pad.t + ih + 4}"
                stroke="rgba(126,166,214,.4)" stroke-width="1"/>
               <text x="${x.toFixed(1)}" y="${pad.t + ih + 15}" fill="#4A5F78" font-size="9"
                font-family="IBM Plex Mono,monospace" text-anchor="middle">${mid}°</text>`;
    }
  }
  const axis = `
    <line x1="${pad.l}" y1="${pad.t + ih}" x2="${W - pad.r}" y2="${pad.t + ih}" stroke="rgba(126,166,214,.35)" stroke-width="1"/>`;
  $("#hist").innerHTML = `<svg viewBox="0 0 ${W} ${H}">${bars}${axis}</svg>`;
  $$("#hist .hgroup").forEach(g => {
    g.addEventListener("mousemove", e => showTip(e.clientX, e.clientY,
      `<b>${(+g.dataset.lo * DEG).toFixed(1)}° … ${(+g.dataset.hi * DEG).toFixed(1)}°</b><br>` +
      `raw ${g.dataset.raw} · kept ${g.dataset.bal}`));
    g.addEventListener("mouseleave", hideTip);
  });
}

/* ───────────────────────── training view ───────────────────────── */

async function loadRuns() {
  try {
    state.runs = await jget("/api/runs");
    const list = $("#runs-list");
    list.innerHTML = state.runs.length ? "" :
      `<div class="maps-empty">no runs — train something first</div>`;
    state.runs.forEach((r, i) => {
      const b = document.createElement("button");
      b.className = "run-item";
      b.innerHTML = `<span class="rid">${esc(r.id)}</span>
        <span class="rmeta">${r.epochs} epochs · best val mse ${r.best_val_mse.toFixed(5)}</span>`;
      b.addEventListener("click", () => selectRun(r.id, b));
      list.appendChild(b);
      if (i === 0) selectRun(r.id, b);
    });
  } catch (e) { console.error("runs failed", e); }
}

async function selectRun(id, btn) {
  $$(".run-item").forEach(x => x.classList.toggle("sel", x === btn));
  state.run = id;
  $("#run-meta").textContent = id;
  try {
    const r = await jget("/api/runs/" + encodeURIComponent(id));
    if (r.history) drawCurve(r.history);
    if (r.eval) renderEval(r.eval);
  } catch (e) { console.error(e); }
}

function drawCurve(hist) {
  const W = 760, H = 330, pad = { l: 52, r: 18, t: 16, b: 30 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const xs = hist.map(h => h.epoch), maxY = Math.max(...hist.map(h => h.train_mse), ...hist.map(h => h.val_mse));
  const X = e => pad.l + (e - xs[0]) / Math.max(1, xs[xs.length - 1] - xs[0]) * iw;
  const Y = v => pad.t + ih - (v / maxY) * ih;
  const path = key => hist.map((h, i) => `${i ? "L" : "M"}${X(h.epoch).toFixed(1)} ${Y(h[key]).toFixed(1)}`).join(" ");
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = maxY * i / 4, y = Y(v).toFixed(1);
    grid += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="rgba(126,166,214,.12)"/>
             <text x="${pad.l - 8}" y="${y}" fill="#4A5F78" font-size="9" font-family="IBM Plex Mono,monospace"
              text-anchor="end" dominant-baseline="middle">${v.toFixed(3)}</text>`;
  }
  let xt = "";
  const step = Math.ceil(xs.length / 10);
  xs.filter(e => e % step === 0 || e === 1).forEach(e => {
    xt += `<text x="${X(e).toFixed(1)}" y="${H - 8}" fill="#4A5F78" font-size="9"
            font-family="IBM Plex Mono,monospace" text-anchor="middle">${e}</text>`;
  });
  const marks = hist.map(h =>
    `<circle cx="${X(h.epoch).toFixed(1)}" cy="${Y(h.val_mse).toFixed(1)}" r="2.6" fill="#F5A94B"/>` +
    `<circle cx="${X(h.epoch).toFixed(1)}" cy="${Y(h.train_mse).toFixed(1)}" r="2.6" fill="#3EE6C1"/>`).join("");
  $("#losscurve").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}">
      ${grid}${xt}
      <path d="${path("train_mse")}" fill="none" stroke="#3EE6C1" stroke-width="1.8"/>
      <path d="${path("val_mse")}" fill="none" stroke="#F5A94B" stroke-width="1.8" stroke-dasharray="5 4"/>
      ${marks}
      <text x="${pad.l}" y="${H - 8}" fill="#4A5F78" font-size="9" font-family="IBM Plex Mono,monospace">epoch →</text>
      <text x="${W - pad.r}" y="${pad.t + 2}" fill="#3EE6C1" font-size="9.5" font-family="IBM Plex Mono,monospace" text-anchor="end">— train mse</text>
      <text x="${W - pad.r}" y="${pad.t + 16}" fill="#F5A94B" font-size="9.5" font-family="IBM Plex Mono,monospace" text-anchor="end">-- val mse</text>
    </svg>
    <div class="hint maps-empty" style="padding:0 0 10px;text-align:left">hover the dots — val MAE shown in degrees</div>`;
  $$("#losscurve circle").forEach(c => {
    c.addEventListener("mousemove", e => {
      const h = hist[ Math.round(((+c.getAttribute("cx") - pad.l) / iw) * (hist.length - 1)) ];
      if (!h) return;
      showTip(e.clientX, e.clientY,
        `<b>epoch ${h.epoch}</b><br>train mse ${h.train_mse.toFixed(5)}<br>val mse ${h.val_mse.toFixed(5)}` +
        (h.val_mae_deg != null ? `<br>val MAE ≈ ${fmt(h.val_mae_deg, 2)}°` : ""));
    });
    c.addEventListener("mouseleave", hideTip);
  });
}

async function loadStory() {
  try {
    const r = await fetch("/api/story");
    if (!r.ok) throw new Error("no story");
    const blob = await r.blob();
    $("#story-img").src = URL.createObjectURL(blob);
    $("#story-img").hidden = false;
    $("#story-empty").style.display = "none";
  } catch (_) { /* stays hidden */ }
}

/* ───────────────────────── boot ───────────────────────── */

buildWheel();
wireUpload();
loadStatus();
loadSamples();
loadHist();
loadRuns();
loadStory();
