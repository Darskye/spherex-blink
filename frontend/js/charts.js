// Spectrum and light-curve charts (plain SVG, no dependencies).
// Colours come from a palette validated for colour-vision deficiency on the dark
// chart surface; text always uses ink tokens, never the series colour.
// Developed with AI assistance (Claude, Anthropic). See README "Use of AI".

import { prettyDate, monthYear, mjdToIso } from "./sky.js";

const SVG = "http://www.w3.org/2000/svg";
const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"];
// One-hue ramp for ordered epochs when there are more than three (earliest = darkest).
const RAMP = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"];
const INK = { muted: "#7f8797", grid: "#1c212b", axis: "#2c3340", surface: "#0f1218", ink2: "#b6bcc8" };
const W = 560, H = 300, M = { l: 54, r: 14, t: 20, b: 40 };

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, content, attrs) {
  const node = el("text", attrs, parent);
  node.textContent = content;
  return node;
}

function niceTicks(lo, hi, target = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return ticks;
}

function formatFlux(v) {
  const a = Math.abs(v);
  if (a >= 1000) return (v / 1000).toFixed(a >= 10000 ? 0 : 1) + " Jy";
  if (a >= 100) return v.toFixed(0) + " mJy";
  if (a >= 10) return v.toFixed(1) + " mJy";
  if (a >= 1) return v.toFixed(2) + " mJy";
  return v.toFixed(3) + " mJy";
}

function tickFlux(v, step) {
  if (Math.abs(v) >= 1000) return (v / 1000) + "K";
  const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3;
  return v.toFixed(digits);
}

export function epochColors(epochs) {
  const sorted = [...epochs].sort((a, b) => a - b);
  const map = new Map();
  if (sorted.length <= 3) {
    sorted.forEach((e, i) => map.set(e, SERIES[i]));
  } else {
    sorted.forEach((e, i) => map.set(e, RAMP[Math.round((i / (sorted.length - 1)) * (RAMP.length - 1))]));
  }
  return map;
}

function legendKey(parent, label, color, shape) {
  const key = document.createElement("span");
  key.className = "key";
  const swatch = document.createElement("span");
  swatch.className = "swatch" + (shape ? " " + shape : "");
  swatch.style.background = color;
  const name = document.createElement("span");
  name.textContent = label;
  key.append(swatch, name);
  parent.appendChild(key);
}

function tooltipEl() { return document.getElementById("tooltip"); }

function showTooltip(build, clientX, clientY) {
  const tip = tooltipEl();
  tip.replaceChildren();
  build(tip);
  tip.hidden = false;
  const rect = tip.getBoundingClientRect();
  let x = clientX + 14, y = clientY + 14;
  if (x + rect.width > window.innerWidth - 8) x = clientX - rect.width - 14;
  if (y + rect.height > window.innerHeight - 8) y = clientY - rect.height - 14;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}

function hideTooltip() { tooltipEl().hidden = true; }

function line(tip, cls, content) {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = content;
  tip.appendChild(div);
  return div;
}

function emptyPlot(container, message) {
  container.replaceChildren();
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": message }, container);
  // Wrap long messages onto several centred lines.
  const words = message.split(" ");
  const lines = [];
  let current = "";
  for (const word of words) {
    if ((current + " " + word).trim().length > 52) { lines.push(current); current = word; }
    else current = (current + " " + word).trim();
  }
  if (current) lines.push(current);
  lines.forEach((content, i) => {
    text(svg, content, { x: W / 2, y: H / 2 + (i - (lines.length - 1) / 2) * 20, "text-anchor": "middle", class: "empty" });
  });
}

/** Replace a chart with a plain statement (used when the data cannot support a chart). */
export function renderMessage(container, legend, message) {
  legend.replaceChildren();
  emptyPlot(container, message);
}

// ----------------------------------------------------------------------------- spectrum

/**
 * Scatter of flux against wavelength. Colour = survey pass (epoch).
 * points: [{wl, flux, err, epoch, detector, date, flagged, flag_names, snr}]
 */
export function renderSpectrum(container, legend, result, detectors) {
  legend.replaceChildren();
  const points = result.points.filter((p) => Number.isFinite(p.wl) && Number.isFinite(p.flux));
  if (!points.length) { emptyPlot(container, "No measurements at this position yet"); return; }

  const good = points.filter((p) => !p.flagged);
  const basis = good.length >= 5 ? good : points;
  const values = basis.map((p) => p.flux).sort((a, b) => a - b);
  const q = (f) => values[Math.min(values.length - 1, Math.max(0, Math.round(f * (values.length - 1))))];
  const typicalErr = basis.map((p) => p.err).sort((a, b) => a - b)[Math.floor(basis.length / 2)] || 0;
  let yMax = Math.max(q(0.99), q(0.5) + 4 * typicalErr, 1e-6);
  let yMin = Math.min(0, q(0.01));
  const padY = (yMax - yMin) * 0.08;
  yMax += padY; yMin -= yMin < 0 ? padY : 0;
  const xMin = 0.7, xMax = 5.1;
  const x = (v) => M.l + ((v - xMin) / (xMax - xMin)) * (W - M.l - M.r);
  const y = (v) => H - M.b - ((Math.min(Math.max(v, yMin), yMax) - yMin) / (yMax - yMin)) * (H - M.t - M.b);

  const epochs = [...new Set(points.map((p) => p.epoch))].sort((a, b) => a - b);
  const colors = epochColors(epochs);

  container.replaceChildren();
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Spectrum: brightness against wavelength" }, container);

  // Detector bands as faint alternating columns with labels.
  Object.entries(detectors).forEach(([det, range], i) => {
    const x0 = x(range.min_um), x1 = x(range.max_um);
    if (i % 2 === 0) el("rect", { x: x0, y: M.t, width: x1 - x0, height: H - M.t - M.b, fill: "#ffffff", opacity: 0.03 }, svg);
    text(svg, `Band ${det}`, { x: (x0 + x1) / 2, y: M.t - 6, "text-anchor": "middle", class: "band-label" });
  });

  const yTicks = niceTicks(yMin, yMax, 5);
  const yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1;
  for (const t of yTicks) {
    el("line", { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), stroke: t === 0 ? INK.axis : INK.grid, "stroke-width": 1 }, svg);
    text(svg, tickFlux(t, yStep), { x: M.l - 8, y: y(t) + 4, "text-anchor": "end" });
  }
  for (let t = 1; t <= 5; t += 0.5) {
    el("line", { x1: x(t), x2: x(t), y1: H - M.b, y2: H - M.b + 4, stroke: INK.axis, "stroke-width": 1 }, svg);
    if (Number.isInteger(t)) text(svg, t.toFixed(0), { x: x(t), y: H - M.b + 17, "text-anchor": "middle" });
  }
  el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: INK.axis, "stroke-width": 1 }, svg);
  text(svg, "Wavelength (µm)", { x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" });
  text(svg, "Brightness (mJy)", { x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})` });

  // Marks: error bars first, then dots. Flagged points are hollow.
  const screen = [];
  const marks = el("g", {}, svg);
  for (const p of points) {
    const cx = x(p.wl), cy = y(p.flux);
    const color = colors.get(p.epoch);
    if (!p.flagged && p.err > 0) {
      el("line", { x1: cx, x2: cx, y1: y(p.flux - p.err), y2: y(p.flux + p.err), stroke: color, "stroke-width": 1, opacity: 0.45 }, marks);
    }
    screen.push({ p, cx, cy, color });
  }
  for (const s of screen) {
    if (s.p.flagged) {
      el("circle", { cx: s.cx, cy: s.cy, r: 3, fill: "none", stroke: INK.muted, "stroke-width": 1.2, opacity: 0.8 }, marks);
    } else if (s.p.blended) {
      // Open ring in the pass colour: a real measurement, but a neighbour's light is mixed in.
      el("circle", { cx: s.cx, cy: s.cy, r: 3.2, fill: INK.surface, stroke: s.color, "stroke-width": 1.6 }, marks);
    } else {
      el("circle", { cx: s.cx, cy: s.cy, r: 3.6, fill: s.color, stroke: INK.surface, "stroke-width": 1 }, marks);
    }
  }

  // Hover: the nearest point wins, so the pointer never has to hit a dot exactly.
  const focus = el("circle", { r: 7, fill: "none", stroke: "#f0efec", "stroke-width": 1.5, visibility: "hidden" }, svg);
  const hit = el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const py = ((ev.clientY - rect.top) / rect.height) * H;
    let best = null, bestD = 40 * 40;
    for (const s of screen) {
      const d = (s.cx - px) ** 2 + (s.cy - py) ** 2;
      if (d < bestD) { bestD = d; best = s; }
    }
    if (!best) { focus.setAttribute("visibility", "hidden"); hideTooltip(); return; }
    focus.setAttribute("cx", best.cx); focus.setAttribute("cy", best.cy); focus.setAttribute("visibility", "visible");
    const p = best.p;
    showTooltip((tip) => {
      line(tip, "value", `${formatFlux(p.flux)} ± ${formatFlux(p.err).replace(" mJy", "").replace(" Jy", "")}`);
      line(tip, "", `${p.wl.toFixed(3)} µm · Band ${p.detector}`);
      line(tip, "", `${prettyDate(p.date)} · signal-to-noise ${p.snr.toFixed(1)}`);
      if (p.flagged || (p.flag_names && p.flag_names.length)) {
        line(tip, "tflag", (p.flagged ? "Unreliable: " : "Flags: ") + (p.flag_names.length ? p.flag_names.join(", ") : "core pixels flagged"));
      }
      if (p.blended) {
        line(tip, "tflag", `Blended: a known star${p.blend_g != null ? ` (Gaia G ${p.blend_g.toFixed(1)})` : ""} is ${p.blend_sep.toFixed(1)}″ away`);
      }
    }, ev.clientX, ev.clientY);
  });
  hit.addEventListener("pointerleave", () => { focus.setAttribute("visibility", "hidden"); hideTooltip(); });

  // Legend: always present for two or more passes.
  const dateOf = new Map();
  for (const p of points) if (!dateOf.has(p.epoch)) dateOf.set(p.epoch, p.date);
  if (epochs.length <= 6) {
    for (const e of epochs) legendKey(legend, `Pass ${e + 1} · ${monthYear(dateOf.get(e))}`, colors.get(e));
  } else {
    legendKey(legend, `Earliest pass · ${monthYear(dateOf.get(epochs[0]))}`, colors.get(epochs[0]));
    legendKey(legend, `Latest pass · ${monthYear(dateOf.get(epochs[epochs.length - 1]))} (${epochs.length} passes, darker = earlier)`, colors.get(epochs[epochs.length - 1]));
  }
  if (points.some((p) => p.blended && !p.flagged)) legendKey(legend, "Open ring: blended with a neighbouring star", "transparent", "hollow ring");
  if (points.some((p) => p.flagged)) legendKey(legend, "Grey ring: flagged, unreliable", "transparent", "hollow");
}

// ----------------------------------------------------------------------------- light curve

/**
 * Lines of band-average brightness against date, one line per detector band,
 * each relative to its own average (so all six share one axis).
 */
export function renderLightCurve(container, legend, result, detectors) {
  legend.replaceChildren();
  const series = [];
  for (const det of Object.keys(detectors).map(Number)) {
    const pts = result.light_curve.filter((p) => p.detector === det && Number.isFinite(p.rel)).sort((a, b) => a.mjd - b.mjd);
    if (pts.length < 2) continue;
    series.push({ det, method: pts[0].method, pts: pts.map((p) => ({ ...p, relErr: p.rel_err })) });
  }
  if (!series.length) {
    emptyPlot(container, "Not enough clear measurements in two or more passes");
    return;
  }

  const all = series.flatMap((s) => s.pts);
  const tMin = Math.min(...all.map((p) => p.mjd)), tMax = Math.max(...all.map((p) => p.mjd));
  const padT = Math.max(10, (tMax - tMin) * 0.06);
  const xMin = tMin - padT, xMax = tMax + padT;
  let yMin = Math.min(0.9, ...all.map((p) => p.rel - p.relErr));
  let yMax = Math.max(1.1, ...all.map((p) => p.rel + p.relErr));
  const padY = (yMax - yMin) * 0.08;
  yMin = Math.max(0, yMin - padY); yMax += padY;
  const x = (v) => M.l + ((v - xMin) / (xMax - xMin)) * (W - M.l - M.r);
  const y = (v) => H - M.b - ((Math.min(Math.max(v, yMin), yMax) - yMin) / (yMax - yMin)) * (H - M.t - M.b);

  container.replaceChildren();
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Light curve: relative brightness against date for each band" }, container);

  for (const t of niceTicks(yMin, yMax, 5)) {
    const isBase = Math.abs(t - 1) < 1e-9;
    el("line", { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), stroke: isBase ? INK.axis : INK.grid, "stroke-width": 1 }, svg);
    text(svg, `${Math.round(t * 100)}%`, { x: M.l - 8, y: y(t) + 4, "text-anchor": "end" });
  }
  // Month ticks along the date axis.
  const first = new Date((xMin - 40587) * 86400000), last = new Date((xMax - 40587) * 86400000);
  const months = (last.getUTCFullYear() - first.getUTCFullYear()) * 12 + last.getUTCMonth() - first.getUTCMonth();
  const every = Math.max(1, Math.ceil(months / 6));
  const cursor = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 1));
  for (let i = 0; cursor <= last; i++) {
    const mjd = cursor.getTime() / 86400000 + 40587;
    el("line", { x1: x(mjd), x2: x(mjd), y1: H - M.b, y2: H - M.b + 4, stroke: INK.axis, "stroke-width": 1 }, svg);
    if (i % every === 0) text(svg, monthYear(cursor.toISOString().slice(0, 10)), { x: x(mjd), y: H - M.b + 17, "text-anchor": "middle" });
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
  }
  el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: INK.axis, "stroke-width": 1 }, svg);
  text(svg, "Date", { x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-title" });
  text(svg, "Brightness vs. band average", { x: 14, y: (M.t + H - M.b) / 2, "text-anchor": "middle", class: "axis-title", transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})` });

  const crosshair = el("line", { y1: M.t, y2: H - M.b, stroke: INK.muted, "stroke-width": 1, visibility: "hidden" }, svg);
  for (const s of series) {
    const color = SERIES[s.det - 1];
    const d = s.pts.map((p, i) => `${i ? "L" : "M"}${x(p.mjd).toFixed(1)},${y(p.rel).toFixed(1)}`).join(" ");
    el("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    for (const p of s.pts) {
      el("line", { x1: x(p.mjd), x2: x(p.mjd), y1: y(p.rel - p.relErr), y2: y(p.rel + p.relErr), stroke: color, "stroke-width": 1, opacity: 0.6 }, svg);
      el("circle", { cx: x(p.mjd), cy: y(p.rel), r: 4.5, stroke: INK.surface, "stroke-width": 2, fill: color }, svg);
    }
    const range = detectors[String(s.det)];
    legendKey(legend, `Band ${s.det} · ${range.min_um.toFixed(2)}–${range.max_um.toFixed(2)} µm`, color, "line");
  }

  // Hover: a crosshair snaps to the nearest pass and lists every band there.
  const epochTimes = [...new Set(all.map((p) => p.epoch))].map((e) => {
    const members = all.filter((p) => p.epoch === e);
    return { epoch: e, mjd: members.reduce((a, p) => a + p.mjd, 0) / members.length };
  });
  const hit = el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    let best = epochTimes[0];
    for (const e of epochTimes) if (Math.abs(x(e.mjd) - px) < Math.abs(x(best.mjd) - px)) best = e;
    crosshair.setAttribute("x1", x(best.mjd)); crosshair.setAttribute("x2", x(best.mjd)); crosshair.setAttribute("visibility", "visible");
    showTooltip((tip) => {
      line(tip, "", `Pass ${best.epoch + 1} · ${monthYear(mjdToIso(best.mjd))}`);
      for (const s of series) {
        const p = s.pts.find((q) => q.epoch === best.epoch);
        if (!p) continue;
        const row = document.createElement("div");
        row.className = "trow";
        const left = document.createElement("span");
        const key = document.createElement("span");
        key.className = "tkey"; key.style.background = SERIES[s.det - 1];
        left.append(key, document.createTextNode(`Band ${s.det}`));
        const right = document.createElement("span");
        right.className = "value";
        const pct = (p.rel - 1) * 100;
        const digits = Math.abs(pct) < 10 ? 1 : 0;
        right.textContent = `${pct >= 0 ? "+" : "−"}${Math.abs(pct).toFixed(digits)}% ± ${(p.relErr * 100).toFixed(digits)}%` +
          (p.flux != null ? ` · ${formatFlux(p.flux)}` : "");
        row.append(left, right);
        tip.appendChild(row);
      }
    }, ev.clientX, ev.clientY);
  });
  hit.addEventListener("pointerleave", () => { crosshair.setAttribute("visibility", "hidden"); hideTooltip(); });
}

// ----------------------------------------------------------------------------- table + CSV

export function renderTable(container, result) {
  container.replaceChildren();
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const h of ["Date", "Band", "Wavelength (µm)", "Brightness (mJy)", "Uncertainty (mJy)", "Signal-to-noise", "Flags"]) {
    const th = document.createElement("th");
    th.textContent = h;
    head.appendChild(th);
  }
  const body = table.createTBody();
  for (const p of [...result.points].sort((a, b) => a.mjd - b.mjd || a.wl - b.wl)) {
    const row = body.insertRow();
    const cells = [p.date, p.detector, p.wl.toFixed(4), p.flux.toFixed(4), p.err.toFixed(4), p.snr.toFixed(1),
      (p.flagged ? "UNRELIABLE " : "") + (p.flag_names || []).join(" ")];
    for (const c of cells) row.insertCell().textContent = String(c);
  }
  container.appendChild(table);
}

export function toCsv(result) {
  const rows = [["mjd", "date", "detector", "wavelength_um", "bandwidth_um", "flux_mJy", "flux_err_mJy", "snr", "unreliable", "flags", "obs_id"]];
  for (const p of [...result.points].sort((a, b) => a.mjd - b.mjd || a.wl - b.wl)) {
    rows.push([p.mjd.toFixed(6), p.date, p.detector, p.wl.toFixed(5), p.bw == null ? "" : p.bw.toFixed(5), p.flux.toFixed(5),
      p.err.toFixed(5), p.snr.toFixed(2), p.flagged ? 1 : 0, (p.flag_names || []).join("|"), p.obs_id]);
  }
  return rows.map((r) => r.join(",")).join("\n") + "\n";
}

export { formatFlux };
