// Canvas image viewer: grey or false-colour frames, blink / difference / sum,
// zoom and pan, and sky overlays.
// Developed with AI assistance (Claude, Anthropic). See README "Use of AI".

import { pixToSky, skyToPix, propagate, formatCoords, monthYear, prettyDate, separationArcsec } from "./sky.js";
import { labelFor } from "./names.js";

const ASINH_A = 8;
const ASINH_NORM = Math.asinh(ASINH_A);
const OVERLAY = {
  gaia: "rgba(150, 190, 245, 0.85)",
  gaiaFast: "#a9cdfa",
  track: "#ffb84d",
  infrared: "rgba(255, 205, 140, 0.8)",
  name: "#eef1f6",
  kind: "#b6bcc8",
  select: "#ffffff",
  chrome: "rgba(238, 241, 246, 0.9)",
  candidate: { unexplained: "#ff7a3d", known: "#58c98a", doubtful: "rgba(182, 188, 200, 0.75)" },
  flag: [224, 72, 72],
};
const FAST_PM_MASYR = 150;      // stars moving faster than this get a motion arrow
const MAX_ZOOM = 8;

function median(sorted) {
  const n = sorted.length;
  if (!n) return 0;
  return n % 2 ? sorted[(n - 1) / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

/** Robust centre and scatter of a set of frames (sampled). */
function robustStats(frames) {
  let total = 0;
  for (const f of frames) total += f.length;
  const step = Math.max(1, Math.floor(total / 120000));
  const sample = [];
  for (const f of frames) {
    for (let i = 0; i < f.length; i += step) {
      const v = f[i];
      if (Number.isFinite(v)) sample.push(v);
    }
  }
  if (!sample.length) return { med: 0, sigma: 1 };
  const sorted = Float32Array.from(sample).sort();
  const med = median(sorted);
  const dev = Float32Array.from(sorted, (v) => Math.abs(v - med)).sort();
  const sigma = 1.4826 * median(dev);
  return { med, sigma: sigma > 0 ? sigma : Math.max(1e-6, Math.abs(med) * 0.01) };
}

export class Viewer {
  constructor({ image, overlay, stage, onSelect, onHover, onFrame, onView }) {
    this.image = image;
    this.overlay = overlay;
    this.stage = stage;
    this.ctx = image.getContext("2d");
    this.octx = overlay.getContext("2d");
    this.buffer = document.createElement("canvas");      // one image pixel per grid pixel
    this.bctx = this.buffer.getContext("2d");
    this.onSelect = onSelect || (() => {});
    this.onHover = onHover || (() => {});
    this.onFrame = onFrame || (() => {});
    this.onView = onView || (() => {});

    this.grid = null;
    this.base = [];          // frames as delivered: {pixels} or {rgb: [r, g, b]}, plus flags and meta
    this.extra = null;       // optional comparison frame from another mission (WISE)
    this.kind = "epochs";
    this.colour = false;
    this.display = [];       // frames after matching and applying the mode
    this.index = 0;
    this.stats = { med: 0, sigma: 1 };
    this.view = { zoom: 1, cx: 0, cy: 0 };
    this.options = {
      mode: "blink", black: 35, white: 38, invert: false, smooth: true, match: true,
      showFlags: false, showGaia: true, showTrack: true, showExtra: false, showNames: true,
    };
    this.gaia = [];
    this.cats = { simbad: [], twomass: [], allwise: [] };
    this._irOnly = null;
    this.sim = null;         // simulation state (see _buildSimulation)
    this.simOffset = 0;      // years from today
    this.measured = null;    // motion measured here for the selected object
    this.candidates = [];    // what the candidate finder flagged: {id, ra, dec, verdict}
    this.bottomInset = 0;    // CSS pixels at the bottom covered by the playback bar
    this.target = null;      // {name, ra_j2000, dec_j2000, pmra, pmdec}
    this.selection = null;   // {ra, dec}
    this._drag = null;

    new ResizeObserver(() => this._present()).observe(stage);
    overlay.addEventListener("pointerdown", (ev) => this._down(ev));
    overlay.addEventListener("pointermove", (ev) => this._move(ev));
    overlay.addEventListener("pointerup", (ev) => this._up(ev));
    overlay.addEventListener("pointercancel", () => { this._drag = null; });
    overlay.addEventListener("pointerleave", () => this.onHover(null));
    overlay.addEventListener("dblclick", (ev) => { ev.preventDefault(); this._zoomAt(ev, 2); });
    // Ctrl/Cmd + wheel (and trackpad pinch) zooms; a plain wheel still scrolls the page.
    overlay.addEventListener("wheel", (ev) => {
      if (!ev.ctrlKey && !ev.metaKey) return;
      ev.preventDefault();
      this._zoomAt(ev, ev.deltaY < 0 ? 1.25 : 0.8);
    }, { passive: false });
  }

  // ------------------------------------------------------------------ data

  clear() {
    this.grid = null;
    this.base = [];
    this.extra = null;
    this.display = [];
    this.index = 0;
    this.ctx.clearRect(0, 0, this.image.width, this.image.height);
    this.drawOverlay();
  }

  setData({ header, data, flags }) {
    const n = header.n;
    const size = n * n;
    const sameField = this.grid && this.grid.n === n && this.grid.ra === header.ra && this.grid.dec === header.dec
      && this.grid.fov_arcmin === header.fov_arcmin;
    this.grid = { n, ra: header.ra, dec: header.dec, scale_arcsec: header.scale_arcsec, fov_arcmin: header.fov_arcmin };
    if (!sameField) this.view = { zoom: 1, cx: n / 2, cy: n / 2 };
    this.kind = header.kind;
    this.colour = header.channels === 3;
    this.channelRanges = header.channel_ranges || null;
    if (this.colour) {
      this.base = header.frames.map((meta, i) => ({
        rgb: [0, 1, 2].map((c) => data.subarray((3 * i + c) * size, (3 * i + c + 1) * size)),
        flags: flags.subarray(i * size, (i + 1) * size),
        meta,
      }));
    } else {
      this.base = header.frames.map((meta, i) => ({
        pixels: data.subarray(i * size, (i + 1) * size),
        flags: flags.subarray(i * size, (i + 1) * size),
        meta,
      }));
    }
    if (this.buffer.width !== n) { this.buffer.width = n; this.buffer.height = n; }
    this._rebuild();
  }

  /** A comparison frame from another mission, on the same grid (or null to remove it). */
  setExtra(result) {
    if (!result) { this.extra = null; }
    else {
      const size = result.header.n * result.header.n;
      this.extra = { pixels: result.data.subarray(0, size), flags: null, meta: result.header.frames[0], credit: result.header.credit };
    }
    if (this.grid) this._rebuild();
  }

  setOption(key, value) {
    if (this.options[key] === value) return;
    this.options[key] = value;
    if (["mode", "match", "showExtra"].includes(key)) this._rebuild();
    else this.draw();
  }

  setGaia(stars) { this.gaia = stars || []; this._irOnly = null; this.drawOverlay(); }
  /** Catalogued objects in the field: {simbad: [...], twomass: [...], allwise: [...]}. */
  setCatalogues(cats) {
    this.cats = { simbad: cats.simbad || [], twomass: cats.twomass || [], allwise: cats.allwise || [] };
    this._irOnly = null;
    this.drawOverlay();
  }

  /** Infrared catalogue sources that are not already marked as a Gaia star (worked out once per field). */
  _infraredOnly() {
    if (this._irOnly) return this._irOnly;
    const gaiaNow = this.gaia.map((s) => propagate(s.ra, s.dec, s.pmra, s.pmdec, s.epoch || 2016, 2026));
    const near = (ra, dec, list) => list.some(([r2, d2]) => Math.abs(d2 - dec) < 0.001 && separationArcsec(ra, dec, r2, d2) < 3);
    const kept = [];
    // Only sources bright enough for SPHEREx to show; the WISE catalogue goes far deeper.
    const wise = this.cats.allwise.filter((o) => (o.w1 != null && o.w1 < 16.5) || (o.w2 != null && o.w2 < 15.5));
    for (const o of [...wise, ...this.cats.twomass]) {
      if (near(o.ra, o.dec, gaiaNow) || near(o.ra, o.dec, kept)) continue;
      kept.push([o.ra, o.dec]);
    }
    this._irOnly = kept;
    return kept;
  }
  setTarget(target) { this.target = target; this.drawOverlay(); }
  setSelection(sel) {
    this.selection = sel;
    // Brightness matching leaves out the area round the selection, so redo it.
    if (this.grid && this.base.length >= 2 && this.options.match) this._rebuild();
    else this.drawOverlay();
  }

  // ------------------------------------------------------------------ frame preparation

  /**
   * False colour: balance the three channels so ordinary stars come out white,
   * leaving anything unusually red (cold) or blue (hot) to stand out.
   */
  _balance(frame) {
    const size = this.grid.n * this.grid.n;
    const [r0, g0, b0] = frame.rgb;
    const mean0 = new Float32Array(size);
    for (let p = 0; p < size; p++) mean0[p] = (r0[p] + g0[p] + b0[p]) / 3;
    const { med, sigma } = robustStats([mean0]);
    const threshold = med + 12 * sigma;
    // One vote per star (its peak), so a single very bright or saturated star
    // with a huge halo cannot set the balance for the whole field.
    let ratios = this._starRatios(mean0, frame.rgb, frame.flags, med + 15 * sigma);
    if (ratios[0].length < 6) {
      ratios = [[], [], []];
      for (let p = 0; p < size; p += 1) {
        const m = mean0[p];
        if (!(m > threshold) || (frame.flags && frame.flags[p] >= 128)) continue;
        ratios[0].push(r0[p] / m); ratios[1].push(g0[p] / m); ratios[2].push(b0[p] / m);
      }
    }
    const out = [r0, g0, b0].map((channel, c) => {
      const list = ratios[c].filter(Number.isFinite).sort((a, b) => a - b);
      const ratio = list.length >= 6 ? Math.max(0.05, median(list)) : 1;
      const level = robustStats([channel]).med;
      const scaled = new Float32Array(size);
      for (let p = 0; p < size; p++) scaled[p] = (channel[p] - level) / ratio;
      return scaled;
    });
    const pixels = new Float32Array(size);
    for (let p = 0; p < size; p++) pixels[p] = (out[0][p] + out[1][p] + out[2][p]) / 3;
    return { ...frame, rgb: out, pixels };
  }

  /** Channel-to-mean brightness ratio at the peak of each unflagged star: [[red...], [green...], [blue...]]. */
  _starRatios(mean0, channels, flags, threshold) {
    const n = this.grid.n;
    const ratios = [[], [], []];
    for (let r = 2; r < n - 2; r++) {
      for (let c = 2; c < n - 2; c++) {
        const p = r * n + c;
        const m = mean0[p];
        if (!(m > threshold) || (flags && flags[p] >= 128)) continue;
        let peak = true;
        for (let dr = -2; dr <= 2 && peak; dr++) {
          for (let dc = -2; dc <= 2; dc++) {
            if (mean0[p + dr * n + dc] > m) { peak = false; break; }
          }
        }
        if (!peak) continue;
        let total = 0;
        const sums = [0, 0, 0];
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const q = p + dr * n + dc;
            total += mean0[q];
            for (let k = 0; k < 3; k++) sums[k] += channels[k][q];
          }
        }
        if (total > 0) for (let k = 0; k < 3; k++) ratios[k].push(sums[k] / total);
      }
    }
    return ratios;
  }

  /**
   * Scale every frame to a common brightness using the field's own stars.
   *
   * Each SPHEREx pass samples slightly different wavelengths inside a band, so
   * ordinary stars come out several percent brighter or fainter from pass to
   * pass (and a frame from another mission is in different units altogether).
   * The scale for a frame is the median ratio between the average frame and that
   * frame over bright pixels. The area around the selected object is left out,
   * so the object's own changes are not normalised away; with too few other
   * stars the frame is left as it is.
   */
  _matched(frames) {
    const size = this.grid.n * this.grid.n, n = this.grid.n;
    const foreign = frames.some((f) => f.meta.source);
    if (frames.length < 2 || !(this.options.match || foreign)) return frames.map((f) => ({ ...f, scale: 1 }));
    const own = frames.filter((f) => !f.meta.source);
    const reference = this._mean((own.length ? own : frames).map((f) => f.pixels));
    const { med, sigma } = robustStats([reference]);
    const threshold = med + 12 * sigma;
    let exclude = null;
    if (this.selection) {
      const [c0, r0] = skyToPix(this.grid, this.selection.ra, this.selection.dec);
      exclude = { c0, r0, radius: 25 / this.grid.scale_arcsec };
    }
    const bright = [];
    for (let p = 0; p < size; p++) {
      if (!(reference[p] > threshold)) continue;
      if (exclude) {
        const dc = (p % n) - exclude.c0, dr = Math.floor(p / n) - exclude.r0;
        if (dc * dc + dr * dr < exclude.radius * exclude.radius) continue;
      }
      bright.push(p);
    }
    return frames.map((f) => {
      const isForeign = Boolean(f.meta.source);
      if (!isForeign && !this.options.match) return { ...f, scale: 1 };
      // A frame from another mission has its own background and noise level (and
      // units), so it is judged by its own statistics and shifted onto ours.
      const self = isForeign ? robustStats([f.pixels]) : { med: 0, sigma: 0 };
      const floor = isForeign ? 8 * self.sigma : threshold * 0.5;
      const ratios = [];
      for (const p of bright) {
        const v = f.pixels[p] - self.med;
        if (Number.isFinite(v) && v > floor) ratios.push((reference[p] - med) / v);
      }
      if (ratios.length < 40) return { ...f, scale: 1 };
      ratios.sort((a, b) => a - b);
      const scale = Math.min(1e4, Math.max(1e-4, median(ratios)));
      const shift = isForeign ? med : 0;
      const rescale = (src) => {
        const out = new Float32Array(size);
        for (let p = 0; p < size; p++) out[p] = (src[p] - self.med) * scale + shift;
        return out;
      };
      return { ...f, pixels: rescale(f.pixels), rgb: f.rgb ? f.rgb.map(rescale) : undefined, scale };
    });
  }

  _rebuild() {
    if (!this.grid) return;
    const mode = this.options.mode;
    const size = this.grid.n * this.grid.n;
    let frames = this.base.map((f) => (f.rgb ? this._balance(f) : f));
    if (this.extra && this.options.showExtra && this.kind === "epochs" && frames.length) {
      frames = [this.extra, ...frames].sort((a, b) => a.meta.mjd - b.meta.mjd);
    }
    const base = this._matched(frames);
    this.frames = base;

    let display = [];
    if (!base.length) {
      display = [];
    } else if (mode === "diff" && base.length >= 2) {
      // Survey passes: subtract the first frame. Single exposures: subtract the
      // average of all of them, so anything fixed in the sky cancels.
      let reference, refFlags, refLabel;
      if (this.kind === "epochs") {
        reference = base[0].pixels; refFlags = base[0].flags; refLabel = this._when(base[0]);
      } else {
        reference = this._mean(base.map((f) => f.pixels)); refFlags = null; refLabel = "average";
      }
      const start = this.kind === "epochs" ? 1 : 0;
      // SPHEREx undersamples its PSF, so the same star is a little sharper in one
      // coadd than another and never subtracts perfectly. Smoothing the difference
      // by about one detector pixel suppresses those ring-shaped leftovers, while a
      // real mover keeps its one-sided dark/bright pair.
      const blurSigma = this.kind === "epochs" ? 6.15 / this.grid.scale_arcsec : 0;
      for (let i = start; i < base.length; i++) {
        let out = new Float32Array(size);
        const flags = new Uint8Array(size);
        const src = base[i].pixels, sf = base[i].flags;
        for (let p = 0; p < size; p++) {
          out[p] = src[p] - reference[p];
          flags[p] = Math.max(sf ? sf[p] : 0, refFlags ? refFlags[p] : 0);
        }
        if (blurSigma > 0.3) out = this._blur(out, blurSigma);
        display.push({ pixels: out, flags, meta: base[i].meta, diff: true, refLabel, source: i });
      }
    } else if (mode === "sim") {
      const own = base.filter((f) => !f.meta.source);
      this.sim = this._buildSimulation(own.length ? own : base);
      display.push(this._simFrame(this.simOffset));
    } else if (mode === "sum") {
      const own = base.filter((f) => !f.meta.source);
      const set = own.length ? own : base;
      const pixels = this._mean(set.map((f) => f.pixels));
      const rgb = set.every((f) => f.rgb) ? [0, 1, 2].map((c) => this._mean(set.map((f) => f.rgb[c]))) : undefined;
      const nexp = set.reduce((acc, f) => acc + (f.meta.nexp || 1), 0);
      const meta = {
        ...set[0].meta, nexp, sum: true, count: set.length,
        mjd: set.reduce((a, f) => a + f.meta.mjd, 0) / set.length,
        year: set.reduce((a, f) => a + f.meta.year, 0) / set.length,
      };
      display.push({ pixels, rgb, flags: this._flagMajority(set), meta, source: 0 });
    } else {
      display = base.map((f, i) => ({ pixels: f.pixels, rgb: f.rgb, flags: f.flags, meta: f.meta, source: i }));
    }
    this.display = display;
    // The stretch is set by this mission's frames, so a deeper comparison frame does not change it.
    const forStats = display.filter((f) => !f.meta.source);
    this.stats = mode === "sim" && this.sim
      ? this.sim.stats
      : robustStats((forStats.length ? forStats : display).map((f) => f.pixels));
    if (this.index >= display.length) this.index = 0;
    this.draw();
  }

  /** Gaussian blur that ignores missing (NaN) pixels. */
  _blur(pixels, sigma) {
    const n = this.grid.n;
    const radius = Math.max(1, Math.ceil(sigma * 2.5));
    const kernel = new Float32Array(2 * radius + 1);
    for (let k = -radius; k <= radius; k++) kernel[k + radius] = Math.exp(-(k * k) / (2 * sigma * sigma));
    const pass = (src, horizontal) => {
      const out = new Float32Array(src.length);
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          let sum = 0, weight = 0;
          for (let k = -radius; k <= radius; k++) {
            const rr = horizontal ? r : r + k, cc = horizontal ? c + k : c;
            if (rr < 0 || cc < 0 || rr >= n || cc >= n) continue;
            const v = src[rr * n + cc];
            if (!Number.isFinite(v)) continue;
            const w = kernel[k + radius];
            sum += w * v; weight += w;
          }
          out[r * n + c] = weight > 0 ? sum / weight : NaN;
        }
      }
      return out;
    };
    return pass(pass(pixels, true), false);
  }

  /** Pixels flagged in at least half of the frames (null when no frame carries flags). */
  _flagMajority(frames) {
    const withFlags = frames.filter((f) => f.flags);
    if (!withFlags.length) return null;
    const size = withFlags[0].flags.length;
    const count = new Uint8Array(size);
    for (const f of withFlags) for (let p = 0; p < size; p++) if (f.flags[p] >= 128) count[p]++;
    const need = Math.ceil(withFlags.length / 2);
    const out = new Uint8Array(size);
    for (let p = 0; p < size; p++) if (count[p] >= need) out[p] = 255;
    return out;
  }

  /**
   * Where the colour cannot be trusted: 1 on flagged pixels (saturated cores
   * above all), fading to 0 over about a detector pixel around them. Drawn
   * neutral, this looks like a burnt-out star in a photograph: a white core
   * with the colour returning smoothly in the halo, instead of a hard-edged block.
   */
  _softFlags(frame) {
    if (!frame.flags) return null;
    if (frame.soft !== undefined) return frame.soft;
    const fl = frame.flags;
    const mask = new Float32Array(fl.length);
    let any = false;
    for (let p = 0; p < fl.length; p++) if (fl[p] >= 128) { mask[p] = 1; any = true; }
    if (!any) { frame.soft = null; return null; }
    const blurred = this._blur(mask, (0.8 * 6.15) / this.grid.scale_arcsec);
    for (let p = 0; p < mask.length; p++) if (!mask[p]) mask[p] = Math.min(1, 2 * blurred[p]);
    frame.soft = mask;
    return mask;
  }

  _mean(frames) {
    const size = frames[0].length;
    const sum = new Float32Array(size);
    const count = new Uint16Array(size);
    for (const f of frames) {
      for (let p = 0; p < size; p++) {
        const v = f[p];
        if (Number.isFinite(v)) { sum[p] += v; count[p]++; }
      }
    }
    for (let p = 0; p < size; p++) sum[p] = count[p] ? sum[p] / count[p] : NaN;
    return sum;
  }

  // ------------------------------------------------------------------ simulation

  setCandidates(list) { this.candidates = list || []; this.drawOverlay(); }

  setMeasuredMotion(measured) { this.measured = measured; if (this.options.mode === "sim") this._rebuild(); }

  setSimOffset(years) {
    this.simOffset = years;
    if (this.options.mode !== "sim" || !this.sim) return;
    this.display = [this._simFrame(years)];
    this.index = 0;
    this.draw();
  }

  /**
   * Prepare the "time machine". Starting from today's combined image, every
   * source with a measured motion (Gaia, SIMBAD, the named target, or a motion
   * measured in this tool) is lifted out of the image as a small sprite, leaving
   * the local sky behind. Drawing the sprites displaced by motion x time gives
   * the field at another date. This is an extrapolation at constant velocity,
   * not an observation, and the page says so wherever it is shown.
   */
  _buildSimulation(own) {
    const n = this.grid.n, size = n * n, scale = this.grid.scale_arcsec;
    const colour = own.every((f) => f.rgb);
    const planes = [this._mean(own.map((f) => f.pixels))];
    if (colour) for (let c = 0; c < 3; c++) planes.push(this._mean(own.map((f) => f.rgb[c])));
    const year0 = own.reduce((a, f) => a + f.meta.year, 0) / own.length;
    const stats = robustStats([planes[0]]);
    const { med, sigma } = stats;

    // Everything with a measured motion, placed where it is today.
    const candidates = [];
    const add = (ra, dec, pmra, pmdec, epoch, label) => {
      if (!(Math.hypot(pmra || 0, pmdec || 0) > 0)) return;
      const [ra0, dec0] = propagate(ra, dec, pmra, pmdec, epoch, year0);
      const [col, row] = skyToPix(this.grid, ra0, dec0);
      if (!(col >= 1 && row >= 1 && col <= n - 2 && row <= n - 2)) return;
      // Grid pixels per year. East is to the left and north is up, hence the signs.
      candidates.push({ col, row, vx: -(pmra || 0) / 1000 / scale, vy: -(pmdec || 0) / 1000 / scale, label });
    };
    const t = this.target;
    if (t && (t.pmra || t.pmdec)) add(t.ra_j2000, t.dec_j2000, t.pmra, t.pmdec, 2000, t.name);
    for (const star of this.gaia) add(star.ra, star.dec, star.pmra, star.pmdec, star.epoch || 2016, null);
    for (const o of this.cats.simbad) add(o.ra, o.dec, o.pmra, o.pmdec, o.epoch || 2000, null);
    if (this.measured) add(this.measured.ra, this.measured.dec, this.measured.pmra, this.measured.pmdec, this.measured.epoch, "measured here");

    // One entry per source (the lists overlap), brightest first, visible ones only.
    const movers = [];
    for (const m of candidates) {
      if (movers.some((k) => Math.abs(k.col - m.col) < 2 && Math.abs(k.row - m.row) < 2)) continue;
      const c0 = Math.round(m.col), r0 = Math.round(m.row);
      let peak = -Infinity;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const v = planes[0][(r0 + dr) * n + (c0 + dc)];
        if (Number.isFinite(v) && v > peak) peak = v;
      }
      m.peak = peak - med;
      if (m.peak > 5 * sigma) movers.push(m);
    }
    movers.sort((a, b) => b.peak - a.peak);
    movers.length = Math.min(movers.length, 400);

    // A last plane records where the data are flagged (saturated cores), so that
    // travels with the star too and its colour stays neutral.
    const flagged = new Float32Array(size);
    const majority = this._flagMajority(own);
    if (majority) for (let q = 0; q < size; q++) if (majority[q]) flagged[q] = 1;
    planes.push(flagged);
    const nImage = planes.length - 1;
    const noise = planes.slice(0, nImage).map((plane, k) => (k ? robustStats([plane]).sigma : sigma));

    const native = 6.15 / scale;                       // grid pixels per detector pixel
    // Grain with the look of the image noise (unit spread, smooth over about one detector
    // pixel), from a fixed seed so the picture is the same every time.
    let grain = new Float32Array(size);
    let seed = 0x9e3779b9;
    const uniform = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let q = 0; q < size; q++) grain[q] = uniform() + uniform() + uniform() + uniform() - 2;
    grain = this._blur(grain, 0.45 * native);
    let spread = 0;
    for (let q = 0; q < size; q++) spread += grain[q] * grain[q];
    spread = Math.sqrt(spread / size) || 1;
    for (let q = 0; q < size; q++) grain[q] /= spread;
    const years = own.map((f) => f.meta.year);
    const yearSpan = Math.max(...years) - Math.min(...years);
    const kept = [];
    for (const m of movers) {
      const c0 = Math.round(m.col), r0 = Math.round(m.row);
      // Already carried off inside a brighter neighbour's core: it travels with that star.
      if (!(planes[0][r0 * n + c0] - med > 4 * sigma)) continue;

      // A saturated star has a flagged core. Find how far it reaches, so the
      // whole core travels as one piece and nothing of it is left behind.
      const reach = Math.ceil(12 * native);
      const total = new Uint16Array(reach + 1), bad = new Uint16Array(reach + 1);
      for (let r = Math.max(0, r0 - reach); r <= Math.min(n - 1, r0 + reach); r++) {
        for (let c = Math.max(0, c0 - reach); c <= Math.min(n - 1, c0 + reach); c++) {
          const b = Math.round(Math.hypot(c - m.col, r - m.row));
          if (b > reach) continue;
          total[b]++;
          if (flagged[r * n + c] > 0) bad[b]++;
        }
      }
      let saturated = 0;
      for (let b = 0; b <= reach; b++) {
        if (!total[b]) continue;                       // no pixel centre falls in this ring
        if (bad[b] < 0.15 * total[b]) break;
        saturated = b + 1;
      }
      // Inside `core` the star's actual pixels are used; beyond `edge`, a smooth profile.
      // A fast star is drawn out along its path over the months of observing, so its core is wider.
      const smear = Math.min(3 * native, 0.5 * Math.hypot(m.vx, m.vy) * yearSpan);
      const core = Math.max(1.8 * native, saturated + 1.2 * native) + smear;
      const edge = core + 1.5 * native;

      // How far to look for this star's light: brighter stars have wider halos.
      const strength = Math.log10(Math.max(1, m.peak / (15 * sigma)));
      const limit = Math.ceil(Math.max(native * Math.min(30, 4 + 7 * strength), 2.5 * edge));
      const bx0 = Math.max(0, c0 - limit), bx1 = Math.min(n - 1, c0 + limit);
      const by0 = Math.max(0, r0 - limit), by1 = Math.min(n - 1, r0 + limit);

      // Radial profile of each image plane: the median in rings one pixel wide.
      // A median ignores neighbouring stars, so they are left where they are.
      const bins = limit + 1;
      const rings = planes.slice(0, nImage).map(() => Array.from({ length: bins }, () => []));
      for (let r = by0; r <= by1; r++) for (let c = bx0; c <= bx1; c++) {
        const b = Math.floor(Math.hypot(c - m.col, r - m.row));
        if (b >= bins) continue;
        for (let k = 0; k < nImage; k++) {
          const v = planes[k][r * n + c];
          if (Number.isFinite(v)) rings[k][b].push(v);
        }
      }
      const profiles = [], skies = [];
      for (let k = 0; k < nImage; k++) {
        const raw = rings[k].map((values) => {
          if (!values.length) return NaN;
          values.sort((a, b) => a - b);
          return values[Math.floor(values.length / 2)];
        });
        // Sky level: the outermost rings.
        const outer = raw.slice(Math.floor(bins * 0.85)).filter(Number.isFinite).sort((a, b) => a - b);
        const sky = outer.length ? outer[Math.floor(outer.length / 2)] : med;
        // Above the sky, never negative, and never rising outwards. Rings with no
        // data (the middle of a saturated core) take the nearest value outside them.
        const profile = new Float32Array(bins);
        let first = 0;
        while (first < bins - 1 && !Number.isFinite(raw[first])) first++;
        let ceiling = Number.isFinite(raw[first]) ? Math.max(0, raw[first] - sky) : 0;
        for (let b = 0; b < bins; b++) {
          if (b > first && Number.isFinite(raw[b])) ceiling = Math.min(ceiling, Math.max(0, raw[b] - sky));
          profile[b] = ceiling;
        }
        profiles.push(profile);
        skies.push(sky);
      }
      // The sprite ends where the star's light has sunk well into the noise in
      // every plane, and fades out over its last third so no edge is left.
      let R = limit;
      for (let b = Math.ceil(edge) + 1; b < bins - 1; b++) {
        let faint = true;
        for (let k = 0; k < nImage && faint; k++) faint = profiles[k][b] < 0.2 * noise[k];
        if (faint) { R = b + 1; break; }
      }
      R = Math.min(limit, Math.max(R, Math.ceil(edge * 1.5)));

      const x0 = Math.max(0, c0 - R), x1 = Math.min(n - 1, c0 + R);
      const y0 = Math.max(0, r0 - R), y1 = Math.min(n - 1, r0 + R);
      const w = x1 - x0 + 1, h = y1 - y0 + 1;
      const sprite = { x0, y0, w, h, planes: planes.map(() => new Float32Array(w * h)) };
      for (let r = y0; r <= y1; r++) for (let c = x0; c <= x1; c++) {
        const d = Math.hypot(c - m.col, r - m.row);
        if (d >= R) continue;
        const q = r * n + c, t = (r - y0) * w + (c - x0);
        // 1 in the core (actual pixels), fading to 0 (smooth profile) just outside it.
        const blend = d <= core ? 1 : d >= edge ? 0 : (edge - d) / (edge - core);
        const taper = d > 0.7 * R ? (R - d) / (0.3 * R) : 1;
        const b = Math.min(bins - 2, Math.floor(d)), frac = d - b;
        for (let k = 0; k < nImage; k++) {
          const v = planes[k][q];
          const model = (profiles[k][b] * (1 - frac) + profiles[k][b + 1] * frac) * taper;
          // A missing pixel in a saturated core takes the profile's value, so the star has no holes.
          const actual = Number.isFinite(v) ? v - skies[k] : model;
          // Outside the core only the smooth profile is lifted, so the sky's own noise and any
          // neighbouring star stay where they are; never much more than is there, which would leave a hole.
          const lifted = blend * actual + (1 - blend) * Math.max(0, Math.min(model, actual + 2 * noise[k]));
          sprite.planes[k][t] = lifted;
          planes[k][q] = (Number.isFinite(v) ? v : skies[k] + model) - lifted;
        }
        if (blend > 0 && flagged[q] > 0) { sprite.planes[nImage][t] = flagged[q]; flagged[q] = 0; }
      }
      // What lies behind the star is unknown. That patch is filled with the sky
      // level plus a little grain like the image noise, not left as a flat hole.
      for (let r = Math.max(0, Math.floor(m.row - edge)); r <= Math.min(n - 1, Math.ceil(m.row + edge)); r++) {
        for (let c = Math.max(0, Math.floor(m.col - edge)); c <= Math.min(n - 1, Math.ceil(m.col + edge)); c++) {
          const d = Math.hypot(c - m.col, r - m.row);
          if (d >= edge) continue;
          const blend = d <= core ? 1 : (edge - d) / (edge - core);
          const q = r * n + c;
          for (let k = 0; k < nImage; k++) {
            planes[k][q] = (1 - blend) * planes[k][q] + blend * (skies[k] + 0.5 * grain[q] * noise[k]);
          }
        }
      }
      m.sprite = sprite;
      m.shape = { saturated, core, radius: R, limit };
      kept.push(m);
    }
    return { year0, planes, movers: kept, colour, stats };
  }

  /** The simulated field `offset` years from today. */
  _simFrame(offset) {
    const sim = this.sim, n = this.grid.n;
    const out = sim.planes.map((plane) => Float32Array.from(plane));
    for (const m of sim.movers) {
      const dx = m.vx * offset, dy = m.vy * offset;
      const ix = Math.floor(dx), iy = Math.floor(dy);
      const fx = dx - ix, fy = dy - iy;
      const { x0, y0, w, h } = m.sprite;
      if (x0 + ix > n || y0 + iy > n || x0 + ix + w < -1 || y0 + iy + h < -1) continue;   // has left the frame
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
      for (let k = 0; k < out.length; k++) {
        const data = m.sprite.planes[k], plane = out[k];
        for (let r = 0; r < h; r++) {
          const ty = y0 + r + iy;
          if (ty < 0 || ty >= n - 1) continue;
          for (let c = 0; c < w; c++) {
            const v = data[r * w + c];
            if (v === 0) continue;
            const tx = x0 + c + ix;
            if (tx < 0 || tx >= n - 1) continue;
            const q = ty * n + tx;
            plane[q] += v * w00; plane[q + 1] += v * w10; plane[q + n] += v * w01; plane[q + n + 1] += v * w11;
          }
        }
      }
    }
    const flagPlane = out[out.length - 1];
    const flags = new Uint8Array(flagPlane.length);
    for (let q = 0; q < flags.length; q++) if (flagPlane[q] >= 0.5) flags[q] = 255;
    return {
      pixels: out[0], rgb: sim.colour ? [out[1], out[2], out[3]] : undefined, flags, source: 0,
      meta: { sim: true, year: sim.year0 + offset, offset, movers: sim.movers.length, mjd: 0, nexp: 0 },
    };
  }

  get simInfo() { return this.sim ? { year0: this.sim.year0, movers: this.sim.movers.length } : null; }

  // ------------------------------------------------------------------ frames and labels

  get frameCount() { return this.display.length; }
  get current() { return this.display[this.index] || null; }

  setIndex(i) {
    if (!this.display.length) return;
    this.index = ((i % this.display.length) + this.display.length) % this.display.length;
    this.draw();
  }
  step(delta) { this.setIndex(this.index + delta); }

  _when(frame) {
    const m = frame.meta;
    if (m.source) return `${m.source} ${Math.floor(m.year)}`;
    if (this.kind === "exposures") return prettyDate(m.date);
    // Near the ecliptic poles two passes can fall in the same month; show the day then.
    const months = (this.frames || []).filter((f) => !f.meta.source).map((f) => monthYear(f.meta.date));
    return new Set(months).size < months.length ? prettyDate(m.date) : monthYear(m.date);
  }

  /** Short stamp shown on the image, e.g. "AUG 2025". */
  stamp(i = this.index) {
    const f = this.display[i];
    if (!f) return { main: "", sub: "" };
    const m = f.meta;
    if (m.sim) {
      const year = Math.round(m.year);
      const main = year >= 1 ? `YEAR ${year}` : `${1 - year} BC`;
      const away = Math.abs(Math.round(m.offset));
      const sub = away === 0 ? "Today" : `${away.toLocaleString("en-US")} years ${m.offset > 0 ? "from now" : "ago"}`;
      return { main, sub };
    }
    if (m.sum) return { main: "ALL COMBINED", sub: `${m.count} ${this.kind === "epochs" ? "passes" : "exposures"}` };
    const own = this.frames.filter((x) => !x.meta.source);
    const position = own.indexOf(this.frames[f.source]) + 1;
    let sub;
    if (m.source) sub = `${m.source} mission · ${m.wise_band || ""}`.trim();
    else if (this.kind === "exposures") sub = `Exposure ${position} of ${own.length}`;
    else sub = `Pass ${position} of ${own.length}`;
    if (f.diff) sub += ` minus ${f.refLabel}`;
    return { main: this._when(f).toUpperCase(), sub };
  }

  frameLabel(i = this.index) {
    const f = this.display[i];
    if (!f) return "";
    const m = f.meta;
    if (m.sim) {
      return `Simulated: ${m.movers} stars moved in straight lines at their measured speeds (Gaia, SIMBAD). Not an observation. Stars with no measured motion, and galaxies, stay put.`;
    }
    if (m.sum) return `${m.count} ${this.kind === "epochs" ? "passes" : "exposures"} combined · ${m.nexp} exposures`;
    if (m.source) return `${m.source} ${m.wise_band || ""} image, ${prettyDate(m.date)} · ${m.wl.toFixed(1)} µm`;
    const wl = m.wl_min != null && m.wl_max != null
      ? (Math.abs(m.wl_max - m.wl_min) < 0.005 ? `${m.wl_min.toFixed(3)} µm` : `${m.wl_min.toFixed(2)}–${m.wl_max.toFixed(2)} µm`)
      : "";
    if (this.kind === "exposures") return `${prettyDate(m.date)}${wl ? " · " + wl : ""}`;
    const span = m.date_min && m.date_max && m.date_min !== m.date_max ? `${prettyDate(m.date_min)} – ${prettyDate(m.date_max)}` : prettyDate(m.date);
    return `${span} · ${m.nexp} exposure${m.nexp === 1 ? "" : "s"}${wl ? " · " + wl : ""}`;
  }

  shortLabel(i) {
    const f = this.display[i];
    if (!f) return "";
    if (f.meta.sum) return "Sum";
    if (f.meta.sim) return "Simulation";
    return this._when(f);
  }

  // ------------------------------------------------------------------ drawing

  draw() {
    const frame = this.current;
    if (!frame || !this.grid) { this._present(); this.onFrame(); return; }
    const n = this.grid.n;
    const img = this.bctx.createImageData(n, n);
    const out = img.data;
    const { med, sigma } = this.stats;
    const { black, white, invert, showFlags } = this.options;
    const kHi = 5 * Math.pow(10, (white / 100) * 2.8);      // white point, in units of the noise
    const px = frame.pixels, fl = frame.flags;
    const [fr, fg, fb] = OVERLAY.flag;
    const colour = Boolean(frame.rgb) && !frame.diff;

    let lo, span, symmetric = false;
    if (frame.diff) {
      symmetric = true;
      span = Math.max(sigma * kHi / 2, 1e-9);
      lo = 0;
    } else {
      lo = med + ((black - 50) / 10) * sigma;
      span = Math.max(sigma * kHi, 1e-9);
    }
    const fade = 6 * sigma;                                   // colour fades out in the noise
    const soft = colour ? this._softFlags(frame) : null;

    for (let p = 0, o = 0; p < px.length; p++, o += 4) {
      const v = px[p];
      let r, g, b;
      if (!Number.isFinite(v)) {
        r = g = b = symmetric ? 0.5 : 0;
      } else if (symmetric) {
        r = g = b = 0.5 + 0.5 * Math.asinh((ASINH_A * v) / span) / ASINH_NORM;
      } else {
        const above = v - lo;
        let t = above > 0 ? Math.asinh((ASINH_A * above) / span) / ASINH_NORM : 0;
        if (t > 1) t = 1;
        // Flagged pixels (saturated cores above all) have no trustworthy colour: they are
        // shown neutral, and the colour returns gradually around them.
        if (colour && above > 0 && !(soft && soft[p] >= 1)) {
          let s = (v - med) / fade;
          s = s < 0 ? 0 : s > 1 ? 1 : s;
          if (soft) s *= 1 - soft[p];
          const tint = (channel) => {
            let rel = (channel[p] - lo) / above;
            rel = rel < 0 ? 0 : rel > 3 ? 3 : rel;
            return t * (1 + s * (rel - 1));
          };
          r = tint(frame.rgb[0]); g = tint(frame.rgb[1]); b = tint(frame.rgb[2]);
          const peak = Math.max(r, g, b);
          if (peak > 1) { r /= peak; g /= peak; b /= peak; }
        } else {
          r = g = b = t;
        }
      }
      r = r < 0 ? 0 : r > 1 ? 1 : r; g = g < 0 ? 0 : g > 1 ? 1 : g; b = b < 0 ? 0 : b > 1 ? 1 : b;
      if (invert) { r = 1 - r; g = 1 - g; b = 1 - b; }
      if (showFlags && fl && fl[p] >= 128) {
        out[o] = Math.round(r * 255 * 0.45 + fr * 0.55);
        out[o + 1] = Math.round(g * 255 * 0.45 + fg * 0.55);
        out[o + 2] = Math.round(b * 255 * 0.45 + fb * 0.55);
      } else {
        out[o] = Math.round(r * 255); out[o + 1] = Math.round(g * 255); out[o + 2] = Math.round(b * 255);
      }
      out[o + 3] = 255;
    }
    this.bctx.putImageData(img, 0, 0);
    this._present();
    this.onFrame();
  }

  /** Visible part of the grid, in grid pixels. */
  _window() {
    const n = this.grid.n;
    const span = n / this.view.zoom;
    const x0 = Math.min(Math.max(this.view.cx - span / 2, 0), n - span);
    const y0 = Math.min(Math.max(this.view.cy - span / 2, 0), n - span);
    return { x0, y0, span };
  }

  /** Copy the visible window of the frame onto the screen canvas, then draw the overlays. */
  _present() {
    const rect = this.stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
    for (const canvas of [this.image, this.overlay]) {
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    }
    this.ctx.clearRect(0, 0, w, h);
    if (this.grid && this.display.length) {
      const { x0, y0, span } = this._window();
      this.ctx.imageSmoothingEnabled = this.options.smooth;
      this.ctx.imageSmoothingQuality = "high";
      this.ctx.drawImage(this.buffer, x0, y0, span, span, 0, 0, w, h);
    }
    this.drawOverlay();
  }

  /** Sky position -> overlay coordinates in CSS pixels. */
  _toScreen(ra, dec, size) {
    const [col, row] = skyToPix(this.grid, ra, dec);
    const { x0, y0, span } = this._window();
    const k = size / span;
    return [(col + 0.5 - x0) * k, (row + 0.5 - y0) * k];
  }

  drawOverlay() {
    const rect = this.stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const size = rect.width;
    const c = this.octx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, size, size);
    if (!this.grid) return;
    const frame = this.current;
    const year = frame ? frame.meta.year : 2026;
    const inside = (x, y, m = 0) => x >= -m && y >= -m && x <= size + m && y <= size + m;
    c.lineCap = "round";
    c.font = "12px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";

    // Simulation: a fading trail from where each star is today to where it has got to.
    if (frame && frame.meta.sim && this.sim && this.simOffset !== 0) {
      const { x0, y0, span } = this._window();
      const k = size / span;
      let drawn = 0;
      for (const m of this.sim.movers) {
        const dx = m.vx * this.simOffset, dy = m.vy * this.simOffset;
        if (Math.hypot(dx, dy) < 2.5) continue;
        const ax = (m.col + 0.5 - x0) * k, ay = (m.row + 0.5 - y0) * k;
        const bx = ax + dx * k, by = ay + dy * k;
        if (!inside(ax, ay, 60) && !inside(bx, by, 60)) continue;
        const gradient = c.createLinearGradient(ax, ay, bx, by);
        gradient.addColorStop(0, "rgba(255, 255, 255, 0)");
        gradient.addColorStop(1, m.label ? "rgba(255, 184, 77, 0.75)" : "rgba(190, 215, 255, 0.5)");
        c.strokeStyle = gradient; c.lineWidth = m.label ? 2 : 1.2;
        c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
        if (++drawn > 160) break;
      }
    }

    // Known stars from Gaia, moved to the date of this frame.
    if (this.options.showGaia && this.gaia.length) {
      // Stars arrive brightest first. Draw fewer on a small screen so the image stays readable.
      const limit = Math.min(140, Math.round(size / 4.5));
      const radius = size < 420 ? 4 : 5;
      let drawn = 0;
      for (const star of this.gaia) {
        const pm = Math.hypot(star.pmra || 0, star.pmdec || 0);
        const fast = pm >= FAST_PM_MASYR;
        if (!fast && drawn >= limit) continue;
        const [ra, dec] = propagate(star.ra, star.dec, star.pmra, star.pmdec, star.epoch || 2016, year);
        const [x, y] = this._toScreen(ra, dec, size);
        if (!inside(x, y)) continue;
        drawn++;
        c.beginPath();
        c.arc(x, y, fast ? 9 : radius, 0, 2 * Math.PI);
        c.strokeStyle = fast ? OVERLAY.gaiaFast : OVERLAY.gaia;
        c.lineWidth = fast ? 1.8 : 1;
        c.globalAlpha = fast ? 1 : 0.55;
        c.stroke();
        c.globalAlpha = 1;
        if (fast) {
          // Arrow: where this star will be in 10 years.
          const [ra2, dec2] = propagate(ra, dec, star.pmra, star.pmdec, year, year + 10);
          const [x2, y2] = this._toScreen(ra2, dec2, size);
          this._arrow(c, x, y, x2, y2, 11, OVERLAY.gaiaFast);
          this._text(c, `${(pm / 1000).toFixed(pm >= 1000 ? 1 : 2)}″/yr`, x + 14, y + 22);
        }
      }
    }

    // Infrared catalogue sources (2MASS, WISE) that Gaia does not list: small diamonds.
    if (this.options.showGaia && (this.cats.twomass.length || this.cats.allwise.length)) {
      c.strokeStyle = OVERLAY.infrared; c.lineWidth = 1; c.globalAlpha = 0.7;
      const d = 4;
      for (const [ra, dec] of this._infraredOnly()) {
        const [x, y] = this._toScreen(ra, dec, size);
        if (!inside(x, y)) continue;
        c.beginPath(); c.moveTo(x, y - d); c.lineTo(x + d, y); c.lineTo(x, y + d); c.lineTo(x - d, y); c.closePath(); c.stroke();
      }
      c.globalAlpha = 1;
    }

    // Names of catalogued objects (SIMBAD), moved to the date of this frame.
    if (this.options.showNames && this.cats.simbad.length) this._drawNames(c, size, year, inside);

    // Where the named target should be, from its published position and motion.
    if (this.options.showTrack && this.target && (this.target.pmra || this.target.pmdec)) {
      const t = this.target;
      c.strokeStyle = OVERLAY.track; c.fillStyle = OVERLAY.track;
      for (const f of this.display) {
        const [ra, dec] = propagate(t.ra_j2000, t.dec_j2000, t.pmra, t.pmdec, 2000, f.meta.year);
        const [x, y] = this._toScreen(ra, dec, size);
        if (!inside(x, y)) continue;
        c.globalAlpha = 0.55;
        c.beginPath(); c.arc(x, y, 2, 0, 2 * Math.PI); c.fill();
        c.globalAlpha = 1;
      }
      const [ra, dec] = propagate(t.ra_j2000, t.dec_j2000, t.pmra, t.pmdec, 2000, year);
      const [x, y] = this._toScreen(ra, dec, size);
      if (inside(x, y)) {
        c.lineWidth = 1.6;
        c.beginPath(); c.arc(x, y, 14, 0, 2 * Math.PI); c.stroke();
        this._text(c, `${t.name} (predicted)`, x + 18, y - 10);
      }
    }

    // What the candidate finder flagged: numbered corner brackets, brightest for the unexplained.
    if (this.options.mode !== "sim") {
      c.font = "600 12px Inter, system-ui, sans-serif";
      for (const cand of this.candidates) {
        const [x, y] = this._toScreen(cand.ra, cand.dec, size);
        if (!inside(x, y)) continue;
        c.strokeStyle = OVERLAY.candidate[cand.verdict] || OVERLAY.candidate.doubtful;
        c.lineWidth = cand.verdict === "unexplained" ? 2.2 : 1.5;
        const h = 16, k = 7;
        for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
          c.beginPath();
          c.moveTo(x + sx * h, y + sy * (h - k)); c.lineTo(x + sx * h, y + sy * h); c.lineTo(x + sx * (h - k), y + sy * h);
          c.stroke();
        }
        this._text(c, String(cand.id), x - h - 2, y - h - 4);
      }
    }

    // The source being measured.
    if (this.selection) {
      const [x, y] = this._toScreen(this.selection.ra, this.selection.dec, size);
      if (inside(x, y)) {
        c.strokeStyle = OVERLAY.select; c.lineWidth = 1.6;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          c.beginPath(); c.moveTo(x + dx * 9, y + dy * 9); c.lineTo(x + dx * 20, y + dy * 20); c.stroke();
        }
      }
    }

    // Scale bar (a round length near a fifth of the width) and compass.
    const pxPerArcsec = size / (this._window().span * this.grid.scale_arcsec);
    const choices = [2, 5, 10, 20, 30, 60, 120, 300];
    const length = choices.reduce((a, b) => (Math.abs(b * pxPerArcsec - size * 0.2) < Math.abs(a * pxPerArcsec - size * 0.2) ? b : a));
    const bar = length * pxPerArcsec;
    c.strokeStyle = OVERLAY.chrome; c.lineWidth = 2;
    const barY = size - 22 - (this.bottomInset || 0);
    c.beginPath(); c.moveTo(16, barY); c.lineTo(16 + bar, barY); c.stroke();
    this._text(c, length >= 60 ? `${length / 60} arcmin` : `${length} arcsec`, 16, barY - 8);
    const cx = size - 28, cy = 48;
    this._arrow(c, cx, cy, cx, cy - 26, 0, OVERLAY.chrome);
    this._arrow(c, cx, cy, cx - 26, cy, 0, OVERLAY.chrome);
    this._text(c, "N", cx - 4, cy - 30);
    this._text(c, "E", cx - 40, cy + 4);
  }

  _drawNames(c, size, year, inside) {
    // The searched-for target has its own label; do not print its name twice.
    let targetXY = null;
    if (this.options.showTrack && this.target && (this.target.pmra || this.target.pmdec)) {
      const t = this.target;
      const [ra, dec] = propagate(t.ra_j2000, t.dec_j2000, t.pmra, t.pmdec, 2000, year);
      targetXY = this._toScreen(ra, dec, size);
    }
    const candidates = [];
    for (const o of this.cats.simbad) {
      const label = labelFor(o);
      if (label.plainStar && !label.proper) continue;          // "Star" with only a serial number: the Gaia circle says enough
      const [ra, dec] = propagate(o.ra, o.dec, o.pmra, o.pmdec, o.epoch || 2000, year);
      const [x, y] = this._toScreen(ra, dec, size);
      if (!inside(x, y, -6)) continue;
      if (targetXY && Math.hypot(x - targetXY[0], y - targetXY[1]) < 16) continue;
      candidates.push({ x, y, ...label });
    }
    // Real names first, then typed objects; cap the count so the image stays readable.
    candidates.sort((a, b) => Number(b.proper) - Number(a.proper));
    const limit = size < 420 ? 10 : 26;
    const placed = [];
    c.font = "11px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
    for (const item of candidates) {
      if (placed.length >= limit) break;
      const width = c.measureText(item.text).width;
      // Try right, left, below, above; skip the label if every spot collides or leaves the image.
      const spots = [[9, 4], [-9 - width, 4], [-width / 2, 17], [-width / 2, -9]];
      let box = null;
      for (const [dx, dy] of spots) {
        const candidate = { x0: item.x + dx - 2, y0: item.y + dy - 11, x1: item.x + dx + width + 2, y1: item.y + dy + 3 };
        if (candidate.x0 < 2 || candidate.y0 < 2 || candidate.x1 > size - 2 || candidate.y1 > size - 2) continue;
        if (placed.some((b) => candidate.x0 < b.x1 && candidate.x1 > b.x0 && candidate.y0 < b.y1 && candidate.y1 > b.y0)) continue;
        box = { ...candidate, tx: item.x + dx, ty: item.y + dy };
        break;
      }
      if (!box) continue;
      placed.push(box);
      c.fillStyle = item.proper ? OVERLAY.name : OVERLAY.kind;
      c.beginPath(); c.arc(item.x, item.y, 1.6, 0, 2 * Math.PI); c.fill();
      c.lineWidth = 3; c.strokeStyle = "rgba(0, 0, 0, 0.8)"; c.lineJoin = "round";
      c.strokeText(item.text, box.tx, box.ty);
      c.fillText(item.text, box.tx, box.ty);
    }
    c.font = "12px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
  }

  _text(c, text, x, y) {
    c.lineWidth = 3; c.strokeStyle = "rgba(0, 0, 0, 0.75)"; c.lineJoin = "round";
    c.strokeText(text, x, y);
    c.fillStyle = "#eef1f6";
    c.fillText(text, x, y);
  }

  _arrow(c, x1, y1, x2, y2, gap, color) {
    const len = Math.hypot(x2 - x1, y2 - y1);
    if (len < gap + 6) return;
    const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
    const sx = x1 + ux * gap, sy = y1 + uy * gap;
    c.strokeStyle = color; c.lineWidth = 1.6;
    c.beginPath(); c.moveTo(sx, sy); c.lineTo(x2, y2); c.stroke();
    const h = 6;
    c.beginPath();
    c.moveTo(x2, y2); c.lineTo(x2 - ux * h - uy * h * 0.6, y2 - uy * h + ux * h * 0.6);
    c.moveTo(x2, y2); c.lineTo(x2 - ux * h + uy * h * 0.6, y2 - uy * h - ux * h * 0.6);
    c.stroke();
  }

  // ------------------------------------------------------------------ zoom and pan

  get zoom() { return this.view.zoom; }

  zoomBy(factor) {
    if (!this.grid) return;
    this.view.zoom = Math.min(MAX_ZOOM, Math.max(1, this.view.zoom * factor));
    if (this.view.zoom === 1) { this.view.cx = this.grid.n / 2; this.view.cy = this.grid.n / 2; }
    this._clampView();
    this._present();
    this.onView();
  }

  resetView() {
    if (!this.grid) return;
    this.view = { zoom: 1, cx: this.grid.n / 2, cy: this.grid.n / 2 };
    this._present();
    this.onView();
  }

  _clampView() {
    const n = this.grid.n, half = n / this.view.zoom / 2;
    this.view.cx = Math.min(Math.max(this.view.cx, half), n - half);
    this.view.cy = Math.min(Math.max(this.view.cy, half), n - half);
  }

  _zoomAt(ev, factor) {
    if (!this.grid) return;
    const before = this._pixelAt(ev);
    const zoom = Math.min(MAX_ZOOM, Math.max(1, this.view.zoom * factor));
    if (zoom === this.view.zoom) return;
    if (before) {
      // Keep the point under the cursor where it is.
      const rect = this.overlay.getBoundingClientRect();
      const fx = (ev.clientX - rect.left) / rect.width, fy = (ev.clientY - rect.top) / rect.height;
      const span = this.grid.n / zoom;
      this.view.cx = before.col + 0.5 - fx * span + span / 2;
      this.view.cy = before.row + 0.5 - fy * span + span / 2;
    }
    this.view.zoom = zoom;
    this._clampView();
    this._present();
    this.onView();
  }

  // ------------------------------------------------------------------ pointer

  _pixelAt(ev) {
    if (!this.grid) return null;
    const rect = this.overlay.getBoundingClientRect();
    const { x0, y0, span } = this._window();
    const col = x0 + ((ev.clientX - rect.left) / rect.width) * span - 0.5;
    const row = y0 + ((ev.clientY - rect.top) / rect.height) * span - 0.5;
    const n = this.grid.n;
    if (col < -0.5 || row < -0.5 || col > n - 0.5 || row > n - 0.5) return null;
    return { col, row };
  }

  _down(ev) {
    if (ev.button !== 0) return;
    this._drag = { x: ev.clientX, y: ev.clientY, cx: this.view.cx, cy: this.view.cy, moved: false };
    this.overlay.setPointerCapture(ev.pointerId);
  }

  _move(ev) {
    const drag = this._drag;
    if (drag && this.grid) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 5) drag.moved = true;
      if (drag.moved && this.view.zoom > 1) {
        const rect = this.overlay.getBoundingClientRect();
        const k = this.grid.n / this.view.zoom / rect.width;
        this.view.cx = drag.cx - dx * k;
        this.view.cy = drag.cy - dy * k;
        this._clampView();
        this._present();
        return;
      }
    }
    const p = this._pixelAt(ev);
    const frame = this.current;
    if (!p || !frame) { this.onHover(null); return; }
    const [ra, dec] = pixToSky(this.grid, p.col, p.row);
    const n = this.grid.n;
    const idx = Math.min(n - 1, Math.max(0, Math.round(p.row))) * n + Math.min(n - 1, Math.max(0, Math.round(p.col)));
    this.onHover({ ra, dec, value: frame.pixels[idx], flagged: frame.flags ? frame.flags[idx] >= 128 : false, text: formatCoords(ra, dec) });
  }

  _up(ev) {
    const drag = this._drag;
    this._drag = null;
    if (!drag || drag.moved) return;
    if (this.current && this.current.meta.sim && Math.abs(this.simOffset) > 0.5) return;   // simulated positions are not real ones
    const p = this._pixelAt(ev);
    if (!p || !this.display.length) return;
    const snapped = this._snap(p.col, p.row);
    const [ra, dec] = pixToSky(this.grid, snapped.col, snapped.row);
    this.onSelect({ ra, dec });
  }

  /** Move a click onto the nearest bright peak (within about one detector pixel). */
  _snap(col, row) {
    const frame = this.current && !this.current.diff ? this.current : this.frames[this.index % this.frames.length];
    if (!frame) return { col, row };
    const n = this.grid.n, px = frame.pixels;
    const radius = Math.max(2, Math.round(7 / this.grid.scale_arcsec));
    const c0 = Math.round(col), r0 = Math.round(row);
    let best = -Infinity, bc = c0, br = r0;
    for (let r = Math.max(1, r0 - radius); r <= Math.min(n - 2, r0 + radius); r++) {
      for (let c = Math.max(1, c0 - radius); c <= Math.min(n - 2, c0 + radius); c++) {
        const v = px[r * n + c];
        if (Number.isFinite(v) && v > best) { best = v; bc = c; br = r; }
      }
    }
    // Only snap when there is a clear source there; otherwise keep the exact click.
    if (!(best > this.stats.med + 6 * this.stats.sigma)) return { col, row };
    let sw = 0, sc = 0, sr = 0;
    for (let r = Math.max(0, br - 2); r <= Math.min(n - 1, br + 2); r++) {
      for (let c = Math.max(0, bc - 2); c <= Math.min(n - 1, bc + 2); c++) {
        const v = px[r * n + c] - this.stats.med;
        if (Number.isFinite(v) && v > 0) { sw += v; sc += v * c; sr += v * r; }
      }
    }
    return sw > 0 ? { col: sc / sw, row: sr / sw } : { col: bc, row: br };
  }
}
