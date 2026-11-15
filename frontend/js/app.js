// SPHEREx Blink: page controller (state, URL, loading, controls).
// Developed with AI assistance (Claude, Anthropic). See README "Use of AI".

import { getJSON, getFrames } from "./api.js";
import { Viewer } from "./viewer.js";
import { renderSpectrum, renderLightCurve, renderMessage, renderTable, toCsv, formatFlux } from "./charts.js";
import { formatCoords, prettyDate } from "./sky.js";
import { identify, isSurveyId, MATCH_RADIUS_ARCSEC } from "./names.js";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const round5 = (v) => Math.round(v * 1e5) / 1e5;

const DEFAULTS = { fov: 6, band: "colour", mode: "blink", kind: "epochs", speed: 800 };
const FALLBACK_BAND = "2";      // shown while the bands needed for Colour or All are still arriving
const MODE_HINTS = {
  blink: "Flip between frames. Fixed stars stay put; anything that moved or changed jumps.",
  diff: "Each frame minus the first. Unchanged sky cancels to grey. A mover leaves a dark spot where it was and a bright spot where it went; a variable star leaves a single spot. Bright stars can leave faint rings.",
  sum: "All frames added together: the deepest view, for finding faint objects.",
  sim: "A simulation, not an observation. Drag the slider or press play: each star moves in a straight line at its measured speed, so you can see the field as it was, or will be.",
};

const state = {
  q: "", name: null, target: null,
  ra: null, dec: null, fov: DEFAULTS.fov,
  band: DEFAULTS.band, mode: DEFAULTS.mode, kind: DEFAULTS.kind, speed: DEFAULTS.speed,
  mask: true, gaia: true, track: true, flags: false, invert: false, smooth: true, match: true, wise: false, names: true,
  black: 35, white: 38,
  sra: null, sdec: null,
  embed: false,
};

let config = null;
let viewer = null;
let session = 0;            // bumped for every new view; stale async work checks it
let status = null;
let playing = false;
let userPaused = false;
let playTimer = null;
let lastSingleBand = "2";
let noticeAction = null;
const sim = { span: 1000, offset: 0, running: false, started: 0, frame: 0 };

const frameCache = new Map();
let framesInFlight = null;
let lastPartial = { key: null, count: 0, time: 0 };
let shown = null;
const wiseCache = new Map();
let wiseInFlight = null;
const spectrum = { key: null, result: null, final: false, inFlight: false, lastCount: 0, lastTime: 0, tableOpen: false };
// What the catalogues list in this field, and whether each has answered yet.
const CATALOGUE_NAMES = { simbad: "SIMBAD", gaia: "Gaia DR3", twomass: "2MASS", allwise: "AllWISE" };
const catalogues = { simbad: [], twomass: [], allwise: [], gaia: [], status: {} };
// The automatic search of the field: idle | loading | done | error.
const finder = { status: "idle", report: null, error: null };

// --------------------------------------------------------------------------- helpers

function notice(message, { error = false, actionLabel = null, action = null } = {}) {
  const box = $("notice");
  if (!message) { box.hidden = true; noticeAction = null; return; }
  $("notice-text").textContent = message;
  box.classList.toggle("error", error);
  const button = $("notice-action");
  button.hidden = !actionLabel;
  button.textContent = actionLabel || "";
  noticeAction = action;
  box.hidden = false;
}

function stageMessage(title, body) {
  const box = $("stage-msg");
  if (!title) { box.hidden = true; return; }
  const inner = $("stage-msg-inner");
  inner.replaceChildren();
  const t = document.createElement("p");
  t.className = "stage-title";
  t.textContent = title;
  inner.appendChild(t);
  if (body) {
    const b = document.createElement("p");
    b.textContent = body;
    inner.appendChild(b);
  }
  box.hidden = false;
}

function viewParams(extra = {}) {
  return { ra: state.ra, dec: state.dec, fov: state.fov, ...extra };
}

const needsAllBands = (band) => band === "colour" || band === "all";

/** The band actually on screen: a single band stands in until Colour or All can be built. */
function effectiveBand() {
  if (needsAllBands(state.band) && (!status || status.stage !== "done")) return FALLBACK_BAND;
  return state.band;
}

function framesKey() {
  return `${effectiveBand()}|${state.kind}|${state.mask ? 1 : 0}`;
}

function wiseBand() {
  return ["1", "2", "3", "4"].includes(state.band) ? "W1" : "W2";
}

function selection() {
  if (state.ra == null) return null;
  return { ra: state.sra ?? state.ra, dec: state.sdec ?? state.dec };
}

function setPressed(container, attr, value) {
  for (const button of container.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset[attr] === String(value)));
  }
}

// --------------------------------------------------------------------------- URL

function readUrl() {
  const p = new URLSearchParams(location.search);
  const num = (key) => (p.has(key) && p.get(key) !== "" && Number.isFinite(Number(p.get(key))) ? Number(p.get(key)) : null);
  state.q = p.get("q") || "";
  state.ra = num("ra");
  state.dec = num("dec");
  if (state.ra == null || state.dec == null) { state.ra = null; state.dec = null; }
  state.fov = num("fov") ?? DEFAULTS.fov;
  const band = (p.get("band") || "").toLowerCase().replace("color", "colour");
  state.band = /^(colour|all|[1-6])$/.test(band) ? band : DEFAULTS.band;
  state.mode = ["blink", "diff", "sum", "sim"].includes(p.get("mode")) ? p.get("mode") : DEFAULTS.mode;
  state.kind = p.get("frames") === "exposures" ? "exposures" : "epochs";
  state.speed = num("speed") ?? DEFAULTS.speed;
  state.sra = num("sra");
  state.sdec = num("sdec");
  state.embed = p.get("embed") === "1";
  state.gaia = p.get("gaia") !== "0";
  state.mask = p.get("mask") !== "0";
  state.wise = p.get("wise") === "1";
  state.names = p.get("names") !== "0";
  state.invert = p.get("invert") === "1";
  state.flags = p.get("flags") === "1";
  if (state.kind === "exposures" && needsAllBands(state.band)) state.band = lastSingleBand;
  if (!needsAllBands(state.band)) lastSingleBand = state.band;
}

function buildUrl(extra = {}) {
  const p = new URLSearchParams();
  if (state.ra != null) { p.set("ra", state.ra); p.set("dec", state.dec); }
  if (state.q) p.set("q", state.q);
  if (state.fov !== DEFAULTS.fov) p.set("fov", state.fov);
  if (state.band !== DEFAULTS.band) p.set("band", state.band);
  if (state.mode !== DEFAULTS.mode) p.set("mode", state.mode);
  if (state.kind !== DEFAULTS.kind) p.set("frames", state.kind);
  if (state.speed !== DEFAULTS.speed) p.set("speed", state.speed);
  if (state.sra != null && state.sdec != null) { p.set("sra", state.sra); p.set("sdec", state.sdec); }
  if (state.wise) p.set("wise", "1");
  if (!state.gaia) p.set("gaia", "0");
  if (!state.names) p.set("names", "0");
  if (!state.mask) p.set("mask", "0");
  if (state.invert) p.set("invert", "1");
  if (state.flags) p.set("flags", "1");
  if (state.embed || extra.embed) p.set("embed", "1");
  const qs = p.toString();
  return location.origin + location.pathname + (qs ? "?" + qs : "");
}

function writeUrl() {
  history.replaceState(null, "", buildUrl());
}

// --------------------------------------------------------------------------- loading

async function submitQuery(text, { keepNotice = false } = {}) {
  const query = text.trim();
  if (!query) return;
  if (!keepNotice) notice(null);
  let resolved;
  try {
    resolved = await getJSON("api/resolve", { q: query });
  } catch (err) {
    notice(err.message, { error: true });
    return;
  }
  state.q = resolved.kind === "name" ? query : "";
  state.target = resolved.kind === "name" ? resolved : null;
  state.name = resolved.name;
  state.ra = round5(resolved.ra);
  state.dec = round5(resolved.dec);
  state.sra = null;
  state.sdec = null;
  startView();
  const s = resolved.suggest;
  if (s) {
    // The coordinates are an old position of something that has since moved away.
    notice(
      `These coordinates are where ${s.name} was around ${s.match_year}. It has since moved ${(s.moved_arcsec / 60).toFixed(1)} arcminutes.`,
      { actionLabel: `Centre on ${s.name} today`, action: () => { $("search-input").value = s.name; submitQuery(s.name); } },
    );
  }
}

function startView() {
  session += 1;
  const my = session;
  status = null;
  frameCache.clear();
  wiseCache.clear();
  framesInFlight = null;
  wiseInFlight = null;
  lastPartial = { key: null, count: 0, time: 0 };
  shown = null;
  Object.assign(spectrum, { key: null, result: null, final: false, inFlight: false, lastCount: 0, lastTime: 0 });
  setPlaying(false);
  userPaused = false;
  stopSimulation();
  if (state.mode === "sim") resetSimulation();

  Object.assign(catalogues, {
    simbad: [], twomass: [], allwise: [], gaia: [],
    status: { simbad: "loading", gaia: "loading", twomass: "loading", allwise: "loading" },
  });
  Object.assign(finder, { status: "idle", report: null, error: null });
  viewer.clear();
  viewer.setCandidates([]);
  renderFinder();
  viewer.setGaia([]);
  viewer.setCatalogues(catalogues);
  viewer.setTarget(state.target);
  viewer.setSelection(selection());
  renderHeader();
  renderSource();
  $("passes-box").hidden = true;
  $("timeline").replaceChildren();
  $("caption").textContent = "";
  $("hud-stamp").hidden = true;
  $("hud-zoom").hidden = true;
  stageMessage("Searching the SPHEREx archive…", "Finding every image NASA has released of this spot.");
  $("progress").hidden = false;
  $("progress-fill").style.width = "0%";
  $("progress-text").textContent = "Searching…";
  writeUrl();
  markExample();
  renderBandButtons();

  loadGaia(my);
  loadCatalogues(my);
  ensureWise();
  poll(my);
}

async function loadGaia(my, attempt = 0) {
  catalogues.status.gaia = "loading";
  try {
    const result = await getJSON("api/gaia", viewParams());
    if (session !== my) return;
    catalogues.gaia = result.stars;
    catalogues.status.gaia = "ready";
    viewer.setGaia(result.stars);
    renderIdentity();
    // A spectrum measured before the catalogue arrived could not check for blended
    // neighbours; measure again now that it can.
    if (spectrum.result && !spectrum.result.blend_checked) {
      Object.assign(spectrum, { final: false, key: null, lastCount: 0, lastTime: 0 });
      ensureSpectrum();
    }
  } catch {
    // The overlay is optional and the viewer works without it, but the catalogue
    // services are sometimes busy, so try once more a little later.
    if (attempt === 0 && session === my) {
      await sleep(20000);
      if (session === my) loadGaia(my, 1);
    } else if (session === my) {
      catalogues.status.gaia = "failed";
      renderIdentity();
    }
  }
}

/** Names and types of what is already catalogued in this field (three services, each in its own time). */
function loadCatalogues(my) {
  for (const source of ["simbad", "twomass", "allwise"]) {
    catalogues.status[source] = "loading";
    getJSON("api/catalogue", viewParams({ source })).then((result) => {
      if (session !== my) return;
      catalogues[source] = result.objects;
      catalogues.status[source] = "ready";
      viewer.setCatalogues(catalogues);
      renderIdentity();
    }).catch(() => {
      if (session !== my) return;
      catalogues.status[source] = "failed";
      renderIdentity();
    });
  }
}

async function ensureWise() {
  viewer.setOption("showExtra", state.wise);
  if (!state.wise || state.ra == null) return;
  const band = wiseBand();
  const key = `${session}|${band}`;
  if (wiseCache.has(key)) { viewer.setExtra(wiseCache.get(key)); afterFramesChange(); return; }
  if (wiseInFlight === key) return;
  wiseInFlight = key;
  const my = session;
  try {
    const result = await getFrames(viewParams({ band }), null, "api/wise");
    if (session !== my) return;
    wiseCache.set(key, result);
    if (state.wise && wiseBand() === band) { viewer.setExtra(result); afterFramesChange(); }
  } catch (err) {
    if (session === my) notice(`The 2010 WISE image could not be loaded: ${err.message}`, { error: true });
  } finally {
    if (wiseInFlight === key) wiseInFlight = null;
  }
}

async function poll(my) {
  let failures = 0;
  while (session === my) {
    try {
      const priority = needsAllBands(state.band) ? FALLBACK_BAND : state.band;
      status = await getJSON("api/status", viewParams({ band: priority }));
      failures = 0;
    } catch (err) {
      failures += 1;
      if (failures >= 4) {
        if (session === my) { notice(`Lost contact with the server: ${err.message}`, { error: true }); $("progress").hidden = true; }
        return;
      }
      await sleep(1500);
      continue;
    }
    if (session !== my) return;
    renderProgress();
    if (status.stage === "error") {
      notice(`The archive could not be read just now. ${status.error || ""} Try again in a minute.`, { error: true });
      stageMessage("Could not load this field", "NASA's archive did not answer. Please try again shortly.");
      return;
    }
    if (status.stage !== "searching" && status.total === 0) {
      stageMessage("No SPHEREx images here yet", "The archive has no public exposures covering this position. New data arrives weekly.");
      return;
    }
    ensureFrames();
    ensureSpectrum();
    if (status.stage === "done") { ensureCandidates(my); return; }
    await sleep(700);
  }
}

/** Once every image is in, search the field for anything that moved, changed, or is unaccounted for. */
async function ensureCandidates(my) {
  if (finder.status !== "idle" || document.body.classList.contains("embed")) return;
  finder.status = "loading";
  renderFinder();
  try {
    const report = await getJSON("api/candidates", viewParams());
    if (session !== my) return;
    Object.assign(finder, { status: "done", report });
    viewer.setCandidates(report.candidates);
  } catch (err) {
    if (session !== my) return;
    Object.assign(finder, { status: "error", error: err.message });
  }
  renderFinder();
}

const VERDICT_LABEL = { unexplained: "Unexplained", known: "Already known", doubtful: "Doubtful" };

function renderFinder() {
  const card = $("finder-card");
  card.hidden = state.ra == null || finder.status === "idle";
  card.classList.toggle("loading", finder.status === "loading");
  const list = $("finder-list"), notes = $("finder-notes"), tally = $("finder-tally"), method = $("finder-method");
  list.replaceChildren(); notes.replaceChildren();
  tally.hidden = true; method.hidden = true;
  if (finder.status === "loading") {
    $("finder-sub").textContent = "Measuring every source in every survey pass, then checking Gaia, SIMBAD, 2MASS and AllWISE. A new field can take a minute.";
    return;
  }
  if (finder.status === "error") {
    $("finder-sub").textContent = `The search could not be completed: ${finder.error}`;
    return;
  }
  if (finder.status !== "done") return;
  const { candidates, counts, field } = finder.report;
  const span = field.baseline_yr >= 0.1 ? ` spanning ${field.baseline_yr.toFixed(1)} years` : "";
  const measured = `Measured ${counts.sources} source${counts.sources === 1 ? "" : "s"} in ${field.passes} survey pass${field.passes === 1 ? "" : "es"}${span}.`;
  $("finder-sub").textContent = candidates.length
    ? `${measured} Flagged by measurement alone; the catalogues were consulted only afterwards.`
    : `${measured} Nothing moved by ${"0.8″"} a year or more, nothing changed brightness by 15% or more, and nothing is missing from, or new to, the catalogues.`;
  if (candidates.length) {
    tally.hidden = false;
    tally.replaceChildren(...["unexplained", "known", "doubtful"].filter((v) => counts[v] > 0).map((v) => {
      const chip = document.createElement("span");
      chip.className = v;
      const n = document.createElement("b");
      n.textContent = counts[v];
      chip.append(n, VERDICT_LABEL[v].toLowerCase());
      return chip;
    }));
  }
  for (const c of candidates) {
    const li = document.createElement("li");
    li.className = "finding";
    li.dataset.verdict = c.verdict;
    const num = document.createElement("span");
    num.className = "num"; num.textContent = c.id;
    const title = document.createElement("h3");
    title.textContent = c.title;
    const who = document.createElement("p");
    who.className = "who";
    who.textContent = c.known?.name || c.name || "No catalogue entry";
    const body = document.createElement("div");
    body.className = "body";
    for (const fact of c.facts) {
      const para = document.createElement("p");
      para.textContent = fact;
      body.append(para);
    }
    const verdict = document.createElement("p");
    verdict.className = "verdict";
    const label = document.createElement("b");
    label.textContent = `${VERDICT_LABEL[c.verdict]}. `;
    const rest = c.explanation.replace(/^Already known: /, "");
    verdict.append(label, rest.charAt(0).toUpperCase() + rest.slice(1));
    body.append(verdict);
    for (const caution of c.caveats) {
      const para = document.createElement("p");
      para.className = "caution";
      para.textContent = `Caution: ${caution}`;
      body.append(para);
    }
    const actions = document.createElement("div");
    actions.className = "actions";
    const show = document.createElement("button");
    show.type = "button";
    show.textContent = c.kind === "gone" ? "Show the spot" : "Show and measure";
    show.addEventListener("click", () => { selectSky(c.ra, c.dec); $("stage").scrollIntoView({ behavior: "smooth", block: "center" }); });
    actions.append(show);
    if (c.kind === "gone" || c.kind === "drifted" || c.kind === "uncatalogued") {
      const compare = document.createElement("button");
      compare.type = "button";
      compare.textContent = "Blink against 2010";
      compare.addEventListener("click", () => {
        selectSky(c.ra, c.dec);
        state.wise = true; $("wise-toggle").checked = true;
        if (state.mode !== "blink") setMode("blink");
        ensureWise(); afterFramesChange(); writeUrl();
        setPlaying(true);
        $("stage").scrollIntoView({ behavior: "smooth", block: "center" });
      });
      actions.append(compare);
    }
    if (c.known?.now_ra != null) {
      const now = document.createElement("button");
      now.type = "button";
      now.textContent = "Show where it is now";
      now.addEventListener("click", () => { selectSky(c.known.now_ra, c.known.now_dec); $("stage").scrollIntoView({ behavior: "smooth", block: "center" }); });
      actions.append(now);
    }
    body.append(actions);
    li.append(num, title, who, body);
    list.append(li);
  }
  for (const text of finder.report.notes) {
    const li = document.createElement("li");
    li.textContent = text;
    notes.append(li);
  }
  method.hidden = false;
  const unconfirmed = counts.not_confirmed > 0
    ? ` ${counts.not_confirmed} first-look flag${counts.not_confirmed === 1 ? "" : "s"} did not survive the exposure-by-exposure check and ${counts.not_confirmed === 1 ? "is" : "are"} not shown.`
    : "";
  method.textContent = `${finder.report.method}${unconfirmed}`;
}

/** Select a sky position as if it had been clicked: mark it, identify it, measure it. */
function selectSky(ra, dec) {
  state.sra = round5(ra);
  state.sdec = round5(dec);
  viewer.setSelection(selection());
  Object.assign(spectrum, { key: null, result: null, final: false, lastCount: 0, lastTime: 0 });
  renderSource();
  $("source-sub").textContent = "Measuring…";
  writeUrl();
  ensureSpectrum();
}

function bandReady(band) {
  if (!status) return false;
  return needsAllBands(band) ? status.stage === "done" : status.bands_ready.includes(Number(band));
}

function ensureFrames() {
  if (!status || state.ra == null) return;
  const key = framesKey();
  const band = effectiveBand();
  if (frameCache.has(key)) { showFrames(frameCache.get(key), key); return; }
  if (framesInFlight) return;
  if (bandReady(band)) { fetchFrames(key, false, Infinity); return; }
  const loaded = status.loaded_per_band[band] || 0;
  const now = performance.now();
  const fresh = lastPartial.key !== key;
  if (loaded > 0 && (fresh || (loaded > lastPartial.count && now - lastPartial.time > 3500))) {
    fetchFrames(key, true, loaded);
  }
}

async function fetchFrames(key, partial, loaded) {
  const my = session;
  framesInFlight = key;
  const [band, kind, mask] = key.split("|");
  let ok = false;
  try {
    const result = await getFrames(viewParams({ band, kind, mask, partial: partial ? 1 : 0 }));
    if (session !== my) return;
    if (result.header.complete) frameCache.set(key, result);
    lastPartial = { key, count: loaded, time: performance.now() };
    if (framesKey() === key) showFrames(result, key);
    ok = true;
  } catch (err) {
    if (session === my) notice(`Could not build the images: ${err.message}`, { error: true });
  } finally {
    if (framesInFlight === key) framesInFlight = null;
    // The view may have changed while this was loading; pick up the new request.
    if (ok && session === my && framesKey() !== key) ensureFrames();
  }
}

function showFrames(result, key) {
  const tag = `${session}|${key}|${result.header.complete ? "final" : result.header.n_exposures}`;
  if (shown === tag) return;
  shown = tag;
  viewer.setData(result);
  if (result.header.nframes === 0) {
    stageMessage("No usable images in this view", "Try another band, or All.");
  } else {
    stageMessage(null);
  }
  afterFramesChange();
}

/** Refresh everything that depends on the list of frames on screen. */
function afterFramesChange() {
  renderTimeline();
  renderFrame();
  renderBandButtons();
  const has = viewer.frameCount > 0;
  const simulating = state.mode === "sim";
  $("hud-stamp").hidden = !has;
  $("hud-zoom").hidden = !has;
  $("transport").hidden = !has;
  $("blinkbar").hidden = simulating;
  $("simbar").hidden = !simulating;
  $("sim-badge").hidden = !(has && simulating);
  $("stage").classList.toggle("sim", simulating);
  viewer.bottomInset = has ? 58 : 0;
  viewer.drawOverlay();
  if (simulating) {
    setPlaying(false);
    return;
  }
  stopSimulation();
  if (!userPaused && viewer.frameCount >= 2 && !matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(true);
  if (viewer.frameCount < 2) setPlaying(false);
}

// ---- simulation playback: the clock swings smoothly between -span and +span years

function setSimOffset(years, fromSlider = false) {
  sim.offset = Math.max(-sim.span, Math.min(sim.span, years));
  if (!fromSlider) $("sim-slider").value = String(Math.round(sim.offset));
  viewer.setSimOffset(sim.offset);
}

/** Put the clock back to today and pick a time span that suits the target. */
function resetSimulation() {
  // A fast-moving target crosses the frame in decades, so start with a shorter span for it.
  const t = state.target;
  const fast = t && Math.hypot(t.pmra || 0, t.pmdec || 0) >= 1000;
  sim.span = fast ? 100 : 1000;
  const slider = $("sim-slider");
  $("sim-span").value = String(sim.span);
  slider.min = String(-sim.span); slider.max = String(sim.span); slider.step = "1";
  sim.offset = 0; slider.value = "0";
  viewer.simOffset = 0;
}

function stopSimulation() {
  sim.running = false;
  cancelAnimationFrame(sim.frame);
  if (state.mode === "sim") showPlayIcon(false);
}

function startSimulation() {
  if (state.mode !== "sim" || !viewer.frameCount) return;
  sim.running = true;
  showPlayIcon(true);
  // Start the swing from wherever the slider is now.
  const period = 16000;
  const phase = Math.asin(Math.max(-1, Math.min(1, sim.offset / sim.span)));
  sim.started = performance.now() - (phase / (2 * Math.PI)) * period;
  const step = (now) => {
    if (!sim.running) return;
    setSimOffset(sim.span * Math.sin(((now - sim.started) / period) * 2 * Math.PI));
    sim.frame = requestAnimationFrame(step);
  };
  sim.frame = requestAnimationFrame(step);
}

function showPlayIcon(isPlaying) {
  // SVG elements have no .hidden property, so toggle the attribute itself.
  $("icon-play").toggleAttribute("hidden", isPlaying);
  $("icon-pause").toggleAttribute("hidden", !isPlaying);
  $("btn-play").setAttribute("aria-label", isPlaying ? "Pause" : "Play");
}

function ensureSpectrum() {
  const sel = selection();
  if (!status || !sel) return;
  const key = `${sel.ra},${sel.dec}`;
  if (spectrum.key === key && spectrum.final) return;
  if (spectrum.inFlight) return;
  if (status.stage === "done") { fetchSpectrum(key, sel, false); return; }
  const now = performance.now();
  const fresh = spectrum.key !== key;
  if (status.loaded > 0 && (fresh || (status.loaded >= spectrum.lastCount + 8 && now - spectrum.lastTime > 6000))) {
    fetchSpectrum(key, sel, true);
  }
}

async function fetchSpectrum(key, sel, partial) {
  const my = session;
  spectrum.inFlight = true;
  const loadedAtRequest = status ? status.loaded : 0;
  // The default selection of a named, moving target is measured along its track.
  // (Only worth doing above 0.1 arcsec per year; slower than that it never leaves a pixel.)
  const t = state.target;
  const motion = t && state.sra == null && Math.hypot(t.pmra || 0, t.pmdec || 0) >= 100
    ? { pmra: t.pmra || 0, pmdec: t.pmdec || 0, epoch: t.epoch }
    : {};
  try {
    const result = await getJSON("api/spectrum", viewParams({ sra: sel.ra, sdec: sel.dec, partial: partial ? 1 : 0, ...motion }));
    if (session !== my) return;
    const current = selection();
    if (!current || `${current.ra},${current.dec}` !== key) return;
    Object.assign(spectrum, { key, result, final: Boolean(result.complete), lastCount: loadedAtRequest, lastTime: performance.now() });
    const measured = result.motion && result.motion.significant
      ? { ra: sel.ra, dec: sel.dec, pmra: result.motion.pm_east * 1000, pmdec: result.motion.pm_north * 1000, epoch: config.view_epoch }
      : null;
    viewer.setMeasuredMotion(measured);
    renderSource();
  } catch (err) {
    if (session === my) $("source-sub").textContent = `Could not measure this position: ${err.message}`;
  } finally {
    spectrum.inFlight = false;
    if (session === my) {
      const current = selection();
      if (current && `${current.ra},${current.dec}` !== key) ensureSpectrum();
    }
  }
}

// --------------------------------------------------------------------------- rendering

function renderHeader() {
  const name = $("target-name"), coords = $("target-coords");
  if (state.ra == null) return;
  name.textContent = state.name || "Sky position";
  let line = `${formatCoords(state.ra, state.dec)}  ·  RA ${state.ra.toFixed(5)}°  Dec ${state.dec >= 0 ? "+" : ""}${state.dec.toFixed(5)}°`;
  const t = state.target;
  if (t && t.moved_arcsec > 5) {
    line += `  ·  moved ${Math.round(t.moved_arcsec)}″ since 2000; centred on where it is in ${Math.round(t.epoch)}`;
  }
  coords.textContent = line;
  document.title = `${state.name || "Sky position"} · SPHEREx Blink`;
}

function renderProgress() {
  const box = $("progress");
  if (!status) return;
  if (status.stage === "searching") {
    box.hidden = false;
    $("progress-text").textContent = "Searching the archive…";
    return;
  }
  renderPasses();
  renderBandButtons();
  if (status.stage === "fetching") {
    box.hidden = false;
    const pct = status.total ? (100 * status.done) / status.total : 0;
    $("progress-fill").style.width = pct.toFixed(1) + "%";
    // A field seen before is read back from the server's cache, not downloaded again.
    const saved = status.from_cache > 0 && status.from_cache >= status.done;
    $("progress-text").textContent = saved
      ? `Opening saved images: ${status.done} of ${status.total}`
      : `Downloading from NASA's archive: ${status.done} of ${status.total} exposures`;
    if (!viewer.frameCount) {
      stageMessage(saved ? "Opening SPHEREx images…" : "Downloading SPHEREx images…",
        `${status.total} exposures cover this spot across ${status.epochs.length} survey pass${status.epochs.length === 1 ? "" : "es"}.` +
        (saved ? "" : " The first view appears within seconds and sharpens as more arrive."));
    }
  } else {
    box.hidden = true;
    // Everything has arrived; the server is aligning and combining it.
    if (!viewer.frameCount && status.total > 0) {
      stageMessage("Building the view…", `Aligning and combining ${status.loaded} exposures.`);
    }
  }
}

function renderPasses() {
  const list = $("passes-list");
  list.replaceChildren();
  if (!status || !status.epochs.length) { $("passes-box").hidden = true; return; }
  for (const ep of status.epochs) {
    const li = document.createElement("li");
    const when = document.createElement("span");
    when.textContent = ep.date_min === ep.date_max ? prettyDate(ep.date_min) : `${prettyDate(ep.date_min)} – ${prettyDate(ep.date_max)}`;
    const count = document.createElement("span");
    count.className = "n";
    count.textContent = `${ep.n} exp.`;
    li.append(when, count);
    list.appendChild(li);
  }
  const notes = [];
  if (status.n_available > status.total) notes.push(`Showing an even sample of ${status.total} from ${status.n_available} exposures.`);
  if (status.stage === "done" && status.failed > 0) notes.push(`${status.failed} exposure${status.failed === 1 ? "" : "s"} could not be loaded.`);
  if (notes.length) {
    const li = document.createElement("li");
    li.className = "n";
    li.textContent = notes.join(" ");
    list.appendChild(li);
  }
  $("passes-box").hidden = false;
}

function renderBandButtons() {
  if (!config) return;
  const box = $("band-buttons");
  setPressed(box, "band", state.band);
  const range = needsAllBands(state.band) ? { min_um: 0.75, max_um: 5.0 } : config.detectors[state.band];
  $("band-caption").textContent = `· ${range.min_um.toFixed(2)}–${range.max_um.toFixed(2)} µm`;
  for (const button of box.querySelectorAll("button")) {
    const b = button.dataset.band;
    const ready = !status ? true : bandReady(b);
    button.style.opacity = ready ? "1" : "0.6";
  }
  const waiting = needsAllBands(state.band) && status && status.stage !== "done" && state.ra != null;
  let hint;
  if (waiting) {
    hint = `${state.band === "colour" ? "Colour" : "All"} needs all six bands. Showing band ${FALLBACK_BAND} while they arrive.`;
  } else if (state.band === "colour") {
    hint = "False colour from all six bands. Ordinary stars look white; cold objects look red.";
  } else if (state.band === "all") {
    hint = "All six bands added together: the deepest image.";
  } else {
    hint = `Infrared band ${state.band} only.`;
  }
  $("band-hint").textContent = hint;
}

function renderTimeline() {
  const box = $("timeline");
  box.replaceChildren();
  const count = viewer.frameCount;
  box.classList.toggle("dense", count > 14);
  for (let i = 0; i < count; i++) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = viewer.shortLabel(i);
    button.title = viewer.frameLabel(i);
    if (viewer.display[i].meta.source) button.classList.add("foreign");
    button.addEventListener("click", () => { userPaused = true; setPlaying(false); viewer.setIndex(i); });
    box.appendChild(button);
  }
  highlightTimeline();
}

function highlightTimeline() {
  const buttons = $("timeline").children;
  for (let i = 0; i < buttons.length; i++) buttons[i].setAttribute("aria-current", String(i === viewer.index));
}

function renderFrame() {
  highlightTimeline();
  const caption = $("caption");
  caption.replaceChildren();
  if (!viewer.frameCount) return;
  const stamp = viewer.stamp();
  $("stamp-main").textContent = stamp.main;
  $("stamp-sub").textContent = stamp.sub;
  caption.append(document.createTextNode(viewer.frameLabel()));
  const frame = viewer.current;
  if (frame && frame.rgb && !frame.diff && viewer.channelRanges) {
    const colours = ["#ff5a4d", "#58d68d", "#5b8dff"];
    viewer.channelRanges.forEach((range, i) => {
      const key = document.createElement("span");
      key.className = "key";
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.style.background = colours[i];
      key.append(chip, document.createTextNode(range));
      caption.appendChild(key);
    });
  }
}

function stat(parent, label, value, sub, warn = false) {
  const tile = document.createElement("div");
  tile.className = "stat" + (warn ? " warn" : "");
  const l = document.createElement("div"); l.className = "label"; l.textContent = label;
  const v = document.createElement("div"); v.className = "value"; v.textContent = value;
  tile.append(l, v);
  if (sub) { const s = document.createElement("div"); s.className = "sub"; s.textContent = sub; tile.appendChild(s); }
  parent.appendChild(tile);
}

/** "What is it?": everything the catalogues list at the selected position. */
function renderIdentity() {
  const box = $("identity");
  const sel = selection();
  if (!sel || state.ra == null || !config) { box.hidden = true; return; }
  const matches = identify(sel, config.view_epoch, catalogues, catalogues.gaia);
  const keys = Object.keys(CATALOGUE_NAMES);
  const waiting = keys.filter((k) => catalogues.status[k] === "loading").map((k) => CATALOGUE_NAMES[k]);
  const failed = keys.filter((k) => catalogues.status[k] === "failed").map((k) => CATALOGUE_NAMES[k]);
  const checked = keys.filter((k) => catalogues.status[k] === "ready").map((k) => CATALOGUE_NAMES[k]);
  const name = $("identity-name"), kind = $("identity-kind"), list = $("identity-list"), note = $("identity-note");
  list.replaceChildren();
  const best = matches[0];
  const speed = (m) => (m.motion >= 100 ? `moves ${(m.motion / 1000).toFixed(2)}″ per year` : null);
  box.classList.toggle("unknown", !best && !waiting.length);
  if (best && best.catalogue === "SIMBAD") {
    const survey = isSurveyId(best.id);
    // The object you searched for keeps the name you know it by.
    const searched = state.sra == null && state.target && best.sep < 4 ? state.target.name : null;
    name.textContent = searched || (survey ? best.kind : best.name);
    kind.textContent = [searched || survey ? best.kind : null, survey || searched ? best.id : best.kind, best.detail, speed(best)]
      .filter((v, i, all) => v && all.indexOf(v) === i).join(" · ");
  } else if (best && best.catalogue === "Gaia DR3") {
    name.textContent = "A catalogued star";
    kind.textContent = [best.id, best.detail, speed(best)].filter(Boolean).join(" · ");
  } else if (best) {
    name.textContent = "A catalogued infrared source";
    kind.textContent = [best.id, best.detail].filter(Boolean).join(" · ");
  } else if (waiting.length) {
    name.textContent = "Checking the catalogues…";
    kind.textContent = "";
  } else {
    name.textContent = "Nothing catalogued here";
    kind.textContent = "";
  }
  for (const m of matches) {
    const li = document.createElement("li");
    const cat = document.createElement("span"); cat.className = "cat"; cat.textContent = m.catalogue;
    const what = document.createElement("span"); what.className = "what";
    if (m.catalogue === "SIMBAD") {
      const link = document.createElement("a");
      link.href = `https://simbad.harvard.edu/simbad/sim-id?Ident=${encodeURIComponent(m.id)}`;
      link.target = "_blank"; link.rel = "noopener";
      link.textContent = m.id;
      what.append(link, document.createTextNode(` · ${m.kind}`));
    } else {
      what.textContent = [m.id, m.detail].filter(Boolean).join(" · ");
    }
    const sep = document.createElement("span"); sep.className = "sep"; sep.textContent = `${m.sep.toFixed(1)}″ away`;
    li.append(cat, what, sep);
    list.appendChild(li);
  }
  const parts = [];
  if (!best && !waiting.length) {
    parts.push(`No object within ${MATCH_RADIUS_ARCSEC}″ in ${checked.join(", ") || "any catalogue"}. That does not make it new: it could be an image artefact, two blended stars, or simply too faint for those surveys. Check that it appears in every pass, turn on “Show flagged pixels”, and compare with 2010.`);
  }
  if (waiting.length) parts.push(`Still checking: ${waiting.join(", ")}.`);
  if (failed.length) parts.push(`Could not reach: ${failed.join(", ")}.`);
  note.replaceChildren(document.createTextNode(parts.join(" ") + (parts.length ? " " : "")));
  const search = document.createElement("a");
  const decText = `${sel.dec >= 0 ? "%2B" : ""}${sel.dec.toFixed(5)}`;
  search.href = `https://simbad.harvard.edu/simbad/sim-coo?Coord=${sel.ra.toFixed(5)}+${decText}&CooFrame=ICRS&Radius=20&Radius.unit=arcsec`;
  search.target = "_blank"; search.rel = "noopener";
  search.textContent = "Search this position in SIMBAD";
  note.appendChild(search);
  box.hidden = false;
}

function renderSource() {
  renderIdentity();
  const result = spectrum.result;
  const sel = selection();
  const has = Boolean(result && sel);
  $("charts").hidden = !has;
  $("source-stats").hidden = !has;
  $("source-actions").hidden = !has;
  $("source-method").hidden = !has;
  $("table-wrap").hidden = !(has && spectrum.tableOpen);
  if (!has) {
    $("source-sub").textContent = state.ra == null
      ? "Click any object in the image to measure it at every wavelength and date."
      : "Measuring the object at the centre… click any other object in the image to measure it instead.";
    return;
  }
  const loading = !result.complete;
  $("source-sub").textContent = `Measured at ${formatCoords(sel.ra, sel.dec)}` +
    (loading ? ` · still downloading (${result.n_exposures} exposures so far)` : ` · ${result.n_points} exposures`) +
    " · click another object to measure it";

  const stats = $("source-stats");
  stats.replaceChildren();
  const trusted = result.n_points - result.n_flagged;
  const mostlyFlagged = result.n_points > 0 && result.n_flagged > 0.5 * result.n_points;
  const saturated = result.n_points > 0 && result.n_saturated > 0.8 * result.n_points;
  const usesWings = result.light_curve.some((p) => p.method === "wings");
  if (saturated) {
    stat(stats, "Data quality", "Saturated", `Too bright for SPHEREx: the core overflows the detector in ${result.n_saturated} of ${result.n_points} exposures. Its light curve uses the unsaturated outer glow instead.`, true);
  } else if (mostlyFlagged) {
    stat(stats, "Data quality", "Partly saturated", `Flagged in ${result.n_flagged} of ${result.n_points} exposures. Grey rings in the spectrum are unreliable.`, true);
  } else if (trusted > 0) {
    const fraction = result.n_detected / trusted;
    stat(stats, "Detected", fraction >= 0.15 ? `${result.n_detected} of ${trusted}` : "Not clearly",
      fraction >= 0.15 ? "exposures at 3σ or better" : `Only ${result.n_detected} of ${trusted} exposures reach 3σ`);
  }
  const blend = result.blend;
  if (blend) {
    stat(stats, "Blended with a neighbour", `${blend.n} of ${result.n_points} exposures`,
      `A catalogued star${blend.g != null ? ` (Gaia G ${blend.g.toFixed(1)})` : ""} comes within ${blend.closest_arcsec.toFixed(1)}″ of the measuring spot, closer than SPHEREx can separate. Open rings in the spectrum include its light and are left out of the light curve.`, true);
  }
  const clean = result.points.filter((p) => !p.flagged && !p.blended && p.snr >= 3);
  const usable = clean.length >= 5 ? clean : result.points.filter((p) => !p.flagged && p.snr >= 3);
  if (usable.length >= 5) {
    const top = [...usable].sort((a, b) => b.flux - a.flux).slice(0, Math.max(3, Math.round(usable.length * 0.08)));
    const wl = top.map((p) => p.wl).sort((a, b) => a - b)[Math.floor(top.length / 2)];
    const peak = top.map((p) => p.flux).sort((a, b) => a - b)[Math.floor(top.length / 2)];
    stat(stats, "Brightest near", `${wl.toFixed(2)} µm`, `about ${formatFlux(peak)}`);
    stat(stats, "Typical signal-to-noise", result.median_snr.toFixed(result.median_snr >= 10 ? 0 : 1), "per exposure");
  }
  const v = result.variability;
  const caveat = result.tracked ? " The measurement follows the moving target, so a background star it passes can mimic a change." : "";
  if (v && v.significance >= 5 && v.fraction >= 0.1) {
    stat(stats, "Change between passes", `${Math.round(v.fraction * 100)}% in band ${v.detector}`,
      `Brighter around ${prettyDate(v.bright_date)} than ${prettyDate(v.faint_date)} (${v.significance.toFixed(0)}σ).${caveat}`);
  } else if (v) {
    stat(stats, "Change between passes", "No clear change", `Largest: ${Math.round(v.fraction * 100)}% in band ${v.detector} (${v.significance.toFixed(1)}σ)`);
  }
  const mo = result.motion;
  if (mo) {
    const t = state.target;
    const catalogue = t && state.sra == null && (t.pmra || t.pmdec)
      ? ` Catalogue value: ${(Math.hypot(t.pmra || 0, t.pmdec || 0) / 1000).toFixed(2)}″ per year.` : "";
    if (mo.significant) {
      const compass = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][Math.round(mo.pa_deg / 45) % 8];
      stat(stats, "Measured motion", `${mo.total.toFixed(mo.total >= 10 ? 1 : 2)} ± ${mo.total_err.toFixed(1)}″ per year`,
        `Heading ${compass}, from its position in ${mo.passes.length} passes over ${mo.baseline_yr.toFixed(1)} years.${catalogue}`);
    } else {
      stat(stats, "Measured motion", "None detected",
        `Anything slower than about ${Math.max(1, 4 * mo.total_err).toFixed(1)}″ per year would not show over ${mo.baseline_yr.toFixed(1)} years.${catalogue}`);
    }
  }
  if (result.tracked) {
    stat(stats, "Tracking", "Following its motion", "The measuring spot moves with the target from exposure to exposure.");
  }

  if (saturated) {
    renderMessage($("spectrum-plot"), $("spectrum-legend"), "This star saturates the detector, so its spectrum cannot be measured here. See the light curve for how its brightness changes.");
  } else {
    renderSpectrum($("spectrum-plot"), $("spectrum-legend"), result, config.detectors);
  }
  renderLightCurve($("lc-plot"), $("lc-legend"), result, config.detectors);
  $("lc-sub").textContent = usesWings
    ? "Brightness over time in each band, relative to that band's average. Measured from the star's outer glow where its core saturates."
    : "Brightness over time in each band, relative to that band's average at the same wavelengths";
  if (spectrum.tableOpen) renderTable($("table-wrap"), result);
  $("source-method").textContent = `How this is measured: ${result.method} Each exposure records the object at one wavelength, so the dots trace its spectrum. For calibrated results use IRSA's SPHEREx Spectrophotometry Tool.`;
}

// --------------------------------------------------------------------------- playback

function setPlaying(value) {
  playing = Boolean(value) && viewer.frameCount >= 2 && state.mode !== "sim";
  if (state.mode !== "sim") showPlayIcon(playing);
  clearTimeout(playTimer);
  if (playing) tick();
}

function tick() {
  clearTimeout(playTimer);
  if (!playing) return;
  playTimer = setTimeout(() => {
    if (!playing) return;
    viewer.step(1);
    tick();
  }, state.speed);
}

// --------------------------------------------------------------------------- controls

function applyViewerOptions() {
  viewer.setOption("black", state.black);
  viewer.setOption("white", state.white);
  viewer.setOption("invert", state.invert);
  viewer.setOption("smooth", state.smooth);
  viewer.setOption("match", state.match);
  viewer.setOption("showFlags", state.flags);
  viewer.setOption("showGaia", state.gaia);
  viewer.setOption("showTrack", state.track);
  viewer.setOption("showExtra", state.wise);
  viewer.setOption("showNames", state.names);
  viewer.setOption("mode", state.mode);
}

function setBand(band) {
  state.band = band;
  if (!needsAllBands(band)) lastSingleBand = band;
  if (needsAllBands(band) && state.kind === "exposures") setKind("epochs", false);
  renderBandButtons();
  writeUrl();
  ensureFrames();
  ensureWise();
}

function setKind(kind, refresh = true) {
  state.kind = kind;
  if (kind === "exposures" && needsAllBands(state.band)) { state.band = lastSingleBand; renderBandButtons(); }
  setPressed($("kind-buttons"), "kind", state.kind);
  state.speed = kind === "exposures" ? Math.min(state.speed, 400) : Math.max(state.speed, 800);
  $("speed-select").value = String(state.speed);
  if (refresh) { writeUrl(); ensureFrames(); }
}

function setMode(mode) {
  state.mode = mode;
  setPressed($("mode-buttons"), "mode", mode);
  $("mode-hint").textContent = MODE_HINTS[mode];
  if (mode === "sim") resetSimulation();
  viewer.setOption("mode", mode);
  afterFramesChange();
  writeUrl();
  // Open the simulation already moving, so it is obvious what it does.
  if (mode === "sim" && !matchMedia("(prefers-reduced-motion: reduce)").matches) startSimulation();
}

function markExample() {
  for (const card of $("examples").querySelectorAll(".example")) {
    card.setAttribute("aria-current", String(Boolean(state.q) && card.dataset.query.toLowerCase() === state.q.toLowerCase()));
  }
}

function openExample(example, { announce = true } = {}) {
  state.fov = example.fov; $("fov-select").value = String(example.fov);
  state.band = example.band;
  if (!needsAllBands(example.band)) lastSingleBand = example.band;
  state.kind = "epochs"; setPressed($("kind-buttons"), "kind", "epochs");
  stopSimulation();
  state.mode = example.mode; setPressed($("mode-buttons"), "mode", example.mode);
  $("mode-hint").textContent = MODE_HINTS[example.mode];
  viewer.setOption("mode", example.mode);
  state.wise = Boolean(example.wise); $("wise-toggle").checked = state.wise;
  $("search-input").value = example.query;
  const text = `${example.name}: ${example.blurb} ${example.look_for}`;
  if (announce) notice(text);
  return submitQuery(example.query, { keepNotice: true });
}

async function copy(text, button) {
  const original = button.lastChild.textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.lastChild.textContent = " Copied";
  } catch {
    window.prompt("Copy this:", text);
  }
  setTimeout(() => { button.lastChild.textContent = original; }, 1400);
}

function wire() {
  $("search-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    state.fov = Number($("fov-select").value);
    submitQuery($("search-input").value);
  });
  $("fov-select").addEventListener("change", () => {
    state.fov = Number($("fov-select").value);
    if (state.ra != null) startView();
  });
  $("notice-close").addEventListener("click", () => notice(null));
  $("notice-action").addEventListener("click", () => { const action = noticeAction; notice(null); if (action) action(); });

  $("band-buttons").addEventListener("click", (ev) => {
    const button = ev.target.closest("button");
    if (button) setBand(button.dataset.band);
  });
  $("mode-buttons").addEventListener("click", (ev) => {
    const button = ev.target.closest("button");
    if (button) setMode(button.dataset.mode);
  });
  $("kind-buttons").addEventListener("click", (ev) => {
    const button = ev.target.closest("button");
    if (button) setKind(button.dataset.kind);
  });

  $("black-slider").addEventListener("input", (ev) => { state.black = Number(ev.target.value); viewer.setOption("black", state.black); });
  $("white-slider").addEventListener("input", (ev) => { state.white = Number(ev.target.value); viewer.setOption("white", state.white); });
  const toggle = (id, key, option, after) => $(id).addEventListener("change", (ev) => {
    state[key] = ev.target.checked;
    if (option) viewer.setOption(option, state[key]);
    writeUrl();
    if (after) after();
  });
  toggle("invert-toggle", "invert", "invert");
  toggle("smooth-toggle", "smooth", "smooth");
  toggle("match-toggle", "match", "match", afterFramesChange);
  toggle("gaia-toggle", "gaia", "showGaia");
  toggle("names-toggle", "names", "showNames");
  toggle("track-toggle", "track", "showTrack");
  toggle("flags-toggle", "flags", "showFlags");
  toggle("mask-toggle", "mask", null, ensureFrames);
  toggle("wise-toggle", "wise", null, () => { ensureWise(); afterFramesChange(); });

  $("btn-play").addEventListener("click", () => {
    if (state.mode === "sim") { if (sim.running) stopSimulation(); else startSimulation(); return; }
    userPaused = playing; setPlaying(!playing);
  });
  $("sim-slider").addEventListener("input", (ev) => { stopSimulation(); setSimOffset(Number(ev.target.value), true); });
  $("sim-span").addEventListener("change", (ev) => {
    sim.span = Number(ev.target.value);
    const slider = $("sim-slider");
    slider.min = String(-sim.span); slider.max = String(sim.span); slider.step = String(Math.max(1, sim.span / 1000));
    setSimOffset(Math.max(-sim.span, Math.min(sim.span, sim.offset)));
  });
  $("btn-sim-now").addEventListener("click", () => { stopSimulation(); setSimOffset(0); });
  $("btn-fullscreen").addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", () => viewer.draw());
  $("btn-prev").addEventListener("click", () => { userPaused = true; setPlaying(false); viewer.step(-1); });
  $("btn-next").addEventListener("click", () => { userPaused = true; setPlaying(false); viewer.step(1); });
  $("speed-select").addEventListener("change", (ev) => { state.speed = Number(ev.target.value); writeUrl(); if (playing) tick(); });
  $("btn-zoom-in").addEventListener("click", () => viewer.zoomBy(1.6));
  $("btn-zoom-out").addEventListener("click", () => viewer.zoomBy(1 / 1.6));
  $("btn-zoom-reset").addEventListener("click", () => viewer.resetView());

  $("btn-copy-link").addEventListener("click", (ev) => copy(buildUrl(), ev.currentTarget));
  $("btn-copy-embed").addEventListener("click", (ev) => copy(
    `<iframe src="${buildUrl({ embed: true })}" width="640" height="820" style="border:0" loading="lazy" title="SPHEREx Blink"></iframe>`, ev.currentTarget));

  $("btn-table").addEventListener("click", (ev) => {
    spectrum.tableOpen = !spectrum.tableOpen;
    ev.currentTarget.textContent = spectrum.tableOpen ? "Hide table" : "Show table";
    renderSource();
  });
  $("btn-csv").addEventListener("click", () => {
    if (!spectrum.result) return;
    const blob = new Blob([toCsv(spectrum.result)], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `spherex_${spectrum.result.ra.toFixed(5)}_${spectrum.result.dec.toFixed(5)}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 2000);
  });

  document.addEventListener("keydown", (ev) => {
    const tag = (ev.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag) || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.key === " " && tag !== "button" && tag !== "summary") {
      ev.preventDefault();
      if (state.mode === "sim") { if (sim.running) stopSimulation(); else startSimulation(); }
      else { userPaused = playing; setPlaying(!playing); }
    }
    else if (ev.key === "f" || ev.key === "F") toggleFullscreen();
    else if (ev.key === "ArrowRight") { userPaused = true; setPlaying(false); viewer.step(1); }
    else if (ev.key === "ArrowLeft") { userPaused = true; setPlaying(false); viewer.step(-1); }
    else if (ev.key === "+" || ev.key === "=") viewer.zoomBy(1.6);
    else if (ev.key === "-") viewer.zoomBy(1 / 1.6);
    else if (ev.key === "0") viewer.resetView();
  });
}

function toggleFullscreen() {
  const wrap = $("stage-wrap");
  if (document.fullscreenElement) document.exitFullscreen();
  else if (wrap.requestFullscreen) wrap.requestFullscreen().catch(() => {});
}

function buildStaticControls() {
  const fov = $("fov-select");
  for (const choice of config.fov_choices_arcmin) {
    const option = document.createElement("option");
    option.value = String(choice);
    option.textContent = `${choice}′ field`;
    fov.appendChild(option);
  }
  const snapped = config.fov_choices_arcmin.reduce((a, b) => (Math.abs(b - state.fov) < Math.abs(a - state.fov) ? b : a));
  state.fov = snapped;
  fov.value = String(snapped);

  const bands = $("band-buttons");
  for (const band of ["colour", "all", ...Object.keys(config.detectors)]) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.band = band;
    if (band === "colour") {
      const tri = document.createElement("span");
      tri.className = "tri";
      tri.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
      button.append(tri, document.createTextNode("Colour"));
      button.title = "False colour from all six bands: blue 0.75–1.6 µm, green 1.6–3.8 µm, red 3.8–5.0 µm";
    } else if (band === "all") {
      button.textContent = "All";
      button.title = "All six bands combined (0.75–5.0 µm): the deepest image";
    } else {
      button.textContent = band;
      button.title = `Band ${band}: ${config.detectors[band].min_um.toFixed(2)}–${config.detectors[band].max_um.toFixed(2)} µm`;
    }
    bands.appendChild(button);
  }

  const examples = $("examples");
  for (const example of config.examples) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "example";
    card.dataset.query = example.query;
    card.title = `${example.blurb} ${example.look_for}`;
    const img = document.createElement("img");
    img.alt = "";
    img.loading = "lazy";
    img.src = `assets/examples/${example.id}.jpg`;
    img.addEventListener("error", () => {
      const fallback = document.createElement("span");
      fallback.className = "thumb-fallback";
      img.replaceWith(fallback);
    });
    const text = document.createElement("span");
    text.className = "text";
    const name = document.createElement("span"); name.className = "name"; name.textContent = example.name;
    const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = example.tag;
    text.append(name, tag);
    card.append(img, text);
    card.addEventListener("click", () => openExample(example));
    examples.appendChild(card);
  }
}

function syncControls() {
  setPressed($("mode-buttons"), "mode", state.mode);
  setPressed($("kind-buttons"), "kind", state.kind);
  $("mode-hint").textContent = MODE_HINTS[state.mode];
  $("speed-select").value = String([1600, 800, 400, 180].includes(state.speed) ? state.speed : 800);
  state.speed = Number($("speed-select").value);
  $("black-slider").value = String(state.black);
  $("white-slider").value = String(state.white);
  $("invert-toggle").checked = state.invert;
  $("smooth-toggle").checked = state.smooth;
  $("match-toggle").checked = state.match;
  $("gaia-toggle").checked = state.gaia;
  $("names-toggle").checked = state.names;
  $("track-toggle").checked = state.track;
  $("flags-toggle").checked = state.flags;
  $("mask-toggle").checked = state.mask;
  $("wise-toggle").checked = state.wise;
  renderBandButtons();
}

async function init() {
  readUrl();
  document.body.classList.toggle("embed", state.embed);
  viewer = new Viewer({
    image: $("image"), overlay: $("overlay"), stage: $("stage"),
    onSelect: (sel) => selectSky(sel.ra, sel.dec),
    onHover: (info) => {
      const box = $("hover-readout");
      if (!info) { box.hidden = true; return; }
      box.textContent = `${info.text}${info.flagged ? " · flagged" : ""}`;
      box.hidden = false;
    },
    onFrame: () => renderFrame(),
    onView: () => $("stage").classList.toggle("zoomed", viewer.zoom > 1),
  });
  // For checking the tool from the browser console: add ?debug to the address.
  if (new URLSearchParams(location.search).has("debug")) window.__spx = { viewer, state };

  try {
    config = await getJSON("api/config");
  } catch (err) {
    notice(`The server is not responding: ${err.message}`, { error: true });
    return;
  }
  buildStaticControls();
  syncControls();
  applyViewerOptions();
  wire();

  if (state.ra != null) {
    state.name = state.q || null;
    startView();
    if (state.q) {
      // Coordinates came from the link; look the name up only for its motion and label.
      getJSON("api/resolve", { q: state.q }).then((resolved) => {
        if (resolved.kind !== "name" || state.q !== resolved.query) return;
        state.target = resolved;
        state.name = resolved.name;
        viewer.setTarget(resolved);
        renderHeader();
        if (state.sra == null && Math.hypot(resolved.pmra || 0, resolved.pmdec || 0) >= 100) {
          // Now that its motion is known, measure the target along its track.
          Object.assign(spectrum, { key: null, final: false, lastCount: 0, lastTime: 0 });
          ensureSpectrum();
        }
      }).catch(() => {});
      $("search-input").value = state.q;
    } else {
      $("search-input").value = `${state.ra} ${state.dec}`;
    }
  } else if (state.q) {
    $("search-input").value = state.q;
    submitQuery(state.q);
  } else if (!state.embed) {
    // First visit: open on a real example so there is something moving to look at.
    const featured = config.examples.find((e) => e.featured) || config.examples[0];
    if (featured) {
      await openExample(featured, { announce: false });
      notice(`You are watching ${featured.name}. ${featured.blurb} ${featured.look_for} Try the other examples, or enter any coordinates above.`);
      $("search-input").value = "";
    }
  }
}

init();
