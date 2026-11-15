// Sky geometry and formatting helpers.
// Developed with AI assistance (Claude, Anthropic). See README "Use of AI".

const D2R = Math.PI / 180;

/** Grid pixel (col, row; row 0 = north edge, col 0 = east edge) -> [ra, dec] in degrees. */
export function pixToSky(grid, col, row) {
  const c = (grid.n - 1) / 2;
  const s = (grid.scale_arcsec / 3600) * D2R;
  const xi = -(col - c) * s;
  const eta = -(row - c) * s;
  const a0 = grid.ra * D2R;
  const d0 = grid.dec * D2R;
  const den = Math.cos(d0) - eta * Math.sin(d0);
  const ra = a0 + Math.atan2(xi, den);
  const dec = Math.atan2(Math.sin(d0) + eta * Math.cos(d0), Math.hypot(xi, den));
  return [(((ra / D2R) % 360) + 360) % 360, dec / D2R];
}

/** [ra, dec] in degrees -> fractional grid pixel [col, row] (gnomonic projection). */
export function skyToPix(grid, ra, dec) {
  const a = ra * D2R, d = dec * D2R, a0 = grid.ra * D2R, d0 = grid.dec * D2R;
  const cosc = Math.sin(d0) * Math.sin(d) + Math.cos(d0) * Math.cos(d) * Math.cos(a - a0);
  if (cosc <= 0) return [NaN, NaN];
  const xi = (Math.cos(d) * Math.sin(a - a0)) / cosc;
  const eta = (Math.cos(d0) * Math.sin(d) - Math.sin(d0) * Math.cos(d) * Math.cos(a - a0)) / cosc;
  const s = (grid.scale_arcsec / 3600) * D2R;
  const c = (grid.n - 1) / 2;
  return [c - xi / s, c - eta / s];
}

/** Move a position along its proper motion (mas/yr; pmra already includes cos dec). */
export function propagate(ra, dec, pmra, pmdec, fromYear, toYear) {
  const dt = toYear - fromYear;
  const newDec = dec + ((pmdec || 0) * dt) / 3.6e6;
  const newRa = ra + ((pmra || 0) * dt) / 3.6e6 / Math.max(Math.cos(dec * D2R), 1e-6);
  return [((newRa % 360) + 360) % 360, newDec];
}

const pad = (v, n = 2) => String(v).padStart(n, "0");

export function formatRa(raDeg) {
  let total = Math.round((((raDeg % 360) + 360) % 360) / 15 * 3600 * 100);    // hundredths of a second of time
  total %= 24 * 3600 * 100;
  const h = Math.floor(total / 360000);
  const m = Math.floor((total % 360000) / 6000);
  const s = (total % 6000) / 100;
  return `${pad(h)}h${pad(m)}m${s.toFixed(2).padStart(5, "0")}s`;
}

export function formatDec(decDeg) {
  const sign = decDeg < 0 ? "−" : "+";
  const total = Math.round(Math.abs(decDeg) * 3600 * 10);                      // tenths of an arcsecond
  const d = Math.floor(total / 36000);
  const m = Math.floor((total % 36000) / 600);
  const s = (total % 600) / 10;
  return `${sign}${pad(d)}°${pad(m)}′${s.toFixed(1).padStart(4, "0")}″`;
}

export function formatCoords(ra, dec) {
  return `${formatRa(ra)} ${formatDec(dec)}`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** 'YYYY-MM-DD' -> '17 Mar 2026'. */
export function prettyDate(iso) {
  if (!iso) return "";
  const [y, m, d] = iso.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
}

/** 'YYYY-MM-DD' -> 'Mar 2026'. */
export function monthYear(iso) {
  if (!iso) return "";
  const [y, m] = iso.split("-").map(Number);
  return `${MONTHS[m - 1]} ${y}`;
}

export function mjdToIso(mjd) {
  return new Date((mjd - 40587) * 86400000).toISOString().slice(0, 10);
}

/** Angular separation in arcseconds (small angles). */
export function separationArcsec(ra1, dec1, ra2, dec2) {
  const dra = (((ra1 - ra2 + 540) % 360) - 180) * Math.cos(((dec1 + dec2) / 2) * D2R);
  return Math.hypot(dra, dec1 - dec2) * 3600;
}
