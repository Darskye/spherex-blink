// Turning catalogue entries into names people can read, and matching a sky
// position against everything catalogued in the field.
// Developed with AI assistance (Claude, Anthropic). See README "Use of AI".

import { propagate, separationArcsec } from "./sky.js";

export const MATCH_RADIUS_ARCSEC = 9;       // about one and a half SPHEREx pixels

// Identifiers that are survey serial numbers rather than names. For these the
// object's type ("Galaxy", "RR Lyrae Variable") says more than the number does.
const SURVEY_ID = /^(Gaia (DR|EDR)\d|2MASS J|2MASSI J|2MASSW J|2MASX J|WISEA? J|WISEP J|TYC |UCAC\d|GSC |SDSS J|ZTF J|ATO J|CRTS J|ASAS J|ASASSN|1RXS J|2RXS J|1WGA J|2E |GES J|2dFGRS|6dFGS|PMN J|NVSS J|GALEX|USNO|LEDA|PGC|TIC |OGLE|PSO J|DENIS|\[)/;

/** SIMBAD identifier without its bookkeeping prefix: "* eps Ind B" -> "eps Ind B". */
export function cleanName(id) {
  return String(id).replace(/^(NAME|V\*|\*\*|\*|EM\*|SV\*|Cl\*|PN|SNR)\s+/, "").replace(/\s+/g, " ").trim();
}

export function isSurveyId(id) {
  return SURVEY_ID.test(String(id));
}

/** What to print next to a SIMBAD object in the image. */
export function labelFor(object) {
  const survey = isSurveyId(object.name);
  return {
    text: survey ? (object.kind || cleanName(object.name)) : cleanName(object.name),
    proper: !survey,                           // a real name, not a serial number
    plainStar: object.otype === "*" || !object.kind,
  };
}

const magnitude = (label, value) => (value == null ? null : `${label} ${value.toFixed(1)}`);

/**
 * Everything catalogued within MATCH_RADIUS_ARCSEC of a position, nearest first
 * within each catalogue. Positions with a known motion are moved to `year`.
 */
export function identify(position, year, catalogues, gaiaStars) {
  const matches = [];
  const consider = (ra, dec, entry) => {
    const sep = separationArcsec(position.ra, position.dec, ra, dec);
    if (sep <= MATCH_RADIUS_ARCSEC) matches.push({ ...entry, sep });
  };
  for (const o of catalogues.simbad || []) {
    const [ra, dec] = propagate(o.ra, o.dec, o.pmra, o.pmdec, o.epoch || 2000, year);
    consider(ra, dec, {
      catalogue: "SIMBAD", rank: 0, id: o.name,
      name: cleanName(o.name), kind: o.kind || "Object", detail: o.sptype ? `spectral type ${o.sptype}` : null,
      motion: Math.hypot(o.pmra || 0, o.pmdec || 0),
    });
  }
  for (const s of gaiaStars || []) {
    const [ra, dec] = propagate(s.ra, s.dec, s.pmra, s.pmdec, s.epoch || 2016, year);
    consider(ra, dec, {
      catalogue: "Gaia DR3", rank: 1, id: `Gaia DR3 ${s.id}`, name: `Gaia DR3 ${s.id}`, kind: "Star",
      detail: [magnitude("G", s.g), s.plx != null && s.plx > 1 ? `${(3261.6 / s.plx).toFixed(0)} light-years away` : null].filter(Boolean).join(" · ") || null,
      motion: Math.hypot(s.pmra || 0, s.pmdec || 0),
    });
  }
  for (const o of catalogues.twomass || []) {
    consider(o.ra, o.dec, {
      catalogue: "2MASS", rank: 2, id: o.name, name: o.name, kind: "Near-infrared source",
      detail: [magnitude("J", o.j), magnitude("H", o.h), magnitude("Ks", o.k)].filter(Boolean).join(" · ") || null,
    });
  }
  for (const o of catalogues.allwise || []) {
    consider(o.ra, o.dec, {
      catalogue: "AllWISE", rank: 3, id: o.name, name: o.name, kind: "Mid-infrared source",
      detail: [magnitude("W1", o.w1), magnitude("W2", o.w2)].filter(Boolean).join(" · ") || null,
    });
  }
  matches.sort((a, b) => a.rank - b.rank || a.sep - b.sep);
  return matches;
}
