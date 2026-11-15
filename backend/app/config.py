"""Central configuration for the SPHEREx blink viewer.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("SPX_CACHE_DIR", ROOT / "cache"))
FRONTEND_DIR = ROOT / "frontend"
BUNDLED_DATA_DIR = Path(__file__).resolve().parent / "data"

# --- NASA/IPAC Infrared Science Archive (IRSA) -------------------------------
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
# Wide-survey and deep-field collections. DR1 and QR3 are a uniformly processed
# set; a position may appear in any of them, so all are searched and merged.
SIA_COLLECTIONS = (
    "spherex_dr1",
    "spherex_qr3",
    "spherex_dr1_deep",
    "spherex_qr3_deep",
)
SIA_RADIUS_DEG = 0.0003          # ~1 arcsec: "images whose footprint contains the point"
EXPOSURE_LIST_TTL_S = 6 * 3600   # re-ask the archive for new exposures every 6 h

USER_AGENT = "spherex-blink/1.0 (NASA Space Apps 2026 project; astronomy viewer)"

# IRSA's cutout service answers HTTP 503 "Server busy" on roughly one request in
# five whatever the concurrency, and sustains about four cutouts per second
# (measured 2026-11-13). More parallelism does not help, so stay modest and
# retry patiently instead.
FETCH_CONCURRENCY = int(os.environ.get("SPX_FETCH_CONCURRENCY", "6"))
FETCH_RETRIES = 4            # network errors per cutout
FETCH_BUSY_RETRIES = 25      # "server busy" answers per cutout
FETCH_TIMEOUT_S = 60.0

# --- Instrument facts (verified against DR1 file headers) --------------------
NATIVE_PIXEL_ARCSEC = 6.15
ARCSEC2_TO_SR = (1.0 / 206264.80624709636) ** 2
DETECTORS = (1, 2, 3, 4, 5, 6)
DETECTOR_RANGES_UM = {
    1: (0.75, 1.09),
    2: (1.10, 1.62),
    3: (1.63, 2.41),
    4: (2.42, 3.82),
    5: (3.83, 4.41),
    6: (4.42, 5.00),
}

# --- Viewer geometry ---------------------------------------------------------
FOV_CHOICES_ARCMIN = (3.0, 6.0, 9.0, 12.0)
DEFAULT_FOV_ARCMIN = 6.0
GRID_N = 192                      # output frames are GRID_N x GRID_N, north up, east left
# Cutouts are in detector orientation (any roll angle), so they must cover the
# diagonal of the north-up square, plus a small margin for the photometry annulus.
CUTOUT_PAD_FACTOR = 1.45

# --- Sampling limits ---------------------------------------------------------
# Near the ecliptic poles a position is observed thousands of times. Cap the
# number of exposures per detector and sample evenly in time.
MAX_EXPOSURES_PER_BAND = int(os.environ.get("SPX_MAX_PER_BAND", "60"))

# --- Epoch grouping ----------------------------------------------------------
EPOCH_GAP_DAYS = 45.0             # a gap this long starts a new survey pass
EPOCH_MAX_SPAN_DAYS = 60.0        # longer passes (polar regions) are split ...
EPOCH_SPLIT_DAYS = 30.0           # ... into windows of about this length

# --- Flags -------------------------------------------------------------------
# Flag names are read from each file's FLAGS header (MSKNnnnn / MSKMnnnn). These
# are the planes treated as "do not trust this pixel" when building coadds and
# doing photometry. SOURCE is deliberately absent: it marks known sources, not
# bad data.
BAD_FLAG_NAMES = (
    "TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT",
    "COLD", "NONLINEAR", "PERSIST", "OUTLIER", "CROSSTALK", "GHOST",
    "GHOST_FPA", "GHOST_EXT", "STREAK", "BLOOM", "SNOWBALL", "HALO",
)
# Fallback bit positions if a FLAGS header is ever missing its definitions
# (values copied from a DR1 file header on 2026-11-13).
DEFAULT_FLAG_BITS = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "NONFUNC": 6, "DICHROIC": 7,
    "MISSING_DATA": 9, "HOT": 10, "COLD": 11, "FULLSAMPLE": 12, "PHANMISS": 14,
    "NONLINEAR": 15, "PERSIST": 17, "OUTLIER": 19, "CROSSTALK": 20,
    "SOURCE": 21, "GHOST": 22, "GHOST_FPA": 23, "GHOST_EXT": 24, "STREAK": 25,
    "BLOOM": 26, "SNOWBALL": 27, "HALO": 28,
}
DEFAULT_FLAG_DESCRIPTIONS = {
    "TRANSIENT": "Transient detected during sample-up-the-ramp",
    "OVERFLOW": "Overflow (saturation) detected",
    "SUR_ERROR": "Error in onboard processing",
    "NONFUNC": "Permanently unusable pixel",
    "DICHROIC": "Low efficiency due to dichroic",
    "MISSING_DATA": "Onboard data lost",
    "HOT": "Hot pixel",
    "COLD": "Anomalously low signal",
    "FULLSAMPLE": "Full sample history is available",
    "PHANMISS": "Phantom correction was not applied",
    "NONLINEAR": "Nonlinearity correction unreliable",
    "PERSIST": "Persistent charge above threshold",
    "OUTLIER": "Flagged by outlier detection",
    "CROSSTALK": "Affected by crosstalk",
    "SOURCE": "Mapped to a known source",
    "GHOST": "Ghost from a bright source in the exposure",
    "GHOST_FPA": "Ghost from a bright source on another detector",
    "GHOST_EXT": "Ghost from a bright source outside the field",
    "STREAK": "Affected by a streak",
    "BLOOM": "Part of a bloom artifact",
    "SNOWBALL": "Part of a snowball artifact",
    "HALO": "Part of a transient halo",
}

# --- Memory limits -----------------------------------------------------------
STACK_CACHE_MAX_BYTES = int(os.environ.get("SPX_STACK_CACHE_MB", "600")) * 1024 * 1024
FRAME_CACHE_MAX_ITEMS = 48

# --- External catalogs -------------------------------------------------------
SESAME_URL = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_MAG_LIMIT = 19.0
GAIA_MAX_ROWS = 1500
# Epoch that resolved names are propagated to before centring the view.
VIEW_EPOCH_YEAR = 2026.0
