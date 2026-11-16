# SPHEREx Blink

**Watch the infrared sky change.** Enter any sky position and blink NASA SPHEREx images taken months apart, so anything that moved or changed jumps out. In false colour, ordinary stars are white and cold objects glow red. Click any object to get its 102-colour infrared spectrum and its brightness over time, measured on the spot.

**Try it: https://spherex-blink.azurewebsites.net** (free hosting: the first page after a quiet spell can take a minute or two to wake, and the site rests when a daily allowance runs out; see "Deploy").

Built for the 2026 NASA Space Apps Challenge, **"Planet X and SPHEREx"**, by team The Celestials (Columbus, GA). A seven-slide demo is in [`submission/SPHEREx-Blink-demo.pdf`](submission/SPHEREx-Blink-demo.pdf).

## What it does

| Challenge asks for | How SPHEREx Blink answers |
|---|---|
| Enter RA and Dec, see a "little video" of that patch | Decimal or sexagesimal coordinates, or an object name. Frames are aligned north-up and blinked, with play, step, speed, a timeline, and zoom and pan. |
| Easy to update with the latest data releases | Nothing is pre-baked. Every view asks NASA's archive what exists for that position; cutouts are cached per exposure, so a return visit downloads only what is new. |
| Use the 102 bands; add up each of the six detector arrays | **Colour view**: three broad channels from the six detectors, balanced so stars are white and cold objects stand out red. Also each detector on its own, or all combined. Each survey pass is coadded with flagged pixels and single-exposure outliers rejected. |
| Spectra on the fly ("extra kudos") | Click any source: its brightness in every exposure, plotted against the wavelength that exposure sampled. |
| Add or subtract images taken at different times | Difference mode (each pass minus the first) and Sum mode (everything combined). |
| Choose the field of view | 3, 6, 9 or 12 arcminutes, plus zoom up to 8x inside the field. |
| Overlay other telescopes, such as Gaia or WISE | **Compare with 2010**: adds the WISE mission's image of the same field to the blink, a 15-year baseline. Gaia DR3 stars are drawn at their position on each frame's date, with arrows on fast movers; sources known only in the infrared (2MASS, AllWISE) are marked too. |
| Coordinates in the URL, embeddable | Every view is a link (`?ra=…&dec=…&fov=…&band=…&mode=…`). "Embed" gives an iframe (`&embed=1`). |
| Show metadata such as contamination flags | Flagged-pixel overlay; per-measurement flag names; saturated and blended measurements are marked and explained. |

Beyond the list:

- **Looks for candidates by itself.** Once a field has loaded, the candidate finder measures every source in every survey pass and lists what moved, what changed brightness, what no catalogue lists, and what a 2010 catalogue lists that is not there now. It works by measurement alone and consults the catalogues only afterwards, so it can say which findings are already known and which are unexplained. See "Candidate finder" below.
- **Says what is already known.** Click anything and "What is it?" gives its name and type from SIMBAD (for example "Brown Dwarf, spectral type Y2, moves 8.15″ per year"), with its Gaia, 2MASS and AllWISE entries. Named objects are labelled on the image. If no catalogue has anything at that spot, it says so, which is the first thing to check before calling something new.
- **Time machine (Simulate).** Runs the clock up to 10,000 years forward or back: every star with a measured motion is lifted out of today's image and redrawn where that motion puts it. It is clearly marked as a simulation, not an observation; stars with no measured motion, and galaxies, stay put.
- **A large image.** The viewer fills the height of the window, with the controls laid over it like a video player, and can go full screen (F).
- **Follows moving targets.** A named object's position is propagated from its catalogue epoch to today (Barnard's Star has moved 270″ since 2000), and its spectrum is measured along its track.
- **Knows old coordinates when it sees them.** Paste the WISE 0855 coordinates from the challenge statement and it says they are where the object was around 2010, and offers to centre on where it is now.
- **Measures motion.** The object's position in each pass is fitted for a drift in arcseconds per year.
- **Honest about what it cannot measure.** Stars too bright for SPHEREx are labelled "Saturated" and followed through their unsaturated outer wings instead; measurements with a known neighbour inside the 6″ pixel are labelled "Blended".

## Results on the five challenge targets

Measured by this tool from public SPHEREx data on 13 Nov 2026. Reproduce with `python scripts/try_target.py "<name>"`.

| Target | What the tool shows | Independent check |
|---|---|---|
| **Barnard's Star** | Hops north between three passes; against the 2010 WISE frame it jumps 2.7 arcminutes. Measured motion **9.95 ± 0.61″/yr**. Saturated, so its light curve comes from the wings: steady within 1–3% in four of six bands. | Gaia DR3: 10.39″/yr |
| **Epsilon Indi B** | A vivid red dot among white stars in Colour view; a clear dark/bright pair in Difference view. Measured motion **4.49 ± 0.30″/yr** to the south-east. Spectrum shows the brown-dwarf peaks near 1.3, 1.6, 2.1 and 4.0 µm. Brightness steady within 1–3% over four passes. | SIMBAD (Gaia DR3): 4.68″/yr |
| **WISE 0855−0714** | A faint red dot at its predicted position, brightest at 4–5 µm (median 0.56 mJy in band 6, signal-to-noise about 5 per exposure). Colour view shows it drifting into a blue background star; the tool flags every exposure as **blended** (Gaia G 16.3, now 4.4″ away). The 2010 WISE frame shows it two arcminutes east, on its predicted 2010 position. No light curve or motion is reported for it. | Band-6 brightness is about what its WISE 4.6 µm magnitude implies (roughly 0.5 mJy) |
| **RW Aurigae** | Between Sep 2025 and Sep 2026 it brightened by roughly half at 0.9 µm while fading by about 46% at 4 µm (16σ). Partly saturated; affected bands use the wings. | Known irregular variable. No independent check of these numbers was made. |
| **R Draconis** | Fully saturated. The wing light curve shows a 45% change in band 6 between Feb and May 2026 (15σ), and bright phases about eight months apart across the bands. | Catalogued pulsation period is about 245 days |

**Brightness calibration check.** For isolated field stars, the tool's fluxes were compared with the 2MASS (J, H, Ks) and WISE (W1, W2) catalogues at matching wavelengths: `python scripts/validate_flux.py`. Across 41 comparisons on 9 stars in three fields the median ratio is **1.02** (tool / catalogue) with 4% scatter; by filter: J 1.03, H 1.03, Ks 1.02, W1 0.99, W2 1.01. The same check first read 14% high, which exposed a real error (the PSF model counted the pixel smoothing twice); it was fixed at the source, not with a correction factor. Nine stars is a small sample: treat this as a sanity check, not a calibration.

Limits worth knowing: SPHEREx pixels are 6.15″, so over its roughly one-year baseline even the fastest stars move one or two pixels (which is why the 2010 comparison helps). A typical position has three survey passes so far. WISE 0855 is at the edge of what a single pass can show.

## Candidate finder

The challenge is named after the search for Planet X, and the way Pluto was found: blink two images and see what moved. The finder does that by measurement, for every source in the field, with no person watching.

1. **Find** the sources in the combined image, and in each pass alone (to catch what is there only once).
2. **Measure** each one in every pass: its position, and its brightness in three wavelength ranges. Saturated stars are measured through their outer glow.
3. **Flag** what appears to move (0.8″ per year or more) or change brightness (15% or more). No catalogue has been consulted at this point.
4. **Confirm** each flag with the slower exposure-by-exposure measurement behind the spectrum panel, which compares like wavelength with like. A motion counts only if the pass images and the single exposures agree on it.
5. **Explain** with the catalogues (Gaia DR3, SIMBAD, 2MASS, AllWISE): which flagged sources are already known; which detected sources no catalogue lists; and which AllWISE sources from 2010 are missing, far fainter, or somewhere else now (skipping sources AllWISE itself marks as artefacts).
6. **Check like for like.** Anything that differs from the 2010 AllWISE catalogue is measured again in the wavelength range WISE itself used (2.8–3.8 µm for its W1 band, 4.1–5.0 µm for W2), exposure by exposure, and scaled by ordinary stars in the same field. A "faded" source must be below 75% of its 2010 brightness with three times the measurement error to spare. A "moved" source must have left its 2010 position: the light still measured there must be no more than a star's own spill at that distance.

Each finding is labelled **already known**, **unexplained** or **doubtful**, with the measurements, the reasons for caution, and buttons to show it, measure it, or blink it against 2010. Findings are candidates for a person to look at. None is a discovery until someone else confirms it.

**A blind test on the five challenge targets.** The finder was run on each field without being told what was there. It flagged all five, and nothing else:

| Field | What the finder flagged on its own | What the catalogues then said |
|---|---|---|
| Epsilon Indi B | Moved 4.5″ per year | Known brown dwarf, catalogued at 4.68″ per year |
| Barnard's Star | Moved 9.9″ per year | Known star, catalogued at 10.39″ per year |
| WISE 0855−0714 | A source AllWISE saw in 2010 is only 23% as bright at that spot now | Known brown dwarf moving 8.15″ per year; it has travelled 125″ since |
| RW Aurigae | Changed brightness by 47% | Known T Tauri star |
| R Draconis | Changed brightness by 46% | Known Mira variable |

Across those fields (200 sources measured, including the field at the WISE 0855 coordinates printed in the challenge) there were no unexplained or doubtful findings. WISE 0855 itself is found through its 2010 position, not through its motion between passes: it currently sits on top of a background star, which hides the small shift.

**Slow movers.** Two more known brown dwarfs were tried because they move too slowly to show between SPHEREx passes. The finder was given only their coordinates, and found both through their drift since 2010:

| Field | What the finder flagged on its own | What the catalogues then said |
|---|---|---|
| SDSS J1624+0029 (T6 dwarf) | 5.8″ from its 2010 position, about 0.37″ per year | Catalogued at 0.37″ per year |
| 2MASS J0243−2453 (T6 dwarf) | 5.1″ from its 2010 position, about 0.33″ per year | Catalogued at 0.36″ per year |

In the like-for-like check their 2010 positions held 51% and 60% of the light measured where they are now. Ordinary stars in the same fields, stepped the same distance, leave 53% and 59%.

**Fresh sky, and what it taught.** A first scan of 100 random fields away from the Milky Way (13 November 2026) was cut short by an archive outage. 52 fields loaded (2,033 sources), 31 of them with every catalogue answering (1,194 sources). Between passes, 72 first-look flags were raised and none survived the exposure-by-exposure check, so the scan produced no false mover or variable, and no new one. Against the 2010 catalogue it listed three findings. Each was then measured by hand, and two turned out to be mistakes of the method:

- A quasar listed as "32% as bright as in 2010" is in fact at 88% of its 2010 brightness at 2.8–3.8 µm (give or take 4%), which is ordinary for a quasar. It is a red object, so it is faint in the broad 1.6–3.8 µm image that the first version compared with WISE.
- A source listed as "6.3″ from its 2010 position" is two objects one pixel apart, one bluer than the other. The tool centred on the blue one; WISE had centred on the red one. Nothing moved.
- A second quasar, SDSS J012322.36+004917.9, really has faded. It is at 50% (give or take 5%) of its 2010 brightness at 2.8–3.8 µm, and about 60% at 4.1–5.0 µm. The CatWISE2020 catalogue, which averages 2010 to 2018 and is not used by the tool, already has it 0.18 magnitudes fainter than AllWISE at 3.4 µm and 0.28 at 4.6 µm.

Step 6 was added because of the first two. With it the finder drops both by itself, and keeps the faded quasar, WISE 0855 and the two slow movers above.

**The same 100 fields, scanned again.** With step 6 in place the scan was repeated the same evening and took three hours. This time all 100 fields loaded and every catalogue answered for each of them: 3,869 sources over one square degree in total (each field is 6 arcminutes across), 99 of the fields with two or more passes. 147 first-look flags were raised, counting both apparent motion or change between passes and differences from the 2010 catalogue. 146 of them did not survive the checks. The one finding left is the faded quasar above. Nothing unexplained and nothing doubtful was found.

One square degree is about one part in 41,000 of the sky. The result shows that the finder stays quiet on ordinary sky while still finding the seven known objects above. It does not show that there is nothing new to find.

To search more sky, use `scripts/scan.py` (below). It takes names, coordinates, a file of them, or random positions away from the Milky Way, and writes a report with a viewer link for every candidate. When the archive stops answering it waits and comes back to the fields it could not finish.

## How it works

```
browser ──/api/status──▶ FastAPI ──SIA search──▶ NASA/IPAC IRSA   (which exposures cover this spot?)
        ◀─ progress ───           ──cutouts────▶ IRSA cutout service (only the start of each file; see below)
        ──/api/frames──▶ align, mask, coadd ──▶ float32 frames ──▶ stretched, coloured, blinked, subtracted in the browser
        ──/api/spectrum▶ forced photometry per exposure ──▶ spectrum, light curve, motion
        ──/api/wise────▶ AllWISE Atlas cutout, resampled onto the same grid
        ──/api/candidates▶ every source measured in every pass ──▶ what moved, changed, is uncatalogued, or has gone
```

- **Finding images**: IRSA's Simple Image Access service (about 1 s; the TAP route took about 15 s).
- **Fast cutouts**: each cutout file is about 4 MB, of which 3.9 MB is a PSF table the viewer does not need and which comes last. The download stops once the image, flags, variance and zodiacal-light extensions have arrived (about 130–200 kB, depending on field size).
- **Polite to the archive**: IRSA answers "server busy" on roughly one request in five and sustains about four cutouts per second. The loader uses six connections, backs off together on a busy answer, and retries patiently. One band is fetched first and shown while the rest arrives.
- **Archive outages**: if the archive does not answer, or only some of its collections answer, the tool keeps the exposure list it already holds for that position, adds anything new that did arrive, and asks again later. With a list on record it waits at most 12 s for a fresh one. (During an outage on 15 Nov 2026 a partial answer had replaced a full list, and Barnard's Star showed one pass instead of three; there is now a test for it.)
- **Wavelengths**: SPHEREx has a linear variable filter, so wavelength depends on where a source lands on the detector. The per-detector lookup table is interpolated for each exposure (checked against `astropy.wcs`).
- **Colour**: red = detectors 5+6 (3.8–5.0 µm), green = 3+4 (1.6–3.8 µm), blue = 1+2 (0.75–1.6 µm). Channels are balanced with one vote per star so the typical star is white; colour fades to grey in the noise and on flagged pixels.
- **Names**: SIMBAD (through its Harvard mirror, then CDS), and the 2MASS and AllWISE catalogues at IRSA, are searched once per field and cached. SIMBAD positions are for the year 2000, so the search reaches 5 arcminutes beyond the field and objects are moved to the frame's date before matching.
- **Simulation**: each moving star is separated from the combined image as its own small picture: its actual pixels in the core (the whole flagged core for a saturated star), and a smooth radial profile for its halo so neighbours are left in place. The patch where it stood is filled with the local sky level and a little grain. Stars move in straight lines at constant speed; parallax and orbital motion are ignored.
- **Fair comparison between passes**: each pass samples slightly different wavelengths within a band, which makes ordinary stars several percent brighter or fainter. Frames are matched using the field's own stars, and light curves compare each measurement with the source's typical brightness at the same wavelength.

Code map: `backend/app/` — `discovery.py` (search, epochs), `fetch.py` (cutouts), `stack.py` (jobs, cache), `process.py` (alignment, coadds, colour), `photometry.py` (spectrum, light curve, motion, blends), `finder.py` (candidate finder), `wise.py` (2010 comparison frame), `catalogs.py` (names: SIMBAD, 2MASS, AllWISE), `payloads.py` (stored views), `wavecal.py`, `resolver.py`, `gaia.py`, `main.py` (API). `frontend/` — plain HTML, CSS and JavaScript modules, no build step.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt      # macOS/Linux: .venv/bin/python
.venv/Scripts/python -m uvicorn app.main:app --app-dir backend --port 8765
```

Open http://localhost:8765. Then, optionally:

```bash
.venv/Scripts/python scripts/prewarm.py            # cache the examples (data, stored views, WISE, Gaia, names, candidates) so they open in a second or two
.venv/Scripts/python scripts/make_thumbnails.py    # redraw the example cards' thumbnails from that data
.venv/Scripts/python scripts/validate_flux.py      # compare star brightnesses with 2MASS and WISE
.venv/Scripts/python scripts/scan.py --examples    # candidate finder on the five challenge targets (the blind test above)
.venv/Scripts/python scripts/scan.py --random 40   # ... on 40 random fields; see --help for names, files and options
.venv/Scripts/python -m pytest                     # 50 offline tests
SPX_LIVE=1 .venv/Scripts/python -m pytest -m live  # end-to-end test against NASA's archive
```

Developed on Python 3.14; the Dockerfile uses 3.13.

## Deploy

One container serves the API and the page:

```bash
docker build -t spherex-blink .
docker run -p 8000:8000 -v spherex-cache:/data spherex-blink
```

Works on any host that runs a container or a Python web app (Azure App Service, Hugging Face Spaces, Render, Fly.io). Give it persistent storage at `SPX_CACHE_DIR` (default `/data/cache` in the container) so cached cutouts survive restarts, and run `scripts/prewarm.py` once after deploying. The first view of a new position shows a first image in under 10 s, a complete band in roughly 15–30 s, and all six bands (needed for colour) in one to three minutes, limited by the archive. Positions seen before, including everything `prewarm.py` covers, open in one to two seconds.

**The live site** runs on Azure App Service's free tier: Python 3.14, the start command above without Docker (`python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000`), the cache at `/home/spx-cache`, `SPX_DISK_CACHE_MB=700` and `SPX_STACK_CACHE_MB=200`. The cache for the five examples was uploaded once from a development machine instead of running `prewarm.py` on the server. That tier's limits, read from the plan on 15 Nov 2026: 60 minutes of processor time and 165 MB of outbound data a day, 1 GB of disk, no always-on (the first page after a quiet spell takes a minute or two), and the site is switched off for the rest of the hour after 15 restarts. If the site says it is stopped or disabled, an allowance has run out and it will come back by itself; the tool also runs locally with the three commands under "Run it".

Settings (environment variables): `SPX_CACHE_DIR`, `SPX_FETCH_CONCURRENCY` (default 6), `SPX_MAX_PER_BAND` (default 60 exposures per detector), `SPX_STACK_CACHE_MB` (default 600), `SPX_DISK_CACHE_MB` (default 0, no limit: on a host with a small disk, stop storing cutouts and views once the cache folder reaches this size; the viewer keeps working from memory).

## API

All endpoints are GET and CORS-open.

| Endpoint | Returns |
|---|---|
| `/api/resolve?q=` | Coordinates for a name or coordinate string, with proper motion; flags coordinates that are an old position of a known fast mover |
| `/api/status?ra=&dec=&fov=&band=` | Starts or reports loading for a position: progress, passes, bands ready |
| `/api/frames?ra=&dec=&fov=&band=colour\|all\|1-6&kind=epochs\|exposures&mask=&partial=` | Aligned frames (binary: `SPXB`, header length, JSON header, float32 planes, uint8 flag planes; three planes per frame for colour) |
| `/api/wise?ra=&dec=&fov=&band=W1\|W2` | The WISE 2010 image of the field on the same grid, same binary layout |
| `/api/spectrum?ra=&dec=&fov=&sra=&sdec=[&pmra=&pmdec=&epoch=]` | Per-exposure fluxes, light curve, variability, measured motion, blend report |
| `/api/candidates?ra=&dec=&fov=` | The candidate finder's report for the field: findings with measurements, verdict, cautions, and what was checked |
| `/api/gaia?ra=&dec=&fov=` | Gaia DR3 stars covering the field |
| `/api/catalogue?ra=&dec=&fov=&source=simbad\|twomass\|allwise` | Catalogued objects in and around the field: names, types, spectral types, proper motions |
| `/api/config` | Bands, field choices, flag definitions, examples |

## Data and references

**NASA data**
- SPHEREx Spectral Images, Data Release 1 and Quick Release 3, from the NASA/IPAC Infrared Science Archive (IRSA): https://irsa.ipac.caltech.edu/Missions/spherex.html (DOI 10.26131/IRSA678)
- AllWISE Atlas images (WISE mission), from IRSA: https://irsa.ipac.caltech.edu/Missions/wise.html
- 2MASS Point Source Catalog and AllWISE Source Catalog, from IRSA (object identification, and `validate_flux.py`)
- IRSA Simple Image Access, cutout service and TAP: https://irsa.ipac.caltech.edu/
- IRSA SPHEREx tutorial notebooks (data layout, cutout URL syntax, wavelength WCS): https://caltech-ipac.github.io/irsa-tutorials/spherex/
- WiseView, the model for this kind of tool (Backyard Worlds): http://byw.tools/wiseview

**Other data and services**
- Gaia DR3, European Space Agency, via the ESA Gaia Archive and the copy hosted at IRSA
- SIMBAD (object names, types, spectral types and proper motions; queried through the mirror at the Center for Astrophysics | Harvard & Smithsonian) and the Sesame name resolver, CDS Strasbourg
- Astrometry for WISE 0855−0714: Kirkpatrick et al. 2021 (ApJS 253, 7), via SIMBAD
- CatWISE2020 catalogue (Marocco et al. 2021, ApJS 253, 8), from IRSA: looked up by hand to check the faded quasar reported under "Candidate finder"; the tool itself does not query it
- Photometric zero points: Cohen et al. 2003 (2MASS), Jarrett et al. 2011 (WISE)

**Software and assets**: Python, FastAPI, Uvicorn, HTTPX, NumPy, SciPy, Astropy (Pillow for the thumbnail script). The browser side uses no JavaScript libraries. Fonts: Inter and Space Grotesk (SIL Open Font License), loaded from Google Fonts. Chart colours follow a palette checked for colour-vision deficiency.

## Use of AI

This project was developed with AI assistance: Claude (Anthropic), used through Claude Code.

- **Code**: the backend, frontend and tests were written with Claude under the team's direction. Every source file carries a header saying so, and the page has a `generator` meta tag.
- **Analysis**: Claude ran the exploratory checks against the archive (timings, file structure, throttling) and the validation runs reported above.
- **Design**: Claude proposed the architecture and the build plan, and suggested several of the features (false-colour view, the 2010 WISE comparison, following saturated stars through their wings, blend detection from Gaia, the motion fit, and brightness matching between passes), as well as the visual design. The team asked for objects to be named, for a larger viewer, and for a mode that simulates change from the data; Claude designed and built them. The team also asked whether the tool could find anything new, the way Pluto was found; Claude designed and built the candidate finder in answer, and ran the tests reported above. After the first sky scan Claude measured each finding by hand against the WISE catalogues, found that two of the three were mistakes of its own method, added the like-for-like check (step 6), and repeated the tests.
- **Decisions**: the team chose the challenge and the signature feature (click-to-spectrum and light curve), and approved the plan.
- **Not AI-generated**: every image the tool shows, including the example thumbnails, is real SPHEREx and WISE data rendered by the tool. No AI-generated video or audio is used, and no NASA logos or mission insignia appear.
- **Demo deck**: `submission/SPHEREx-Blink-demo.pdf` (source in `submission/demo/`) was written and laid out by Claude under the team's direction. Its screenshots are captures of the working tool, not generated images.
- **AI-generated image**: the team logo (`submission/team-logo.png`) was drawn by Claude as HTML and CSS to the team's brief. It carries a visible "AI-generated image" mark.

## Limitations

- Quick-look measurements only. Fluxes use a Gaussian PSF approximation; they matched 2MASS and WISE to about 2% for nine isolated stars, but they are not the SPHEREx project's calibrated spectrophotometry. Use IRSA's Spectrophotometry Tool for science.
- Colour is for spotting unusual objects, not for measuring them: the balance is set by the stars in each field.
- The 2010 frame comes from a different telescope with different depth, so faint stars can appear in it and not in SPHEREx frames. Use it for big movers, not for subtle changes.
- Simulate mode is an extrapolation: straight-line motion at today's catalogue speed. Over thousands of years real stars also change speed and brightness, and the sky behind a star is unknown, so the patch it leaves is filled in.
- Names come from SIMBAD, 2MASS and AllWISE. "Nothing catalogued here" means none of those lists has an entry within 9″; it does not by itself mean the source is new.
- The candidate finder sees motion between passes only above about 1″ per year, because SPHEREx pixels are 6″ and the survey is about a year old. Slower motion shows only against the 2010 and 1999 catalogue positions, only for sources Gaia does not list, and only once the source is 3″ or more from where AllWISE put it (about 0.2″ per year).
- A distant planet would not be found this way. At 500 times Earth's distance from the Sun it would appear to swing about 7 arcminutes each way as Earth orbits, so it would leave a 6-arcminute field between passes; and it is expected to be fainter than SPHEREx can see.
- "Unexplained" means unexplained by Gaia DR3, SIMBAD, 2MASS and AllWISE. Other lists (AAVSO VSX and ZTF for variables, CatWISE for motions, the Minor Planet Center for asteroids) are not checked. In crowded fields a chance catalogue match is likely, and the report says so.
- Parallax is ignored in the motion fit (it is at most about 0.5″, under a tenth of a pixel).
- Pass coadds deliberately reject anything seen in only one exposure, so fast asteroids appear only in "Every exposure" mode.
- Near the ecliptic poles a position has thousands of exposures; the tool samples 60 per detector evenly in time.
- Not affiliated with or endorsed by NASA.
