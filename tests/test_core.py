"""Offline tests of the pieces the science depends on.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import io
import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from app import config, discovery, fetch, photometry, process, resolver, wavecal
from app.discovery import Exposure
from app.stack import ExposureData

# --------------------------------------------------------------------------- epochs


def test_epochs_typical_position_has_one_epoch_per_pass():
    # Dates measured for WISE 0855: three passes, the middle one with a late straggler group.
    mjd = np.concatenate([
        np.linspace(60800.7, 60801.5, 30),
        np.linspace(60979.7, 60993.9, 110),
        np.linspace(61020.8, 61022.4, 18),
        np.linspace(61151.2, 61174.3, 100),
    ])
    labels = discovery.assign_epochs(mjd)
    assert sorted(set(labels)) == [0, 1, 2]
    assert set(labels[30:158]) == {1}          # straggler group stays with its pass


def test_epochs_continuous_coverage_is_split_into_windows():
    mjd = np.linspace(60790.0, 61180.0, 600)   # near an ecliptic pole: no gaps at all
    labels = discovery.assign_epochs(mjd)
    n = len(set(labels))
    assert 10 <= n <= 16
    assert list(labels) == sorted(labels)      # labels increase with time
    for label in set(labels):
        span = np.ptp(mjd[labels == label])
        assert span <= config.EPOCH_SPLIT_DAYS + 1


def test_epochs_are_independent_of_input_order():
    rng = np.random.default_rng(1)
    mjd = np.concatenate([rng.uniform(60800, 60815, 20), rng.uniform(60980, 60995, 20)])
    labels = discovery.assign_epochs(mjd)
    assert all(labels[mjd < 60900] == 0) and all(labels[mjd > 60900] == 1)


def test_cap_keeps_even_time_coverage():
    exposures = [Exposure(f"o{i}", 3, "u", 60800.0 + i, 100.0, "c", "") for i in range(500)]
    kept = discovery.cap_exposures(exposures, 60)
    assert len(kept) == 60
    times = [e.mjd for e in kept]
    assert times[0] == 60800.0 and times[-1] == 61299.0
    assert max(np.diff(times)) < 12


# --------------------------------------------------------------------------- download prefix


def _fake_cutout(n: int = 20) -> bytes:
    rng = np.random.default_rng(0)
    image = rng.normal(1.0, 0.1, (n, n)).astype(np.float32)
    header = fits.Header()
    header["DETECTOR"] = 6
    header["HISTORY"] = "[CALIB] spectral_wcs abc-123"
    flag_header = fits.Header()
    flag_header["MSKN0000"] = "TRANSIENT"
    flag_header["MSKM0000"] = 1
    flag_header["MSKN0021"] = "SOURCE"
    flag_header["MSKM0021"] = 1 << 21
    psf = fits.BinTableHDU.from_columns(
        [fits.Column(name="EPSF", format="1089D", array=np.zeros((441, 1089)))], name="EPSF"
    )
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(image, header=header, name="IMAGE"),
        fits.ImageHDU(np.zeros((n, n), dtype=np.int32), header=flag_header, name="FLAGS"),
        fits.ImageHDU(np.full((n, n), 0.01, dtype=np.float32), name="VARIANCE"),
        fits.ImageHDU(np.full((n, n), 0.4, dtype=np.float32), name="ZODI"),
        psf,
    ])
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdul.writeto(buffer)
    return buffer.getvalue()


def test_prefix_reader_stops_before_the_psf_table():
    data = _fake_cutout()
    reader = fetch.FitsPrefixReader()
    done_at = None
    for start in range(0, len(data), 4096):
        if reader.feed(data[start:start + 4096]):
            done_at = start + 4096
            break
    assert done_at is not None
    assert reader.end_offset % 2880 == 0
    assert reader.end_offset < 0.1 * len(data)       # the ~3.8 MB PSF table was never needed
    cut = fetch.parse_cutout(reader.result())
    assert cut.sci.shape == (20, 20)
    assert abs(float(np.median(cut.sci)) - 0.6) < 0.05   # IMAGE - ZODI
    assert cut.calib_id == "abc-123"
    assert cut.flag_bits == {"TRANSIENT": 0, "SOURCE": 21}
    assert "HISTORY" not in cut.header


# --------------------------------------------------------------------------- wavelength


def test_wavelength_lookup_is_exact_for_a_bilinear_table():
    x = np.array([1, 256, 511, 766, 1021, 1275, 1530, 1785, 2040], dtype=float)
    y = x.copy()
    yy, xx = np.meshgrid(y, x, indexing="ij")
    values = np.stack([4.4 + 0.0002 * yy + 0.00001 * xx, 0.03 + 0.000001 * yy], axis=-1)
    table = wavecal.WaveTable(6, "t", x, y, values)
    header = {"CRPIX1W": -1597.0, "CRPIX2W": -237.0, "CDELT1W": 1, "CDELT2W": 1, "CRVAL1W": 1, "CRVAL2W": 1}
    wl, bw = table.evaluate(header, 10.3, 40.7)
    xd, yd = 10.3 + 1 + 1597 + 1, 40.7 + 1 + 237 + 1
    assert wl == pytest.approx(4.4 + 0.0002 * yd + 0.00001 * xd, abs=1e-9)
    assert bw == pytest.approx(0.03 + 0.000001 * yd, abs=1e-12)


# --------------------------------------------------------------------------- coordinates


@pytest.mark.parametrize("text, ra, dec", [
    ("269.4463 4.7682", 269.4463, 4.7682),
    ("269.4463, +4.7682", 269.4463, 4.7682),
    ("133.795129 -7.245147", 133.795129, -7.245147),
    ("08h55m10.831s -7d14m42.53s", 133.795129, -7.245147),     # as written in the challenge statement
    ("08:55:10.831 -07:14:42.53", 133.795129, -7.245147),
    ("17 57 48.5 +04 41 36", 269.452083, 4.693333),
])
def test_coordinate_formats(text, ra, dec):
    got = resolver.parse_coordinates(text)
    assert got is not None
    assert got[0] == pytest.approx(ra, abs=2e-5)
    assert got[1] == pytest.approx(dec, abs=2e-5)


@pytest.mark.parametrize("text", ["Barnard's Star", "R Dra", "M 31", "WISE 0855", "400 10", "10 95", ""])
def test_names_and_nonsense_are_not_coordinates(text):
    assert resolver.parse_coordinates(text) is None


def test_proper_motion_propagation_reproduces_gaia():
    # Barnard's Star: SIMBAD J2000 position + proper motion must land on Gaia DR3's epoch-2016 position.
    t = resolver.BUILTIN_TARGETS["barnard's star"]
    ra, dec = resolver.propagate(t["ra"], t["dec"], t["pmra"], t["pmdec"], 2000.0, 2016.0)
    gaia_ra, gaia_dec = 269.44850252543836, 4.739420051112412
    sep = math.hypot((ra - gaia_ra) * math.cos(math.radians(dec)), dec - gaia_dec) * 3600
    assert sep < 0.1


# --------------------------------------------------------------------------- image grid


def test_grid_is_north_up_east_left_and_centred():
    grid = process.Grid(150.0, 20.0, 6.0)
    ra, dec = grid.world
    n = grid.n
    assert grid.scale_arcsec == pytest.approx(360.0 / n)
    centre = (n - 1) / 2
    cra, cdec = grid.wcs.all_pix2world(centre, centre, 0)
    assert float(cra) == pytest.approx(150.0, abs=1e-9) and float(cdec) == pytest.approx(20.0, abs=1e-9)
    assert dec[0, n // 2] > dec[-1, n // 2]            # row 0 is the north edge
    assert ra[n // 2, 0] > ra[n // 2, -1]              # column 0 is the east edge
    assert (dec[0, n // 2] - dec[-1, n // 2]) * 60 == pytest.approx(6.0 * (n - 1) / n, rel=1e-3)


def _gauss(n, x0, y0, sigma, flux):
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return (flux * g / g.sum()).astype(np.float32)


def test_coadd_rejects_a_cosmic_ray_but_keeps_the_star():
    rng = np.random.default_rng(2)
    n = 48
    star = _gauss(n, 24, 24, 2.0, 400.0)
    images, bad = [], []
    for i in range(8):
        frame = star * rng.uniform(0.8, 1.2) + rng.normal(0, 0.02, (n, n)).astype(np.float32)
        if i == 3:
            frame[10, 10] += 5.0                       # a hit only this exposure sees
        images.append(frame.astype(np.float32))
        bad.append(np.zeros((n, n), dtype=np.float32))
    out, flag_fraction, count = process.coadd(images, bad, [1 / 0.02 ** 2] * 8, True)
    assert abs(out[10, 10]) < 0.1                      # cosmic ray gone
    assert count[10, 10] == 7
    assert out[24, 24] == pytest.approx(star[24, 24], rel=0.12)   # star untouched
    assert count[24, 24] == 8
    assert flag_fraction.max() == 0


def test_coadd_shows_fully_flagged_pixels_instead_of_holes():
    n = 16
    images = [np.full((n, n), 5.0, dtype=np.float32) for _ in range(4)]
    bad = [np.zeros((n, n), dtype=np.float32) for _ in range(4)]
    for b in bad:
        b[8, 8] = 1.0                                  # flagged in every exposure (e.g. saturated core)
    out, flag_fraction, count = process.coadd(images, bad, [1.0] * 4, True)
    assert out[8, 8] == pytest.approx(5.0)
    assert flag_fraction[8, 8] == 1.0 and count[8, 8] == 0


# --------------------------------------------------------------------------- photometry


def _exposure(image, mjd=60900.0, detector=3, flags=None, epoch=0, ra0=150.0, dec0=20.0):
    n = image.shape[0]
    header = fits.Header()
    header["NAXIS"] = 2
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = ra0, dec0
    header["CRPIX1"], header["CRPIX2"] = (n + 1) / 2, (n + 1) / 2
    header["CDELT1"], header["CDELT2"] = -config.NATIVE_PIXEL_ARCSEC / 3600, config.NATIVE_PIXEL_ARCSEC / 3600
    header["MJD-AVG"] = mjd
    header["PSF_FWHM"] = 7.0
    header["OMEGA_MEDIAN"] = config.NATIVE_PIXEL_ARCSEC ** 2
    cut = fetch.RawCutout(
        sci=image.astype(np.float32),
        variance=np.full(image.shape, 0.01 ** 2, dtype=np.float32),
        flags=np.zeros(image.shape, dtype=np.int32) if flags is None else flags,
        header=header, zodi_median=0.4, calib_id="t",
        flag_bits=dict(config.DEFAULT_FLAG_BITS), flag_descriptions={},
    )
    exp = Exposure(f"obs{mjd}", detector, "u", mjd, 100.0, "c", "")
    return ExposureData(exp=exp, cut=cut, epoch=epoch)


def _job():
    bits = config.DEFAULT_FLAG_BITS
    bad = sum(1 << bits[name] for name in config.BAD_FLAG_NAMES)
    return SimpleNamespace(
        bad_mask=bad, wave={}, flag_descriptions={},
        flag_names=lambda value: [k for k, b in bits.items() if value & (1 << b)],
    )


def test_psf_photometry_recovers_a_known_flux():
    rng = np.random.default_rng(3)
    n = 41
    sigma_pix = 7.0 / 2.3548 / config.NATIVE_PIXEL_ARCSEC
    flux_mjy = 12.0
    per_pixel = flux_mjy / (config.NATIVE_PIXEL_ARCSEC ** 2 * photometry.MJY_PER_MJYSR_ARCSEC2)
    xs = np.arange(n)
    image = per_pixel * photometry._psf_weights(20.3, 19.6, sigma_pix, xs, xs)
    image = image + rng.normal(0, 0.01, (n, n)) + 0.25           # sky pedestal must be removed
    ed = _exposure(image)
    ra, dec = ed.wcs.all_pix2world(20.3, 19.6, 0)
    point = photometry.measure(ed, _job(), float(ra), float(dec))
    assert point["flux"] == pytest.approx(flux_mjy, rel=0.03)
    assert not point["flagged"]
    assert point["snr"] > 50
    assert math.hypot(point["east"], point["north"]) < 2.5      # centroid lands on the source


def test_saturated_core_is_reported_as_unreliable():
    n = 41
    sigma_pix = 7.0 / 2.3548 / config.NATIVE_PIXEL_ARCSEC
    xs = np.arange(n)
    image = 5000.0 * photometry._psf_weights(20.0, 20.0, sigma_pix, xs, xs)
    flags = np.zeros((n, n), dtype=np.int32)
    flags[18:23, 18:23] = 1 << config.DEFAULT_FLAG_BITS["OVERFLOW"]
    image[19:22, 19:22] = np.nan
    ed = _exposure(image, flags=flags)
    point = photometry.measure(ed, _job(), 150.0, 20.0)
    assert point["flagged"]
    assert point["_sat_radius"] >= 2


def test_light_curve_ignores_wavelength_sampling_but_sees_real_change():
    rng = np.random.default_rng(4)
    # A source whose brightness falls steeply with wavelength, sampled at different
    # wavelengths in each pass. A naive mean per pass would show a false change.
    def spectrum(wl):
        return 100.0 * (wl / 1.6) ** -4

    wl = np.concatenate([np.linspace(1.63, 2.0, 20), np.linspace(1.9, 2.41, 20), np.linspace(1.63, 2.41, 20)])
    epoch = np.repeat([0, 1, 2], 20)
    mjd = 60800.0 + epoch * 180.0
    steady = spectrum(wl) * rng.normal(1.0, 0.01, wl.size)
    naive = [steady[epoch == e].mean() for e in (0, 1)]
    assert abs(naive[0] / naive[1] - 1) > 0.3                     # the trap is real
    series = photometry._relative_series(wl, steady, epoch, mjd)
    assert len(series) == 3
    for s in series:
        assert s["rel"] == pytest.approx(1.0, abs=0.05)

    varying = steady * np.where(epoch == 2, 1.5, 1.0)
    series = photometry._relative_series(wl, varying, epoch, mjd)
    rel = {s["epoch"]: s["rel"] for s in series}
    assert rel[2] / rel[0] == pytest.approx(1.5, rel=0.12)


def test_blend_flags_a_neighbour_but_not_the_target_itself():
    points = [{"mjd": 60800.0 + 180 * i, "_ra": 150.0, "_dec": 20.0, "blended": False, "blend_sep": None, "blend_g": None}
              for i in range(3)]
    arcsec = 1 / 3600
    stars = [
        {"ra": 150.0, "dec": 20.0, "pmra": 0, "pmdec": 0, "epoch": 2016.0, "g": 12.0},             # the target
        {"ra": 150.0, "dec": 20.0 + 4 * arcsec, "pmra": 0, "pmdec": 0, "epoch": 2016.0, "g": 13.0},  # real blend
        {"ra": 150.0, "dec": 20.0 - 5 * arcsec, "pmra": 0, "pmdec": 0, "epoch": 2016.0, "g": 18.5},  # negligible
        {"ra": 150.0, "dec": 20.0 + 60 * arcsec, "pmra": 0, "pmdec": 0, "epoch": 2016.0, "g": 10.0}, # far away
    ]
    summary = photometry._mark_blends(points, stars)
    assert summary["n"] == 3
    assert summary["closest_arcsec"] == pytest.approx(4.0, abs=0.05)
    assert summary["g"] == 13.0
    assert all(p["blended"] for p in points)

    alone = [dict(p, blended=False, blend_sep=None) for p in points]
    assert photometry._mark_blends(alone, [stars[0], stars[2], stars[3]]) is None


def test_measured_motion_recovers_a_known_drift():
    rng = np.random.default_rng(5)
    points = []
    for epoch, year in enumerate([2025.6, 2026.2, 2026.7]):
        for _ in range(30):
            mjd = 51544.5 + (year - 2000.0) * 365.25
            dt = year - 2026.0
            points.append({"epoch": epoch, "mjd": mjd, "snr": 50.0, "blended": False,
                           "east": -0.8 * dt + rng.normal(0, 1.0), "north": 10.4 * dt + rng.normal(0, 1.0)})
    motion = photometry._measured_motion(points)
    assert motion["significant"]
    assert motion["pm_north"] == pytest.approx(10.4, abs=0.8)
    assert motion["pm_east"] == pytest.approx(-0.8, abs=0.8)
    assert motion["total"] == pytest.approx(10.43, abs=0.8)

    still = [dict(p, east=rng.normal(0, 1.0), north=rng.normal(0, 1.0)) for p in points]
    assert not photometry._measured_motion(still)["significant"]


# --------------------------------------------------------------------------- live smoke test


@pytest.mark.live
def test_live_barnards_star_end_to_end(tmp_path, monkeypatch):
    """Talks to IRSA: finds exposures, downloads one band, builds frames."""
    import asyncio

    from app.stack import StackManager

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "MAX_EXPOSURES_PER_BAND", 8)

    async def run():
        manager = StackManager()
        await manager.start()
        try:
            target = await resolver.resolve(manager.client, "Barnard's Star")
            job = manager.get_job(target["ra"], target["dec"], 3.0, 2)
            await job.wait_for(lambda: job.finished, timeout=300)
            assert job.stage == "done", job.error
            assert len(job.data) >= 0.8 * job.total > 0
            assert len(job.epochs_summary()) >= 2
            header, data, flags = process.build_frames(job, 2, "epochs", True)
            assert data.shape[0] >= 2 and np.isfinite(data).mean() > 0.9
            assert all(frame["wl"] and 1.0 < frame["wl"] < 1.7 for frame in header["frames"])
        finally:
            await manager.stop()

    asyncio.run(run())


# --------------------------------------------------------------------------- colour, old coordinates, WISE


def test_challenge_coordinates_are_recognised_as_an_old_position():
    # The challenge statement gives WISE 0855's coordinates as WISE measured them around 2010.
    coords = resolver.parse_coordinates("08h55m10.831s -7d14m42.53s")
    hint = resolver.moved_target_near(*coords)
    assert hint is not None and hint["name"] == "WISE 0855-0714"
    assert 2009 <= hint["match_year"] <= 2012
    assert 110 < hint["moved_arcsec"] < 140                # about two arcminutes away by 2026

    # Its present position needs no hint, and neither does empty sky or a slow star.
    now = resolver.propagate(133.81890825, -7.247048, -8123.7, 673.2, 2000.0, config.VIEW_EPOCH_YEAR)
    assert resolver.moved_target_near(*now) is None
    assert resolver.moved_target_near(10.0, 10.0) is None
    assert resolver.moved_target_near(248.16761, 66.75498) is None       # R Dra barely moves


def _colour_job(exposures, ra0=150.0, dec0=20.0):
    job = _job()
    job.ra, job.dec, job.fov_arcmin = ra0, dec0, 3.0
    job.saturation_mask = 0
    job.band_exposures = lambda detector: [e for e in exposures if detector is None or e.exp.detector == detector]
    return job


def test_colour_frames_show_a_cold_object_red():
    rng = np.random.default_rng(6)
    n = 41
    sigma_pix = 7.0 / 2.3548 / config.NATIVE_PIXEL_ARCSEC
    xs = np.arange(n)
    star = photometry._psf_weights(12.0, 12.0, sigma_pix, xs, xs)      # ordinary star: all bands
    cold = photometry._psf_weights(28.0, 26.0, sigma_pix, xs, xs)      # cold object: long wavelengths only
    exposures = []
    for epoch, mjd0 in enumerate((60800.0, 60980.0)):
        for detector in config.DETECTORS:
            for k in range(3):
                image = 300.0 * star + (300.0 * cold if detector >= 5 else 0.0) + rng.normal(0, 0.02, (n, n))
                exposures.append(_exposure(image, mjd=mjd0 + detector + 0.1 * k, detector=detector, epoch=epoch))
    header, data, flags = process.build_colour_frames(_colour_job(exposures), True, exposures)
    assert header["channels"] == 3 and len(header["frames"]) == 2
    assert data.shape[0] == 6 and flags.shape[0] == 2                    # red, green, blue per pass
    assert [f["nexp"] for f in header["frames"]] == [18, 18]

    grid = process.Grid(150.0, 20.0, 3.0)

    def brightness(plane, x, y):
        ra, dec = exposures[0].wcs.all_pix2world(x, y, 0)
        col, row = grid.wcs.all_world2pix(ra, dec, 0)
        return float(plane[int(round(float(row))), int(round(float(col)))])

    red, green, blue = data[0], data[1], data[2]
    assert brightness(red, 12, 12) == pytest.approx(brightness(blue, 12, 12), rel=0.25)   # star: similar in every channel
    assert brightness(red, 28, 26) > 20 * abs(brightness(blue, 28, 26)) + 1.0             # cold object: red only
    assert brightness(red, 28, 26) > 20 * abs(brightness(green, 28, 26)) + 1.0


def test_wise_tiles_are_tried_nearest_first():
    from astropy.io.votable.tree import VOTableFile
    from astropy.table import Table

    from app import wise

    base = "https://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/x/"
    table = Table({
        "s_ra": [150.9, 150.1, 150.1, 150.4],
        "s_dec": [20.0, 20.05, 20.05, 20.3],
        "energy_bandpassname": ["W2", "W2", "W1", "W2"],
        "access_url": [base + "far-w2-int-3.fits", base + "near-w2-int-3.fits", base + "near-w1-int-3.fits", base + "mid-w2-unc-3.fits"],
    })
    buffer = io.BytesIO()
    VOTableFile.from_table(table).to_xml(buffer)
    urls = wise._parse_sia(buffer.getvalue(), "W2", 150.0, 20.0)
    assert urls == [base + "near-w2-int-3.fits", base + "far-w2-int-3.fits"]     # science images of the right band only


# --------------------------------------------------------------------------- catalogue names


def test_simbad_lookup_falls_back_to_the_second_service_and_parses_objects():
    import asyncio

    import httpx

    from app import catalogs

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "simbad.harvard.edu":
            return httpx.Response(404, text="<html>Page unavailable</html>", headers={"content-type": "text/html"})
        return httpx.Response(200, json={
            "metadata": [{"name": k} for k in ("main_id", "ra", "dec", "pmra", "pmdec", "otype", "description", "sp_type")],
            "data": [
                ["* eps Ind B", 331.04414, -56.78284, 3981.977, -2466.832, "BD*", "Brown Dwarf", "T1V+T6V"],
                ["LEDA  395751", 331.2, -56.8, None, None, "G", "Galaxy", None],
                [None, 331.3, -56.9, None, None, "*", "Star", None],            # no identifier: dropped
            ],
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await catalogs._simbad(client, 331.0966, -56.8007, 6.0)

    objects = asyncio.run(run())
    assert calls == ["simbad.harvard.edu", "simbad.cds.unistra.fr"]
    assert [o["name"] for o in objects] == ["* eps Ind B", "LEDA  395751"]
    dwarf = objects[0]
    assert dwarf["kind"] == "Brown Dwarf" and dwarf["sptype"] == "T1V+T6V" and dwarf["epoch"] == 2000.0
    assert dwarf["pmra"] == pytest.approx(3981.977)
    assert objects[1]["pmra"] is None and objects[1]["sptype"] is None


def test_catalogue_search_reaches_back_far_enough_for_fast_movers():
    from app import catalogs

    # SIMBAD positions are for the year 2000; Barnard's Star has moved about 4.5 arcminutes since.
    assert catalogs._field_radius_deg(6.0) * 60 == pytest.approx(3 * math.sqrt(2) + 0.5)
    assert (catalogs._field_radius_deg(6.0) + 5.0 / 60.0) * 60 > 4.5 + 3 * math.sqrt(2)


def test_centre_is_not_dragged_onto_a_brighter_neighbour():
    # A faint source with a star ten times brighter 20 arcsec away. The measured
    # centre must stay on the faint source in every exposure; if it jumps to the
    # neighbour, the motion fit reports a motion that is not there.
    rng = np.random.default_rng(6)
    n = 41
    sigma_pix = 7.0 / 2.3548 / config.NATIVE_PIXEL_ARCSEC
    xs = np.arange(n)
    points = []
    for epoch, year in enumerate([2025.4, 2025.9, 2026.4]):
        for k in range(12):
            jx, jy = rng.uniform(-0.5, 0.5, 2)                        # the source lands differently on the pixels each time
            image = 4.0 * photometry._psf_weights(20.0 + jx, 20.0 + jy, sigma_pix, xs, xs)
            image = image + 40.0 * photometry._psf_weights(20.0 + jx, 16.7 + jy, sigma_pix, xs, xs)
            image = image + rng.normal(0, 0.01, (n, n))
            ed = _exposure(image, mjd=51544.5 + (year - 2000.0) * 365.25 + k, epoch=epoch)
            ra, dec = ed.wcs.all_pix2world(20.0 + jx, 20.0 + jy, 0)
            point = photometry.measure(ed, _job(), float(ra), float(dec))
            assert point is not None and point["east"] is not None
            assert math.hypot(point["east"], point["north"]) < 6.0   # not 20 arcsec away on the neighbour
            points.append(point)
    motion = photometry._measured_motion(points)
    assert motion is None or not motion["significant"]


# --------------------------------------------------------------------------- archive outages


def test_partial_archive_answer_is_reported_as_partial(monkeypatch):
    """If one collection fails and another answers, the caller is told the list is not whole."""
    import asyncio

    arrived = Exposure("2026W34_1A_0001_1", 3, "u", 61270.0, 113.6, "spherex_qr3", "")

    async def fake(client, collection, ra, dec):
        if collection == config.SIA_COLLECTIONS[0]:
            raise RuntimeError("SIA HTTP 502")
        return [arrived] if collection == config.SIA_COLLECTIONS[1] else []

    monkeypatch.setattr(discovery, "_query_collection", fake)
    with pytest.raises(discovery.PartialSearch) as caught:
        asyncio.run(discovery.search_exposures(None, 10.0, 20.0))
    assert caught.value.exposures == [arrived]


def test_partial_archive_answer_never_shrinks_the_stored_list(monkeypatch, tmp_path):
    """Regression (15 Nov 2026): during an archive outage one collection answered, and its 119
    exposures replaced a stored list of 356, so Barnard's Star showed one pass instead of three."""
    import asyncio
    import json
    import os
    import time

    from app.stack import StackManager, _write_json

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    old = [Exposure(f"2025W33_2C_{i:04d}_1", 3, "u", 60900.0 + i, 113.6, "spherex_dr1", "") for i in range(5)]
    new = Exposure("2026W34_1A_0001_1", 3, "u", 61270.0, 113.6, "spherex_qr3", "")
    path = tmp_path / "exposures" / "10.00000_+20.00000.json"
    _write_json(path, [e.to_dict() for e in old])
    long_ago = time.time() - 10 * config.EXPOSURE_LIST_TTL_S
    os.utime(path, (long_ago, long_ago))

    async def partial(client, ra, dec):
        raise discovery.PartialSearch([new], [RuntimeError("SIA HTTP 502")])

    monkeypatch.setattr(discovery, "search_exposures", partial)
    manager = StackManager()
    got = asyncio.run(manager._exposure_list(10.0, 20.0))
    assert {e.key for e in got} == {e.key for e in old} | {new.key}
    assert len(json.loads(path.read_text("utf-8"))) == 5      # the stored list is untouched

    # For the next few minutes the archive is not asked again about a position on record.
    async def must_not_run(client, ra, dec):
        raise AssertionError("archive asked again too soon")

    monkeypatch.setattr(discovery, "search_exposures", must_not_run)
    assert len(asyncio.run(manager._exposure_list(10.0, 20.0))) == 5


def test_slow_archive_does_not_hold_up_a_position_on_record(monkeypatch, tmp_path):
    """With a list on record, the tool waits only briefly for a fresh one."""
    import asyncio
    import os
    import time

    from app.stack import StackManager, _write_json

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "SEARCH_PATIENCE_S", 0.05)
    old = [Exposure("2025W33_2C_0001_1", 3, "u", 60900.0, 113.6, "spherex_dr1", "")]
    path = tmp_path / "exposures" / "10.00000_+20.00000.json"
    _write_json(path, [e.to_dict() for e in old])
    long_ago = time.time() - 10 * config.EXPOSURE_LIST_TTL_S
    os.utime(path, (long_ago, long_ago))

    async def hangs(client, ra, dec):
        await asyncio.sleep(30)

    monkeypatch.setattr(discovery, "search_exposures", hangs)
    started = time.time()
    got = asyncio.run(StackManager()._exposure_list(10.0, 20.0))
    assert [e.key for e in got] == [old[0].key]
    assert time.time() - started < 5


# --------------------------------------------------------------------------- disk limit


def test_disk_limit_stops_large_cache_writes(tmp_path, monkeypatch):
    from app import diskbudget

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "DISK_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(diskbudget, "_checked_at", 0.0)
    monkeypatch.setattr(diskbudget, "_has_room", True)

    monkeypatch.setattr(config, "DISK_CACHE_MAX_BYTES", 0)          # no limit set
    assert diskbudget.has_room()

    (tmp_path / "cutouts").mkdir()
    (tmp_path / "cutouts" / "a.npz").write_bytes(b"x" * 4096)
    monkeypatch.setattr(config, "DISK_CACHE_MAX_BYTES", 8192)       # under the limit
    assert diskbudget.has_room()
    monkeypatch.setattr(config, "DISK_CACHE_MAX_BYTES", 1024)       # over the limit
    assert not diskbudget.has_room()


# --------------------------------------------------------------------------- following a clicked star


def _star(ra, dec, pmra, pmdec, epoch=2016.0, g=12.0):
    return {"ra": ra, "dec": dec, "pmra": pmra, "pmdec": pmdec, "epoch": epoch, "g": g}


def test_click_on_a_fast_star_is_followed():
    # A star moving 4.7 arcsec/yr, clicked where it was in the last of four passes.
    mjds = [60800.0, 60960.0, 61150.0, 61310.0]                    # about 1.4 years
    star = _star(331.0, -56.8, 4000.0, -2500.0)
    last_year = discovery.mjd_to_decimal_year(mjds[-1])
    ra_click, dec_click = resolver.propagate(star["ra"], star["dec"], star["pmra"], star["pmdec"], star["epoch"], last_year)

    followed = photometry.track_for_click([star], ra_click, dec_click, mjds)
    assert followed is not None
    ra, dec, motion = followed
    assert motion["pmra"] == 4000.0 and motion["pmdec"] == -2500.0
    assert motion["epoch"] == pytest.approx(last_year)
    assert abs(dec - dec_click) * 3600.0 < 0.1
    assert abs(ra - ra_click) * 3600.0 * math.cos(math.radians(dec)) < 0.1


def test_click_on_a_slow_star_or_empty_sky_is_not_followed():
    mjds = [60800.0, 60960.0, 61150.0, 61310.0]
    slow = _star(10.0, 20.0, 12.0, -8.0)
    assert photometry.track_for_click([slow], 10.0, 20.0, mjds) is None            # barely moves: fixed spot is right
    fast = _star(10.0, 20.0, 4000.0, 0.0)
    assert photometry.track_for_click([fast], 10.02, 20.02, mjds) is None          # the click is 100 arcsec from its path
    assert photometry.track_for_click([], 10.0, 20.0, mjds) is None
