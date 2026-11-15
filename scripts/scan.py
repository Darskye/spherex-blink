"""Search fields for candidates: sources that moved, changed brightness, are in
no catalogue, or were catalogued in 2010 and are not there now.

    python scripts/scan.py --examples                 # the five challenge targets: does the finder rediscover them?
    python scripts/scan.py "Wolf 359" "150.1 2.2"     # any names or coordinates
    python scripts/scan.py --file fields.txt          # one name or "ra dec" per line
    python scripts/scan.py --random 40 --seed 7       # random fields away from the Milky Way
    python scripts/scan.py --random 100 --keep-awake  # an all-night run (about 4 minutes per new field)

Each field costs the archive a few hundred small downloads (one to three
minutes) the first time, and nothing after that. Results go to
scan_results/<time>/report.json and candidates.csv; every candidate has a link
that opens it in the viewer. They are candidates to look at, not discoveries.

When the archive or a catalogue stops answering, the scan waits, carries on,
and comes back to the fields it could not finish (--retries, twice by default).
The last line of the log says how many fields were searched in full.

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config, finder, resolver  # noqa: E402
from app.main import EXAMPLES  # noqa: E402
from app.stack import StackManager  # noqa: E402


def keep_awake() -> bool:
    """Ask Windows not to fall asleep from idleness while this scan runs.

    Nothing is changed permanently: the request belongs to this process and
    ends with it. A closed lid, or choosing Sleep, still sleeps.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        continuous, system_required = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(continuous | system_required))
    except Exception:                                          # noqa: BLE001 - a convenience only
        return False


def random_fields(count: int, seed: int, min_lat: float) -> list[str]:
    """Random positions at least `min_lat` degrees from the plane of the Milky Way."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    rng = random.Random(seed)
    fields: list[str] = []
    while len(fields) < count:
        ra = rng.uniform(0.0, 360.0)
        dec = math.degrees(math.asin(rng.uniform(-1.0, 1.0)))
        if abs(SkyCoord(ra * u.deg, dec * u.deg).galactic.b.deg) >= min_lat:
            fields.append(f"{ra:.4f} {dec:+.4f}")
    return fields


FIELD_TIME_LIMIT_S = 25 * 60        # a field that takes longer than this is left for later (the archive is having a bad hour)
ARCHIVE_PAUSE_S = 5 * 60            # how long to wait when the archive stops answering
_log_file = None


def say(text: str = "") -> None:
    """Print, and keep a copy in the run's log file so progress can be read while it runs."""
    print(text, flush=True)
    if _log_file is not None:
        _log_file.write(text + "\n")
        _log_file.flush()


LATER = "later"                     # the archive did not answer: worth another try
NEVER = "never"                     # nothing to be had: a name nobody knows, or sky SPHEREx has not covered


async def scan_field(manager: StackManager, query: str, fov: float, fresh: bool) -> dict | str:
    """The report for one field, or LATER / NEVER when there is none."""
    try:
        target = await resolver.resolve(manager.client, query)
    except Exception as err:                                   # noqa: BLE001 - a name service being down must not stop a long scan
        say(f"  {query}: could not look the name up ({err.__class__.__name__})")
        return LATER
    if target is None:
        say(f"  could not understand {query!r}")
        return NEVER
    ra, dec = round(target["ra"], 5), round(target["dec"], 5)
    started = time.time()
    job = manager.get_job(ra, dec, fov, None)
    while not job.finished:
        if time.time() - started > FIELD_TIME_LIMIT_S:
            say(f"  {query}: still loading after {FIELD_TIME_LIMIT_S // 60} minutes ({job.done}/{job.total} exposures)")
            return LATER
        await asyncio.sleep(2.0)
    if job.stage == "error":
        say(f"  {query}: no data ({job.error})")
        return LATER
    if not job.data:
        say(f"  {query}: SPHEREx has not observed this position yet")
        return NEVER
    try:
        report = await asyncio.wait_for(finder.run(manager.client, job, fresh=fresh), timeout=FIELD_TIME_LIMIT_S)
    except Exception as err:                                   # noqa: BLE001 - one bad field must not stop a long scan
        say(f"  {query}: search failed ({err!r})")
        return LATER
    report = dict(report)
    report["query"] = query
    report["name"] = target.get("name") or query
    report["wall_seconds"] = round(time.time() - started, 1)
    return report


def viewer_link(base: str, report: dict, candidate: dict) -> str:
    f = report["field"]
    return (f"{base}/?ra={f['ra']:.5f}&dec={f['dec']:.5f}&fov={f['fov_arcmin']:g}"
            f"&sra={candidate['ra']:.5f}&sdec={candidate['dec']:.5f}")


def describe(report: dict, base: str) -> None:
    counts = report["counts"]
    say(f"  {report['name']}: {counts['sources']} sources, {report['field']['passes']} passes, "
          f"{counts['first_pass_flags']} first-pass flags ({counts['not_confirmed']} not confirmed) -> "
          f"{counts['unexplained']} unexplained, {counts['known']} already known, {counts['doubtful']} doubtful "
        f"[{report['wall_seconds']:.0f}s]")
    for c in report["candidates"]:
        mark = {"unexplained": "??", "known": "ok", "doubtful": " ~"}[c["verdict"]]
        say(f"    {mark} {c['title']}" + (f" - {c['name']}" if c.get("name") else "") + f"  [{c['kind']}]")
        say(f"       {c['explanation']}")
        if c["verdict"] == "unexplained":
            say(f"       {viewer_link(base, report, c)}")
    for note in report["notes"]:
        say(f"    note: {note}")


async def main() -> int:
    # The reports use symbols such as the arcsecond mark; a Windows console may not have them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="names or coordinates")
    parser.add_argument("--examples", action="store_true", help="the five challenge targets")
    parser.add_argument("--file", help="text file with one name or coordinate pair per line")
    parser.add_argument("--random", type=int, default=0, metavar="N", help="N random fields")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-lat", type=float, default=30.0, help="keep random fields this far from the Milky Way (degrees)")
    parser.add_argument("--fov", type=float, default=config.DEFAULT_FOV_ARCMIN)
    parser.add_argument("--fresh", action="store_true", help="search again even if a stored result exists")
    parser.add_argument("--keep-awake", action="store_true", help="on Windows, stop the computer idle-sleeping while the scan runs")
    parser.add_argument("--retries", type=int, default=2, help="how many more times to try fields the archive or a catalogue failed on")
    parser.add_argument("--out", default=str(ROOT / "scan_results"))
    parser.add_argument("--viewer", default="http://localhost:8765", help="address of the viewer, for the links")
    args = parser.parse_args()

    queries = list(args.targets)
    if args.examples:
        queries += [example["query"] for example in EXAMPLES]
    if args.file:
        queries += [line.strip() for line in Path(args.file).read_text("utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
    if args.random:
        queries += random_fields(args.random, args.seed, args.min_lat)
    if not queries:
        parser.print_help()
        return 2

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    global _log_file
    _log_file = open(out_dir / "log.txt", "w", encoding="utf-8")
    logging.getLogger("httpx").setLevel(logging.WARNING)       # one line per download would bury the progress
    say(f"Scanning {len(queries)} fields. Results so far are always in {out_dir}")
    if args.keep_awake:
        say("Idle sleep is held off while this runs." if keep_awake() else "Could not hold off idle sleep; keep the computer awake yourself.")
    began = time.time()
    manager = StackManager()
    await manager.start()
    found: dict[str, dict] = {}                                # query -> its latest report

    def complete(report: dict) -> bool:
        """Every catalogue answered, so nothing was left unsearched."""
        return all(count is not None for count in report["catalogues"].values())

    def save() -> None:
        reports = [found[q] for q in queries if q in found]
        (out_dir / "report.json").write_text(json.dumps(reports, indent=1, default=float), "utf-8")
        with open(out_dir / "candidates.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["field", "field_ra", "field_dec", "verdict", "kind", "finding", "ra", "dec", "name",
                             "explanation", "measurements", "cautions", "link"])
            for r in reports:
                for c in r["candidates"]:
                    writer.writerow([
                        r["name"], r["field"]["ra"], r["field"]["dec"], c["verdict"], c["kind"], c["title"],
                        f"{c['ra']:.5f}", f"{c['dec']:.5f}", c.get("name") or "", c["explanation"],
                        " ".join(c["facts"]), " ".join(c["caveats"]), viewer_link(args.viewer, r, c),
                    ])

    try:
        todo = list(dict.fromkeys(queries))
        for attempt in range(1 + args.retries):
            again: list[str] = []
            failures_in_a_row = 0
            for index, query in enumerate(todo, start=1):
                say(f"[{index}/{len(todo)}] {datetime.now():%H:%M} {query}")
                outcome = await scan_field(manager, query, args.fov, args.fresh)
                if isinstance(outcome, dict):
                    failures_in_a_row = 0
                    describe(outcome, args.viewer)
                    if query not in found or complete(outcome) or not complete(found[query]):
                        found[query] = outcome
                    if not complete(outcome):
                        again.append(query)                    # a catalogue was down: the field is only half searched
                    save()                                     # after every field, so a long scan can be stopped at any time
                elif outcome == LATER:
                    again.append(query)
                    failures_in_a_row += 1
                    if failures_in_a_row >= 3 and index < len(todo):
                        say(f"  The archive is not answering. Waiting {ARCHIVE_PAUSE_S // 60} minutes before going on.")
                        await asyncio.sleep(ARCHIVE_PAUSE_S)
                        failures_in_a_row = 0
            if not again or attempt == args.retries:
                break
            todo = again
            say(f"\n{len(todo)} fields were not searched in full. Trying them again in {ARCHIVE_PAUSE_S // 60} minutes.")
            await asyncio.sleep(ARCHIVE_PAUSE_S)
    finally:
        await manager.stop()

    reports = [found[q] for q in queries if q in found]
    full = [r for r in reports if complete(r)]
    total = {key: sum(r["counts"][key] for r in reports) for key in ("sources", "unexplained", "known", "doubtful")}
    say(f"\n{len(reports)} of {len(set(queries))} fields searched in {(time.time() - began) / 3600:.1f} hours "
        f"({len(full)} in full, with {sum(r['counts']['sources'] for r in full)} sources; "
        f"{len(reports) - len(full)} without every catalogue). "
        f"{total['sources']} sources measured in all: {total['unexplained']} unexplained, "
        f"{total['known']} already known, {total['doubtful']} doubtful.")
    say(f"Results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
