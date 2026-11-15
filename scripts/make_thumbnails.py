"""Render the example cards' thumbnails from real SPHEREx data.

Each thumbnail is the false-colour view of the example's latest survey pass,
made with the same recipe as the viewer (stars balanced to white, cold objects
red). Run after `prewarm.py`, so the data is already cached:

    python scripts/make_thumbnails.py

Developed with AI assistance (Claude, Anthropic). See README "Use of AI".
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import process, resolver  # noqa: E402
from app.main import EXAMPLES  # noqa: E402
from app.stack import StackManager  # noqa: E402

OUT = ROOT / "frontend" / "assets" / "examples"


def robust(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    med = float(np.median(finite))
    return med, float(1.4826 * np.median(np.abs(finite - med)))


def false_colour(
    red: np.ndarray, green: np.ndarray, blue: np.ndarray, flagged: np.ndarray | None = None,
    white: float = 58.0, black: float = -1.5,
) -> np.ndarray:
    """Same recipe as frontend/js/viewer.js: balance to white stars (one vote per star),
    asinh intensity, colour fading out in the noise, flagged pixels neutral."""
    from scipy import ndimage

    mean0 = (red + green + blue) / 3
    med, sigma = robust(mean0)
    clean = np.isfinite(mean0) if flagged is None else np.isfinite(mean0) & ~flagged
    peaks = clean & (mean0 > med + 15 * sigma) & (mean0 == ndimage.maximum_filter(np.nan_to_num(mean0, nan=-np.inf), size=5))
    box = np.ones((3, 3))
    total = ndimage.convolve(np.nan_to_num(mean0), box, mode="nearest")
    channels = []
    for channel in (red, green, blue):
        if peaks.sum() >= 6:
            ratio = float(np.median(ndimage.convolve(np.nan_to_num(channel), box, mode="nearest")[peaks] / total[peaks]))
        else:
            bright = clean & (mean0 > med + 12 * sigma)
            ratio = float(np.median(channel[bright] / mean0[bright])) if bright.sum() >= 6 else 1.0
        level, _ = robust(channel)
        channels.append((channel - level) / max(ratio, 0.05))
    intensity = sum(channels) / 3
    med, sigma = robust(intensity)
    lo, span = med + black * sigma, white * sigma
    above = np.clip(intensity - lo, 0, None)
    tone = np.clip(np.arcsinh(8 * above / span) / np.arcsinh(8), 0, 1)
    saturation = np.clip((intensity - med) / (6 * sigma), 0, 1)
    if flagged is not None:
        saturation = np.where(flagged, 0.0, saturation)
    planes = []
    for channel in channels:
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(above > 0, (channel - lo) / above, 1.0)
        planes.append(tone * (1 + saturation * (np.clip(rel, 0, 3) - 1)))
    rgb = np.stack(planes, axis=-1)
    peak = np.nanmax(rgb, axis=-1, keepdims=True)
    rgb = np.where(peak > 1, rgb / peak, rgb)
    return np.nan_to_num(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manager = StackManager()
    await manager.start()
    try:
        for example in EXAMPLES:
            target = await resolver.resolve(manager.client, example["query"])
            job = manager.get_job(round(target["ra"], 5), round(target["dec"], 5), float(example["fov"]), None)
            await job.wait_for(lambda: job.finished, timeout=900)
            if job.stage == "error":
                print(f"{example['id']}: FAILED {job.error}")
                continue
            header, data, flags = process.build_colour_frames(job, True)
            if not header["frames"]:
                print(f"{example['id']}: no colour frames")
                continue
            last = len(header["frames"]) - 1
            image = false_colour(data[3 * last], data[3 * last + 1], data[3 * last + 2], flags[last] >= 128)
            n = image.shape[0]
            crop = image[n // 6: n - n // 6, n // 6: n - n // 6]          # centre two thirds
            path = OUT / f"{example['id']}.jpg"
            Image.fromarray(crop).resize((208, 208), Image.LANCZOS).save(path, quality=88)
            print(f"{example['id']}: {path.relative_to(ROOT)} ({header['frames'][last]['date']})")
    finally:
        await manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
