"""Image loading and ImageIO/FFmpeg video decoding."""

from __future__ import annotations

import numpy as np
import imageio.v3 as iio
import logging
from contextlib import contextmanager
import torch

from . import color


class _IgnoreExpectedFFmpegShutdown(logging.Filter):
    """Hide ImageIO's noisy single-frame shutdown warning, and nothing else."""

    def filter(self, record):
        return "We had to kill ffmpeg to stop it" not in record.getMessage()


@contextmanager
def _quiet_single_frame_shutdown():
    logger = logging.getLogger("imageio_ffmpeg")
    warning_filter = _IgnoreExpectedFFmpegShutdown()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)



def probe_video(path):
    meta = iio.immeta(path, plugin="FFMPEG", exclude_applied=False)
    w, h = meta.get("size", (None, None))
    pix_fmt = meta.get("pix_fmt", "")
    color_range = None
    if "(" in pix_fmt:
        tags = pix_fmt[pix_fmt.find("(") + 1: pix_fmt.find(")")]
        for t in (s.strip() for s in tags.split(",")):
            if t in ("tv", "pc"):
                color_range = t
    return {
        "width": w, "height": h, "pix_fmt": pix_fmt,
        "codec_name": meta.get("codec"), "fps": meta.get("fps"),
        "duration": meta.get("duration"), "color_range": color_range,
    }


def _timestamp_to_index(timestamp, fps):
    if isinstance(timestamp, (int, float)):
        seconds = float(timestamp)
    else:
        seconds = 0.0
        for p in str(timestamp).split(":"):
            seconds = seconds * 60 + float(p)
    return max(0, int(round(seconds * (fps or 25.0))))


def extract_frame(path, timestamp="00:00:01", full_16bit=True, size=None):
    """Extract one frame; ``size=(width, height)`` asks ffmpeg to resize while decoding."""
    info = probe_video(path)
    index = _timestamp_to_index(timestamp, info.get("fps"))
    dtype = "uint16" if full_16bit else "uint8"
    # ImageIO must terminate its still-running decoder after retrieving one
    # requested frame. On some HEVC MOV files it logs that expected cleanup as
    # a warning; suppress only that exact message while retaining real errors.
    with _quiet_single_frame_shutdown():
        frame = np.asarray(
            iio.imread(path, index=index, plugin="FFMPEG", dtype=dtype, size=size),
            dtype=np.float32)
    if full_16bit:
        bit_depth = 10
        pf = info.get("pix_fmt", "")
        for d in (16, 14, 12, 10, 8):
            if f"p{d}le" in pf or f"p{d}be" in pf:
                bit_depth = d
                break
        frame = (frame / float(1 << (16 - bit_depth))) / float((1 << bit_depth) - 1)
    else:
        frame = frame / 255.0
    return np.clip(frame, 0.0, 1.0).astype(np.float32), info


# Image and source-frame loading


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_display_image(path: str, size: int | None = 512) -> torch.Tensor:
    """Load a film still as display-encoded [1,3,H,W] in [0,1] (Rec.709/sRGB)."""
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3].astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if size is not None:
        t = torch.nn.functional.interpolate(t, size=(size, size),
                                             mode="area")
    return t


def load_flog2_frame(path: str, timestamp: str = "00:00:01",
                     max_side: int | None = 512,
                     decoder_resize: bool = False) -> torch.Tensor:
    """Extract an F-Log2 frame [1,3,H,W] via your extract module.

    Downscales so the longer side is at most `max_side`. A native movie frame is
    often 4K+ (a single 3840x2160 float32 frame is ~100 MB, and stacking several
    for the training batch multiplies that). The statistic-matching loss only
    reads pixel *distributions*, so grading at reduced resolution is loss-
    equivalent and dramatically cheaper. Set max_side=None to keep native res.

    Downscaling uses area interpolation on the DWG-linear signal (correct place
    to resample light), not on the F-Log2 code values, to avoid resampling
    artifacts across the log curve. We convert F-Log2 -> DWG-linear, resize,
    then re-encode to F-Log2 so the returned tensor is still F-Log2 as callers
    expect."""
    decode_size = None
    if decoder_resize and max_side is not None:
        info = probe_video(path)
        width, height = info.get("width"), info.get("height")
        if width and height and max(width, height) > max_side:
            scale = max_side / max(width, height)
            decode_size = (max(2, int(round(width * scale))),
                           max(2, int(round(height * scale))))
    flog2_numpy, _info = extract_frame(path, timestamp=timestamp, size=decode_size)
    frame = torch.from_numpy(flog2_numpy).permute(2, 0, 1).unsqueeze(0).float()

    if max_side is not None and decode_size is None:
        _, _, h, w = frame.shape
        long_side = max(h, w)
        if long_side > max_side:
            scale = max_side / long_side
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            # resample in linear light, then back to F-Log2
            lin = color.flog2_to_scene_linear(frame)
            lin = torch.nn.functional.interpolate(
                lin, size=(new_h, new_w), mode="area")
            # re-encode: scene-linear F-Gamut -> F-Log2 code. We only have the
            # The inverse encoder below restores the caller's expected format.
            frame = _scene_linear_to_flog2(lin)
    return frame


def _scene_linear_to_flog2(lin: torch.Tensor) -> torch.Tensor:
    """Inverse of ``color.flog2_to_scene_linear``."""
    a, b, c, d = color._FLOG2_A, color._FLOG2_B, color._FLOG2_C, color._FLOG2_D
    e, f = color._FLOG2_E, color._FLOG2_F
    # linear branch cut in *linear* domain corresponds to code < CUT2:
    lin_cut = (color._FLOG2_CUT2 - f) / e
    log_branch = c * torch.log10((a * lin.clamp_min(0) + b).clamp_min(1e-10)) + d
    lin_branch = e * lin + f
    return torch.where(lin < lin_cut, lin_branch, log_branch)


def load_film_titles(csv_path: str,
                     acquisition_types=("Film",),
                     title_col: str = "original_title",
                     type_col: str = "acquisition_type") -> set[str]:
    """Read the acquisition CSV and return the set of movie titles whose
    acquisition_type is in `acquisition_types` (default: photochemical Film only).

    Titles are normalized (stripped, case-folded) for robust folder matching.
    To also include, e.g., 'Mixed', pass acquisition_types=('Film','Mixed')."""
    import csv

    wanted = {t.strip().casefold() for t in acquisition_types}
    titles: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get(type_col, "").strip().casefold() in wanted:
                titles.add(row[title_col].strip().casefold())
    return titles


def find_film_stills(root: str = "film_corpus", verbose: bool = True,
                     per_movie_cap: int | None = None,
                     allowed_movies: set[str] | None = None):
    """Recursively find film stills under `root`, which is organized as
    `root/<movie_name>/*.{jpg,jpeg,png,tif,tiff,webp}` (any nesting depth).

    Returns a flat, sorted list of image paths grouped by their top-level
    movie directory.

    per_movie_cap : if set, keep at most this many paths PER movie. Bounds memory
        AND balances the corpus so no single film dominates the pooled look.
    allowed_movies : if set, only include movie folders whose (normalized) name
        is in this set. Use with load_film_titles(...) to restrict the corpus to
        photochemical-film acquisitions only. Folder names are matched
        case-insensitively against the CSV titles."""
    import os

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    per_movie: dict[str, list[str]] = {}
    n_files = 0
    skipped_movies: set[str] = set()
    for dirpath, _dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        movie = rel.split(os.sep)[0] if rel != "." else "(root)"
        # skip whole folders not in the allow-list (film-only filter)
        if allowed_movies is not None and movie != "(root)":
            if movie.strip().casefold() not in allowed_movies:
                skipped_movies.add(movie)
                continue
        bucket = per_movie.setdefault(movie, [])
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                if per_movie_cap is not None and len(bucket) >= per_movie_cap:
                    break                       # stop early; don't hold all paths
                bucket.append(os.path.join(dirpath, fn))
                n_files += 1
        if verbose and n_files and n_files % 50_000 == 0:
            print(f"  scanning... {n_files} kept so far", flush=True)

    # drop empty buckets (movies with no matching images)
    per_movie = {m: v for m, v in per_movie.items() if v}
    paths = sorted(p for hits in per_movie.values() for p in hits)
    if verbose and per_movie:
        counts = [len(v) for v in per_movie.values()]
        print(f"found {len(paths)} stills across {len(per_movie)} movies "
              f"(min {min(counts)}, max {max(counts)}, "
              f"mean {sum(counts)//len(counts)} per movie)")
        if allowed_movies is not None:
            print(f"  film-only filter: kept {len(per_movie)} movies, "
                  f"skipped {len(skipped_movies)} non-film/unmatched folders")
        if len(per_movie) <= 30:
            for movie in sorted(per_movie):
                print(f"    {len(per_movie[movie]):5d}  {movie}")
    return paths
