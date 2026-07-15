"""Train the image-conditioned, spatially constant colour-grade model.

Example:
    uv run python train_grade.py input1.MOV --look "Moonrise Kingdom (2012)"
"""

import argparse
from pathlib import Path
import re
import unicodedata

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from colorgrade import color
from colorgrade.model import ImageConditionedColorGrade
from colorgrade.media import load_flog2_frame, probe_video
from colorgrade.losses import FilmLook
from colorgrade.corpus import (FilmFeatureEncoder, MovieStyleLoss,
                               distribution_target, load_or_build_index,
                               style_target, retrieve_look)
from colorgrade.style import load_style_encoder
from apply_grade import render_full_resolution


def filename_slug(value: str) -> str:
    """Convert a human label into a readable, shell-friendly filename part."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def default_output_paths(source: str, movie: str, timestamp,
                         output_dir: str) -> dict[str, Path]:
    prefix = "__".join((filename_slug(Path(source).stem),
                        filename_slug(movie),
                        "t" + filename_slug(timestamp)))
    root = Path(output_dir)
    return {
        "checkpoint": root / f"{prefix}__grade.pt",
        "preview": root / f"{prefix}__preview.png",
        "fullres": root / f"{prefix}__fullres.png",
    }


def structural_loss(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """Keep scene edges while allowing a global tone and colour transform.

    Each gradient map is normalized independently, making this mostly
    insensitive to an intended exposure/contrast change while discouraging
    clipped or flattened images.
    """
    w = before.new_tensor(color.REC709_LUMA).view(1, 3, 1, 1)
    def edges(x):
        y = (x * w).sum(1, keepdim=True)
        dx, dy = y[..., :, 1:] - y[..., :, :-1], y[..., 1:, :] - y[..., :-1, :]
        e = torch.cat((F.pad(dx, (0, 1)), F.pad(dy, (0, 0, 0, 1))), dim=1)
        return (e - e.mean((1, 2, 3), keepdim=True)) / e.std((1, 2, 3), keepdim=True).clamp_min(1e-4)
    return F.mse_loss(edges(after), edges(before))


def load_training_frames(source: str, count: int, max_side: int,
                         timestamp="00:00:01") -> torch.Tensor:
    info = probe_video(source)
    duration = float(info.get("duration") or 1.0)
    # Avoid potentially incomplete first/last frames.
    # A single-frame grade is the normal first use case; one second is a
    # reliable seek point for camera MOV files and matches the preview script.
    seconds = [timestamp] if count == 1 else \
        torch.linspace(duration * 0.10, duration * 0.80, count).tolist()
    frames = []
    for n, sec in enumerate(seconds, 1):
        print(f"loading source frame {n}/{count} at {sec}", flush=True)
        # Decode at training resolution: 6K camera frames otherwise make each
        # ffmpeg extraction unnecessarily memory-heavy before PyTorch downsizes.
        frames.append(load_flog2_frame(source, timestamp=sec, max_side=max_side,
                                       decoder_resize=True))
    return torch.cat(frames, dim=0)


def save_preview(model: ImageConditionedColorGrade, source: str, output: str,
                 timestamp: float, device: str, max_side: int):
    frame = load_flog2_frame(source, timestamp=timestamp, max_side=max_side,
                             decoder_resize=True).to(device)
    model.eval()
    with torch.no_grad():
        display = color.working_to_rec709_display(model(frame), clamp=True)
    image = (display[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    iio.imwrite(output, image)
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="input1.MOV")
    parser.add_argument("--timestamp", default="00:00:01")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--index", default="film_index.pt")
    parser.add_argument("--index-images", type=int, default=4000)
    parser.add_argument("--look", required=True,
                        help="movie folder defining the desired aesthetic")
    parser.add_argument("--style-encoder", default="movie_style_encoder.pt")
    parser.add_argument("--retrieved-frames", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--preview", default=None)
    parser.add_argument("--fullres-output", default=None)
    parser.add_argument("--tile-height", type=int, default=256)
    args = parser.parse_args()

    defaults = default_output_paths(args.source, args.look, args.timestamp,
                                    args.output_dir)
    args.checkpoint = str(Path(args.checkpoint) if args.checkpoint else defaults["checkpoint"])
    args.preview = str(Path(args.preview) if args.preview else defaults["preview"])
    args.fullres_output = str(Path(args.fullres_output) if args.fullres_output else defaults["fullres"])
    for path in (args.checkpoint, args.preview, args.fullres_output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    frames = load_training_frames(args.source, args.frames, args.max_side,
                                  args.timestamp).to(device)
    feature_encoder = FilmFeatureEncoder().to(device).eval()
    index = load_or_build_index(args.index, device=device,
                                max_images=args.index_images)
    source_display = color.working_to_rec709_display(
        color.flog2_to_working(frames[:1]), clamp=True)
    selection = retrieve_look(index, source_display, feature_encoder,
                              top_frames=args.retrieved_frames,
                              requested_movie=args.look)
    print(f"\nCHOSEN MOVIE LOOK: {selection['movie']}\n"
          f"using {len(selection['paths'])} related frames", flush=True)
    print(f"outputs:\n  grade:   {args.checkpoint}\n"
          f"  preview: {args.preview}\n  fullres: {args.fullres_output}", flush=True)
    target = distribution_target(selection["paths"]).to(device)
    style_encoder = load_style_encoder(args.style_encoder, device)
    target_style = style_target(selection["all_movie_paths"],
                                style_encoder).to(device)
    model = ImageConditionedColorGrade(input_flog2=True).to(device)
    look_loss = FilmLook(target, w_sw=0.25, w_cov=0.25, w_luma=0.10,
                         n_proj=16, subsample=4_096,
                         input_space="linear").to(device)
    movie_style_loss = MovieStyleLoss(style_encoder, target_style).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.train()
    for step in range(1, args.steps + 1):
        index = torch.randint(frames.shape[0], (min(args.batch_size, frames.shape[0]),), device=device)
        source = frames[index]
        before = color.working_to_rec709_display(color.flog2_to_working(source), clamp=True)
        working = model(source)
        # The corpus loss sees unclipped linear values. Out-of-range channels
        # therefore remain costly and differentiable instead of disappearing
        # behind a display clamp.
        after_linear = color.working_to_rec709_linear(working)
        after = color.working_to_rec709_display(working, clamp=True)
        distribution = look_loss(after_linear)
        learned_style = movie_style_loss(after)
        structure = structural_loss(before, after)
        total = 0.5 * learned_style + distribution["total"] + 0.10 * structure
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step {step:4d} total {total.item():.5f} "
                  f"feature_style {learned_style.item():.5f} "
                  f"distribution {distribution['total'].item():.5f} "
                  f"structure {structure.item():.5f}", flush=True)

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "frames": args.frames,
                "steps": args.steps, "input_flog2": True,
                "architecture_version": 2,
                "training_objective_version": 2,
                "selected_movie": selection["movie"],
                "source": args.source, "timestamp": args.timestamp,
                "style_encoder": args.style_encoder}, args.checkpoint)
    print(f"wrote {args.checkpoint}")
    save_preview(model, args.source, args.preview, args.timestamp, device,
                 args.max_side)
    print(f"\nFINAL MOVIE LOOK: {selection['movie']}\n", flush=True)
    render_full_resolution(model, args.source, args.timestamp,
                           args.fullres_output, device,
                           conditioning_size=384,
                           tile_height=args.tile_height)


if __name__ == "__main__":
    main()
