"""Apply a trained ImageConditionedColorGrade checkpoint at native resolution.

The model chooses one grade from a small copy of the complete frame.  That
same transform is then applied in horizontal tiles to the full-resolution
source, so there are no spatially varying grading decisions or tile seams.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from colorgrade import color
from colorgrade.model import GradeParameters, ImageConditionedColorGrade
from colorgrade.media import extract_frame, load_flog2_frame


def move_grade_parameters(params: GradeParameters, device: str) -> GradeParameters:
    values = {}
    for name in params.__dataclass_fields__:
        value = getattr(params, name)
        values[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return GradeParameters(**values)


def render_full_resolution(model: ImageConditionedColorGrade, source: str,
                           timestamp, output_path: str, device: str,
                           conditioning_size: int = 384,
                           tile_height: int = 256) -> None:
    """Render one model-selected constant grade onto a native-resolution frame."""
    model.to(device).eval()
    conditioning = load_flog2_frame(
        source, timestamp=timestamp, max_side=conditioning_size,
        decoder_resize=True).to(device)
    with torch.no_grad():
        params, curve, weights = model.predicted_parameters(conditioning)

    full, info = extract_frame(source, timestamp=timestamp)
    height, width = full.shape[:2]
    output = np.empty((height, width, 3), dtype=np.uint8)
    params = move_grade_parameters(params, device)
    curve, weights = curve.to(device), weights.to(device)

    print(f"applying one constant grade to {width}x{height} on {device}", flush=True)
    with torch.no_grad():
        for top in range(0, height, tile_height):
            bottom = min(top + tile_height, height)
            tile = torch.from_numpy(full[top:bottom]).permute(2, 0, 1).unsqueeze(0).to(device)
            working = model.apply_predicted(tile, params, curve, weights)
            display = color.working_to_rec709_display(working, clamp=True)
            rendered = display[0].permute(1, 2, 0).cpu().numpy()
            output[top:bottom] = (rendered.clip(0, 1) * 255).round().astype(np.uint8)
            print(f"  rows {top}:{bottom}/{height}", flush=True)

    iio.imwrite(output_path, output)
    print(f"wrote {output_path}")
    print("source:", info.get("width"), "x", info.get("height"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="input1.MOV")
    parser.add_argument("--timestamp", default="00:00:01")
    parser.add_argument("--checkpoint", default="conditioned_grade_v2.pt")
    parser.add_argument("--output", default=None)
    parser.add_argument("--conditioning-size", type=int, default=384)
    parser.add_argument("--tile-height", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output is None:
        checkpoint = Path(args.checkpoint)
        stem = checkpoint.stem
        stem = stem[:-len("__grade")] if stem.endswith("__grade") else stem
        args.output = str(checkpoint.with_name(stem + "__fullres.png"))

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else \
            ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    model = ImageConditionedColorGrade(input_flog2=True)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and blob.get("architecture_version") != 2:
        raise SystemExit(
            "This checkpoint uses the retired independent-RGB architecture. "
            "Retrain it with train_grade.py before full-resolution rendering."
        )
    if isinstance(blob, dict) and blob.get("training_objective_version") != 2:
        raise SystemExit(
            "This checkpoint was trained with the retired generic ImageNet-style loss. "
            "Retrain it with train_grade.py and movie_style_encoder.pt."
        )
    model.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
    model.to(device).eval()
    chosen_movie = blob.get("selected_movie", "not recorded") if isinstance(blob, dict) else "not recorded"
    print(f"\nCHOSEN MOVIE LOOK: {chosen_movie}\n", flush=True)
    print(f"FULL-RESOLUTION OUTPUT: {args.output}\n", flush=True)
    render_full_resolution(model, args.source, args.timestamp, args.output,
                           device, args.conditioning_size, args.tile_height)


if __name__ == "__main__":
    main()
