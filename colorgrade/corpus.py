"""Film-corpus discovery, indexing, retrieval, and target construction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .media import find_film_stills, load_display_image, load_film_titles
from .losses import FilmTarget, _pixels, to_loss_space
from .style import MovieStyleEncoder


INDEX_VERSION = 2


def crop_black_borders(image: Tensor, threshold: float = 0.025,
                       variation: float = 0.015) -> Tensor:
    """Remove contiguous near-black, nearly uniform bars from a CHW image."""
    if image.ndim != 3:
        raise ValueError("crop_black_borders expects [3,H,W]")
    luma = (image * image.new_tensor((0.2126, 0.7152, 0.0722))[:, None, None]).sum(0)
    row_bar = (luma.mean(1) < threshold) & (luma.std(1) < variation)
    col_bar = (luma.mean(0) < threshold) & (luma.std(0) < variation)

    def edge_count(mask: Tensor, reverse: bool = False) -> int:
        values = mask.flip(0) if reverse else mask
        non_bar = (~values).nonzero()
        return int(non_bar[0]) if non_bar.numel() else 0

    top, bottom = edge_count(row_bar), edge_count(row_bar, True)
    left, right = edge_count(col_bar), edge_count(col_bar, True)
    h, w = image.shape[-2:]
    # Never let an unusually dark photograph be cropped aggressively.
    if top + bottom > int(0.35 * h):
        top = bottom = 0
    if left + right > int(0.35 * w):
        left = right = 0
    return image[:, top:h-bottom if bottom else h, left:w-right if right else w]


def _square_resize(image: Tensor, size: int = 224) -> Tensor:
    """Resize without stretching and pad with the image's edge-neutral mean."""
    _, h, w = image.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    image = F.interpolate(image[None], (nh, nw), mode="bilinear",
                          align_corners=False)[0]
    pad_h, pad_w = size - nh, size - nw
    fill = image.mean(dim=(1, 2), keepdim=True)
    canvas = fill.expand(3, size, size).clone()
    top, left = pad_h // 2, pad_w // 2
    canvas[:, top:top+nh, left:left+nw] = image
    return canvas


def photographic_features(image: Tensor) -> Tensor:
    """Exposure/contrast/chroma descriptor used alongside semantic retrieval."""
    px = image.permute(1, 2, 0).reshape(-1, 3)
    luma = (px * image.new_tensor((0.2126, 0.7152, 0.0722))).sum(1)
    chroma = torch.sqrt(((px - luma[:, None]) ** 2).mean(1) + 1e-8)
    lq = torch.quantile(luma, torch.linspace(0.02, 0.98, 16, device=image.device))
    cq = torch.quantile(chroma, torch.linspace(0.10, 0.95, 8, device=image.device))
    return torch.cat((lq, cq))


class FilmFeatureEncoder(nn.Module):
    """Frozen ImageNet encoder used only for semantic frame retrieval."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def forward(self, display: Tensor) -> Tensor:
        x = (display - self.mean) / self.std
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        semantic = self.layer4(self.layer3(f2)).mean((2, 3))
        semantic = F.normalize(semantic, dim=1)
        return semantic


def prepare_image(display: Tensor, size: int = 224) -> Tensor:
    image = display[0] if display.ndim == 4 else display
    return _square_resize(crop_black_borders(image), size)


def build_film_index(root: str = "film_corpus", output: str = "film_index.pt",
                     max_images: int = 4000, per_movie_cap: int = 64,
                     batch_size: int = 32,
                     film_only_csv: str | None = "movie_titles_acquisition_annotated.csv",
                     device: str = "cpu") -> dict:
    allowed = load_film_titles(film_only_csv) if film_only_csv else None
    paths = find_film_stills(root, per_movie_cap=per_movie_cap,
                             allowed_movies=allowed)
    random.Random(0).shuffle(paths)
    paths = paths[:max_images]
    encoder = FilmFeatureEncoder().to(device)
    semantics, photos, kept_paths, movies = [], [], [], []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start+batch_size]
        images, valid = [], []
        for path in batch_paths:
            try:
                image = prepare_image(load_display_image(path, size=None))
                images.append(image)
                valid.append(path)
            except Exception as error:
                print(f"skip {path}: {error}", flush=True)
        if not images:
            continue
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            semantic = encoder(batch)
        semantics.append(semantic.cpu())
        photos.append(torch.stack([photographic_features(x) for x in images]))
        kept_paths.extend(valid)
        movies.extend([Path(path).relative_to(root).parts[0] for path in valid])
        print(f"indexed {len(kept_paths)}/{len(paths)} frames", flush=True)

    semantic = torch.cat(semantics)
    photo = torch.cat(photos)
    grouped = defaultdict(list)
    for index, movie in enumerate(movies):
        grouped[movie].append(index)
    movie_names = sorted(grouped)
    index = {
        "version": INDEX_VERSION, "root": root, "paths": kept_paths,
        "movies": movies, "semantic": semantic, "photo": photo,
        "photo_mean": photo.mean(0), "photo_std": photo.std(0).clamp_min(1e-4),
        "movie_names": movie_names,
    }
    torch.save(index, output)
    print(f"wrote {output}: {len(kept_paths)} frames, {len(movie_names)} movie looks")
    return index


def load_or_build_index(path: str, device: str, **build_kwargs) -> dict:
    if Path(path).exists():
        index = torch.load(path, map_location="cpu", weights_only=True)
        if index.get("version") == 1:
            # One-time in-place migration: discard the retired generic ResNet
            # moment "style" fields while keeping costly semantic embeddings.
            for key in ("style", "style_mean", "style_std", "movie_style"):
                index.pop(key, None)
            index["version"] = INDEX_VERSION
            torch.save(index, path)
            print(f"migrated {path} to semantic-only index v{INDEX_VERSION}")
        if index.get("version") != INDEX_VERSION:
            raise ValueError(f"{path} has an unsupported index version")
        return index
    return build_film_index(output=path, device=device, **build_kwargs)


def retrieve_look(index: dict, source_display: Tensor,
                  encoder: FilmFeatureEncoder, top_frames: int = 64,
                  photo_weight: float = 0.15,
                  requested_movie: str | None = None) -> dict:
    prepared = prepare_image(source_display).unsqueeze(0).to(next(encoder.parameters()).device)
    with torch.no_grad():
        source_semantic = encoder(prepared)
    source_photo = photographic_features(prepare_image(source_display)).cpu()
    semantic_score = index["semantic"] @ source_semantic[0].cpu()
    photo_z = (index["photo"] - source_photo) / index["photo_std"]
    score = semantic_score - photo_weight * photo_z.square().mean(1).sqrt()

    movie_to_indices = defaultdict(list)
    for i, movie in enumerate(index["movies"]):
        movie_to_indices[movie].append(i)
    if not requested_movie:
        suggestions = sorted(index["movie_names"], key=lambda name:
            torch.topk(score[movie_to_indices[name]],
                       min(3, len(movie_to_indices[name]))).values.mean().item(),
            reverse=True)[:5]
        raise ValueError("--look is required; source content cannot determine the desired "
                         "aesthetic. Source-like suggestions: " + ", ".join(suggestions))
    matches = [name for name in index["movie_names"]
               if name.casefold() == requested_movie.casefold()]
    if not matches:
        raise ValueError(f"movie look not found in index: {requested_movie}")
    movie = matches[0]
    candidates = torch.tensor(movie_to_indices[movie])
    order = candidates[torch.argsort(score[candidates], descending=True)]
    chosen = order[:top_frames].tolist()
    return {
        "movie": movie,
        "paths": [index["paths"][i] for i in chosen],
        "all_movie_paths": [index["paths"][i] for i in movie_to_indices[movie]],
        "scores": score[chosen],
    }


def distribution_target(paths: list[str], per_image_pixels: int = 2000) -> FilmTarget:
    """Build colour-distribution statistics from selected reference frames."""
    chunks = []
    for path in paths:
        display = crop_black_borders(load_display_image(path, size=None)[0]).unsqueeze(0)
        pixels = _pixels(to_loss_space(display))
        if len(pixels) > per_image_pixels:
            pixels = pixels[torch.randperm(len(pixels))[:per_image_pixels]]
        chunks.append(pixels)
    return FilmTarget(torch.cat(chunks), max_pixels=len(chunks) * per_image_pixels)


def style_target(paths: list[str], encoder: MovieStyleEncoder,
                 views: int = 2) -> Tensor:
    """Average different scenes and shuffled views into one movie-look embedding."""
    embeddings = []
    device = next(encoder.parameters()).device
    for path in paths:
        image = prepare_image(load_display_image(path, size=None)).unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings.extend(encoder(image, destroy_content=True) for _ in range(views))
    return F.normalize(torch.cat(embeddings).mean(0), dim=0)


class MovieStyleLoss(nn.Module):
    """Compare a grade to a dedicated, contrastively learned movie-look embedding."""

    def __init__(self, encoder: MovieStyleEncoder, target: Tensor):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("target", target)

    def forward(self, display: Tensor) -> Tensor:
        batch = torch.stack([_square_resize(x, 224) for x in display])
        # Patch shuffling is differentiable and blocks composition shortcuts.
        style = self.encoder(batch, destroy_content=True)
        return (1.0 - style @ self.target).mean()
