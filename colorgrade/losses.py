"""Distribution losses for matching an explicitly selected movie look.

Idea (distribution matching, no paired data): given graded-digital images and a
selected movie's stills (UNPAIRED), pull the graded colour/tone distribution
toward the reference distribution.

Three complementary terms, each handling the content-vs-look tradeoff:
    1. Sliced-Wasserstein in OKLab  -> primary distributional distance on the
                                       color cloud (a true metric, not binned
                                       histograms), perceptually weighted.
    2. Mean + covariance (Gram)     -> channel relationships; largely content-
                                       INvariant, where much film look lives.
    3. Luma quantile (tone CDF)     -> the film tone response (toe/shoulder).

Film targets are decoded from display RGB into linear Rec.709. Model outputs
should be compared in *unclipped* linear Rec.709 so invalid negative and
greater-than-one colours remain visible to both the loss and its gradients.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Loss color space: OKLab (perceptually uniform, differentiable)
# ---------------------------------------------------------------------------

def _linear_to_oklab_nchw(rgb: Tensor) -> Tensor:
    """Signed light-linear RGB -> OKLab without hiding out-of-gamut values."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    def signed_cbrt(v):
        return torch.sign(v) * v.abs().clamp_min(1e-12).pow(1 / 3)
    l_, m_, s_ = signed_cbrt(l), signed_cbrt(m), signed_cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.stack([L, a, bb], dim=1)


def to_loss_space(display_rgb: Tensor, gamma: float = 2.4) -> Tensor:
    """Display-encoded [0,1] (Rec.709 Gamma 2.4) -> OKLab for the loss.
    Undo the 2.4 display gamma to light-linear, then OKLab. Use the SAME call
    for both graded-digital and film images so they are compared consistently."""
    # Signed power is an invertible extension of the display curve.  Corpus
    # images are normally [0,1], while malformed inputs remain observable.
    lin = torch.sign(display_rgb) * display_rgb.abs().pow(gamma)
    return _linear_to_oklab_nchw(lin)


def linear_to_loss_space(linear_rgb: Tensor) -> Tensor:
    """Unclipped linear Rec.709 -> signed OKLab for model-output matching."""
    return _linear_to_oklab_nchw(linear_rgb)


def _pixels(x_oklab: Tensor) -> Tensor:
    """[B,3,H,W] -> [N,3] pooled point cloud."""
    return x_oklab.permute(0, 2, 3, 1).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Distance terms
# ---------------------------------------------------------------------------

def sliced_wasserstein(X: Tensor, Y: Tensor, n_proj: int = 64,
                       generator=None) -> Tensor:
    """X:[N,3], Y:[M,3]. Mean squared 1-D Wasserstein over random projections.

    SW needs the two sorted sets to be COMPARED elementwise, which requires
    equal counts. Instead of resampling a 500k-wide sorted tensor with
    F.interpolate every step (slow, and it holds large autograd buffers -> the
    step-0 memory blowup), we simply draw a random size-N subsample of Y once
    per call. This is a valid, cheap Monte-Carlo estimate of the same SW
    distance and keeps every intermediate at size [N, n_proj]."""
    N = X.shape[0]
    if Y.shape[0] > N:
        idx = torch.randint(0, Y.shape[0], (N,), device=Y.device, generator=generator)
        Y = Y[idx]
    elif Y.shape[0] < N:
        # rare: film pool smaller than the graded sample -> match by tiling
        reps = (N + Y.shape[0] - 1) // Y.shape[0]
        Y = Y.repeat(reps, 1)[:N]

    d = X.shape[1]
    dirs = torch.randn(n_proj, d, device=X.device, dtype=X.dtype, generator=generator)
    dirs = F.normalize(dirs, dim=1)
    xp = X @ dirs.T                       # [N, n_proj]
    yp = Y @ dirs.T                       # [N, n_proj]
    xs, _ = torch.sort(xp, dim=0)
    ys, _ = torch.sort(yp, dim=0)
    return (xs - ys).pow(2).mean()


def _mean_cov(X: Tensor):
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    return mu, cov


def _luma_quantiles(luma: Tensor, n: int = 256) -> Tensor:
    xs, _ = torch.sort(luma)
    return F.interpolate(xs.view(1, 1, -1), size=n, mode="linear",
                         align_corners=True).view(-1)


# ---------------------------------------------------------------------------
# Film target (precomputed over the whole corpus for a single global look)
# ---------------------------------------------------------------------------

class FilmTarget:
    """OKLab distribution statistics for retrieved frames of one movie."""

    def __init__(self, oklab_pixels: Tensor, max_pixels: int = 200_000):
        X = oklab_pixels
        if X.shape[0] > max_pixels:
            X = X[torch.randperm(X.shape[0])[:max_pixels]]
        self.pixels = X
        self.mean, self.cov = _mean_cov(X)
        self.luma_q = _luma_quantiles(X[:, 0], 256)


    def to(self, device):
        self.pixels = self.pixels.to(device)
        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        self.luma_q = self.luma_q.to(device)
        return self


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class FilmLook(nn.Module):
    """
    Statistic-matching loss comparing graded output against a FilmTarget.
    For training a model, use ``input_space='linear'`` and pass unclipped
    linear Rec.709. Display mode remains for backward compatibility.

    Weights (defaults: covariance + SW dominate, luma assists):
        w_sw, w_cov, w_luma
    subsample caps SW cost per step; n_proj controls SW resolution.
    """

    def __init__(self, film_target: FilmTarget, w_sw: float = 1.0,
                 w_cov: float = 1.0, w_luma: float = 0.5, n_proj: int = 64,
                 gamma: float = 2.4, subsample: int = 20_000,
                 input_space: str = "display"):
        super().__init__()
        if input_space not in ("display", "linear"):
            raise ValueError("input_space must be 'display' or 'linear'")
        self.film = film_target
        self.w_sw, self.w_cov, self.w_luma = w_sw, w_cov, w_luma
        self.n_proj, self.gamma, self.subsample = n_proj, gamma, subsample
        self.input_space = input_space

    def forward(self, graded: Tensor) -> dict:
        lab = (linear_to_loss_space(graded) if self.input_space == "linear"
               else to_loss_space(graded, self.gamma))
        X = _pixels(lab)
        if X.shape[0] > self.subsample:
            X = X[torch.randperm(X.shape[0], device=X.device)[:self.subsample]]
        Y = self.film.pixels

        L_sw = sliced_wasserstein(X, Y, self.n_proj) if self.w_sw else X.new_zeros(())
        mx, cx = _mean_cov(X)
        L_cov = (mx - self.film.mean).pow(2).mean() + (cx - self.film.cov).pow(2).mean()
        xq = _luma_quantiles(X[:, 0], 256)
        L_luma = (xq - self.film.luma_q).pow(2).mean()

        total = self.w_sw * L_sw + self.w_cov * L_cov + self.w_luma * L_luma
        return {"total": total, "sw": L_sw.detach(),
                "cov": L_cov.detach(), "luma": L_luma.detach()}
