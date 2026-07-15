"""Global, image-conditioned colour-grade model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .color import (
    REC709_LUMA, _to_nchw, apply_lut, apply_matrix, di_inverse, di_oetf,
    exposure_linear, flog2_to_dwg_linear, flog2_to_working,
    monotone_curve_knots, perceptual_grade_linear_dwg,
    white_balance_linear, working_to_rec709_display,
)

@dataclass
class GradeParameters:
    exposure_stops: Union[float, Tensor] = 0.0
    temperature: Union[float, Tensor] = 0.0
    tint: Union[float, Tensor] = 0.0
    color_matrix: Optional[Tensor] = None
    perceptual_saturation: Union[float, Tensor] = 1.0
    hue_radians: Union[float, Tensor] = 0.0


class GlobalTransform(nn.Module):
    """Apply the constrained global transform predicted by the network."""

    def __init__(self, input_flog2: bool = False):
        super().__init__()
        self.input_flog2 = input_flog2

    def forward(
        self,
        x: Tensor,
        params: GradeParameters,
        curve_y_knots: Tensor,
    ) -> Tensor:
        if self.input_flog2:
            linear = flog2_to_dwg_linear(x)
        else:
            linear = di_inverse(x)

        linear = exposure_linear(linear, params.exposure_stops)
        linear = white_balance_linear(
            linear, temperature=params.temperature, tint=params.tint
        )
        if params.color_matrix is not None:
            linear = apply_matrix(linear, params.color_matrix)

        linear = perceptual_grade_linear_dwg(
            linear, curve_y_knots=curve_y_knots,
            saturation_amount=params.perceptual_saturation,
            hue_radians=params.hue_radians,
        )
        return di_oetf(linear)


# ============================================================================
# Image-conditioned, but spatially constant, grading
# ============================================================================

def frame_colour_statistics(display_rgb: Tensor) -> Tensor:
    """Return compact, differentiable whole-frame colour statistics.

    The returned vector deliberately has no pixel coordinates.  It complements
    a semantic image encoder with the quantities a colourist would inspect:
    RGB correlation, luma distribution, chroma distribution, and the relative
    amount of shadow, midtone, and highlight content.  Every item is computed
    per image, so it is safe to feed to a head predicting one constant grade.
    """
    image, _ = _to_nchw(display_rgb)
    B = image.shape[0]
    px = image.permute(0, 2, 3, 1).reshape(B, -1, 3)
    mean = px.mean(dim=1)
    centered = px - mean[:, None, :]
    cov = centered.transpose(1, 2).bmm(centered) / max(px.shape[1] - 1, 1)

    luma = (px * image.new_tensor(REC709_LUMA)).sum(dim=-1)
    chroma = torch.sqrt(((px - luma[..., None]) ** 2).mean(dim=-1) + 1e-8)
    # Sorting gives a differentiable-almost-everywhere quantile representation.
    luma_q = F.interpolate(torch.sort(luma, dim=1).values[:, None], size=32,
                           mode="linear", align_corners=True).squeeze(1)
    chroma_q = F.interpolate(torch.sort(chroma, dim=1).values[:, None], size=16,
                             mode="linear", align_corners=True).squeeze(1)
    zones = torch.stack((
        (luma < 0.2).to(image.dtype).mean(dim=1),
        ((luma >= 0.2) & (luma < 0.8)).to(image.dtype).mean(dim=1),
        (luma >= 0.8).to(image.dtype).mean(dim=1),
    ), dim=1)
    return torch.cat((mean, cov.flatten(1), luma_q, chroma_q, zones), dim=1)


class SceneEncoder(nn.Module):
    """Small dependency-free scene encoder used by :class:`ImageConditionedColorGrade`.

    It is intentionally replaceable: for production, substitute a frozen
    self-supervised encoder (for example DINOv2) with the same ``[B, D]``
    output contract.  The default keeps this repository runnable without a
    model download.
    """

    def __init__(self, out_features: int = 256):
        super().__init__()
        channels = (3, 32, 64, 128, 192)
        layers = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers.extend((
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.GELU(),
            ))
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Linear(channels[-1], out_features)
        self.out_features = out_features

    def forward(self, display_rgb: Tensor) -> Tensor:
        x, _ = _to_nchw(display_rgb)
        return self.projection(self.features(x).flatten(1))


class ImageConditionedColorGrade(nn.Module):
    """Predict one global, image-specific grade and residual LUT blend.

    The model predicts a separate set of controls for each image in a batch,
    but applies each set uniformly to all of that image's pixels.  It therefore
    preserves the requested *constant-grade* constraint while allowing a
    portrait, night exterior, or high-key scene to select different grading
    decisions.

    The photographic parameterization uses one shared monotone luminance
    curve, a neutral-preserving linear colour matrix, perceptual chroma/hue,
    and a tightly bounded residual LUT.  It cannot create the independent RGB
    highlight curves that caused the earlier neon-green failure.
    """

    def __init__(self, curve_knots: int = 9, lut_size: int = 17,
                 lut_count: int = 8, encoder: Optional[nn.Module] = None,
                 encoder_features: int = 256, input_flog2: bool = False):
        super().__init__()
        if curve_knots < 2 or lut_size < 2 or lut_count < 1:
            raise ValueError("curve_knots >= 2, lut_size >= 2, and lut_count >= 1 are required")
        self.input_flog2 = input_flog2
        self.curve_knots = curve_knots
        self.lut_count = lut_count
        self.encoder = encoder if encoder is not None else SceneEncoder(encoder_features)
        # 3 mean + 9 covariance + 32 luma quantiles + 16 chroma quantiles + 3 zones
        stats_features = 63
        if encoder is not None and not hasattr(encoder, "out_features"):
            raise ValueError("A custom encoder must expose its output width as .out_features")
        encoded_features = getattr(self.encoder, "out_features", encoder_features)
        head_in = encoded_features + stats_features
        # exposure/temperature/tint + neutral matrix + chroma/hue + shared curve
        self.parameter_count = 3 + 9 + 2 + (curve_knots - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, self.parameter_count + lut_count),
        )
        # Zero output means the identity controls and an equal LUT mixture.
        # This makes the model safe to insert into an existing grading path.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # Tiny zero-mean differences break LUT-mixture symmetry while their
        # equal initial blend remains exactly zero (identity).
        lut_init = torch.randn(lut_count, lut_size, lut_size, lut_size, 3) * 1e-4
        lut_init -= lut_init.mean(dim=0, keepdim=True)
        self.residual_lut_bank = nn.Parameter(lut_init)
        self.grade = GlobalTransform(input_flog2=input_flog2)

    def _display_for_conditioning(self, x: Tensor) -> Tensor:
        if self.input_flog2:
            return working_to_rec709_display(flog2_to_working(x), clamp=True)
        return working_to_rec709_display(x, clamp=True)

    def predicted_parameters(self, x: Tensor) -> Tuple[GradeParameters, Tensor, Tensor]:
        """Return ``(GradeParameters, curve_knots, LUT blend weights)`` for ``x``."""
        display = self._display_for_conditioning(x)
        raw = self.head(torch.cat((self.encoder(display), frame_colour_statistics(display)), dim=1))
        p, lut_logits = raw[:, :self.parameter_count], raw[:, self.parameter_count:]
        i = 0
        def take(n):
            nonlocal i
            out = p[:, i:i + n]
            i += n
            return out
        params = GradeParameters(
            exposure_stops=2.0 * torch.tanh(take(1).squeeze(1)),
            temperature=1.5 * torch.tanh(take(1).squeeze(1)),
            tint=1.5 * torch.tanh(take(1).squeeze(1)),
        )
        raw_matrix = 0.12 * torch.tanh(take(9).reshape(-1, 3, 3))
        # Each row of the residual sums to zero: M*[1,1,1] = [1,1,1].
        raw_matrix = raw_matrix - raw_matrix.mean(dim=-1, keepdim=True)
        eye = torch.eye(3, dtype=p.dtype, device=p.device).unsqueeze(0)
        params.color_matrix = eye + raw_matrix
        params.perceptual_saturation = torch.exp(0.45 * torch.tanh(take(1).squeeze(1)))
        params.hue_radians = 0.35 * torch.tanh(take(1).squeeze(1))
        shared_curve = monotone_curve_knots(take(self.curve_knots - 1))
        # Explicit [B,3,K] avoids the B==3 ambiguity of a two-dimensional curve.
        curve = shared_curve[:, None, :].expand(-1, 3, -1)
        return params, curve, torch.softmax(lut_logits, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        params, curve, weights = self.predicted_parameters(x)
        return self.apply_predicted(x, params, curve, weights)

    def apply_predicted(self, x: Tensor, params: GradeParameters,
                        curve: Tensor, weights: Tensor) -> Tensor:
        """Apply already-predicted controls to ``x``.

        This is useful for full-resolution tiled rendering: predict once from a
        small copy of the complete frame, then reuse the exact same global
        controls for every native-resolution tile.
        """
        working = self.grade(x, params, curve_y_knots=curve)
        # A convex blend keeps LUT selection stable; tanh bounds residual strength.
        residual = torch.einsum("bk,k...c->b...c", weights, self.residual_lut_bank)
        return working + 0.03 * torch.tanh(apply_lut(working, residual))
