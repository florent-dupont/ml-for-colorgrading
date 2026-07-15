"""Differentiable colour science and pixel operations.

Images pass through F-Log2/F-Gamut, scene-linear DaVinci Wide Gamut,
DaVinci Intermediate, and linear/display Rec.709 representations.

Tensor conventions
------------------
Accepted shapes:
    (3, H, W), (B, 3, H, W), (H, W, 3), (B, H, W, 3)

Internally all operations use NCHW. Inputs and outputs retain their original shape.

Pipeline
--------
F-Log2 code
    -> scene-linear F-Gamut
    -> scene-linear DWG
    -> linear-light controls (exposure, white balance)
    -> shared perceptual lightness/chroma transform
    -> DaVinci Intermediate and bounded residual LUT
    -> Rec.709 / Gamma 2.4 display transform
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

_FLOG2_CUT2 = 0.100686685370811
_FLOG2_A = 5.555556
_FLOG2_B = 0.064829
_FLOG2_C = 0.245281
_FLOG2_D = 0.384316
_FLOG2_E = 8.799461
_FLOG2_F = 0.092864

_DI_A = 0.0075
_DI_B = 7.0
_DI_C = 0.07329248
_DI_M = 10.44426855
_DI_LIN_CUT = 0.00262409
_DI_LOG_CUT = 0.02740668

DISPLAY_GAMMA = 2.4

M_SRC_TO_WORKING = (
    (0.8921121209464459, 0.02436917587121653, 0.08351870318233734),
    (0.032616601764084874, 0.7861375169041127, 0.18124588133180272),
    (0.06997705118563409, 0.10474949190382742, 0.8252734569105387),
)

M_WORKING_TO_DISPLAY = (
    (1.8986148993059053, -0.7921761834040437, -0.10643871590186235),
    (-0.16894878647615938, 1.4889757541181161, -0.3200269676419564),
    (-0.12153916060431863, -0.31567585305224316, 1.4372150136565618),
)

# DWG RGB <-> XYZ matrices, computed from the published DWG primaries and D65.
# Row-vector values are transposed when registered as torch matrices.
M_DWG_TO_XYZ = (
    (0.700622320335921, 0.148774815012217, 0.101058729111070),
    (0.274118483092156, 0.873631895166417, -0.147750378258063),
    (-0.098962912575016, -0.137895325075852, 1.325915987837112),
)
M_XYZ_TO_DWG = (
    (1.516672040260000, -0.281478047499000, -0.146963628770000),
    (-0.464917101908000, 1.251423775570000, 0.174884608563000),
    (0.064849047069000, 0.109139342794000, 0.761414616139000),
)

# Bradford chromatic adaptation.
M_BRADFORD = (
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
)
M_BRADFORD_INV = (
    (0.986992905466712, -0.147054256420990, 0.159962651663731),
    (0.432305269723394, 0.518360271536777, 0.049291228212856),
    (-0.008528664575177, 0.040042821654085, 0.968486695787550),
)

# Rec.709 luminance coefficients. Useful as an approximately perceptual luma
# axis for saturation in a log-encoded working image.
REC709_LUMA = (0.2126, 0.7152, 0.0722)


# ============================================================================
# Shape and matrix helpers
# ============================================================================

def _to_nchw(x: Tensor) -> Tuple[Tensor, str]:
    if x.ndim == 3:
        if x.shape[0] == 3:
            return x.unsqueeze(0), "chw"
        if x.shape[-1] == 3:
            return x.permute(2, 0, 1).unsqueeze(0), "hwc"
    elif x.ndim == 4:
        if x.shape[1] == 3:
            return x, "nchw"
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2), "nhwc"
    raise ValueError(
        f"Expected (3,H,W), (B,3,H,W), (H,W,3), or (B,H,W,3); got {tuple(x.shape)}"
    )


def _from_nchw(x: Tensor, layout: str) -> Tensor:
    if layout == "chw":
        return x.squeeze(0)
    if layout == "hwc":
        return x.squeeze(0).permute(1, 2, 0)
    if layout == "nchw":
        return x
    if layout == "nhwc":
        return x.permute(0, 2, 3, 1)
    raise RuntimeError(f"Unknown layout: {layout}")


def _matrix(
    values: Sequence[Sequence[float]], ref: Tensor
) -> Tensor:
    return torch.as_tensor(values, dtype=ref.dtype, device=ref.device)


def apply_matrix(x: Tensor, matrix: Tensor) -> Tensor:
    """Apply a shared ``[3,3]`` or per-image ``[B,3,3]`` RGB matrix."""
    y, layout = _to_nchw(x)
    matrix = torch.as_tensor(matrix, dtype=y.dtype, device=y.device)
    if matrix.ndim == 2:
        out = torch.einsum("ij,bjhw->bihw", matrix, y)
    elif matrix.ndim == 3 and matrix.shape[0] in (1, y.shape[0]):
        out = torch.einsum("bij,bjhw->bihw", matrix.expand(y.shape[0], -1, -1), y)
    else:
        raise ValueError("matrix must have shape (3,3), (1,3,3), or (B,3,3)")
    return _from_nchw(out, layout)


def _batch_scalar(v: Union[float, Tensor], ref_nchw: Tensor) -> Tensor:
    """
    Convert scalar or shape-(B,) parameter to shape (B,1,1,1).
    """
    t = torch.as_tensor(v, dtype=ref_nchw.dtype, device=ref_nchw.device)
    if t.ndim == 0:
        return t.reshape(1, 1, 1, 1)
    if t.ndim == 1 and t.shape[0] in (1, ref_nchw.shape[0]):
        return t.reshape(-1, 1, 1, 1)
    raise ValueError(f"Expected scalar or (B,), got {tuple(t.shape)}")




# ============================================================================
# Camera and working-space transforms
# ============================================================================

def flog2_to_scene_linear(x: Tensor) -> Tensor:
    """
    F-Log2 normalized code value -> scene-linear F-Gamut RGB.

    torch.where evaluates both branches. Clamping the logarithmic branch's
    power exponent is unnecessary for normal camera values, but limiting it
    prevents accidental overflow during unconstrained optimization.
    """
    v = x
    linear_branch = (v - _FLOG2_F) / _FLOG2_E
    exponent = torch.clamp((v - _FLOG2_D) / _FLOG2_C, -32.0, 32.0)
    log_branch = torch.pow(v.new_tensor(10.0), exponent) / _FLOG2_A - _FLOG2_B / _FLOG2_A
    return torch.where(v < _FLOG2_CUT2, linear_branch, log_branch)


def di_oetf(x: Tensor) -> Tensor:
    """Scene-linear DWG RGB -> DaVinci Intermediate."""
    # DI is conventionally defined on nonnegative light.
    z = torch.clamp_min(x, 0.0)
    linear_branch = z * _DI_M
    log_branch = _DI_C * (torch.log2(z + _DI_A) + _DI_B)
    return torch.where(z <= _DI_LIN_CUT, linear_branch, log_branch)


def di_inverse(x: Tensor) -> Tensor:
    """DaVinci Intermediate -> scene-linear DWG RGB."""
    linear_branch = x / _DI_M
    exponent = torch.clamp(x / _DI_C - _DI_B, -32.0, 32.0)
    log_branch = torch.pow(x.new_tensor(2.0), exponent) - _DI_A
    return torch.where(x <= _DI_LOG_CUT, linear_branch, log_branch)


def scene_linear_f_gamut_to_dwg(x: Tensor) -> Tensor:
    return apply_matrix(x, _matrix(M_SRC_TO_WORKING, x))


def flog2_to_dwg_linear(x: Tensor) -> Tensor:
    return scene_linear_f_gamut_to_dwg(flog2_to_scene_linear(x))


def flog2_to_working(x: Tensor) -> Tensor:
    return di_oetf(flog2_to_dwg_linear(x))


def working_to_rec709_display(
    x: Tensor,
    gamma: float = DISPLAY_GAMMA,
    clamp: bool = False,
) -> Tensor:
    lin_709 = working_to_rec709_linear(x)
    positive = lin_709 > 0.0
    safe = lin_709.clamp_min(torch.finfo(lin_709.dtype).tiny)
    display = safe.pow(1.0 / gamma) * positive.to(lin_709.dtype)
    return display.clamp(0.0, 1.0) if clamp else display


def working_to_rec709_linear(x: Tensor) -> Tensor:
    """DaVinci Intermediate / DWG -> unclipped scene-linear Rec.709.

    Unlike the display helper this deliberately preserves negative and
    greater-than-one values.  It is the correct model-output domain for a loss:
    no invalid colour can disappear behind a display clamp.
    """
    return apply_matrix(di_inverse(x), _matrix(M_WORKING_TO_DISPLAY, x))


def _signed_cbrt(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.abs(x).clamp_min(1e-12).pow(1.0 / 3.0)


def linear_rec709_to_oklab(x: Tensor) -> Tensor:
    """Signed linear Rec.709 RGB -> OKLab, retaining out-of-gamut values."""
    rgb, layout = _to_nchw(x)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    l = _signed_cbrt(0.4122214708*r + 0.5363325363*g + 0.0514459929*b)
    m = _signed_cbrt(0.2119034982*r + 0.6806995451*g + 0.1073969566*b)
    s = _signed_cbrt(0.0883024619*r + 0.2817188376*g + 0.6299787005*b)
    lab = torch.stack((
        0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
        1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
        0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
    ), dim=1)
    return _from_nchw(lab, layout)


def oklab_to_linear_rec709(x: Tensor) -> Tensor:
    """OKLab -> signed linear Rec.709 RGB."""
    lab, layout = _to_nchw(x)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    l = (L + 0.3963377774*a + 0.2158037573*b).pow(3)
    m = (L - 0.1055613458*a - 0.0638541728*b).pow(3)
    s = (L - 0.0894841775*a - 1.2914855480*b).pow(3)
    rgb = torch.stack((
        4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
       -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
       -0.0041960863*l - 0.7034186147*m + 1.7076147010*s,
    ), dim=1)
    return _from_nchw(rgb, layout)


# ============================================================================
# Primary grading operators
# ============================================================================

def exposure_linear(x: Tensor, stops: Union[float, Tensor]) -> Tensor:
    """
    Linear exposure:
        y = 2^stops x.
    """
    y, layout = _to_nchw(x)
    s = _batch_scalar(stops, y)
    out = y * torch.exp2(s)
    return _from_nchw(out, layout)




def _xy_from_cct_approx(cct: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Differentiable approximation to the Planckian locus for 1667 K--25000 K.

    Formula: Hernández-Andrés / standard polynomial approximation.
    The piecewise boundaries are nondifferentiable but harmless in practice.
    """
    t = cct.clamp(1667.0, 25000.0)
    t2, t3 = t * t, t * t * t

    x_lo = -0.2661239e9 / t3 - 0.2343580e6 / t2 + 0.8776956e3 / t + 0.179910
    x_hi = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    x = torch.where(t <= 4000.0, x_lo, x_hi)

    y_1 = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    y_2 = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    y_3 = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    y = torch.where(t <= 2222.0, y_1, torch.where(t <= 4000.0, y_2, y_3))
    return x, y


def _xy_to_xyz_white(x: Tensor, y: Tensor) -> Tensor:
    yy = y.clamp_min(torch.finfo(y.dtype).eps)
    X = x / yy
    Y = torch.ones_like(X)
    Z = (1.0 - x - y) / yy
    return torch.stack((X, Y, Z), dim=-1)


def white_balance_linear(
    x: Tensor,
    temperature: Union[float, Tensor] = 0.0,
    tint: Union[float, Tensor] = 0.0,
    *,
    base_cct: float = 6504.0,
    mired_scale: float = 50.0,
    tint_scale: float = 0.05,
) -> Tensor:
    """
    Chromatic adaptation in scene-linear DWG.

    Parameters
    ----------
    temperature:
        Dimensionless UI-like value. Positive values warm the image.
        It shifts reciprocal temperature by `mired_scale * temperature`.

        target_mired = 1e6/base_cct + mired_scale*temperature
        target_cct   = 1e6/target_mired

    tint:
        Dimensionless green-magenta displacement. Positive values add magenta.
        It displaces the target white's y chromaticity by
        `-tint_scale * tint`.

    This is an explicit Bradford adaptation from the adjusted illuminant back
    to D65. Thus a warmer assumed illuminant produces compensating warm RGB
    gains, matching the usual grading-control direction.
    """
    image, layout = _to_nchw(x)
    B = image.shape[0]
    temp = torch.as_tensor(temperature, dtype=image.dtype, device=image.device)
    tint_t = torch.as_tensor(tint, dtype=image.dtype, device=image.device)
    if temp.ndim == 0:
        temp = temp.expand(B)
    if tint_t.ndim == 0:
        tint_t = tint_t.expand(B)
    if temp.shape != (B,) or tint_t.shape != (B,):
        raise ValueError("temperature and tint must be scalars or shape (B,)")

    base_mired = 1.0e6 / base_cct
    target_mired = (base_mired + mired_scale * temp).clamp(40.0, 600.0)
    target_cct = 1.0e6 / target_mired

    x_t, y_t = _xy_from_cct_approx(target_cct)
    y_t = (y_t - tint_scale * tint_t).clamp(0.05, 0.90)
    source_white = _xy_to_xyz_white(x_t, y_t)

    # Use the same locus approximation for the reference white. This makes
    # temperature=0 and tint=0 exactly neutral despite approximation error
    # relative to the tabulated D65 chromaticity.
    base_t = image.new_full((B,), base_cct)
    base_x, base_y = _xy_from_cct_approx(base_t)
    target_white = _xy_to_xyz_white(base_x, base_y)

    bradford = _matrix(M_BRADFORD, image)
    bradford_inv = _matrix(M_BRADFORD_INV, image)

    src_lms = torch.einsum("ij,bj->bi", bradford, source_white)
    dst_lms = torch.einsum("ij,bj->bi", bradford, target_white)
    ratio = dst_lms / src_lms.clamp_min(torch.finfo(image.dtype).eps)

    # A_b = B^{-1} diag(ratio_b) B.
    adaptation = torch.einsum(
        "ij,bj,jk->bik", bradford_inv, ratio, bradford
    )

    dwg_to_xyz = _matrix(M_DWG_TO_XYZ, image)
    # Inverting the registered forward matrix avoids neutral drift caused by
    # rounded independently tabulated inverse coefficients.
    xyz_to_dwg = torch.linalg.inv(dwg_to_xyz)
    rgb_matrix = torch.einsum(
        "ij,bjk,kl->bil", xyz_to_dwg, adaptation, dwg_to_xyz
    )
    out = torch.einsum("bij,bjhw->bihw", rgb_matrix, image)
    return _from_nchw(out, layout)




def monotone_curve_knots(
    raw_steps: Tensor,
    *,
    start: float = 0.0,
    end: float = 1.0,
    min_step: float = 1e-4,
) -> Tensor:
    """
    Convert unconstrained parameters to strictly increasing curve ordinates.

    For raw_steps of shape (..., K-1), returns shape (..., K):
        d_i = softplus(raw_i) + min_step
        y_0 = start
        y_j = start + (end-start) sum_{i<j} d_i / sum_i d_i
    """
    steps = F.softplus(raw_steps) + min_step
    cumulative = torch.cumsum(steps, dim=-1)
    normalized = cumulative / cumulative[..., -1:].clamp_min(min_step)
    first = torch.zeros_like(normalized[..., :1])
    unit = torch.cat((first, normalized), dim=-1)
    return start + (end - start) * unit


def _apply_scalar_monotone_curve(values: Tensor, y_knots: Tensor) -> Tensor:
    """Apply one shared monotone curve to ``values`` shaped ``[B,H,W]``."""
    B = values.shape[0]
    yk = torch.as_tensor(y_knots, dtype=values.dtype, device=values.device)
    if yk.ndim == 1:
        yk = yk.unsqueeze(0).expand(B, -1)
    if yk.ndim != 2 or yk.shape[0] not in (1, B):
        raise ValueError("shared curve must have shape (K), (1,K), or (B,K)")
    yk = yk.expand(B, -1)
    K = yk.shape[-1]
    xk = torch.linspace(0.0, 1.0, K, dtype=values.dtype, device=values.device)
    flat = values.reshape(B, -1)
    idx = (torch.searchsorted(xk, flat.contiguous(), right=True) - 1).clamp(0, K - 2)
    lo, hi = torch.gather(yk, 1, idx), torch.gather(yk, 1, idx + 1)
    t = (flat - xk[idx]) / (xk[idx + 1] - xk[idx])
    return (lo + t * (hi - lo)).reshape_as(values)


def perceptual_grade_linear_dwg(
    x: Tensor,
    curve_y_knots: Optional[Tensor] = None,
    saturation_amount: Union[float, Tensor] = 1.0,
    hue_radians: Union[float, Tensor] = 0.0,
) -> Tensor:
    """Grade tone and chroma in OKLab, returning scene-linear DWG.

    Tone is a single monotone L curve; chroma is scaled and rotated without
    independent RGB curves.  Consequently a neutral pixel stays neutral unless
    the explicit hue/chroma controls request otherwise.
    """
    dwg, layout = _to_nchw(x)
    to_709 = _matrix(M_WORKING_TO_DISPLAY, dwg)
    to_dwg = torch.linalg.inv(to_709)
    lab, _ = _to_nchw(linear_rec709_to_oklab(apply_matrix(dwg, to_709)))
    if curve_y_knots is not None:
        if curve_y_knots.ndim == 3:
            curve_y_knots = curve_y_knots[:, 0]
        lab = lab.clone()
        lab[:, 0] = _apply_scalar_monotone_curve(lab[:, 0], curve_y_knots)
    sat = _batch_scalar(saturation_amount, lab)
    hue = _batch_scalar(hue_radians, lab)
    a, b = lab[:, 1:2], lab[:, 2:3]
    cos_h, sin_h = torch.cos(hue), torch.sin(hue)
    lab = torch.cat((lab[:, :1], sat * (cos_h*a - sin_h*b),
                     sat * (sin_h*a + cos_h*b)), dim=1)
    out = apply_matrix(oklab_to_linear_rec709(lab), to_dwg)
    out, _ = _to_nchw(out)
    return _from_nchw(out, layout)




# ============================================================================
# 3D LUT
# ============================================================================

def apply_lut(
    x: Tensor,
    lut: Tensor,
    *,
    clamp_input: bool = True,
) -> Tensor:
    """Apply a trilinear 3D RGB lookup table to an image.

    ``lut`` may have shape ``(S,S,S,3)`` (the common .cube-style layout),
    ``(3,S,S,S)``, or either layout with a leading batch dimension. Its axes
    are ordered red, green, blue and its values are expected to be in the same
    code-value domain as ``x``. LUT values remain differentiable, so a LUT can
    be optimized with the rest of a model.

    Inputs are clamped to [0, 1] by default because a finite LUT has no defined
    extrapolation outside that range.  Set ``clamp_input=False`` to reject
    neither values nor gradients; ``grid_sample`` will then use zero padding
    for out-of-range coordinates.
    """
    image, layout = _to_nchw(x)
    table = torch.as_tensor(lut, dtype=image.dtype, device=image.device)

    if table.ndim == 4 and table.shape[-1] == 3:
        table = table.permute(3, 2, 1, 0).unsqueeze(0)
    elif table.ndim == 4 and table.shape[0] == 3:
        table = table.permute(0, 3, 2, 1).unsqueeze(0)
    elif table.ndim == 5 and table.shape[-1] == 3:
        table = table.permute(0, 4, 3, 2, 1)
    elif table.ndim == 5 and table.shape[1] == 3:
        pass
    else:
        raise ValueError("lut must have shape (S,S,S,3), (3,S,S,S), or batched equivalent")

    if table.shape[2] != table.shape[3] or table.shape[3] != table.shape[4]:
        raise ValueError("lut must be cubic: (S,S,S,3) or (3,S,S,S)")
    if table.shape[2] < 2:
        raise ValueError("lut size S must be at least 2")

    values = image.clamp(0.0, 1.0) if clamp_input else image
    # grid_sample's coordinates are x=width, y=height, z=depth.  The LUT has
    # been arranged as (blue, green, red), hence this RGB coordinate order.
    grid = values.permute(0, 2, 3, 1).unsqueeze(1).mul(2.0).sub(1.0)
    if table.shape[0] not in (1, image.shape[0]):
        raise ValueError("Batched LUT must have batch size 1 or match image batch size")
    sampled = F.grid_sample(
        table.expand(image.shape[0], -1, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return _from_nchw(sampled.squeeze(2), layout)


# ============================================================================
