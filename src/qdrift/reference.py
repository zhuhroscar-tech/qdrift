"""Exact reference arithmetic for affine (scale/zero-point) INT8 quantization.

Uses Python's ``fractions.Fraction`` so the "ground truth" itself never
carries float rounding error -- the oracle a real (float-based) kernel is
compared against must not share float's own approximation with the thing
being tested.

Affine quantization (the scheme used by ONNX QuantizeLinear/DequantizeLinear,
TensorFlow Lite, and ARM Compute Library's NEQuantizationLayer family):

    q = clamp(round(x / scale) + zero_point, qmin, qmax)
    x_approx = (q - zero_point) * scale

``round`` here is round-half-to-even (banker's rounding), which is the
documented ONNX spec behavior -- see microsoft/onnxruntime#18576, which
reports a QuantizeLinear kernel that instead rounds half-down, a real
observed divergence from spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union

Number = Union[int, float, Fraction]


def _to_fraction(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    # float -> exact Fraction of the float's own binary value (no decimal
    # re-rounding); this is what "the real number a float represents" means.
    return Fraction(x)


def round_half_even(x: Fraction) -> int:
    """Round a Fraction to the nearest int, ties to even -- exactly the
    IEEE-754 / ONNX QuantizeLinear default rounding rule, computed exactly
    (no intermediate float rounding) so this is a trustworthy oracle.
    """
    floor_val = x.numerator // x.denominator
    remainder = x - floor_val
    half = Fraction(1, 2)
    if remainder < half:
        return floor_val
    if remainder > half:
        return floor_val + 1
    # exact tie: round to even
    return floor_val if floor_val % 2 == 0 else floor_val + 1


def round_half_down(x: Fraction) -> int:
    """Round half toward negative infinity's neighbor-down variant: ties
    round to the lower integer. This is the buggy behavior reported in
    microsoft/onnxruntime#18576 ('consistently rounding to the lower
    integer instead of round-to-nearest-even'). Included so a kernel
    exhibiting exactly this bug is caught by ``check_round_mode``.
    """
    floor_val = x.numerator // x.denominator
    remainder = x - floor_val
    half = Fraction(1, 2)
    if remainder <= half:
        return floor_val
    return floor_val + 1


@dataclass(frozen=True)
class QParams:
    """Affine quantization parameters for one tensor."""

    scale: float
    zero_point: int
    qmin: int = -128
    qmax: int = 127

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        if self.qmin >= self.qmax:
            raise ValueError(f"qmin ({self.qmin}) must be < qmax ({self.qmax})")
        if not (self.qmin <= self.zero_point <= self.qmax):
            raise ValueError(
                f"zero_point {self.zero_point} outside [{self.qmin}, {self.qmax}]"
            )


def exact_quantize(x: Number, qp: QParams) -> int:
    """Quantize a real value to an integer code, exactly (Fraction math),
    using round-half-to-even and clamping -- the reference oracle."""
    scale = _to_fraction(qp.scale)
    real = _to_fraction(x)
    q = round_half_even(real / scale) + qp.zero_point
    return max(qp.qmin, min(qp.qmax, q))


def exact_dequantize(q: int, qp: QParams) -> Fraction:
    """Dequantize an integer code back to an exact real value (Fraction)."""
    return (Fraction(q) - qp.zero_point) * _to_fraction(qp.scale)


def exact_requantize_add(
    qa: int,
    qp_a: QParams,
    qb: int,
    qp_b: QParams,
    qp_out: QParams,
) -> int:
    """The mathematically correct cross-scale element-wise add:

        real_sum = dequantize(qa, qp_a) + dequantize(qb, qp_b)
        result   = quantize(real_sum, qp_out)

    This is the ground truth for "INT8 tensor A (its own scale/zero-point)
    plus INT8 tensor B (a *different* scale/zero-point) equals INT8 tensor
    C (a third scale/zero-point)" -- exactly the operation reported broken
    on Apple M4 in openvinotoolkit/openvino#34673 ('the ARM Compute Library
    SVE2/Neon kernel is failing the requantization/scale-alignment step
    during the addition of tensors with different scales').
    """
    real_a = exact_dequantize(qa, qp_a)
    real_b = exact_dequantize(qb, qp_b)
    real_sum = real_a + real_b
    return exact_quantize(real_sum, qp_out)
