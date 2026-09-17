"""Adversarial fixtures for qdrift: quantization parameter sets and probe
values chosen to exercise the specific reported failure patterns, not
generic random inputs.

Each fixture documents which real-world report it reconstructs the shape
of (never claiming to be the exact reported model/data, since none of the
original bug reports published a minimal byte-for-byte repro dataset).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .reference import QParams


@dataclass(frozen=True)
class RoundBoundaryFixture:
    name: str
    description: str
    qp: QParams
    probe_values: List[float]


@dataclass(frozen=True)
class CrossScaleAddFixture:
    name: str
    description: str
    qp_a: QParams
    qp_b: QParams
    qp_out: QParams
    pairs: List[tuple]  # (qa, qb) integer code pairs to probe


ROUND_BOUNDARY_FIXTURES: List[RoundBoundaryFixture] = [
    RoundBoundaryFixture(
        name="onnx_qlinear_half_ties_fp16_input",
        description=(
            "Reconstructs the shape of microsoft/onnxruntime#18576: a "
            "QuantizeLinear layer over fp16-precision inputs where "
            "value/scale lands exactly on a .5 boundary. The issue reports "
            "the runtime consistently rounds such ties DOWN instead of the "
            "ONNX-spec round-half-to-even."
        ),
        qp=QParams(scale=0.1, zero_point=0, qmin=-128, qmax=127),
        # scale=0.1 chosen so several small multiples land on exact .5
        # quotients when divided by scale (e.g. 0.05/0.1 = 0.5 exactly).
        probe_values=[0.05, 0.15, 0.25, 0.35, -0.05, -0.15, -0.25, 1.05, 2.55],
    ),
    RoundBoundaryFixture(
        name="symmetric_zero_point_ties",
        description=(
            "Ties at zero_point != 0 boundaries -- a distinct code path in "
            "many kernels (offset added before vs after rounding) from the "
            "zero_point=0 case above."
        ),
        qp=QParams(scale=0.25, zero_point=10, qmin=-128, qmax=127),
        probe_values=[0.125, 0.375, 0.625, -0.125, -0.375, 3.125],
    ),
]


CROSS_SCALE_ADD_FIXTURES: List[CrossScaleAddFixture] = [
    CrossScaleAddFixture(
        name="arm_residual_add_disparate_scales",
        description=(
            "Reconstructs the shape of openvinotoolkit/openvino#34673: an "
            "INT8 residual (element-wise Add) connection combining two "
            "tensors with different learned scales -- exactly the "
            "'Add_2' node the issue identifies as the point of catastrophic "
            "failure ('int32 accumulator overflow, incorrect broadcast of "
            "scale/zero-point vectors, or requantization/scale-alignment "
            "failure'). Scale ratio is deliberately not a clean power of "
            "two. Note: qdrift's own int32_fixedpoint reference kernel "
            "passes this fixture exactly (0 mismatches) -- it is NOT a "
            "byte-for-byte reproduction of the ARM Compute Library SVE2/Neon "
            "bug (that requires the actual ACL binary on M4 hardware, "
            "unavailable to this checker); this fixture demonstrates the "
            "*problem shape* (disparate-scale INT8 add) and gives a "
            "reusable oracle other real kernels can be checked against, "
            "not a confirmed repro of the specific ACL defect."
        ),
        qp_a=QParams(scale=0.031, zero_point=-5, qmin=-128, qmax=127),
        qp_b=QParams(scale=0.0074, zero_point=12, qmin=-128, qmax=127),
        qp_out=QParams(scale=0.045, zero_point=0, qmin=-128, qmax=127),
        pairs=[
            (100, 100), (127, 127), (-128, -128), (0, 0),
            (100, -128), (-128, 100), (50, -50), (127, -128),
            (10, 10), (-1, 1), (64, 63),
        ],
    ),
    CrossScaleAddFixture(
        name="near_identical_scales_sanity",
        description=(
            "A near-identical-scale fixture, included to test the (false) "
            "assumption that only widely disparate scales are at risk of "
            "fixed-point requantization error. Empirically, qdrift's own "
            "int32_fixedpoint reference kernel DOES diverge here (qa=50, "
            "qb=-50 rounds to -1 instead of the exact 0) even though the "
            "scale ratio is ~1.005 -- the true sum sits at -0.005, right at "
            "a rounding boundary that a 16-bit fixed-point rescale nudges "
            "across. This is a genuine measured property of that kernel, "
            "not a designed 'should always pass' control -- treat the "
            "verdict as data, not as confirmation of an assumption."
        ),
        qp_a=QParams(scale=0.02, zero_point=0, qmin=-128, qmax=127),
        qp_b=QParams(scale=0.0201, zero_point=0, qmin=-128, qmax=127),
        qp_out=QParams(scale=0.02, zero_point=0, qmin=-128, qmax=127),
        pairs=[(100, 100), (50, -50), (127, 127), (-128, -128), (10, 10)],
    ),
]


def list_round_boundary_fixtures() -> List[str]:
    return [f.name for f in ROUND_BOUNDARY_FIXTURES]


def list_cross_scale_fixtures() -> List[str]:
    return [f.name for f in CROSS_SCALE_ADD_FIXTURES]


def get_round_boundary_fixture(name: str) -> RoundBoundaryFixture:
    for f in ROUND_BOUNDARY_FIXTURES:
        if f.name == name:
            return f
    raise KeyError(f"unknown round-boundary fixture: {name}")


def get_cross_scale_fixture(name: str) -> CrossScaleAddFixture:
    for f in CROSS_SCALE_ADD_FIXTURES:
        if f.name == name:
            return f
    raise KeyError(f"unknown cross-scale fixture: {name}")
