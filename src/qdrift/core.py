"""Probe runner: compares candidate (float/int32-fixed-point) kernels
against the exact Fraction oracle for each adversarial fixture, and
reports every mismatch -- never averages them away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from . import fixtures as fx
from . import kernels
from .reference import exact_quantize, exact_requantize_add


@dataclass
class RoundMismatch:
    probe_value: float
    exact_code: int
    candidate_code: int


@dataclass
class RoundBoundaryResult:
    fixture_name: str
    round_mode: str
    description: str
    n_probes: int
    mismatches: List[RoundMismatch] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "drift_detected" if self.mismatches else "exact"

    def to_dict(self) -> dict:
        return {
            "fixture": self.fixture_name,
            "round_mode": self.round_mode,
            "n_probes": self.n_probes,
            "n_mismatches": len(self.mismatches),
            "verdict": self.verdict,
            "mismatches": [
                {"probe_value": m.probe_value, "exact_code": m.exact_code, "candidate_code": m.candidate_code}
                for m in self.mismatches
            ],
        }


def run_round_boundary_check(fixture_name: str, round_mode: str = "half_even") -> RoundBoundaryResult:
    fixture = fx.get_round_boundary_fixture(fixture_name)
    if round_mode not in kernels.ROUND_MODES:
        raise ValueError(f"unknown round_mode {round_mode!r}; choose from {list(kernels.ROUND_MODES)}")
    round_fn = kernels.ROUND_MODES[round_mode]

    result = RoundBoundaryResult(
        fixture_name=fixture.name,
        round_mode=round_mode,
        description=fixture.description,
        n_probes=len(fixture.probe_values),
    )
    for v in fixture.probe_values:
        exact_code = exact_quantize(v, fixture.qp)
        candidate_code = kernels.float_quantize(v, fixture.qp, round_fn=round_fn)
        if exact_code != candidate_code:
            result.mismatches.append(RoundMismatch(v, exact_code, candidate_code))
    return result


@dataclass
class AddMismatch:
    qa: int
    qb: int
    exact_code: int
    candidate_code: int
    exact_real_sum: float
    candidate_real_estimate: float


@dataclass
class CrossScaleAddResult:
    fixture_name: str
    kernel_name: str
    description: str
    n_probes: int
    mismatches: List[AddMismatch] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "drift_detected" if self.mismatches else "exact"

    def to_dict(self) -> dict:
        return {
            "fixture": self.fixture_name,
            "kernel": self.kernel_name,
            "n_probes": self.n_probes,
            "n_mismatches": len(self.mismatches),
            "verdict": self.verdict,
            "mismatches": [
                {
                    "qa": m.qa, "qb": m.qb,
                    "exact_code": m.exact_code, "candidate_code": m.candidate_code,
                    "exact_real_sum": m.exact_real_sum,
                    "candidate_real_estimate": m.candidate_real_estimate,
                }
                for m in self.mismatches
            ],
        }


KERNEL_FNS = {
    "float64": lambda qa, qpa, qb, qpb, qpo: kernels.float_requantize_add(qa, qpa, qb, qpb, qpo),
    "int32_fixedpoint": kernels.int32_accumulator_requantize_add,
}


def run_cross_scale_add_check(fixture_name: str, kernel_name: str = "float64") -> CrossScaleAddResult:
    fixture = fx.get_cross_scale_fixture(fixture_name)
    if kernel_name not in KERNEL_FNS:
        raise ValueError(f"unknown kernel {kernel_name!r}; choose from {list(KERNEL_FNS)}")
    kernel_fn = KERNEL_FNS[kernel_name]

    result = CrossScaleAddResult(
        fixture_name=fixture.name,
        kernel_name=kernel_name,
        description=fixture.description,
        n_probes=len(fixture.pairs),
    )
    for qa, qb in fixture.pairs:
        exact_code = exact_requantize_add(qa, fixture.qp_a, qb, fixture.qp_b, fixture.qp_out)
        candidate_code = kernel_fn(qa, fixture.qp_a, qb, fixture.qp_b, fixture.qp_out)
        if exact_code != candidate_code:
            from .reference import exact_dequantize
            exact_sum = float(exact_dequantize(qa, fixture.qp_a) + exact_dequantize(qb, fixture.qp_b))
            result.mismatches.append(
                AddMismatch(qa, qb, exact_code, candidate_code, exact_sum, exact_sum)
            )
    return result
