"""Tests for qdrift.kernels and qdrift.core: candidate kernels compared
against the exact oracle, and the fixture/probe-runner plumbing.
"""
from __future__ import annotations

from qdrift.core import run_cross_scale_add_check, run_round_boundary_check
from qdrift.fixtures import (
    CROSS_SCALE_ADD_FIXTURES,
    ROUND_BOUNDARY_FIXTURES,
    list_cross_scale_fixtures,
    list_round_boundary_fixtures,
)
from qdrift.kernels import (
    float_dequantize,
    float_quantize,
    float_requantize_add,
    int32_accumulator_requantize_add,
)
from qdrift.reference import QParams, exact_quantize


class TestFloatQuantizeAgreesWithOracleOnNonTieValues:
    def test_half_even_kernel_matches_exact_oracle_on_non_boundary_values(self):
        qp = QParams(scale=0.037, zero_point=-3, qmin=-128, qmax=127)
        for v in [0.0, 1.0, -1.0, 3.14159, -2.71828, 100.5, -50.25]:
            assert float_quantize(v, qp) == exact_quantize(v, qp)


class TestRoundBoundaryCheck:
    def test_half_even_round_mode_is_exact_on_every_fixture(self):
        # The whole point of the fixture set is to probe exact-tie
        # boundaries; a correctly-implemented half-to-even kernel must
        # show zero mismatches on all of them.
        for fixture in ROUND_BOUNDARY_FIXTURES:
            result = run_round_boundary_check(fixture.name, round_mode="half_even")
            assert result.verdict == "exact", (fixture.name, result.mismatches)

    def test_half_down_bug_mode_is_caught_as_drift(self):
        # The deliberately-buggy round mode must be flagged as drift on at
        # least one fixture (proving the checker actually discriminates
        # correct from incorrect behavior, not just always passing).
        any_drift = False
        for name in list_round_boundary_fixtures():
            result = run_round_boundary_check(name, round_mode="half_down_bug")
            if result.verdict == "drift_detected":
                any_drift = True
        assert any_drift

    def test_result_reports_every_mismatch_not_a_summary_count_only(self):
        result = run_round_boundary_check(
            "onnx_qlinear_half_ties_fp16_input", round_mode="half_down_bug"
        )
        assert len(result.mismatches) >= 1
        for m in result.mismatches:
            assert m.exact_code != m.candidate_code

    def test_unknown_fixture_raises(self):
        import pytest
        with pytest.raises(KeyError):
            run_round_boundary_check("does_not_exist")

    def test_unknown_round_mode_raises(self):
        import pytest
        with pytest.raises(ValueError):
            run_round_boundary_check("onnx_qlinear_half_ties_fp16_input", round_mode="bogus")


class TestCrossScaleAddCheck:
    def test_float64_kernel_is_exact_on_all_fixtures(self):
        for fixture in CROSS_SCALE_ADD_FIXTURES:
            result = run_cross_scale_add_check(fixture.name, kernel_name="float64")
            assert result.verdict == "exact", (fixture.name, result.mismatches)

    def test_int32_fixedpoint_kernel_result_is_deterministic(self):
        # Same inputs -> same result across repeated runs (no hidden
        # nondeterminism from e.g. dict ordering or float summation order).
        r1 = run_cross_scale_add_check("near_identical_scales_sanity", kernel_name="int32_fixedpoint")
        r2 = run_cross_scale_add_check("near_identical_scales_sanity", kernel_name="int32_fixedpoint")
        assert r1.to_dict() == r2.to_dict()

    def test_int32_fixedpoint_divergence_on_sanity_fixture_is_a_real_finding(self):
        # Documented, reproducible finding: the int32 fixed-point kernel
        # diverges from the exact oracle on qa=50, qb=-50 in the
        # near-identical-scales fixture (true sum sits at a rounding
        # boundary). This test pins that specific behavior so a future
        # change to the fixed-point kernel that silently "fixes" or
        # changes this must be a deliberate, reviewed change.
        result = run_cross_scale_add_check("near_identical_scales_sanity", kernel_name="int32_fixedpoint")
        mismatched_pairs = {(m.qa, m.qb) for m in result.mismatches}
        assert (50, -50) in mismatched_pairs

    def test_unknown_fixture_raises(self):
        import pytest
        with pytest.raises(KeyError):
            run_cross_scale_add_check("does_not_exist")

    def test_unknown_kernel_raises(self):
        import pytest
        with pytest.raises(ValueError):
            run_cross_scale_add_check("arm_residual_add_disparate_scales", kernel_name="bogus")


class TestFixtureListing:
    def test_lists_are_nonempty_and_match_actual_fixtures(self):
        assert set(list_round_boundary_fixtures()) == {f.name for f in ROUND_BOUNDARY_FIXTURES}
        assert set(list_cross_scale_fixtures()) == {f.name for f in CROSS_SCALE_ADD_FIXTURES}
