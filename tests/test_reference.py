"""Correctness tests for qdrift.reference: the exact-oracle arithmetic.

Every claim is checked against an independent method:
- round_half_even is checked against Python's own round() (which is
  documented to use round-half-to-even for floats) on plain floats, AND
  against hand-picked exact ties where the answer is unambiguous.
- exact_quantize/dequantize round-trip and clamp correctly.
- exact_requantize_add is checked against a from-scratch decimal.Decimal
  computation (a second, independent high-precision arithmetic library),
  not just re-deriving the same Fraction code path.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from qdrift.reference import (
    QParams,
    exact_dequantize,
    exact_quantize,
    exact_requantize_add,
    round_half_down,
    round_half_even,
)


class TestRoundHalfEven:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (Fraction(1, 2), 0),   # 0.5 -> 0 (even)
            (Fraction(3, 2), 2),   # 1.5 -> 2 (even)
            (Fraction(5, 2), 2),   # 2.5 -> 2 (even)
            (Fraction(7, 2), 4),   # 3.5 -> 4 (even)
            (Fraction(-1, 2), 0),  # -0.5 -> 0 (even)
            (Fraction(-3, 2), -2), # -1.5 -> -2 (even)
        ],
    )
    def test_exact_ties_round_to_even(self, value, expected):
        assert round_half_even(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (Fraction(11, 10), 1),   # 1.1 -> 1
            (Fraction(19, 10), 2),   # 1.9 -> 2
            (Fraction(-11, 10), -1), # -1.1 -> -1
            (Fraction(-19, 10), -2), # -1.9 -> -2
        ],
    )
    def test_non_tie_values_round_normally(self, value, expected):
        assert round_half_even(value) == expected

    def test_matches_python_round_for_many_non_adversarial_floats(self):
        # Python's round() uses round-half-to-even for floats (documented
        # behavior); cross-check our exact implementation against it on a
        # sweep of ordinary (non-exact-tie) values as an independent check.
        import random
        rng = random.Random(0)
        for _ in range(500):
            v = rng.uniform(-1000, 1000)
            frac = Fraction(v)
            # Skip the vanishingly-rare case where the float's exact
            # Fraction representation happens to be an exact .5 tie --
            # those are covered by the dedicated tie tests above with
            # unambiguous rational inputs.
            if (frac * 2).denominator == 1:
                continue
            assert round_half_even(frac) == round(v)


class TestRoundHalfDownBug:
    def test_ties_round_down_not_to_even(self):
        # This reproduces the buggy behavior reported in
        # microsoft/onnxruntime#18576: exact ties go to the lower integer
        # regardless of even/odd.
        assert round_half_down(Fraction(1, 2)) == 0
        assert round_half_down(Fraction(3, 2)) == 1  # NOT 2 (differs from half-even)
        assert round_half_down(Fraction(5, 2)) == 2
        assert round_half_down(Fraction(-1, 2)) == -1  # differs from half-even's 0

    def test_differs_from_half_even_on_odd_ties(self):
        # The whole point of the bug: half_even and half_down_bug disagree
        # exactly on ties where the lower integer is odd.
        assert round_half_even(Fraction(3, 2)) != round_half_down(Fraction(3, 2))
        assert round_half_even(Fraction(-1, 2)) != round_half_down(Fraction(-1, 2))

    def test_agrees_with_half_even_on_even_ties(self):
        assert round_half_even(Fraction(1, 2)) == round_half_down(Fraction(1, 2))
        assert round_half_even(Fraction(5, 2)) == round_half_down(Fraction(5, 2))

    def test_non_tie_values_round_normally_not_just_ties(self):
        # Every existing round_half_down test uses an exact .5 tie, so the
        # function's non-tie branch (remainder > half -> round up) was
        # never exercised: a bug that broke ordinary (non-tie) rounding
        # here would have gone undetected. round_half_down must agree
        # with round_half_even away from ties (they only differ AT ties).
        for value, expected in [
            (Fraction(11, 10), 1),    # 1.1 -> 1 (below tie)
            (Fraction(19, 10), 2),    # 1.9 -> 2 (above tie)
            (Fraction(-11, 10), -1),  # -1.1 -> -1
            (Fraction(-19, 10), -2),  # -1.9 -> -2
        ]:
            assert round_half_down(value) == expected
            assert round_half_down(value) == round_half_even(value)


class TestQParams:
    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError):
            QParams(scale=0.0, zero_point=0)
        with pytest.raises(ValueError):
            QParams(scale=-1.0, zero_point=0)

    def test_rejects_zero_point_outside_range(self):
        with pytest.raises(ValueError):
            QParams(scale=1.0, zero_point=200, qmin=-128, qmax=127)

    def test_rejects_inverted_qmin_qmax(self):
        with pytest.raises(ValueError):
            QParams(scale=1.0, zero_point=0, qmin=100, qmax=-100)


class TestExactQuantizeDequantize:
    def test_round_trip_is_exact_for_exact_multiples(self):
        qp = QParams(scale=0.5, zero_point=0)
        for code in range(-10, 11):
            real = exact_dequantize(code, qp)
            assert exact_quantize(real, qp) == code

    def test_clamps_to_qmax(self):
        qp = QParams(scale=1.0, zero_point=0, qmin=-128, qmax=127)
        assert exact_quantize(1e9, qp) == 127

    def test_clamps_to_qmin(self):
        qp = QParams(scale=1.0, zero_point=0, qmin=-128, qmax=127)
        assert exact_quantize(-1e9, qp) == -128

    def test_zero_point_offset_applied(self):
        qp = QParams(scale=1.0, zero_point=50, qmin=-128, qmax=127)
        assert exact_quantize(0.0, qp) == 50

    def test_matches_independent_decimal_computation(self):
        # Cross-check exact_dequantize against decimal.Decimal (a second,
        # independent high-precision library) rather than re-deriving the
        # same Fraction path.
        qp = QParams(scale=0.037, zero_point=-8, qmin=-128, qmax=127)
        for code in (-100, -1, 0, 1, 42, 100):
            got = exact_dequantize(code, qp)
            expected = (Decimal(code) - Decimal(qp.zero_point)) * Decimal(str(qp.scale))
            assert abs(Decimal(got.numerator) / Decimal(got.denominator) - expected) < Decimal("1e-15")

    def test_exact_quantize_accepts_plain_int_input(self):
        # exact_quantize's internal _to_fraction() has a dedicated
        # isinstance(x, int) branch distinct from its Fraction and float
        # branches; every other test in this suite passes a float or an
        # already-Fraction value, so the plain-int input path (a legitimate
        # public-API input per the Number type alias) was never exercised.
        qp = QParams(scale=1.0, zero_point=0, qmin=-128, qmax=127)
        assert exact_quantize(5, qp) == 5
        assert exact_quantize(-5, qp) == -5


class TestExactRequantizeAdd:
    def test_matches_independent_decimal_computation(self):
        qp_a = QParams(scale=0.031, zero_point=-5, qmin=-128, qmax=127)
        qp_b = QParams(scale=0.0074, zero_point=12, qmin=-128, qmax=127)
        qp_out = QParams(scale=0.045, zero_point=0, qmin=-128, qmax=127)

        for qa, qb in [(100, 100), (-128, 127), (0, 0), (50, -50)]:
            got = exact_requantize_add(qa, qp_a, qb, qp_b, qp_out)

            # Independent Decimal-based recomputation of the same math.
            real_a = (Decimal(qa) - Decimal(qp_a.zero_point)) * Decimal(str(qp_a.scale))
            real_b = (Decimal(qb) - Decimal(qp_b.zero_point)) * Decimal(str(qp_b.scale))
            real_sum = real_a + real_b
            scaled = real_sum / Decimal(str(qp_out.scale))
            # Decimal round-half-even via quantize
            from decimal import ROUND_HALF_EVEN
            rounded = int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
            expected = max(qp_out.qmin, min(qp_out.qmax, rounded + qp_out.zero_point))
            assert got == expected, (qa, qb, got, expected)

    def test_output_is_always_within_range(self):
        qp_a = QParams(scale=1.0, zero_point=0, qmin=-128, qmax=127)
        qp_b = QParams(scale=1.0, zero_point=0, qmin=-128, qmax=127)
        qp_out = QParams(scale=1.0, zero_point=0, qmin=-128, qmax=127)
        result = exact_requantize_add(127, qp_a, 127, qp_b, qp_out)
        assert qp_out.qmin <= result <= qp_out.qmax
