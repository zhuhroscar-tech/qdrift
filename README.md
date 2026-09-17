[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# qdrift

Check affine INT8 quantization arithmetic (`QuantizeLinear`/`DequantizeLinear`-style
scale + zero-point math, as used by ONNX Runtime, TensorFlow Lite, and ARM
Compute Library) against an exact rational-number oracle. `qdrift` does not
run a real inference framework or hardware kernel — it reconstructs the
*shape* of two independently reported divergence classes and gives a
reusable exact reference other tools or hand-written kernels can be
checked against.

## What it checks

**Round-boundary behavior** (`check-round`): whether a candidate
quantization kernel rounds ties (values landing exactly on a `.5`
quotient after dividing by `scale`) to the ONNX-spec round-half-to-even,
or to some other rule. Reconstructs the shape of
[microsoft/onnxruntime#18576](https://github.com/microsoft/onnxruntime/issues/18576),
which reported a `QuantizeLinear` kernel consistently rounding ties down
instead of to-even.

**Cross-scale element-wise add** (`check-add`): whether adding two INT8
tensors that have *different* scale/zero-point pairs and requantizing to
a third scale produces the mathematically correct result. Reconstructs
the problem shape of
[openvinotoolkit/openvino#34673](https://github.com/openvinotoolkit/openvino/issues/34673),
an open, independently-reproduced report of INT8 residual-add corruption
on Apple M4 hardware.

**Important limitation, stated plainly:** `qdrift` does not have access
to ONNX Runtime's actual `QuantizeLinear` kernel or ARM Compute Library's
actual NEQuantizationLayer/Add kernels, and cannot run on M4-specific ACL
code paths. It ships two illustrative candidate implementations (a plain
float64 kernel and a 16-bit fixed-point accumulator kernel) built the way
real runtime kernels are commonly built, checked against the exact
oracle — a reusable correctness harness for *this class* of bug, not a
confirmed reproduction of either specific cited issue's root cause. If
you maintain a real quantization kernel, adapt `qdrift.reference` as the
oracle and swap in your own kernel call.

## Install

Requires Python 3.9+, no runtime dependencies (uses only `fractions` and
`decimal` from the standard library).

```bash
git clone https://github.com/zhuhroscar-tech/qdrift.git
cd qdrift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Release wheels are available from [Releases](https://github.com/zhuhroscar-tech/qdrift/releases);
check the accompanying `SHA256SUMS.txt` before installation.

## Quick start

```bash
qdrift list-fixtures
qdrift check-round                                   # default: half_even (spec-correct)
qdrift check-round --round-mode half_down_bug         # deliberately buggy mode, for comparison
qdrift check-add                                      # default: float64 kernel
qdrift check-add --kernel int32_fixedpoint            # fixed-point accumulator kernel
qdrift check-round --json
```

Exit code is `0` when every probed value matches the exact oracle, `1`
when any mismatch (drift) is found.

## How the oracle works

All "ground truth" math uses Python's `fractions.Fraction`, so the
reference itself never carries float rounding error. `round_half_even`
computes IEEE-754/ONNX-spec ties-to-even rounding exactly on a `Fraction`
— no intermediate float conversion. `exact_requantize_add` dequantizes
both inputs to exact rationals, adds them exactly, and requantizes once
— the unambiguous "correct answer" a real float or fixed-point kernel is
compared against.

## Design notes

- One shared status-glyph system, semantic-only ANSI color, and
  `NO_COLOR`/`--no-color` support, consistent with this project's other
  CLIs.
- Every finding reports the specific input, exact code, and candidate
  code — never a pass/fail summary alone.
- The `near_identical_scales_sanity` fixture is not a "should always
  pass" control: the shipped `int32_fixedpoint` kernel measurably
  diverges on it (documented in the fixture's own description and pinned
  by a regression test), because near-equal scales can still land a true
  sum exactly on a rounding boundary that fixed-point rescaling nudges
  across.

## License

MIT
