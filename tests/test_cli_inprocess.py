"""In-process CLI tests: exercise qdrift.cli.main() via direct calls
(not subprocess), so pytest-cov actually measures cli.py's statements.

qdrift's existing test_cli.py drives the CLI exclusively through
`subprocess.run([sys.executable, "-m", "qdrift.cli", ...])`. A subprocess
is a separate interpreter process that coverage.py cannot instrument, so
qdrift's actual coverage on cli.py was 0% (76/76 lines "missing") despite
10 passing subprocess tests -- silently hiding the one file most likely
to have an untested branch. This mirrors the fleet-wide fix already
applied to recallwatch (test_cli_inprocess.py) and matches the pattern
used by blasdrift, numguard, causality-audit, and other owned CLIs
(a companion in-process file alongside the subprocess-based one). The
working subprocess tests are kept untouched; this file only adds
coverage.
"""
from __future__ import annotations

import json

import pytest

from qdrift import cli


def _run(args, capsys):
    exit_code = cli.main(args)
    return exit_code, capsys.readouterr().out


def test_version_flag_inprocess(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "qdrift" in out


def test_list_fixtures_inprocess(capsys):
    exit_code, out = _run(["list-fixtures", "--no-color"], capsys)
    assert exit_code == 0
    assert "onnx_qlinear_half_ties_fp16_input" in out
    assert "arm_residual_add_disparate_scales" in out


def test_check_round_default_mode_exits_zero_inprocess(capsys):
    exit_code, out = _run(["check-round", "--no-color"], capsys)
    assert exit_code == 0, out


def test_check_round_buggy_mode_exits_nonzero_inprocess(capsys):
    exit_code, out = _run(
        ["check-round", "--round-mode", "half_down_bug", "--no-color"], capsys
    )
    assert exit_code == 1, out
    assert "drift_detected" in out


def test_check_round_single_fixture_ok_text_output_inprocess(capsys):
    exit_code, out = _run(
        ["check-round", "--fixture", "onnx_qlinear_half_ties_fp16_input", "--no-color"],
        capsys,
    )
    assert exit_code == 0, out
    assert "round-boundary: onnx_qlinear_half_ties_fp16_input" in out


def test_check_round_json_output_is_valid_json_inprocess(capsys):
    exit_code, out = _run(
        ["check-round", "--fixture", "onnx_qlinear_half_ties_fp16_input", "--json"],
        capsys,
    )
    assert exit_code == 0
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["verdict"] == "exact"


def test_check_add_default_kernel_exits_zero_inprocess(capsys):
    exit_code, out = _run(["check-add", "--no-color"], capsys)
    assert exit_code == 0, out


def test_check_add_fixedpoint_kernel_on_sanity_fixture_exits_nonzero_inprocess(capsys):
    exit_code, out = _run(
        [
            "check-add", "--fixture", "near_identical_scales_sanity",
            "--kernel", "int32_fixedpoint", "--no-color",
        ],
        capsys,
    )
    assert exit_code == 1, out


def test_check_add_json_output_is_valid_json_inprocess(capsys):
    exit_code, out = _run(["check-add", "--json"], capsys)
    assert exit_code == 0
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert all("verdict" in r for r in parsed)


def test_check_add_text_output_shows_mismatch_details_on_drift_inprocess(capsys):
    exit_code, out = _run(
        [
            "check-add", "--fixture", "near_identical_scales_sanity",
            "--kernel", "int32_fixedpoint", "--no-color",
        ],
        capsys,
    )
    assert exit_code == 1
    assert "qa=" in out
    assert "exact_code=" in out


def test_unknown_command_exits_nonzero_inprocess():
    with pytest.raises(SystemExit):
        cli.main(["not-a-real-command"])


def test_invalid_round_mode_rejected_by_argparse_inprocess():
    with pytest.raises(SystemExit):
        cli.main(["check-round", "--round-mode", "not-a-real-mode"])
