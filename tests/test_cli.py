"""CLI integration tests: end-to-end subprocess invocation, exit codes,
JSON output validity.
"""
from __future__ import annotations

import json
import subprocess
import sys


def run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "qdrift.cli", *args],
        capture_output=True,
        text=True,
    )


def test_version_flag():
    result = run_cli(["--version"])
    assert result.returncode == 0
    assert "qdrift" in result.stdout


def test_list_fixtures():
    result = run_cli(["list-fixtures", "--no-color"])
    assert result.returncode == 0
    assert "onnx_qlinear_half_ties_fp16_input" in result.stdout
    assert "arm_residual_add_disparate_scales" in result.stdout


def test_check_round_default_mode_exits_zero():
    result = run_cli(["check-round", "--no-color"])
    assert result.returncode == 0, result.stdout


def test_check_round_buggy_mode_exits_nonzero():
    result = run_cli(["check-round", "--round-mode", "half_down_bug", "--no-color"])
    assert result.returncode == 1, result.stdout
    assert "drift_detected" in result.stdout


def test_check_round_json_output_is_valid_json():
    result = run_cli(["check-round", "--fixture", "onnx_qlinear_half_ties_fp16_input", "--json"])
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)
    assert parsed[0]["verdict"] == "exact"


def test_check_add_default_kernel_exits_zero():
    result = run_cli(["check-add", "--no-color"])
    assert result.returncode == 0, result.stdout


def test_check_add_fixedpoint_kernel_on_sanity_fixture_exits_nonzero():
    result = run_cli([
        "check-add", "--fixture", "near_identical_scales_sanity",
        "--kernel", "int32_fixedpoint", "--no-color",
    ])
    assert result.returncode == 1, result.stdout


def test_check_add_json_output_is_valid_json():
    result = run_cli(["check-add", "--json"])
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)
    assert all("verdict" in r for r in parsed)


def test_unknown_command_exits_nonzero():
    result = run_cli(["not-a-real-command"])
    assert result.returncode != 0
