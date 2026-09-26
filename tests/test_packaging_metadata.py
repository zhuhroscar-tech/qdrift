"""Regression checks for modern packaging metadata.

Setuptools 77+ warns on the legacy ``license = {text = ...}`` table and
on the deprecated MIT trove classifier. Keep the project on the current
SPDX-style metadata so package builds stay warning-free.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _pyproject_text() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def test_license_metadata_uses_spdx_string_and_license_files():
    text = _pyproject_text()
    assert 'license = "MIT"' in text
    assert 'license = { text = "MIT" }' not in text
    assert 'license-files = ["LICENSE"]' in text


def test_deprecated_license_classifier_is_not_reintroduced():
    text = _pyproject_text()
    assert 'License :: OSI Approved :: MIT License' not in text


def test_setuptools_floor_supports_spdx_license_metadata():
    text = _pyproject_text()
    assert 'setuptools>=77' in text


def test_project_metadata_links_maintenance_resources():
    text = _pyproject_text()
    assert 'Homepage = "https://github.com/zhuhroscar-tech/qdrift"' in text
    assert 'Issues = "https://github.com/zhuhroscar-tech/qdrift/issues"' in text
    assert 'Changelog = "https://github.com/zhuhroscar-tech/qdrift/blob/main/CHANGELOG.md"' in text
