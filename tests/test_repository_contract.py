"""Repository-level completeness contract checks."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
CHANGELOG = ROOT / "CHANGELOG.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CODEQL_WORKFLOW = ROOT / ".github" / "workflows" / "codeql.yml"

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def _project_version() -> str:
    match = _VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    assert match, "pyproject.toml must define [project].version"
    return match.group(1)


def test_core_project_files_are_present():
    for path in (README, README_ZH, CHANGELOG, ROOT / "LICENSE", CI_WORKFLOW, CODEQL_WORKFLOW):
        assert path.exists(), f"Missing required repository file: {path.relative_to(ROOT)}"


def test_readmes_link_release_history_license_and_downloads():
    english = README.read_text(encoding="utf-8")
    chinese = README_ZH.read_text(encoding="utf-8")
    for text in (english, chinese):
        assert "CHANGELOG.md" in text, "README must link release history"
        assert "LICENSE" in text or "License" in text or "许可证" in text
        assert "https://github.com/zhuhroscar-tech/qdrift/releases" in text
        assert "SHA256SUMS.txt" in text


def test_changelog_documents_current_version():
    changelog = CHANGELOG.read_text(encoding="utf-8")
    version = _project_version()
    assert f"## v{version}" in changelog
    assert "v0.1.1" in changelog
    assert "v0.1.0" in changelog


def test_ci_builds_release_artifacts_and_checksums():
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "python -m build" in workflow
    assert "sha256sum" in workflow
    assert "actions/upload-artifact" in workflow
    assert "dist/" in workflow


def test_codeql_workflow_analyzes_python():
    workflow = CODEQL_WORKFLOW.read_text(encoding="utf-8")
    assert "github/codeql-action/init" in workflow
    assert "languages: python" in workflow
