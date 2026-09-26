# Changelog

## v0.1.3 — 2026-09-26

- Added package metadata links for issue reporting and release history.
- Made release-tag CI coverage explicit for `v*` tags so source releases rerun the same validation as main pushes.
- Added repository-contract coverage for package resource links and tag-triggered CI wiring.

## v0.1.2 — 2026-09-24

- Added release-history documentation and repository-contract checks for project completeness.
- Linked the changelog from both English and Chinese READMEs.
- Kept the package metadata, runtime `__version__`, and CLI release version aligned.

## v0.1.1 — 2026-09-24

- Modernized packaging license metadata to use an SPDX license string and `license-files`.
- Added regression coverage so deprecated license metadata is not reintroduced.
- Verified local tests, package build, CLI version output, GitHub Actions CI, CodeQL, and release artifacts.

## v0.1.0 — 2026-09-17

- Initial public release of the exact-oracle affine INT8 quantization arithmetic checker.
- Added round-boundary and cross-scale-add probes, CLI commands, bilingual documentation, tests, CI, and packaged release artifacts.
