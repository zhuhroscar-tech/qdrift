"""qdrift CLI: exact-oracle checks for affine INT8 quantize/dequantize/
requantize-add arithmetic against real-runtime-style float kernels."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import __version__
from . import fixtures as fx
from .core import run_cross_scale_add_check, run_round_boundary_check
from .kernels import ROUND_MODES
from .style import Style, print_fields, resolve_style, section, status_headline


def cmd_check_round(args: argparse.Namespace) -> int:
    style = resolve_style(args.no_color)
    names = fx.list_round_boundary_fixtures() if args.fixture == "all" else [args.fixture]
    results = [run_round_boundary_check(n, round_mode=args.round_mode) for n in names]
    any_drift = any(r.verdict == "drift_detected" for r in results)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            level = "fail" if r.verdict == "drift_detected" else "ok"
            print(status_headline(style, level, f"round-boundary: {r.fixture_name} [{r.round_mode}]"))
            print_fields([
                ("probes", str(r.n_probes)),
                ("mismatches", str(len(r.mismatches))),
                ("verdict", r.verdict),
            ])
            for m in r.mismatches:
                print(f"    value={m.probe_value!r}  exact={m.exact_code}  candidate={m.candidate_code}")
            print()
    return 1 if any_drift else 0


def cmd_check_add(args: argparse.Namespace) -> int:
    style = resolve_style(args.no_color)
    names = fx.list_cross_scale_fixtures() if args.fixture == "all" else [args.fixture]
    results = [run_cross_scale_add_check(n, kernel_name=args.kernel) for n in names]
    any_drift = any(r.verdict == "drift_detected" for r in results)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for r in results:
            level = "fail" if r.verdict == "drift_detected" else "ok"
            print(status_headline(style, level, f"cross-scale add: {r.fixture_name} [{r.kernel_name}]"))
            print_fields([
                ("probes", str(r.n_probes)),
                ("mismatches", str(len(r.mismatches))),
                ("verdict", r.verdict),
            ])
            for m in r.mismatches:
                print(
                    f"    qa={m.qa} qb={m.qb}  exact_code={m.exact_code}  "
                    f"candidate_code={m.candidate_code}  exact_real_sum={m.exact_real_sum:.6f}"
                )
            print()
    return 1 if any_drift else 0


def cmd_list_fixtures(args: argparse.Namespace) -> int:
    style = resolve_style(args.no_color)
    section("round-boundary fixtures")
    for name in fx.list_round_boundary_fixtures():
        print(f"  {name}")
    section("cross-scale-add fixtures")
    for name in fx.list_cross_scale_fixtures():
        print(f"  {name}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdrift",
        description="Exact-oracle correctness checks for affine INT8 quantization arithmetic.",
    )
    parser.add_argument("--version", action="version", version=f"qdrift {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    round_p = sub.add_parser(
        "check-round",
        help="check round-to-nearest-even boundary behavior against the exact oracle",
    )
    round_p.add_argument("--fixture", default="all", help="fixture name, or 'all' (default)")
    round_p.add_argument(
        "--round-mode", choices=list(ROUND_MODES), default="half_even",
        help="candidate kernel's rounding mode to test (default: half_even, the ONNX spec)",
    )
    round_p.add_argument("--json", action="store_true")
    round_p.add_argument("--no-color", action="store_true")
    round_p.set_defaults(func=cmd_check_round)

    add_p = sub.add_parser(
        "check-add",
        help="check cross-scale INT8 element-wise add + requantize against the exact oracle",
    )
    add_p.add_argument("--fixture", default="all", help="fixture name, or 'all' (default)")
    add_p.add_argument(
        "--kernel", choices=["float64", "int32_fixedpoint"], default="float64",
        help="candidate kernel implementation strategy to test (default: float64)",
    )
    add_p.add_argument("--json", action="store_true")
    add_p.add_argument("--no-color", action="store_true")
    add_p.set_defaults(func=cmd_check_add)

    list_p = sub.add_parser("list-fixtures", help="list available fixtures")
    list_p.add_argument("--no-color", action="store_true")
    list_p.set_defaults(func=cmd_list_fixtures)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
