"""pitight CLI — pipeline integrity checker.

Usage:
    pitight status              Show artifact registry summary
    pitight check               Run all integrity checks
    pitight check --stale       Check for config-drifted artifacts
    pitight check --missing     Check for missing artifacts on disk
"""

from __future__ import annotations

import argparse
import json
import sys

from pitight.artifact import ArtifactRegistry


def cmd_status(args: argparse.Namespace) -> None:
    registry = ArtifactRegistry(args.registry)
    summary = registry.summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_check(args: argparse.Namespace) -> None:
    registry = ArtifactRegistry(args.registry)
    issues: list[str] = []

    # Check missing files
    if not args.stale_only:
        missing = registry.find_missing()
        for art in missing:
            issues.append(f"MISSING: {art.name} -> {art.path}")

    # Check stale (requires current configs)
    if not args.missing_only and args.config_file:
        configs = json.loads(open(args.config_file).read())
        stale = registry.find_stale(configs)
        for art in stale:
            issues.append(
                f"STALE: {art.name} (registered={art.config_hash[:10]}..., "
                f"current={configs[art.name][:10]}...)"
            )

    if issues:
        for issue in issues:
            print(f"  {issue}")
        print(f"\n{len(issues)} issue(s) found.")
        sys.exit(1)
    else:
        print("All checks passed.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pitight",
        description="Pipeline integrity checker for ML workflows.",
    )
    parser.add_argument(
        "--registry",
        default=".pitight/artifacts.json",
        help="Path to artifact registry file",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show artifact registry summary")

    check_p = sub.add_parser("check", help="Run integrity checks")
    check_p.add_argument("--stale-only", action="store_true")
    check_p.add_argument("--missing-only", action="store_true")
    check_p.add_argument("--config-file", help="JSON file of name→current_hash")

    args = parser.parse_args(argv)

    if args.command == "status":
        cmd_status(args)
    elif args.command == "check":
        cmd_check(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
