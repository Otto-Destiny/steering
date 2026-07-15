"""Capture a sequential range of X cases from the STEERING evaluation dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluate/examples.yaml"))
    parser.add_argument("--start", type=int, required=True, help="One-based case number")
    parser.add_argument("--end", type=int, required=True, help="Inclusive one-based case number")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--login-timeout",
        type=int,
        default=60,
        help="Maximum seconds to wait for each X thread to become available",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=14,
        help="Maximum thread scrolls per X candidate",
    )
    parser.add_argument(
        "--max-thread-span-hours",
        type=int,
        default=48,
        help="Keep same-author posts within this distance from the root post",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.start < 1 or args.end < args.start or args.end > len(cases):
        raise SystemExit(f"Invalid case range {args.start}..{args.end}")

    capture_script = Path(__file__).with_name("capture_x_thread.py")
    output_directory = Path(__file__).with_name("captures")
    output_directory.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for case_number in range(args.start, args.end + 1):
        case = cases[case_number - 1]
        source = case["sources"][0]
        case_id = case["id"]
        if source["platform"] != "x":
            print(f"[{case_number}] SKIP {case_id}: platform={source['platform']}", flush=True)
            continue

        output = output_directory / f"{case_id}.json"
        if output.exists() and not args.force:
            print(f"[{case_number}] EXISTS {case_id}", flush=True)
            continue

        url = source.get("canonical_url") or source["url"]
        print(f"[{case_number}] CAPTURE {case_id}", flush=True)
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and repository script
            [
                sys.executable,
                str(capture_script),
                str(url),
                "--candidate-id",
                case_id,
                "--output",
                str(output),
                "--login-timeout",
                str(args.login_timeout),
                "--max-scrolls",
                str(args.max_scrolls),
                "--max-thread-span-hours",
                str(args.max_thread_span_hours),
                "--headless",
            ],
            check=False,
        )
        if completed.returncode:
            failures.append(case_id)
            print(f"[{case_number}] FAILED {case_id}", flush=True)
        else:
            print(f"[{case_number}] COMPLETE {case_id}", flush=True)

    if failures:
        print("Failed cases: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
