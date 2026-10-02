from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.quality_gates.gates import report_to_markdown, run_all_gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage 1 WP1/WP2/WP3 artifact sufficiency.")
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "tests" / "quality_gates" / "quality_contract.json",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs" / "quality_gates",
    )
    parser.add_argument(
        "--fail-on",
        choices=["fail", "pending", "never"],
        default="fail",
        help="Exit non-zero on FAIL only, on FAIL/PENDING, or never.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_all_gates(ROOT, args.contract)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "stage1_quality_report.json"
    markdown_path = args.report_dir / "stage1_quality_report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(report_to_markdown(report), encoding="utf-8")

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    if args.fail_on == "fail" and report.status == "FAIL":
        raise SystemExit(1)
    if args.fail_on == "pending" and report.status in {"FAIL", "PENDING"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
