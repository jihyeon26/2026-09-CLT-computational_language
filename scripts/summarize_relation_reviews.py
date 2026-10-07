from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp2_open_models.review import review_summary_markdown, summarize_relation_reviews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and validate WP2 relation reviews.")
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp2_open_models_pilot" / "relation_review_sample.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "relation_review",
    )
    parser.add_argument("--minimum-reviewed", type=int, default=90)
    parser.add_argument("--minimum-precision", type=float, default=0.80)
    parser.add_argument(
        "--fail-on",
        choices=["never", "fail", "pending"],
        default="never",
        help="Choose which report status produces a non-zero exit code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviews = pd.read_csv(args.reviews)
    summary = summarize_relation_reviews(
        reviews,
        minimum_reviewed=args.minimum_reviewed,
        minimum_precision=args.minimum_precision,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "relation_review_report.json"
    markdown_path = args.output_dir / "relation_review_report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(review_summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path.resolve()}")
    print(f"Markdown report: {markdown_path.resolve()}")

    should_fail = args.fail_on == "pending" and summary["status"] != "PASS"
    should_fail = should_fail or (args.fail_on == "fail" and summary["status"] == "FAIL")
    if should_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
