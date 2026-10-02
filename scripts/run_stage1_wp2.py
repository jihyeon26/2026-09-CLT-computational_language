from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp2_open_models import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 1 WP2 Open Models extraction baseline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "dataset" / "ai_media_dataset_20260911.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp2_open_models_pilot",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=300,
        help="Maximum stratified candidate documents. Use 0 for all candidates.",
    )
    parser.add_argument("--review-sample-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        input_csv=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        max_docs=args.max_docs,
        review_sample_size=args.review_sample_size,
    )
    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

