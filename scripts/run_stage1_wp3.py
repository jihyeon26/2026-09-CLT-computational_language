from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp3_knowledge_graph import WP3PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the contract-driven Stage 1 WP3 knowledge graph pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp2_open_models_pilot",
        help="Directory containing a WP2 delivery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp3_open_models_pilot",
        help="Directory for WP3 graph artifacts.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "config" / "wp3_open_models_contract.json",
        help="JSON contract defining filenames, column aliases, and graph policies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        WP3PipelineConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            contract_path=args.contract,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
