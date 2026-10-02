from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage3_graph_rag import GraphEvidenceRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve source passages through the Stage 1 knowledge graph."
    )
    parser.add_argument("query", help="Question or text containing one or more entity names.")
    parser.add_argument(
        "--wp2-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp2_open_models_pilot",
    )
    parser.add_argument(
        "--wp3-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp3_open_models_pilot",
    )
    parser.add_argument("--max-hops", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--predicate", action="append", dest="predicates")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    retriever = GraphEvidenceRetriever.from_directories(args.wp3_dir, args.wp2_dir)
    result = retriever.retrieve(
        args.query,
        max_hops=args.max_hops,
        predicates=args.predicates,
        start_date=args.start_date,
        end_date=args.end_date,
        top_k=args.top_k,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
