from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "exercise"
KEYWORDS = (
    "import ",
    "spacy",
    "transformers",
    "entity",
    "relation",
    "networkx",
    "knowledge graph",
    "co-occurrence",
    "cooccurrence",
)


def main() -> None:
    for path in sorted(ROOT.glob("*.ipynb")):
        print(f"\n===== {path.name} =====")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            source = "".join(cell.get("source", []))
            if any(keyword.lower() in source.lower() for keyword in KEYWORDS):
                print(f"\n--- CELL {index} [{cell.get('cell_type')}] ---")
                print(source[:5000])


if __name__ == "__main__":
    main()
