from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "stage1_wp2_open_models.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Stage 1 — WP2: Open Models knowledge extraction

This notebook documents a **precision-first, auditable baseline** for entity extraction,
typed relation extraction, and sentence-level entity co-occurrence. It uses the AI Media
Dataset from September 2024 through August 2026 and retains source sentences and offsets
for every extracted assertion.

The pipeline combines a curated Open Models entity catalogue with spaCy statistical NER.
Its `rule_score` is a heuristic ranking score—not a probability or a claim of correctness.
The review sample must be manually labelled before reporting relation precision."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "dataset").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "outputs" / "stage1_wp2_open_models_pilot"
RUN_PIPELINE = False  # Set True only when regenerating the extraction artifacts.
ROOT, OUTPUT_DIR"""
        ),
        nbf.v4.new_markdown_cell(
            """## Reproducible extraction

The pilot selects up to 300 Open Models candidates using titles and curated tags, stratified
across months. Set `max_docs=0` only after manually reviewing the pilot. The complete source
text remains in Parquet; the original CSV is never modified."""
        ),
        nbf.v4.new_code_cell(
            """if RUN_PIPELINE:
    from wp2_open_models.pipeline import PipelineConfig, run_pipeline

    run_pipeline(
        PipelineConfig(
            input_csv=ROOT / "dataset" / "ai_media_dataset_20260911.csv",
            output_dir=OUTPUT_DIR,
            max_docs=300,
        )
    )"""
        ),
        nbf.v4.new_markdown_cell("## QA and coverage"),
        nbf.v4.new_code_cell(
            """qa = json.loads((OUTPUT_DIR / "qa_report.json").read_text(encoding="utf-8"))
summary_keys = [
    "rows_in_date_range", "open_model_candidates", "document_count", "sentence_count",
    "mention_count", "entity_count", "relation_count", "documents_with_mentions",
    "documents_with_relations", "document_mention_coverage_pct",
    "document_relation_coverage_pct", "invalid_relation_signatures",
    "mention_offset_failures", "hard_validation_passed",
]
pd.DataFrame({"metric": summary_keys, "value": [qa[key] for key in summary_keys]})"""
        ),
        nbf.v4.new_code_cell(
            """assert qa["hard_validation_passed"], "WP2 hard validation failed; inspect qa_report.json"
assert qa["mention_offset_failures"] == 0
assert qa["invalid_relation_signatures"] == 0"""
        ),
        nbf.v4.new_markdown_cell("## Entities"),
        nbf.v4.new_code_cell(
            """entities = pd.read_parquet(OUTPUT_DIR / "entities.parquet")
relations = pd.read_parquet(OUTPUT_DIR / "relations.parquet")
cooccurrence = pd.read_parquet(OUTPUT_DIR / "cooccurrence_edges.parquet")

entity_type_counts = entities["entity_type"].value_counts().sort_values()
ax = entity_type_counts.plot.barh(figsize=(8, 4), color="#4C78A8")
ax.set(title="Canonical entities by type", xlabel="Entity count", ylabel="")
plt.tight_layout()
plt.show()

entities.sort_values(["document_count", "mention_count"], ascending=False).head(20)[
    ["canonical_name", "entity_type", "mention_count", "document_count", "first_seen", "last_seen"]
]"""
        ),
        nbf.v4.new_markdown_cell(
            """Generic `PERSON`, `ORGANIZATION`, and `PRODUCT_OTHER` nodes come from statistical
NER and are retained only in sentences that also contain a domain entity. Curated model,
family, licence, architecture, and capability entities use stable IDs."""
        ),
        nbf.v4.new_markdown_cell("## Typed relations with evidence"),
        nbf.v4.new_code_cell(
            """relation_type_counts = relations["predicate"].value_counts().sort_values()
ax = relation_type_counts.plot.barh(figsize=(8, 4), color="#F58518")
ax.set(title="Extracted relation assertions by type", xlabel="Evidence-bearing assertions", ylabel="")
plt.tight_layout()
plt.show()

labels = entities.set_index("entity_id")["canonical_name"]
relation_view = relations.assign(
    subject_label=relations["subject_id"].map(labels),
    object_label=relations["object_id"].map(labels),
)
relation_view[relation_view["predicate"] != "BELONGS_TO_FAMILY"].sample(
    min(20, (relation_view["predicate"] != "BELONGS_TO_FAMILY").sum()), random_state=42
)[["subject_label", "predicate", "object_label", "assertion_status", "rule_score", "evidence_text"]]"""
        ),
        nbf.v4.new_markdown_cell("## Entity co-occurrence"),
        nbf.v4.new_code_cell(
            """cooccurrence_view = cooccurrence.assign(
    source_label=cooccurrence["source_id"].map(labels),
    target_label=cooccurrence["target_id"].map(labels),
)
cooccurrence_view.sort_values(
    ["document_count", "sentence_count"], ascending=False
).head(25)[
    ["source_label", "target_label", "sentence_count", "document_count", "first_seen", "last_seen"]
]"""
        ),
        nbf.v4.new_markdown_cell(
            """## Manual review protocol

Open `relation_review_sample.csv` and label `review_correct` as `TRUE`, `FALSE`, or
`AMBIGUOUS` using the quoted evidence sentence and linked source. Rejected rows also require
`review_error_type`. At least two team members should independently review disagreements.
Report precision by predicate; do not call coverage “recall” without an annotated gold set."""
        ),
        nbf.v4.new_code_cell(
            """review = pd.read_csv(OUTPUT_DIR / "relation_review_sample.csv")
review[[
    "subject_label", "predicate", "object_label", "evidence_text", "title", "url",
    "review_correct", "reviewer", "review_error_type", "review_notes",
]].head(10)"""
        ),
        nbf.v4.new_markdown_cell(
            """## WP3 handoff

- `entities.parquet` and `relations.parquet` are the canonical graph tables.
- `knowledge_graph.graphml` is a visualization-friendly export.
- `knowledge_graph.json` is a NetworkX node-link export for programmatic loading.
- `schema.json` defines allowed edge signatures.
- `DATA_DICTIONARY.md`, `qa_report.json`, and `run_manifest.json` document semantics,
  integrity checks, configuration, versions, and input provenance.

For Stage 3, retrieve text evidence by `document_id`/`text_unit_id`; graph triples alone
should not be presented as sufficient evidence."""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
