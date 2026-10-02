# Stage 1 WP2 - Open Models

This implementation is a reproducible pilot for entity extraction, relation extraction, and entity co-occurrence over the AI Media Dataset.

## Method

1. Restrict dates to September 2024 through August 2026.
2. Select Open Models candidates from titles and curated dataset tags.
3. Parse the serialized paragraph lists and preserve a cleaned text view.
4. Segment the text into sentence-level evidence units.
5. Combine spaCy `en_core_web_sm` NER with an auditable Open Models `EntityRuler` catalogue.
6. Link mentions to stable entity IDs while preserving mention offsets.
7. Extract schema-constrained relation candidates with evidence sentences.
8. Build sentence-level entity co-occurrence edges and a top-75 matrix.
9. Export validation, review, and provenance artifacts for WP3.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\run_stage1_wp2.py --max-docs 300
```

Use `--max-docs 0` only after reviewing the pilot and confirming the candidate filter, catalogue, and relation rules.

## Limitations

- The relation extractor is a transparent schema-constrained baseline, not a replacement for manual review or a stronger structured/LLM extractor.
- `rule_score` is a heuristic score, not a calibrated probability.
- Generic spaCy entities are retained only in sentences containing a domain entity.
- Entity resolution is catalogue-based and does not yet perform contextual disambiguation.
- Counts and trends must not be interpreted until near-duplicates and extraction precision are reviewed.

