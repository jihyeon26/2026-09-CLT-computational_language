# Contract-driven Stage 1 WP3 pipeline

This pipeline is intentionally decoupled from the current WP2 implementation. It reads a JSON contract that
defines candidate filenames, accepted column aliases, optional tables, validation strictness, and graph policies.

## Why this design

The official WP2 delivery is not yet fixed. A future delivery may rename `entities.parquet` to `nodes.csv`, use
`source/relation/target` instead of `subject_id/predicate/object_id`, omit optional evidence columns, or add new
entity and relation types. These changes should normally require editing only
`config/wp3_open_models_contract.json`.

The graph analysis code always receives the same canonical internal fields after adaptation.

## Run with the current pilot

```powershell
.\.venv\Scripts\python.exe scripts\run_stage1_wp3.py
```

The executed analysis notebook is `notebooks/stage1_wp3_knowledge_graph.ipynb`. Its first configuration
cell contains the upstream input directory, output directory, contract path, and `RUN_PIPELINE` switch.

## Run with a future WP2 delivery

```powershell
.\.venv\Scripts\python.exe scripts\run_stage1_wp3.py `
  --input-dir path\to\official_wp2 `
  --output-dir outputs\stage1_wp3_official `
  --contract config\wp3_open_models_contract.json
```

Before running, add any new filenames or source-column aliases to the contract. Do not rename the official WP2
files by hand; the contract report should document how they were adapted.

## Pipeline layers

1. Contract adapter: locates Parquet/CSV tables and maps aliases to canonical fields.
2. Validation: checks IDs, endpoints, self-relations, and upstream relation signatures.
3. Policy filter: handles assertion status, manual reviews, self-edges, confidence, and co-occurrence thresholds.
4. Graph construction: creates assertion, collapsed typed, and co-occurrence graph views.
5. Analysis: computes whole-graph and predicate-specific PageRank, weighted degree, betweenness, and clustering.
6. Trend/community analysis: creates monthly relation/entity/edge tables and weighted co-occurrence communities.
7. Human-review preparation: ranks evidence-linked finding candidates without treating them as final conclusions.
8. Export: writes tables, GraphML, node-link JSON, eight figures, QA reports, graph schema, contract, and manifest.

## Canonical minimum contract

Required entity fields:

- `entity_id`
- `canonical_name`
- `entity_type`

Required relation fields:

- `subject_id`
- `predicate`
- `object_id`

Evidence, dates, scores, documents, text units, co-occurrence, reviews, and schema are optional at the adapter
level. The contract report warns when optional sources are absent. For the final project submission, evidence and
source linkage should be treated as required because Stage 3 RAG needs provenance.

## Finalization boundary

The pipeline automates graph construction, metrics, temporal summaries, community detection, visualization,
schema export, and reproducibility checks. After an official WP2 delivery arrives, the remaining human decisions
are to approve the WP2 relation review, freeze the adapter contract with strict provenance, and approve the final
visuals and evidence-backed findings. `graph_finding_candidates.*` is deliberately a review queue, not a list of
claims that may be copied into the report without checking source passages.

## Change scenarios

- Filename changed: add it to the table's `files` list.
- Column renamed: add the new name to the canonical field's alias list.
- New assertion status: update `allowed_assertion_statuses`.
- New relation type: update the upstream `schema.json`; strict validation will then accept it.
- Different review policy: change `exclude_reviewed_false` or `require_reviewed_true`.
- Dense co-occurrence graph: raise `minimum_cooccurrence_documents`.
- Official delivery cannot be adapted: validation fails before graph artifacts are produced.
