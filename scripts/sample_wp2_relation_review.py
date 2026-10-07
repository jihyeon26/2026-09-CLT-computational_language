from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRIPLE_KEY = ["head_id", "relation", "tail_id"]
REVIEWERS = ["Reviewer_A", "Reviewer_B", "Reviewer_C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible stratified sample for WP2 relation review."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp2_open_models_official_20261007",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "stage1_wp2" / "evaluation",
    )
    parser.add_argument("--sample-size", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-per-predicate", type=int, default=2)
    parser.add_argument("--minimum-sentence-support", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, purpose: str, *parts: Any) -> str:
    value = "|".join([str(seed), purpose, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score_band(value: float) -> str:
    if value < 0.60:
        return "0.50-0.59"
    if value < 0.75:
        return "0.60-0.74"
    return "0.75-1.00"


def allocate_proportionally(capacities: dict[str, int], slots: int) -> dict[str, int]:
    allocation = {name: 0 for name in capacities}
    if slots <= 0 or not capacities:
        return allocation

    available = sum(capacities.values())
    if slots > available:
        raise ValueError(f"Cannot allocate {slots} slots from {available} available rows.")

    # Give every non-empty confidence band one row when possible.
    non_empty = [name for name, capacity in capacities.items() if capacity > 0]
    if slots >= len(non_empty):
        for name in non_empty:
            allocation[name] = 1
        slots -= len(non_empty)

    remaining_capacity = {
        name: capacity - allocation[name] for name, capacity in capacities.items()
    }
    while slots:
        active = {name: cap for name, cap in remaining_capacity.items() if cap > 0}
        total = sum(active.values())
        if not total:
            raise ValueError("Sampling allocation exhausted before reaching the requested size.")

        raw = {name: slots * cap / total for name, cap in active.items()}
        whole = {
            name: min(int(raw[name]), remaining_capacity[name]) for name in active
        }
        used = sum(whole.values())
        for name, value in whole.items():
            allocation[name] += value
            remaining_capacity[name] -= value
        slots -= used

        if slots and used == 0:
            order = sorted(
                active,
                key=lambda name: (raw[name] - int(raw[name]), active[name], name),
                reverse=True,
            )
            for name in order:
                if slots == 0:
                    break
                if remaining_capacity[name] > 0:
                    allocation[name] += 1
                    remaining_capacity[name] -= 1
                    slots -= 1
    return allocation


def final_node_type_valid(
    row: pd.Series,
    node_types: dict[str, str],
    relation_types: dict[str, dict[str, Any]],
) -> bool:
    relation = relation_types[row["relation"]]
    return (
        node_types[row["head_id"]] in relation["head_types"]
        and node_types[row["tail_id"]] in relation["tail_types"]
    )


def build_population(
    input_dir: Path,
    minimum_sentence_support: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = {
        "nodes": input_dir / "nodes.csv",
        "edges": input_dir / "edges_relations.csv",
        "evidence": input_dir / "relations_evidence.csv",
        "schema": input_dir / "schema.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing WP2 input files: {missing}")

    nodes = pd.read_csv(paths["nodes"], low_memory=False)
    edges = pd.read_csv(paths["edges"], low_memory=False)
    evidence = pd.read_csv(paths["evidence"], low_memory=False)
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))

    node_types = nodes.set_index("entity_id")["entity_type"].to_dict()
    node_names = nodes.set_index("entity_id")["entity"].to_dict()
    relation_types = schema["edges"]["relation"]["types"]

    unknown_endpoints = sorted(
        (set(edges["head_id"]) | set(edges["tail_id"])) - set(node_types)
    )
    if unknown_endpoints:
        raise ValueError(f"Relation endpoints missing from nodes.csv: {unknown_endpoints[:10]}")

    edges["final_node_type_valid"] = edges.apply(
        final_node_type_valid,
        axis=1,
        node_types=node_types,
        relation_types=relation_types,
    )
    candidate_edges = edges[
        (edges["source"] == "glirel")
        & edges["type_valid"].fillna(False)
        & edges["final_node_type_valid"]
        & (edges["n_sentences"].fillna(0) >= minimum_sentence_support)
        & (edges["head_id"] != edges["tail_id"])
    ].copy()

    triples = candidate_edges[TRIPLE_KEY].drop_duplicates().reset_index(drop=True)
    if triples.empty:
        raise ValueError("No GLiREL triples remain after the configured filters.")

    candidate_evidence = evidence[evidence["source"] == "glirel"].merge(
        triples,
        on=TRIPLE_KEY,
        how="inner",
        validate="many_to_one",
    )
    candidate_evidence = candidate_evidence.dropna(
        subset=["doc_id", "sentence_id", "sentence_text", "score"]
    ).copy()
    candidate_evidence["sentence_id"] = candidate_evidence["sentence_id"].astype(int)
    candidate_evidence["evidence_key"] = [
        stable_key(seed, "evidence", h, r, t, d, s, score)
        for h, r, t, d, s, score in zip(
            candidate_evidence["head_id"],
            candidate_evidence["relation"],
            candidate_evidence["tail_id"],
            candidate_evidence["doc_id"],
            candidate_evidence["sentence_id"],
            candidate_evidence["score"],
        )
    ]

    support = (
        candidate_evidence.assign(
            sentence_key=(
                candidate_evidence["doc_id"].astype(str)
                + ":"
                + candidate_evidence["sentence_id"].astype(str)
            )
        )
        .groupby(TRIPLE_KEY, as_index=False)
        .agg(
            n_sentences=("sentence_key", "nunique"),
            n_docs=("doc_id", "nunique"),
            mean_score=("score", "mean"),
        )
    )

    selected_evidence = (
        candidate_evidence.sort_values("evidence_key")
        .drop_duplicates(TRIPLE_KEY, keep="first")
        [
            TRIPLE_KEY
            + [
                "doc_id",
                "sentence_id",
                "sentence_text",
                "score",
                "date",
                "quarter",
            ]
        ]
    )

    population = triples.merge(support, on=TRIPLE_KEY, validate="one_to_one").merge(
        selected_evidence,
        on=TRIPLE_KEY,
        validate="one_to_one",
    )
    population["head"] = population["head_id"].map(node_names)
    population["head_type"] = population["head_id"].map(node_types)
    population["tail"] = population["tail_id"].map(node_names)
    population["tail_type"] = population["tail_id"].map(node_types)
    population["score_band"] = population["mean_score"].map(score_band)
    population["sample_key"] = [
        stable_key(seed, "sample", h, r, t)
        for h, r, t in zip(
            population["head_id"], population["relation"], population["tail_id"]
        )
    ]

    audit = {
        "input_files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "raw_counts": {
            "nodes": int(len(nodes)),
            "aggregated_relation_rows": int(len(edges)),
            "relation_evidence_rows": int(len(evidence)),
            "glirel_evidence_rows": int((evidence["source"] == "glirel").sum()),
            "curated_wp1_lexicon_rows": int((evidence["source"] == "wp1_lexicon").sum()),
        },
        "population_counts": {
            "filtered_edge_rows": int(len(candidate_edges)),
            "unique_glirel_triples": int(len(population)),
            "candidate_evidence_rows": int(len(candidate_evidence)),
            "relation_types": int(population["relation"].nunique()),
        },
    }
    return population, audit


def sample_population(
    population: pd.DataFrame,
    sample_size: int,
    minimum_per_predicate: int,
    seed: int,
) -> pd.DataFrame:
    if sample_size > len(population):
        raise ValueError(
            f"Requested sample size {sample_size} exceeds population {len(population)}."
        )

    base_parts = []
    for _, group in population.groupby("relation", sort=True):
        take = min(minimum_per_predicate, len(group))
        base_parts.append(group.sort_values("sample_key").head(take))
    base = pd.concat(base_parts, ignore_index=False).drop_duplicates(TRIPLE_KEY)

    if len(base) > sample_size:
        raise ValueError(
            f"Predicate minimums require {len(base)} rows, above sample size {sample_size}."
        )

    remaining_slots = sample_size - len(base)
    remaining = population.merge(
        base[TRIPLE_KEY], on=TRIPLE_KEY, how="left", indicator=True
    )
    remaining = remaining[remaining["_merge"] == "left_only"].drop(columns="_merge")

    capacities = {
        str(name): int(count)
        for name, count in remaining["score_band"].value_counts().sort_index().items()
    }
    allocation = allocate_proportionally(capacities, remaining_slots)
    additional_parts = []
    for band, take in allocation.items():
        if take:
            band_rows = remaining[remaining["score_band"] == band]
            additional_parts.append(band_rows.sort_values("sample_key").head(take))
    additional = (
        pd.concat(additional_parts, ignore_index=False)
        if additional_parts
        else population.iloc[0:0].copy()
    )

    sample = pd.concat([base, additional], ignore_index=True).drop_duplicates(TRIPLE_KEY)
    if len(sample) != sample_size:
        raise AssertionError(f"Expected {sample_size} sampled triples, got {len(sample)}.")

    sample["selection_reason"] = "confidence_band_allocation"
    base_keys = set(map(tuple, base[TRIPLE_KEY].itertuples(index=False, name=None)))
    sample.loc[
        [tuple(row) in base_keys for row in sample[TRIPLE_KEY].to_numpy()],
        "selection_reason",
    ] = "predicate_minimum"

    sample["assignment_key"] = [
        stable_key(seed, "assignment", h, r, t)
        for h, r, t in zip(sample["head_id"], sample["relation"], sample["tail_id"])
    ]
    sample = sample.sort_values("assignment_key").reset_index(drop=True)
    sample["reviewer_assignment"] = [REVIEWERS[i % len(REVIEWERS)] for i in range(len(sample))]
    sample = sample.sort_values(
        ["reviewer_assignment", "relation", "sample_key"]
    ).reset_index(drop=True)

    sample.insert(0, "sample_id", [f"S{i:03d}" for i in range(1, len(sample) + 1)])
    sample.insert(
        1,
        "relation_id",
        [
            "rel_" + stable_key(seed, "relation_id", h, r, t)[:16]
            for h, r, t in zip(
                sample["head_id"], sample["relation"], sample["tail_id"]
            )
        ],
    )
    sample["text_unit_id"] = (
        sample["doc_id"].astype(str) + ":" + sample["sentence_id"].astype(str)
    )
    sample["extraction_method"] = "glirel"
    sample["review_correct"] = ""
    sample["reviewer"] = ""
    sample["review_error_type"] = ""
    sample["review_notes"] = ""

    columns = [
        "sample_id",
        "relation_id",
        "reviewer_assignment",
        "head_id",
        "head",
        "head_type",
        "relation",
        "tail_id",
        "tail",
        "tail_type",
        "doc_id",
        "text_unit_id",
        "date",
        "quarter",
        "sentence_text",
        "score",
        "mean_score",
        "score_band",
        "n_sentences",
        "n_docs",
        "extraction_method",
        "selection_reason",
        "review_correct",
        "reviewer",
        "review_error_type",
        "review_notes",
    ]
    return sample[columns].rename(
        columns={
            "relation": "predicate",
            "doc_id": "document_id",
            "date": "published_at",
            "sentence_text": "evidence_text",
            "score": "evidence_score",
        }
    )


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if args.minimum_per_predicate <= 0:
        raise ValueError("--minimum-per-predicate must be positive.")

    population, audit = build_population(
        args.input_dir,
        minimum_sentence_support=args.minimum_sentence_support,
        seed=args.seed,
    )
    sample = sample_population(
        population,
        sample_size=args.sample_size,
        minimum_per_predicate=args.minimum_per_predicate,
        seed=args.seed,
    )

    if sample["relation_id"].duplicated().any():
        raise AssertionError("Sample contains duplicate relation_id values.")
    if sample[TRIPLE_KEY[0]].isna().any() or sample["tail_id"].isna().any():
        raise AssertionError("Sample contains missing relation endpoints.")
    if sample["evidence_text"].isna().any():
        raise AssertionError("Sample contains missing evidence text.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"relation_review_sample_seed{args.seed}.csv"
    manifest_path = args.output_dir / "sampling_manifest.json"
    sample.to_csv(csv_path, index=False, encoding="utf-8")

    manifest = {
        "version": "wp2-relation-review-sample-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sampling_unit": "unique (head_id, predicate, tail_id) GLiREL triple",
        "selection_policy": {
            "source": "glirel",
            "original_type_valid_required": True,
            "final_node_type_valid_required": True,
            "minimum_sentence_support": args.minimum_sentence_support,
            "self_relations_excluded": True,
            "curated_wp1_lexicon_relations_excluded": True,
            "minimum_per_predicate": args.minimum_per_predicate,
            "remaining_rows": "proportional allocation across mean-score bands",
            "evidence_selection": "one deterministic random evidence sentence per triple",
        },
        "random_seed": args.seed,
        "requested_sample_size": args.sample_size,
        **audit,
        "sample_counts": {
            "rows": int(len(sample)),
            "relation_types": int(sample["predicate"].nunique()),
            "by_predicate": {
                str(name): int(count)
                for name, count in sample["predicate"].value_counts().sort_index().items()
            },
            "by_score_band": {
                str(name): int(count)
                for name, count in sample["score_band"].value_counts().sort_index().items()
            },
            "by_reviewer_assignment": {
                str(name): int(count)
                for name, count in sample["reviewer_assignment"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "outputs": {
            "sample_csv": str(csv_path.resolve()),
            "sample_csv_sha256": sha256_file(csv_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["population_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["sample_counts"], ensure_ascii=False, indent=2))
    print(f"Sample: {csv_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
