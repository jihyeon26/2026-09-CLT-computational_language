from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage3_kg_revision import (
    aggregate_support,
    build_pairs,
    edge_layers,
    hypothesis_name,
    lexicon_conflicts,
    score_pairs,
    threshold_table,
)


def test_edge_layers_count_article_support_without_lexicon_rows() -> None:
    assertions = pd.DataFrame([
        {"subject_id": "model:llama", "predicate": "developed by", "object_id": "org:meta", "document_id": "d1",
         "text_unit_id": "d1:1", "published_at": "2025-01-01", "extraction_method": "glirel"},
        {"subject_id": "model:llama", "predicate": "developed by", "object_id": "org:meta", "document_id": None,
         "text_unit_id": None, "published_at": None, "extraction_method": "wp1_lexicon"},
        {"subject_id": "model:phi", "predicate": "developed by", "object_id": "org:microsoft", "document_id": None,
         "text_unit_id": None, "published_at": None, "extraction_method": "wp1_lexicon"},
        {"subject_id": "model:qwen3_8b", "predicate": "member of family", "object_id": "model:qwen", "document_id": "d2",
         "text_unit_id": "d2:4", "published_at": "2025-02-01", "extraction_method": "model_name"},
    ])
    collapsed = assertions.drop_duplicates(["subject_id", "predicate", "object_id"])[["subject_id", "predicate", "object_id"]]
    edges = edge_layers(collapsed, assertions).set_index("subject_id")
    assert edges.loc["model:llama", "knowledge_layer"] == "article_claim"
    assert edges.loc["model:llama", "lexicon_agreement"] == "lexicon_and_articles"
    assert edges.loc["model:llama", "article_sentences"] == 1
    assert edges.loc["model:phi", "knowledge_layer"] == "curated_lexicon"
    assert edges.loc["model:phi", "article_sentences"] == 0
    assert edges.loc["model:qwen3_8b", "knowledge_layer"] == "name_pattern"


def test_lexicon_conflicts_marks_relations_excluded_only_by_the_lexicon() -> None:
    tiers = pd.DataFrame([
        {"head_id": "model:llama", "relation": "developed by", "tail_id": "org:nvidia", "head": "Llama", "tail": "NVIDIA",
         "n_distinct_sentences": 5, "n_docs": 4, "exclusion_reason": "conflicts_lexicon_developer"},
        {"head_id": "model:llama", "relation": "developed by", "tail_id": "org:x", "head": "Llama", "tail": "x",
         "n_distinct_sentences": 1, "n_docs": 1, "exclusion_reason": "generic_node;conflicts_lexicon_developer"},
        {"head_id": "model:llama", "relation": "developed by", "tail_id": "org:meta", "head": "Llama", "tail": "Meta",
         "n_distinct_sentences": 9, "n_docs": 9, "exclusion_reason": None},
    ])
    conflicts = lexicon_conflicts(tiers)
    assert conflicts["tail_id"].tolist() == ["org:nvidia", "org:x"]
    assert conflicts["only_reason"].tolist() == [True, False]


def test_hypothesis_uses_graph_names_and_the_part_present_in_the_sentence() -> None:
    assert hypothesis_name("Command R / Aya", "Cohere released Aya Vision.") == "Aya"
    assert hypothesis_name("united_states", "anything") == "united states"
    assert hypothesis_name("Llama", "NVIDIA released Llama Nemotron.") == "Llama"


def test_symmetric_relations_are_scored_in_both_orders_and_the_best_is_kept(tmp_path: Path) -> None:
    evidence = pd.DataFrame([
        {"key": "e1", "predicate": "competes with", "head_name": "OpenAI", "tail_name": "DeepSeek",
         "sentence_text": "DeepSeek rivals OpenAI."},
        {"key": "e2", "predicate": "developed by", "head_name": "Qwen", "tail_name": "Alibaba",
         "sentence_text": "Alibaba, Google and Meta released models."},
    ])
    pairs = build_pairs(evidence)
    assert pairs["direction"].tolist() == ["forward", "reverse", "forward"]

    class FakeVerifier:
        def predict(self, premises, hypotheses):
            return np.array([[0.9 if h.startswith("DeepSeek") else 0.2, 0.1] for h in hypotheses])

    scored = score_pairs(pairs, FakeVerifier(), tmp_path / "cache.parquet", "fake")
    sentences, edges = aggregate_support(evidence, scored, ["key"])
    assert sentences["support_score"].tolist() == [np.float32(0.9), np.float32(0.2)]
    assert edges.set_index("key")["support_status"].to_dict() == {"e1": "supported", "e2": "not_supported"}
    cached = score_pairs(pairs, None, tmp_path / "cache.parquet", "fake")  # served from the cache
    assert cached["entailment"].tolist() == scored["entailment"].tolist()


def test_threshold_table_reports_precision_of_kept_relations() -> None:
    table = threshold_table(pd.Series([True, True, False, False]), pd.Series([0.9, 0.4, 0.8, 0.1]), thresholds=[0.5])
    row = table.iloc[0]
    assert (row["kept"], row["precision"], row["recall_of_true"]) == (2, 0.5, 0.5)
    assert (row["removed_false"], row["removed_true"]) == (1, 1)


def test_list_sentences_do_not_support_rivalry() -> None:
    from stage3_kg_revision import sentence_actor_counts

    mentions = pd.DataFrame([
        {"entity_id": e, "doc_id": "d1", "sentence_id": 1} for e in ["org:openai", "org:google", "org:anthropic_2", "tech:ai"]
    ] + [{"entity_id": e, "doc_id": "d1", "sentence_id": 2} for e in ["org:openai", "org:deepseek"]])
    id_map = pd.Series({"org:anthropic_2": "org:anthropic"})
    compact = pd.Series({"org:openai": "ORGANIZATION", "org:google": "ORGANIZATION", "org:anthropic": "ORGANIZATION",
                         "org:deepseek": "ORGANIZATION", "tech:ai": "TECHNICAL_COMPONENT"})
    counts = sentence_actor_counts(mentions, id_map, compact, generic_ids={"tech:ai"})
    assert counts.to_dict() == {("d1", 1): 3, ("d1", 2): 2}

    evidence = pd.DataFrame([
        {"key": "e1", "predicate": "competes with", "n_actors": 3},
        {"key": "e1", "predicate": "competes with", "n_actors": 2},
        {"key": "e2", "predicate": "developed by", "n_actors": 3},
    ])
    scored = pd.DataFrame({"row_index": [0, 1, 2], "entailment": [0.99, 0.2, 0.9], "contradiction": [0.0, 0.1, 0.0],
                           "hypothesis": ["h"] * 3})
    sentences, edges = aggregate_support(evidence, scored, ["key"])
    assert sentences["supported"].tolist() == [False, False, True]
    assert edges.set_index("key")["support_status"].to_dict() == {"e1": "not_supported", "e2": "supported"}
