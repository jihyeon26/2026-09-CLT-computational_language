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
    triple_key,
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



def test_competes_with_shares_one_key_for_both_orders() -> None:
    assert triple_key("org:openai", "competes with", "org:deepseek") == triple_key("org:deepseek", "competes with", "org:openai")
    assert triple_key("model:qwen", "developed by", "org:alibaba") != triple_key("org:alibaba", "developed by", "model:qwen")
