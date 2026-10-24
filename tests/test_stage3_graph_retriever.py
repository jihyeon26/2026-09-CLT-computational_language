from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage3_graph_rag import GraphEvidenceRetriever


def retriever() -> GraphEvidenceRetriever:
    nodes = pd.DataFrame(
        [
            {"entity_id": "model:llama3", "canonical_name": "Llama 3", "entity_type": "MODEL", "aliases": json.dumps(["Llama-3"])},
            {"entity_id": "org:meta", "canonical_name": "Meta", "entity_type": "ORGANIZATION", "aliases": "[]"},
            {"entity_id": "license:llama", "canonical_name": "Llama Community License", "entity_type": "LICENSE", "aliases": "[]"},
        ]
    )
    assertions = pd.DataFrame(
        [
            {"relation_id": "r1", "subject_id": "model:llama3", "predicate": "DEVELOPED_BY", "object_id": "org:meta", "document_id": "d1", "text_unit_id": "s1", "evidence_text": "Meta developed Llama 3.", "published_at": "2025-01-10", "rule_score": 0.9},
            {"relation_id": "r2", "subject_id": "model:llama3", "predicate": "LICENSED_UNDER", "object_id": "license:llama", "document_id": "d2", "text_unit_id": "s2", "evidence_text": "Llama uses its community license.", "published_at": "2025-02-10", "rule_score": 0.8},
        ]
    )
    text_units = pd.DataFrame(
        [
            {"text_unit_id": "s1", "document_id": "d1", "text_unit_text": "Meta is the developer of Llama 3."},
            {"text_unit_id": "s2", "document_id": "d2", "text_unit_text": "The model is released under the Llama Community License."},
        ]
    )
    documents = pd.DataFrame(
        [
            {"document_id": "d1", "title": "Llama launch", "url": "https://example.test/d1", "domain": "example.test", "published_at": "2025-01-10"},
            {"document_id": "d2", "title": "Llama license", "url": "https://example.test/d2", "domain": "example.test", "published_at": "2025-02-10"},
        ]
    )
    return GraphEvidenceRetriever(nodes, assertions, text_units, documents)


def test_entity_resolution_and_evidence_provenance() -> None:
    result = retriever().retrieve("Who developed Llama-3?", top_k=5)
    assert result["matched_entities"][0]["entity_id"] == "model:llama3"
    assert result["evidence"][0]["predicate"] == "DEVELOPED_BY"
    assert result["evidence"][0]["passage"] == "Meta is the developer of Llama 3."
    assert result["evidence"][0]["url"] == "https://example.test/d1"
    assert result["diagnostics"]["missing_text_units"] == 0
    assert result["diagnostics"]["missing_documents"] == 0


def test_predicate_and_date_filters_are_applied() -> None:
    result = retriever().retrieve(
        "Llama 3 license",
        predicates=["LICENSED_UNDER"],
        start_date="2025-02-01",
        top_k=5,
    )
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["relation_id"] == "r2"


def test_no_entity_match_returns_diagnostic_instead_of_unrelated_evidence() -> None:
    result = retriever().retrieve("What happened to an unknown model?")
    assert result["evidence"] == []
    assert result["diagnostics"]["reason"] == "no_entity_match"


def test_explicit_unknown_entity_id_is_rejected() -> None:
    try:
        retriever().retrieve("ignored", entity_ids=["model:missing"])
    except ValueError as exc:
        assert "Unknown entity_ids" in str(exc)
    else:
        raise AssertionError("Expected unknown entity ID to be rejected")


def test_min_support_filters_and_ranks_passages() -> None:
    base = retriever()
    assertions = base.assertions.assign(support_score=[0.2, 0.9])
    supported = GraphEvidenceRetriever(base.nodes, assertions, base.text_units, base.documents)
    ranked = supported.retrieve("Llama 3", top_k=5)
    assert [item["relation_id"] for item in ranked["evidence"]] == ["r2", "r1"]
    filtered = supported.retrieve("Llama 3", top_k=5, min_support=0.5)
    assert [item["relation_id"] for item in filtered["evidence"]] == ["r2"]
    assert filtered["evidence"][0]["support_score"] == 0.9
    unchecked = GraphEvidenceRetriever(base.nodes, base.assertions.assign(support_score=[None, 0.2]), base.text_units,
                                       base.documents)
    kept = unchecked.retrieve("Llama 3", top_k=5, min_support=0.5)
    assert [item["relation_id"] for item in kept["evidence"]] == ["r1"]
    assert kept["evidence"][0]["support_score"] is None


def test_min_support_requires_support_scores() -> None:
    try:
        retriever().retrieve("Llama 3", min_support=0.5)
    except ValueError as exc:
        assert "support_score" in str(exc)
    else:
        raise AssertionError("Expected min_support without support scores to be rejected")


def test_background_facts_are_kept_apart_from_evidence() -> None:
    base = retriever()
    background = pd.DataFrame(
        [{"subject_id": "model:llama3", "predicate": "DEVELOPED_BY", "object_id": "org:meta",
          "knowledge_source": "curated_lexicon"}]
    )
    with_background = GraphEvidenceRetriever(base.nodes, base.assertions, base.text_units, base.documents, background)
    result = with_background.retrieve("Who developed Llama 3?", predicates=["DEVELOPED_BY"])
    assert result["background"] == [
        {"subject": {"entity_id": "model:llama3", "name": "Llama 3"}, "predicate": "DEVELOPED_BY",
         "object": {"entity_id": "org:meta", "name": "Meta"}, "knowledge_source": "curated_lexicon"}
    ]
    assert all(item["knowledge_source"] == "article_claim" for item in result["evidence"])
