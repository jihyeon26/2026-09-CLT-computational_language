from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_wp3_candidate_sensitivity import (
    ANALYSIS_STATUS,
    build_cooccurrence_graph,
    build_relation_graph,
    collapse_candidate_relations,
    graph_statistics,
)


def test_candidate_relation_graph_collapses_evidence_and_keeps_status() -> None:
    relations = pd.DataFrame(
        [
            {
                "subject_id": "model:a",
                "predicate": "developed by",
                "predicate_group": "PROVENANCE",
                "object_id": "org:x",
                "rule_score": 0.7,
                "published_at": "2025-01-01",
                "document_id": "d1",
                "text_unit_id": "d1:0",
            },
            {
                "subject_id": "model:a",
                "predicate": "developed by",
                "predicate_group": "PROVENANCE",
                "object_id": "org:x",
                "rule_score": 0.9,
                "published_at": "2025-02-01",
                "document_id": "d2",
                "text_unit_id": "d2:0",
            },
        ]
    )
    nodes = pd.DataFrame(
        [
            {
                "entity_id": "model:a",
                "canonical_name": "A",
                "raw_entity_type": "AI_MODEL",
                "entity_type_compact": "MODEL",
            },
            {
                "entity_id": "org:x",
                "canonical_name": "X",
                "raw_entity_type": "COMPANY",
                "entity_type_compact": "ORGANIZATION",
            },
        ]
    )

    collapsed = collapse_candidate_relations(relations)
    graph = build_relation_graph(nodes, collapsed, "test")
    summary = graph_statistics(graph, assertion_count=len(relations))

    assert len(collapsed) == 1
    assert collapsed.loc[0, "evidence_count"] == 2
    assert collapsed.loc[0, "document_count"] == 2
    assert graph.graph["validation_status"] == ANALYSIS_STATUS
    assert summary["assertion_count"] == 2
    assert summary["collapsed_edge_count"] == 1


def test_cooccurrence_threshold_is_applied_by_document_count() -> None:
    edges = pd.DataFrame(
        [
            {
                "source_id": "a",
                "target_id": "b",
                "document_count": 1,
                "sentence_count": 2,
            },
            {
                "source_id": "b",
                "target_id": "c",
                "document_count": 3,
                "sentence_count": 5,
            },
        ]
    )

    graph = build_cooccurrence_graph(edges, threshold=2)

    assert set(graph.nodes) == {"b", "c"}
    assert graph.number_of_edges() == 1
    assert graph.graph["minimum_document_count"] == 2
