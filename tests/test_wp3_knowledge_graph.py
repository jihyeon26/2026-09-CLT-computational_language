from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp3_knowledge_graph.contract import ContractError, load_contract_tables
from wp3_knowledge_graph.pipeline import (
    WP3PipelineConfig,
    build_graph_schema,
    build_temporal_tables,
    collapse_relations,
    detect_cooccurrence_communities,
    run_pipeline,
    validate_inputs,
)


def minimal_contract() -> dict:
    return {
        "contract_version": "test-v1",
        "tables": {
            "entities": {
                "required": True,
                "files": ["nodes.csv"],
                "required_fields": ["entity_id", "canonical_name", "entity_type"],
                "columns": {
                    "entity_id": ["id"],
                    "canonical_name": ["label"],
                    "entity_type": ["type"],
                },
            },
            "relations": {
                "required": True,
                "files": ["edges.csv"],
                "required_fields": ["subject_id", "predicate", "object_id"],
                "columns": {
                    "subject_id": ["source"],
                    "predicate": ["relation"],
                    "object_id": ["target"],
                },
            },
        },
        "policies": {"strict_schema": False},
        "analysis": {"betweenness_sample_size": 10, "maximum_visual_nodes": 10},
    }


def test_contract_adapts_alias_columns(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"id": "model:a", "label": "Model A", "type": "MODEL"}]
    ).to_csv(tmp_path / "nodes.csv", index=False)
    pd.DataFrame(
        [{"source": "model:a", "relation": "DEVELOPED_BY", "target": "org:x"}]
    ).to_csv(tmp_path / "edges.csv", index=False)

    tables, report = load_contract_tables(tmp_path, minimal_contract())
    assert report["passed"]
    assert {"entity_id", "canonical_name", "entity_type"}.issubset(tables["entities"].columns)
    assert {"subject_id", "predicate", "object_id"}.issubset(tables["relations"].columns)


def test_contract_fails_when_required_field_is_missing(tmp_path: Path) -> None:
    pd.DataFrame([{"id": "model:a", "label": "Model A"}]).to_csv(
        tmp_path / "nodes.csv", index=False
    )
    pd.DataFrame([{"source": "model:a", "relation": "REL", "target": "org:x"}]).to_csv(
        tmp_path / "edges.csv", index=False
    )
    with pytest.raises(ContractError):
        load_contract_tables(tmp_path, minimal_contract())


def test_validation_rejects_dangling_relation() -> None:
    entities = pd.DataFrame(
        [{"entity_id": "model:a", "canonical_name": "Model A", "entity_type": "MODEL"}]
    )
    relations = pd.DataFrame(
        [
            {
                "relation_id": "r1",
                "subject_id": "model:a",
                "predicate": "DEVELOPED_BY",
                "object_id": "org:missing",
            }
        ]
    )
    report = validate_inputs(entities, relations, pd.DataFrame(), None, {})
    assert not report["passed"]
    assert report["dangling_relation_objects"] == 1


def test_collapse_preserves_counts_and_provenance() -> None:
    relations = pd.DataFrame(
        [
            {
                "relation_id": "r1",
                "subject_id": "model:a",
                "predicate": "DEVELOPED_BY",
                "object_id": "org:x",
                "document_id": "d1",
                "text_unit_id": "s1",
                "published_at": "2025-01-01",
                "assertion_status": "asserted",
                "rule_score": 0.8,
            },
            {
                "relation_id": "r2",
                "subject_id": "model:a",
                "predicate": "DEVELOPED_BY",
                "object_id": "org:x",
                "document_id": "d2",
                "text_unit_id": "s2",
                "published_at": "2025-02-01",
                "assertion_status": "asserted",
                "rule_score": 0.9,
            },
        ]
    )
    documents = pd.DataFrame(
        [{"document_id": "d1", "domain": "a.example"}, {"document_id": "d2", "domain": "b.example"}]
    )
    collapsed = collapse_relations(relations, documents)
    assert len(collapsed) == 1
    assert collapsed.iloc[0]["evidence_count"] == 2
    assert collapsed.iloc[0]["document_count"] == 2
    assert collapsed.iloc[0]["domain_count"] == 2
    assert json.loads(collapsed.iloc[0]["text_unit_ids"]) == ["s1", "s2"]


def test_end_to_end_pipeline_accepts_alias_contract(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    pd.DataFrame(
        [
            {"id": "model:a", "label": "Model A", "type": "MODEL"},
            {"id": "org:x", "label": "Org X", "type": "ORGANIZATION"},
        ]
    ).to_csv(input_dir / "nodes.csv", index=False)
    pd.DataFrame(
        [
            {
                "source": "model:a",
                "relation": "DEVELOPED_BY",
                "target": "org:x",
            }
        ]
    ).to_csv(input_dir / "edges.csv", index=False)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(minimal_contract()), encoding="utf-8")

    summary = run_pipeline(
        WP3PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            contract_path=contract_path,
        )
    )
    assert summary["collapsed_typed_edges"] == 1
    assert (output_dir / "typed_graph_collapsed.graphml").exists()
    assert (output_dir / "contract_report.json").exists()
    assert (output_dir / "graph_schema.json").exists()
    assert (output_dir / "relation_type_summary.parquet").exists()
    assert (output_dir / "monthly_relation_assertions.parquet").exists()
    assert (output_dir / "graph_finding_candidates.csv").exists()


def test_temporal_tables_separate_month_and_predicate() -> None:
    entities = pd.DataFrame(
        [
            {"entity_id": "model:a", "entity_type": "MODEL", "first_seen": "2025-01-03"},
            {"entity_id": "org:x", "entity_type": "ORGANIZATION", "first_seen": "2025-02-04"},
        ]
    )
    relations = pd.DataFrame(
        [
            {"relation_id": "r1", "document_id": "d1", "predicate": "DEVELOPED_BY", "published_at": "2025-01-05"},
            {"relation_id": "r2", "document_id": "d2", "predicate": "DEVELOPED_BY", "published_at": "2025-02-05"},
        ]
    )
    collapsed = pd.DataFrame(
        [
            {"edge_id": "e1", "predicate": "DEVELOPED_BY", "first_seen": "2025-01-05"},
            {"edge_id": "e2", "predicate": "RELEASED_BY", "first_seen": "2025-02-05"},
        ]
    )
    tables = build_temporal_tables(entities, relations, collapsed)
    assert set(tables["monthly_relation_assertions"]["month"]) == {"2025-01", "2025-02"}
    assert tables["monthly_new_entities"]["new_entity_count"].sum() == 2
    assert tables["monthly_new_typed_edges"]["new_typed_edge_count"].sum() == 2


def test_cooccurrence_community_membership_is_complete() -> None:
    graph = nx.Graph()
    graph.add_node("a", label="A", entity_type="MODEL")
    graph.add_node("b", label="B", entity_type="MODEL")
    graph.add_node("c", label="C", entity_type="ORGANIZATION")
    graph.add_node("d", label="D", entity_type="ORGANIZATION")
    graph.add_edge("a", "b", document_count=10)
    graph.add_edge("c", "d", document_count=10)
    membership, summary = detect_cooccurrence_communities(graph)
    assert set(membership["entity_id"]) == {"a", "b", "c", "d"}
    assert len(summary) == 2
    assert sorted(summary["node_count"].tolist()) == [2, 2]


def test_graph_schema_declares_three_views_and_provenance() -> None:
    schema = build_graph_schema(minimal_contract(), None)
    assert set(schema["graph_views"]) == {"assertion", "collapsed_typed", "cooccurrence"}
    assert schema["graph_views"]["assertion"]["multigraph"] is True
    assert "text_unit_id" in schema["provenance_chain"]
