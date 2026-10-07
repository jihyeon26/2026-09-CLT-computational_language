from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_STATUS = "candidate_pending_human_validation"
RELATION_VARIANTS = {
    "all_candidates": {
        "quality_tiers": [
            "low_score_candidate",
            "medium_score_candidate",
            "high_score_candidate",
            "curated_without_article_evidence",
        ],
        "description": "All structurally valid relation candidates.",
    },
    "medium_plus": {
        "quality_tiers": ["medium_score_candidate", "high_score_candidate"],
        "description": "GLiREL relation candidates with score >= 0.60.",
    },
    "high_score": {
        "quality_tiers": ["high_score_candidate"],
        "description": "GLiREL relation candidates with score >= 0.75.",
    },
}
COOCCURRENCE_THRESHOLDS = [1, 2, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare unvalidated WP3 candidate graph variants before human review."
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp3_open_models_official_20261007" / "prepared",
    )
    parser.add_argument(
        "--wp2-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp2_open_models_official_20261007",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp3_open_models_official_20261007" / "sensitivity",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts" / "stage1_wp3" / "sensitivity",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def collapse_candidate_relations(relations: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (subject_id, predicate, object_id), group in relations.groupby(
        ["subject_id", "predicate", "object_id"], sort=True, dropna=False
    ):
        scores = pd.to_numeric(group["rule_score"], errors="coerce").dropna()
        dates = pd.to_datetime(group["published_at"], errors="coerce").dropna()
        documents = sorted({str(value) for value in group["document_id"].dropna()})
        text_units = sorted({str(value) for value in group["text_unit_id"].dropna()})
        records.append(
            {
                "subject_id": subject_id,
                "predicate": predicate,
                "predicate_group": group["predicate_group"].iloc[0],
                "object_id": object_id,
                "evidence_count": int(len(group)),
                "document_count": int(len(documents)),
                "mean_score": float(scores.mean()) if not scores.empty else np.nan,
                "max_score": float(scores.max()) if not scores.empty else np.nan,
                "first_seen": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
                "last_seen": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
                "document_ids": json.dumps(documents, ensure_ascii=False),
                "text_unit_ids": json.dumps(text_units, ensure_ascii=False),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["predicate", "subject_id", "object_id"]
    ).reset_index(drop=True)


def build_relation_graph(
    nodes: pd.DataFrame, collapsed: pd.DataFrame, variant: str
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        graph_view="candidate_typed",
        variant=variant,
        validation_status=ANALYSIS_STATUS,
    )
    referenced = set(collapsed["subject_id"]) | set(collapsed["object_id"])
    for row in nodes[nodes["entity_id"].isin(referenced)].itertuples(index=False):
        graph.add_node(
            row.entity_id,
            label=scalar(row.canonical_name),
            raw_entity_type=scalar(row.raw_entity_type),
            entity_type_compact=scalar(row.entity_type_compact),
        )
    for row in collapsed.itertuples(index=False):
        graph.add_edge(
            row.subject_id,
            row.object_id,
            key=row.predicate,
            predicate=row.predicate,
            predicate_group=row.predicate_group,
            document_count=int(row.document_count),
            evidence_count=int(row.evidence_count),
            mean_score=scalar(row.mean_score),
            validation_status=ANALYSIS_STATUS,
        )
    return graph


def graph_statistics(graph: nx.Graph, assertion_count: int) -> dict[str, Any]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    components = list(nx.weakly_connected_components(graph)) if graph.is_directed() else list(nx.connected_components(graph))
    largest = max((len(component) for component in components), default=0)
    return {
        "assertion_count": int(assertion_count),
        "node_count": int(node_count),
        "collapsed_edge_count": int(edge_count),
        "component_count": int(len(components)),
        "largest_component_nodes": int(largest),
        "largest_component_share": float(largest / node_count) if node_count else 0.0,
        "density": float(nx.density(graph)) if node_count > 1 else 0.0,
    }


def top_nodes(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    variant: str,
    top_k: int,
    weight: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(), pd.Series(dtype=float)
    pagerank = pd.Series(nx.pagerank(graph, weight=weight), name="pagerank")
    weighted_degree = pd.Series(dict(graph.degree(weight=weight)), name="weighted_degree")
    lookup = nodes.set_index("entity_id")
    ranked = (
        pd.concat([pagerank, weighted_degree], axis=1)
        .sort_values("pagerank", ascending=False)
        .head(top_k)
        .reset_index(names="entity_id")
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.insert(0, "variant", variant)
    ranked["canonical_name"] = ranked["entity_id"].map(lookup["canonical_name"])
    ranked["entity_type_compact"] = ranked["entity_id"].map(
        lookup["entity_type_compact"]
    )
    ranked["validation_status"] = ANALYSIS_STATUS
    return ranked, pagerank


def build_cooccurrence_graph(edges: pd.DataFrame, threshold: int) -> nx.Graph:
    selected = edges[edges["document_count"] >= threshold]
    graph = nx.Graph(
        graph_view="cooccurrence",
        minimum_document_count=threshold,
        validation_status="descriptive_cooccurrence_not_semantic_relation",
    )
    for row in selected.itertuples(index=False):
        graph.add_edge(
            row.source_id,
            row.target_id,
            document_count=int(row.document_count),
            sentence_count=int(row.sentence_count),
        )
    return graph


def write_graph(graph: nx.Graph, path: Path) -> None:
    safe = graph.__class__()
    safe.graph.update({key: scalar(value) for key, value in graph.graph.items()})
    safe.add_nodes_from(
        (node, {key: scalar(value) for key, value in data.items()})
        for node, data in graph.nodes(data=True)
    )
    if graph.is_multigraph():
        for source, target, key, data in graph.edges(keys=True, data=True):
            safe.add_edge(
                source,
                target,
                key=key,
                **{name: scalar(value) for name, value in data.items()},
            )
    else:
        for source, target, data in graph.edges(data=True):
            safe.add_edge(
                source,
                target,
                **{name: scalar(value) for name, value in data.items()},
            )
    nx.write_graphml(safe, path)


def save_figures(
    relation_summary: pd.DataFrame,
    stability: pd.DataFrame,
    cooccurrence_summary: pd.DataFrame,
    relation_group_counts: pd.DataFrame,
    artifact_dir: Path,
) -> list[Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    created: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(relation_summary["variant"], relation_summary["collapsed_edge_count"], color="#4C78A8")
    axes[0].set_ylabel("Collapsed edges")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(relation_summary["variant"], relation_summary["node_count"], color="#F58518")
    axes[1].set_ylabel("Referenced nodes")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Candidate relation graph size by quality tier")
    fig.text(0.5, 0.01, "Pending human validation", ha="center", color="#B22222")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = artifact_dir / "relation_variant_sizes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    names = relation_summary["variant"].tolist()
    matrix = stability.pivot(index="variant_a", columns="variant_b", values="top20_jaccard")
    matrix = matrix.reindex(index=names, columns=names)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for row in range(len(names)):
        for column in range(len(names)):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.6 else "#1A1A1A",
            )
    ax.set_title("Top-20 PageRank overlap")
    fig.colorbar(image, ax=ax, label="Jaccard similarity")
    fig.tight_layout()
    path = artifact_dir / "pagerank_top20_stability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        cooccurrence_summary["minimum_document_count"],
        cooccurrence_summary["edge_count"],
        marker="o",
        label="Edges",
    )
    ax.plot(
        cooccurrence_summary["minimum_document_count"],
        cooccurrence_summary["node_count"],
        marker="o",
        label="Nodes",
    )
    ax.set_xlabel("Minimum document co-occurrence")
    ax.set_ylabel("Count")
    ax.set_title("Co-occurrence graph threshold sensitivity")
    ax.legend()
    fig.tight_layout()
    path = artifact_dir / "cooccurrence_threshold_sensitivity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    pivot = relation_group_counts.pivot(
        index="predicate_group", columns="variant", values="assertion_count"
    ).fillna(0)
    pivot = pivot.reindex(columns=relation_summary["variant"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Relation assertions")
    ax.set_xlabel("Predicate group")
    ax.set_title("Predicate-group counts across candidate variants")
    ax.tick_params(axis="x", rotation=35)
    fig.text(0.5, 0.01, "Pending human validation", ha="center", color="#B22222")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = artifact_dir / "relation_group_sensitivity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)
    return created


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    node_path = args.prepared_dir / "nodes_prepared.parquet"
    relation_path = args.prepared_dir / "typed_edges_assertions.parquet"
    cooccurrence_path = args.wp2_dir / "edges_cooccurrence.csv"
    for path in [node_path, relation_path, cooccurrence_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    nodes = pd.read_parquet(node_path)
    relations = pd.read_parquet(relation_path)
    cooccurrence_raw = pd.read_csv(cooccurrence_path, low_memory=False).rename(
        columns={
            "entity1_id": "source_id",
            "entity2_id": "target_id",
            "sentence_cooccurrence": "sentence_count",
            "document_cooccurrence": "document_count",
        }
    )
    for column in ["sentence_count", "document_count"]:
        cooccurrence_raw[column] = pd.to_numeric(
            cooccurrence_raw[column], errors="coerce"
        ).fillna(0).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = args.output_dir / "graphs"
    table_dir = args.output_dir / "tables"
    graph_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    relation_summaries: list[dict[str, Any]] = []
    top_frames: list[pd.DataFrame] = []
    pageranks: dict[str, pd.Series] = {}
    relation_groups: list[pd.DataFrame] = []
    graph_files: list[Path] = []

    for variant, specification in RELATION_VARIANTS.items():
        selected = relations[
            relations["quality_tier"].isin(specification["quality_tiers"])
        ].copy()
        collapsed = collapse_candidate_relations(selected)
        graph = build_relation_graph(nodes, collapsed, variant)
        statistics = graph_statistics(graph, len(selected))
        statistics.update(
            {
                "variant": variant,
                "description": specification["description"],
                "predicate_count": int(selected["predicate"].nunique()),
                "predicate_group_count": int(selected["predicate_group"].nunique()),
                "validation_status": ANALYSIS_STATUS,
            }
        )
        relation_summaries.append(statistics)

        top, pagerank = top_nodes(
            graph, nodes, variant, args.top_k, weight="document_count"
        )
        top_frames.append(top)
        pageranks[variant] = pagerank
        group_counts = (
            selected["predicate_group"]
            .value_counts()
            .rename_axis("predicate_group")
            .reset_index(name="assertion_count")
        )
        group_counts.insert(0, "variant", variant)
        relation_groups.append(group_counts)

        collapsed.to_parquet(table_dir / f"{variant}_collapsed_edges.parquet", index=False)
        graph_path = graph_dir / f"{variant}.graphml"
        write_graph(graph, graph_path)
        graph_files.append(graph_path)

    relation_summary = pd.DataFrame(relation_summaries)[
        [
            "variant", "description", "assertion_count", "collapsed_edge_count",
            "node_count", "predicate_count", "predicate_group_count", "component_count",
            "largest_component_nodes", "largest_component_share", "density",
            "validation_status",
        ]
    ]
    top_nodes_table = pd.concat(top_frames, ignore_index=True)
    relation_group_counts = pd.concat(relation_groups, ignore_index=True)

    stability_records: list[dict[str, Any]] = []
    for variant_a in RELATION_VARIANTS:
        for variant_b in RELATION_VARIANTS:
            top_a = set(
                top_nodes_table.loc[
                    top_nodes_table["variant"] == variant_a, "entity_id"
                ]
            )
            top_b = set(
                top_nodes_table.loc[
                    top_nodes_table["variant"] == variant_b, "entity_id"
                ]
            )
            union = top_a | top_b
            jaccard = len(top_a & top_b) / len(union) if union else 1.0
            aligned = pd.concat(
                [pageranks[variant_a], pageranks[variant_b]], axis=1
            ).fillna(0.0)
            spearman = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
            stability_records.append(
                {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "top20_jaccard": float(jaccard),
                    "pagerank_spearman": float(spearman),
                }
            )
    stability = pd.DataFrame(stability_records)

    cooccurrence_summaries: list[dict[str, Any]] = []
    cooccurrence_top_frames: list[pd.DataFrame] = []
    community_records: list[dict[str, Any]] = []
    for threshold in COOCCURRENCE_THRESHOLDS:
        graph = build_cooccurrence_graph(cooccurrence_raw, threshold)
        statistics = graph_statistics(graph, graph.number_of_edges())
        statistics.update(
            {
                "minimum_document_count": threshold,
                "edge_count": statistics.pop("collapsed_edge_count"),
                "validation_status": "descriptive_cooccurrence_not_semantic_relation",
            }
        )
        communities = nx.community.louvain_communities(
            graph, weight="document_count", seed=args.seed
        )
        statistics["community_count"] = int(len(communities))
        statistics["largest_community_nodes"] = int(
            max((len(community) for community in communities), default=0)
        )
        cooccurrence_summaries.append(statistics)

        variant = f"cooccurrence_min_docs_{threshold}"
        top, _ = top_nodes(
            graph, nodes, variant, args.top_k, weight="document_count"
        )
        cooccurrence_top_frames.append(top)
        for community_id, community in enumerate(
            sorted(communities, key=len, reverse=True), start=1
        ):
            community_records.append(
                {
                    "minimum_document_count": threshold,
                    "community_id": community_id,
                    "node_count": int(len(community)),
                }
            )

        if threshold in {2, 5}:
            graph_path = graph_dir / f"{variant}.graphml"
            write_graph(graph, graph_path)
            graph_files.append(graph_path)

    cooccurrence_summary = pd.DataFrame(cooccurrence_summaries)
    cooccurrence_top_nodes = pd.concat(cooccurrence_top_frames, ignore_index=True)
    community_summary = pd.DataFrame(community_records)

    artifact_tables = {
        "relation_variant_summary.csv": relation_summary,
        "relation_top_nodes.csv": top_nodes_table,
        "relation_variant_stability.csv": stability,
        "relation_group_counts.csv": relation_group_counts,
        "cooccurrence_threshold_summary.csv": cooccurrence_summary,
        "cooccurrence_top_nodes.csv": cooccurrence_top_nodes,
        "cooccurrence_community_sizes.csv": community_summary,
    }
    for filename, table in artifact_tables.items():
        table.to_csv(args.artifact_dir / filename, index=False, encoding="utf-8")

    figures = save_figures(
        relation_summary,
        stability,
        cooccurrence_summary,
        relation_group_counts,
        args.artifact_dir,
    )

    manifest = {
        "analysis_version": "stage1-wp3-candidate-sensitivity-v0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_status": ANALYSIS_STATUS,
        "random_seed": args.seed,
        "top_k": args.top_k,
        "relation_variants": RELATION_VARIANTS,
        "cooccurrence_thresholds": COOCCURRENCE_THRESHOLDS,
        "inputs": {
            "nodes_prepared": {"path": portable_path(node_path), "sha256": sha256_file(node_path)},
            "typed_edges_assertions": {"path": portable_path(relation_path), "sha256": sha256_file(relation_path)},
            "cooccurrence": {"path": portable_path(cooccurrence_path), "sha256": sha256_file(cooccurrence_path)},
        },
        "tracked_outputs": {
            path.name: {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in [
                *(args.artifact_dir / filename for filename in artifact_tables),
                *figures,
            ]
        },
        "generated_graphs": {
            path.stem: {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in graph_files
        },
        "interpretation_policy": {
            "human_validated": False,
            "allowed_use": "exploratory sensitivity and review prioritization",
            "prohibited_use": "final factual claims or relation-grounded RAG answers",
        },
    }
    manifest_path = args.artifact_dir / "sensitivity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = """# WP3 candidate graph 민감도 분석

이 폴더의 수치는 사람의 relation 검수 전에 생성한 탐색 결과다. 세 typed graph는 구조 검사를
통과한 candidate relation을 score tier별로 나눈 것이며 검증된 사실 그래프가 아니다.

- `all_candidates`: 모든 구조적으로 유효한 candidate와 curated relation
- `medium_plus`: GLiREL score 0.60 이상
- `high_score`: GLiREL score 0.75 이상

공동출현 graph는 semantic relation이 아니며 같은 문서에서 함께 등장한 정도만 나타낸다.
PageRank, component와 community 결과는 threshold 안정성을 비교하는 용도로만 사용한다.

최종 보고서의 사실 주장과 Stage 3 relation-grounded retrieval에는 사람 검수 결과 또는 연결된
원문 passage 확인이 필요하다.
"""
    (args.artifact_dir / "README_KO.md").write_text(readme, encoding="utf-8")

    relation_rows = []
    for row in relation_summary.itertuples(index=False):
        relation_rows.append(
            f"| {row.variant} | {row.assertion_count:,} | {row.collapsed_edge_count:,} | "
            f"{row.node_count:,} | {row.component_count:,} | {row.largest_component_share:.3f} |"
        )
    comparison_rows = []
    for variant_a, variant_b in [
        ("all_candidates", "medium_plus"),
        ("all_candidates", "high_score"),
        ("medium_plus", "high_score"),
    ]:
        row = stability[
            (stability["variant_a"] == variant_a)
            & (stability["variant_b"] == variant_b)
        ].iloc[0]
        comparison_rows.append(
            f"| {variant_a} vs {variant_b} | {row['top20_jaccard']:.3f} | "
            f"{row['pagerank_spearman']:.3f} |"
        )
    cooccurrence_rows = []
    for row in cooccurrence_summary.itertuples(index=False):
        cooccurrence_rows.append(
            f"| {row.minimum_document_count} | {row.edge_count:,} | {row.node_count:,} | "
            f"{row.component_count:,} | {row.community_count:,} | "
            f"{row.largest_component_share:.3f} |"
        )
    assessment = """# WP3 candidate graph 민감도 평가

이 평가는 사람의 relation 검수 전에 graph 결과가 score·support threshold에 얼마나 민감한지
확인한다. 수치는 탐색용이며 실제 산업 영향력이나 사실 관계를 확정하지 않는다.

## Relation graph 크기와 연결성

| Variant | Assertions | Collapsed edges | Nodes | Components | Largest component share |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(relation_rows) + """

## PageRank 안정성

| Comparison | Top-20 Jaccard | PageRank Spearman |
|---|---:|---:|
""" + "\n".join(comparison_rows) + """

`all_candidates`와 `high_score`의 top-20 중복과 전체 PageRank 순위 상관이 모두 낮다. 현재
중심성 순위는 confidence threshold 선택에 민감하므로 최종 insight로 사용할 수 없다.

## Co-occurrence threshold 민감성

| Minimum documents | Edges | Nodes | Components | Communities | Largest component share |
|---:|---:|---:|---:|---:|---:|
""" + "\n".join(cooccurrence_rows) + """

문서 1회 공동출현은 coverage가 크지만 희소 node와 작은 community가 많다. 문서 5회 기준은
구조가 안정적으로 연결되지만 coverage가 크게 줄어든다. 현재는 `minimum documents = 2`를
탐색용 기본값으로 사용하고 1과 5를 sensitivity view로 함께 보존한다.

## 검수 전 결정

- `medium_plus`를 탐색용 typed graph의 임시 기본값으로 사용한다.
- `all_candidates`와 `high_score` 결과를 함께 보고 metric 안정성을 확인한다.
- 어느 variant도 validated knowledge graph로 부르지 않는다.
- predicate group별 최종 포함 여부는 사람 검수 후 정한다.
- 최종 centrality와 graph insight는 human-validated graph에서 다시 계산한다.
"""
    (args.artifact_dir / "ASSESSMENT_KO.md").write_text(
        assessment, encoding="utf-8"
    )

    print(relation_summary.to_string(index=False))
    print(cooccurrence_summary.to_string(index=False))
    print(f"Artifacts: {args.artifact_dir.resolve()}")
    print(f"Generated graphs: {graph_dir.resolve()}")


if __name__ == "__main__":
    main()
