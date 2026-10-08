"""Part 3 of the WP3 notebook: build the final Stage 1 knowledge graph from the corrected relations.

The cells are kept in a separate module so that `build_wp3_notebook.py` stays readable. Part 3 starts with its own
setup cell and only reads files from disk, so it can be re-run without re-running Parts 1-2.
"""
from __future__ import annotations

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def part3_cells() -> list:
    return [
        markdown(
            """
# WP3 3단계 — 보정 관계로 최종 지식그래프 구축 (Stage 1 완결)

2단계까지의 결론은 **raw candidate graph는 그대로 사실 그래프로 쓸 수 없다**는 것이었다. 이후
`notebooks/stage1_wp3_relation_correction.ipynb`에서 오류 원인별 규칙으로 relation을 보정했다.

- 이름 변형·별칭 병합, 잘못된 node type 수정, 일반명사 node 표시
- `Llama Nemotron`처럼 다른 제품명 안의 family 이름 연결 해제
- WP1 lexicon 개발사와 충돌하는 `developed by` 제외, 자기 회사 제품과의 `competes with` 제외
- 같은 문장 반복을 제거한 뒤 서로 다른 근거 문장이 2개 이상인 relation만 Tier B로 분류

3단계는 이 보정 결과(Tier A + B)로 과제 요구사항의 Knowledge Graph Construction을 완성한다.

1. compact topic-specific schema 확정
2. assertion·collapsed typed·co-occurrence graph 생성
3. centrality, clustering, community, 시간 분석
4. 목적별 시각화
5. 근거 문장이 연결된 insight 후보
6. Stage 3에서 다시 쓸 graph·schema·data dictionary 저장과 재로딩 검증

3단계는 디스크의 파일만 읽으므로 1·2단계를 다시 실행하지 않고 이 셀부터 실행할 수 있다.
"""
        ),
        markdown(
            """
## 22. 3단계 실행 환경

입력은 세 곳에서 읽는다.

- `outputs/stage1_wp2_open_models_official_20261007/`: 공식 WP2 handover (수정하지 않음)
- `outputs/stage1_wp3_relation_correction_20261008/`: relation correction notebook 산출물
- `artifacts/stage1_wp3/evaluation/relation_correction/`: 사람 검수용 Tier B 표본

대용량 결과는 Git-ignored `outputs/stage1_wp3_knowledge_graph_final/`에, 제출·공유용 작은 파일은
`artifacts/stage1_wp3/final/`에 저장한다.
"""
        ),
        code(
            """
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import Image, display
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path.cwd()
if not (ROOT / "config").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_graph_rag import GraphEvidenceRetriever
from wp3_knowledge_graph.pipeline import (
    build_relation_type_summary,
    calculate_node_metrics,
    calculate_relation_type_metrics,
    collapse_relations,
    detect_cooccurrence_communities,
    graph_summary,
)

WP2_INPUT_DIR = ROOT / "outputs" / "stage1_wp2_open_models_official_20261007"
CORRECTION_DIR = ROOT / "outputs" / "stage1_wp3_relation_correction_20261008"
CORRECTED_REVIEW_PATH = (
    ROOT / "artifacts" / "stage1_wp3" / "evaluation" / "relation_correction"
    / "relation_review_corrected_seed2026.csv"
)
COMPACT_SCHEMA_PATH = ROOT / "config" / "wp3_compact_schema.json"
WP1_CORE_PATH = ROOT / "dataset" / "processed" / "open_models_core.parquet"  # 선택: 기사 제목·URL
FINAL_OUTPUT_DIR = ROOT / "outputs" / "stage1_wp3_knowledge_graph_final"
FINAL_ARTIFACT_DIR = ROOT / "artifacts" / "stage1_wp3" / "final"
FIGURE_DIR = FINAL_ARTIFACT_DIR / "figures"
for directory in [FINAL_OUTPUT_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MIN_COOCCURRENCE_DOCUMENTS = 2   # 19절 민감도 분석에서 정한 탐색용 기본값
DRAW_COOCCURRENCE_DOCUMENTS = 5  # community 그림에서만 쓰는 더 높은 기준
SEED = 42
PERIODS = {"P1 (2024-09~2025-08)": ("2024-09-01", "2025-08-31"),
           "P2 (2025-09~2026-08)": ("2025-09-01", "2026-08-31")}

for required_path in [WP2_INPUT_DIR, CORRECTION_DIR, CORRECTED_REVIEW_PATH, COMPACT_SCHEMA_PATH]:
    assert required_path.exists(), f"필수 입력이 없습니다: {required_path}"

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

def save_figure(fig, filename):
    # The WP3 pipeline module uses the non-interactive Agg backend, so saved PNGs are displayed instead.
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(path)))

def pretty(label):
    return str(label).replace("_", " ")
"""
        ),
        markdown(
            """
## 23. 보정 relation 입력과 단계별 감소

보정 notebook의 결과를 읽고, WP2 원본에서 최종 graph 후보까지 relation이 어떻게 줄었는지 확인한다.
Tier A는 WP1 lexicon의 model family → developer, Tier B는 모든 보정 검사를 통과하고 서로 다른 근거
문장이 2개 이상인 GLiREL relation이다. 같은 triple이 Tier A와 B에 모두 있으면 하나로 합치고
`sources`에 두 출처를 함께 기록했다.
"""
        ),
        code(
            """
nodes_corrected = pd.read_csv(
    CORRECTION_DIR / "nodes_corrected.csv", keep_default_na=False, na_values=[""]
)
edges_corrected = pd.read_csv(CORRECTION_DIR / "edges_relations_corrected.csv")
graph_candidates = pd.read_csv(CORRECTION_DIR / "edges_graph_candidates.csv")
evidence_corrected = pd.read_csv(CORRECTION_DIR / "relations_evidence_corrected.csv")
wp2_edges = pd.read_csv(WP2_INPUT_DIR / "edges_relations.csv")
wp2_nodes = pd.read_csv(WP2_INPUT_DIR / "nodes.csv", keep_default_na=False, na_values=[""])
wp2_cooccurrence = pd.read_csv(WP2_INPUT_DIR / "edges_cooccurrence.csv")
nodes_quarterly = pd.read_csv(WP2_INPUT_DIR / "nodes_quarterly.csv")
sentences = pd.read_csv(WP2_INPUT_DIR / "sentences.csv")
wp2_schema = json.loads((WP2_INPUT_DIR / "schema.json").read_text(encoding="utf-8"))
compact_schema = json.loads(COMPACT_SCHEMA_PATH.read_text(encoding="utf-8"))

glirel_raw = wp2_edges[wp2_edges["source"] == "glirel"]
relation_funnel = pd.DataFrame(
    [
        ("WP2 GLiREL 원본 triple", len(glirel_raw)),
        ("WP2 권장 필터: type_valid & 문장 2개 이상", int((glirel_raw["type_valid"] & (glirel_raw["n_sentences"] >= 2)).sum())),
        ("보정 후 Tier B: 모든 검사 통과 & 서로 다른 문장 2개 이상", int((edges_corrected["tier"] == "B").sum())),
        ("Tier A: WP1 lexicon model family → developer", int((edges_corrected["tier"] == "A").sum())),
        ("최종 graph 후보: Tier A+B, 동일 triple 병합", len(graph_candidates)),
    ],
    columns=["단계", "triple 수"],
)
display(relation_funnel)

exclusion_reasons = (
    edges_corrected["exclusion_reason"].dropna().str.split(";").explode()
    .value_counts().rename_axis("제외 사유").reset_index(name="triple 수")
)
display(exclusion_reasons)
display(graph_candidates["sources"].value_counts().rename_axis("출처").reset_index(name="triple 수"))
"""
        ),
        markdown(
            """
WP2 권장 필터의 1,629개 대신 1,067개 triple을 사용한다. 줄어든 이유는 점수 threshold가 아니라 구체적인
오류 원인이다. 가장 큰 제외 사유는 schema type 위반과 `model`, `company`, `AI` 같은 일반명사 node다.
Tier C(근거 문장 1개)와 X(제외)도 삭제하지 않고 보정 산출물에 사유와 함께 남아 있다.
"""
        ),
        markdown(
            """
## 24. 사람 검수 결과 반영 규칙

Tier B 표본 90개(`relation_review_corrected_seed2026.csv`)는 팀원 3명이 30개씩 검수한다. 이 셀은
검수 파일을 읽어 다음 규칙을 적용한다. 검수가 아직 없으면 graph는 `candidate` 상태로 유지된다.

- `FALSE`로 판정된 triple은 graph에서 제외한다.
- 검수 행이 90개 이상이고 precision이 0.80 이상이면 graph 상태를 `human_validated_sample`로 바꾼다.
- `AMBIGUOUS`는 precision 계산에서 제외하고 개수만 보고한다.

AI가 작성한 판정 초안은 이 계산에 사용하지 않는다.
"""
        ),
        code(
            """
TRIPLE = ["head_id", "relation", "tail_id"]
review = pd.read_csv(CORRECTED_REVIEW_PATH)
review_labels = review["review_correct"].astype("string").str.strip().str.upper()
review_counts = review_labels.value_counts().reindex(["TRUE", "FALSE", "AMBIGUOUS"], fill_value=0)
decided = int(review_counts["TRUE"] + review_counts["FALSE"])
review_precision = review_counts["TRUE"] / decided if decided else np.nan
GRAPH_STATUS = (
    "human_validated_sample"
    if review_labels.notna().sum() >= 90 and review_precision >= 0.80
    else "candidate_pending_human_review"
)

review_keys = review.assign(review_label=review_labels).rename(columns={"predicate": "relation"})[
    TRIPLE + ["review_label"]
]
final_edges = graph_candidates.merge(review_keys, on=TRIPLE, how="left", validate="one_to_one")
label = final_edges["review_label"].astype(object).fillna("")
final_edges["review_status"] = np.select(
    [label.eq("TRUE"), label.eq("FALSE"), label.eq("AMBIGUOUS"), final_edges["tier"].eq("A")],
    ["reviewed_true", "reviewed_false", "reviewed_ambiguous", "curated_lexicon"],
    default="unreviewed",
)
final_edges = final_edges[final_edges["review_status"] != "reviewed_false"].reset_index(drop=True)

display(pd.DataFrame(
    [
        ("검수 표본 행 수", len(review)),
        ("검수 완료 행 수", int(review_labels.notna().sum())),
        ("TRUE / FALSE / AMBIGUOUS", " / ".join(str(int(v)) for v in review_counts)),
        ("Tier B 표본 precision", "검수 전" if np.isnan(review_precision) else f"{review_precision:.3f}"),
        ("FALSE로 제외한 triple", len(graph_candidates) - len(final_edges)),
        ("최종 graph triple", len(final_edges)),
        ("graph 상태", GRAPH_STATUS),
    ],
    columns=["항목", "값"],
))
"""
        ),
        markdown(
            """
## 25. Compact topic-specific schema 확정

과제는 graph를 만들기 전에 compact schema를 정의하도록 요구한다. WP2의 20개 entity type과 36개
relation type은 추출 recall을 위한 넓은 schema이므로, 분석에는 2단계에서 정한 상위 그룹을 쓴다.

- **node**: 9개 compact entity type. 원래 type은 `entity_type`에 그대로 남긴다.
- **edge**: 실제 graph에 남은 raw relation type과 이를 묶는 8개 predicate group
- `competes with`는 대칭 관계라 방향이 없다. 나머지는 head → tail 방향이 의미를 가진다
  (예: `Llama → developed by → Meta`).
- 날짜는 **기사 발행일**이다. 모델 출시일이나 사건 발생일이 아니다.
"""
        ),
        code(
            """
ENTITY_GROUP = compact_schema["entity_type_mapping"]
PREDICATE_GROUP = compact_schema["predicate_group_mapping"]
SYMMETRIC_PREDICATES = {"competes with"}
COMPACT_TYPE_DEFINITIONS = {
    "MODEL": "AI model or open model family (WP1 lexicon families and other models)",
    "ORGANIZATION": "company, lab, institution or agency",
    "PERSON": "named person",
    "INFRASTRUCTURE": "hardware, platform or product used to build, host or run models",
    "TECHNICAL_COMPONENT": "technology, tool or model architecture",
    "MODEL_ATTRIBUTE": "capability, openness concept or model specification",
    "EVALUATION_RESOURCE": "benchmark or dataset",
    "GOVERNANCE": "license, standard or law",
    "LOCATION_OR_GROUP": "country, city, region or nationality group",
}

nodes_kg = nodes_corrected.rename(
    columns={"entity": "canonical_name", "mentions": "mention_count", "n_docs": "document_count"}
)
nodes_kg["mention_count"] = nodes_kg["mention_count"].fillna(0).astype(int)
nodes_kg["document_count"] = nodes_kg["document_count"].fillna(0).astype(int)
nodes_kg["entity_type_compact"] = nodes_kg["entity_type"].map(ENTITY_GROUP)
nodes_kg["is_open_model_family"] = nodes_kg["canonical_id"].notna() & nodes_kg["entity_type"].eq("FOUNDATION_MODEL")
final_edges["predicate_group"] = final_edges["relation"].map(PREDICATE_GROUP)
final_edges["directed"] = ~final_edges["relation"].isin(SYMMETRIC_PREDICATES)
assert final_edges["predicate_group"].notna().all()

wp2_relation_types = wp2_schema["edges"]["relation"]["types"]
endpoint_types = final_edges.assign(
    head_compact=final_edges["head_type"].map(ENTITY_GROUP),
    tail_compact=final_edges["tail_type"].map(ENTITY_GROUP),
)
relation_schema_table = (
    endpoint_types.groupby(["predicate_group", "relation"])
    .agg(triples=("head_id", "size"),
         head_types=("head_compact", lambda x: ", ".join(sorted(set(x)))),
         tail_types=("tail_compact", lambda x: ", ".join(sorted(set(x)))),
         directed=("directed", "first"))
    .reset_index()
    .sort_values(["predicate_group", "triples"], ascending=[True, False])
)
display(relation_schema_table)

graph_schema = {
    "schema_version": "stage1-wp3-open-models-kg-v1.0",
    "status": GRAPH_STATUS,
    "track": "Open Models",
    "node_key": "entity_id",
    "node_attributes": ["canonical_name", "entity_type", "entity_type_compact", "mention_count",
                        "document_count", "first_seen", "last_seen", "is_open_model_family", "aliases"],
    "entity_types": {
        compact: {"definition": COMPACT_TYPE_DEFINITIONS[compact],
                  "wp2_entity_types": sorted(t for t, c in ENTITY_GROUP.items() if c == compact)}
        for compact in COMPACT_TYPE_DEFINITIONS
    },
    "relation_types": {
        row.relation: {
            "group": row.predicate_group,
            "directed": bool(row.directed),
            "observed_head_types": row.head_types.split(", "),
            "observed_tail_types": row.tail_types.split(", "),
            "definition": wp2_relation_types[row.relation]["definition"],
        }
        for row in relation_schema_table.itertuples(index=False)
    },
    "edge_quality_tiers": {
        "A": "curated WP1 lexicon model family -> developer (no article sentence)",
        "B": "GLiREL relation passing all WP3 correction checks with >= 2 distinct evidence sentences",
    },
    "review_status_values": ["curated_lexicon", "unreviewed", "reviewed_true", "reviewed_ambiguous"],
    "graph_views": {
        "assertion": "directed multigraph, one edge per evidence sentence (edge key = relation_id)",
        "typed": "directed multigraph, one edge per unique (head, relation, tail); competes with is symmetric",
        "cooccurrence": f"undirected weighted graph of same-sentence co-mentions in >= {MIN_COOCCURRENCE_DOCUMENTS} documents",
    },
    "edge_weight": "document_count = number of distinct articles supporting the edge",
    "date_semantics": "published_at / first_seen / last_seen are article publication dates, not release or event dates",
    "provenance_chain": ["entity_id", "relation_id", "text_unit_id (doc_id:sentence_id)", "document_id", "published_at"],
}
compact_counts = (
    nodes_kg[nodes_kg["entity_id"].isin(set(final_edges["head_id"]) | set(final_edges["tail_id"]))]
    ["entity_type_compact"].value_counts().rename_axis("compact entity type").reset_index(name="typed graph node 수")
)
display(compact_counts)
"""
        ),
        markdown(
            """
## 26. 세 가지 graph view 생성

과제의 "nodes represent entities and edges represent co-mentions or extracted relations"를 세 view로
나누어 구현한다. typed relation과 co-occurrence는 의미가 다르므로 같은 graph에 섞지 않는다.

| view | NetworkX 구조 | edge 단위 | 용도 |
|---|---|---|---|
| assertion | `MultiDiGraph` | 근거 문장 1개 | provenance, Stage 3 근거 passage 회수 |
| typed | `MultiDiGraph` | unique triple 1개 | centrality, relation별 분석, 발표용 graph |
| co-occurrence | `Graph` | entity 쌍 1개 | community, clustering, 담론 구조 |

Co-occurrence는 WP2의 `edges_cooccurrence.csv`를 보정 node ID로 다시 매핑하고, 일반명사 node와
문서 2개 미만의 쌍을 제외한다.
"""
        ),
        code(
            """
def text_unit(doc_ids, sentence_ids):
    return [f"{d}:{int(s)}" for d, s in zip(doc_ids, sentence_ids)]

# 1) Assertion table: GLiREL evidence of the graph triples + curated lexicon rows
ok_evidence = evidence_corrected[evidence_corrected["evidence_status"] == "ok"]
glirel_keys = final_edges.loc[final_edges["sources"].str.contains("glirel"), TRIPLE]
glirel_assertions = ok_evidence.merge(glirel_keys, on=TRIPLE, how="inner")
glirel_assertions = pd.DataFrame({
    "subject_id": glirel_assertions["head_id"],
    "predicate": glirel_assertions["relation"],
    "object_id": glirel_assertions["tail_id"],
    "document_id": glirel_assertions["doc_id"],
    "text_unit_id": text_unit(glirel_assertions["doc_id"], glirel_assertions["sentence_id"]),
    "evidence_text": glirel_assertions["sentence_text"],
    "published_at": glirel_assertions["date"],
    "extraction_method": "glirel",
    "rule_score": glirel_assertions["score"],
    "assertion_status": "asserted",
})
lexicon_rows = final_edges[final_edges["sources"].str.contains("wp1_lexicon")]
lexicon_assertions = pd.DataFrame({
    "subject_id": lexicon_rows["head_id"], "predicate": lexicon_rows["relation"],
    "object_id": lexicon_rows["tail_id"], "document_id": pd.NA, "text_unit_id": pd.NA,
    "evidence_text": pd.NA, "published_at": pd.NA, "extraction_method": "wp1_lexicon",
    "rule_score": np.nan, "assertion_status": "asserted",
})
assertions = pd.concat([glirel_assertions, lexicon_assertions], ignore_index=True)
assertions.insert(0, "relation_id", [
    "rel_" + hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:16]
    for key in assertions[["subject_id", "predicate", "object_id", "text_unit_id", "extraction_method"]]
    .itertuples(index=False, name=None)
])
assertions = assertions.drop_duplicates("relation_id").reset_index(drop=True)
assert set(map(tuple, glirel_keys.values)) <= set(
    zip(glirel_assertions["subject_id"], glirel_assertions["predicate"], glirel_assertions["object_id"])
), "every GLiREL graph triple needs at least one evidence sentence"

# 2) Collapsed typed edges: evidence aggregates from the WP3 pipeline + correction attributes
collapsed = collapse_relations(assertions, documents=None)
edge_attributes = final_edges.rename(columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id"})[
    ["subject_id", "predicate", "object_id", "tier", "sources", "review_status", "predicate_group",
     "directed", "n_distinct_sentences", "mean_score"]
]
collapsed = collapsed.merge(edge_attributes, on=["subject_id", "predicate", "object_id"], how="left", validate="one_to_one")
collapsed["weight"] = collapsed["document_count"].clip(lower=1)
assert len(collapsed) == len(final_edges)

# 3) Co-occurrence edges remapped to the corrected node ids
id_map = (nodes_corrected.assign(original_id=nodes_corrected["merged_ids"].str.split(" | ", regex=False))
          .explode("original_id").set_index("original_id")["entity_id"])
generic_ids = set(nodes_corrected.loc[nodes_corrected["is_generic"], "entity_id"])
co = wp2_cooccurrence.assign(
    a=wp2_cooccurrence["entity1_id"].map(id_map), b=wp2_cooccurrence["entity2_id"].map(id_map)
)
co = co[(co["a"] != co["b"]) & ~co["a"].isin(generic_ids) & ~co["b"].isin(generic_ids)].copy()
ordered = np.sort(co[["a", "b"]].to_numpy(dtype=str), axis=1)
co["source_id"], co["target_id"] = ordered[:, 0], ordered[:, 1]
cooccurrence_all = (
    co.groupby(["source_id", "target_id"])
    .agg(sentence_count=("sentence_cooccurrence", "sum"), document_count=("document_cooccurrence", "sum"),
         npmi=("npmi", "max"), first_seen=("first_seen", "min"), last_seen=("last_seen", "max"),
         merged_pairs=("entity1_id", "size"))
    .reset_index()
)
cooccurrence_edges = cooccurrence_all[cooccurrence_all["document_count"] >= MIN_COOCCURRENCE_DOCUMENTS].reset_index(drop=True)

# 4) NetworkX graphs
node_index = nodes_kg.set_index("entity_id")
def node_attributes(entity_id):
    row = node_index.loc[entity_id]
    return {"label": row["canonical_name"], "entity_type": row["entity_type"],
            "entity_type_compact": row["entity_type_compact"], "mention_count": int(row["mention_count"]),
            "document_count": int(row["document_count"]), "first_seen": row["first_seen"] or "",
            "last_seen": row["last_seen"] or "", "is_open_model_family": bool(row["is_open_model_family"])}

def scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if pd.notna(value) else -1.0
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)

assertion_graph = nx.MultiDiGraph(graph_view="assertion", schema_version=graph_schema["schema_version"])
typed_graph = nx.MultiDiGraph(graph_view="typed", schema_version=graph_schema["schema_version"])
cooccurrence_graph = nx.Graph(graph_view="cooccurrence", schema_version=graph_schema["schema_version"])
for graph, endpoints in [
    (assertion_graph, set(assertions["subject_id"]) | set(assertions["object_id"])),
    (typed_graph, set(collapsed["subject_id"]) | set(collapsed["object_id"])),
    (cooccurrence_graph, set(cooccurrence_edges["source_id"]) | set(cooccurrence_edges["target_id"])),
]:
    graph.add_nodes_from((entity_id, node_attributes(entity_id)) for entity_id in sorted(endpoints))

for row in assertions.itertuples(index=False):
    assertion_graph.add_edge(row.subject_id, row.object_id, key=row.relation_id, predicate=row.predicate,
                             document_id=scalar(row.document_id), text_unit_id=scalar(row.text_unit_id),
                             published_at=scalar(row.published_at), extraction_method=row.extraction_method,
                             rule_score=scalar(row.rule_score))
typed_columns = ["predicate", "predicate_group", "directed", "tier", "sources", "review_status", "evidence_count",
                 "document_count", "n_distinct_sentences", "weight", "first_seen", "last_seen", "mean_rule_score",
                 "text_unit_ids"]
for row in collapsed.itertuples(index=False):
    typed_graph.add_edge(row.subject_id, row.object_id, key=row.predicate,
                         **{column: scalar(getattr(row, column)) for column in typed_columns})
for row in cooccurrence_edges.itertuples(index=False):
    cooccurrence_graph.add_edge(row.source_id, row.target_id, sentence_count=int(row.sentence_count),
                                document_count=int(row.document_count), npmi=float(row.npmi),
                                first_seen=scalar(row.first_seen), last_seen=scalar(row.last_seen))

display(pd.DataFrame(
    [
        ("assertion", assertion_graph.number_of_nodes(), assertion_graph.number_of_edges(), "근거 문장 + lexicon"),
        ("typed", typed_graph.number_of_nodes(), typed_graph.number_of_edges(), "unique triple"),
        ("co-occurrence", cooccurrence_graph.number_of_nodes(), cooccurrence_graph.number_of_edges(),
         f"문서 {MIN_COOCCURRENCE_DOCUMENTS}개 이상 공동출현"),
    ],
    columns=["view", "node", "edge", "edge 단위"],
))
print(f"co-occurrence: 일반명사 제외 후 {len(cooccurrence_all):,}쌍 중 문서 {MIN_COOCCURRENCE_DOCUMENTS}개 이상 "
      f"{len(cooccurrence_edges):,}쌍 | 병합으로 합쳐진 쌍 {int((cooccurrence_all['merged_pairs'] > 1).sum()):,}개")
"""
        ),
        markdown(
            """
보정으로 합쳐진 node 쌍의 co-occurrence 문서 수는 원래 쌍의 문서 수를 더한 값이라, 같은 기사가 두
번 세어질 수 있는 상한값이다. 해당 쌍의 수를 위에 함께 출력했다.
"""
        ),
        markdown(
            """
## 27. 기본 구조 통계

node·edge 수, density, component, relation type별 edge 수를 먼저 본다. density가 매우 낮고 component가
많으면 graph metric을 전체 산업 구조로 일반화하기 어렵다는 뜻이므로, 이후 해석의 범위를 정하는 기준이 된다.
"""
        ),
        code(
            """
structure = pd.DataFrame(
    {name: graph_summary(graph) for name, graph in
     [("assertion", assertion_graph), ("typed", typed_graph), ("co-occurrence", cooccurrence_graph)]}
).T
typed_undirected = nx.Graph(typed_graph.to_undirected())
largest_component = max(nx.connected_components(typed_undirected), key=len)
structure.loc["typed", "largest_component_nodes"] = len(largest_component)
cooc_largest = max(nx.connected_components(cooccurrence_graph), key=len)
structure.loc["co-occurrence", "largest_component_nodes"] = len(cooc_largest)
display(structure)

relation_summary = build_relation_type_summary(collapsed)
relation_summary.insert(1, "predicate_group", relation_summary["predicate"].map(PREDICATE_GROUP))
display(relation_summary)

GROUP_COLORS = {
    "PROVENANCE": "#4C78A8", "EVALUATION": "#E45756", "ORGANIZATION_RELATION": "#F58518",
    "DEPLOYMENT": "#54A24B", "GOVERNANCE_RELATION": "#B279A2", "LINEAGE": "#72B7B2",
    "ATTRIBUTE": "#9D755D", "TRAINING": "#EECA3B",
}
COMPACT_COLORS = {
    "MODEL": "#4C78A8", "ORGANIZATION": "#F58518", "PERSON": "#9D755D", "INFRASTRUCTURE": "#72B7B2",
    "TECHNICAL_COMPONENT": "#54A24B", "MODEL_ATTRIBUTE": "#B279A2", "EVALUATION_RESOURCE": "#EECA3B",
    "GOVERNANCE": "#E45756", "LOCATION_OR_GROUP": "#BAB0AC",
}
plot_data = relation_summary.sort_values("unique_edges")
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(plot_data["predicate"], plot_data["unique_edges"],
        color=plot_data["predicate_group"].map(GROUP_COLORS))
for y, value in enumerate(plot_data["unique_edges"]):
    ax.text(value + 3, y, str(value), va="center", fontsize=8)
ax.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COLORS.items() if g in set(plot_data["predicate_group"])],
          title="Predicate group", frameon=False, loc="lower right", fontsize=8)
ax.set(title="Typed graph: unique edges by relation type", xlabel="Unique edges (triples)", ylabel="")
save_figure(fig, "01_relation_type_counts.png")
"""
        ),
        markdown(
            """
## 28. Centrality 분석

**왜**: 어떤 entity가 관계망의 중심에 있는지 보기 위해서다. 세 가지 지표를 함께 본다.

- weighted degree: 연결된 relation의 근거 기사 수 합
- PageRank: 중요한 node와 연결될수록 커지는 점수 (edge weight = 근거 기사 수)
- betweenness: 서로 다른 영역을 잇는 다리 역할 (샘플 기반 근사)

typed graph의 PageRank와 betweenness는 방향을 무시한 projection에서 계산한다. 방향이 의미를 가지는
relation은 아래에서 relation별 in/out degree로 따로 해석한다. 예를 들어 `developed by`는 모델 → 조직
방향이므로 조직의 in-degree가 "graph에서 확인된 개발 산출물 수"가 된다.
"""
        ),
        code(
            """
typed_metrics = calculate_node_metrics(typed_graph, "typed", "weight", sample_size=1000, seed=SEED)
cooc_metrics = calculate_node_metrics(cooccurrence_graph, "cooccurrence", "document_count", sample_size=500, seed=SEED)
for metrics in (typed_metrics, cooc_metrics):
    metrics["entity_type_compact"] = metrics["entity_id"].map(node_index["entity_type_compact"])

metric_columns = ["canonical_name", "entity_type_compact", "degree", "in_degree", "out_degree",
                  "weighted_degree", "pagerank", "betweenness"]
print("Typed graph - PageRank 상위 15")
display(typed_metrics.head(15)[metric_columns].round(4))
print("Co-occurrence graph - PageRank 상위 15")
display(cooc_metrics.head(15)[[c for c in metric_columns if c in cooc_metrics]].round(4))

relation_metrics = calculate_relation_type_metrics(nodes_kg, collapsed, sample_size=300, seed=SEED)
relation_views = {
    "developed by": ("in_degree", "개발 산출물이 가장 많이 연결된 조직"),
    "competes with": ("degree", "경쟁 관계가 가장 많은 entity"),
    "works for": ("in_degree", "소속 인물이 가장 많이 연결된 조직"),
    "outperforms": ("out_degree", "다른 모델보다 낫다고 서술된 횟수가 많은 모델"),
}
for predicate, (column, description) in relation_views.items():
    view = relation_metrics[relation_metrics["predicate"] == predicate]
    print(f"[{predicate}] {description} ({column} 기준)")
    display(view.sort_values([column, "weighted_degree"], ascending=False)
                .head(8)[["canonical_name", "entity_type", column, "weighted_degree"]])
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, metrics, title in [(axes[0], typed_metrics, "Typed relation graph"),
                           (axes[1], cooc_metrics, "Co-occurrence graph")]:
    top = metrics.head(15).iloc[::-1]
    ax.barh(top["canonical_name"].map(pretty), top["pagerank"], color=top["entity_type_compact"].map(COMPACT_COLORS))
    ax.set(title=f"Top 15 PageRank - {title}", xlabel="PageRank (weight = supporting articles)")
present = set(typed_metrics.head(15)["entity_type_compact"]) | set(cooc_metrics.head(15)["entity_type_compact"])
fig.legend(handles=[Patch(color=COMPACT_COLORS[t], label=t) for t in COMPACT_COLORS if t in present],
           loc="lower center", ncol=len(present), frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save_figure(fig, "02_pagerank_typed_vs_cooccurrence.png")

overlap = set(typed_metrics.head(20)["entity_id"]) & set(cooc_metrics.head(20)["entity_id"])
print(f"두 view의 PageRank 상위 20 중 겹치는 entity: {len(overlap)}개 -> "
      + ", ".join(sorted(node_index.loc[list(overlap), "canonical_name"])))
"""
        ),
        markdown(
            """
**해석 (팀 검토 전 초안)**

- typed graph의 PageRank 상위 7개는 NVIDIA, OpenAI, Google, DeepSeek, Meta, Anthropic, Microsoft로 모두
  조직이다. 모델 family 중에서는 Llama, Nemotron, Gemma, Qwen, Kimi가 상위 15위 안에 든다.
- NVIDIA가 1위인 이유는 방향을 보면 설명된다. `developed by` in-degree가 83으로 가장 크다(Nemotron,
  DGX Spark, NemoClaw, Blackwell architecture 등). 하드웨어 공급자이면서 모델·소프트웨어 개발사로 가장 많은
  산출물과 연결되어 있다.
- OpenAI는 weighted degree(880)와 betweenness(0.30)가 가장 크다. 주로 `competes with`를 통해 여러 조직·모델을
  잇는 위치에 있다.
- co-occurrence graph에서는 `United States`, `China`, `open source model`처럼 맥락을 나타내는 entity가 상위에
  들어온다. 같은 문장에 함께 쓰인 빈도라서 담론의 배경(미·중 경쟁, open vs closed 논쟁)을 반영한다.
- 두 view의 PageRank 상위 20 중 14개가 겹쳐, **핵심 행위자 집합**은 view에 덜 민감하다. 그러나 18절에서 본
  것처럼 **순위**는 threshold에 따라 바뀌므로 산업 영향력 순위로 표현하지 않는다.
"""
        ),
        markdown(
            """
## 29. Clustering과 community

**왜**: 함께 언급되는 entity 묶음은 담론의 하위 주제(예: 중국 오픈 모델, NVIDIA 하드웨어 생태계)를
보여 준다. clustering coefficient는 triangle이 많은 co-occurrence graph에서 의미가 있고, 희소한 typed
graph에서는 낮게 나오는 것이 정상이다. community는 문서 수를 weight로 한 greedy modularity로 찾는다.
"""
        ),
        code(
            """
clustering_table = pd.DataFrame(
    [
        ("typed (undirected projection)", nx.average_clustering(typed_undirected), np.nan,
         nx.transitivity(typed_undirected)),
        ("co-occurrence", nx.average_clustering(cooccurrence_graph),
         nx.average_clustering(cooccurrence_graph, weight="document_count"), nx.transitivity(cooccurrence_graph)),
    ],
    columns=["view", "average clustering", "weighted average clustering", "transitivity"],
).round(4)
display(clustering_table)

community_members, community_summary = detect_cooccurrence_communities(cooccurrence_graph)
community_sets = [set(group["entity_id"]) for _, group in community_members.groupby("community_id")]
modularity = nx.community.modularity(cooccurrence_graph, community_sets, weight="document_count")
community_summary["top_entities"] = community_summary["top_entities"].map(lambda v: ", ".join(json.loads(v)[:8]))
community_summary["share_of_nodes"] = (community_summary["node_count"] / cooccurrence_graph.number_of_nodes()).round(3)
print(f"community {len(community_summary)}개 | modularity {modularity:.3f}")
display(community_summary.head(10))
"""
        ),
        code(
            """
TOP_COMMUNITIES = 6
NODES_PER_COMMUNITY = 9
top_members = community_members[
    (community_members["community_id"] <= TOP_COMMUNITIES)
    & (community_members["rank_in_community"] <= NODES_PER_COMMUNITY)
]
draw = cooccurrence_graph.subgraph(top_members["entity_id"]).copy()
draw.remove_edges_from([(a, b) for a, b, d in draw.edges(data=True)
                        if d["document_count"] < DRAW_COOCCURRENCE_DOCUMENTS])
draw.remove_nodes_from(list(nx.isolates(draw)))
community_of = top_members.set_index("entity_id")["community_id"]
palette = plt.get_cmap("tab10")
positions = nx.spring_layout(draw, seed=SEED, weight="document_count", k=2.4 / np.sqrt(max(len(draw), 1)), iterations=500)
fig, ax = plt.subplots(figsize=(13, 10))
widths = [0.3 + 2.5 * d["document_count"] / max(nx.get_edge_attributes(draw, "document_count").values())
          for _, _, d in draw.edges(data=True)]
nx.draw_networkx_edges(draw, positions, width=widths, alpha=0.25, edge_color="#6B7280", ax=ax)
nx.draw_networkx_nodes(draw, positions, ax=ax, linewidths=0.6, edgecolors="white",
                       node_color=[palette(community_of[n] - 1) for n in draw],
                       node_size=[120 + 0.6 * draw.nodes[n]["document_count"] for n in draw])
nx.draw_networkx_labels(draw, positions, labels={n: pretty(draw.nodes[n]["label"]) for n in draw}, font_size=7, ax=ax)
legend_labels = community_summary.set_index("community_id")["top_entities"]
ax.legend(handles=[Patch(color=palette(c - 1), label=f"C{c}: " + ", ".join(legend_labels[c].split(", ")[:3]))
                   for c in range(1, TOP_COMMUNITIES + 1)], loc="upper left", frameon=False, fontsize=8)
ax.set_title(f"Co-occurrence communities (top {NODES_PER_COMMUNITY} entities of the {TOP_COMMUNITIES} largest; "
             f"edges with >= {DRAW_COOCCURRENCE_DOCUMENTS} documents)")
ax.axis("off")
save_figure(fig, "03_cooccurrence_communities.png")
"""
        ),
        markdown(
            """
**해석 (팀 검토 전 초안)**

- 평균 clustering은 co-occurrence graph가 0.54, typed graph가 0.10이다. 공동출현은 한 기사 안의 여러 entity가
  서로 묶여 triangle이 많고, typed graph는 모델 → 개발사처럼 별 모양 구조라 낮게 나오는 것이 정상이다.
- greedy modularity로 192개 community를 찾았고 modularity는 0.236이다. 값이 낮은 것은 주요 조직·모델이
  대부분의 기사에서 함께 언급되어 경계가 뚜렷하지 않기 때문이다.
- 그래도 상위 3개 community(전체 node의 약 80%)는 해석 가능한 담론 묶음으로 보인다.
  - **C1 (34%)** OpenAI, Google, open source model, Llama, Anthropic: 미국 frontier lab과 open vs closed 모델 논의
  - **C2 (28%)** United States, DeepSeek, China, NVIDIA, Microsoft: DeepSeek 이후 미·중 경쟁, 칩, 수출 통제 담론
  - **C3 (18%)** vLLM, Nemotron, Google Cloud, IBM, Dell: 기업용 배포·추론 인프라
  - 작은 community는 특정 기사 주제다: LLM gateway 운영(C4), attention 구조(C5: MLA, GQA), Meta 앱(C6).
- community 이름은 상위 entity를 보고 붙인 해석이므로 팀 검토 대상이다.
"""
        ),
        markdown(
            """
## 30. 시간 분석

모든 날짜는 기사 발행일이므로 "그 시점에 데이터셋에서 관찰된 관계"로 해석한다.

1. 분기별 relation 근거 문장 수 (predicate group별)
2. 분기별 새로 처음 관찰된 typed triple 수
3. 주요 open model family의 분기별 언급 기사 비율 (WP2 `nodes_quarterly`)
4. 2026년에 처음 관찰된 relation (emerging relationship 후보)
5. 전반기(P1)와 후반기(P2) 사이 relation 근거 비중이 가장 크게 늘어난 entity
"""
        ),
        code(
            """
dated = assertions[assertions["extraction_method"] == "glirel"].copy()
dated["quarter"] = pd.to_datetime(dated["published_at"]).dt.to_period("Q").astype(str)
dated["predicate_group"] = dated["predicate"].map(PREDICATE_GROUP)
quarterly_assertions = dated.pivot_table(index="quarter", columns="predicate_group", values="relation_id",
                                         aggfunc="count", fill_value=0)

new_edges = collapsed[collapsed["first_seen"].notna()].copy()
new_edges["quarter"] = pd.to_datetime(new_edges["first_seen"]).dt.to_period("Q").astype(str)
quarterly_new_edges = new_edges.pivot_table(index="quarter", columns="predicate_group", values="predicate",
                                            aggfunc="count", fill_value=0)

mentions = pd.read_csv(WP2_INPUT_DIR / "entity_mentions.csv", usecols=["doc_id", "quarter"])
documents_per_quarter = mentions.drop_duplicates("doc_id")["quarter"].value_counts().sort_index()
families = nodes_kg[nodes_kg["is_open_model_family"]].set_index("entity_id")["canonical_name"]
family_quarterly = nodes_quarterly.assign(entity_id=nodes_quarterly["entity_id"].map(id_map))
family_quarterly = (family_quarterly[family_quarterly["entity_id"].isin(families.index)]
                    .groupby(["entity_id", "quarter"])["n_docs"].sum().unstack(fill_value=0))
family_share = family_quarterly.div(documents_per_quarter.reindex(family_quarterly.columns), axis=1)
top_families = family_quarterly.sum(axis=1).nlargest(10).index
family_share = family_share.loc[top_families].rename(index=families)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, table, title in [(axes[0], quarterly_assertions, "Relation evidence sentences per quarter"),
                         (axes[1], quarterly_new_edges, "First-seen typed triples per quarter")]:
    table.plot(ax=ax, kind="bar", stacked=True, width=0.8,
               color=[GROUP_COLORS.get(c, "#BAB0AC") for c in table.columns], legend=False)
    ax.set(title=title, xlabel="Quarter (article publication date)", ylabel="Count")
handles = [Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_COLORS if g in set(quarterly_assertions.columns)]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save_figure(fig, "04_quarterly_relations.png")

fig, ax = plt.subplots(figsize=(11, 5))
image = ax.imshow(family_share.values, aspect="auto", cmap="Blues")
ax.set_xticks(range(family_share.shape[1]), family_share.columns, rotation=45, ha="right")
ax.set_yticks(range(family_share.shape[0]), family_share.index)
for (i, j), value in np.ndenumerate(family_share.values):
    ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7,
            color="white" if value > family_share.values.max() * 0.6 else "black")
fig.colorbar(image, ax=ax, label="Share of articles mentioning the family")
ax.set_title("Top 10 open model families: share of Open Models articles per quarter")
save_figure(fig, "05_family_quarterly_share.png")
display(documents_per_quarter.rename("기사 수").to_frame().T)
"""
        ),
        code(
            """
labels = node_index["canonical_name"]
emerging = collapsed[(collapsed["first_seen"].fillna("") >= "2026-01-01") & (collapsed["document_count"] >= 2)].copy()
emerging["edge"] = emerging["subject_id"].map(labels) + " -> " + emerging["predicate"] + " -> " + emerging["object_id"].map(labels)
print(f"2026년에 처음 관찰되고 2개 이상 기사로 뒷받침되는 triple: {len(emerging)}개")
display(emerging.sort_values(["document_count", "first_seen"], ascending=[False, True])
        .head(15)[["edge", "predicate_group", "document_count", "first_seen", "last_seen"]])

def period_share(start, end):
    rows = dated[(dated["published_at"] >= start) & (dated["published_at"] <= end)]
    long = pd.concat([rows[["subject_id", "document_id"]].rename(columns={"subject_id": "entity_id"}),
                      rows[["object_id", "document_id"]].rename(columns={"object_id": "entity_id"})])
    per_entity = long.drop_duplicates().groupby("entity_id")["document_id"].nunique()
    return per_entity / rows["document_id"].nunique()

shares = pd.DataFrame({name: period_share(*span) for name, span in PERIODS.items()}).fillna(0)
p1, p2 = shares.columns
shares["change_pp"] = (shares[p2] - shares[p1]) * 100
shares["canonical_name"] = shares.index.map(labels)
shares["entity_type_compact"] = shares.index.map(node_index["entity_type_compact"])
print("relation 근거 기사 비중 변화 (P2 - P1, %p)")
display(pd.concat([shares.nlargest(10, "change_pp"), shares.nsmallest(5, "change_pp")])
        [["canonical_name", "entity_type_compact", p1, p2, "change_pp"]].round(3))
"""
        ),
        markdown(
            """
**해석 (팀 검토 전 초안)**

- **2025Q1 집중**: relation 근거 문장이 약 1,700개로 다른 분기의 2~4배이고, 기사 수도 349개로 가장 많다
  (DeepSeek-R1 공개 직후). 처음 관찰된 triple도 이 분기에 가장 많다. 따라서 분기 비교는 기사 수로 정규화한
  지표(family 비율, 근거 기사 비중)를 우선한다.
- **family 담론의 이동**: Llama를 언급한 기사 비율은 2024Q3 68%에서 2026Q3 19%로 계속 낮아졌다. 반면 Qwen은
  5% → 30%대, Kimi는 0% → 40%(2026Q3), GLM은 0% → 27%로 올라갔다. Nemotron은 2026Q1에 25%로 가장 높았고,
  gpt-oss는 공개 직후인 2025Q3에 16%까지 올랐다가 줄었다.
- **relation 근거 비중 변화 (P1 → P2)**: Anthropic +14.9%p, Nemotron +9.9%p, Kimi +7.4%p, NVIDIA +6.7%p,
  GLM +5.0%p가 늘었고, DeepSeek −27.0%p, Meta −13.0%p, Llama −9.8%p가 줄었다. DeepSeek 감소는 2025Q1 집중 보도의
  반동으로 볼 수 있다.
- **2026년 신규 relation**: 125개 중 상위는 NemoClaw·OpenShell → NVIDIA(agent 소프트웨어), Muse Spark → Meta,
  GPT-5.5 → OpenAI, Claude Mythos 5·Fable 5 → Anthropic이다. 다만 `Anthropic → headquartered in → China`,
  `hugging_face → developed by → Google DeepMind`처럼 명백한 오류도 섞여 있다. **Tier B에도 오류가 남아 있으므로**
  사람 검수 전에는 개별 relation을 사실로 인용하지 않는다.
- 2024Q3은 9월, 2026Q3은 7~8월만 포함한다. 날짜는 모두 기사 발행일이다.
"""
        ),
        markdown(
            """
## 31. 목적별 graph 시각화

전체 graph를 한 그림에 그리면 읽을 수 없으므로 질문별로 subgraph를 나눈다. 모든 그림에서 node 색은
compact entity type, edge 색은 predicate group이고, threshold는 제목에 적었다.

1. **핵심 typed graph**: typed PageRank 상위 entity 사이의 relation
2. **Open model family → developer**: lexicon(Tier A)과 기사 근거(Tier B)를 선 모양으로 구분
3. **경쟁·성능 비교**: `competes with`(점선, 방향 없음)와 `outperforms`(화살표)
"""
        ),
        code(
            """
def draw_typed_subgraph(edges, title, filename, max_labels=40, figsize=(14, 10), k=None):
    graph = nx.MultiDiGraph()
    for row in edges.itertuples(index=False):
        for entity_id in (row.subject_id, row.object_id):
            graph.add_node(entity_id, label=labels[entity_id],
                           compact=node_index.loc[entity_id, "entity_type_compact"])
        graph.add_edge(row.subject_id, row.object_id, key=row.predicate, group=row.predicate_group,
                       weight=row.weight, directed=row.directed, sources=row.sources)
    simple = nx.Graph(graph.to_undirected())
    positions = nx.spring_layout(simple, seed=SEED, k=k or 1.6 / np.sqrt(len(simple)), iterations=400)
    fig, ax = plt.subplots(figsize=figsize)
    max_weight = max(d["weight"] for _, _, d in graph.edges(data=True))
    for source, target, data in graph.edges(data=True):
        nx.draw_networkx_edges(
            graph, positions, edgelist=[(source, target)], ax=ax, width=0.6 + 3.0 * data["weight"] / max_weight,
            edge_color=GROUP_COLORS.get(data["group"], "#6B7280"), alpha=0.55,
            style="solid" if "glirel" in data["sources"] else "dashed",
            arrows=True, arrowstyle="-|>" if data["directed"] else "-", arrowsize=10,
            connectionstyle="arc3,rad=0.06", node_size=300)
    degree = dict(simple.degree())
    nx.draw_networkx_nodes(graph, positions, ax=ax, linewidths=0.6, edgecolors="white",
                           node_color=[COMPACT_COLORS[graph.nodes[n]["compact"]] for n in graph],
                           node_size=[180 + 60 * degree[n] for n in graph])
    shown = sorted(graph, key=lambda n: -degree[n])[:max_labels]
    nx.draw_networkx_labels(graph, positions, labels={n: pretty(graph.nodes[n]["label"]) for n in shown}, font_size=8, ax=ax)
    groups = sorted({d["group"] for _, _, d in graph.edges(data=True)})
    types = sorted({graph.nodes[n]["compact"] for n in graph})
    handles = ([Patch(color=COMPACT_COLORS[t], label=t) for t in types]
               + [Line2D([0], [0], color=GROUP_COLORS[g], lw=2, label=g) for g in groups])
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, ncol=2)
    ax.set_title(title)
    ax.axis("off")
    save_figure(fig, filename)

core_ids = set(typed_metrics.head(45)["entity_id"])
core_edges = collapsed[collapsed["subject_id"].isin(core_ids) & collapsed["object_id"].isin(core_ids)]
draw_typed_subgraph(core_edges, f"Core typed graph: relations among the 45 highest-PageRank entities ({len(core_edges)} edges)",
                    "06_core_typed_graph.png")

family_ids = set(nodes_kg.loc[nodes_kg["is_open_model_family"], "entity_id"])
family_edges = collapsed[collapsed["subject_id"].isin(family_ids)
                         & collapsed["predicate"].isin(["developed by", "released by"])]
# Family -> developer pairs are mostly disconnected, so a ranked bar chart is clearer than a network layout.
family_plot = family_edges.assign(
    pair=family_edges["subject_id"].map(labels).map(pretty) + "  ->  " + family_edges["object_id"].map(labels).map(pretty)
         + np.where(family_edges["predicate"].eq("released by"), "  (released by)", ""),
    evidence=np.select(
        [family_edges["sources"].str.contains("glirel") & family_edges["sources"].str.contains("wp1_lexicon"),
         family_edges["sources"].str.contains("glirel")],
        ["lexicon + article evidence", "article evidence only"], default="lexicon only (no article sentence)"),
).sort_values(["document_count", "pair"]).reset_index(drop=True)
EVIDENCE_COLORS = {"lexicon + article evidence": "#4C78A8", "article evidence only": "#F58518",
                   "lexicon only (no article sentence)": "#BAB0AC"}
fig, ax = plt.subplots(figsize=(10, 0.28 * len(family_plot) + 1.5))
ax.barh(family_plot["pair"], family_plot["document_count"].clip(lower=0.3),
        color=family_plot["evidence"].map(EVIDENCE_COLORS))
for y, value in enumerate(family_plot["document_count"]):
    ax.text(value + 1, y, str(int(value)), va="center", fontsize=7)
ax.legend(handles=[Patch(color=c, label=l) for l, c in EVIDENCE_COLORS.items()], loc="lower right", frameon=False, fontsize=8)
ax.set(title="Open model families -> developer: supporting articles per edge", xlabel="Supporting articles", ylabel="")
ax.tick_params(axis="y", labelsize=8)
save_figure(fig, "07_family_developer_edges.png")
display(family_plot["evidence"].value_counts().rename_axis("근거 유형").reset_index(name="edge 수"))

compete = collapsed[collapsed["predicate"].isin(["competes with", "outperforms"])]
compete_ids = set(pd.concat([compete["subject_id"], compete["object_id"]]).value_counts().head(35).index)
compete = compete[compete["subject_id"].isin(compete_ids) & compete["object_id"].isin(compete_ids)]
draw_typed_subgraph(compete, f"Competition and benchmark claims among the 35 most connected entities ({len(compete)} edges)",
                    "08_competition_graph.png", figsize=(13, 10))
"""
        ),
        markdown(
            """
## 32. Insight 후보와 근거 문장

graph pattern을 insight로 쓰기 전에 원문 근거와 대조해야 한다. 아래 표는 graph에서 자동으로 고른
후보이며, 각 후보에 근거 기사 수와 대표 근거 문장(점수가 가장 높은 문장, 기사 발행일 포함)을 붙인다.
`review_status=pending_team_review`이므로 팀 검토 전에는 확정된 결론이 아니다.
"""
        ),
        code(
            """
def evidence_examples(subject_id, predicate, object_id, n=1):
    rows = dated[(dated["subject_id"] == subject_id) & (dated["predicate"] == predicate) & (dated["object_id"] == object_id)]
    rows = rows.sort_values("rule_score", ascending=False).drop_duplicates("document_id").head(n)
    return " || ".join(f"[{r.published_at}] {str(r.evidence_text)[:220]}" for r in rows.itertuples(index=False))

findings = []
developers = relation_metrics[relation_metrics["predicate"] == "developed by"].nlargest(5, "in_degree")
for row in developers.itertuples(index=False):
    products = collapsed[(collapsed["predicate"] == "developed by") & (collapsed["object_id"] == row.entity_id)]
    top = products.nlargest(4, "document_count")
    findings.append({"finding_type": "developer_hub", "entity_or_edge": row.canonical_name,
                     "metric": f"developed by in-degree {int(row.in_degree)}",
                     "documents": int(products["document_count"].sum()),
                     "detail": ", ".join(top["subject_id"].map(labels)),
                     "evidence": evidence_examples(top.iloc[0]["subject_id"], "developed by", row.entity_id)})
for predicate in ["competes with", "outperforms"]:
    for row in collapsed[collapsed["predicate"] == predicate].nlargest(4, "document_count").itertuples(index=False):
        findings.append({"finding_type": f"strong_{predicate.replace(' ', '_')}",
                         "entity_or_edge": f"{labels[row.subject_id]} -> {predicate} -> {labels[row.object_id]}",
                         "metric": "supporting articles", "documents": int(row.document_count),
                         "detail": f"{row.first_seen} ~ {row.last_seen}",
                         "evidence": evidence_examples(row.subject_id, predicate, row.object_id)})
for row in emerging.nlargest(5, "document_count").itertuples(index=False):
    findings.append({"finding_type": "emerging_2026", "entity_or_edge": row.edge, "metric": "first seen",
                     "documents": int(row.document_count), "detail": row.first_seen,
                     "evidence": evidence_examples(row.subject_id, row.predicate, row.object_id)})
for row in shares.nlargest(3, "change_pp").itertuples():
    findings.append({"finding_type": "rising_entity", "entity_or_edge": row.canonical_name,
                     "metric": f"relation evidence share {row.change_pp:+.1f}pp (P2 - P1)", "documents": None,
                     "detail": row.entity_type_compact, "evidence": ""})
finding_candidates = pd.DataFrame(findings).assign(review_status="pending_team_review")
with pd.option_context("display.max_colwidth", 160):
    display(finding_candidates)
"""
        ),
        markdown(
            """
### 핵심 insight 초안 (팀 검토 전)

각 항목은 관찰 → 사용한 graph·지표 → 근거 → 한계 순서로 적었다. 위 후보 표의 근거 문장을 원문과 대조한 뒤
확정한다.

**1. 오픈 모델 담론의 중심이 Llama에서 중국 family와 Nemotron으로 분산되었다.**
30절 family 기사 비율과 relation 근거 비중을 사용했다. Llama 68% → 19%, Qwen 5% → 32%, Kimi 0% → 40%,
GLM 0% → 27%이고, `GLM → outperforms → Claude Opus`(2026-08-11, "GLM-5.2 (Zhipu AI) beats Claude Opus on some
coding benchmarks at 46% of the cost") 같은 2026년 relation이 나타났다.
한계: 언급 비율은 실제 사용량이나 성능이 아니라 기사 노출이다.

**2. NVIDIA는 typed graph에서 하드웨어 공급자를 넘어 모델·agent 소프트웨어 개발사로 연결된다.**
typed PageRank 1위, `developed by` in-degree 83. Nemotron 기사 비율은 2026Q1 25%로 최고였고, 2026년 신규
relation에 NemoClaw·OpenShell이 있다 (2026-03-16, "Nvidia says it developed NemoClaw in collaboration with
OpenClaw founder Peter Steinberger").
한계: 같은 2026년 신규 relation의 `OpenClaw → developed by → NVIDIA`는 근거 문장상 오류 가능성이 높다.

**3. 경쟁 관계는 OpenAI를 중심으로 형성된다.**
`competes with` 상위 쌍은 DeepSeek–OpenAI(103개 기사), Anthropic–OpenAI(97), Google–OpenAI(80)이고, OpenAI의
`competes with` weighted degree가 가장 크다 (2024-12-17, "DeepSeek said last month that its R1 model competes
with OpenAI's o1").
한계: 여러 회사를 나열한 문장에서 추출된 경쟁 관계가 섞여 있다.

**4. 2025Q1 DeepSeek 보도 집중이 전체 graph 구조에 큰 영향을 준다.**
DeepSeek의 relation 근거 기사 비중은 P1 39% → P2 12%이고, 2025Q1에 근거 문장과 신규 triple이 집중된다.
전체 기간을 합친 centrality는 이 분기의 영향을 크게 받으므로, 기간별 지표와 함께 해석해야 한다.

**insight로 쓰지 않는 것**: `outperforms` 상위 edge(`Qwen → Gemma`, `GPT-4o → Claude` 등)는 근거 문장을 보면
"it outperforms Qwen 2.5, Gemma 3 and Llama 3"처럼 제3의 모델이 둘 다를 능가한다는 서술에서 잘못 추출된 경우가
많다. 이 relation 유형은 사람 검수 전까지 insight에서 제외한다.
"""
        ),
        markdown(
            """
## 33. 한계와 타당성 위협

- **Tier B precision 미확인**: 보정 전 표본의 precision은 AI 초안 기준 약 0.47, 단어 목록 필터 후
  약 0.71이었다. 보정 규칙의 효과는 새 Tier B 표본 90개의 사람 검수로만 확정할 수 있다. 그 전까지
  graph 상태는 `candidate_pending_human_review`다.
- **Tier A는 기사 근거가 없다**: WP1 lexicon의 curated relation이다. Stage 3에서 기사 근거가 필요한
  답변에는 Tier B assertion만 인용한다.
- **날짜 의미**: 모든 날짜는 기사 발행일이다. 출시일이나 사건 발생일로 표현하지 않는다.
- **co-occurrence ≠ 의미 관계**: 같은 문장에 함께 등장했다는 뜻일 뿐, 경쟁·협력·인과를 뜻하지 않는다.
  병합 node 쌍의 문서 수는 상한값이다.
- **중심성 ≠ 산업 영향력**: 1,210개 Open Models 기사에서의 언급·관계 구조다. 2단계에서 확인했듯 score와
  threshold에 따라 순위가 크게 바뀐다.
- **일반명사 규칙과 별칭 표의 한계**: 규칙이 놓친 일반명사(예: `Fox`, `Nari`)와 등록되지 않은 별칭이
  남아 있을 수 있다. `outputs/stage1_wp3_relation_correction_20261008/corrections/`를 검토해 보완한다.
- **family 단위 축약**: `MedGemma` → `Gemma`, `Mixtral` → `Mistral`처럼 WP1 lexicon이 family 단위로
  묶으므로, 버전별 lineage(`successor of`, `fine-tuned from`)는 거의 추출되지 않았다.
"""
        ),
        markdown(
            """
## 34. 저장과 재로딩 검증

Stage 3에서 다시 쓰도록 graph와 schema를 저장한다.

- **Parquet**: 기계가 읽는 기준 파일. CSV는 사람이 확인하기 위한 사본
- **GraphML**: Gephi·Cytoscape 등 외부 도구용. list 속성은 JSON 문자열로 저장
- **node-link JSON**: NetworkX와 Stage 3 코드에서 바로 로드
- `graph_schema.json`, `DATA_DICTIONARY_KO.md`, `run_manifest.json`

대용량 파일은 `outputs/stage1_wp3_knowledge_graph_final/`(Git 제외, 공유 Drive로 전달)에, 작은
schema·요약·그림은 `artifacts/stage1_wp3/final/`에 저장한다.
"""
        ),
        code(
            """
alias_names = wp2_nodes.set_index("entity_id")["entity"]
nodes_kg["aliases"] = [
    json.dumps(sorted({alias_names.get(i, "") for i in merged.split(" | ")} - {name, ""}), ensure_ascii=False)
    for merged, name in zip(nodes_kg["merged_ids"], nodes_kg["canonical_name"])
]
graph_node_ids = set(typed_graph) | set(cooccurrence_graph) | set(assertion_graph)
nodes_out = nodes_kg[nodes_kg["entity_id"].isin(graph_node_ids)].merge(
    typed_metrics[["entity_id", "pagerank", "betweenness", "weighted_degree"]].add_prefix("typed_")
    .rename(columns={"typed_entity_id": "entity_id"}), on="entity_id", how="left",
).merge(
    cooc_metrics[["entity_id", "pagerank", "betweenness", "clustering"]].add_prefix("cooc_")
    .rename(columns={"cooc_entity_id": "entity_id"}), on="entity_id", how="left",
).merge(community_members[["entity_id", "community_id"]], on="entity_id", how="left")

tables = {
    "nodes": nodes_out,
    "typed_edges_assertions": assertions,
    "typed_edges_collapsed": collapsed,
    "cooccurrence_edges": cooccurrence_edges,
    "node_metrics": pd.concat([typed_metrics, cooc_metrics], ignore_index=True),
    "relation_type_metrics": relation_metrics,
    "relation_type_summary": relation_summary,
    "node_communities": community_members,
    "community_summary": community_summary,
    "quarterly_relation_assertions": quarterly_assertions.reset_index(),
    "quarterly_new_typed_edges": quarterly_new_edges.reset_index(),
    "family_quarterly_share": family_share.reset_index(names="family"),
    "emerging_relations_2026": emerging,
    "entity_period_share": shares.reset_index(names="entity_id"),
    "graph_finding_candidates": finding_candidates,
}
for name, table in tables.items():
    table.to_parquet(FINAL_OUTPUT_DIR / f"{name}.parquet", index=False)
    table.to_csv(FINAL_OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

def graphml_safe(graph):
    safe = graph.copy()
    for _, data in safe.nodes(data=True):
        data.update({k: scalar(v) for k, v in data.items()})
    edge_data = safe.edges(keys=True, data=True) if safe.is_multigraph() else safe.edges(data=True)
    for *_, data in edge_data:
        data.update({k: scalar(v) for k, v in data.items()})
    return safe

for stem, graph in [("assertion_graph", assertion_graph), ("typed_graph", typed_graph),
                    ("cooccurrence_graph", cooccurrence_graph)]:
    nx.write_graphml(graphml_safe(graph), FINAL_OUTPUT_DIR / f"{stem}.graphml")
    (FINAL_OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps(nx.node_link_data(graphml_safe(graph), edges="edges"), ensure_ascii=False), encoding="utf-8")

summary = {
    "schema_version": graph_schema["schema_version"], "status": GRAPH_STATUS,
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "graphs": {name: graph_summary(graph) for name, graph in
               [("assertion", assertion_graph), ("typed", typed_graph), ("cooccurrence", cooccurrence_graph)]},
    "relation_types": int(collapsed["predicate"].nunique()),
    "tiers": collapsed["tier"].value_counts().to_dict(),
    "review": {"reviewed_rows": int(review_labels.notna().sum()),
               "precision": None if np.isnan(review_precision) else float(review_precision)},
    "cooccurrence_minimum_documents": MIN_COOCCURRENCE_DOCUMENTS,
    "communities": int(len(community_summary)), "modularity": round(float(modularity), 4),
}
sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = {
    **summary,
    "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in [
        WP2_INPUT_DIR / "nodes.csv", WP2_INPUT_DIR / "relations_evidence.csv",
        WP2_INPUT_DIR / "edges_cooccurrence.csv", CORRECTION_DIR / "edges_graph_candidates.csv",
        CORRECTION_DIR / "nodes_corrected.csv", CORRECTED_REVIEW_PATH]},
    "versions": {"networkx": nx.__version__, "pandas": pd.__version__},
}
for directory in [FINAL_OUTPUT_DIR, FINAL_ARTIFACT_DIR]:
    (directory / "graph_schema.json").write_text(json.dumps(graph_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "graph_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(FINAL_OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
for small in ["typed_edges_collapsed", "relation_type_summary", "community_summary", "graph_finding_candidates"]:
    tables[small].to_csv(FINAL_ARTIFACT_DIR / f"{small}.csv", index=False, encoding="utf-8-sig")

# Re-load check: the saved files must reproduce the in-memory graphs
reloaded = {
    "typed_graph.graphml": nx.read_graphml(FINAL_OUTPUT_DIR / "typed_graph.graphml"),
    "typed_graph.json": nx.node_link_graph(json.loads((FINAL_OUTPUT_DIR / "typed_graph.json").read_text(encoding="utf-8")), edges="edges"),
    "cooccurrence_graph.graphml": nx.read_graphml(FINAL_OUTPUT_DIR / "cooccurrence_graph.graphml"),
    "assertion_graph.json": nx.node_link_graph(json.loads((FINAL_OUTPUT_DIR / "assertion_graph.json").read_text(encoding="utf-8")), edges="edges"),
}
expected = {"typed": typed_graph, "cooccurrence": cooccurrence_graph, "assertion": assertion_graph}
reload_check = pd.DataFrame([
    (name, g.number_of_nodes(), g.number_of_edges(),
     (g.number_of_nodes(), g.number_of_edges()) == (expected[name.split("_")[0]].number_of_nodes(),
                                                   expected[name.split("_")[0]].number_of_edges()))
    for name, g in reloaded.items()
], columns=["file", "node", "edge", "일치"])
display(reload_check)
assert reload_check["일치"].all()
print(sorted(p.name for p in FINAL_OUTPUT_DIR.iterdir())[:40])
"""
        ),
        code(
            """
DATA_DICTIONARY = f'''# Stage 1 WP3 최종 지식그래프 데이터 사전

Schema: `{graph_schema["schema_version"]}` · 상태: `{GRAPH_STATUS}` · Track: Open Models
날짜(`published_at`, `first_seen`, `last_seen`)는 모두 **기사 발행일**이다.

## Node (`nodes.*`)
| 열 | 설명 |
|---|---|
| `entity_id` | 보정 후 node ID. 별칭·이름 변형이 합쳐진 원래 WP2 ID는 `merged_ids` |
| `canonical_name`, `aliases` | 대표 이름과 병합된 다른 이름(JSON list) |
| `entity_type` / `entity_type_compact` | WP2 type / 9개 compact type |
| `mention_count`, `document_count` | 언급 수 / 언급 기사 수 |
| `is_open_model_family` | WP1 lexicon의 open model family 여부 |
| `typed_*`, `cooc_*`, `community_id` | typed·co-occurrence graph metric과 community |

## Typed edge
- `typed_edges_assertions.*`: 근거 문장 1개당 1행. `relation_id`, `subject_id`, `predicate`, `object_id`,
  `document_id`, `text_unit_id`(= `doc_id:sentence_id`), `evidence_text`, `published_at`, `extraction_method`
  (`glirel` 또는 `wp1_lexicon`), `rule_score`
- `typed_edges_collapsed.*`: unique triple 1개당 1행. `evidence_count`, `document_count`(edge weight),
  `n_distinct_sentences`, `first_seen`, `last_seen`, `tier`(A/B), `sources`, `review_status`,
  `predicate_group`, `directed`, `relation_ids`/`text_unit_ids`(JSON list)

## Co-occurrence edge (`cooccurrence_edges.*`)
같은 문장에 함께 언급된 entity 쌍. `sentence_count`, `document_count`(문서 {MIN_COOCCURRENCE_DOCUMENTS}개 이상),
`npmi`, `merged_pairs`(보정으로 합쳐진 원래 쌍 수). 의미 관계가 아니다.

## Graph 파일
`assertion_graph`, `typed_graph`, `cooccurrence_graph` 각각 `.graphml`과 node-link `.json`

## 분석 표
`node_metrics`, `relation_type_metrics`, `relation_type_summary`, `node_communities`, `community_summary`,
`quarterly_relation_assertions`, `quarterly_new_typed_edges`, `family_quarterly_share`,
`emerging_relations_2026`, `entity_period_share`, `graph_finding_candidates`(팀 검토 전 후보)

## Stage 3 provenance
`entity_id` → `relation_id` → `text_unit_id` → `document_id` → `published_at`.
기사 제목·URL은 WP1 `open_models_core.parquet`의 `doc_id`로 연결한다.
'''
for directory in [FINAL_OUTPUT_DIR, FINAL_ARTIFACT_DIR]:
    (directory / "DATA_DICTIONARY_KO.md").write_text(DATA_DICTIONARY, encoding="utf-8")
print(DATA_DICTIONARY[:600])
"""
        ),
        markdown(
            """
## 35. Stage 3 Graph-RAG 연결 smoke test

저장한 graph를 기존 `GraphEvidenceRetriever`로 다시 읽어, 질문 속 entity → typed edge → 근거 문장 →
기사 ID·발행일로 돌아갈 수 있는지 확인한다. WP1 `open_models_core.parquet`이 있으면 기사 제목과 URL도
붙인다.
"""
        ),
        code(
            """
text_units = pd.DataFrame({
    "text_unit_id": text_unit(sentences["doc_id"], sentences["sentence_id"]),
    "document_id": sentences["doc_id"],
    "text_unit_text": sentences["sentence_text"],
})
documents = (assertions.dropna(subset=["document_id"]).groupby("document_id")["published_at"].first()
             .reset_index())
if WP1_CORE_PATH.exists():
    wp1_documents = pd.read_parquet(WP1_CORE_PATH, columns=["doc_id", "title", "url"])
    documents = documents.merge(wp1_documents.rename(columns={"doc_id": "document_id"}), on="document_id", how="left")

retriever = GraphEvidenceRetriever(
    nodes=pd.read_parquet(FINAL_OUTPUT_DIR / "nodes.parquet"),
    assertions=pd.read_parquet(FINAL_OUTPUT_DIR / "typed_edges_assertions.parquet").dropna(subset=["text_unit_id"]),
    text_units=text_units,
    documents=documents,
)
smoke_queries = [
    ("Who developed Qwen?", ["developed by"]),
    ("Which companies compete with DeepSeek?", ["competes with"]),
    ("What does NVIDIA manufacture?", ["manufactured by"]),
]
smoke_rows = []
for query, predicates in smoke_queries:
    result = retriever.retrieve(query, predicates=predicates, top_k=3)
    for item in result["evidence"]:
        smoke_rows.append({"query": query, "subject": item["subject"]["name"], "predicate": item["predicate"],
                           "object": item["object"]["name"], "published_at": item["published_at"],
                           "document_id": item["document_id"], "passage": str(item["passage"])[:160]})
    assert result["evidence"], f"no evidence returned for: {query}"
smoke_test = pd.DataFrame(smoke_rows)
with pd.option_context("display.max_colwidth", 160):
    display(smoke_test)
"""
        ),
        markdown(
            """
## 36. Stage 1 완료 체크리스트

과제 안내서의 Knowledge Graph Construction·Outputs 요구사항을 기준으로 자동 확인 결과와 사람이
해야 할 항목을 구분한다. `PENDING`은 코드로 확정할 수 없는 팀 작업이다.
"""
        ),
        code(
            """
figure_files = sorted(p.name for p in FIGURE_DIR.glob("*.png"))
checklist = pd.DataFrame(
    [
        ("compact topic-specific schema를 graph 전에 정의", "PASS", f"{len(graph_schema['entity_types'])} node type, "
         f"{len(graph_schema['relation_types'])} relation type, graph_schema.json"),
        ("node = entity, edge = co-mention 또는 extracted relation", "PASS",
         f"typed {typed_graph.number_of_edges():,} / co-occurrence {cooccurrence_graph.number_of_edges():,} edge"),
        ("모든 edge endpoint가 node table에 존재", "PASS" if graph_node_ids <= set(nodes_out["entity_id"]) else "FAIL", "nodes.parquet"),
        ("centrality 분석", "PASS", "weighted degree, PageRank, betweenness, relation별 in/out degree"),
        ("clustering·community 분석", "PASS", f"{len(community_summary)} communities, modularity {modularity:.3f}"),
        ("시간 분석 (emerging relationship)", "PASS", f"분기별 relation, family 추이, 2026년 신규 triple {len(emerging)}개"),
        ("NetworkX 시각화와 legend", "PASS", f"{len(figure_files)}개 그림"),
        ("graph와 schema 저장·재로딩", "PASS" if reload_check["일치"].all() else "FAIL", "GraphML, node-link JSON, Parquet"),
        ("short schema / data dictionary", "PASS", "DATA_DICTIONARY_KO.md"),
        ("Stage 3 provenance smoke test", "PASS" if len(smoke_test) else "FAIL", f"{len(smoke_test)}개 근거 반환"),
        ("Tier B relation 사람 검수 90개, precision >= 0.80", "PASS" if GRAPH_STATUS == "human_validated_sample" else "PENDING",
         f"검수 {int(review_labels.notna().sum())}/90"),
        ("insight를 원문 근거와 대조해 확정", "PENDING", "graph_finding_candidates 팀 검토"),
        ("그림 label·legend·threshold 사람 승인", "PENDING", ", ".join(figure_files)),
        ("WP1 topic·temporal 결과와 통합 (stage1_final.ipynb)", "PENDING", "WP1 표와 30절 family 추이 비교"),
        ("팀원 기여·AI 도구 공개, 최종본 영어 전환", "PENDING", "37절 초안 확인"),
    ],
    columns=["요구사항", "상태", "근거"],
)
display(checklist)
assert not checklist["상태"].eq("FAIL").any()
"""
        ),
        markdown(
            """
## 37. 팀원 기여와 AI 도구 공개 (초안)

| 작업 | 담당 | 내용 |
|---|---|---|
| Stage 1 WP1 | Daliasny Gonzalez Guzman | 데이터 정제·전처리, EDA, sentiment/frame, topic modeling, Open Models lexicon |
| Stage 1 WP2 | Jennifer Spieser | spaCy·GLiNER entity 추출, GLiREL relation 추출, co-occurrence, handover schema |
| Stage 1 WP3 | Jihyeon Choung | WP2 인계 검증, relation 검수·보정, compact schema, knowledge graph 구축·분석·시각화, Stage 3 인계 |
| relation 검수 | 팀원 3명 | Tier B 표본 90개 (각 30개) — 진행 예정 |

**AI 도구 사용 공개 (초안)**

- **Claude Code (Anthropic)**: WP2 relation 표본의 1차 판정 초안 작성(공식 검수에 사용하지 않음),
  relation correction notebook 초안과 보정 규칙 설계, 본 notebook 3단계(22–37절)의 코드와 설명 초안
  작성. 모든 수치는 notebook 실행 결과에서 나왔고, 해석과 결론은 팀 검토 대상이다.
- 다른 AI 도구를 사용했다면 도구 이름과 사용 범위를 여기에 추가한다.
"""
        ),
    ]
