from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "stage1_wp3_open_models_contract.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Stage 1 WP3 — Contract-driven Knowledge Graph

이 노트북은 Stage 1 WP2의 공식 산출물이 아직 확정되지 않은 상태에서 WP3 작업을 진행하기
위한 **계약 기반 분석 노트북**이다. 파일명과 열 이름을 분석 코드에 직접 고정하지 않고,
`config/wp3_open_models_contract.json`의 alias와 정책을 통해 WP2 결과를 canonical schema로
변환한다.

현재 실행 결과는 300문서 파일럿을 사용한 개발용 baseline이다. 공식 WP2 결과가 도착하면
입력 경로와 계약만 갱신한 뒤 동일한 셀을 다시 실행해야 한다."""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. 실행 경로와 교체 지점

공식 WP2 결과가 도착했을 때 우선 변경할 부분은 아래 셀의 `WP2_INPUT_DIR`이다. 파일명이나
열 이름이 달라졌다면 원본 파일을 수동으로 바꾸지 말고 `CONTRACT_PATH`의 JSON에 후보 파일명과
alias를 추가한다."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "config").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

WP2_INPUT_DIR = ROOT / "outputs" / "stage1_wp2_open_models_pilot"
WP3_OUTPUT_DIR = ROOT / "outputs" / "stage1_wp3_open_models_pilot"
CONTRACT_PATH = ROOT / "config" / "wp3_open_models_contract.json"

# 공식 WP2 입력으로 graph artifacts를 다시 만들 때만 True로 변경한다.
RUN_PIPELINE = False

ROOT, WP2_INPUT_DIR, WP3_OUTPUT_DIR, CONTRACT_PATH"""
        ),
        nbf.v4.new_markdown_cell("## 2. WP2→WP3 입력 계약 확인"),
        nbf.v4.new_code_cell(
            """contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

contract_overview = []
for table_name, spec in contract["tables"].items():
    contract_overview.append({
        "table": table_name,
        "required": spec.get("required", False),
        "candidate_files": ", ".join(spec.get("files", [])),
        "required_fields": ", ".join(spec.get("required_fields", [])),
    })

print("contract_version:", contract["contract_version"])
pd.DataFrame(contract_overview)"""
        ),
        nbf.v4.new_code_cell(
            """pd.DataFrame([
    {"policy": key, "value": json.dumps(value, ensure_ascii=False)}
    for key, value in contract["policies"].items()
])"""
        ),
        nbf.v4.new_markdown_cell(
            """계약은 다음 변경을 흡수한다.

- Parquet과 CSV 파일명 변경
- `entity_id` 대신 `id`/`node_id` 사용
- `subject_id/predicate/object_id` 대신 `source/relation/target` 사용
- review, documents, text units, co-occurrence 같은 선택 입력의 추가·누락
- assertion status, confidence threshold, provenance 엄격성 변경

공식 계약이 확정되면 `strict_provenance`를 `true`로 바꾸어 evidence 연결이 없는 관계를
graph 생성 전에 차단하는 것이 권장된다."""
        ),
        nbf.v4.new_markdown_cell("## 3. 필요할 때 파이프라인 재실행"),
        nbf.v4.new_code_cell(
            """if RUN_PIPELINE:
    from wp3_knowledge_graph import WP3PipelineConfig, run_pipeline

    pipeline_summary = run_pipeline(
        WP3PipelineConfig(
            input_dir=WP2_INPUT_DIR,
            output_dir=WP3_OUTPUT_DIR,
            contract_path=CONTRACT_PATH,
        )
    )
    display(pd.json_normalize(pipeline_summary))
else:
    print("기존 WP3 산출물을 사용합니다. 공식 WP2 입력으로 교체할 때 RUN_PIPELINE=True로 설정하세요.")"""
        ),
        nbf.v4.new_markdown_cell("## 4. 실제로 해석된 계약과 validation 결과"),
        nbf.v4.new_code_cell(
            """contract_report = json.loads(
    (WP3_OUTPUT_DIR / "contract_report.json").read_text(encoding="utf-8")
)
validation = json.loads(
    (WP3_OUTPUT_DIR / "validation_report.json").read_text(encoding="utf-8")
)

resolved_tables = pd.DataFrame([
    {
        "table": name,
        "rows": details["rows"],
        "resolved_file": Path(details["path"]).name,
        "resolved_columns": json.dumps(details["resolved_columns"], ensure_ascii=False),
    }
    for name, details in contract_report["tables"].items()
])

display(resolved_tables)
display(pd.DataFrame({
    "check": [key for key in validation if key not in {"errors", "warnings"}],
    "result": [validation[key] for key in validation if key not in {"errors", "warnings"}],
}))
print("warnings:", validation["warnings"])
print("errors:", validation["errors"])
assert validation["passed"], '입력 계약 또는 graph integrity validation이 실패했습니다.'"""
        ),
        nbf.v4.new_markdown_cell(
            """Validation 통과는 ID, endpoint, schema signature, provenance 같은 구조적 조건이
충족됐다는 뜻이다. 자동으로 추출된 relation의 의미가 모두 정확하다는 뜻은 아니며, 이는
`relation_review_sample.csv`의 수동 검수로 별도 평가해야 한다."""
        ),
        nbf.v4.new_markdown_cell("## 5. Graph view와 policy 적용 결과"),
        nbf.v4.new_code_cell(
            """summary = json.loads(
    (WP3_OUTPUT_DIR / "graph_summary.json").read_text(encoding="utf-8")
)

display(pd.DataFrame([
    {"graph_view": name, **statistics}
    for name, statistics in summary["graphs"].items()
]))
display(pd.DataFrame([
    {"policy_step": key, "relation_count": value}
    for key, value in summary["policy_audit"].items()
]))"""
        ),
        nbf.v4.new_markdown_cell(
            """세 graph view의 목적은 다르다.

- **Assertion graph**: evidence별 edge를 유지하여 provenance와 Stage 3 retrieval에 사용한다.
- **Collapsed typed graph**: 같은 `(subject, predicate, object)`를 합쳐 metric과 발표용 분석에 사용한다.
- **Co-occurrence graph**: 같은 문장에 나타난 엔티티를 연결하며 community와 담론 구조 분석에 사용한다.

Assertion 수를 unique fact 수로 해석하거나, 공동출현을 의미 관계로 해석하면 안 된다."""
        ),
        nbf.v4.new_markdown_cell("## 6. Collapsed typed relations"),
        nbf.v4.new_code_cell(
            """nodes = pd.read_parquet(WP3_OUTPUT_DIR / "nodes.parquet")
assertions = pd.read_parquet(WP3_OUTPUT_DIR / "typed_edges_assertions.parquet")
typed_edges = pd.read_parquet(WP3_OUTPUT_DIR / "typed_edges_collapsed.parquet")
cooccurrence = pd.read_parquet(WP3_OUTPUT_DIR / "cooccurrence_edges.parquet")
metrics = pd.read_parquet(WP3_OUTPUT_DIR / "node_metrics.parquet")

labels = nodes.set_index("entity_id")["canonical_name"]
typed_view = typed_edges.assign(
    subject_label=typed_edges["subject_id"].map(labels),
    object_label=typed_edges["object_id"].map(labels),
)

display(typed_view["predicate"].value_counts().rename_axis("predicate").reset_index(name="unique_edges"))
typed_view.sort_values(
    ["document_count", "evidence_count"], ascending=False
)[[
    "subject_label", "predicate", "object_label", "evidence_count",
    "document_count", "domain_count", "first_seen", "last_seen",
]].head(25)"""
        ),
        nbf.v4.new_markdown_cell(
            """`document_count`는 서로 다른 source document 수이고 `evidence_count`는 문장별
assertion 수다. 반복 모델 카드나 유사 기사에 덜 민감하도록 기본 시각화와 weighted metric에는
`document_count`를 사용한다."""
        ),
        nbf.v4.new_markdown_cell("## 7. Network metrics"),
        nbf.v4.new_code_cell(
            """metric_columns = [
    "canonical_name", "entity_type", "degree", "weighted_degree",
    "pagerank", "betweenness", "clustering",
]

print("Collapsed typed graph")
display(
    metrics[metrics["graph_view"] == "typed"]
    .sort_values("pagerank", ascending=False)
    .head(15)[metric_columns]
)

print("Co-occurrence graph")
display(
    metrics[metrics["graph_view"] == "cooccurrence"]
    .sort_values("pagerank", ascending=False)
    .head(15)[metric_columns]
)"""
        ),
        nbf.v4.new_markdown_cell(
            """현재 metric은 300문서 파일럿과 현재 graph policy 안에서의 연결성을 나타낸다.
실제 산업 영향력이나 모델 성능 순위로 해석할 수 없다. 또한 typed graph의 방향은 relation에
따라 의미가 다르므로, 최종 분석에서는 predicate별 subgraph를 함께 제시해야 한다."""
        ),
        nbf.v4.new_markdown_cell("## 8. 관계 유형별 지표와 시간 변화"),
        nbf.v4.new_code_cell(
            """relation_summary = pd.read_parquet(WP3_OUTPUT_DIR / "relation_type_summary.parquet")
monthly_assertions = pd.read_parquet(WP3_OUTPUT_DIR / "monthly_relation_assertions.parquet")
monthly_entities = pd.read_parquet(WP3_OUTPUT_DIR / "monthly_new_entities.parquet")
monthly_edges = pd.read_parquet(WP3_OUTPUT_DIR / "monthly_new_typed_edges.parquet")

display(relation_summary)
display(monthly_assertions.tail(20))

predicate_metrics = metrics[metrics["graph_view"].str.startswith("predicate:", na=False)]
display(
    predicate_metrics.sort_values(["graph_view", "pagerank"], ascending=[True, False])
    .groupby("graph_view", as_index=False)
    .head(5)[["graph_view", *metric_columns]]
)"""
        ),
        nbf.v4.new_markdown_cell(
            """관계 유형별 graph를 분리하면 방향과 의미가 다른 edge를 한꺼번에 합산하는 오류를
줄일 수 있다. 월별 표는 데이터 수집량 변화에도 영향을 받으므로, 상승·하락을 실제 시장 변화로
확정하기 전에 월별 document 수와 원문 evidence를 함께 확인해야 한다."""
        ),
        nbf.v4.new_markdown_cell("## 9. Co-occurrence community와 검토 후보"),
        nbf.v4.new_code_cell(
            """communities = pd.read_parquet(WP3_OUTPUT_DIR / "node_communities.parquet")
community_summary = pd.read_parquet(WP3_OUTPUT_DIR / "community_summary.parquet")
finding_candidates = pd.read_parquet(WP3_OUTPUT_DIR / "graph_finding_candidates.parquet")

display(community_summary.head(15))
display(
    communities.sort_values(["community_id", "rank_in_community"])
    .groupby("community_id", as_index=False)
    .head(5)
    .head(50)
)
display(finding_candidates.head(30))
assert set(finding_candidates["review_status"]) <= {"unreviewed", "accepted", "rejected"}"""
        ),
        nbf.v4.new_markdown_cell(
            """`graph_finding_candidates`는 결론이 아니라 사람의 검토 대기열이다. 최종 보고서에
사용할 항목은 연결된 entity/relation/text unit을 원문과 대조한 뒤 `accepted` 또는 `rejected`로
표시해야 한다. 공동출현 community도 의미 관계나 인과관계를 직접 뜻하지 않는다."""
        ),
        nbf.v4.new_markdown_cell("## 10. 자동 생성된 시각화 점검"),
        nbf.v4.new_code_cell(
            """for filename in [
    "typed_graph_top.png",
    "cooccurrence_graph_top.png",
    "typed_pagerank_top15.png",
    "developer_release_subgraph.png",
    "model_attributes_subgraph.png",
    "relation_type_counts.png",
    "monthly_relation_assertions.png",
    "cooccurrence_community_sizes.png",
]:
    path = WP3_OUTPUT_DIR / "figures" / filename
    print(filename)
    display(Image(filename=str(path)))"""
        ),
        nbf.v4.new_markdown_cell(
            """전체 graph와 관계별 subgraph, 시간 변화, community 크기를 각각 분리했다.
최종 제출 전에는 label 겹침과 threshold를 확인하고, 발표에 사용할 그림은 해당 finding의
원문 evidence와 함께 선택한다."""
        ),
        nbf.v4.new_markdown_cell("## 11. 공식 WP2가 도착했을 때 교체 절차"),
        nbf.v4.new_code_cell(
            """handoff_checklist = pd.DataFrame([
    (1, "새 WP2 폴더를 별도 위치에 보존한다."),
    (2, "entities/relations 및 선택 테이블의 실제 파일명을 확인한다."),
    (3, "계약 JSON의 files와 column aliases를 갱신한다."),
    (4, "공식 evidence 필드가 확정되면 strict_provenance=true로 바꾼다."),
    (5, "새 output directory를 지정하고 RUN_PIPELINE=True로 실행한다."),
    (6, "contract_report와 validation_report를 먼저 확인한다."),
    (7, "assertion 수, collapsed edge 수, relation type 분포를 파일럿과 비교한다."),
    (8, "수동 review 결과를 반영한 뒤 metric과 시각화를 확정한다."),
], columns=["step", "action"])
handoff_checklist"""
        ),
        nbf.v4.new_markdown_cell(
            """## 12. 평가기준과 Stage 3 handoff

이 파이프라인은 Stage 1 평가기준의 schema 정의, node/edge 구성, centrality와 clustering,
NetworkX 시각화, graph/schema 저장을 지원한다. 하지만 최종 Stage 1 제출을 위해서는 다음이
추가되어야 한다. Predicate별 분석·시간 변화·community·evidence 후보는 자동 생성된다.
공식 입력 후 남는 일은 수동 검수 precision 확정, evidence 기반 insight 승인, 팀원 contribution과
AI tool disclosure 기록이다.

공식 Stage 3 Graph-RAG를 위해서는 다음 chain을 보존한다.

`entity_id → relation_id → text_unit_id → document_id → URL / published_at`

그래프가 찾은 relation만 답변 근거로 사용하지 않고, 연결된 원문 passage를 함께 retrieval해야
faithfulness와 temporal grounding을 평가할 수 있다."""
        ),
        nbf.v4.new_markdown_cell(
            """## 13. 현재 상태

이 노트북과 출력은 **WP3 pipeline integration baseline**이다. 공식 WP2 계약이 도착하기 전에는
파일럿의 graph 수치로 최종 결론을 내리지 않는다. 계약이 확정되면 adapter validation을 먼저
통과시키고, 동일 notebook을 재실행해 최종 analysis artifact를 만든다."""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
