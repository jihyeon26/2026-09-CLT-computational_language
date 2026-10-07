from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "stage1_wp3_open_models_contract.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = [
        markdown(
            """
# Stage 1 WP3 — 공식 WP2 인계 연결 및 검증

이 노트북은 팀원의 공식 WP1·WP2 산출물을 사용해 Stage 1 WP3 지식그래프 작업을 시작하는
**한국어 검토용 작업본**이다. 기존 300문서 파일럿 결과는 사용하지 않는다.

현재 범위는 WP3 1단계인 다음 작업까지다.

1. 공식 WP2 원본을 수정하지 않고 별도 입력 폴더에서 읽는다.
2. 실제 파일명과 열 이름을 WP3 canonical contract에 연결한다.
3. 문서별 `sentence_id`를 전역 고유 `text_unit_id`로 변환한다.
4. WP1 문서 ID, relation endpoint와 evidence 문장 연결을 검증한다.
5. 최종 node type, 중복, self-loop와 sentence-local offset을 그래프 계산 전에 교정한다.

분석과 팀 검토가 끝난 뒤 최종 제출본에서는 설명, 셀 제목과 주석을 영어로 바꾼다.
"""
        ),
        markdown(
            """
## 1. 공식 입력과 출력 경로

WP1 processed 데이터와 WP2 handover는 다운로드·팀 저장소의 원본을 변경하지 않고 현재 프로젝트의
Git-ignored 경로에 복사해 두었다. WP3 산출물도 파일럿 폴더와 분리된 official 경로를 사용한다.
"""
        ),
        code(
            """
from pathlib import Path
import json
import sys

import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "config").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

WP1_PROCESSED_DIR = ROOT / "dataset" / "processed"
WP2_INPUT_DIR = ROOT / "outputs" / "stage1_wp2_open_models_official_20261007"
WP3_OUTPUT_DIR = ROOT / "outputs" / "stage1_wp3_open_models_official_20261007"
CONTRACT_PATH = ROOT / "config" / "wp3_open_models_contract.json"
REVIEW_SAMPLE_PATH = (
    ROOT / "artifacts" / "stage1_wp2" / "evaluation" / "relation_review_sample_seed42.csv"
)

for required_path in [WP1_PROCESSED_DIR, WP2_INPUT_DIR, CONTRACT_PATH]:
    assert required_path.exists(), f"필수 입력이 없습니다: {required_path}"

ROOT, WP1_PROCESSED_DIR, WP2_INPUT_DIR, WP3_OUTPUT_DIR, CONTRACT_PATH
"""
        ),
        markdown("## 2. 팀 handover 파일 확인"),
        code(
            """
expected_wp1 = [
    "corpus_clean.parquet",
    "lemma_cache.parquet",
    "open_models_core.parquet",
    "open_models_target_sentences.parquet",
    "track_scan_cache.parquet",
]
expected_wp2 = [
    "nodes.csv",
    "nodes_quarterly.csv",
    "entity_mentions.csv",
    "sentences.csv",
    "edges_relations.csv",
    "relations_evidence.csv",
    "edges_cooccurrence.csv",
    "cooccurrence_quarterly.csv",
    "schema.json",
    "data_dictionary.md",
]

handover_files = []
for work_package, directory, names in [
    ("WP1", WP1_PROCESSED_DIR, expected_wp1),
    ("WP2", WP2_INPUT_DIR, expected_wp2),
]:
    for name in names:
        path = directory / name
        handover_files.append(
            {
                "work_package": work_package,
                "file": name,
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1_000_000, 2) if path.exists() else None,
            }
        )

handover_table = pd.DataFrame(handover_files)
display(handover_table)
assert handover_table["exists"].all(), "공식 handover 파일이 일부 누락됐습니다."
"""
        ),
        markdown(
            """
WP1 대용량 Parquet과 WP2 CSV는 Git에 커밋하지 않는다. 팀 공유 저장소에는 schema, data dictionary,
최종 notebook처럼 협업과 제출에 필요한 작은 파일만 올리고, 데이터는 공유 Drive에서 관리한다.
"""
        ),
        markdown("## 3. WP2→WP3 contract 적용"),
        code(
            """
from wp3_knowledge_graph.contract import load_contract_tables, read_contract
from wp3_knowledge_graph.pipeline import (
    annotate_final_relation_types,
    apply_relation_policy,
    collapse_relations,
    correct_sentence_local_offsets,
    normalize_relation_schema,
    prepare_cooccurrence,
    prepare_entities,
    prepare_relations,
    validate_inputs,
)

contract = read_contract(CONTRACT_PATH)
tables, contract_report = load_contract_tables(WP2_INPUT_DIR, contract)

entities = prepare_entities(tables["entities"])
relations = prepare_relations(tables["relations"])
cooccurrence = prepare_cooccurrence(tables.get("cooccurrence"))
text_units = tables["text_units"]

resolved_tables = pd.DataFrame(
    [
        {
            "table": name,
            "rows": details["rows"],
            "resolved_file": Path(details["path"]).name,
            "resolved_columns": json.dumps(details["resolved_columns"], ensure_ascii=False),
        }
        for name, details in contract_report["tables"].items()
    ]
)

print("contract_version:", contract["contract_version"])
display(resolved_tables)
assert contract_report["passed"]
"""
        ),
        markdown(
            """
공식 WP2 파일은 다음과 같이 선택된다.

- `nodes.csv` → canonical entities
- `relations_evidence.csv` → 문장 근거를 가진 relation assertions
- `edges_cooccurrence.csv` → co-occurrence edges
- `sentences.csv` → text units

집계본인 `edges_relations.csv`보다 `relations_evidence.csv`를 우선하는 이유는 assertion마다
`document_id`, `text_unit_id`, evidence와 발행일을 보존하기 위해서다.
"""
        ),
        markdown("## 4. WP1에서 WP2로 이어진 문서 범위 확인"),
        code(
            """
wp1_core = pd.read_parquet(
    WP1_PROCESSED_DIR / "open_models_core.parquet",
    columns=["doc_id", "date", "title", "url"],
)
wp1_doc_ids = set(wp1_core["doc_id"].astype(str))
wp2_doc_ids = set(text_units["document_id"].astype(str))

lineage_check = pd.DataFrame(
    [
        ("WP1 핵심 문서 수", len(wp1_doc_ids)),
        ("WP2 문장 데이터에 포함된 문서 수", len(wp2_doc_ids)),
        ("WP2에서 누락된 WP1 문서 수", len(wp1_doc_ids - wp2_doc_ids)),
        ("WP1에 없지만 WP2에 추가된 문서 수", len(wp2_doc_ids - wp1_doc_ids)),
    ],
    columns=["검사항목", "개수"],
)
display(lineage_check)

assert wp1_doc_ids == wp2_doc_ids
"""
        ),
        markdown("## 5. 전역 text-unit ID와 provenance 연결 확인"),
        code(
            """
known_text_units = set(text_units["text_unit_id"].dropna().astype(str))
relation_text_units = set(relations["text_unit_id"].dropna().astype(str))
glirel_relations = relations[relations["extraction_method"] == "glirel"]
curated_relations = relations[relations["extraction_method"] == "wp1_lexicon"]

provenance_checks = pd.DataFrame(
    [
        ("text-unit ID가 모두 고유함", bool(text_units["text_unit_id"].is_unique)),
        ("relation ID가 모두 고유함", bool(relations["relation_id"].is_unique)),
        ("문장 데이터에서 찾을 수 없는 relation text unit 수", len(relation_text_units - known_text_units)),
        ("문서 ID가 없는 GLiREL relation 수", int(glirel_relations["document_id"].isna().sum())),
        ("근거 문장이 없는 GLiREL relation 수", int(glirel_relations["evidence_text"].isna().sum())),
        ("기사 근거가 없는 curated relation 수", int(curated_relations["evidence_text"].isna().sum())),
    ],
    columns=["검사항목", "결과"],
)
display(provenance_checks)

assert text_units["text_unit_id"].is_unique
assert relations["relation_id"].is_unique
assert not (relation_text_units - known_text_units)
assert glirel_relations["document_id"].notna().all()
assert glirel_relations["evidence_text"].notna().all()
"""
        ),
        markdown(
            """
WP2의 `sentence_id`는 각 문서 안에서 다시 0부터 시작한다. Contract adapter는 이를
`doc_id:sentence_id` 형식으로 변환해 전체 corpus에서 고유한 `text_unit_id`를 만든다.

`wp1_lexicon`에서 추가된 35개 `developed by` 관계는 기사 문장 근거가 없는 curated relation이다.
오류로 숨기지 않고 별도 source로 유지하며, Stage 3에서 기사 evidence가 필요한 답변에는 직접
사용하지 않는다.
"""
        ),
        markdown("## 6. 공식 산출물의 구조적 한계 재현"),
        code(
            """
# 원본 집계 파일은 감사 목적으로만 읽으며 분석 입력으로 사용하지 않는다.
aggregated_upstream = pd.read_csv(WP2_INPUT_DIR / "edges_relations.csv")
aggregate_duplicate_rows = int(
    aggregated_upstream.duplicated(["head_id", "relation", "tail_id"]).sum()
)

wp2_schema = json.loads((WP2_INPUT_DIR / "schema.json").read_text(encoding="utf-8"))
canonical_schema = normalize_relation_schema(wp2_schema)
typed_relations = annotate_final_relation_types(entities, relations, canonical_schema)

aggregate_for_check = aggregated_upstream.rename(
    columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id"}
)
typed_aggregate = annotate_final_relation_types(
    entities, aggregate_for_check, canonical_schema
)
stored_valid = typed_aggregate["type_valid"].fillna(False).astype(bool)
recommended_filter = (
    (pd.to_numeric(typed_aggregate["n_sentences"], errors="coerce").fillna(0) >= 2)
    | (typed_aggregate["source"] == "wp1_lexicon")
)

limitation_audit = pd.DataFrame(
    [
        ("원본 집계 triple의 추가 중복 행", aggregate_duplicate_rows),
        (
            "저장 당시 valid였지만 최종 node type에서는 invalid인 집계 행",
            int((stored_valid & ~typed_aggregate["final_type_valid"]).sum()),
        ),
        (
            "권장 필터 후에도 최종 node type에서 invalid인 집계 행",
            int((recommended_filter & stored_valid & ~typed_aggregate["final_type_valid"]).sum()),
        ),
        ("원본 relation assertion의 self-loop", int((relations["subject_id"] == relations["object_id"]).sum())),
        ("최종 node type에서 invalid인 relation assertion", int((~typed_relations["final_type_valid"]).sum())),
    ],
    columns=["확인 항목", "개수"],
)
display(limitation_audit)

assert aggregate_duplicate_rows == 266
assert int((stored_valid & ~typed_aggregate["final_type_valid"]).sum()) == 417
assert int((recommended_filter & stored_valid & ~typed_aggregate["final_type_valid"]).sum()) == 84
assert int((relations["subject_id"] == relations["object_id"]).sum()) == 1
"""
        ),
        markdown(
            """
위 수치는 노트북 미실행 때문에 생긴 것이 아니라 WP2의 집계 기준과 mention 단계 type을 사용한
검증 방식에서 생긴다. 따라서 WP2 원본 파일을 고쳐 쓰지 않고, 아래에서 최종 node type을 적용한
WP3 분석용 relation view를 새로 만든다.
"""
        ),
        markdown("## 7. 그래프 계산 전 relation 교정"),
        code(
            """
analysis_relations, relation_policy_audit = apply_relation_policy(
    typed_relations,
    reviews=None,
    policies=contract["policies"],
)
collapsed_relations = collapse_relations(analysis_relations, documents=None)

relation_correction_summary = pd.DataFrame(
    [
        ("원본 relation assertion", len(relations)),
        ("self-loop 제외", relation_policy_audit.get("excluded_self_relations", 0)),
        ("최종 node type signature 불일치 제외", relation_policy_audit.get("excluded_by_final_signature", 0)),
        ("분석용 relation assertion", len(analysis_relations)),
        ("triple 기준으로 collapse한 분석용 edge", len(collapsed_relations)),
        (
            "collapsed edge의 추가 중복",
            int(collapsed_relations.duplicated(["subject_id", "predicate", "object_id"]).sum()),
        ),
    ],
    columns=["교정 단계", "개수"],
)
display(relation_correction_summary)

assert analysis_relations["final_type_valid"].all()
assert not (analysis_relations["subject_id"] == analysis_relations["object_id"]).any()
assert not collapsed_relations.duplicated(["subject_id", "predicate", "object_id"]).any()
"""
        ),
        markdown(
            """
여기서 제거되는 것은 원본 evidence가 아니라 **분석용 그래프에 포함할지 여부**다.
`relations_evidence.csv`는 그대로 보존하며, 분석용 assertion에는 최종 node type 기준으로 유효한
관계만 포함한다. 동일 triple의 여러 문장 근거는 `collapsed_relations`의 evidence 집계에 보존된다.
"""
        ),
        markdown("## 8. Sentence-local mention offset 교정"),
        code(
            """
corrected_mentions = correct_sentence_local_offsets(tables["mentions"], text_units)
offset_summary = (
    corrected_mentions["offset_status"]
    .value_counts(dropna=False)
    .rename_axis("교정 상태")
    .reset_index(name="mention 수")
)
display(offset_summary)
display(
    corrected_mentions.loc[
        ~corrected_mentions["offset_resolved"],
        ["entity_id", "mention_text", "text_unit_id", "sentence_text"],
    ]
)

resolved_mentions = corrected_mentions[corrected_mentions["offset_resolved"]]
resolved_span_text = resolved_mentions.apply(
    lambda row: row["sentence_text"][
        int(row["sentence_local_start"]):int(row["sentence_local_end"])
    ],
    axis=1,
)
resolved_span_matches = (
    resolved_span_text.str.casefold().to_numpy()
    == resolved_mentions["mention_text"].str.casefold().to_numpy()
).all()
assert resolved_span_matches
assert int((~corrected_mentions["offset_resolved"]).sum()) == 2
"""
        ),
        markdown(
            """
기사 전체 기준 `article_start_char`와 `article_end_char`는 수정하지 않는다. 별도
`sentence_local_start`·`sentence_local_end`를 추가했으며, 12,975개 mismatch를 자동 교정했다.
문장 하나의 범위를 넘어선 잘못된 mention 2개는 자동 교정하지 않고 `unresolved`로 격리한다.
Stage 3에서 정확한 span 강조가 필요할 때는 `offset_resolved=True`인 행만 사용한다.
"""
        ),
        markdown("## 9. 교정 후 구조 무결성 검사"),
        code(
            """
validation = validate_inputs(
    entities,
    analysis_relations,
    cooccurrence,
    schema=canonical_schema,
    policies=contract["policies"],
)

validation_labels = {
    "duplicate_entity_ids": "중복 entity ID 수",
    "duplicate_relation_ids": "중복 relation ID 수",
    "null_entity_keys": "필수 key가 비어 있는 entity 수",
    "null_relation_keys": "필수 key가 비어 있는 relation 수",
    "dangling_relation_subjects": "존재하지 않는 subject를 참조하는 relation 수",
    "dangling_relation_objects": "존재하지 않는 object를 참조하는 relation 수",
    "self_relations": "self-loop 수",
    "invalid_relation_signatures": "최종 schema signature 위반 수",
    "dangling_cooccurrence_endpoints": "존재하지 않는 endpoint를 참조하는 공동출현 edge 수",
}
validation_table = pd.DataFrame(
    [
        {"검사항목": validation_labels.get(key, key), "결과": value}
        for key, value in validation.items()
        if key not in {"errors", "warnings", "passed", "missing_relation_provenance"}
    ]
)
display(validation_table)
print("근거 정보가 없는 relation 수:", validation["missing_relation_provenance"])
print("경고:", validation["warnings"])
print("오류:", validation["errors"])

assert validation["passed"], validation["errors"]
"""
        ),
        markdown(
            """
교정 후에는 schema 불일치, self-loop와 triple 중복이 분석용 relation에 남지 않는다. 원본의
curated `wp1_lexicon` relation 35개 중 최종 type·self-loop 정책을 통과한 28개는 기사 근거가 없어
경고로 남는다. 이들은 source를 구분해 보존하되 기사 evidence가 필요한 Stage 3 답변에는 직접
사용하지 않는다.
"""
        ),
        markdown("## 10. 공식 schema의 현재 범위"),
        code(
            """
wp2_schema = json.loads((WP2_INPUT_DIR / "schema.json").read_text(encoding="utf-8"))
entity_types = wp2_schema["node"]["entity_types"]
relation_types = wp2_schema["edges"]["relation"]["types"]

schema_summary = pd.DataFrame(
    [
        ("엔티티 유형", len(entity_types)),
        ("관계 유형", len(relation_types)),
        ("노드", len(entities)),
        ("원본 관계 assertion", len(relations)),
        ("교정 후 분석용 관계 assertion", len(analysis_relations)),
        ("교정 후 collapsed relation", len(collapsed_relations)),
        ("공동출현 edge", len(cooccurrence)),
        ("text unit", len(text_units)),
    ],
    columns=["항목", "개수"],
)
display(schema_summary)

display(pd.DataFrame(
    [
        {
            "relation": name,
            "head_types": ", ".join(spec["head_types"]),
            "tail_types": ", ".join(spec["tail_types"]),
        }
        for name, spec in relation_types.items()
    ]
))
"""
        ),
        markdown(
            """
20개 entity type과 36개 relation type은 extraction recall을 위한 넓은 schema다. 과제에서 요구하는
compact topic-specific schema를 만들기 위해 다음 단계에서 원본 label을 보존하면서 분석용
canonical type과 predicate를 별도로 정의한다.
"""
        ),
        markdown("## 11. 교정된 실제 데이터 미리보기"),
        code(
            """
display(entities.sort_values("mention_count", ascending=False).head(15))
display(
    analysis_relations[
        [
            "subject_id", "predicate", "object_id", "document_id",
            "text_unit_id", "evidence_text", "published_at",
            "extraction_method", "rule_score",
        ]
    ].head(15)
)
display(cooccurrence.sort_values("sentence_count", ascending=False).head(15))
display(collapsed_relations.head(15))
"""
        ),
        markdown("## 12. 수동 relation 검수 준비 상태"),
        code(
            """
assert REVIEW_SAMPLE_PATH.exists(), f"검수 표본이 없습니다: {REVIEW_SAMPLE_PATH}"
review_sample = pd.read_csv(REVIEW_SAMPLE_PATH)

review_status = pd.DataFrame(
    [
        ("표본 행 수", len(review_sample)),
        ("표본에 포함된 관계 유형 수", review_sample["predicate"].nunique()),
        ("검수자 배정 수", review_sample["reviewer_assignment"].nunique()),
        ("검수가 완료된 행 수", review_sample["review_correct"].notna().sum()),
    ],
    columns=["항목", "개수"],
)
display(review_status)
display(review_sample["reviewer_assignment"].value_counts().rename("rows"))
"""
        ),
        markdown(
            """
검수 표본은 90개 unique GLiREL triple이며 세 명에게 30개씩 배정되어 있다. 현재는 검수 전 상태다.
팀원은 추출된 열을 수정하지 않고 `review_correct`, `reviewer`, `review_error_type`, `review_notes`만
작성한다.
"""
        ),
        markdown("## 13. WP3 1단계 완료 판정"),
        code(
            """
step1_checks = pd.DataFrame(
    [
        ("공식 WP1 파일이 모두 있음", handover_table.query("work_package == 'WP1'")["exists"].all()),
        ("공식 WP2 파일이 모두 있음", handover_table.query("work_package == 'WP2'")["exists"].all()),
        ("계약이 공식 WP2 파일을 올바르게 선택함", contract_report["passed"]),
        ("WP1과 WP2의 문서 ID 집합이 일치함", wp1_doc_ids == wp2_doc_ids),
        ("text-unit ID가 전체 데이터에서 고유함", text_units["text_unit_id"].is_unique),
        ("GLiREL 근거가 text unit에 연결됨", not (relation_text_units - known_text_units)),
        ("최종 node type 기준으로 모든 분석용 relation이 유효함", analysis_relations["final_type_valid"].all()),
        ("분석용 relation에 self-loop가 없음", not (analysis_relations["subject_id"] == analysis_relations["object_id"]).any()),
        ("collapsed relation에 중복 triple이 없음", not collapsed_relations.duplicated(["subject_id", "predicate", "object_id"]).any()),
        ("교정 가능한 sentence-local offset이 모두 일치함", resolved_span_matches),
        ("자동 교정 불가능한 mention 2개가 격리됨", int((~corrected_mentions["offset_resolved"]).sum()) == 2),
        ("교정 후 엔티티와 edge 구조가 유효함", validation["passed"]),
        ("수동 검수 표본 90개가 준비됨", len(review_sample) == 90),
    ],
    columns=["완료 기준", "통과"],
)
display(step1_checks)
assert step1_checks["통과"].all()
print("WP3 1단계 완료: WP1·WP2 원본을 수정하지 않고 공식 인계 산출물을 연결했습니다.")
"""
        ),
        markdown(
            """
## 14. 다음 단계

WP3 2단계에서는 다음 작업을 수행한다.

1. 20개 entity type과 36개 relation type을 과제에 맞는 compact schema로 매핑한다.
2. generic entity와 저빈도 noise 처리 기준을 확정한다.
3. raw assertion graph와 분석용 filtered graph의 출력 형식을 확정한다.
4. relation 수동 검수 결과를 반영할 수 있는 policy를 연결한다.
5. unresolved mention 2개는 정확한 span 강조 대상에서 제외한다.

그 이후에만 centrality, community, temporal analysis와 시각화를 계산한다. 이 순서를 지키면
파일럿 graph 결과와 팀원의 공식 산출물이 섞이는 것을 방지할 수 있다.
"""
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
