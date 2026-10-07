from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp3_knowledge_graph.contract import find_optional_json, load_contract_tables, read_contract
from wp3_knowledge_graph.pipeline import (
    PIPELINE_VERSION,
    annotate_final_relation_types,
    apply_compact_schema,
    apply_relation_policy,
    collapse_relations,
    correct_sentence_local_offsets,
    normalize_relation_schema,
    prepare_cooccurrence,
    prepare_entities,
    prepare_relations,
    validate_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the validated WP2-to-WP3 preparation snapshot before graph analysis."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp2_open_models_official_20261007",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "stage1_wp3_open_models_official_20261007" / "prepared",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "artifacts" / "stage1_wp3" / "preparation",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "config" / "wp3_open_models_contract.json",
    )
    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=ROOT / "config" / "wp3_compact_schema.json",
    )
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


def exclusion_reason(row: pd.Series, policies: dict[str, Any]) -> str:
    reasons: list[str] = []
    allowed_statuses = policies.get("allowed_assertion_statuses")
    if allowed_statuses and row["assertion_status"] not in allowed_statuses:
        reasons.append("assertion_status")
    if policies.get("drop_self_relations", True) and row["subject_id"] == row["object_id"]:
        reasons.append("self_loop")
    if policies.get("require_valid_final_signature", False) and not bool(
        row["final_type_valid"]
    ):
        reasons.append("invalid_final_signature")
    minimum_score = policies.get("minimum_rule_score")
    if minimum_score is not None:
        score = pd.to_numeric(row["rule_score"], errors="coerce")
        if pd.isna(score) or score < float(minimum_score):
            reasons.append("below_minimum_score")
    return "|".join(reasons) if reasons else "included"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    contract = read_contract(args.contract)
    compact_schema = json.loads(args.compact_schema.read_text(encoding="utf-8"))
    tables, contract_report = load_contract_tables(args.input_dir, contract)
    schema_path, upstream_schema = find_optional_json(
        args.input_dir, contract.get("schema_files", [])
    )
    canonical_schema = normalize_relation_schema(upstream_schema)

    entities = prepare_entities(tables["entities"])
    relations = prepare_relations(tables["relations"])
    cooccurrence = prepare_cooccurrence(tables.get("cooccurrence"))
    policies = contract.get("policies", {})

    raw_validation = validate_inputs(entities, relations, cooccurrence, None, policies)
    if not raw_validation["passed"]:
        raise ValueError(f"Raw structural validation failed: {raw_validation['errors']}")

    typed_relations = annotate_final_relation_types(entities, relations, canonical_schema)
    compact_entities, compact_candidates = apply_compact_schema(
        entities, typed_relations, compact_schema
    )
    included_relations, policy_audit = apply_relation_policy(
        compact_candidates, reviews=None, policies=policies
    )
    final_validation = validate_inputs(
        compact_entities, included_relations, cooccurrence, canonical_schema, policies
    )
    if not final_validation["passed"]:
        raise ValueError(f"Prepared relation validation failed: {final_validation['errors']}")

    decisions = compact_candidates.copy()
    decisions["exclusion_reason"] = decisions.apply(
        exclusion_reason, axis=1, policies=policies
    )
    included_ids = set(included_relations["relation_id"])
    decisions["included_in_wp3_candidate_graph"] = decisions["relation_id"].isin(
        included_ids
    )
    if not (
        decisions["included_in_wp3_candidate_graph"]
        == decisions["exclusion_reason"].eq("included")
    ).all():
        raise AssertionError("Policy decision table does not match the included relations.")

    collapsed = collapse_relations(included_relations, documents=None)
    collapsed["raw_predicate"] = collapsed["predicate"]
    collapsed["predicate_group"] = collapsed["raw_predicate"].map(
        compact_schema["predicate_group_mapping"]
    )
    if collapsed["predicate_group"].isna().any():
        raise AssertionError("Collapsed relations contain unmapped predicates.")

    corrected_mentions = None
    unresolved_mentions = pd.DataFrame()
    if tables.get("mentions") is not None and tables.get("text_units") is not None:
        corrected_mentions = correct_sentence_local_offsets(
            tables["mentions"], tables["text_units"]
        )
        unresolved_mentions = corrected_mentions[
            ~corrected_mentions["offset_resolved"]
        ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output_tables = {
        "nodes_prepared": compact_entities,
        "relation_policy_decisions": decisions,
        "typed_edges_assertions": included_relations,
        "typed_edges_collapsed": collapsed,
    }
    if corrected_mentions is not None:
        output_tables["entity_mentions_corrected"] = corrected_mentions

    output_files: dict[str, dict[str, Any]] = {}
    for name, table in output_tables.items():
        path = args.output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        output_files[name] = {
            "path": portable_path(path),
            "rows": int(len(table)),
            "sha256": sha256_file(path),
        }

    unresolved_path = args.artifact_dir / "unresolved_mentions.csv"
    unresolved_mentions.to_csv(unresolved_path, index=False, encoding="utf-8")

    policy_report = {
        **policy_audit,
        "excluded_relation_assertions": int(
            (~decisions["included_in_wp3_candidate_graph"]).sum()
        ),
        "exclusion_reasons": {
            str(name): int(count)
            for name, count in decisions.loc[
                ~decisions["included_in_wp3_candidate_graph"], "exclusion_reason"
            ].value_counts().sort_index().items()
        },
        "quality_tiers_in_candidate_graph": {
            str(name): int(count)
            for name, count in included_relations["quality_tier"]
            .value_counts()
            .sort_index()
            .items()
        },
        "human_validated_relations": 0,
        "ai_prescreen_used_as_exclusion_policy": False,
    }
    validation_report = {
        "raw_structural_validation": raw_validation,
        "prepared_candidate_validation": final_validation,
        "offset_status": (
            {
                str(name): int(count)
                for name, count in corrected_mentions["offset_status"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }
            if corrected_mentions is not None
            else None
        ),
    }
    write_json(args.artifact_dir / "policy_audit.json", policy_report)
    write_json(args.artifact_dir / "validation_report.json", validation_report)
    write_json(args.artifact_dir / "canonical_schema.json", canonical_schema)
    write_json(args.artifact_dir / "compact_schema_resolved.json", compact_schema)

    manifest = {
        "snapshot_version": "stage1-wp3-preparation-v0.1",
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "candidate_graph_only_pending_human_relation_review",
        "contract_version": contract["contract_version"],
        "compact_schema_version": compact_schema["schema_version"],
        "input_files": {
            name: {
                "path": portable_path(Path(details["path"])),
                "rows": details["rows"],
                "sha256": sha256_file(Path(details["path"])),
            }
            for name, details in contract_report["tables"].items()
        },
        "schema_input": (
            {
                "path": portable_path(schema_path),
                "sha256": sha256_file(schema_path),
            }
            if schema_path
            else None
        ),
        "output_files": output_files,
        "counts": {
            "nodes": int(len(compact_entities)),
            "raw_relation_assertions": int(len(relations)),
            "candidate_relation_assertions": int(len(included_relations)),
            "collapsed_candidate_edges": int(len(collapsed)),
            "unresolved_mentions": int(len(unresolved_mentions)),
        },
        "manual_review": {
            "required": True,
            "completed": False,
            "minimum_review_rows": 90,
            "minimum_precision": 0.80,
        },
    }
    write_json(args.artifact_dir / "preparation_manifest.json", manifest)

    dictionary = """# WP3 준비 스냅샷 데이터 사전

이 스냅샷은 공식 WP2 원본을 변경하지 않고 WP3 그래프 분석 전에 만든 파생 입력이다.

- `nodes_prepared.parquet`: WP2 원본 type과 compact 분석 type을 함께 보존한 node 테이블
- `relation_policy_decisions.parquet`: 모든 relation assertion과 포함 여부·제외 사유
- `typed_edges_assertions.parquet`: 구조 정책을 통과한 candidate relation assertions
- `typed_edges_collapsed.parquet`: 동일 raw triple을 합친 candidate graph edges
- `entity_mentions_corrected.parquet`: article offset을 보존하고 sentence-local offset을 추가한 mention
- `unresolved_mentions.csv`: 자동으로 보정할 수 없는 mention 검토 목록
- `policy_audit.json`: relation 제외 수와 candidate quality tier 분포
- `validation_report.json`: 교정 전후 구조 검사와 offset 보정 결과
- `canonical_schema.json`: WP2 schema를 WP3 signature 형식으로 변환한 결과
- `compact_schema_resolved.json`: 분석용 compact type·predicate group 매핑
- `preparation_manifest.json`: 입력·출력 해시, 버전과 완료 상태

`candidate`는 사람이 의미 정확성을 승인했다는 뜻이 아니다. 현재 AI 1차 스크리닝 결과는
edge 삭제 정책에 사용하지 않았으며, 검증된 최종 그래프에는 사람의 relation 검수가 필요하다.
"""
    (args.artifact_dir / "DATA_DICTIONARY_KO.md").write_text(
        dictionary, encoding="utf-8"
    )

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(policy_report, ensure_ascii=False, indent=2))
    print(f"Prepared data: {args.output_dir.resolve()}")
    print(f"Tracked audit artifacts: {args.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
