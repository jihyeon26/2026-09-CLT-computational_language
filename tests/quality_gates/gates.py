from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import nbformat
import networkx as nx
import pandas as pd
from PIL import Image


STATUS_PRIORITY = {"PASS": 0, "WARN": 1, "PENDING": 2, "FAIL": 3}


@dataclass(frozen=True)
class CheckResult:
    stage: str
    check_id: str
    status: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    action: str | None = None


@dataclass
class StageReport:
    stage: str
    checks: list[CheckResult]

    @property
    def status(self) -> str:
        if not self.checks:
            return "FAIL"
        return max((check.status for check in self.checks), key=STATUS_PRIORITY.get)

    @property
    def counts(self) -> dict[str, int]:
        return {
            status: sum(check.status == status for check in self.checks)
            for status in STATUS_PRIORITY
        }


@dataclass
class QualityReport:
    contract_version: str
    workspace: str
    stages: list[StageReport]

    @property
    def status(self) -> str:
        return max((stage.status for stage in self.stages), key=STATUS_PRIORITY.get)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "workspace": self.workspace,
            "overall_status": self.status,
            "stages": [
                {
                    "stage": stage.stage,
                    "status": stage.status,
                    "counts": stage.counts,
                    "checks": [asdict(check) for check in stage.checks],
                }
                for stage in self.stages
            ],
        }


def _result(
    stage: str,
    check_id: str,
    condition: bool,
    success: str,
    failure: str,
    *,
    evidence: dict[str, Any] | None = None,
    failure_status: str = "FAIL",
    action: str | None = None,
) -> CheckResult:
    return CheckResult(
        stage=stage,
        check_id=check_id,
        status="PASS" if condition else failure_status,
        message=success if condition else failure,
        evidence=evidence or {},
        action=None if condition else action,
    )


def _first_match(root: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _notebook_check(stage: str, root: Path, path: Path | None) -> list[CheckResult]:
    if path is None or not path.exists():
        return [
            CheckResult(
                stage,
                "notebook_exists",
                "FAIL",
                "실행 가능한 notebook을 찾지 못했습니다.",
                action="계약의 notebook 경로를 수정하거나 notebook을 생성하세요.",
            )
        ]
    try:
        notebook = nbformat.read(path, as_version=4)
    except Exception as exc:
        return [
            CheckResult(
                stage,
                "notebook_readable",
                "FAIL",
                f"Notebook을 읽을 수 없습니다: {exc}",
                evidence={"path": str(path.relative_to(root))},
            )
        ]

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    executed = [cell for cell in code_cells if cell.get("execution_count") is not None]
    errors = [
        {"cell": index, "ename": output.get("ename"), "evalue": output.get("evalue")}
        for index, cell in enumerate(notebook.cells)
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    relative = str(path.relative_to(root))
    return [
        _result(
            stage,
            "notebook_exists",
            True,
            "Notebook이 존재합니다.",
            "",
            evidence={"path": relative},
        ),
        _result(
            stage,
            "notebook_executed",
            len(code_cells) > 0 and len(executed) == len(code_cells),
            "모든 code cell이 실행된 상태입니다.",
            "실행되지 않은 code cell이 있습니다.",
            evidence={"code_cells": len(code_cells), "executed": len(executed)},
            action="깨끗한 kernel에서 notebook을 처음부터 끝까지 실행하세요.",
        ),
        _result(
            stage,
            "notebook_no_errors",
            not errors,
            "저장된 notebook output에 실행 오류가 없습니다.",
            "Notebook output에 실행 오류가 남아 있습니다.",
            evidence={"errors": errors},
            action="오류 셀을 수정하고 전체 notebook을 다시 실행하세요.",
        ),
    ]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _file_group_check(
    stage: str,
    check_id: str,
    root: Path,
    patterns: list[str],
    label: str,
) -> tuple[CheckResult, Path | None]:
    path = _first_match(root, patterns)
    check = _result(
        stage,
        check_id,
        path is not None,
        f"{label} 산출물을 찾았습니다.",
        f"{label} 산출물을 찾지 못했습니다.",
        evidence={"path": str(path.relative_to(root)) if path else None, "patterns": patterns},
        action=f"{label} 결과를 생성하거나 quality contract의 파일 패턴을 수정하세요.",
    )
    return check, path


def check_wp1(root: Path, config: dict[str, Any], manual_flags: dict[str, bool]) -> StageReport:
    stage = "WP1"
    checks: list[CheckResult] = []
    notebook = _first_match(root, config["notebook_candidates"])
    checks.extend(_notebook_check(stage, root, notebook))

    resolved: dict[str, Path] = {}
    labels = {
        "cleaned_documents": "정제 문서",
        "temporal_analysis": "시간 추이 분석",
        "topic_model": "토픽 모델링",
        "analysis_summary": "분석 요약/데이터 사전",
    }
    for name, patterns in config["artifact_groups"].items():
        check, path = _file_group_check(stage, f"artifact_{name}", root, patterns, labels[name])
        checks.append(check)
        if path:
            resolved[name] = path

    cleaned = resolved.get("cleaned_documents")
    if cleaned and cleaned.suffix.lower() in {".parquet", ".csv"}:
        try:
            frame = _read_table(cleaned)
            checks.append(
                _result(
                    stage,
                    "minimum_documents",
                    len(frame) >= int(config["minimum_documents"]),
                    "정제 문서 수가 최소 기준을 충족합니다.",
                    "정제 문서 수가 최소 기준보다 적습니다.",
                    evidence={"rows": len(frame), "minimum": config["minimum_documents"]},
                )
            )
            if "document_id" in frame.columns:
                duplicates = int(frame["document_id"].duplicated().sum())
                checks.append(
                    _result(
                        stage,
                        "document_ids_unique",
                        duplicates == 0,
                        "정제 문서 ID가 고유합니다.",
                        "정제 문서 ID에 중복이 있습니다.",
                        evidence={"duplicates": duplicates},
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        stage,
                        "document_ids_unique",
                        "WARN",
                        "정제 문서에 document_id가 없어 중복 ID 검사를 건너뛰었습니다.",
                    )
                )
        except Exception as exc:
            checks.append(CheckResult(stage, "cleaned_data_readable", "FAIL", f"정제 데이터를 읽을 수 없습니다: {exc}"))

    checks.append(
        _result(
            stage,
            "manual_track_insights",
            bool(manual_flags.get("wp1_track_insights_approved", False)),
            "Open Models 주요 추이와 topic 해석을 사람이 승인했습니다.",
            "Open Models 주요 추이와 topic 해석에 대한 수동 승인이 남아 있습니다.",
            failure_status="PENDING",
            action="시간 추이와 topic 결과를 검토하고 manual flag를 true로 변경하세요.",
        )
    )
    return StageReport(stage, checks)


def _required_files(stage: str, output_dir: Path, names: list[str]) -> list[CheckResult]:
    checks = []
    for name in names:
        path = output_dir / name
        checks.append(
            _result(
                stage,
                f"file_{name}",
                path.exists() and path.stat().st_size > 0,
                f"필수 파일 `{name}`이 존재합니다.",
                f"필수 파일 `{name}`이 없거나 비어 있습니다.",
                evidence={"path": str(path), "bytes": path.stat().st_size if path.exists() else 0},
                action="해당 stage 파이프라인을 재실행하세요.",
            )
        )
    return checks


def _truthy_review(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "correct", "accepted"}:
        return True
    if normalized in {"false", "0", "no", "n", "incorrect", "rejected"}:
        return False
    return None


def check_wp2(root: Path, config: dict[str, Any], manual_flags: dict[str, bool]) -> StageReport:
    stage = "WP2"
    output_dir = root / config["output_dir"]
    checks = _required_files(stage, output_dir, config["required_files"])
    checks.extend(_notebook_check(stage, root, root / config["notebook"]))

    core_paths = {
        "documents": output_dir / "documents.parquet",
        "mentions": output_dir / "entity_mentions.parquet",
        "entities": output_dir / "entities.parquet",
        "relations": output_dir / "relations.parquet",
        "cooccurrence": output_dir / "cooccurrence_edges.parquet",
    }
    if all(path.exists() for path in core_paths.values()):
        documents = pd.read_parquet(core_paths["documents"])
        mentions = pd.read_parquet(core_paths["mentions"])
        entities = pd.read_parquet(core_paths["entities"])
        relations = pd.read_parquet(core_paths["relations"])
        cooccurrence = pd.read_parquet(core_paths["cooccurrence"])
        entity_ids = set(entities["entity_id"])

        for name, count, minimum in [
            ("entities", len(entities), config["minimum_entities"]),
            ("relations", len(relations), config["minimum_relations"]),
            ("cooccurrence_edges", len(cooccurrence), config["minimum_cooccurrence_edges"]),
        ]:
            checks.append(
                _result(
                    stage,
                    f"minimum_{name}",
                    count >= int(minimum),
                    f"{name} 수가 최소 기준을 충족합니다.",
                    f"{name} 수가 최소 기준보다 적습니다.",
                    evidence={"count": count, "minimum": minimum},
                )
            )

        duplicate_entities = int(entities["entity_id"].duplicated().sum())
        duplicate_relations = int(relations["relation_id"].duplicated().sum())
        dangling = int(
            (~relations["subject_id"].isin(entity_ids) | ~relations["object_id"].isin(entity_ids)).sum()
        )
        checks.extend(
            [
                _result(stage, "unique_entity_ids", duplicate_entities == 0, "Entity ID가 고유합니다.", "Entity ID 중복이 있습니다.", evidence={"duplicates": duplicate_entities}),
                _result(stage, "unique_relation_ids", duplicate_relations == 0, "Relation ID가 고유합니다.", "Relation ID 중복이 있습니다.", evidence={"duplicates": duplicate_relations}),
                _result(stage, "relation_endpoints", dangling == 0, "모든 relation endpoint가 존재합니다.", "존재하지 않는 node를 가리키는 relation이 있습니다.", evidence={"dangling": dangling}),
            ]
        )

        text_by_document = documents.set_index("document_id")["text_clean"].to_dict()
        offset_failures = 0
        required_offset_columns = {"document_id", "char_start", "char_end", "surface_form"}
        if required_offset_columns.issubset(mentions.columns):
            for row in mentions.itertuples(index=False):
                text = text_by_document.get(row.document_id)
                if text is None or text[row.char_start : row.char_end] != row.surface_form:
                    offset_failures += 1
            checks.append(
                _result(stage, "mention_offsets", offset_failures == 0, "Mention offset이 원문과 일치합니다.", "원문과 일치하지 않는 mention offset이 있습니다.", evidence={"failures": offset_failures})
            )

        provenance_fields = ["document_id", "text_unit_id", "evidence_text"]
        missing_provenance = {
            field: int((relations[field].isna() | relations[field].astype("string").str.strip().eq("")).sum())
            if field in relations.columns else len(relations)
            for field in provenance_fields
        }
        checks.append(
            _result(stage, "relation_provenance", not any(missing_provenance.values()), "모든 relation에 document, text unit, evidence가 연결되어 있습니다.", "Relation provenance가 누락되었습니다.", evidence=missing_provenance)
        )

    qa_path = output_dir / "qa_report.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        checks.extend(
            [
                _result(stage, "upstream_hard_validation", bool(qa.get("hard_validation_passed")), "WP2 hard validation이 통과했습니다.", "WP2 hard validation이 실패했습니다.", evidence={"hard_validation_passed": qa.get("hard_validation_passed")}),
                _result(stage, "mention_coverage", float(qa.get("document_mention_coverage_pct", 0)) >= float(config["minimum_mention_document_coverage_pct"]), "Mention document coverage가 최소 기준을 충족합니다.", "Mention document coverage가 최소 기준보다 낮습니다.", evidence={"actual": qa.get("document_mention_coverage_pct"), "minimum": config["minimum_mention_document_coverage_pct"]}),
                _result(stage, "relation_coverage", float(qa.get("document_relation_coverage_pct", 0)) >= float(config["minimum_relation_document_coverage_pct"]), "Relation document coverage가 최소 기준을 충족합니다.", "Relation document coverage가 최소 기준보다 낮습니다.", evidence={"actual": qa.get("document_relation_coverage_pct"), "minimum": config["minimum_relation_document_coverage_pct"]}),
            ]
        )

    review_path = output_dir / "relation_review_sample.csv"
    if review_path.exists():
        reviews = pd.read_csv(review_path)
        required_review_columns = {
            "review_correct", "reviewer", "review_error_type", "review_notes"
        }
        missing_review_columns = sorted(required_review_columns - set(reviews.columns))
        checks.append(
            _result(
                stage,
                "relation_review_schema",
                not missing_review_columns,
                "Relation review sheet에 판정·검수자·오류 유형·메모 열이 있습니다.",
                "Relation review sheet의 검수 열이 불완전합니다.",
                evidence={"missing": missing_review_columns},
            )
        )
        judged = reviews["review_correct"].map(_truthy_review) if "review_correct" in reviews.columns else pd.Series(dtype=object)
        reviewed = int(judged.notna().sum())
        correct = int((judged == True).sum())  # noqa: E712
        precision = correct / reviewed if reviewed else None
        raw_labels = reviews.get("review_correct", pd.Series("", index=reviews.index)).fillna("").astype(str).str.strip().str.upper()
        labeled = raw_labels.ne("")
        valid_labels = {"TRUE", "FALSE", "AMBIGUOUS", "YES", "NO", "Y", "N", "1", "0", "CORRECT", "INCORRECT", "ACCEPT", "ACCEPTED", "REJECT", "REJECTED", "UNSURE", "UNCLEAR"}
        invalid_labels = int((labeled & ~raw_labels.isin(valid_labels)).sum())
        reviewer = reviews.get("reviewer", pd.Series("", index=reviews.index)).fillna("").astype(str).str.strip()
        missing_reviewers = int((labeled & reviewer.eq("")).sum())
        error_type = reviews.get("review_error_type", pd.Series("", index=reviews.index)).fillna("").astype(str).str.strip()
        rejected_without_error = int((raw_labels.isin({"FALSE", "NO", "N", "0", "INCORRECT", "REJECT", "REJECTED"}) & error_type.eq("")).sum())
        checks.append(
            _result(
                stage,
                "relation_review_metadata",
                invalid_labels == 0 and missing_reviewers == 0 and rejected_without_error == 0,
                "작성된 relation review의 label과 검수 metadata가 완전합니다.",
                "작성된 relation review에 잘못된 label 또는 누락된 검수 metadata가 있습니다.",
                evidence={
                    "invalid_labels": invalid_labels,
                    "missing_reviewers": missing_reviewers,
                    "rejected_without_error_type": rejected_without_error,
                },
                action="가이드에 따라 reviewer와 FALSE 행의 review_error_type을 채우세요.",
            )
        )
        checks.append(
            _result(
                stage,
                "manual_relation_review_count",
                reviewed >= int(config["minimum_reviewed_relations"]),
                "수동 relation 검수 표본 수가 최소 기준을 충족합니다.",
                "수동 relation 검수가 충분하지 않습니다.",
                evidence={"reviewed": reviewed, "minimum": config["minimum_reviewed_relations"]},
                failure_status="PENDING",
                action="relation_review_sample.csv의 review_correct, reviewer, review_notes를 채우세요.",
            )
        )
        if precision is not None:
            checks.append(
                _result(
                    stage,
                    "manual_relation_precision",
                    precision >= float(config["minimum_review_precision"]),
                    "수동 검수 precision이 최소 기준을 충족합니다.",
                    "수동 검수 precision이 최소 기준보다 낮습니다.",
                    evidence={"precision": precision, "minimum": config["minimum_review_precision"]},
                    action="오탐 유형을 분석해 extraction rule 또는 catalogue를 수정하세요.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    stage,
                    "manual_relation_precision",
                    "PENDING",
                    "검수된 relation이 없어 precision을 계산할 수 없습니다.",
                    action="최소 검수 표본을 채운 후 다시 실행하세요.",
                )
            )

    checks.append(
        _result(
            stage,
            "manual_review_approval",
            bool(manual_flags.get("wp2_relation_review_approved", False)),
            "팀이 relation review 결과를 승인했습니다.",
            "팀의 relation review 최종 승인이 남아 있습니다.",
            failure_status="PENDING",
            action="검수 결과에 합의한 뒤 manual flag를 true로 변경하세요.",
        )
    )
    return StageReport(stage, checks)


def check_wp3(root: Path, config: dict[str, Any], manual_flags: dict[str, bool]) -> StageReport:
    stage = "WP3"
    output_dir = root / config["output_dir"]
    upstream_dir = root / config["upstream_wp2_dir"]
    checks = _required_files(stage, output_dir, config["required_files"])
    checks.extend(_notebook_check(stage, root, root / config["notebook"]))

    validation_path = output_dir / "validation_report.json"
    contract_report_path = output_dir / "contract_report.json"
    summary_path = output_dir / "graph_summary.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        checks.append(
            _result(stage, "wp3_validation", bool(validation.get("passed")), "WP3 contract/schema/integrity validation이 통과했습니다.", "WP3 validation이 실패했습니다.", evidence={"errors": validation.get("errors"), "warnings": validation.get("warnings")})
        )
    if contract_report_path.exists():
        contract_report = json.loads(contract_report_path.read_text(encoding="utf-8"))
        checks.append(
            _result(stage, "contract_adapter", bool(contract_report.get("passed")), "WP2→WP3 contract adapter가 모든 입력을 해석했습니다.", "Contract adapter가 입력을 해석하지 못했습니다.", evidence={"errors": contract_report.get("errors"), "warnings": contract_report.get("warnings")})
        )
        contract_version = str(contract_report.get("contract_version", ""))
        frozen = "draft" not in contract_version.lower()
        checks.append(
            _result(
                stage,
                "official_contract_frozen",
                frozen or not config.get("require_frozen_contract_for_final", True),
                "공식 WP2→WP3 contract version이 고정되었습니다.",
                "현재 draft contract를 사용하고 있어 최종 수치로 확정할 수 없습니다.",
                evidence={"contract_version": contract_version},
                failure_status="PENDING",
                action="공식 WP2 산출물 형식이 오면 contract version과 strict provenance를 확정하세요.",
            )
        )

    table_paths = {
        "nodes": output_dir / "nodes.parquet",
        "assertions": output_dir / "typed_edges_assertions.parquet",
        "collapsed": output_dir / "typed_edges_collapsed.parquet",
        "cooccurrence": output_dir / "cooccurrence_edges.parquet",
        "metrics": output_dir / "node_metrics.parquet",
    }
    if all(path.exists() for path in table_paths.values()):
        nodes = pd.read_parquet(table_paths["nodes"])
        assertions = pd.read_parquet(table_paths["assertions"])
        collapsed = pd.read_parquet(table_paths["collapsed"])
        cooccurrence = pd.read_parquet(table_paths["cooccurrence"])
        metrics = pd.read_parquet(table_paths["metrics"])
        node_ids = set(nodes["entity_id"])
        dangling = int((~assertions["subject_id"].isin(node_ids) | ~assertions["object_id"].isin(node_ids)).sum())
        duplicate_triples = int(collapsed[["subject_id", "predicate", "object_id"]].duplicated().sum())
        checks.extend(
            [
                _result(stage, "assertion_endpoints", dangling == 0, "Assertion graph endpoint가 모두 존재합니다.", "Assertion graph에 dangling endpoint가 있습니다.", evidence={"dangling": dangling}),
                _result(stage, "collapsed_unique_triples", duplicate_triples == 0, "Collapsed typed edge가 triple별로 고유합니다.", "Collapsed typed edge에 중복 triple이 있습니다.", evidence={"duplicates": duplicate_triples}),
                _result(stage, "minimum_typed_edges", len(collapsed) >= int(config["minimum_typed_edges"]), "Typed graph edge 수가 최소 기준을 충족합니다.", "Typed graph edge 수가 최소 기준보다 적습니다.", evidence={"count": len(collapsed), "minimum": config["minimum_typed_edges"]}),
                _result(stage, "minimum_cooccurrence_edges", len(cooccurrence) >= int(config["minimum_cooccurrence_edges"]), "Co-occurrence edge 수가 최소 기준을 충족합니다.", "Co-occurrence edge 수가 최소 기준보다 적습니다.", evidence={"count": len(cooccurrence), "minimum": config["minimum_cooccurrence_edges"]}),
            ]
        )
        views = set(metrics["graph_view"].dropna()) if "graph_view" in metrics.columns else set()
        missing_views = sorted(set(config["required_metric_views"]) - views)
        metric_columns = {"degree", "weighted_degree", "pagerank", "betweenness", "clustering"}
        missing_metric_columns = sorted(metric_columns - set(metrics.columns))
        checks.extend(
            [
                _result(stage, "metric_views", not missing_views, "필수 graph view별 metric이 존재합니다.", "일부 graph view의 metric이 없습니다.", evidence={"views": sorted(views), "missing": missing_views}),
                _result(stage, "metric_columns", not missing_metric_columns, "필수 network metric 열이 존재합니다.", "필수 network metric 열이 누락됐습니다.", evidence={"missing": missing_metric_columns}),
            ]
        )

        relation_summary_path = output_dir / "relation_type_summary.parquet"
        monthly_assertions_path = output_dir / "monthly_relation_assertions.parquet"
        communities_path = output_dir / "node_communities.parquet"
        community_summary_path = output_dir / "community_summary.parquet"
        candidates_path = output_dir / "graph_finding_candidates.parquet"
        relation_summary = pd.read_parquet(relation_summary_path) if relation_summary_path.exists() else pd.DataFrame()
        monthly_assertions = pd.read_parquet(monthly_assertions_path) if monthly_assertions_path.exists() else pd.DataFrame()
        communities = pd.read_parquet(communities_path) if communities_path.exists() else pd.DataFrame()
        community_summary = pd.read_parquet(community_summary_path) if community_summary_path.exists() else pd.DataFrame()
        candidates = pd.read_parquet(candidates_path) if candidates_path.exists() else pd.DataFrame()
        predicates = set(collapsed["predicate"].dropna().astype(str))
        summary_predicates = set(relation_summary["predicate"].dropna().astype(str)) if "predicate" in relation_summary else set()
        predicate_metric_views = {
            str(view).split(":", 1)[1]
            for view in views
            if str(view).startswith("predicate:")
        }
        cooccurrence_nodes = set(cooccurrence["source_id"]) | set(cooccurrence["target_id"])
        community_nodes = set(communities["entity_id"]) if "entity_id" in communities else set()
        required_candidate_columns = {
            "candidate_type", "entity_or_pair", "metric_name", "metric_value",
            "supporting_ids", "review_status",
        }
        checks.extend(
            [
                _result(stage, "relation_summary_coverage", predicates == summary_predicates, "모든 predicate가 관계 유형 요약에 포함됩니다.", "관계 유형 요약에서 일부 predicate가 빠졌습니다.", evidence={"missing": sorted(predicates - summary_predicates)}),
                _result(stage, "predicate_metric_views", predicates == predicate_metric_views, "모든 predicate에 독립적인 network metric이 있습니다.", "일부 predicate별 metric이 없습니다.", evidence={"missing": sorted(predicates - predicate_metric_views)}),
                _result(stage, "temporal_analysis", not monthly_assertions.empty and "month" in monthly_assertions and monthly_assertions["month"].notna().all(), "월별 relation 변화 표가 생성되었습니다.", "날짜가 있는 relation의 월별 변화 표가 비어 있습니다.", evidence={"rows": len(monthly_assertions)}),
                _result(stage, "community_membership", cooccurrence_nodes == community_nodes and len(community_summary) > 0, "모든 co-occurrence node에 community가 할당되었습니다.", "Community 할당이 누락되었거나 비어 있습니다.", evidence={"missing_nodes": len(cooccurrence_nodes - community_nodes), "communities": len(community_summary)}),
                _result(stage, "finding_candidate_schema", not candidates.empty and required_candidate_columns.issubset(candidates.columns), "근거 검토용 finding candidate가 생성되었습니다.", "Finding candidate가 없거나 필수 열이 누락됐습니다.", evidence={"count": len(candidates), "missing_columns": sorted(required_candidate_columns - set(candidates.columns))}),
            ]
        )

        upstream_documents = upstream_dir / "documents.parquet"
        upstream_text_units = upstream_dir / "text_units.parquet"
        if upstream_documents.exists() and upstream_text_units.exists():
            documents = pd.read_parquet(upstream_documents)
            text_units = pd.read_parquet(upstream_text_units)
            missing_docs = int((~assertions["document_id"].isin(set(documents["document_id"]))).sum())
            missing_text_units = int((~assertions["text_unit_id"].isin(set(text_units["text_unit_id"]))).sum())
            checks.append(
                _result(stage, "rag_provenance_chain", missing_docs == 0 and missing_text_units == 0, "모든 graph assertion이 WP2 document와 text unit으로 역추적됩니다.", "일부 graph assertion의 document/text unit 연결이 끊어졌습니다.", evidence={"missing_documents": missing_docs, "missing_text_units": missing_text_units})
            )
        else:
            checks.append(CheckResult(stage, "rag_provenance_chain", "PENDING", "WP2 document/text unit이 없어 Stage 3 provenance chain을 검사할 수 없습니다."))

    graph_files = [
        output_dir / "assertion_graph.graphml",
        output_dir / "typed_graph_collapsed.graphml",
        output_dir / "cooccurrence_graph.graphml",
    ]
    graph_errors = []
    for path in graph_files:
        try:
            nx.read_graphml(path)
        except Exception as exc:
            graph_errors.append({"file": path.name, "error": str(exc)})
    checks.append(
        _result(stage, "graphml_reload", not graph_errors, "모든 GraphML 파일을 다시 로드할 수 있습니다.", "일부 GraphML 파일을 다시 로드할 수 없습니다.", evidence={"errors": graph_errors})
    )

    graph_schema_path = output_dir / "graph_schema.json"
    if graph_schema_path.exists():
        graph_schema = json.loads(graph_schema_path.read_text(encoding="utf-8"))
        expected_views = {"assertion", "collapsed_typed", "cooccurrence"}
        actual_views = set(graph_schema.get("graph_views", {}))
        provenance = set(graph_schema.get("provenance_chain", []))
        checks.extend(
            [
                _result(stage, "graph_schema_views", expected_views.issubset(actual_views), "세 graph view의 의미가 schema에 명시됐습니다.", "Graph schema에 필수 view가 누락됐습니다.", evidence={"missing": sorted(expected_views - actual_views)}),
                _result(stage, "graph_schema_provenance", {"relation_id", "text_unit_id", "document_id"}.issubset(provenance), "Stage 3용 provenance chain이 schema에 명시됐습니다.", "Graph schema의 Stage 3 provenance chain이 불완전합니다."),
            ]
        )

    figures = sorted((output_dir / "figures").glob("*.png")) if (output_dir / "figures").exists() else []
    invalid_figures = []
    for path in figures:
        try:
            with Image.open(path) as image:
                if image.width < 640 or image.height < 480:
                    invalid_figures.append({"file": path.name, "size": [image.width, image.height]})
                image.verify()
        except Exception as exc:
            invalid_figures.append({"file": path.name, "error": str(exc)})
    checks.extend(
        [
            _result(stage, "minimum_figures", len(figures) >= int(config["minimum_figures"]), "시각화 수가 최소 기준을 충족합니다.", "시각화 수가 최소 기준보다 적습니다.", evidence={"count": len(figures), "minimum": config["minimum_figures"]}),
            _result(stage, "figures_readable", not invalid_figures, "모든 PNG 시각화가 읽을 수 있는 크기와 형식입니다.", "읽을 수 없거나 너무 작은 시각화가 있습니다.", evidence={"invalid": invalid_figures}),
        ]
    )

    for flag, check_id, message, action in [
        ("wp3_visualizations_approved", "manual_visual_review", "팀의 시각화 수동 검수가 남아 있습니다.", "label 겹침, legend, threshold를 확인한 뒤 manual flag를 true로 변경하세요."),
        ("wp3_graph_insights_approved", "manual_graph_insights", "Evidence 기반 graph insight의 수동 승인이 남아 있습니다.", "metric과 source evidence를 함께 검토한 뒤 manual flag를 true로 변경하세요."),
    ]:
        checks.append(
            _result(stage, check_id, bool(manual_flags.get(flag, False)), message.replace("남아 있습니다", "완료됐습니다"), message, failure_status="PENDING", action=action)
        )
    return StageReport(stage, checks)


def run_all_gates(root: Path, contract_path: Path) -> QualityReport:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    flags = contract.get("manual_flags", {})
    stages = [
        check_wp1(root, contract["wp1"], flags),
        check_wp2(root, contract["wp2"], flags),
        check_wp3(root, contract["wp3"], flags),
    ]
    return QualityReport(
        contract_version=contract["contract_version"],
        workspace=str(root.resolve()),
        stages=stages,
    )


def report_to_markdown(report: QualityReport) -> str:
    lines = [
        "# Stage 1 WP1/WP2/WP3 quality gate report",
        "",
        f"- Contract: `{report.contract_version}`",
        f"- Overall status: **{report.status}**",
        "",
        "## Summary",
        "",
        "| Work package | Status | PASS | WARN | PENDING | FAIL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in report.stages:
        counts = stage.counts
        lines.append(
            f"| {stage.stage} | {stage.status} | {counts['PASS']} | {counts['WARN']} | {counts['PENDING']} | {counts['FAIL']} |"
        )
    for stage in report.stages:
        lines.extend(["", f"## {stage.stage}: {stage.status}", ""])
        for check in stage.checks:
            lines.append(f"- **[{check.status}] {check.check_id}** — {check.message}")
            if check.evidence:
                lines.append(f"  - Evidence: `{json.dumps(check.evidence, ensure_ascii=False)}`")
            if check.action:
                lines.append(f"  - Next action: {check.action}")
    lines.append("")
    return "\n".join(lines)
