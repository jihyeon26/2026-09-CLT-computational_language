from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.quality_gates.gates import CheckResult, QualityReport, StageReport, report_to_markdown


def test_stage_status_uses_most_severe_check() -> None:
    stage = StageReport(
        "WP2",
        [
            CheckResult("WP2", "a", "PASS", "ok"),
            CheckResult("WP2", "b", "PENDING", "review"),
            CheckResult("WP2", "c", "WARN", "warning"),
        ],
    )
    assert stage.status == "PENDING"


def test_report_status_uses_most_severe_stage() -> None:
    report = QualityReport(
        "v1",
        "workspace",
        [
            StageReport("WP1", [CheckResult("WP1", "a", "FAIL", "missing")]),
            StageReport("WP2", [CheckResult("WP2", "b", "PASS", "ok")]),
        ],
    )
    assert report.status == "FAIL"
    assert report.to_dict()["overall_status"] == "FAIL"


def test_markdown_report_contains_actions() -> None:
    report = QualityReport(
        "v1",
        "workspace",
        [
            StageReport(
                "WP3",
                [CheckResult("WP3", "review", "PENDING", "pending", action="Review it")],
            )
        ],
    )
    markdown = report_to_markdown(report)
    assert "WP3: PENDING" in markdown
    assert "Review it" in markdown
