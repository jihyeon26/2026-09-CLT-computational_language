from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp2_open_models.review import review_summary_markdown, summarize_relation_reviews


def test_review_summary_excludes_ambiguous_from_precision() -> None:
    reviews = pd.DataFrame(
        [
            {"relation_id": "r1", "predicate": "DEVELOPED_BY", "review_correct": "TRUE", "reviewer": "A", "review_error_type": ""},
            {"relation_id": "r2", "predicate": "DEVELOPED_BY", "review_correct": "FALSE", "reviewer": "A", "review_error_type": "wrong_predicate"},
            {"relation_id": "r3", "predicate": "RELEASED_BY", "review_correct": "AMBIGUOUS", "reviewer": "B", "review_error_type": ""},
        ]
    )
    summary = summarize_relation_reviews(reviews, minimum_reviewed=2, minimum_precision=0.5)
    assert summary["status"] == "PASS"
    assert summary["counts"]["reviewed_for_precision"] == 2
    assert summary["counts"]["ambiguous"] == 1
    assert summary["precision"] == 0.5
    assert "wrong_predicate" in summary["error_type_counts"]
    assert "Predicate breakdown" in review_summary_markdown(summary)


def test_review_summary_blocks_incomplete_rejected_rows() -> None:
    reviews = pd.DataFrame(
        [
            {"relation_id": "r1", "predicate": "DEVELOPED_BY", "review_correct": "FALSE", "reviewer": "", "review_error_type": ""},
        ]
    )
    summary = summarize_relation_reviews(reviews, minimum_reviewed=1, minimum_precision=0.0)
    assert summary["status"] == "FAIL"
    assert "missing_reviewer" in summary["problems"]
    assert "rejected_rows_without_error_type" in summary["problems"]


def test_review_summary_reports_pending_when_sample_is_too_small() -> None:
    reviews = pd.DataFrame(
        [
            {"relation_id": "r1", "predicate": "DEVELOPED_BY", "review_correct": "TRUE", "reviewer": "A", "review_error_type": ""},
        ]
    )
    summary = summarize_relation_reviews(reviews, minimum_reviewed=50)
    assert summary["status"] == "PENDING"
    assert summary["pending"] == ["insufficient_reviewed_relations"]
