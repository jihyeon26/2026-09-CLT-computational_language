from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


ACCEPTED_REVIEW_LABELS = {"TRUE", "FALSE", "AMBIGUOUS"}
REVIEW_ERROR_TYPES = {
    "entity_boundary",
    "entity_linking",
    "wrong_predicate",
    "reversed_direction",
    "negation_or_uncertainty",
    "insufficient_context",
    "duplicate_or_noise",
    "schema_or_catalogue",
    "other",
}


def normalize_review_label(value: Any) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    normalized = str(value).strip().upper()
    aliases = {
        "1": "TRUE",
        "YES": "TRUE",
        "Y": "TRUE",
        "CORRECT": "TRUE",
        "ACCEPT": "TRUE",
        "ACCEPTED": "TRUE",
        "0": "FALSE",
        "NO": "FALSE",
        "N": "FALSE",
        "INCORRECT": "FALSE",
        "REJECT": "FALSE",
        "REJECTED": "FALSE",
        "UNSURE": "AMBIGUOUS",
        "UNCLEAR": "AMBIGUOUS",
    }
    return aliases.get(normalized, normalized)


def summarize_relation_reviews(
    reviews: pd.DataFrame,
    minimum_reviewed: int = 90,
    minimum_precision: float = 0.80,
) -> dict[str, Any]:
    required = {"relation_id", "predicate", "review_correct"}
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"Missing required review columns: {missing}")

    frame = reviews.copy()
    frame["normalized_review"] = frame["review_correct"].map(normalize_review_label)
    invalid_mask = frame["normalized_review"].notna() & ~frame["normalized_review"].isin(
        ACCEPTED_REVIEW_LABELS
    )
    labeled_mask = frame["normalized_review"].isin(ACCEPTED_REVIEW_LABELS)
    judged_mask = frame["normalized_review"].isin({"TRUE", "FALSE"})
    accepted = int((frame["normalized_review"] == "TRUE").sum())
    rejected = int((frame["normalized_review"] == "FALSE").sum())
    ambiguous = int((frame["normalized_review"] == "AMBIGUOUS").sum())
    reviewed = accepted + rejected
    precision = accepted / reviewed if reviewed else None

    reviewer = frame.get("reviewer", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    missing_reviewer = int((labeled_mask & reviewer.eq("")).sum())
    error_type = (
        frame.get("review_error_type", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    rejected_mask = frame["normalized_review"] == "FALSE"
    rejected_missing_error_type = int((rejected_mask & error_type.eq("")).sum())
    invalid_error_types = sorted(
        set(error_type[error_type.ne("")]) - REVIEW_ERROR_TYPES
    )

    by_predicate = []
    for predicate, group in frame.groupby("predicate", dropna=False):
        labels = group["normalized_review"]
        predicate_accepted = int((labels == "TRUE").sum())
        predicate_rejected = int((labels == "FALSE").sum())
        predicate_reviewed = predicate_accepted + predicate_rejected
        by_predicate.append(
            {
                "predicate": str(predicate),
                "sampled": int(len(group)),
                "accepted": predicate_accepted,
                "rejected": predicate_rejected,
                "ambiguous": int((labels == "AMBIGUOUS").sum()),
                "unreviewed": int(labels.isna().sum()),
                "precision": (
                    predicate_accepted / predicate_reviewed if predicate_reviewed else None
                ),
            }
        )

    error_counts = Counter(error_type[rejected_mask & error_type.ne("")])
    problems = []
    pending = []
    if invalid_mask.any():
        problems.append("invalid_review_labels")
    if invalid_error_types:
        problems.append("invalid_error_types")
    if missing_reviewer:
        problems.append("missing_reviewer")
    if rejected_missing_error_type:
        problems.append("rejected_rows_without_error_type")
    if precision is not None and precision < minimum_precision:
        problems.append("precision_below_threshold")
    if reviewed < minimum_reviewed:
        pending.append("insufficient_reviewed_relations")
    if precision is None:
        pending.append("precision_unavailable")

    status = "FAIL" if problems else ("PENDING" if pending else "PASS")
    return {
        "status": status,
        "thresholds": {
            "minimum_reviewed": int(minimum_reviewed),
            "minimum_precision": float(minimum_precision),
        },
        "counts": {
            "sampled": int(len(frame)),
            "labeled": int(labeled_mask.sum()),
            "reviewed_for_precision": reviewed,
            "accepted": accepted,
            "rejected": rejected,
            "ambiguous": ambiguous,
            "unreviewed": int(frame["normalized_review"].isna().sum()),
            "invalid_labels": int(invalid_mask.sum()),
            "missing_reviewer": missing_reviewer,
            "rejected_missing_error_type": rejected_missing_error_type,
        },
        "precision": precision,
        "coverage": float(labeled_mask.mean()) if len(frame) else 0.0,
        "invalid_error_types": invalid_error_types,
        "error_type_counts": dict(sorted(error_counts.items())),
        "by_predicate": sorted(by_predicate, key=lambda row: row["predicate"]),
        "problems": problems,
        "pending": pending,
    }


def review_summary_markdown(summary: dict[str, Any]) -> str:
    precision = summary["precision"]
    precision_text = "N/A" if precision is None else f"{precision:.3f}"
    counts = summary["counts"]
    lines = [
        "# WP2 relation review report",
        "",
        f"- Status: **{summary['status']}**",
        f"- Precision: **{precision_text}**",
        f"- Reviewed for precision: **{counts['reviewed_for_precision']}**",
        f"- Ambiguous: **{counts['ambiguous']}**",
        f"- Unreviewed: **{counts['unreviewed']}**",
        "",
        "## Predicate breakdown",
        "",
        "| Predicate | Sampled | Accepted | Rejected | Ambiguous | Unreviewed | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_predicate"]:
        value = "N/A" if row["precision"] is None else f"{row['precision']:.3f}"
        lines.append(
            f"| {row['predicate']} | {row['sampled']} | {row['accepted']} | "
            f"{row['rejected']} | {row['ambiguous']} | {row['unreviewed']} | {value} |"
        )
    lines.extend(["", "## Error taxonomy", ""])
    if summary["error_type_counts"]:
        for name, count in summary["error_type_counts"].items():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- No rejected rows with an error type yet.")
    lines.extend(["", "## Blocking issues", ""])
    issues = summary["problems"] + summary["pending"]
    lines.extend([f"- `{issue}`" for issue in issues] or ["- None"])
    return "\n".join(lines) + "\n"
