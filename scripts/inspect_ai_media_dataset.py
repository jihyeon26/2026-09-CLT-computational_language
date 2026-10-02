from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = Path(__file__).resolve().parents[1] / "dataset" / "ai_media_dataset_20260911.csv"


def compact(value: object, limit: int = 500) -> str | None:
    if pd.isna(value):
        return None
    return str(value)[:limit]


def parse_list(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


TRACK_PATTERNS = {
    "ai_governance": {
        "regulation_policy": r"\b(?:ai act|artificial intelligence act|regulat(?:e|ed|es|ing|ion|ions|ory)|legislat(?:e|ed|ion|ive)|executive order|public policy|compliance|governance)\b",
        "safety_security": r"\b(?:ai safety|artificial intelligence safety|model safety|ai security|red[ -]?team(?:ing)?|guardrail(?:s)?|alignment|catastrophic risk|existential risk)\b",
        "accountability_fairness": r"\b(?:accountab(?:ility|le)|algorithmic bias|ai bias|fairness|discrimination|responsible ai|ai ethics|algorithmic transparency|model transparency|ai audit|algorithmic audit)\b",
        "rights_risk": r"\b(?:privacy|data protection|copyright|intellectual property|risk management|human rights|deepfake regulation|content provenance)\b",
    },
    "open_models": {
        "openness": r"\b(?:open[ -]source(?:d)?|open[ -]weight(?:s|ed)?|open model(?:s)?|model weights|permissive licen[cs]e|apache 2(?:\.0)?|mit licen[cs]e)\b",
        "families": r"\b(?:llama ?[23]?|mixtral|mistral|gemma|qwen(?:2|2\.5|3)?|deepseek|falcon|bloom|olmo|phi[ -]?[234]|smollm|apertus)\b",
        "ecosystem": r"\b(?:hugging ?face|model hub|model card|weights available|downloadable weights)\b",
    },
    "agentic_web": {
        "agents": r"\b(?:agentic|ai agents?|artificial intelligence agents?|autonomous agents?|web agents?|browser agents?|multi[ -]agent(?: systems?)?)\b",
        "protocols": r"\b(?:model context protocol|\bmcp\b|agent2agent|agent[ -]to[ -]agent|\ba2a\b)\b",
        "computer_web": r"\b(?:computer use|browser use|browser automation|web browsing agent|web automation agent)\b",
        "tooling": r"\b(?:tool calling|function calling|tool use|agent framework|agent workflow|agent orchestration)\b",
    },
}


def print_ranked(label: str, values: pd.Series, n: int = 15) -> None:
    print(label)
    print(values.head(n).to_string())


def main() -> None:
    print("FILE", DATASET)
    print("FILE_BYTES", os.path.getsize(DATASET))

    df = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(DATASET, encoding=encoding, low_memory=False)
            print("ENCODING", encoding)
            break
        except UnicodeDecodeError as exc:
            print("FAILED_ENCODING", encoding, str(exc)[:180])
    if df is None:
        raise RuntimeError("Could not decode dataset")

    print("SHAPE", df.shape)
    print("COLUMNS", json.dumps(df.columns.tolist(), ensure_ascii=False))
    print("DTYPES")
    print(df.dtypes.to_string())
    print("HEAD")
    for index, row in df.head(3).iterrows():
        print(
            "ROW",
            index,
            json.dumps({key: compact(value) for key, value in row.items()}, ensure_ascii=False),
        )
    print("MISSING")
    print(df.isna().sum().sort_values(ascending=False).to_string())
    print("NUNIQUE")
    print(df.nunique(dropna=True).sort_values().to_string())
    print("EXACT_DUP_ROWS", int(df.duplicated().sum()))

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    print("INVALID_DATES", int(parsed_dates.isna().sum()))
    print("DATE_MIN", parsed_dates.min().date())
    print("DATE_MAX", parsed_dates.max().date())
    month_counts = parsed_dates.dt.to_period("M").value_counts().sort_index()
    print("MONTH_COUNTS")
    print(month_counts.to_string())

    print_ranked("TOP_DOMAINS", df["domain"].value_counts(), 20)
    print("DUPLICATE_CONTENT_ROWS", int(df.duplicated("content").sum()))
    print("DUPLICATE_CONTENT_GROUPS", int((df.groupby("content").size() > 1).sum()))
    print("DUPLICATE_NORMALIZED_URLS", int(df["url"].str.lower().str.rstrip("/").duplicated().sum()))

    title_chars = df["title"].str.len()
    content_chars = df["content"].str.len()
    for name, series in (("TITLE_CHARS", title_chars), ("CONTENT_SERIALIZED_CHARS", content_chars)):
        print(name, series.quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).round(0).astype(int).to_dict())

    tag_lists = df["tags"].map(parse_list)
    tag_counts = Counter(tag for tags in tag_lists for tag in tags)
    print("TAG_COUNT_TOTAL", sum(tag_counts.values()))
    print("TAG_VOCAB_SIZE", len(tag_counts))
    print("TOP_TAGS")
    for tag, count in tag_counts.most_common(40):
        print(f"{tag}\t{count}")

    searchable = (
        df["title"].fillna("").str.lower()
        + "\n"
        + df["content"].fillna("").str.lower()
        + "\n"
        + df["tags"].fillna("").str.lower()
    )
    title_text = df["title"].fillna("").str.lower()
    track_masks: dict[str, pd.Series] = {}
    print("TRACK_COVERAGE")
    for track, groups in TRACK_PATTERNS.items():
        group_masks = {
            group: searchable.str.contains(pattern, regex=True, na=False)
            for group, pattern in groups.items()
        }
        mask = pd.concat(group_masks, axis=1).any(axis=1)
        track_masks[track] = mask
        title_mask = pd.concat(
            [title_text.str.contains(pattern, regex=True, na=False) for pattern in groups.values()],
            axis=1,
        ).any(axis=1)
        print(track, "DOCS", int(mask.sum()), "PCT", round(mask.mean() * 100, 2), "TITLE_HITS", int(title_mask.sum()))
        print(track, "GROUP_HITS", json.dumps({group: int(group_mask.sum()) for group, group_mask in group_masks.items()}))
        per_month = parsed_dates[mask].dt.to_period("M").value_counts().sort_index()
        print(track, "MONTHLY", json.dumps({str(k): int(v) for k, v in per_month.items()}))
        print(track, "TOP_DOMAINS", json.dumps(df.loc[mask, "domain"].value_counts().head(12).to_dict()))
        track_tag_counts = Counter(tag for tags in tag_lists[mask] for tag in tags)
        print(track, "TOP_TAGS", json.dumps(dict(track_tag_counts.most_common(30)), ensure_ascii=False))
        samples = df.loc[mask, ["date", "title", "domain"]].copy()
        if len(samples) > 12:
            sample_positions = np.linspace(0, len(samples) - 1, 12, dtype=int)
            samples = samples.iloc[sample_positions]
        print(track, "SAMPLES")
        for record in samples.to_dict("records"):
            print(json.dumps(record, ensure_ascii=False))

    track_frame = pd.DataFrame(track_masks)
    track_count = track_frame.sum(axis=1)
    print("TRACK_OVERLAP_COUNT")
    print(track_count.value_counts().sort_index().to_string())
    combo = track_frame.apply(lambda row: "+".join(row.index[row].tolist()) or "none", axis=1)
    print("TRACK_COMBINATIONS")
    print(combo.value_counts().to_string())

    print("CONTENT_DUPLICATE_EXAMPLES")
    dup_content = df[df.duplicated("content", keep=False)].sort_values("content")
    for _, group in list(dup_content.groupby("content"))[:8]:
        print(json.dumps(group[["date", "title", "domain", "url"]].to_dict("records"), ensure_ascii=False))


if __name__ == "__main__":
    main()
