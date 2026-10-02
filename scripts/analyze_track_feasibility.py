from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


DATASET = Path(__file__).resolve().parents[1] / "dataset" / "ai_media_dataset_20260911.csv"

TRACKS = {
    "ai_governance": {
        "tags": {
            "accountability",
            "fairness",
            "aisafety",
            "aiethics",
            "responsibleai",
            "aigovernance",
            "airegulation",
        },
        "specific": [
            r"\bai governance\b",
            r"\bartificial intelligence governance\b",
            r"\bai regulat(?:ion|ions|ory|ed|ing)\b",
            r"\bartificial intelligence regulat(?:ion|ions|ory|ed|ing)\b",
            r"\bai act\b",
            r"\bartificial intelligence act\b",
            r"\bresponsible ai\b",
            r"\bai ethics\b",
            r"\bai safety\b",
            r"\bmodel safety\b",
            r"\bai security\b",
            r"\bai accountab(?:ility|le)\b",
            r"\balgorithmic accountab(?:ility|le)\b",
            r"\balgorithmic bias\b",
            r"\bai bias\b",
            r"\balgorithmic fairness\b",
            r"\bai fairness\b",
            r"\balgorithmic transparency\b",
            r"\bai transparency\b",
            r"\bai audit(?:ing|s)?\b",
            r"\bred[ -]?team(?:ing)?\b",
        ],
    },
    "open_models": {
        "tags": {
            "opensource",
            "openmodel",
            "openweights",
            "deepseek",
            "huggingface",
            "llama",
            "mistral",
            "mixtral",
            "qwen",
            "gemma",
            "falcon",
            "bloom",
            "olmo",
            "phi",
            "smollm",
            "apertus",
        },
        "specific": [
            r"\bopen[ -]source (?:ai |language |foundation |large language )?models?\b",
            r"\bopen[ -]weight(?:ed)? (?:ai |language |foundation |large language )?models?\b",
            r"\bopen models?\b",
            r"\bmodel weights\b",
            r"\bweights (?:are |were )?(?:open|available|released|downloadable)\b",
            r"\b(?:llama|mixtral|mistral|gemma|qwen|deepseek|falcon|bloom|olmo|phi[ -]?[234]|smollm|apertus)\b",
        ],
    },
    "agentic_web": {
        "tags": {
            "aiagent",
            "agenticai",
            "autonomousagent",
            "multiagentsystem",
            "modelcontextprotocol",
            "toolcalling",
            "functioncalling",
            "computeruse",
        },
        "specific": [
            r"\bagentic(?: ai)?\b",
            r"\bai agents?\b",
            r"\bartificial intelligence agents?\b",
            r"\bautonomous (?:ai )?agents?\b",
            r"\bmulti[ -]agent(?: systems?)?\b",
            r"\bmodel context protocol\b",
            r"\bagent2agent\b",
            r"\bagent[ -]to[ -]agent\b",
            r"\bcomputer use\b",
            r"\bbrowser agents?\b",
            r"\bweb agents?\b",
            r"\btool calling\b",
            r"\bfunction calling\b",
            r"\bagent orchestration\b",
        ],
    },
}


def parse_tags(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = [value]
    return [str(tag) for tag in parsed] if isinstance(parsed, list) else [str(parsed)]


def norm_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tag.lower())


def yearly_summary(mask: pd.Series, dates: pd.Series) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    periods = {
        "2024-09_to_2024-12": ("2024-09-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026-01_to_2026-08": ("2026-01-01", "2026-08-31"),
    }
    for label, (start, end) in periods.items():
        period = dates.between(start, end)
        result[label] = {
            "docs": int((mask & period).sum()),
            "all_docs": int(period.sum()),
            "share_pct": round((mask & period).sum() / period.sum() * 100, 2),
        }
    return result


def main() -> None:
    df = pd.read_csv(
        DATASET,
        usecols=["title", "date", "content", "domain", "url", "tags"],
        low_memory=False,
    )
    dates = pd.to_datetime(df["date"])
    title = df["title"].str.lower()
    content = df["content"].str.lower()
    tag_lists = df["tags"].map(parse_tags)
    normalized_tag_sets = tag_lists.map(lambda tags: {norm_tag(tag) for tag in tags})

    all_tag_counts = Counter(tag for tags in tag_lists for tag in tags)
    print("TRACK_TAG_VOCAB_MATCHES")
    for track, config in TRACKS.items():
        matched = {
            tag: count
            for tag, count in all_tag_counts.items()
            if norm_tag(tag) in config["tags"]
        }
        print(track, json.dumps(dict(sorted(matched.items(), key=lambda item: -item[1])), ensure_ascii=False))

    results: dict[str, dict[str, pd.Series]] = {}
    for track, config in TRACKS.items():
        regex = "(?:" + "|".join(config["specific"]) + ")"
        title_mask = title.str.contains(regex, regex=True, na=False)
        tag_mask = normalized_tag_sets.map(lambda tags: bool(tags & config["tags"]))
        content_mask = content.str.contains(regex, regex=True, na=False)
        seed = title_mask | tag_mask
        expanded = seed | content_mask
        results[track] = {
            "title": title_mask,
            "tag": tag_mask,
            "content": content_mask,
            "seed": seed,
            "expanded": expanded,
        }

        print("TRACK", track)
        print(
            "COUNTS",
            json.dumps(
                {
                    "title": int(title_mask.sum()),
                    "tag": int(tag_mask.sum()),
                    "title_or_tag_seed": int(seed.sum()),
                    "expanded_with_body": int(expanded.sum()),
                    "seed_pct": round(seed.mean() * 100, 2),
                    "expanded_pct": round(expanded.mean() * 100, 2),
                }
            ),
        )
        print("SEED_PERIOD_SHARES", json.dumps(yearly_summary(seed, dates)))
        print("EXPANDED_PERIOD_SHARES", json.dumps(yearly_summary(expanded, dates)))
        seed_month = dates[seed].dt.to_period("M").value_counts().sort_index()
        all_month = dates.dt.to_period("M").value_counts().sort_index()
        month_share = (seed_month / all_month * 100).fillna(0)
        print("SEED_MONTHLY_DOCS", json.dumps({str(k): int(v) for k, v in seed_month.items()}))
        print("SEED_MONTHLY_SHARE_PCT", json.dumps({str(k): round(float(v), 2) for k, v in month_share.items()}))
        print("SEED_TOP_DOMAINS", json.dumps(df.loc[seed, "domain"].value_counts().head(15).to_dict()))
        seed_tags = Counter(tag for tags in tag_lists[seed] for tag in tags)
        print("SEED_TOP_TAGS", json.dumps(dict(seed_tags.most_common(35)), ensure_ascii=False))

        sample = df.loc[seed, ["date", "title", "domain"]].sort_values("date")
        if len(sample) > 10:
            positions = [round(i * (len(sample) - 1) / 9) for i in range(10)]
            sample = sample.iloc[positions]
        print("SEED_SAMPLES")
        for record in sample.to_dict("records"):
            print(json.dumps(record, ensure_ascii=False))

    seed_frame = pd.DataFrame({track: masks["seed"] for track, masks in results.items()})
    expanded_frame = pd.DataFrame({track: masks["expanded"] for track, masks in results.items()})
    for name, frame in (("SEED", seed_frame), ("EXPANDED", expanded_frame)):
        combo = frame.apply(lambda row: "+".join(row.index[row].tolist()) or "none", axis=1)
        print(name + "_COMBINATIONS")
        print(combo.value_counts().to_string())


if __name__ == "__main__":
    main()
