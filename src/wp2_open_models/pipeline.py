from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import spacy

from .catalog import (
    CATALOGUE_BY_ID,
    ENTITY_CATALOGUE,
    OPEN_MODEL_TAGS,
    OPEN_MODEL_TEXT_PATTERN,
    RELATION_SIGNATURES,
)


PIPELINE_VERSION = "stage1-wp2-open-models-v0.1"
MODEL_TYPES = {"MODEL", "MODEL_FAMILY"}


@dataclass(frozen=True)
class PipelineConfig:
    input_csv: Path
    output_dir: Path
    max_docs: int = 300
    random_seed: int = 42
    spacy_model: str = "en_core_web_sm"
    min_date: str = "2024-09-01"
    max_date: str = "2026-08-31"
    review_sample_size: int = 100


def parse_list_field(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\u00a0", " ").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "unknown"


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratified_limit(frame: pd.DataFrame, max_docs: int) -> pd.DataFrame:
    if max_docs <= 0 or len(frame) <= max_docs:
        return frame.copy()
    months = sorted(frame["month"].unique())
    quota = max(1, math.ceil(max_docs / len(months)))
    chunks = []
    for month in months:
        month_rows = frame[frame["month"] == month].sort_values(
            ["track_score", "date", "document_id"], ascending=[False, True, True]
        )
        chunks.append(month_rows.head(quota))
    selected = pd.concat(chunks, ignore_index=True)
    if len(selected) > max_docs:
        selected = selected.sort_values(
            ["track_score", "date", "document_id"], ascending=[False, True, True]
        ).head(max_docs)
    return selected.sort_values(["date", "document_id"]).reset_index(drop=True)


def load_open_model_documents(config: PipelineConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    source = pd.read_csv(
        config.input_csv,
        usecols=["title", "date", "content", "domain", "url", "tags"],
        low_memory=False,
    )
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    valid_date = source["date"].between(config.min_date, config.max_date)
    source = source.loc[valid_date].copy()

    title_hit = source["title"].str.contains(OPEN_MODEL_TEXT_PATTERN, case=False, regex=True, na=False)
    tag_regex = "|".join(re.escape(tag) for tag in sorted(OPEN_MODEL_TAGS))
    tag_hit = source["tags"].str.contains(tag_regex, case=False, regex=True, na=False)
    content_hit = source["content"].str.contains(OPEN_MODEL_TEXT_PATTERN, case=False, regex=True, na=False)
    source["track_score"] = title_hit.astype(int) * 3 + tag_hit.astype(int) * 2 + content_hit.astype(int)
    source = source[title_hit | tag_hit].copy()

    source["document_id"] = source["url"].map(lambda value: f"doc:{short_hash(str(value))}")
    source["month"] = source["date"].dt.to_period("M").astype(str)
    source["tags_list"] = source["tags"].map(parse_list_field)
    source["text_clean"] = source["content"].map(
        lambda value: normalize_text("\n".join(parse_list_field(value)))
    )
    source["published_at"] = source["date"].dt.strftime("%Y-%m-%d")
    source["selection_reason"] = np.select(
        [title_hit.loc[source.index] & tag_hit.loc[source.index], title_hit.loc[source.index], tag_hit.loc[source.index]],
        ["title_and_tag", "title", "tag"],
        default="unknown",
    )
    candidates = len(source)
    selected = _stratified_limit(source, config.max_docs)
    keep = [
        "document_id",
        "title",
        "published_at",
        "month",
        "domain",
        "url",
        "text_clean",
        "tags_list",
        "track_score",
        "selection_reason",
    ]
    audit = {
        "rows_in_date_range": int(valid_date.sum()),
        "open_model_candidates": int(candidates),
        "selected_documents": int(len(selected)),
    }
    return selected[keep].copy(), audit


def build_nlp(model_name: str):
    nlp = spacy.load(
        model_name,
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
    )
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True},
    )
    patterns = []
    for row in ENTITY_CATALOGUE:
        for alias in row["aliases"]:
            patterns.append({"label": row["entity_type"], "pattern": alias, "id": row["entity_id"]})
    ruler.add_patterns(patterns)
    nlp.max_length = 250_000
    return nlp


def _generic_entity_id(entity_type: str, text: str) -> str:
    prefix = {
        "ORGANIZATION": "org",
        "PERSON": "person",
        "PRODUCT_OTHER": "product",
    }[entity_type]
    return f"{prefix}:{slugify(text)}"


def extract_mentions_and_sentences(
    documents: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nlp = build_nlp(model_name)
    sentence_rows: list[dict[str, Any]] = []
    mention_rows: list[dict[str, Any]] = []

    inputs = ((row.text_clean, row) for row in documents.itertuples(index=False))
    for parsed, row in nlp.pipe(inputs, as_tuples=True, batch_size=8, n_process=1):
        sentence_ids: dict[tuple[int, int], str] = {}
        sentence_domain_presence: set[tuple[int, int]] = set()
        for index, sentence in enumerate(parsed.sents, start=1):
            sentence_id = f"{row.document_id}:S{index:04d}"
            sentence_ids[(sentence.start_char, sentence.end_char)] = sentence_id
            sentence_rows.append(
                {
                    "text_unit_id": sentence_id,
                    "document_id": row.document_id,
                    "sentence_index": index,
                    "start_char": sentence.start_char,
                    "end_char": sentence.end_char,
                    "text_unit_text": sentence.text,
                    "published_at": row.published_at,
                }
            )
        for ent in parsed.ents:
            if ent.ent_id_:
                sentence_domain_presence.add((ent.sent.start_char, ent.sent.end_char))

        for ent in parsed.ents:
            if ent.ent_id_:
                catalogue = CATALOGUE_BY_ID[ent.ent_id_]
                entity_id = ent.ent_id_
                entity_type = catalogue["entity_type"]
                canonical_name = catalogue["canonical_name"]
                extraction_method = "domain_entity_ruler"
                extraction_score = 1.0
                score_type = "exact_dictionary_match"
            elif ent.label_ in {"ORG", "PERSON", "PRODUCT"} and (
                ent.sent.start_char,
                ent.sent.end_char,
            ) in sentence_domain_presence:
                entity_type = {"ORG": "ORGANIZATION", "PERSON": "PERSON", "PRODUCT": "PRODUCT_OTHER"}[ent.label_]
                canonical_name = normalize_text(ent.text)
                entity_id = _generic_entity_id(entity_type, canonical_name)
                extraction_method = model_name
                extraction_score = np.nan
                score_type = "not_available"
            else:
                continue

            sentence_id = sentence_ids[(ent.sent.start_char, ent.sent.end_char)]
            mention_id = f"mention:{short_hash(f'{row.document_id}|{ent.start_char}|{ent.end_char}|{entity_id}') }"
            mention_rows.append(
                {
                    "mention_id": mention_id,
                    "entity_id": entity_id,
                    "document_id": row.document_id,
                    "text_unit_id": sentence_id,
                    "surface_form": ent.text,
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                    "char_start": ent.start_char,
                    "char_end": ent.end_char,
                    "sentence_start": ent.sent.start_char,
                    "sentence_end": ent.sent.end_char,
                    "evidence_text": ent.sent.text,
                    "published_at": row.published_at,
                    "extraction_method": extraction_method,
                    "extraction_score": extraction_score,
                    "score_type": score_type,
                }
            )

    sentences = pd.DataFrame(sentence_rows)
    mentions = pd.DataFrame(mention_rows).drop_duplicates("mention_id")
    return mentions, sentences


def build_entity_table(mentions: pd.DataFrame) -> pd.DataFrame:
    if mentions.empty:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "canonical_name",
                "entity_type",
                "aliases",
                "mention_count",
                "document_count",
                "first_seen",
                "last_seen",
                "extraction_methods",
            ]
        )
    records = []
    for entity_id, group in mentions.groupby("entity_id", sort=True):
        type_counts = group["entity_type"].value_counts()
        canonical_counts = group["canonical_name"].value_counts()
        records.append(
            {
                "entity_id": entity_id,
                "canonical_name": canonical_counts.index[0],
                "entity_type": type_counts.index[0],
                "aliases": json.dumps(sorted(set(group["surface_form"])), ensure_ascii=False),
                "mention_count": int(len(group)),
                "document_count": int(group["document_id"].nunique()),
                "first_seen": group["published_at"].min(),
                "last_seen": group["published_at"].max(),
                "extraction_methods": json.dumps(sorted(set(group["extraction_method"])), ensure_ascii=False),
            }
        )
    return pd.DataFrame(records)


def _assertion_status(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(?:not|never|no evidence|didn't|did not|hasn't|has not)\b", lower):
        return "negated"
    if re.search(r"\b(?:may|might|could|reportedly|allegedly|expected to|plans to)\b", lower):
        return "uncertain"
    return "asserted"


def _relation_record(
    subject: pd.Series,
    predicate: str,
    object_: pd.Series,
    method: str,
    rule_score: float,
) -> dict[str, Any]:
    key = f"{subject.document_id}|{subject.text_unit_id}|{subject.entity_id}|{predicate}|{object_.entity_id}|{method}"
    return {
        "relation_id": f"relation:{short_hash(key)}",
        "subject_id": subject.entity_id,
        "predicate": predicate,
        "object_id": object_.entity_id,
        "document_id": subject.document_id,
        "text_unit_id": subject.text_unit_id,
        "evidence_text": subject.evidence_text,
        "published_at": subject.published_at,
        "assertion_status": "asserted" if method == "catalogue_parent_v1" else _assertion_status(subject.evidence_text),
        "extraction_method": method,
        "rule_score": rule_score,
        "score_type": "heuristic_rule_score_not_probability",
    }


def _directional_model_pair(
    models: pd.DataFrame,
    trigger: re.Match[str],
    *,
    allow_leading_trigger: bool,
) -> tuple[pd.Series, pd.Series] | None:
    """Return (derived/successor, base/predecessor) around a directional trigger.

    Mention offsets are document-relative while the trigger is sentence-relative.
    Ambiguous sentences are deliberately skipped: precision matters more than recall
    for graph edges used downstream by RAG.
    """
    specific = models[models["entity_type"] == "MODEL"].copy()
    if len(specific) < 2:
        return None
    specific["local_start"] = specific["char_start"] - specific["sentence_start"]
    specific = specific.sort_values("local_start")
    before = specific[specific["local_start"] < trigger.start()]
    after = specific[specific["local_start"] >= trigger.end()]

    # Canonical form: "derived model ... based on ... base model".
    if not before.empty and not after.empty:
        subject, object_ = before.iloc[-1], after.iloc[0]
        return None if subject["entity_id"] == object_["entity_id"] else (subject, object_)

    # Inverted form: "Based on base model, derived model ...".  Only accept a
    # near-sentence-leading trigger and a nearby base mention to avoid linking
    # unrelated models in long comparison sentences.
    if allow_leading_trigger and trigger.start() <= 20 and len(after) >= 2:
        if int(after.iloc[0]["local_start"]) - trigger.end() <= 60:
            subject, object_ = after.iloc[1], after.iloc[0]
            return None if subject["entity_id"] == object_["entity_id"] else (subject, object_)
    return None


def _org_model_pair_for_trigger(
    group: pd.DataFrame,
    trigger: re.Match[str],
) -> tuple[pd.Series, pd.Series] | None:
    """Find an explicit (model, organisation) pair around a release/development verb."""
    positioned = group.copy()
    positioned["local_start"] = positioned["char_start"] - positioned["sentence_start"]
    positioned["local_end"] = positioned["char_end"] - positioned["sentence_start"]
    models = positioned[positioned["entity_type"].isin(MODEL_TYPES)].sort_values("local_start")
    curated_org_ids = {
        entity_id
        for entity_id, metadata in CATALOGUE_BY_ID.items()
        if metadata["entity_type"] == "ORGANIZATION"
    }
    orgs = positioned[positioned["entity_id"].isin(curated_org_ids)].sort_values("local_start")
    if models.empty or orgs.empty:
        return None

    # "Meta released Llama 3".
    org_before = orgs[orgs["local_start"] < trigger.start()]
    model_after = models[models["local_start"] >= trigger.end()]
    if not org_before.empty and not model_after.empty:
        org = org_before.iloc[-1]
        model = model_after.iloc[0]
        nearby_model_before = models[
            (models["local_start"] < trigger.start())
            & (models["local_start"] >= trigger.start() - 50)
        ]
        explicit_org_after_prior_model = (
            not nearby_model_before.empty
            and int(org["local_start"]) > int(nearby_model_before.iloc[-1]["local_start"])
        )
        if (nearby_model_before.empty or explicit_org_after_prior_model) and trigger.start() - int(org["local_start"]) <= 120 and int(model["local_start"]) - trigger.end() <= 90:
            between = str(group.iloc[0]["evidence_text"])[trigger.end() : int(model["local_start"])].lower()
            after_surface = str(group.iloc[0]["evidence_text"])[int(model["local_end"]) : int(model["local_end"]) + 20]
            blocked_context = re.search(
                r"\b(?:(?:variant|version|derivative|fine-tune)\s+of|rival(?:s|ing)?|outperform(?:s|ing)?|"
                r"compared with|versus|built on|availability|(?:app|service|platform|framework|tool)\b.{0,35}\b(?:run|runs|using|uses|with|of))\b",
                between,
            )
            speculative_release = re.search(r"to[- ]be[- ]$", str(group.iloc[0]["evidence_text"])[: trigger.start()], re.I)
            embedded_name = re.match(r"-[A-Za-z0-9]", after_surface)
            if not blocked_context and not speculative_release and not embedded_name:
                return model, org

    # "Meta's Llama 3 ... was released".
    model_before = models[models["local_start"] < trigger.start()]
    if not model_before.empty:
        model = model_before.iloc[-1]
        org_before_model = orgs[orgs["local_start"] < int(model["local_start"])]
        if not org_before_model.empty:
            org = org_before_model.iloc[-1]
            intervening_org = orgs[
                (orgs["local_start"] > int(model["local_start"]))
                & (orgs["local_start"] < trigger.start())
            ]
            ownership_text = str(group.iloc[0]["evidence_text"])[int(org["local_end"]) : int(model["local_start"])]
            close_name = int(model["local_start"]) - int(org["local_end"]) <= 20
            possessive_name = re.search(r"(?:['’]s\s+|\bof\s+)$", ownership_text, re.I)
            hyphenated_compound = re.fullmatch(r"\s*-\s*", ownership_text)
            if intervening_org.empty and trigger.start() - int(org["local_start"]) <= 180 and (close_name or possessive_name) and not hyphenated_compound:
                return model, org

        # "Llama 3 was developed by Meta".
        org_after = orgs[orgs["local_start"] >= trigger.end()]
        if not org_after.empty:
            org = org_after.iloc[0]
            between = str(group.iloc[0]["evidence_text"])[trigger.end() : int(org["local_start"])].lower()
            if re.search(r"\bby\b", between) and int(org["local_start"]) - trigger.end() <= 90:
                return model, org
    return None


def extract_relations(mentions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if mentions.empty:
        return pd.DataFrame()

    release_re = re.compile(r"\b(?:released|launched|unveiled|introduced|announced|open-sourced|published)\b", re.I)
    develop_re = re.compile(
        r"\b(?:developed|created(?!\s+(?:using|with|from)\b)|"
        r"built(?![- ]?in\b|\s+(?:on|from|with|into)\b)|trained)\b",
        re.I,
    )
    successor_re = re.compile(r"\b(?:successor to|succeeds|replaces|next generation of)\b", re.I)
    derived_re = re.compile(r"\b(?:based on|built on|derived from|fine-tuned from|fork of)\b", re.I)
    license_re = re.compile(r"\b(?:licensed under|available under|released under|licen[cs]e|terms of use)\b", re.I)
    architecture_re = re.compile(r"\b(?:architecture|based on|uses|built with|powered by)\b", re.I)
    capability_re = re.compile(r"\b(?:supports|capable of|designed for|optimized for|can|ability to|excels at)\b", re.I)

    for _, group in mentions.groupby("text_unit_id", sort=False):
        # Keep repeated mentions here: relation direction depends on the mention
        # nearest to the trigger even when the same entity appears twice.
        group = group.sort_values("char_start")
        text = str(group.iloc[0]["evidence_text"])
        models = group[group["entity_type"].isin(MODEL_TYPES)]
        orgs = group[group["entity_type"] == "ORGANIZATION"]
        licenses = group[group["entity_type"] == "LICENSE"]
        architectures = group[group["entity_type"] == "ARCHITECTURE"]
        capabilities = group[group["entity_type"] == "CAPABILITY"]

        verb_matches = [
            *(('DEVELOPED_BY', match) for match in develop_re.finditer(text)),
            *(('RELEASED_BY', match) for match in release_re.finditer(text)),
        ]
        for predicate, match in sorted(verb_matches, key=lambda item: item[1].start()):
            pair = _org_model_pair_for_trigger(group, match)
            if pair:
                records.append(_relation_record(pair[0], predicate, pair[1], "directional_org_model_rule_v2", 0.85))

        for _, model in models.iterrows():
            if not licenses.empty and license_re.search(text):
                license_ = min(licenses.itertuples(index=False), key=lambda row: abs(row.char_start - model.char_start))
                records.append(_relation_record(model, "LICENSED_UNDER", pd.Series(license_._asdict()), "sentence_schema_rule_v1", 0.90))

            if not architectures.empty and architecture_re.search(text):
                architecture = min(architectures.itertuples(index=False), key=lambda row: abs(row.char_start - model.char_start))
                records.append(_relation_record(model, "USES_ARCHITECTURE", pd.Series(architecture._asdict()), "sentence_schema_rule_v1", 0.70))

            if not capabilities.empty and capability_re.search(text):
                capability = min(capabilities.itertuples(index=False), key=lambda row: abs(row.char_start - model.char_start))
                records.append(_relation_record(model, "HAS_CAPABILITY", pd.Series(capability._asdict()), "sentence_schema_rule_v1", 0.65))

        successor_match = successor_re.search(text)
        derived_match = derived_re.search(text)
        if successor_match:
            pair = _directional_model_pair(models, successor_match, allow_leading_trigger=False)
            if pair:
                records.append(_relation_record(pair[0], "SUCCESSOR_OF", pair[1], "directional_sentence_rule_v2", 0.85))
        elif derived_match:
            pair = _directional_model_pair(models, derived_match, allow_leading_trigger=True)
            if pair:
                records.append(_relation_record(pair[0], "DERIVED_FROM", pair[1], "directional_sentence_rule_v2", 0.80))

        for _, model in models[models["entity_type"] == "MODEL"].iterrows():
            parent_id = CATALOGUE_BY_ID.get(model.entity_id, {}).get("parent_family_id")
            if parent_id:
                parent = CATALOGUE_BY_ID[parent_id]
                family = model.copy()
                family["entity_id"] = parent_id
                family["canonical_name"] = parent["canonical_name"]
                family["entity_type"] = parent["entity_type"]
                records.append(_relation_record(model, "BELONGS_TO_FAMILY", family, "catalogue_parent_v1", 1.0))

    if not records:
        return pd.DataFrame(
            columns=[
                "relation_id", "subject_id", "predicate", "object_id", "document_id",
                "text_unit_id", "evidence_text", "published_at", "assertion_status",
                "extraction_method", "rule_score", "score_type",
            ]
        )
    return pd.DataFrame(records).drop_duplicates("relation_id").sort_values(
        ["published_at", "text_unit_id", "predicate", "subject_id", "object_id"]
    ).reset_index(drop=True)


def ensure_relation_endpoints(entities: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    known = set(entities["entity_id"]) if not entities.empty else set()
    missing = (set(relations.get("subject_id", [])) | set(relations.get("object_id", []))) - known
    additions = []
    for entity_id in sorted(missing):
        row = CATALOGUE_BY_ID.get(entity_id)
        if row is None:
            continue
        additions.append(
            {
                "entity_id": entity_id,
                "canonical_name": row["canonical_name"],
                "entity_type": row["entity_type"],
                "aliases": json.dumps(row["aliases"], ensure_ascii=False),
                "mention_count": 0,
                "document_count": 0,
                "first_seen": None,
                "last_seen": None,
                "extraction_methods": json.dumps(["catalogue_parent_v1"]),
            }
        )
    if additions:
        entities = pd.concat([entities, pd.DataFrame(additions)], ignore_index=True)
    return entities.sort_values(["entity_type", "canonical_name", "entity_id"]).reset_index(drop=True)


def build_cooccurrence(
    mentions: pd.DataFrame,
    entities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"sentence_ids": set(), "document_ids": set(), "first_seen": None, "last_seen": None}
    )
    for _, group in mentions.groupby("text_unit_id"):
        unique = sorted(set(group["entity_id"]))
        if len(unique) < 2:
            continue
        row0 = group.iloc[0]
        for source_id, target_id in combinations(unique, 2):
            record = counts[(source_id, target_id)]
            record["sentence_ids"].add(row0.text_unit_id)
            record["document_ids"].add(row0.document_id)
            date = row0.published_at
            record["first_seen"] = date if record["first_seen"] is None else min(record["first_seen"], date)
            record["last_seen"] = date if record["last_seen"] is None else max(record["last_seen"], date)

    rows = [
        {
            "source_id": source,
            "target_id": target,
            "sentence_count": len(values["sentence_ids"]),
            "document_count": len(values["document_ids"]),
            "first_seen": values["first_seen"],
            "last_seen": values["last_seen"],
        }
        for (source, target), values in counts.items()
    ]
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.sort_values(["document_count", "sentence_count", "source_id", "target_id"], ascending=[False, False, True, True]).reset_index(drop=True)

    top_ids = entities.sort_values(["document_count", "mention_count"], ascending=False).head(75)["entity_id"].tolist()
    matrix = pd.DataFrame(0, index=top_ids, columns=top_ids, dtype=int)
    if not edges.empty:
        for row in edges.itertuples(index=False):
            if row.source_id in matrix.index and row.target_id in matrix.columns:
                matrix.loc[row.source_id, row.target_id] = row.document_count
                matrix.loc[row.target_id, row.source_id] = row.document_count
    matrix.index.name = "entity_id"
    return edges, matrix


def build_review_sample(
    relations: pd.DataFrame,
    entities: pd.DataFrame,
    documents: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if relations.empty:
        return relations.copy()
    predicates = sorted(relations["predicate"].unique())
    per_group = max(1, math.ceil(sample_size / len(predicates)))
    samples = []
    for predicate in predicates:
        group = relations[relations["predicate"] == predicate]
        samples.append(group.sample(min(per_group, len(group)), random_state=seed))
    result = pd.concat(samples, ignore_index=True).head(sample_size).copy()
    labels = entities.set_index("entity_id")["canonical_name"]
    result.insert(2, "subject_label", result["subject_id"].map(labels))
    result.insert(5, "object_label", result["object_id"].map(labels))
    metadata = documents.set_index("document_id")[["title", "domain", "url"]]
    result = result.join(metadata, on="document_id")
    result["review_correct"] = ""
    result["reviewer"] = ""
    result["review_error_type"] = ""
    result["review_notes"] = ""
    return result


def validate_outputs(
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    mentions: pd.DataFrame,
    entities: pd.DataFrame,
    relations: pd.DataFrame,
) -> dict[str, Any]:
    text_by_doc = documents.set_index("document_id")["text_clean"].to_dict()
    offset_failures = 0
    for row in mentions.itertuples(index=False):
        text = text_by_doc[row.document_id]
        if text[row.char_start : row.char_end] != row.surface_form:
            offset_failures += 1

    entity_ids = set(entities["entity_id"])
    sentence_ids = set(sentences["text_unit_id"])
    document_ids = set(documents["document_id"])
    dangling_subjects = int((~relations["subject_id"].isin(entity_ids)).sum()) if not relations.empty else 0
    dangling_objects = int((~relations["object_id"].isin(entity_ids)).sum()) if not relations.empty else 0
    missing_evidence = int((~relations["text_unit_id"].isin(sentence_ids)).sum()) if not relations.empty else 0
    missing_documents = int((~relations["document_id"].isin(document_ids)).sum()) if not relations.empty else 0

    type_by_id = entities.set_index("entity_id")["entity_type"].to_dict()
    invalid_signatures = 0
    if not relations.empty:
        for row in relations.itertuples(index=False):
            signature = (type_by_id.get(row.subject_id), type_by_id.get(row.object_id))
            if signature not in RELATION_SIGNATURES.get(row.predicate, set()):
                invalid_signatures += 1

    qa = {
        "document_count": int(len(documents)),
        "sentence_count": int(len(sentences)),
        "mention_count": int(len(mentions)),
        "entity_count": int(len(entities)),
        "relation_count": int(len(relations)),
        "documents_with_mentions": int(mentions["document_id"].nunique()) if not mentions.empty else 0,
        "documents_with_relations": int(relations["document_id"].nunique()) if not relations.empty else 0,
        "document_mention_coverage_pct": round(mentions["document_id"].nunique() / len(documents) * 100, 2) if len(documents) else 0,
        "document_relation_coverage_pct": round(relations["document_id"].nunique() / len(documents) * 100, 2) if len(documents) else 0,
        "duplicate_document_ids": int(documents["document_id"].duplicated().sum()),
        "duplicate_mention_ids": int(mentions["mention_id"].duplicated().sum()) if not mentions.empty else 0,
        "duplicate_entity_ids": int(entities["entity_id"].duplicated().sum()) if not entities.empty else 0,
        "duplicate_relation_ids": int(relations["relation_id"].duplicated().sum()) if not relations.empty else 0,
        "mention_offset_failures": int(offset_failures),
        "dangling_relation_subjects": dangling_subjects,
        "dangling_relation_objects": dangling_objects,
        "relations_missing_evidence_sentence": missing_evidence,
        "relations_missing_document": missing_documents,
        "invalid_relation_signatures": int(invalid_signatures),
        "entity_types": entities["entity_type"].value_counts().to_dict() if not entities.empty else {},
        "relation_types": relations["predicate"].value_counts().to_dict() if not relations.empty else {},
        "assertion_statuses": relations["assertion_status"].value_counts().to_dict() if not relations.empty else {},
    }
    hard_failures = [
        "duplicate_document_ids",
        "duplicate_mention_ids",
        "duplicate_entity_ids",
        "duplicate_relation_ids",
        "mention_offset_failures",
        "dangling_relation_subjects",
        "dangling_relation_objects",
        "relations_missing_evidence_sentence",
        "relations_missing_document",
        "invalid_relation_signatures",
    ]
    qa["hard_validation_passed"] = all(qa[name] == 0 for name in hard_failures)
    return qa


def build_entity_graph(entities: pd.DataFrame, relations: pd.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(track="open_models", pipeline_version=PIPELINE_VERSION)
    for row in entities.itertuples(index=False):
        graph.add_node(
            row.entity_id,
            label=row.canonical_name,
            entity_type=row.entity_type,
            mention_count=int(row.mention_count),
            document_count=int(row.document_count),
        )
    for row in relations.itertuples(index=False):
        graph.add_edge(
            row.subject_id,
            row.object_id,
            key=row.relation_id,
            relation_id=row.relation_id,
            predicate=row.predicate,
            document_id=row.document_id,
            text_unit_id=row.text_unit_id,
            published_at=row.published_at,
            assertion_status=row.assertion_status,
            extraction_method=row.extraction_method,
            rule_score=float(row.rule_score),
        )
    return graph


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def export_outputs(
    config: PipelineConfig,
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    mentions: pd.DataFrame,
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    cooccurrence_edges: pd.DataFrame,
    cooccurrence_matrix: pd.DataFrame,
    review_sample: pd.DataFrame,
    qa: dict[str, Any],
    audit: dict[str, int],
) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "documents": documents,
        "text_units": sentences,
        "entity_mentions": mentions,
        "entities": entities,
        "relations": relations,
        "cooccurrence_edges": cooccurrence_edges,
    }
    for name, table in tables.items():
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
    entities.to_csv(output_dir / "entities.csv", index=False, encoding="utf-8-sig")
    relations.to_csv(output_dir / "relations.csv", index=False, encoding="utf-8-sig")
    cooccurrence_edges.to_csv(output_dir / "cooccurrence_edges.csv", index=False, encoding="utf-8-sig")
    cooccurrence_matrix.to_csv(output_dir / "cooccurrence_matrix_top75.csv", encoding="utf-8-sig")
    review_sample.to_csv(output_dir / "relation_review_sample.csv", index=False, encoding="utf-8-sig")

    graph = build_entity_graph(entities, relations)
    node_link = nx.node_link_data(graph, edges="edges")
    (output_dir / "knowledge_graph.json").write_text(
        json.dumps(_json_safe(node_link), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    nx.write_graphml(graph, output_dir / "knowledge_graph.graphml")

    schema = {
        "pipeline_version": PIPELINE_VERSION,
        "entity_types": sorted(set(entities["entity_type"])),
        "relation_signatures": {
            predicate: [list(signature) for signature in sorted(signatures)]
            for predicate, signatures in RELATION_SIGNATURES.items()
        },
        "notes": {
            "rule_score": "Heuristic ranking score, not a calibrated probability.",
            "assertion_status": "Shallow lexical status; must be manually reviewed.",
        },
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output_dir / "qa_report.json").write_text(
        json.dumps(_json_safe({**audit, **qa}), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "input_sha256": file_sha256(config.input_csv),
        "spacy_version": spacy.__version__,
        "networkx_version": nx.__version__,
        "pandas_version": pd.__version__,
        "qa_passed": qa["hard_validation_passed"],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dictionary = """# WP2 output data dictionary

This directory contains a pilot extraction for the Open Models track.

- `documents.parquet`: selected and minimally cleaned source documents.
- `text_units.parquet`: sentence-level evidence units with document offsets.
- `entity_mentions.parquet`: extracted mention spans and canonical entity IDs.
- `entities.parquet` / `entities.csv`: canonical entity catalogue derived from mentions.
- `relations.parquet` / `relations.csv`: evidence-bearing typed relation candidates.
- `cooccurrence_edges.parquet` / `cooccurrence_edges.csv`: sentence co-occurrence edge list.
- `cooccurrence_matrix_top75.csv`: dense matrix for the 75 most frequent entities.
- `relation_review_sample.csv`: stratified sample for manual precision review.
- `knowledge_graph.json`: NetworkX node-link JSON for later reuse.
- `knowledge_graph.graphml`: convenience export for visualization tools.
- `schema.json`: entity types and allowed relation signatures.
- `qa_report.json`: integrity checks and coverage statistics.
- `run_manifest.json`: input hash and software/configuration versions.

`rule_score` is a heuristic score and must not be described as a calibrated probability.
The relation file is a baseline candidate set; manual review and a stronger structured extractor are still required for a final submission.
"""
    (output_dir / "DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    documents, audit = load_open_model_documents(config)
    mentions, sentences = extract_mentions_and_sentences(documents, config.spacy_model)
    entities = build_entity_table(mentions)
    relations = extract_relations(mentions)
    entities = ensure_relation_endpoints(entities, relations)
    cooccurrence_edges, cooccurrence_matrix = build_cooccurrence(mentions, entities)
    review_sample = build_review_sample(
        relations,
        entities,
        documents,
        config.review_sample_size,
        config.random_seed,
    )
    qa = validate_outputs(documents, sentences, mentions, entities, relations)
    export_outputs(
        config,
        documents,
        sentences,
        mentions,
        entities,
        relations,
        cooccurrence_edges,
        cooccurrence_matrix,
        review_sample,
        qa,
        audit,
    )
    return {**audit, **qa}

