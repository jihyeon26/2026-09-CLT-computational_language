from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wp2_open_models.pipeline import extract_relations, normalize_text, parse_list_field


def test_parse_and_normalize_content() -> None:
    assert parse_list_field("['One', 'Two']") == ["One", "Two"]
    assert normalize_text("A\u00a0  B\n\n\nC") == "A B\n\nC"


def test_schema_relation_baseline() -> None:
    sentence = "Meta released Llama 3 under the Llama Community License."
    rows = [
        {
            "mention_id": "m1",
            "entity_id": "org:meta",
            "document_id": "d1",
            "text_unit_id": "d1:S0001",
            "surface_form": "Meta",
            "canonical_name": "Meta",
            "entity_type": "ORGANIZATION",
            "char_start": 0,
            "char_end": 4,
            "sentence_start": 0,
            "sentence_end": len(sentence),
            "evidence_text": sentence,
            "published_at": "2025-01-01",
            "extraction_method": "domain_entity_ruler",
            "extraction_score": 1.0,
            "score_type": "exact_dictionary_match",
        },
        {
            "mention_id": "m2",
            "entity_id": "model:llama_3",
            "document_id": "d1",
            "text_unit_id": "d1:S0001",
            "surface_form": "Llama 3",
            "canonical_name": "Llama 3",
            "entity_type": "MODEL",
            "char_start": 14,
            "char_end": 21,
            "sentence_start": 0,
            "sentence_end": len(sentence),
            "evidence_text": sentence,
            "published_at": "2025-01-01",
            "extraction_method": "domain_entity_ruler",
            "extraction_score": 1.0,
            "score_type": "exact_dictionary_match",
        },
        {
            "mention_id": "m3",
            "entity_id": "license:llama_community",
            "document_id": "d1",
            "text_unit_id": "d1:S0001",
            "surface_form": "Llama Community License",
            "canonical_name": "Llama Community License",
            "entity_type": "LICENSE",
            "char_start": 32,
            "char_end": 55,
            "sentence_start": 0,
            "sentence_end": len(sentence),
            "evidence_text": sentence,
            "published_at": "2025-01-01",
            "extraction_method": "domain_entity_ruler",
            "extraction_score": 1.0,
            "score_type": "exact_dictionary_match",
        },
    ]
    relations = extract_relations(pd.DataFrame(rows))
    assert "RELEASED_BY" in set(relations["predicate"])
    assert "LICENSED_UNDER" in set(relations["predicate"])
    assert "BELONGS_TO_FAMILY" in set(relations["predicate"])


def test_derived_from_direction_for_leading_trigger() -> None:
    sentence = "Based on DeepSeek-V3, DeepSeek-R1 improves mathematical reasoning."
    common = {
        "document_id": "d2",
        "text_unit_id": "d2:S0001",
        "sentence_start": 0,
        "sentence_end": len(sentence),
        "evidence_text": sentence,
        "published_at": "2025-01-01",
        "entity_type": "MODEL",
        "extraction_method": "domain_entity_ruler",
        "extraction_score": 1.0,
        "score_type": "exact_dictionary_match",
    }
    rows = [
        {
            **common,
            "mention_id": "m4",
            "entity_id": "model:deepseek_v3",
            "surface_form": "DeepSeek-V3",
            "canonical_name": "DeepSeek-V3",
            "char_start": 9,
            "char_end": 20,
        },
        {
            **common,
            "mention_id": "m5",
            "entity_id": "model:deepseek_r1",
            "surface_form": "DeepSeek-R1",
            "canonical_name": "DeepSeek-R1",
            "char_start": 22,
            "char_end": 33,
        },
    ]
    relations = extract_relations(pd.DataFrame(rows))
    derived = relations[relations["predicate"] == "DERIVED_FROM"].iloc[0]
    assert derived["subject_id"] == "model:deepseek_r1"
    assert derived["object_id"] == "model:deepseek_v3"


def test_release_uses_org_and_model_nearest_to_trigger() -> None:
    sentence = "The Meta Llama 3.2 launch came before Meta released Llama 3.1."
    specs = [
        ("m6", "org:meta", "Meta", "Meta", "ORGANIZATION", 4, 8),
        ("m7", "model:llama_3_2", "Llama 3.2", "Llama 3.2", "MODEL", 9, 18),
        ("m8", "org:meta", "Meta", "Meta", "ORGANIZATION", 38, 42),
        ("m9", "model:llama_3_1", "Llama 3.1", "Llama 3.1", "MODEL", 52, 61),
    ]
    rows = [
        {
            "mention_id": mention_id,
            "entity_id": entity_id,
            "document_id": "d3",
            "text_unit_id": "d3:S0001",
            "surface_form": surface,
            "canonical_name": canonical,
            "entity_type": entity_type,
            "char_start": start,
            "char_end": end,
            "sentence_start": 0,
            "sentence_end": len(sentence),
            "evidence_text": sentence,
            "published_at": "2025-01-01",
            "extraction_method": "domain_entity_ruler",
            "extraction_score": 1.0,
            "score_type": "exact_dictionary_match",
        }
        for mention_id, entity_id, surface, canonical, entity_type, start, end in specs
    ]
    relations = extract_relations(pd.DataFrame(rows))
    release = relations[relations["predicate"] == "RELEASED_BY"].iloc[0]
    assert release["subject_id"] == "model:llama_3_1"
    assert release["object_id"] == "org:meta"


def test_repeated_model_mentions_do_not_create_self_relation() -> None:
    sentence = "Gemma 4 feedback is based on experience using Gemma 4."
    rows = []
    for index, (start, end) in enumerate(((0, 7), (46, 53)), start=1):
        rows.append(
            {
                "mention_id": f"repeat-{index}",
                "entity_id": "model:gemma_4",
                "document_id": "d4",
                "text_unit_id": "d4:S0001",
                "surface_form": "Gemma 4",
                "canonical_name": "Gemma 4",
                "entity_type": "MODEL",
                "char_start": start,
                "char_end": end,
                "sentence_start": 0,
                "sentence_end": len(sentence),
                "evidence_text": sentence,
                "published_at": "2025-01-01",
                "extraction_method": "domain_entity_ruler",
                "extraction_score": 1.0,
                "score_type": "exact_dictionary_match",
            }
        )
    relations = extract_relations(pd.DataFrame(rows))
    assert "DERIVED_FROM" not in set(relations["predicate"])

