"""Stage 3 revision of the Stage 1 knowledge graph: knowledge layers and evidence support checks."""

from .layers import KNOWLEDGE_SOURCE, edge_layers, label_assertions, lexicon_conflicts
from .support import (
    HYPOTHESES,
    NLIVerifier,
    aggregate_support,
    build_pairs,
    hypothesis_name,
    score_pairs,
    sentence_actor_counts,
    threshold_table,
    wilson,
)

__all__ = [
    "HYPOTHESES",
    "KNOWLEDGE_SOURCE",
    "NLIVerifier",
    "aggregate_support",
    "build_pairs",
    "edge_layers",
    "hypothesis_name",
    "label_assertions",
    "lexicon_conflicts",
    "score_pairs",
    "sentence_actor_counts",
    "threshold_table",
    "wilson",
]
