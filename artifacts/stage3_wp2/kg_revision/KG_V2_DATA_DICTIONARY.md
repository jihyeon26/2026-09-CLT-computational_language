# Knowledge graph v2 (Stage 3 WP2): data dictionary

Built from the submitted Stage 1 knowledge graph by the notebook `stage3_wp2_kg_revision.ipynb`. Node IDs, relation IDs
and text unit IDs are those of Stage 1. All dates are article publication dates.

## `edges` - one row per triple (all Stage 1 graph edges)
Stage 1 columns, plus:
| column | description |
|---|---|
| `knowledge_layer` | `article_claim` (stated in an article sentence), `name_pattern` (model name with size -> family), `curated_lexicon` (WP1 lexicon only, no sentence) |
| `has_article_claim`, `has_name_pattern`, `in_lexicon` | which sources contain the triple |
| `lexicon_agreement` | `lexicon_and_articles` or `lexicon_only` for lexicon triples |
| `article_sentences`, `article_documents`, `article_first_seen`, `article_last_seen` | support counted on article sentences only |
| `support_status` | `supported` (at least one sentence entails the relation), `not_supported`, `no_passage` (lexicon only) |
| `support_max`, `support_mean` | best and mean NLI support score of the sentences |
| `scored_sentences`, `supported_sentences` | sentences checked / supporting |
| `support_disagrees_with_review` | confirmed by a person but no supporting sentence (kept) |
| `relation_reliability` | `low` for `competes with` (cannot be validated at sentence level, Section 3.6), else `standard` |
| `retrieval_eligible` | used by the retriever: article claim or name pattern, supported or confirmed by a person, and `relation_reliability = standard` |

## `assertions` - one row per evidence sentence (`retrieval_assertions`: the rows of eligible edges)
Stage 1 columns (`relation_id`, `subject_id`, `predicate`, `object_id`, `document_id`, `text_unit_id`, `evidence_text`,
`published_at`, `extraction_method`, `rule_score`), plus `knowledge_source` (layer of the row), `support_score`
(NLI entailment probability, MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli), `contradiction_score`, `n_actors` (named models / organisations in the
sentence), `list_sentence` (n_actors >= 3), `supported` (support_score >= 0.5, and for
`competes with` not a list sentence), `hypothesis` (the sentence the relation was checked as), `retrieval_eligible`.

## `background_facts`
Lexicon facts (`subject_id`, `predicate`, `object_id`, `knowledge_source = curated_lexicon`). Returned by the
retriever as `background`, never as cited evidence.

## `text_units`, `documents`
Sentences (`text_unit_id` = `doc_id:sentence_id`) and articles (publication date, title, URL, domain).

## `lexicon_conflicts`
Article relations that Stage 1 excluded because they name another developer of a lexicon family, with their
support status. They stay excluded.

## `typed_graph_v2.graphml`
All edges with `knowledge_layer`, `support_status`, `retrieval_eligible`.
