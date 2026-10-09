# Stage 1 WP3 최종 지식그래프 데이터 사전

Schema: `stage1-wp3-open-models-kg-v1.0` · 상태: `pending_human_review` · Track: Open Models
날짜(`published_at`, `first_seen`, `last_seen`)는 모두 **기사 발행일**이다.

## Node (`nodes.*`)
| 열 | 설명 |
|---|---|
| `entity_id` | 보정 후 node ID. 별칭·이름 변형이 합쳐진 원래 WP2 ID는 `merged_ids` |
| `canonical_name`, `aliases` | 대표 이름과 병합된 다른 이름(JSON list) |
| `entity_type` / `entity_type_compact` | WP2 type / 9개 compact type |
| `mention_count`, `document_count` | 언급 수 / 언급 기사 수 |
| `is_open_model_family` | WP1 lexicon의 open model family 여부 |
| `parameter_sizes_b` | 파라미터 크기(B) 목록(JSON). 모델 node는 이름에 적힌 크기, family node는 소속 모델 node들의 크기 (3-1절 모델명 정제) |
| `typed_*`, `cooc_*`, `community_id` | typed·co-occurrence graph metric과 community |

## Typed edge
- `typed_edges_assertions.*`: 근거 문장 1개당 1행. `relation_id`, `subject_id`, `predicate`, `object_id`,
  `document_id`, `text_unit_id`(= `doc_id:sentence_id`), `evidence_text`, `published_at`, `extraction_method`
  (`glirel`, `wp1_lexicon`, `model_name`), `rule_score`
- `typed_edges_collapsed.*`: unique triple 1개당 1행. `evidence_count`, `document_count`(edge weight),
  `n_distinct_sentences`, `first_seen`, `last_seen`, `tier`(A: lexicon, B: 근거 문장 2개 이상, `competes with`는 3개 이상),
  `sources`, `added_by`(보정 / 관계 기준 type / 모델명 정제), `review_status`(표본 사람 검수 `reviewed_*`, insight 대조 `checked_true_human`·`checked_true_ai`),
  `predicate_group`, `directed`, `relation_ids`/`text_unit_ids`(JSON list)

## Co-occurrence edge (`cooccurrence_edges.*`)
같은 문장에 함께 언급된 entity 쌍. `sentence_count`, `document_count`(문서 2개 이상),
`npmi`, `merged_pairs`(보정으로 합쳐진 원래 쌍 수). 의미 관계가 아니다.

## Graph 파일
`assertion_graph`, `typed_graph`, `cooccurrence_graph` 각각 `.graphml`과 node-link `.json`

## 분석 표
`node_metrics`, `relation_type_metrics`, `relation_type_summary`, `node_communities`, `community_summary`,
`quarterly_relation_assertions`, `quarterly_new_typed_edges`, `family_quarterly_share`,
`entity_period_share`, `graph_finding_candidates`(검토 전 후보),
`model_name_mentions`(모델 node의 근거 문장 전체: family, 모델 node, 크기, 원문 표현 `surface`)

## Stage 3 provenance
`entity_id` → `relation_id` → `text_unit_id` → `document_id` → `published_at`.
기사 제목·URL은 WP1 `open_models_core.parquet`의 `doc_id`로 연결한다.
