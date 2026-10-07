# Stage 1 통합 체크리스트와 WP1 → WP2 → WP3 작업 개요

기준일: 2026-10-07  
트랙: **Open Models**  
범위: Stage 1의 세 work package인 WP1(Data and Topic Analysis), WP2(Knowledge Extraction), WP3(Knowledge Graph and Outputs)

> 이 문서의 `WP1`, `WP2`, `WP3`는 모두 **Stage 1 내부 작업 구분**이다. 공식 프로젝트의 Stage 2는 embeddings/model training이고, 공식 Stage 3는 RAG/question answering이다.

WP1·WP2 노트북의 핵심 셀을 코드, 실행 결과와 해석 순서로 검토하려면
`docs/STAGE1_WP1_WP2_NOTEBOOK_REVIEW_KO.md`를 함께 본다.

## 제출 전 통합 체크리스트

표시 방법:

- `[x]`: 현재 산출물로 확인 완료
- `[ ]`: 미완료, 재검토 또는 최종 통합 시 수행 필요

### 공통 범위와 재현성

- [x] 전 단계에서 Open Models 트랙을 일관되게 사용한다.
- [x] 분석 기간을 2024년 9월부터 2026년 8월까지로 유지한다.
- [x] 문서별 안정적인 `doc_id`와 기사 발행일을 보존한다.
- [ ] 세 WP의 핵심 내용을 하나의 `stage1_final.ipynb`에 통합한다.
- [ ] 최종 notebook을 처음부터 끝까지 순서대로 재실행하고 미실행 셀과 error output이 없는지 확인한다.
- [ ] 설치 패키지와 모델 버전을 `requirements.txt` 또는 실행 환경 설명에 기록한다.
- [ ] Google Drive 입력·출력 경로와 필요한 파일을 notebook 첫 부분에 설명한다.
- [ ] 팀원별 기여 범위를 최종 notebook에 명시한다.
- [ ] 사용한 AI coding tool의 이름과 사용 범위를 공개한다.
- [ ] 그래프와 표의 수치가 notebook 서술과 일치하는지 최종 교차 확인한다.

### WP1: Data and Topic Analysis

- [x] 원본 데이터 로딩, 본문 복원, 결측치 처리와 중복 제거 과정을 설명한다.
- [x] Unicode, 공백, HTML·boilerplate 등 텍스트 정제 기준을 설명한다.
- [x] tokenization, lemmatization, stop-word 처리와 Open Models 용어 정규화를 수행한다.
- [x] 정제된 전체 corpus와 Open Models core corpus를 구분해 저장한다.
- [x] 월·분기별 문서량과 Open Models 트렌드를 분석한다.
- [x] sentiment와 담론 frame을 분석한다.
- [x] topic modeling을 수행하고 topic 수 선정 근거와 시간 변화를 제시한다.
- [x] Open Models 모델 family, developer, ecosystem, openness concept lexicon을 제공한다.
- [ ] WP1의 자동·LLM 보조 validation 결과를 사람이 최종 검토하고 승인한다.
- [ ] WP1의 핵심 insight가 표·그림·원문 근거와 일치하는지 팀이 검토한다.

### WP2: Knowledge Extraction

- [x] WP1의 `open_models_core.parquet`과 lexicon을 입력으로 사용한다.
- [x] spaCy와 GLiNER로 일반·도메인 entity를 추출한다.
- [x] entity mention을 canonical entity와 stable `entity_id`로 연결한다.
- [x] GLiREL 또는 이에 준하는 structured extraction으로 typed relation을 추출한다.
- [x] relation마다 `doc_id`, `sentence_id`, 문장 근거와 발행일을 보존한다.
- [x] sentence-level entity co-occurrence를 계산한다.
- [x] entity type, relation type, 허용 방향을 `schema.json`에 기록한다.
- [x] nodes, relations, evidence, co-occurrence와 data dictionary를 인계한다.
- [ ] 최종 node type을 기준으로 relation의 schema validity를 다시 계산한다.
- [ ] 중복 triple과 self-loop를 제거하거나 명시적으로 제외한다.
- [ ] generic entity와 1회성 noise를 분석용 graph에서 필터링한다.
- [ ] relation 90개를 사람이 검수하고 precision과 오류 유형을 기록한다.
- [ ] `evaluation/` 산출물을 실제 인계 폴더에 포함한다.
- [ ] mention offset과 `sentences.csv` 문자열의 불일치를 수정하거나 한계로 명시한다.
- [ ] WP2 notebook의 미실행 셀, 실행 순서, 고정 Drive 경로와 임시 clone 셀을 정리한다.

### WP3: Knowledge Graph and Outputs

- [x] 공식 WP2 파일명과 열 이름을 WP3 canonical contract에 연결한다.
- [x] `doc_id + sentence_id`로 전역 고유한 text-unit ID를 만든다.
- [ ] compact Open Models graph schema를 확정하고 36개 relation을 필요한 수준으로 통합한다.
- [ ] evidence assertion graph를 `MultiDiGraph`로 생성한다.
- [ ] 동일 triple을 합친 collapsed typed graph를 생성한다.
- [ ] co-occurrence graph를 typed graph와 분리해 생성한다.
- [ ] node·edge 수, density, component, isolate 등 기본 구조 통계를 계산한다.
- [ ] in/out degree, weighted degree, PageRank, betweenness를 계산한다.
- [ ] co-occurrence graph의 clustering coefficient와 community를 분석한다.
- [ ] relation type별 subgraph metric을 계산하고 방향의 의미를 해석한다.
- [ ] 월·분기별 새 node, 새 relation, first/last seen을 분석한다.
- [ ] 전체 요약, 개발 조직, 모델 특성, community, temporal graph를 시각화한다.
- [ ] graph finding을 실제 source sentence와 대조한 뒤 insight로 채택한다.
- [ ] GraphML, node-link JSON, 분석 테이블, figure, schema와 data dictionary를 저장한다.
- [ ] Stage 3 Graph-RAG가 entity·relation에서 원문 passage로 돌아갈 수 있는지 smoke test한다.
- [ ] WP3 시각화의 label, legend, edge 방향과 가독성을 사람이 승인한다.

### 최종 합격 조건

- [ ] 모든 relation endpoint가 `nodes`에 존재한다.
- [ ] 분석용 typed edge가 compact schema를 모두 만족한다.
- [ ] 근거가 필요한 relation은 문서와 문장으로 역추적된다.
- [ ] co-occurrence를 의미 관계나 인과관계로 해석하지 않는다.
- [ ] 발행일을 실제 모델 출시일이나 사건 발생일로 잘못 표현하지 않는다.
- [ ] relation 수동 검수 결과가 팀 합의 기준을 통과한다. 내부 기준은 최소 90건, precision 0.80 이상이다.
- [ ] 그래프 metric을 산업 영향력이나 모델 품질의 직접 척도로 과장하지 않는다.
- [ ] 최종 notebook, graph 파일과 data dictionary가 동일한 schema version을 사용한다.
- [ ] 자동 quality gate에서 `FAIL`이 없고 수동 승인 항목도 완료된다.

---

## 1. 전체 흐름과 각 WP의 경계

```text
AI Media Dataset
    │
    ▼
WP1: 정제 corpus + Open Models 범위 + topic/temporal insight
    │  doc_id, date, title, text, family/concept lexicon
    ▼
WP2: entity mention + canonical node + evidence-bearing relation
    │  nodes, relation assertions, sentences, co-occurrence, schema
    ▼
WP3: 검증·필터링된 knowledge graph + metric + visualization
    │  GraphML/JSON, graph tables, schema, evidence lookup contract
    ▼
공식 Stage 3: text retrieval + graph-enhanced RAG
```

각 WP의 종료 지점은 다음과 같이 구분한다.

| WP | 시작점 | 종료점 | 하지 않는 일 |
|---|---|---|---|
| WP1 | 원본 AI Media Dataset | 정제되고 트랙이 지정된 corpus와 topic·temporal 분석 | 개별 관계를 knowledge graph edge로 확정하지 않음 |
| WP2 | WP1 core corpus와 lexicon | 원문 근거에 연결된 entity·relation·co-occurrence 테이블 | centrality와 graph insight를 최종 해석하지 않음 |
| WP3 | WP2 구조화 테이블 | 저장·재사용 가능한 graph, metric, 시각화와 insight | 원문 근거 없이 새로운 사실을 생성하지 않음 |

WP1은 **분석 대상과 언어적 맥락**을 정하고, WP2는 **텍스트를 구조화된 사실 후보**로 바꾸며, WP3는 **그 후보를 검증 가능한 그래프와 분석 결과**로 완성한다.

---

## 2. WP1에서 수행한 작업

### 목적

WP1은 전체 AI Media Dataset을 정제하고 Open Models 관련 문서를 선별한 뒤, 시간 변화와 담론의 주요 주제를 설명하는 단계다. WP2와 WP3가 사용할 공통 문서 ID, 본문, 날짜와 도메인 lexicon도 여기서 정의한다.

### 현재 완료 범위

| 산출물 | 규모·내용 | 다음 단계에서의 역할 |
|---|---|---|
| `corpus_clean.parquet` | 정제 corpus 45,644개 문서 | 전체 분석 모집단과 원문 추적 |
| `open_models_core.parquet` | Open Models core 1,210개 문서 | WP2 entity/relation extraction의 직접 입력 |
| `open_models_target_sentences.parquet` | 관련 문장 13,587개 | sentiment, frame, evidence 후보 |
| `open_models_lexicon.json` | 35개 model family, developer, ecosystem, concept와 정규화 규칙 | WP2 entity canonicalization과 curated relation |
| EDA 표·그림 | 월·분기, source, family, region, terminology 변화 | WP3 graph trend 해석의 비교 기준 |
| topic 결과 | 16개 NMF topic과 시간별 prevalence | graph community·relation trend 해석의 배경 |
| sentiment·frame 결과 | family·region·분기별 담론 특성 | 그래프 구조와 담론 변화의 연결 |

### WP1이 WP2에 넘긴 핵심 계약

- `doc_id`: URL의 SHA-1 앞 10자리로 만든 안정적 문서 ID
- `date`, `month`, `quarter`: 기사 발행 시점
- `title`, `text`, `url`, `domain`: 근거와 출처 확인에 필요한 본문·메타데이터
- `om_families`, `om_regions`: Open Models family와 developer region
- family·tool·concept 정규화 규칙과 version pattern

### 남은 확인 사항

WP1 notebook은 실행 결과와 주요 표·그림이 갖춰져 있다. 다만 최종 제출 전에는 트랙 validation sample과 핵심 insight를 사람이 검토해야 한다. 현재 validation label은 LLM 보조 결과이므로 사람의 최종 승인을 별도로 남기는 것이 안전하다. 대용량 processed parquet은 Git이 아니라 공유 Drive에 있으므로 최종 notebook에 정확한 위치와 hash도 기록해야 한다.

---

## 3. WP2가 WP1을 이어받아 수행한 작업

### 연결 방식

WP2는 WP1의 1,210개 Open Models core 문서를 그대로 사용했다. `doc_id`와 발행일을 유지하면서 문장을 나누고, WP1 lexicon을 high-precision domain dictionary로 재사용했다. 이 방식으로 WP1의 `Llama`, `Qwen`, `Gemma` 같은 family가 WP2에서도 같은 canonical identity를 유지한다.

### 추출 파이프라인

1. spaCy로 sentence segmentation과 일반 NER을 수행한다.
2. WP1 lexicon으로 foundation model, ecosystem tool과 openness concept를 찾는다.
3. GLiNER로 AI model, company, technology, benchmark, license 등 도메인 타입을 확장한다.
4. overlapping mention을 병합하고 surface form을 canonical entity로 연결한다.
5. GLiREL로 문장 안의 방향성 relation을 추출한다.
6. 같은 문장에 나타난 entity pair의 co-occurrence와 NPMI를 계산한다.
7. node, edge, evidence, quarterly table, schema와 data dictionary를 인계한다.

### 현재 공식 WP2 산출물

| 항목 | 결과 |
|---|---:|
| 대상 문서 | 1,210 |
| 문장 | 99,438 |
| entity mention | 209,148 |
| canonical node | 54,597 |
| relation evidence row | 30,595 |
| aggregated relation edge | 22,992 |
| co-occurrence edge | 186,710 |
| entity type | 20 |
| relation type | 36 |

제공된 주요 파일은 다음과 같다.

- `nodes.csv`, `nodes_quarterly.csv`
- `entity_mentions.csv`, `sentences.csv`
- `relations_evidence.csv`, `edges_relations.csv`
- `edges_cooccurrence.csv`, `cooccurrence_quarterly.csv`
- `schema.json`, `data_dictionary.md`

### 구조적으로 확인된 장점

- WP1의 1,210개 `doc_id`가 유지된다.
- GLiREL relation의 문서·문장 참조가 모두 실제 `sentences.csv`에 존재한다.
- 모든 relation과 co-occurrence endpoint가 `nodes.csv`에 존재한다.
- node, quarter, relation과 co-occurrence 집계가 원본 evidence 계산과 일치한다.
- schema 밖의 entity type 또는 relation label은 없다.
- 날짜 범위가 2024-09부터 2026-08까지 유지된다.

### WP2의 현재 한계와 WP3 인계 조건

WP2는 WP3를 시작하기에 충분하지만 원본 edge를 그대로 그래프에 넣으면 안 된다.

1. **최종 node type과 relation type validation의 불일치**  
   mention 단계 타입으로 계산한 `type_valid` 중 417개는 최종 node type을 적용하면 무효다. WP2 권장 필터 뒤에도 84개가 남으므로 WP3에서 다시 계산해야 한다.

2. **중복과 self-loop**  
   `(head_id, relation, tail_id)` 중복 행 266개와 self-loop 1개가 있다. evidence는 보존하되 분석용 collapsed graph에서는 정리해야 한다.

3. **noise와 graph 규모**  
   node의 71.4%가 한 번만 등장하고 79.1%는 문서 하나에서만 등장한다. `AI`, `model`, `company`, `developers`, `users` 같은 generic entity도 포함되어 있다.

4. **relation 수동 검수 미완료**  
   entity에 대한 자동·silver-standard 평가는 있지만 relation의 의미적 precision은 측정되지 않았다.

5. **근거 예외**  
   GLiREL relation은 문장으로 추적되지만 WP1 lexicon에서 추가한 `developed by` 35개는 기사 문장 근거가 없다. curated source로 분리해 표시해야 한다.

6. **mention offset 불일치**  
   12,977개 mention의 offset이 저장된 `sentence_text`와 맞지 않는다. sentence text에 `.strip()`을 적용하면서 `sentence_start`를 보정하지 않은 것이 주된 원인이다.

7. **재현성과 인계 파일**  
   WP2 notebook의 30개 code cell 중 10개가 미실행 상태이고 실행 번호가 순차적이지 않다. data dictionary가 설명하는 `evaluation/` 파일도 Git 인계본에는 없다.

---

## 4. WP3가 WP2를 이어받는 방법

### WP3의 목적

WP3는 WP2의 결과를 단순히 NetworkX에 넣는 단계가 아니다. 먼저 schema·ID·provenance를 검증하고, 목적별 graph view를 만든 뒤, network metric과 시간 분석을 통해 Open Models 생태계의 구조와 변화를 설명해야 한다.

### 1단계: 원본을 보존하고 contract로 연결

공식 WP2 폴더는 수정하지 않고 snapshot으로 보존한다. WP3 adapter에서 다음을 canonical field로 변환한다.

| WP2 열 | WP3 canonical field |
|---|---|
| `entity` | `canonical_name` |
| `head_id` | `subject_id` |
| `relation` | `predicate` |
| `tail_id` | `object_id` |
| `doc_id` | `document_id` |
| `sentence_text` | `evidence_text` 또는 text-unit `text` |
| `source` | `extraction_method` |
| `score` | `rule_score` |
| `entity1_id`, `entity2_id` | co-occurrence `source_id`, `target_id` |

`sentence_id`는 각 문서 안에서만 유일하므로 다음과 같이 변환한다.

```text
text_unit_id = doc_id + ":" + sentence_id
```

typed relation의 canonical 입력에는 집계본인 `edges_relations.csv`보다 문장 근거가 있는 `relations_evidence.csv`를 우선 사용한다.

### 2단계: relation을 다시 검증하고 분석용 입력을 만든다

권장 순서는 다음과 같다.

1. endpoint가 node table에 존재하는지 검사한다.
2. `nodes.csv`의 최종 `entity_type`으로 relation schema를 다시 평가한다.
3. self-loop를 제거한다.
4. relation evidence 행은 보존하되 동일 triple을 분석용 테이블에서 collapse한다.
5. GLiREL edge에는 `type_valid`, support와 confidence 정책을 적용한다.
6. WP1 curated edge는 `extraction_method=wp1_lexicon`으로 분리한다.
7. generic·저빈도 node 필터를 기록하고 원본 graph와 분석 graph를 모두 저장한다.
8. relation review 결과가 생기면 명시적 FALSE edge를 제외한다.

현재 산출물에서 보수적 필터와 node-type 재검증을 적용하면 약 1,539개의 unique typed triple과 978개의 관련 node를 초기 분석 대상으로 사용할 수 있다. 문장 5개 이상에서 관찰된 co-occurrence는 3,398개다. 이 값은 최종 정책을 확정한 뒤 다시 보고한다.

### 3단계: 세 가지 graph view를 만든다

#### Evidence assertion graph

- 자료구조: directed `MultiDiGraph`
- edge 하나가 relation evidence 한 건을 나타낸다.
- `document_id`, `text_unit_id`, `evidence_text`, 발행일과 score를 유지한다.
- Stage 3 Graph-RAG의 근거 회수에 사용한다.

#### Collapsed typed graph

- 자료구조: directed graph 또는 predicate를 보존하는 `MultiDiGraph`
- 동일 `(subject_id, predicate, object_id)`를 하나로 합친다.
- `evidence_count`, `document_count`, `first_seen`, `last_seen`을 저장한다.
- centrality, relation-specific analysis와 발표용 시각화에 사용한다.

#### Co-occurrence graph

- 자료구조: undirected weighted graph
- 같은 문장에 등장한 entity pair를 나타낸다.
- `sentence_cooccurrence`, `document_cooccurrence`, `npmi`와 기간을 저장한다.
- community와 담론상 연관 구조를 탐색한다.

### 4단계: compact schema를 확정한다

WP2의 20개 entity type과 36개 relation type은 extraction 단계의 recall을 위한 넓은 schema다. 과제는 compact topic-specific schema를 요구하므로 WP3에서 다음을 결정한다.

- `ORG`와 `COMPANY`를 분석에서 통합할지 여부
- `FOUNDATION_MODEL`, `AI_MODEL`, `PRODUCT`, `PLATFORM`의 경계
- `developed by`, `released by`, `announced by`를 별도로 유지할지 상위 relation으로 묶을지 여부
- `based on`, `fine-tuned from`, `distilled from`, `successor of`의 계층
- `MODEL_SPEC`를 node로 유지할지 edge attribute로 바꿀지 여부
- broad·ambiguous relation을 핵심 graph에서 제외할 기준

원본 relation label은 삭제하지 않고 `raw_predicate`로 보존하고, compact schema의 label을 별도 `predicate`로 두는 것이 좋다.

### 5단계: metric과 시간 분석을 수행한다

최소 분석 범위는 다음과 같다.

- node·edge 수, density, weakly connected component와 isolate
- entity type·relation type별 분포
- in-degree, out-degree, weighted degree
- PageRank와 betweenness centrality
- co-occurrence clustering coefficient
- weighted community detection과 community별 대표 entity
- relation-specific centrality
- 월·분기별 새 node와 새 relation
- entity와 relation의 first/last seen

metric은 graph 구조 안의 위치를 나타낸다. PageRank가 높다고 실제 시장 영향력이나 모델 품질이 높다고 단정하지 않는다.

### 6단계: 시각화와 insight를 완성한다

권장 시각화는 다음과 같다.

1. compact typed graph 전체 요약
2. model–developer graph
3. model–architecture/license/capability graph
4. model lineage 또는 earlier-model relation graph
5. co-occurrence community graph
6. relation type 분포
7. centrality 비교
8. 월·분기별 새 node·relation 추이

각 insight는 최소 하나 이상의 source passage로 확인한다. 공출현은 “함께 언급됨”으로 표현하고, relation은 방향과 evidence를 확인한 뒤 서술한다.

### 7단계: 저장과 Stage 3 인계

WP3의 최종 인계 묶음에는 다음이 포함되어야 한다.

- assertion, collapsed typed, co-occurrence graph의 GraphML
- 동일 graph의 node-link JSON
- node·edge·metric·community·temporal Parquet 또는 CSV
- compact `graph_schema.json`
- contract report, validation report와 run manifest
- figure와 해석 가능한 summary table
- `DATA_DICTIONARY.md`
- entity 또는 relation에서 source passage를 찾는 예제 코드

공식 Stage 3에서는 이 결과를 text-only RAG와 graph-enhanced RAG의 비교에 사용한다. 따라서 graph edge에서 원문 passage와 발행일로 돌아가는 provenance chain은 반드시 유지해야 한다.

---

## 5. WP3 작업 완료 판정 기준

WP3는 다음 질문에 모두 답할 수 있을 때 완료로 본다.

1. 어떤 WP1 문서와 WP2 evidence에서 각 graph edge가 왔는가?
2. 어떤 schema version과 필터 정책으로 graph가 만들어졌는가?
3. typed relation과 co-occurrence를 분리해서 분석했는가?
4. 전체 graph와 발표용 filtered graph의 차이를 설명할 수 있는가?
5. 주요 metric이 어떤 graph view와 weight에서 계산됐는가?
6. 시각화에서 node type, relation, 방향과 threshold를 읽을 수 있는가?
7. 주요 finding을 실제 기사 문장과 연결할 수 있는가?
8. 월별 문서량 변화와 실제 관계 변화의 차이를 고려했는가?
9. GraphML과 JSON을 다시 로드할 수 있는가?
10. 공식 Stage 3 retrieval 코드가 entity·relation에서 passage를 가져올 수 있는가?

이 기준을 충족하면 Stage 1의 세 WP는 다음과 같은 하나의 설명 가능한 흐름을 이룬다.

> WP1이 무엇이 변화했는지 보여주고, WP2가 누가 무엇과 어떤 관계인지 구조화하며, WP3가 그 구조의 중심·군집·시간 변화를 분석하고 원문 근거와 함께 저장한다.

---

## 6. 기준 파일

- 프로젝트 안내서: `CLT_AI_Trend_Project.pdf`
- 팀 WP1 notebook: `stage1/work_packages/stage1_wp1_data_topic_analysis.ipynb`
- 팀 WP1 data dictionary: `artifacts/stage1_wp1/data_dictionary.md`
- 팀 WP2 notebook: `stage1/work_packages/stage1_wp2_entity_relation_extraction.ipynb`
- 팀 WP2 schema와 data dictionary: `artifacts/stage1_wp2/schema.json`, `artifacts/stage1_wp2/data_dictionary.md`
- WP3 draft contract: `config/wp3_open_models_contract.json`
- 자동 quality gate: `tests/quality_gates/`
