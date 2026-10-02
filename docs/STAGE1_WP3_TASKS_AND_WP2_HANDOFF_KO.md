# Stage 1 WP3 작업 계획과 WP2 연결 문서

## 1. 문서의 범위와 용어

이 문서에서 **WP2**와 **WP3**는 팀이 Stage 1을 세 개의 work package로 나눈 구조를 뜻한다.

- Stage 1 WP2: Knowledge Extraction
- Stage 1 WP3: Knowledge Graph and Outputs

공식 프로젝트 문서의 **Stage 2**는 embeddings와 model training이고, 공식 **Stage 3**는 RAG와 question answering이다. 이름이 혼동될 수 있으므로 이 문서에서는 팀 내부 작업을 `Stage 1 WP2`, `Stage 1 WP3`로 표기한다.

Stage 1 WP3의 직접 목표는 WP2의 구조화된 엔티티와 관계를 이용해 Open Models knowledge graph를 만들고, 그래프 구조·metric·시각화·저장 형식을 완성하는 것이다. 이 그래프는 나중에 공식 Stage 3의 graph-enhanced RAG에서 다시 사용되어야 한다.

## 2. 프로젝트 평가기준에서 요구하는 WP3의 역할

공식 프로젝트 안내서의 Stage 1 `Knowledge Graph Construction` 및 `Outputs` 요구사항은 다음과 같다.

1. 그래프를 만들기 전에 compact topic-specific schema를 정의한다.
2. 엔티티를 node로, co-mention 또는 extracted relation을 edge로 표현한다.
3. centrality와 clustering 등의 network metric으로 영향력 있는 엔티티와 새롭게 나타나는 관계를 분석한다.
4. NetworkX 또는 Graphviz 등의 도구로 그래프를 시각화한다.
5. graph와 schema를 저장해 공식 Stage 3에서 재사용할 수 있게 한다.
6. topic-focused knowledge graph와 짧은 schema/data dictionary를 제출한다.
7. selected track의 주요 insight와 notable development를 문서화한다.

프로젝트 공통 제출 요건도 WP3 노트북에 반영해야 한다.

- 별도의 final report는 필수가 아니므로 notebook 자체가 설명 가능한 결과물이어야 한다.
- 코드는 구조화되고 주석과 설명이 충분해야 한다.
- 결과 시각화가 포함되어야 한다.
- notebook은 가능한 한 재현 가능한 tutorial처럼 읽혀야 한다.
- 각 팀원의 기여를 명확히 표시해야 한다.
- AI coding tool을 사용했다면 도구 이름과 사용 방식을 공개해야 한다.

즉, 평가는 그래프가 실행되는지만 보는 것이 아니라 **스키마의 타당성, 분석의 해석 가능성, 시각화, 재현성, Stage 3 재사용 가능성**까지 함께 본다.

---

## 3. WP2에서 WP3로 전달되는 현재 결과

WP2 파일럿은 300개 Open Models 문서에서 다음 결과를 만들었다.

| 항목 | 현재 결과 |
|---|---:|
| 선택 문서 | 300 |
| 문장 단위 | 26,579 |
| 엔티티 mention | 8,850 |
| canonical entity | 1,288 |
| relation assertion | 1,647 |
| unique typed triple | 68 |
| co-occurrence edge | 4,230 |
| 관계가 발견된 문서 | 177 |

자동 무결성 검사에서는 다음 항목이 모두 0이었다.

- 중복 ID
- mention offset 오류
- 존재하지 않는 node를 가리키는 edge
- source document 또는 evidence sentence가 없는 관계
- schema에 허용되지 않은 relation signature
- subject와 object가 같은 self-edge

이 결과는 **구조적으로 그래프를 만들 준비가 되었다**는 뜻이다. 다만 의미적 정확도가 완전히 검증됐다는 뜻은 아니다. `relation_review_sample.csv`의 수동 검수가 아직 필요하다.

## 4. WP2 산출물과 WP3 작업의 직접 연결

| WP2 산출물 | 포함 정보 | WP3에서의 역할 |
|---|---|---|
| `entities.parquet` | canonical entity ID, 이름, 타입, mention/document 수, 관찰 기간 | graph node 생성 |
| `relations.parquet` | subject, predicate, object, evidence, 날짜, assertion 상태 | 방향성 typed edge 생성 |
| `entity_mentions.parquet` | 표면형, offset, 문장, canonical ID | node provenance와 원문 위치 확인 |
| `documents.parquet` | 제목, 날짜, 도메인, URL, 정규화 본문 | 출처와 시간 정보 연결 |
| `text_units.parquet` | 문장 ID와 문장 본문 | edge evidence 조회 및 Stage 3 retrieval |
| `cooccurrence_edges.parquet` | 문장·문서 공동출현 빈도와 기간 | 무방향 co-occurrence graph 생성 |
| `cooccurrence_matrix_top75.csv` | 상위 엔티티 dense matrix | heatmap과 clustering 보조 입력 |
| `schema.json` | entity type과 허용 relation signature | graph schema 검증 |
| `relation_review_sample.csv` | 검수 대상 관계와 source URL | edge precision 평가 및 필터링 |
| `qa_report.json` | coverage와 무결성 검사 결과 | 입력 품질 보고 |
| `run_manifest.json` | 데이터 hash, 기간, seed, 버전 | 재현성 기록 |
| `knowledge_graph.graphml` | 현재 assertion 기반 graph export | WP3 초기 로딩 및 시각화 참고 |
| `knowledge_graph.json` | NetworkX node-link 구조 | Python 기반 재사용 |

WP3는 GraphML 파일을 그대로 꾸미는 데서 시작하면 안 된다. 먼저 Parquet 원본에서 목적별 graph view를 명시적으로 만들어야 한다.

---

## 5. WP3에서 만들어야 할 세 가지 graph view

### 5.1 Assertion graph

WP2 `relations.parquet`의 한 행을 하나의 edge로 유지하는 directed MultiGraph다. 동일한 triple이라도 서로 다른 evidence sentence가 있으면 여러 edge가 존재한다.

용도:

- edge에서 원문 문장으로 돌아가기
- 서로 다른 기사나 도메인의 지지 evidence 비교
- 공식 Stage 3 Graph-RAG에서 근거 passage 회수
- 시간별 assertion 추적

권장 edge 속성:

- `predicate`
- `document_id`
- `text_unit_id`
- `published_at`
- `assertion_status`
- `extraction_method`
- `rule_score`

Assertion graph는 provenance에는 유리하지만, 반복 evidence 때문에 centrality가 과도하게 커질 수 있으므로 발표용 graph metric의 기본 그래프로 사용하지 않는 것이 좋다.

### 5.2 Collapsed typed graph

동일한 `(subject_id, predicate, object_id)`를 하나의 edge로 합친 directed graph다. 현재 1,647개 assertion은 68개 unique triple로 축약된다.

권장 집계 속성:

- `evidence_count`: assertion 수
- `document_count`: 서로 다른 기사 수
- `domain_count`: 서로 다른 출처 도메인 수
- `first_seen`: 첫 기사 발행일
- `last_seen`: 마지막 기사 발행일
- `asserted_count`, `negated_count`, `uncertain_count`
- `review_status`

용도:

- typed relation 중심의 centrality와 component 분석
- 모델-조직, 모델-라이선스, 모델-아키텍처 관계 시각화
- 발표용 knowledge graph

### 5.3 Co-occurrence graph

같은 문장에 등장한 엔티티 쌍으로 구성한 undirected weighted graph다.

권장 edge 속성:

- `sentence_count`
- `document_count`
- `first_seen`
- `last_seen`

용도:

- 자주 함께 언급되는 모델·조직 탐색
- community detection
- 모델 간 비교 담론이나 생태계 구조 탐색

Co-occurrence는 의미 관계가 아니다. `A`와 `B`가 자주 같이 등장해도 `A가 B를 개발했다`는 뜻은 아니다. Typed graph와 같은 그림에 표시한다면 edge style과 legend를 명확히 분리해야 한다.

---

## 6. WP3의 구체적인 작업 순서

### 작업 1. 입력 검증과 edge 정책 확정

먼저 `qa_report.json`의 `hard_validation_passed`가 `true`인지 확인한다. 이어서 다음 정책을 코드와 notebook에 명시한다.

- 기본 typed graph에는 `assertion_status == asserted`만 포함할지 여부
- 수동 검수에서 `review_correct == FALSE`인 관계의 제외 방식
- 아직 검수되지 않은 관계를 `unreviewed`로 표시할지 여부
- generic `PERSON`, `ORGANIZATION`, `PRODUCT_OTHER` node를 주 시각화에 포함할지 여부
- edge weight를 `evidence_count`와 `document_count` 중 무엇으로 정의할지 여부

권장 정책은 다음과 같다.

- 분석용 원본에는 모든 assertion을 보존한다.
- 발표용 typed graph에는 asserted relation만 사용한다.
- 주 시각화는 `MODEL`, `MODEL_FAMILY`, curated `ORGANIZATION`, `LICENSE`, `ARCHITECTURE`, `CAPABILITY` 중심으로 제한한다.
- edge thickness에는 `document_count`를 사용하고 `evidence_count`는 tooltip 또는 표에 별도로 제시한다.

### 작업 2. Schema를 WP3 관점에서 확정

WP2 schema를 검토한 뒤 다음을 문서화한다.

- 각 node type의 정의와 포함·제외 기준
- 각 relation의 허용 방향
- directed/undirected 여부
- 필수 node/edge 속성
- canonical ID 규칙
- evidence와 source 연결 규칙
- 날짜가 기사 발행일인지 사건 발생일인지

현재 schema에는 `SUCCESSOR_OF`가 정의되어 있지만 파일럿에서 추출된 edge는 0개다. 이것을 오류로 볼 필요는 없지만, Open Models 평가 목표에 “earlier models와의 관계”가 포함되어 있으므로 catalogue와 relation extraction 규칙을 보완할지 검토해야 한다.

### 작업 3. Collapsed edge table 생성

`relations.parquet`을 다음 key로 groupby한다.

```text
subject_id + predicate + object_id
```

각 group에서 evidence, document, domain, 시간 범위를 집계한다. 결과는 `typed_edges_collapsed.parquet`과 사람이 읽을 수 있는 CSV로 저장한다.

중요한 점은 evidence를 삭제하지 않는 것이다. collapsed edge에는 원문 전체를 중복 저장하기보다 assertion ID나 `text_unit_id` 목록을 연결하고, 실제 evidence는 원본 relation table에서 조회하도록 한다.

### 작업 4. Graph 생성

NetworkX 기준 권장 구조는 다음과 같다.

- Assertion graph: `MultiDiGraph`
- Collapsed typed graph: `DiGraph` 또는 relation type별 `MultiDiGraph`
- Co-occurrence graph: `Graph`

모든 graph node에는 최소한 다음 속성을 둔다.

- `label`
- `entity_type`
- `mention_count`
- `document_count`
- `first_seen`
- `last_seen`

### 작업 5. Network metric 계산

프로젝트 기준에서 centrality와 clustering이 예시로 명시되어 있으므로 최소 다음 분석이 필요하다.

#### 기본 구조 통계

- node 수와 edge 수
- density
- connected component 또는 weakly connected component 수
- isolated node 수
- relation type별 edge 수
- entity type별 node 수

#### Centrality

- in-degree와 out-degree
- weighted degree
- PageRank
- betweenness centrality

Directed typed graph에서 in-degree와 out-degree의 의미가 relation마다 다르므로 전체 graph 결과뿐 아니라 relation-specific subgraph도 함께 봐야 한다. 예를 들어 `DEVELOPED_BY`의 방향은 모델에서 조직으로 향하므로 조직의 in-degree가 높게 나온다.

#### Clustering과 community

- weighted clustering coefficient
- community detection
- community별 대표 엔티티와 주요 공동출현 edge

Clustering은 triangle이 풍부한 co-occurrence graph에서 더 자연스럽다. 희소한 directed typed graph에 clustering 값을 적용했다면 정의와 한계를 설명해야 한다.

#### 시간 분석

- 월별 새 node 수
- 월별 새 typed relation 수
- 모델 패밀리별 first/last seen
- 기간별 centrality 또는 co-occurrence 변화

현재 날짜는 기사 발행일이므로 “출시 시점”이 아니라 “데이터셋에서 관찰된 시점”이라고 표현해야 한다.

### 작업 6. 시각화 생성

최소 다음 시각화를 권장한다.

1. **전체 요약 graph**: 상위 node만 표시하고 node color는 entity type, edge color는 predicate로 구분한다.
2. **모델/패밀리-개발 조직 graph**: `DEVELOPED_BY`, `RELEASED_BY` 중심의 directed view다.
3. **모델-라이선스-아키텍처 graph**: `LICENSED_UNDER`, `USES_ARCHITECTURE`를 비교한다.
4. **Co-occurrence community graph**: document count가 일정 기준 이상인 edge만 표시한다.
5. **Top centrality table 또는 bar chart**: metric 값과 entity type을 함께 표시한다.
6. **Temporal chart**: 주요 모델 패밀리 또는 관계의 월별 관찰 변화를 보여준다.

1,288개 node 전체를 한 그림에 표시하는 것은 정보 전달력이 낮다. 전체 graph 파일은 보존하되, 보고용 그림은 목적별 subgraph와 threshold를 사용해야 한다. 사용한 threshold는 notebook에 명시한다.

### 작업 7. Insight를 작성

평가기준은 단순 metric 계산이 아니라 influential entity와 emerging relationship을 해석하도록 요구한다. 각 insight는 다음 구조로 작성하는 것이 좋다.

1. 관찰한 graph pattern
2. 사용한 graph와 metric
3. 관련 entity와 relation
4. 이를 지지하는 evidence 문장과 기사 날짜
5. 표본·추출 방식에 따른 한계

예를 들어 “Meta가 가장 영향력 있는 조직이다”라고 단정하기보다 “현재 300문서 파일럿의 collapsed developer/release subgraph에서 Meta의 연결 수가 가장 높았다”처럼 분석 단위를 명시해야 한다.

### 작업 8. 저장과 재사용 인터페이스 완성

최종 graph는 최소 두 형식으로 저장한다.

- GraphML: Gephi, Cytoscape 등 외부 시각화 도구와 공유
- JSON node-link: NetworkX와 Stage 3 코드에서 로드

이외에 Parquet node/edge table을 canonical source로 유지한다. GraphML은 list나 nested object 속성 표현에 제한이 있으므로 evidence ID 목록은 별도 table로 보존하는 편이 안전하다.

---

## 7. WP3에서 계산하면 안 되는 잘못된 결과

다음 방식은 평가에서 설명력이 약하거나 잘못된 해석을 만들 수 있다.

- 1,647개 assertion을 모두 독립적인 사실로 간주하기
- `BELONGS_TO_FAMILY` 반복 evidence를 그대로 degree에 반영하기
- typed relation과 co-occurrence를 동일한 의미의 edge로 합치기
- 일반 NER 조직 720개를 모두 검증된 Open Models 개발 조직으로 간주하기
- 파일럿 300문서의 centrality를 전체 산업의 영향력으로 표현하기
- 기사 발행일을 실제 모델 출시일로 표현하기
- graph 그림만 만들고 metric 및 해석을 생략하기
- graph edge에서 source sentence와 URL 연결을 제거하기

---

## 8. WP3 노트북 권장 목차

WP3 notebook이 재현 가능한 tutorial처럼 읽히도록 다음 순서를 권장한다.

1. Objective and project context
2. WP2 input artifacts and validation
3. Final graph schema
4. Edge inclusion and review policy
5. Assertion graph construction
6. Collapsed typed graph construction
7. Co-occurrence graph construction
8. Basic graph statistics
9. Centrality analysis
10. Clustering/community analysis
11. Temporal analysis
12. Graph visualizations
13. Key Open Models insights with evidence
14. Limitations and validity threats
15. Saved artifacts and Stage 3 loading example
16. Team member contributions
17. AI tool disclosure

각 metric과 시각화 앞에는 “왜 이 분석을 하는지”, 뒤에는 “무엇을 의미하며 무엇을 의미하지 않는지”를 짧게 설명한다.

## 9. WP3 제출 산출물 체크리스트

### 코드와 notebook

- [ ] 입력 파일 경로와 실행 환경 설명
- [ ] WP2 QA 결과 확인 셀
- [ ] graph schema를 표 또는 코드로 제시
- [ ] assertion/collapsed/co-occurrence graph 생성 코드
- [ ] metric 계산 코드와 해석
- [ ] 시각화와 legend
- [ ] 주요 insight와 evidence
- [ ] 한계와 개선 방향
- [ ] 팀원 기여 및 AI 도구 공개

### 데이터 파일

- [ ] `nodes.parquet` 및 검토용 CSV
- [ ] `typed_edges_assertions.parquet`
- [ ] `typed_edges_collapsed.parquet` 및 CSV
- [ ] `cooccurrence_edges.parquet`
- [ ] `knowledge_graph.graphml`
- [ ] `knowledge_graph.json`
- [ ] `graph_schema.json`
- [ ] `node_metrics.csv`
- [ ] `graph_summary.json`
- [ ] `DATA_DICTIONARY.md`
- [ ] 실행 설정과 데이터 hash를 담은 manifest

### 그림

- [ ] 전체 또는 핵심 subgraph
- [ ] relation-specific graph
- [ ] co-occurrence community graph
- [ ] centrality 비교 chart
- [ ] temporal trend chart

## 10. WP3 완료 기준

다음 조건을 만족하면 Stage 1 WP3가 실질적으로 완료됐다고 볼 수 있다.

1. 모든 graph edge가 존재하는 node를 가리킨다.
2. typed graph와 co-occurrence graph가 분리되어 있다.
3. assertion과 unique triple의 차이를 notebook이 설명한다.
4. metric 계산에 사용한 graph view와 edge weight가 명시되어 있다.
5. 최소 하나의 centrality 분석과 하나의 clustering/community 분석이 있다.
6. 시각화가 읽을 수 있는 수준으로 필터링되어 있고 legend가 있다.
7. 주요 insight가 원문 evidence와 연결된다.
8. graph와 schema가 파일로 저장되고 다시 로드되는지 검증된다.
9. Stage 3에서 사용할 entity ID, relation type, `document_id`, `text_unit_id`, 날짜가 보존된다.
10. notebook에 팀원 기여와 AI 도구 사용이 공개된다.

---

## 11. 공식 Stage 3 RAG 평가와의 연결

공식 Stage 3에서는 Stage 1 knowledge graph를 실제 RAG 시스템에 통합하고 text-only baseline과 비교해야 한다. 따라서 WP3는 시각화용 그래프뿐 아니라 retrieval이 가능한 데이터 구조를 남겨야 한다.

공식 Stage 3의 주요 평가 요구사항은 다음과 같다.

### Q&A dataset

- Open Models를 대표하는 passage 약 100-150개
- RAG 생성 모델과 다른 LLM을 사용해 Q&A 약 200-300개 생성
- passage당 일반적으로 1-3개 질문
- factual, temporal, relational, analytical, comparative 질문 혼합
- clarity, correctness, diversity, coverage, answerability 수동 검수

### RAG 비교

- text-only RAG baseline 구현
- dense, sparse 또는 hybrid retrieval 실험
- Stage 1 graph를 사용하는 graph-aware 또는 entity-anchored retrieval 구현
- text-only와 graph-enhanced RAG의 retrieval 및 answer quality 비교
- 시간 질문에서 발행일과 chronology 반영

### 평가축

- Faithfulness: source evidence와의 일치
- Relevance: 질문에 직접 답하는 정도
- Context grounding: passage와 graph evidence의 적절한 사용
- Temporal grounding: 날짜와 시간 순서의 정확성

### Enhancement

- 평가 결과에서 baseline 또는 graph-enhanced RAG의 한계 하나를 찾는다.
- re-ranking, query expansion, graph expansion, improved chunking, retrieval filtering 중 하나처럼 실행 가능한 개선을 구현한다.
- 개선 전후의 retrieval 또는 answer quality를 평가한다.
- 효과, 한계, trade-off를 논의한다.

이 요구사항 때문에 WP3 graph에는 다음 연결이 반드시 남아야 한다.

```text
entity_id
  -> relation assertion
  -> text_unit_id
  -> evidence sentence
  -> document_id
  -> title / URL / published_at
```

이 chain이 유지되어야 graph가 검색한 구조적 관계를 원문 passage로 검증할 수 있고, faithfulness와 temporal grounding을 평가할 수 있다.

## 12. WP2와 WP3 사이에서 지금 결정해야 할 사항

WP3 구현 전에 팀이 합의해야 할 핵심 선택은 다음과 같다.

1. 수동 relation review를 WP3 graph 생성 전에 완료할지, 우선 `unreviewed` 표시로 진행할지
2. main visualization에 포함할 entity type과 최소 document count
3. collapsed edge weight를 `document_count`로 할지 다른 값으로 할지
4. `BELONGS_TO_FAMILY`를 centrality 계산에 포함할지 별도 layer로 둘지
5. generic organization/person/product node를 주 graph에서 제외할지
6. 전체 3,584개 후보 실행을 WP3 metric 계산 전에 완료할지
7. 최종 Stage 3 RAG가 사용할 graph serialization 형식

권장 순서는 다음과 같다.

> relation sample 검수 → 규칙 보완 → 전체 후보 실행 → WP2 QA 재확인 → WP3 graph 생성 → metric/시각화 → Stage 3 loading test

## 13. 결론

Stage 1 WP3의 목표는 WP2 결과를 단순히 NetworkX 객체로 변환하는 것이 아니다. 평가기준을 만족하려면 topic-specific schema, 목적별 graph view, network metric, 읽을 수 있는 시각화, evidence 기반 insight, schema/data dictionary, Stage 3 재사용 인터페이스까지 완성해야 한다.

WP2가 제공한 stable entity ID, typed relation, 공동출현, evidence sentence, 문서 URL과 날짜는 WP3의 핵심 입력이다. WP3는 이를 손실 없이 그래프로 구성하면서 반복 assertion과 unique triple을 구분하고, 분석용 graph와 retrieval용 evidence store를 함께 보존해야 한다. 이 설계를 지키면 Stage 1 평가 요구사항과 공식 Stage 3의 graph-enhanced RAG 요구사항을 동시에 만족시키는 기반이 된다.

## 14. 참고한 평가기준

- `CLT_AI_Trend_Project.pdf`, Stage 1 `Knowledge Graph Construction` 및 `Outputs`, 3쪽
- `CLT_AI_Trend_Project.pdf`, Stage 3 `Implementing an RAG System for Question Answering`, 4-5쪽
- `CLT_AI_Trend_Project.pdf`, 공통 `Requirements`, 5쪽
