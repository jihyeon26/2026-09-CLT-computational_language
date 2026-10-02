# Stage 1 WP2 작업 결과 설명서 — Open Models

## 1. 문서 목적

이 문서는 Stage 1의 **WP2: Knowledge Extraction**에서 수행한 작업과 결과를 설명하고, 다음 작업인 **WP3: Knowledge Graph and Outputs**에서 무엇을 이어받아야 하는지 정리한다. 또한 WP2가 Stage 1뿐 아니라 이후 Stage 2의 모델 학습·평가와 Stage 3의 RAG 시스템에서 어떤 역할을 하는지도 설명한다.

현재 결과는 전체 후보를 처리한 최종본이 아니라, 방법과 데이터 구조를 먼저 검증하기 위해 300개 문서를 사용한 **재현 가능한 파일럿 baseline**이다.

---

## 2. 전체 프로젝트에서 WP2의 위치

프로젝트의 데이터 흐름은 다음과 같다.

> 원문 기사 → WP1의 데이터·주제 분석 → **WP2의 구조화된 지식 추출** → WP3의 지식 그래프 → Stage 2의 표현 학습·모델 비교 → Stage 3의 RAG 및 질의응답

WP1은 어떤 문서를 분석할지 정하고 텍스트의 전반적 특성을 파악한다. WP2는 그 비정형 텍스트 안에서 모델, 조직, 라이선스, 아키텍처 등의 엔티티와 관계를 추출해 기계가 사용할 수 있는 구조로 바꾼다. WP3는 WP2가 만든 엔티티와 관계를 실제 그래프로 구성하고, 그래프 통계와 시각화를 만든다.

따라서 WP2는 단순한 중간 전처리가 아니라, **텍스트 분석과 지식 그래프를 연결하는 인터페이스**다. WP2의 엔티티 ID, 관계 방향, 근거 문장 구조가 불안정하면 WP3의 centrality, community, 시각화 결과도 왜곡되고, Stage 3의 Graph-RAG 역시 잘못된 정보를 검색하게 된다.

## 3. WP1에서 선행되어야 하는 최소 작업

WP2를 시작하기 위해 WP1 전체 결과가 완성될 필요는 없다. 하지만 다음의 최소 입력 정의는 필요하다.

- 분석 기간: 2024-09-01부터 2026-08-31까지
- 트랙: Open Models
- 원본 열: `title`, `date`, `content`, `domain`, `url`, `tags`
- 직렬화된 본문 리스트의 파싱과 Unicode/공백 정규화
- 문서와 문장을 추적할 수 있는 안정적인 ID
- 원본 URL 및 발행일 보존

이번 파일럿에서는 위 조건을 WP2 파이프라인 내부에 포함했다. 따라서 팀의 WP1 결과가 나중에 전달되면 후보 문서 목록과 전처리 기준만 비교·조정하면 된다. Topic Modeling 결과는 WP2 실행의 필수 입력은 아니지만, 하위 주제별 추출 품질을 비교하거나 문서 표본을 설계할 때 사용할 수 있다.

---

## 4. 입력 데이터와 파일럿 범위

원본은 `dataset/ai_media_dataset_20260911.csv`이다. 원본 파일은 수정하지 않고 읽기 전용 입력으로 사용했다.

| 항목 | 값 |
|---|---:|
| 지정 기간 내 전체 문서 | 47,508 |
| 제목 또는 태그 기준 Open Models 후보 | 3,584 |
| 파일럿 선택 문서 | 300 |
| 추출된 문장 단위 | 26,579 |
| 랜덤 시드 | 42 |
| 사용 NER 모델 | spaCy `en_core_web_sm` |

300개 문서는 특정 시기의 기사만 과대표집하지 않도록 월별로 나누어 선택했다. 후보 선정은 제목과 데이터셋의 기존 태그를 사용했으며, 본문에 특정 단어가 우연히 한 번 등장했다는 이유만으로 트랙 문서에 포함되는 것을 피했다.

---

## 5. 수행한 작업

### 5.1 문서와 문장 단위 생성

- `content` 열에 저장된 Python 리스트 형태의 문자열을 파싱했다.
- Unicode와 과도한 공백을 정규화했다.
- URL의 해시를 이용해 안정적인 `document_id`를 생성했다.
- 본문을 문장으로 나누고 각 문장에 `text_unit_id`를 부여했다.
- 모든 엔티티 mention에 문서 기준 시작·종료 offset과 근거 문장을 보존했다.

이 구조 덕분에 그래프 edge에서 다시 원문 문장과 기사 URL로 돌아갈 수 있다.

### 5.2 엔티티 추출과 정규화

두 가지 추출 방식을 결합했다.

1. **도메인 EntityRuler**: Open Models 트랙에서 중요한 모델과 개념을 curated catalogue로 정의한다.
2. **통계적 NER**: spaCy 모델을 사용해 추가 조직, 사람, 제품 후보를 찾는다.

통계적 NER 결과는 도메인 엔티티가 함께 등장한 문장에서만 보존했다. 핵심 모델과 모델 패밀리는 stable canonical ID로 연결했다. 예를 들어 문장 표면형이 달라도 동일한 모델로 판단되면 같은 ID를 사용한다.

사용한 엔티티 타입은 다음과 같다.

| 타입 | 의미 | 파일럿 엔티티 수 |
|---|---|---:|
| `MODEL` | 구체적인 모델 또는 모델 버전 | 19 |
| `MODEL_FAMILY` | Llama, Gemma, Qwen과 같은 모델 계열 | 10 |
| `ORGANIZATION` | 기업, 연구기관, 프로젝트 조직 후보 | 720 |
| `PERSON` | 사람 이름 후보 | 442 |
| `PRODUCT_OTHER` | 모델 이외의 제품 후보 | 84 |
| `CAPABILITY` | reasoning, tool use 등의 기능 | 6 |
| `ARCHITECTURE` | Transformer, MoE 등의 구조 | 4 |
| `LICENSE` | Apache 2.0 등의 라이선스 | 3 |

`ORGANIZATION`, `PERSON`, `PRODUCT_OTHER`에는 통계적 NER의 일반 엔티티가 포함되므로, 이 숫자를 곧바로 “Open Models 생태계의 조직·인물 수”로 해석하면 안 된다.

### 5.3 관계 추출

관계는 허용된 subject/object 타입과 방향을 미리 정의한 뒤, 해당 스키마를 만족하는 경우에만 만들었다.

| 관계 | 방향 | 의미 |
|---|---|---|
| `DEVELOPED_BY` | 모델/패밀리 → 조직 | 개발 주체 |
| `RELEASED_BY` | 모델/패밀리 → 조직 | 공개 또는 출시 주체 |
| `BELONGS_TO_FAMILY` | 모델 → 모델 패밀리 | 모델의 상위 계열 |
| `SUCCESSOR_OF` | 모델 → 모델 | 후속 모델 관계 |
| `DERIVED_FROM` | 모델 → 모델/패밀리 | 기반 모델 또는 파생 관계 |
| `LICENSED_UNDER` | 모델/패밀리 → 라이선스 | 적용 라이선스 |
| `USES_ARCHITECTURE` | 모델/패밀리 → 아키텍처 | 사용 구조 |
| `HAS_CAPABILITY` | 모델/패밀리 → 기능 | 명시된 기능 |

조직–모델 관계와 파생 관계에서는 단순 공동등장만 사용하지 않고, `released`, `developed`, `based on` 등의 동사 위치와 엔티티 순서를 이용했다. 특히 다음 오탐을 줄이는 규칙을 추가했다.

- 비교 대상 모델을 개발·출시 대상으로 잘못 연결하는 경우
- `Qwen-Agent`에서 `Qwen`만 잘라 모델 출시 관계로 인식하는 경우
- 같은 문장에 반복 등장한 조직 중 동사와 멀리 떨어진 mention을 선택하는 경우
- 같은 모델의 반복 mention으로 `Gemma 4 → Gemma 4`와 같은 self-edge를 만드는 경우
- 일반 NER가 기사 머리말이나 모델명을 조직으로 잘못 분류하는 경우

모든 관계는 `document_id`, `text_unit_id`, `evidence_text`, 발행일을 가진다. 즉, 그래프의 edge는 근거가 없는 사실 목록이 아니라 **원문 evidence에 연결된 relation assertion**이다.

### 5.4 엔티티 공동출현

같은 문장에서 함께 등장한 서로 다른 엔티티 쌍을 계산했다.

- 공동출현 edge: 4,230개
- 각 edge에 함께 등장한 문장 수와 문서 수 기록
- 처음·마지막 관찰일 기록
- 빈도가 높은 75개 엔티티에 대한 dense matrix 별도 생성

공동출현은 의미 관계가 아니다. 예를 들어 두 모델이 비교 기사에서 함께 등장했다고 해서 한 모델이 다른 모델을 개발하거나 계승했다는 의미는 아니다. WP3에서는 typed relation graph와 co-occurrence graph를 분리해서 분석해야 한다.

---

## 6. 주요 결과와 해석

### 6.1 추출량

| 지표 | 결과 |
|---|---:|
| 엔티티 mention | 8,850 |
| canonical entity | 1,288 |
| relation assertion | 1,647 |
| unique `(subject, predicate, object)` triple | 68 |
| 엔티티가 하나 이상 있는 문서 | 296 / 300 |
| 관계가 하나 이상 있는 문서 | 177 / 300 |
| 엔티티 문서 coverage | 98.67% |
| 관계 문서 coverage | 59.00% |

여기서 **relation assertion**과 **unique triple**은 반드시 구분해야 한다. 같은 triple이 여러 기사 또는 여러 문장에서 반복 관찰되면 각 근거는 별도의 assertion으로 저장된다. 예를 들어 `Llama 3.2 BELONGS_TO_FAMILY Llama`가 여러 문장에 등장할 수 있다.

### 6.2 관계 타입별 assertion 수

| 관계 | assertion 수 | unique triple 수 |
|---|---:|---:|
| `BELONGS_TO_FAMILY` | 1,550 | 19 |
| `USES_ARCHITECTURE` | 36 | 15 |
| `RELEASED_BY` | 17 | 9 |
| `LICENSED_UNDER` | 17 | 9 |
| `DEVELOPED_BY` | 14 | 4 |
| `HAS_CAPABILITY` | 11 | 11 |
| `DERIVED_FROM` | 2 | 1 |
| `SUCCESSOR_OF` | 0 | 0 |

`BELONGS_TO_FAMILY`가 전체 assertion 대부분을 차지한다. 이것은 1,550개의 서로 다른 사실이 있다는 뜻이 아니라, catalogue에 정의된 모델–패밀리 관계가 많은 evidence 문장에서 반복 확인됐다는 뜻이다.

### 6.3 자동 무결성 검사

다음 hard validation 항목을 모두 통과했다.

- 중복 문서 ID: 0
- 중복 mention ID: 0
- 중복 엔티티 ID: 0
- 중복 relation ID: 0
- mention offset 불일치: 0
- 존재하지 않는 엔티티를 가리키는 relation: 0
- evidence 문장 또는 문서가 없는 relation: 0
- 허용되지 않은 relation signature: 0
- subject와 object가 같은 self-relation: 0

이 검사는 파일 구조와 그래프 무결성이 올바르다는 뜻이다. 그러나 relation의 의미가 실제 문장과 일치하는지까지 자동으로 보장하지는 않는다.

---

## 7. 결과를 사용할 때 반드시 알아야 할 점

### 7.1 `rule_score`는 확률이 아니다

`rule_score`는 추출 규칙의 상대적 신뢰도를 나타내기 위한 heuristic ranking 값이다. 예를 들어 `0.85`를 “85% 정확도”라고 보고하면 안 된다. 실제 precision은 수동 검수 결과로 계산해야 한다.

### 7.2 현재 결과는 파일럿이다

전체 Open Models 후보 3,584개 중 300개만 처리했다. 현재의 빈도, 중심성, 시계열 분포를 전체 데이터셋의 결론으로 보고하면 안 된다. 파일럿의 목적은 스키마와 추출 방법을 검증하는 것이다.

### 7.3 자동 NER에는 오탐과 누락이 있다

일반 조직·사람·제품 엔티티에는 spaCy의 통계적 예측이 포함되어 있다. 반대로 catalogue에 아직 없는 새로운 모델명은 일반 제품으로 남거나 추출되지 않을 수 있다. 최종 실행 전에 catalogue를 보강해야 한다.

### 7.4 관계가 없는 문서는 실패 문서가 아니다

300개 중 123개 문서에는 현재 스키마의 관계가 없다. 이는 문서가 무의미하다는 뜻이 아니다. 단순 의견 기사, 벤치마크 비교, 정책 기사처럼 현재 정의한 관계를 명시적으로 말하지 않는 문서일 수 있다.

### 7.5 발행일과 사건 발생일은 다르다

현재 `published_at`은 기사 발행일이다. 문장 안에 별도의 출시일이나 사건 날짜가 있더라도 구조화된 event date로 추출하지 않았다. 따라서 시계열 분석에서 “언급된 날짜”와 “실제 출시일”을 혼동하면 안 된다.

### 7.6 `assertion_status`는 얕은 어휘 규칙이다

부정이나 불확실성을 나타내는 단어를 이용해 `asserted`, `negated`, `uncertain`을 구분하지만, 완전한 자연어 추론 결과가 아니다. 특히 부정 관계는 WP3 그래프에 사실 edge로 넣기 전에 검토해야 한다.

---

## 8. WP3가 이어받아야 할 파일

WP3의 canonical input은 다음 두 파일이다.

- `entities.parquet`: 노드 테이블
- `relations.parquet`: evidence-bearing typed edge 테이블

보조 입력은 다음과 같다.

| 파일 | WP3에서의 용도 |
|---|---|
| `entity_mentions.parquet` | 엔티티가 등장한 원문 위치 추적 |
| `text_units.parquet` | edge evidence와 원문 문장 연결 |
| `documents.parquet` | 제목, 날짜, 도메인, URL 등 문서 메타데이터 |
| `cooccurrence_edges.parquet` | 별도의 무방향 공동출현 그래프 구축 |
| `cooccurrence_matrix_top75.csv` | heatmap 또는 군집화 입력 |
| `schema.json` | 허용 노드·edge 타입 확인 |
| `qa_report.json` | 입력 무결성과 coverage 보고 |
| `run_manifest.json` | 데이터 hash, 설정, 라이브러리 버전 확인 |
| `knowledge_graph.graphml` | Gephi 등 시각화 도구의 초기 입력 |
| `knowledge_graph.json` | NetworkX 기반 코드에서 빠르게 재사용 |

CSV 파일은 사람이 확인하기 위한 편의 형식이고, 코드에서는 타입과 용량 측면에서 Parquet 사용을 권장한다.

---

## 9. WP3에서 결정해야 할 사항

### 9.1 assertion graph와 collapsed graph를 구분한다

`relations.parquet`의 한 행은 하나의 evidence-bearing assertion이다. 따라서 동일 triple이 반복될 수 있다.

WP3에서는 목적에 따라 두 그래프를 만들 수 있다.

1. **Assertion MultiGraph**: 각 evidence를 별도의 edge로 유지한다. 출처 추적과 Stage 3 RAG에 유리하다.
2. **Collapsed Entity Graph**: 동일 `(subject, predicate, object)`를 하나의 edge로 합치고 `evidence_count`, `document_count`, `first_seen`, `last_seen`을 속성으로 저장한다. 통계와 시각화에 유리하다.

발표용 그래프 metric은 collapsed graph를 기본으로 하고, evidence 조회에는 assertion graph 또는 원본 relation table을 사용하는 것이 적절하다.

### 9.2 typed relation과 공동출현을 섞지 않는다

- Typed edge: `DEVELOPED_BY`, `LICENSED_UNDER`처럼 의미와 방향이 있다.
- Co-occurrence edge: 같은 문장에 등장했다는 의미만 있고 기본적으로 무방향이다.

두 종류를 한 그래프에 넣을 경우 edge type을 명시해야 하며, centrality 계산에서는 분리된 결과도 함께 제시해야 한다.

### 9.3 edge weight를 단순 빈도로만 정의하지 않는다

반복된 모델 카드나 중복 기사 때문에 mention 수가 커질 수 있다. 다음 값을 별도로 보존하는 것이 좋다.

- `evidence_count`: relation assertion 수
- `document_count`: 서로 다른 source document 수
- `domain_count`: 서로 다른 출처 도메인 수
- `first_seen`, `last_seen`: 발행일 기준 관찰 범위
- `reviewed_precision` 또는 검수 상태

그래프 시각화의 기본 edge weight로는 단순 문장 수보다 `document_count`가 더 보수적이다.

### 9.4 부정 및 미검수 관계를 처리한다

WP3 기본 그래프에는 다음 정책을 권장한다.

- `assertion_status == asserted`만 기본 edge로 사용
- `negated`와 `uncertain`은 별도 테이블 또는 다른 edge style로 보존
- 수동 검수에서 `FALSE`인 관계는 제외
- 검수되지 않은 관계는 `unreviewed`로 표시

### 9.5 그래프 metric의 해석 단위를 명시한다

권장 metric은 다음과 같다.

- 노드 수와 edge 수
- connected components
- degree 및 weighted degree
- PageRank 또는 eigenvector centrality
- betweenness centrality
- relation type별 subgraph 통계
- 모델 패밀리–조직 bipartite view
- 시간 구간별 등장·관계 변화

다만 현재 파일럿에서는 Llama/Gemma 관련 문서와 catalogue 관계가 많이 반복되므로, centrality를 “실제 산업 영향력”으로 단정해서는 안 된다. 이는 현재 표본과 추출 스키마 안에서의 연결성이다.

---

## 10. WP3 이전에 해야 할 수동 검수

`relation_review_sample.csv`에는 predicate별로 나눈 87개 검수 표본이 있다. 다음 열을 팀원이 채워야 한다.

- `review_correct`: `TRUE` 또는 `FALSE`
- `reviewer`: 검수자 이름 또는 이니셜
- `review_notes`: 오탐 이유 또는 catalogue 수정 제안

검수할 때는 subject와 object 이름만 보지 말고 `evidence_text`와 기사 URL을 함께 확인한다. 특히 다음 오류 유형을 기록한다.

- 관계 방향 오류
- 비교 대상을 실제 주체로 잘못 선택
- 모델 패밀리와 구체 모델 버전 혼동
- 조직명 오인식
- 부정·추정 문장을 확정 사실로 처리
- 문장 경계가 너무 넓어 서로 다른 절을 연결

predicate별 precision은 다음처럼 계산할 수 있다.

> precision = 올바른 것으로 판정된 표본 수 / 해당 predicate의 전체 검수 표본 수

별도의 gold annotation이 없으므로 현재 단계에서 recall을 보고해서는 안 된다.

## 11. WP3 시작 전 권장 완료 조건

- [ ] 87개 relation review sample 검수 완료
- [ ] predicate별 precision 계산
- [ ] 주요 오탐 규칙 수정
- [ ] 새 모델·라이선스·조직 alias catalogue 보강
- [ ] 중복 또는 near-duplicate 기사 처리 방침 결정
- [ ] 전체 3,584개 후보에 파이프라인 실행
- [ ] 전체 실행 후 hard validation 재확인
- [ ] schema와 pipeline version 고정
- [ ] WP3가 사용할 asserted/reviewed edge 필터 결정

---

## 12. 이후 Stage 2와 Stage 3에서의 활용

### Stage 2 — Embeddings and Model Training

WP2 결과는 다음 용도로 사용할 수 있다.

- 동일 엔티티가 등장하는 문장끼리 embedding 구조 비교
- 관계가 있는 엔티티 쌍을 positive pair 또는 평가 예제로 구성
- 모델·조직·라이선스별 문맥 embedding 시각화
- rule-based extraction과 fine-tuned model extraction의 비교 평가

단, WP2의 자동 추출 결과 전체를 그대로 학습 정답으로 쓰면 기존 규칙의 오류를 학습하게 된다. 수동 검수된 subset을 evaluation 또는 fine-tuning 데이터로 사용하는 것이 안전하다.

### Stage 3 — RAG and Question Answering

WP2와 WP3의 결과는 Graph-Enhanced RAG에서 다음처럼 사용된다.

- 질문에서 모델명 또는 조직명을 entity ID로 연결
- 관련 graph neighbor를 탐색
- relation edge의 `text_unit_id`를 통해 근거 문장을 회수
- 동일 triple을 지지하는 서로 다른 문서와 도메인을 비교
- 출시, 라이선스, 개발 조직, 아키텍처 관련 구조적 질문에 답변

그래프 triple만으로 답을 생성하면 안 된다. 최종 답변에는 반드시 `text_units.parquet`와 `documents.parquet`에서 회수한 원문 evidence를 함께 사용해야 한다. 이 원칙이 Stage 3 평가의 faithfulness와 provenance를 높인다.

---

## 13. 재현 방법

프로젝트 루트에서 다음 명령으로 300문서 파일럿을 다시 실행할 수 있다.

```powershell
.\.venv\Scripts\python.exe scripts\run_stage1_wp2.py --max-docs 300
```

파일럿 검수와 catalogue 수정을 완료한 후 전체 후보를 처리하려면 다음을 사용한다.

```powershell
.\.venv\Scripts\python.exe scripts\run_stage1_wp2.py --max-docs 0
```

테스트는 다음과 같이 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_wp2_open_models.py -q
```

현재 테스트 5개가 통과했으며, 실행된 노트북의 코드 셀 8개에서도 오류가 발생하지 않았다.

## 14. 주요 파일 위치

- 실행 노트북: `notebooks/stage1_wp2_open_models.ipynb`
- 파이프라인: `src/wp2_open_models/pipeline.py`
- 엔티티 catalogue와 schema: `src/wp2_open_models/catalog.py`
- 실행 스크립트: `scripts/run_stage1_wp2.py`
- 테스트: `tests/test_wp2_open_models.py`
- 의존성: `requirements-wp2.txt`
- 결과 디렉터리: `outputs/stage1_wp2_open_models_pilot/`
- 결과 데이터 사전: `outputs/stage1_wp2_open_models_pilot/DATA_DICTIONARY.md`

## 15. 결론

이번 WP2 파일럿은 Open Models 관련 기사에서 엔티티, typed relation, 공동출현 정보를 추출하고, 모든 관계를 원문 evidence와 연결하는 구조를 마련했다. 자동 무결성 검사는 통과했지만, 이것은 의미적 정확도가 완성됐다는 뜻은 아니다.

WP3에서 가장 중요한 것은 관계 assertion을 바로 하나의 단순 그래프로 간주하지 않는 것이다. 반복 evidence를 합친 collapsed graph와 출처를 유지한 assertion graph를 구분하고, typed relation과 공동출현을 분리하며, 수동 검수 결과를 edge 필터에 반영해야 한다. 이 과정을 지키면 Stage 1의 그래프 분석뿐 아니라 Stage 3의 evidence-grounded Graph-RAG까지 동일한 데이터 구조를 안정적으로 재사용할 수 있다.
