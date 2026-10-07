# Stage 1 WP2 → WP3 운영·검수·인계 가이드

이 문서는 지금까지 구현된 작업을 정리하고, 팀의 공식 **Stage 1 WP2 지식 추출 결과**가
도착했을 때 WP3 지식그래프를 완성하는 순서와 합격 기준을 정의한다.

## 1. 현재 상태 요약

현재 저장소에는 공식 WP2 결과가 오기 전에 전체 흐름을 검증하기 위한 Open Models 파일럿이
있다. 파일럿 수치는 개발과 구조 검증용이며 최종 프로젝트 결과로 확정하면 안 된다.

구현 완료된 범위는 다음과 같다.

### WP2 지식 추출 기반

- Open Models 문서 선택, 문장 분리, entity mention과 canonical entity 생성
- typed relation extraction과 assertion status 처리
- entity co-occurrence edge 생성
- relation에서 document, text unit, evidence sentence로 이어지는 provenance 보존
- schema, QA report, run manifest, data dictionary 출력
- predicate별 relation review sample 생성

주요 구현:

- `src/wp2_open_models/`
- `scripts/run_stage1_wp2.py`
- `notebooks/stage1_wp2_open_models.ipynb`

### WP2 수동 검수 체계

- `TRUE`, `FALSE`, `AMBIGUOUS` 판정 지원
- predicate별 precision과 전체 precision 계산
- 검수자, 오류 유형, 검수 메모 완전성 검사
- 오류 유형별 집계와 PASS/PENDING/FAIL 판정

주요 구현:

- `docs/WP2_RELATION_REVIEW_GUIDE_KO.md`
- `scripts/summarize_relation_reviews.py`
- `src/wp2_open_models/review.py`

### WP3 계약 기반 지식그래프

- WP2의 파일명과 열 이름 차이를 alias로 흡수하는 입력 계약
- evidence assertion graph, collapsed typed graph, co-occurrence graph 생성
- 전체 graph와 predicate별 PageRank, degree, betweenness, clustering 계산
- 관계 유형 요약, 월별 관계·entity·edge 변화 분석
- weighted co-occurrence community 탐지
- 전체 graph, 관계별 subgraph, 시간 변화와 community 시각화
- GraphML, node-link JSON, Parquet/CSV, graph schema와 manifest 출력
- 원문 검토 전용 finding candidate 생성

주요 구현:

- `config/wp3_open_models_contract.json`
- `src/wp3_knowledge_graph/`
- `scripts/run_stage1_wp3.py`
- `notebooks/stage1_wp3_open_models_contract.ipynb`

### Stage 3 인계 기반

- 질문에서 canonical entity와 alias 식별
- 1~3 hop typed graph expansion
- predicate와 날짜 필터
- graph triple과 함께 source passage, URL, 발행일 반환

주요 구현:

- `src/stage3_graph_rag/`
- `scripts/query_graph_evidence.py`
- `docs/STAGE3_GRAPH_RETRIEVER_KO.md`

### 자동 품질 검사

- WP1/WP2/WP3 필수 파일과 notebook 실행 상태 검사
- ID 고유성, dangling endpoint, mention offset, schema, provenance 검사
- graph 재로딩, metric view, temporal table, community, figure 검사
- 자동화할 수 없는 승인을 PENDING으로 분리

주요 구현:

- `tests/quality_gates/`
- `tests/test_quality_gate_engine.py`

현재 파일럿 기준 상태는 다음과 같다.

| Work package | PASS | PENDING | FAIL | 의미 |
|---|---:|---:|---:|---|
| WP1 | 0 | 1 | 5 | 팀의 공식 WP1 산출물이 아직 없음 |
| WP2 | 27 | 3 | 0 | 구조는 통과, relation 수동 검수 미완료 |
| WP3 | 47 | 3 | 0 | 자동 검사는 통과, 공식 계약·수동 승인 미완료 |

전체 Python test suite는 현재 23개 테스트를 통과한다.

## 2. 공식 WP2에서 받아야 하는 파일

최소 필수 파일은 `entities`와 `relations`이지만, 최종 제출과 Stage 3까지 고려하면 아래 묶음을
받아야 한다.

| 파일 | 필요한 핵심 내용 | 사용처 |
|---|---|---|
| `entities.parquet` 또는 CSV | entity ID, canonical name, entity type | graph node |
| `relations.parquet` 또는 CSV | relation ID, subject, predicate, object | typed graph edge |
| `documents.parquet` 또는 CSV | document ID, title, URL, 발행일 | 출처와 시간 근거 |
| `text_units.parquet` 또는 CSV | text unit ID, document ID, passage text | Stage 3 evidence retrieval |
| `cooccurrence_edges.parquet` 또는 CSV | entity pair, sentence/document count | community 분석 |
| `schema.json` | relation별 허용 subject/object type | schema validation |
| `relation_review_sample.csv` | 표본 relation과 검수 열 | precision 평가 |
| `qa_report.json` | upstream validation 결과 | WP2 품질 확인 |
| `run_manifest.json` | 버전, 설정, input hash | 재현성 확인 |
| `DATA_DICTIONARY.md` | 열의 의미와 단위 | 팀 인계 |

각 relation에는 최종적으로 다음 provenance가 있어야 한다.

`relation_id → text_unit_id → document_id → URL / published_at`

`evidence_text`도 relation 행에 유지한다. Graph triple만 있고 원문으로 돌아갈 수 없다면 Stage 3
faithfulness와 temporal grounding을 검증할 수 없으므로 최종 입력으로 승인하지 않는다.

## 3. 공식 WP2가 도착한 뒤 실행 순서

### Step 1. 원본 전달물을 별도 폴더에 보존한다

예시 경로:

```text
outputs/stage1_wp2_open_models_official_YYYYMMDD/
```

`outputs/`는 Git에서 제외되어 있다. 원본 파일을 직접 수정하거나 파일명을 임의로 바꾸지 않는다.
전달자, 전달 시각, 버전 설명을 함께 기록하고 가능하면 upstream manifest의 hash를 보존한다.

### Step 2. 파일명과 열을 계약에 대응시킨다

`config/wp3_open_models_contract.json`에서 다음만 수정한다.

- 실제 파일명을 각 table의 `files` 후보에 추가
- 다른 열 이름을 canonical field의 `columns` alias에 추가
- 새 predicate가 있으면 WP2 `schema.json`의 relation signature 확인
- assertion status와 confidence 정책을 팀 결정에 맞게 수정

원본 WP2 파일을 WP3 코드에 맞춰 수동 변환하지 않는다. Contract report에 어떤 alias가
사용됐는지가 남아야 한다.

이 시점에는 `contract_version`의 `draft`를 아직 제거하지 않는다.

### Step 3. WP2 자동 무결성을 먼저 확인한다

공식 폴더를 가리키도록 `tests/quality_gates/quality_contract.json`의 `wp2.output_dir`을 변경하거나,
이 파일을 복사한 공식용 quality contract를 만든다. 원본 notebook 경로도 공식 notebook이 있으면
함께 지정한다.

개발 중 상태를 확인한다.

```powershell
.\.venv\Scripts\python.exe tests\quality_gates\run_quality_gates.py `
  --contract path\to\official_quality_contract.json `
  --report-dir outputs\quality_gates_official `
  --fail-on never
```

WP2에 FAIL이 하나라도 있으면 다음 단계로 가지 않고 upstream 문제를 먼저 고친다.

### Step 4. 공식 relation sample을 사람이 검수한다

`docs/WP2_RELATION_REVIEW_GUIDE_KO.md`에 따라 다음 열을 채운다.

- `review_correct`: `TRUE`, `FALSE`, `AMBIGUOUS`
- `reviewer`
- `review_error_type`: FALSE인 행에 필수
- `review_notes`

집계한다.

```powershell
.\.venv\Scripts\python.exe scripts\summarize_relation_reviews.py `
  --reviews outputs\stage1_wp2_open_models_official_YYYYMMDD\relation_review_sample.csv `
  --output-dir outputs\relation_review_official `
  --minimum-reviewed 90 `
  --minimum-precision 0.80 `
  --fail-on pending
```

결과가 FAIL이면 반복 오탐 유형을 확인하여 WP2 extraction rule 또는 catalogue를 수정하고 새로운
표본으로 다시 평가한다. 좋은 edge만 골라 남겨서 precision을 올리는 방식은 사용하지 않는다.

### Step 5. WP2→WP3 계약을 확정한다

다음 조건이 충족된 후에만 계약을 고정한다.

- 필수 파일과 열이 확정됨
- relation schema와 assertion status 의미가 확정됨
- provenance 누락이 없음
- 수동 검수 기준을 통과함

그 다음 `config/wp3_open_models_contract.json`에서 다음을 반영한다.

- `contract_version`에서 `draft` 제거 후 팀이 합의한 버전 부여
- `strict_provenance: true`
- 필요한 경우 co-occurrence threshold와 relation policy 확정

표본만 검수한 경우 `require_reviewed_true`를 켜면 대부분 relation이 제거되므로 일반적으로
`false`를 유지한다. 대신 `exclude_reviewed_false: true`로 명시적으로 거절된 edge를 제외한다.

### Step 6. 공식 WP3 산출물을 새 폴더에 생성한다

```powershell
.\.venv\Scripts\python.exe scripts\run_stage1_wp3.py `
  --input-dir outputs\stage1_wp2_open_models_official_YYYYMMDD `
  --output-dir outputs\stage1_wp3_open_models_official_YYYYMMDD `
  --contract config\wp3_open_models_contract.json
```

파일럿 output을 덮어쓰지 않는다. 실행이 실패하면 `contract_report.json`과
`validation_report.json`을 먼저 확인한다.

### Step 7. 자동 산출물과 수치를 검증한다

공식 WP3 폴더에서 다음을 확인한다.

- 세 graph view의 node/edge 수가 0이 아님
- assertion 수와 collapsed typed edge 수의 차이가 설명 가능함
- 모든 predicate가 relation summary와 predicate metric에 존재함
- 모든 co-occurrence node가 community에 할당됨
- 월별 표가 비어 있지 않고 날짜 범위가 WP2와 일치함
- GraphML 세 파일을 다시 로드할 수 있음
- 모든 assertion이 WP2 document와 text unit으로 역추적됨
- graph schema에 assertion, collapsed typed, co-occurrence view가 명시됨

수치가 파일럿과 달라지는 것 자체는 오류가 아니다. 입력 문서 수, 날짜 범위, 추출 rule 차이로
변화를 설명할 수 있어야 한다.

### Step 8. WP3 notebook을 공식 경로로 실행한다

`notebooks/stage1_wp3_open_models_contract.ipynb`의 설정 셀에서 다음을 공식 경로로 바꾼다.

- `WP2_INPUT_DIR`
- `WP3_OUTPUT_DIR`
- `CONTRACT_PATH`
- 재생성이 필요하면 `RUN_PIPELINE=True`

모든 code cell을 실행하고 저장한다. 실행 번호가 없는 cell이나 error output이 남아 있으면
최종 notebook으로 승인하지 않는다.

### Step 9. 그림과 finding을 사람이 승인한다

자동 생성된 그림에서 label 겹침, legend, 방향 화살표, threshold를 확인한다.
`graph_finding_candidates`의 후보는 다음을 모두 확인한 뒤에만 최종 insight로 채택한다.

- 최소 하나 이상의 실제 source passage가 주장을 직접 지지함
- 반복 문서 때문에 중요도가 과장되지 않았음
- 공동출현을 typed relation이나 인과관계로 표현하지 않음
- PageRank를 실제 산업 영향력이나 모델 품질 순위로 표현하지 않음
- 시간 변화가 월별 수집 문서 수 변화만으로 설명되지 않음

승인된 finding은 WP1 topic/temporal 결과와 비교하여 최종 Stage 1 서술에 통합한다.

### Step 10. Stage 3 retrieval smoke test를 실행한다

공식 폴더를 사용해 알려진 entity와 relation으로 최소 세 질의를 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts\query_graph_evidence.py `
  "Who developed Gemma 4?" `
  --wp2-dir outputs\stage1_wp2_open_models_official_YYYYMMDD `
  --wp3-dir outputs\stage1_wp3_open_models_official_YYYYMMDD `
  --predicate DEVELOPED_BY `
  --top-k 5
```

각 결과는 relation뿐 아니라 passage, `text_unit_id`, `document_id`, URL, 발행일을 반환해야 한다.
`missing_text_units`와 `missing_documents`는 0이어야 한다.

### Step 11. 최종 quality gate를 실행한다

공식 quality contract에서 다음을 수정한다.

- WP2/WP3 공식 output directory
- 공식 notebook 경로
- `manual_flags`의 팀 승인 결과

최종 실행에서는 PENDING도 실패로 처리한다.

```powershell
.\.venv\Scripts\python.exe tests\quality_gates\run_quality_gates.py `
  --contract path\to\official_quality_contract.json `
  --report-dir outputs\quality_gates_official `
  --fail-on pending
```

## 4. 잘 됐는지 판단하는 기준

### WP2 자동 합격 기준

| 기준 | 합격 조건 |
|---|---|
| 필수 산출물 | 계약에 정의한 모든 필수 파일 존재 |
| Entity ID | 중복 0, null key 0 |
| Relation ID | 중복 0 |
| Relation endpoint | dangling subject/object 0 |
| Mention offset | 원문 불일치 0 |
| Relation schema | invalid signature 0 |
| Provenance | document/text unit/evidence 누락 0 |
| Upstream QA | `hard_validation_passed=true` |
| Mention coverage | 내부 기준 80% 이상 |
| Relation document coverage | 내부 기준 20% 이상 |

Coverage와 최소 개수는 프로젝트 rubric의 공식 절대 기준이 아니라 현재 팀의 sanity threshold다.
공식 데이터 특성에 맞춰 변경할 경우 이유를 notebook에 기록한다.

### WP2 수동 합격 기준

| 기준 | 합격 조건 |
|---|---|
| 검수량 | TRUE/FALSE 판정 90개 이상 |
| Precision | `TRUE / (TRUE + FALSE) ≥ 0.80` |
| Ambiguous | precision 분모에서 제외하고 사유 기록 |
| Reviewer | 작성된 모든 판정에 검수자 존재 |
| Error type | 모든 FALSE 행에 표준 오류 유형 존재 |
| 팀 승인 | review 결과와 수정 여부에 팀 합의 |

Predicate별 표본이 너무 적으면 전체 precision만으로 해당 predicate를 신뢰하지 않는다.

### WP3 자동 합격 기준

| 기준 | 합격 조건 |
|---|---|
| Contract adapter | 모든 필수 입력과 alias 해석 성공 |
| Contract status | 최종 version에 `draft`가 없음 |
| Validation | error 0, `passed=true` |
| Assertion graph | endpoint 누락 0 |
| Collapsed graph | 중복 subject-predicate-object 0 |
| Metrics | typed, co-occurrence와 모든 predicate별 view 존재 |
| Temporal analysis | 날짜가 있는 월별 relation 표가 비어 있지 않음 |
| Community | 모든 co-occurrence node에 community 할당 |
| Provenance chain | assertion의 missing document/text unit 0 |
| Serialization | 세 GraphML 파일 재로딩 성공 |
| Graph schema | 세 graph view와 Stage 3 provenance 명시 |
| Visualization | 요구된 8개 PNG가 읽을 수 있는 크기로 존재 |
| Notebook | 모든 code cell 실행, error output 0 |

### WP3 수동 합격 기준

- 최종 그림이 읽을 수 있고 해석하려는 관계를 정확히 보여 줌
- 최종 insight가 source evidence로 직접 뒷받침됨
- graph metric의 의미를 과장하지 않음
- WP1 topic/temporal 결과와 연결된 설명이 있음
- 파일럿이 아니라 공식 WP2 기반 수치를 사용함
- 팀원 contribution과 AI tool disclosure가 기록됨

### 상태 해석

| 상태 | 의미 | 다음 행동 |
|---|---|---|
| PASS | 정의한 자동 또는 수동 조건 충족 | 다음 단계 진행 가능 |
| WARN | 실행은 가능하지만 해석 시 주의 필요 | 원인과 영향 기록 |
| PENDING | 공식 계약 또는 사람의 승인이 남음 | 개발용 사용만 가능, 최종 확정 금지 |
| FAIL | 구조·schema·provenance·실행 문제가 있음 | 다음 단계 중단 후 수정 |

## 5. 최종 Stage 3 인계 묶음

Stage 3 팀에는 최소한 다음을 전달한다.

- `nodes.parquet`
- `typed_edges_assertions.parquet`
- `typed_edges_collapsed.parquet`
- assertion/collapsed/co-occurrence GraphML 또는 JSON
- `graph_schema.json`
- `documents.parquet`
- `text_units.parquet`
- `validation_report.json`
- `run_manifest.json`
- 승인된 graph finding과 사용한 source evidence 목록

Stage 3에서는 text-only retrieval을 먼저 baseline으로 만들고, graph retriever가 반환한 passage를
추가한 graph-enhanced 방식과 동일한 QA set, answer model, prompt로 비교한다.

## 6. 하지 말아야 할 것

- 공식 WP2 파일을 WP3 코드에 맞추기 위해 수동으로 rename 또는 열 변환
- 검수 결과가 낮다고 오답 표본만 삭제
- generated graph table을 직접 수정하고 upstream relation은 그대로 두기
- 공동출현 edge를 사실 관계로 서술
- centrality를 실제 영향력·품질 순위로 단정
- graph triple만 LLM에 넣고 원문 passage를 근거로 제시하지 않기
- 파일럿 output을 공식 output으로 덮어쓰기
- FAIL 또는 PENDING이 남은 quality report를 최종 완료 증거로 사용

## 7. 최종 완료 정의

WP3는 다음이 모두 충족될 때 완료로 본다.

1. 공식 WP2 자동 검사와 relation review가 통과했다.
2. 확정된 계약과 strict provenance로 공식 WP3를 재생성했다.
3. WP3 자동 quality gate에 FAIL과 PENDING이 없다.
4. 그림과 evidence-backed finding을 팀이 승인했다.
5. Stage 3 retriever smoke test에서 provenance 누락이 없다.
6. WP1 결과와 통합된 최종 notebook이 실행 완료 상태다.

현재 저장소는 위 절차를 실행할 코드와 검사 도구를 갖춘 상태다. 아직 남은 것은 공식 WP1/WP2
전달물, relation 수동 검수, 공식 계약 고정과 팀의 최종 해석 승인이다.
