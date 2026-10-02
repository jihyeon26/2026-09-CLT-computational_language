# Stage 1 WP3 준비도와 최종 완료 조건

## 결론

현재 WP3는 **공식 WP2 산출물을 입력하면 동일 파이프라인으로 그래프·분석·시각화·Stage 3용
파일을 다시 생성할 수 있는 수준**이다. 코드와 산출물 구조는 준비됐지만, 파일럿 결과를 최종
결과라고 확정할 수는 없다.

WP3 그래프 생성의 직접 입력은 WP2이다. WP1 결과는 그래프 생성의 필수 입력은 아니지만,
최종 Stage 1 notebook에서 topic/temporal 결과와 graph finding이 일치하는지 비교하여 해석을
통합해야 한다.

## 공식 WP2에서 받아야 할 것

필수 입력:

- `entities`: entity ID, 이름, 유형
- `relations`: subject, predicate, object
- `schema.json`: 허용 relation과 subject/object type 조합
- 각 relation의 `document_id`, `text_unit_id`, `evidence_text`, `published_at`

최종 분석과 Stage 3를 위해 사실상 필요한 입력:

- `documents`와 `text_units`: relation에서 원문 passage와 URL로 돌아가기 위한 provenance
- `cooccurrence_edges`: community와 담론 연결 구조 분석
- `relation_review_sample.csv`: 최소 50개 검수 및 내부 기준 precision 0.80 이상

파일명이나 열 이름이 달라도 계약 JSON alias로 흡수할 수 있다. 공식 형식이 정해지면 계약의
`contract_version`에서 `draft`를 제거하고 `strict_provenance=true`로 변경한다.

## 자동으로 완성되는 부분

- Assertion, collapsed typed, co-occurrence graph 생성
- 전체 graph와 7개 predicate별 centrality/clustering 계산
- 관계 유형별 요약과 월별 relation/entity/edge 변화 표
- weighted co-occurrence community 탐지
- 전체 graph와 관계별 subgraph를 포함한 8개 시각화
- GraphML, node-link JSON, Parquet/CSV 저장
- Stage 3용 graph schema와 provenance chain 저장
- 원문 검토용 finding candidate 생성
- 계약, schema, endpoint, provenance, notebook 실행, 그림 형식에 대한 자동 품질 검사

## 자동화할 수 없어 마지막에 사람이 해야 할 것

1. WP2 relation 표본을 검수하고 precision을 승인한다.
2. 공식 입력으로 재실행한 그림의 label, legend, threshold를 확인한다.
3. `graph_finding_candidates`를 원문 evidence와 대조해 accepted/rejected로 판정한다.
4. 승인된 finding만 WP1 topic/temporal 결과와 함께 최종 notebook 서술에 사용한다.
5. 팀원 contribution과 AI tool disclosure를 최종 notebook에 기록한다.

## 현재 파일럿 상태

- 입력 entity: 1,288
- 포함 relation assertion: 1,644
- unique typed edge: 66
- co-occurrence edge: 4,230
- relation type: 7
- co-occurrence community: 23
- 검토 대기 finding candidate: 50
- WP3 quality gate: PASS 47, PENDING 3, FAIL 0

PENDING 3개는 공식 계약 고정, 시각화 수동 승인, graph insight 수동 승인이다. WP2 자체의
relation 수동 검수도 아직 완료되지 않았다.

## 공식 입력 도착 후 실행 순서

```powershell
.\.venv\Scripts\python.exe scripts\run_stage1_wp3.py `
  --input-dir path\to\official_wp2 `
  --output-dir outputs\stage1_wp3_official `
  --contract config\wp3_open_models_contract.json
```

그 다음 notebook의 입력/output 경로를 공식 폴더로 바꾸고 실행한 뒤 품질 게이트를 실행한다.

```powershell
.\.venv\Scripts\python.exe tests\quality_gates\run_quality_gates.py --fail-on pending
```

모든 자동 검사가 PASS이고 위 수동 항목이 승인되면 WP3를 최종 완료로 간주할 수 있다.

## 추가 준비 도구

- `scripts/summarize_relation_reviews.py`: relation 검수 수, precision, predicate별 결과와 오류 유형 집계
- `docs/WP2_RELATION_REVIEW_GUIDE_KO.md`: TRUE/FALSE/AMBIGUOUS 판단 및 오류 분류 기준
- `src/stage3_graph_rag/`: entity 기반 graph expansion과 source passage 회수 baseline
- `scripts/query_graph_evidence.py`: Stage 3 graph retrieval을 명령행에서 점검하는 도구

Retriever는 relation triple뿐 아니라 `text_unit_id`, passage, document URL, 날짜를 함께 반환한다.
따라서 Stage 3에서는 이를 text-only retrieval 결과와 합쳐 동일한 QA set으로 비교할 수 있다.
