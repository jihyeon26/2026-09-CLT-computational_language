# WP2 relation 수동 검수 가이드

## 목적

이 검수는 relation extraction의 의미적 precision을 측정하고, WP3 knowledge graph에 포함하면
안 되는 edge 유형을 찾기 위한 절차다. 자동 도구나 LLM의 판단만으로 `review_correct`를 확정하지
않고 사람이 evidence 문장과 필요하면 원문 URL을 확인한다.

## 입력 방법

`relation_review_sample.csv`에서 다음 열을 채운다.

- `review_correct`: `TRUE`, `FALSE`, `AMBIGUOUS` 중 하나
- `reviewer`: 검수자 이름 또는 팀에서 정한 ID
- `review_error_type`: `FALSE`일 때 아래 오류 유형 중 하나
- `review_notes`: 판단 근거, 수정 제안 또는 ambiguous 사유

`AMBIGUOUS`는 evidence 문장만으로 참/거짓을 결정할 수 없을 때 사용한다. Precision은
`TRUE / (TRUE + FALSE)`로 계산하며 ambiguous는 분모에서 제외한다.

## 판단 순서

1. Subject와 object가 evidence 문장에 실제로 존재하는지 확인한다.
2. 두 표현이 catalogue의 canonical entity에 올바르게 연결됐는지 확인한다.
3. Predicate 의미가 문장에 명시되어 있는지 확인한다.
4. Relation 방향이 올바른지 확인한다.
5. 부정, 추측, 인용, 비교 문맥을 asserted fact로 잘못 처리하지 않았는지 확인한다.
6. 문장만으로 부족하면 원문 URL의 주변 문맥을 확인한다.

## 오류 유형

| 값 | 의미 |
|---|---|
| `entity_boundary` | mention 범위가 잘못됨 |
| `entity_linking` | 다른 canonical entity에 연결됨 |
| `wrong_predicate` | entity는 맞지만 relation 의미가 틀림 |
| `reversed_direction` | subject와 object 방향이 반대임 |
| `negation_or_uncertainty` | 부정·불확실 표현을 사실로 처리함 |
| `insufficient_context` | 현재 evidence로 판단할 수 없음 |
| `duplicate_or_noise` | 중복·템플릿·노이즈성 관계임 |
| `schema_or_catalogue` | schema 또는 catalogue 자체 수정이 필요함 |
| `other` | 위 유형에 해당하지 않으며 notes에 설명함 |

판단할 수 없는 경우에는 `FALSE + insufficient_context`보다 `AMBIGUOUS`를 우선 사용한다.

## 집계 실행

```powershell
.\.venv\Scripts\python.exe scripts\summarize_relation_reviews.py
```

제출 전 기준을 강제하려면 다음과 같이 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts\summarize_relation_reviews.py --fail-on pending
```

기본 내부 기준은 precision 계산 가능 표본 50개 이상, precision 0.80 이상이다. 보고서는
`outputs/relation_review/`에 JSON과 Markdown으로 생성된다.

## 팀 검수 권장 방식

- Predicate별 표본이 최소 몇 개씩 포함되는지 확인한다.
- 경계 사례 10개 정도는 두 명이 독립적으로 검수하고 불일치를 논의한다.
- 오답이 반복되는 오류 유형은 개별 edge 삭제에 그치지 말고 extraction rule 또는 catalogue를 수정한다.
- Rule 수정 후에는 동일 표본에 맞추는 대신 새로운 표본으로 다시 확인한다.
