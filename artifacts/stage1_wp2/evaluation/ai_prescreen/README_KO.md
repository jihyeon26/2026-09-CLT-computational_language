# Claude relation AI 1차 스크리닝

이 폴더는 `relation_review_sample_seed42.csv` 90행을 Claude로 1차 판정한 결과를 보존한다.
이 결과는 사람의 수동 검수를 대체하지 않으며 WP3 edge를 자동 삭제하는 정책에도 사용하지 않는다.

## 결과

- TRUE: 39
- FALSE: 44
- AMBIGUOUS: 7
- 모호 항목을 제외한 precision: 0.470
- 평가 기준: 최소 90개 확정 판정, precision 0.80 이상
- 현재 상태: FAIL

점수대별 확정 판정 precision은 다음과 같다.

| Mean-score band | TRUE | FALSE | AMBIGUOUS | Precision |
|---|---:|---:|---:|---:|
| 0.50–0.59 | 11 | 24 | 5 | 0.314 |
| 0.60–0.74 | 24 | 19 | 2 | 0.558 |
| 0.75–1.00 | 4 | 1 | 0 | 0.800 |

0.75 이상 구간은 표본이 5개뿐이므로 전체 population의 성능으로 일반화할 수 없다.

## 해석상 주의점

표본 단위는 unique triple이지만 각 triple에는 한 개의 deterministic evidence sentence만 제시됐다.
따라서 FALSE는 다음 두 경우를 함께 포함할 수 있다.

1. triple 자체가 잘못됨
2. triple은 사실일 수 있지만 선택된 문장이 해당 관계를 직접 뒷받침하지 않음

두 번째 경우도 evidence-grounded Graph-RAG에는 실패다. 다만 전체 triple을 graph에서 제거할지 결정하려면
다른 evidence 문장도 확인해야 한다. 사람 검수에서는 `evidence supports relation`과
`triple plausible from other evidence`를 구분해 기록하는 것이 좋다.

## 사용 정책

- AI 판정은 사람 검수 우선순위를 정하는 데만 사용한다.
- FALSE와 AMBIGUOUS 51개를 사람이 먼저 재검토한다.
- TRUE 39개도 일부가 잘못 승인됐을 가능성이 있으므로 전부 또는 별도 표본을 재검토한다.
- 사람이 승인하기 전까지 모든 relation은 `candidate`로 표시한다.
- raw relation graph를 사실 지식그래프로 설명하지 않는다.

`relation_review_claude_draft.csv`의 원본 추출 열은 기준 표본과 동일하다. 11개 evidence 문자열의 차이는
CSV 처리 과정에서 `CRLF`가 `LF`로 정규화된 것이며 문장 내용 변화는 아니다.
