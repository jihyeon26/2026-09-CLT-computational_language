# Stage 1 quality gates

이 폴더는 프로젝트 Stage 1 안의 WP1, WP2, WP3 산출물이 다음 작업으로 넘어갈 만큼 충분한지
검사한다. 일반 단위 테스트와 달리 실제 notebook과 output artifact를 읽는다.

## 상태 의미

- `PASS`: 자동 검사 기준 충족
- `WARN`: 결과를 막지는 않지만 확인이 필요한 항목
- `PENDING`: 사람의 검수 또는 공식 계약 확정이 필요한 항목
- `FAIL`: 필수 파일, schema, ID, provenance 또는 실행 가능성에 문제가 있음

Stage 상태는 가장 심각한 check 상태를 따른다.

## 실행

현재 진행 상황을 보고서로 만들되 실패 코드 없이 종료:

```powershell
.\.venv\Scripts\python.exe tests\quality_gates\run_quality_gates.py --fail-on never
```

CI 또는 제출 직전에는 FAIL과 PENDING을 모두 차단:

```powershell
.\.venv\Scripts\python.exe tests\quality_gates\run_quality_gates.py --fail-on pending
```

결과는 다음 위치에 저장된다.

- `outputs/quality_gates/stage1_quality_report.json`
- `outputs/quality_gates/stage1_quality_report.md`

## 기준 변경

`quality_contract.json`에서 다음을 변경할 수 있다.

- WP1 산출물 파일 패턴
- WP2/WP3 output directory
- 최소 문서, 엔티티, 관계, 시각화 수
- 최소 document coverage
- relation 수동 검수 표본 수와 precision
- 필요한 metric view
- 사람의 최종 승인 flag

WP3 자동 검사는 전체 graph뿐 아니라 모든 predicate별 metric, 월별 relation 표,
co-occurrence community 전체 할당, finding candidate schema, 세 graph view 명세와 Stage 3
provenance chain까지 확인한다. `graph_finding_candidates`의 내용이 사실인지에 대한 판단은
아래 수동 승인 범위에 남겨 둔다.

최소 개수와 coverage는 공식 평가기준의 절대 점수 기준이 아니라 팀의 내부 품질 기준이다.
변경할 경우 이유를 notebook 또는 회의 기록에 남긴다.

## 자동화할 수 없는 부분

다음 항목은 코드로 완전히 판정할 수 없으므로 manual flag로 관리한다.

- WP1 topic과 temporal insight의 타당성
- WP2 relation sample의 의미적 정확성
- WP3 graph visualization의 가독성
- WP3 graph insight가 evidence로 충분히 뒷받침되는지
- 공식 WP2 contract의 최종 확정
