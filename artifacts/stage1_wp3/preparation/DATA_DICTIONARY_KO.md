# WP3 준비 스냅샷 데이터 사전

이 스냅샷은 공식 WP2 원본을 변경하지 않고 WP3 그래프 분석 전에 만든 파생 입력이다.

- `nodes_prepared.parquet`: WP2 원본 type과 compact 분석 type을 함께 보존한 node 테이블
- `relation_policy_decisions.parquet`: 모든 relation assertion과 포함 여부·제외 사유
- `typed_edges_assertions.parquet`: 구조 정책을 통과한 candidate relation assertions
- `typed_edges_collapsed.parquet`: 동일 raw triple을 합친 candidate graph edges
- `entity_mentions_corrected.parquet`: article offset을 보존하고 sentence-local offset을 추가한 mention
- `unresolved_mentions.csv`: 자동으로 보정할 수 없는 mention 검토 목록
- `policy_audit.json`: relation 제외 수와 candidate quality tier 분포
- `validation_report.json`: 교정 전후 구조 검사와 offset 보정 결과
- `canonical_schema.json`: WP2 schema를 WP3 signature 형식으로 변환한 결과
- `compact_schema_resolved.json`: 분석용 compact type·predicate group 매핑
- `preparation_manifest.json`: 입력·출력 해시, 버전과 완료 상태

`candidate`는 사람이 의미 정확성을 승인했다는 뜻이 아니다. 현재 AI 1차 스크리닝 결과는
edge 삭제 정책에 사용하지 않았으며, 검증된 최종 그래프에는 사람의 relation 검수가 필요하다.
