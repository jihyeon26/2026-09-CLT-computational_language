# WP3 candidate graph 민감도 분석

이 폴더의 수치는 사람의 relation 검수 전에 생성한 탐색 결과다. 세 typed graph는 구조 검사를
통과한 candidate relation을 score tier별로 나눈 것이며 검증된 사실 그래프가 아니다.

- `all_candidates`: 모든 구조적으로 유효한 candidate와 curated relation
- `medium_plus`: GLiREL score 0.60 이상
- `high_score`: GLiREL score 0.75 이상

공동출현 graph는 semantic relation이 아니며 같은 문서에서 함께 등장한 정도만 나타낸다.
PageRank, component와 community 결과는 threshold 안정성을 비교하는 용도로만 사용한다.

최종 보고서의 사실 주장과 Stage 3 relation-grounded retrieval에는 사람 검수 결과 또는 연결된
원문 passage 확인이 필요하다.
