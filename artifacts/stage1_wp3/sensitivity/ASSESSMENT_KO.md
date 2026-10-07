# WP3 candidate graph 민감도 평가

이 평가는 사람의 relation 검수 전에 graph 결과가 score·support threshold에 얼마나 민감한지
확인한다. 수치는 탐색용이며 실제 산업 영향력이나 사실 관계를 확정하지 않는다.

## Relation graph 크기와 연결성

| Variant | Assertions | Collapsed edges | Nodes | Components | Largest component share |
|---|---:|---:|---:|---:|---:|
| all_candidates | 19,160 | 13,322 | 8,331 | 600 | 0.804 |
| medium_plus | 9,575 | 6,747 | 4,800 | 468 | 0.746 |
| high_score | 1,766 | 1,458 | 1,323 | 225 | 0.577 |

## PageRank 안정성

| Comparison | Top-20 Jaccard | PageRank Spearman |
|---|---:|---:|
| all_candidates vs medium_plus | 0.739 | 0.395 |
| all_candidates vs high_score | 0.379 | 0.189 |
| medium_plus vs high_score | 0.481 | 0.243 |

`all_candidates`와 `high_score`의 top-20 중복과 전체 PageRank 순위 상관이 모두 낮다. 현재
중심성 순위는 confidence threshold 선택에 민감하므로 최종 insight로 사용할 수 없다.

## Co-occurrence threshold 민감성

| Minimum documents | Edges | Nodes | Components | Communities | Largest component share |
|---:|---:|---:|---:|---:|---:|
| 1 | 186,710 | 48,304 | 1,930 | 2,543 | 0.892 |
| 2 | 51,522 | 8,922 | 81 | 114 | 0.974 |
| 5 | 19,521 | 2,377 | 5 | 13 | 0.996 |

문서 1회 공동출현은 coverage가 크지만 희소 node와 작은 community가 많다. 문서 5회 기준은
구조가 안정적으로 연결되지만 coverage가 크게 줄어든다. 현재는 `minimum documents = 2`를
탐색용 기본값으로 사용하고 1과 5를 sensitivity view로 함께 보존한다.

## 검수 전 결정

- `medium_plus`를 탐색용 typed graph의 임시 기본값으로 사용한다.
- `all_candidates`와 `high_score` 결과를 함께 보고 metric 안정성을 확인한다.
- 어느 variant도 validated knowledge graph로 부르지 않는다.
- predicate group별 최종 포함 여부는 사람 검수 후 정한다.
- 최종 centrality와 graph insight는 human-validated graph에서 다시 계산한다.
