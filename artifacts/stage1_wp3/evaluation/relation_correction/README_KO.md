# Stage 1 WP3 relation correction 검수

이 폴더는 `notebooks/stage1_wp3_relation_correction.ipynb`가 생성한 신규 Tier B 검수 표본을
보관한다. 기존 WP2 표본과 AI-screened 표본에 포함된 triple은 제외했다.

## 파일

- `relation_review_corrected_seed2026.csv`: 90개 신규 검수 표본
- `relation_review_corrected_seed2026.xlsx`: dropdown과 가이드가 포함된 검수용 workbook
- `relation_review_corrected_seed2026_manifest.json`: 입력 hash, 표본 추출 방식과 결과 수

검수자는 `review_correct`, `reviewer`, `review_error_type`, `review_notes`만 입력한다. 현재 상태는
`pending_human_review`이며, 완료 전에는 Tier B를 human-validated relation으로 표현하지 않는다.

대용량 보정 결과는 Git에 커밋하지 않고 다음 위치에서 재생성한다.

```text
outputs/stage1_wp3_relation_correction_20261008/
```

실제 candidate graph 입력은 `edges_graph_candidates.csv`이다. 이 파일은 Tier A와 B를 사용하되
동일 triple 15개를 하나로 합치고 `sources`에 WP1 lexicon과 GLiREL provenance를 함께 보존한다.
