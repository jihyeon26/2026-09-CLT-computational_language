# Stage 3 graph evidence retriever 사용법

## 목적

`GraphEvidenceRetriever`는 질문에서 entity를 찾고 Stage 1 assertion graph를 1~3 hop 확장한 뒤,
각 relation에 연결된 WP2 text unit과 document metadata를 반환한다. Graph triple 자체를 답변
근거로 사용하지 않고 원문 passage와 URL을 함께 제공하는 것이 핵심이다.

## 빠른 실행

```powershell
.\.venv\Scripts\python.exe scripts\query_graph_evidence.py "Who developed Gemma 4?" --predicate DEVELOPED_BY
```

Predicate와 날짜를 제한할 수도 있다.

```powershell
.\.venv\Scripts\python.exe scripts\query_graph_evidence.py `
  "What license is used by Gemma?" `
  --predicate LICENSED_UNDER `
  --start-date 2025-01-01 `
  --max-hops 1 `
  --top-k 5
```

## Python 사용

```python
from stage3_graph_rag import GraphEvidenceRetriever

retriever = GraphEvidenceRetriever.from_directories(WP3_OUTPUT_DIR, WP2_OUTPUT_DIR)
result = retriever.retrieve(
    "Who developed Gemma 4?",
    max_hops=1,
    top_k=5,
)
```

반환값에는 다음이 포함된다.

- `matched_entities`: query에서 해석한 entity와 match 방식
- `evidence`: subject-predicate-object, passage, URL, 날짜, retrieval score
- `diagnostics`: seed entity, 후보 assertion 수, provenance 누락 수, 적용한 filter

## Stage 3 실험에서의 사용 위치

1. Text-only baseline은 질문으로 dense/sparse passage retrieval을 수행한다.
2. Graph-enhanced 방식은 질문 entity를 anchor로 relation evidence를 찾는다.
3. 두 방식의 passage를 합치고 동일한 answer model과 prompt를 사용한다.
4. Faithfulness, relevance, context sufficiency, temporal grounding을 같은 QA set으로 비교한다.

Graph-enhanced 방식만 더 많은 context나 더 좋은 생성 모델을 사용하면 graph 효과를 공정하게
비교할 수 없다.

## 현재 골격의 제한

- Entity resolution은 canonical name과 alias의 phrase/token matching이다.
- Retrieval score는 hop 거리와 WP2 rule score를 결합한 baseline이며 학습된 ranker가 아니다.
- Query relation intent를 자동 분류하지 않으므로 필요한 경우 `predicates`를 명시해야 한다.
- 반환된 passage가 질문에 충분한지는 Stage 3 evaluation에서 별도로 측정해야 한다.

이 제한은 text-only baseline과 비교 가능한 단순하고 설명 가능한 출발점을 제공하기 위한 것이다.
