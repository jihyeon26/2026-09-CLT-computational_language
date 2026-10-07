# Stage 1 WP1·WP2 노트북 코드·결과·해석 검토 가이드

기준일: 2026-10-07  
트랙: **Open Models**  
대상: 팀 저장소의 공식 Stage 1 WP1·WP2 노트북과 공식 인계 산출물

## 1. 문서 목적

이 문서는 Stage 1의 WP1과 WP2 노트북을 나중에 다시 열었을 때 다음 질문에 답할 수 있도록 만든 한국어 검토 자료다.

1. 각 핵심 코드 셀은 무엇을 하는가?
2. 실제 저장된 실행 결과는 무엇인가?
3. 그 결과를 어떻게 해석해야 하는가?
4. 결과를 과대해석하지 않으려면 무엇을 주의해야 하는가?
5. WP3에서 어떤 파일과 규칙을 이어받아야 하는가?

기존 문서와 역할이 다르다.

- `STAGE1_WP1_WP2_WP3_INTEGRATION_OVERVIEW_KO.md`: 제출 체크리스트와 세 WP의 연결
- `STAGE1_WP2_TO_WP3_RUNBOOK_KO.md`: WP2를 받은 뒤 WP3를 실행하는 절차
- **이 문서**: WP1·WP2 노트북의 핵심 셀을 코드, 결과, 해석 순서로 이해하는 자료

셀 번호는 노트북 JSON에서 0부터 센 인덱스다. Jupyter 화면에서는 실행 번호인 `In [n]`을 함께 참고한다.

---

# Part A. WP1: Data and Topic Analysis

## 2. WP1 전체 구조

WP1 노트북은 총 75셀이고, 그중 코드 셀은 33개다. 모든 셀을 같은 깊이로 읽기보다는 다음 여섯 블록으로 보는 것이 좋다.

1. 실행 환경과 원자료
2. 데이터 정제
3. Open Models 문서 선별
4. 탐색적 분석
5. 감성·프레이밍·토픽 모델링
6. WP2 인계 산출물

팀 노트북: `stage1/work_packages/stage1_wp1_data_topic_analysis.ipynb`

## 3. 실행 환경과 원자료

### 셀 3: 라이브러리 준비

pandas, NumPy, spaCy, NLTK와 scikit-learn을 불러오고, 필요한 패키지가 없으면 `pip`로 설치한다. 이 셀은 분석 결과를 만들지 않지만 Colab 실행 환경을 구성한다. 최종 재현성을 높이려면 자동 설치에만 의존하지 말고 패키지 버전을 별도로 고정한다.

### 셀 4: 결과를 결정하는 설정

```python
SEED = 42
STUDY_START = "2024-09-01"
STUDY_END = "2026-08-31"
MIN_CHARS = 1_500
NEAR_DUP_THRESHOLD = 0.80
CORE_SCORE_MIN = 4.0
SPACY_MAX_CHARS = 12_000
K_GRID = [6, 8, 10, 12, 14, 16, 18, 20]
K_SELECT_RANGE = (8, 16)
```

- `MIN_CHARS`: boilerplate 제거 후 분석 가능한 최소 본문 길이
- `NEAR_DUP_THRESHOLD`: MinHash 기반 근접 중복 기준
- `CORE_SCORE_MIN`: Open Models core 문서가 되기 위한 관련성 점수
- `K_SELECT_RANGE`: 최종 topic 수를 선택할 허용 구간

결과가 바뀌었을 때 가장 먼저 확인할 셀이다.

### 셀 6: 원본 파일과 캐시 검증

원본 CSV를 환경변수, 저장소, Google Drive, Kaggle 순서로 찾고 SHA-256을 계산한다.

```text
Raw file: ai_media_dataset_20260911.csv
크기: 약 530 MB
sha256: 82dcea5ee2b15faf…
Cleaning stage: loaded from data/processed (cache, same raw checksum)
```

저장된 WP1 결과는 정제 단계를 다시 계산한 것이 아니라 캐시에서 읽은 결과다. 그러나 현재 원본과 정제 당시 원본의 checksum이 같은지 확인한 뒤 캐시를 사용했다. 최종 재현성 시험이 필요하면 별도 환경에서 `WP1_FORCE_REBUILD=1`로 처음부터 다시 실행한다.

### 셀 8–11: 원자료 검사와 파싱

```text
47,751 rows × 7 columns
결측값: 0
중복 URL: 0
동일한 content 문자열: 66
출처: 392
날짜: 2024-09-11 → 2026-09-07
```

`content`와 `tags`는 문자열로 직렬화된 Python list이므로 `ast.literal_eval`로 파싱한다. URL의 SHA-1 앞 10자리를 사용해 안정적인 `doc_id`도 만든다.

```text
doc_id = "doc_" + sha1(url)[:10]
```

이 `doc_id`가 WP1 → WP2 → WP3를 연결하는 기본 키다.

## 4. 데이터 정제

### 셀 14: Boilerplate 제거

같은 출처의 문서에서 `max(5개 문서, 해당 출처 문서의 1%)` 이상 반복되는 문단을 boilerplate로 본다.

```text
boilerplate 문단 유형: 5,667
제거된 문단: 280,950
제거된 문자 비율: 9.8%
영향받은 문서: 33,717
```

뉴스레터 가입 문구, footer, 광고뿐 아니라 `open source`, `OpenAI` 같은 단어를 포함한 반복 문구도 제거한다. 이 단계가 없으면 Open Models 관련 문서 수가 부풀려질 수 있다.

### 셀 16: 문자 복원

- mojibake 복구
- zero-width character 제거
- apostrophe 공백 복원
- Penn Treebank quote 복원
- 괄호 뒤 잘못된 공백 복원

```text
apostrophe spacing: 681,159건
quote: 160,588건
parenthesis spacing: 269,632건
mojibake 문서: 167개
```

이는 본문을 요약하거나 단어를 제거하는 과정이 아니라 잘못된 표기를 복원하는 과정이다.

### 셀 18: 짧은 문서 제거

Boilerplate 제거 후 1,500자 미만인 992개 문서를 제거한다. AP 영상 캡션이나 거의 비어 있는 API 페이지 등이 주요 대상이다.

### 셀 20: 언어 필터

영어 stopword 비율이 0.25 미만인 문서만 `langdetect`로 검사한다.

```text
langdetect 검사: 2,887개
비영어 제거: 357개
```

### 셀 22: 정확·근접 중복 제거

정확 중복은 정규화된 본문 hash로, 근접 중복은 5-word shingle MinHash로 찾는다.

```text
정확 중복: 119
확인된 근접 중복쌍: 613
제거 문서: 529
중복 cluster: 472
여러 출처에 걸친 cluster: 100
```

같은 기사가 재게시되며 트렌드가 과장되는 것을 방지한다. Cluster에서는 가장 먼저 등장한 문서를 남긴다.

### 셀 24: 정제 corpus 확정

```text
원자료                              47,751
본문 1,500자 이상                   46,759
영어 문서                           46,402
정확·근접 중복 제거                 45,873
연구 기간 적용                      45,644
```

최종 정제 corpus는 45,644문서, 6,870만 단어, 391개 출처이며 분석 기간은 2024-09-11부터 2026-08-31까지다.

## 5. Open Models 문서 선별

### 셀 31: Matching view

Unicode 하이픈과 따옴표를 일반 ASCII 문자로 바꾼 검색용 view를 만든다. 원문 보존용 `text`와 정규식 적용용 view를 분리한 설계다.

### 셀 33: Open Models lexicon

```text
model family: 35개
openness concept: 6개
ecosystem tool: 7개
지역별 family: US 14 / China 10 / Europe 6 / Other 5
```

단순 부분 문자열이 아니라 keyword pre-filter와 정규식을 조합해 `Phi`, `Granite`, `Arctic` 같은 중의적 이름의 오탐을 줄인다.

### 셀 35: 관련성 점수와 tier

```python
om_score = (
    3 * title_hit
    + 2 * lead_hit
    + min(strong_openness_mentions, 4)
    + 0.5 * min(family_mentions, 6)
)
```

```text
reference: API reference 성격의 페이지
core:      om_score >= 4
mention:   신호는 있으나 core 미만
none:      관련 신호 없음
```

| Tier | 문서 수 | 전체 비율 |
|---|---:|---:|
| core | 1,210 | 2.7% |
| reference | 68 | 0.1% |
| mention | 2,827 | 6.2% |
| none | 41,539 | 91.0% |

| Family | 전체 언급 문서 | Core 문서 |
|---|---:|---:|
| DeepSeek | 1,309 | 566 |
| Llama | 1,139 | 438 |
| Mistral | 625 | 237 |
| Qwen | 513 | 298 |
| Gemma | 352 | 194 |

### 셀 38: 관련성 수동 표본 검증

고정 hash로 core 40개와 mention 20개를 뽑아 검토했다.

```text
core 관련성: 37/40 = 92%
core가 기사의 주제 자체인 비율: 18/40 = 45%
mention 관련성: 1/20 = 5%
```

Core는 높은 관련성을 보이지만 모두 Open Models를 주제로 삼은 기사는 아니다. Mention은 recall을 위한 넓은 후보군이며 topic model의 직접 입력으로 쓰지 않는다. 표본이 작으므로 최종 보고서에는 반드시 `n=40`, `n=20`을 함께 적는다.

## 6. WP1 EDA

### 셀 27–28: Corpus 편향 확인

```text
월별 문서 수: 953 → 3,176
최대/최소 비율: 3.3
상위 10개 출처 비중: 23.4%
100개 이상 문서를 가진 출처: 120개
```

따라서 시점별 raw count를 그대로 비교하지 않고 월별 비율이나 문서 1,000개당 언급 수를 사용한다.

### 셀 41: Open Models coverage

```text
2025-01 core share: 6.0%
2025-02 core share: 6.4%
2026-07 core share: 3.8%
첫 12개월 평균: 3.06%
마지막 12개월 평균: 2.23%
```

2025년 초 DeepSeek 보도 spike가 첫 기간 평균을 높였으므로 단순한 지속적 하락으로 해석하면 안 된다.

### 셀 44–45: Family별 관심

한 기사 안에서 같은 family가 여러 번 언급되어도 문서 단위로 한 번만 센다. 월별 수집량 차이를 보정하기 위해 문서 1,000개당 비율을 사용한다.

```text
DeepSeek peak: 2025-02, 116.3‰
Llama peak: 2024-09, 44.1‰
Qwen peak: 2026-08, 22.0‰
Kimi peak: 2026-07, 19.8‰
```

이는 모델 성능이 아니라 미디어 관심도다.

### 셀 48: 지역별 분석

모델 family를 개발사 본거지에 따라 US, China, Europe, Other로 분류한다.

```text
China/US 언급 문서 비율
2024Q4 0.20
2025Q1 1.85
2025Q3 1.15
2026Q1 1.01
2026Q3 1.27
```

이는 기사 출처의 국가나 시장 점유율이 아니라 언급된 모델 개발사의 지역이다. 한 문서가 여러 지역에 동시에 포함될 수 있다.

### 셀 51: Release radar

버전 문자열을 regex로 찾아 첫 등장일과 peak month를 계산한다.

```text
5개 이상 문서에 등장한 버전: 68개
연구 기간 안에서 처음 관찰된 버전: 59개
```

`first_seen`은 공식 출시일이 아니라 이 corpus에서 regex가 처음 포착한 날짜다. 예측 기사, 비교표, 숫자 문맥의 오탐이 있을 수 있으므로 주요 버전은 원문을 수동 확인한다.

### 셀 54: Open-weight 용어 변화

```text
2024Q4 13.4%
2025Q2 29.6%
2025Q3 43.7%
2026Q3 61.4%
```

미디어의 표현이 `open-source model`에서 더 구체적인 `open-weight`로 이동한다는 근거로 사용할 수 있다. 부분 분기는 별도로 표시해야 한다.

## 7. 감성·프레이밍·토픽 모델링

### 셀 57: Topic analysis view

Model family 이름과 `open_weight`, `llm` 등의 중요 phrase를 보호한 뒤 spaCy lemmatization을 수행한다. 명사·고유명사·형용사·동사를 중심으로 남기며 문서당 최대 12,000자를 처리한다.

```text
core 문서: 1,210
전처리 token 중앙값: 624
vocabulary: 26,729
이번 실행에서 새로 lemmatize한 문서: 0, 전부 cache 사용
```

`fidelity text`와 topic model용 `analysis view`를 분리한 점이 중요하다.

### 셀 59–62: 감성과 frame

Open Models family 또는 openness concept가 포함된 문장만 분석한다.

```text
target sentence: 13,587
포함 문서: 1,207/1,210
VADER 평균: +0.237
positive: 56%
negative: 10%
```

Frame은 Performance, Cost, Access, Safety, Geopolitics, Licensing, Enterprise Adoption의 7개 keyword dictionary다. VADER의 positive는 Open Models에 대한 지지와 동일하지 않고, frame도 학습된 분류기가 아니라 keyword presence이므로 탐색적 결과로 제시한다.

### 셀 65–67: Topic model 선택

```text
TF-IDF: 1,210 documents × 12,000 uni/bi-grams
density: 3.21%
```

`k=6, 8, 10, 12, 14, 16, 18, 20`을 비교하지만 최종 허용 범위는 8–16이다. 이 범위 안에서 diversity가 0.75 이상이고 평균 NPMI가 가장 높은 `k=16`을 선택한다.

| 모델 | 평균 NPMI | Diversity |
|---|---:|---:|
| NMF | 0.2387 | 0.865 |
| LDA | 0.1556 | 0.685 |

NMF를 메인 모델로 선택한 근거가 있다. Topic 이름은 상위 세 단어를 붙인 자동 라벨이므로 대표 문서를 본 뒤 사람이 최종 명명한다. `people · ask · find`와 `enterprise · innovation · solution` topic은 coherence가 낮아 특히 검토가 필요하다.

### 셀 70: Topic 변화

상승:

```text
Claude/agent/coding       +8.69%p
Nemotron/NVIDIA           +4.52%p
China/geopolitics         +2.40%p
API/Python                +1.75%p
```

감소:

```text
DeepSeek/R1               -4.02%p
Reasoning/RL              -4.00%p
Multilingual/instruction  -3.43%p
Meta/Llama                -3.26%p
```

DeepSeek 관련 감소는 2025년 초 spike 이후의 정상화 영향을 포함한다.

## 8. WP1 인계 산출물

셀 74에서 생성하는 핵심 파일:

| 파일 | 역할 |
|---|---|
| `corpus_clean.parquet` | 정제된 전체 45,644문서 |
| `open_models_core.parquet` | WP2의 직접 입력인 1,210문서 |
| `open_models_target_sentences.parquet` | WP1 감성·frame용 13,587문장 |
| `lemma_cache.parquet` | Topic 전처리 결과 |
| `track_scan_cache.parquet` | Family·concept 매칭 결과 |
| `open_models_lexicon.json` | WP2 canonicalisation과 dictionary extraction 입력 |
| `data_dictionary.md` | 컬럼 정의 |

WP1의 target sentence 파일은 WP2의 전체 `sentences.csv`와 다르다. WP3 relation provenance에는 WP2의 문장 파일을 사용한다.

---

# Part B. WP2: Knowledge Extraction

## 9. WP2 전체 구조와 실행 상태

WP2 노트북은 총 53셀이고, 비어 있지 않은 코드 셀은 30개다.

```text
실행번호가 있는 코드 셀: 20
실행번호가 없는 코드 셀: 10
```

실행번호가 순차적이지 않고 코드의 `OUT_DIR`과 출력에 표시된 저장 경로도 일치하지 않는다. 따라서 노트북 출력은 여러 Colab 실행의 결과가 합쳐진 상태로 봐야 한다. 공식 CSV는 별도로 직접 검사한다.

팀 노트북: `stage1/work_packages/stage1_wp2_entity_relation_extraction.ipynb`

## 10. WP1 입력 연결

### 셀 3–5

- 팀 GitHub 저장소 clone
- Google Drive mount
- WP1 `open_models_core.parquet` 로딩
- WP1 `open_models_lexicon.json` 로딩

```text
35 foundation models
6 openness concepts
7 ecosystem tools
1,210 documents
```

임시 clone 셀에는 `REMOVE IN FINAL NOTEBOOK` 표시가 있다. 최종 제출본에서는 제거하거나 재현 가능한 일반 경로 설정으로 바꾼다.

## 11. 문장 분리와 spaCy NER

### 셀 7–9

`en_core_web_lg`를 사용해 1,210개 문서를 문장으로 나누고 generic NER을 수행한다.

```text
문장: 99,438
spaCy entity mention: 138,973
ORG: 60,489
PERSON: 15,064
GPE: 9,754
PRODUCT: 3,963
```

spaCy는 일반 뉴스 entity에는 유용하지만 AI model, architecture, benchmark 같은 domain type이 없다.

### Sentence offset 주의점

코드는 `sent.text.strip()`을 저장하면서 원문 기준 `sentence_start`는 보정하지 않는다. 따라서 저장된 sentence 안에서 mention의 상대 위치를 계산하면 다음 불일치가 생긴다.

```text
sentence-local mention offset mismatch: 12,977 / 209,148
```

전체 article text 기준 mention offset은 대부분 정상이다. WP3나 Stage 3에서 sentence 안의 정확한 span을 표시할 때만 보정이 필요하다.

## 12. WP1 dictionary extraction

### 셀 11–15

WP1 lexicon을 전체 본문에 적용한다.

```text
dictionary mentions: 19,793
FOUNDATION_MODEL: 14,580
OPENNESS_CONCEPT: 3,290
TECHNOLOGY: 1,923
```

각 mention은 `doc_id`, `sentence_id`, 원문 offset, canonical ID와 developer를 유지한다. Llama, Qwen, Gemma 같은 핵심 family에 대해 높은 precision을 제공한다.

## 13. GLiNER domain NER

### 셀 18–23

```text
model: urchade/gliner_medium-v2.1
threshold: 0.5
labels: 15
```

```text
GLiNER mentions: 184,339
COMPANY: 29,497
AI_MODEL: 27,220
PERSON: 26,981
TECHNOLOGY: 13,679
DATASET: 12,191
```

spaCy에 없던 domain type을 추가하지만 `I`, `model`, `AI`, `you`, `developers`, `data`, `users` 같은 generic entity도 다수 추출한다. 후속 단계에서 pronoun은 제거하지만 generic noun은 상당수 남는다. Mention 수 증가를 곧바로 품질 향상으로 해석하면 안 된다.

## 14. Entity 병합과 linking

### 셀 24–28

spaCy에서는 `ORG`, `PERSON`, `PRODUCT`, `LAW`, `GPE`, `NORP`만 유지하며 45,185개의 날짜·숫자·금액 등 비관련 mention을 제거한다.

병합 우선순위:

```text
1. WP1 domain dictionary
2. spaCy place types
3. GLiNER
4. 나머지 spaCy types
```

```text
baseline(dictionary + spaCy): 105,886 mentions
improved(dictionary + GLiNER + spaCy): 209,443 mentions
최종 linked mentions: 209,148
surface forms: 59,013
linked entity names: 54,588
```

관사·소유격·구두점 정리, WP1 phrase normalization, casing 통합, 짧은 사람 이름 연결과 entity별 대표 type 결정을 수행한다.

`DeepSeek`, `Mistral`처럼 회사와 model family가 같은 이름은 mention 문맥별 type을 일부 유지한다. 이 때문에 relation 추출 당시 type과 최종 node type이 달라질 수 있다.

## 15. Entity 평가

### 셀 30: WP1 lexicon silver standard

| Lexicon type | spaCy 탐지율 | GLiNER 탐지율 | spaCy 올바른 type | GLiNER 올바른 type |
|---|---:|---:|---:|---:|
| Foundation model | 54.5% | 76.4% | 6.3% | 28.6% |
| Technology | 33.3% | 61.9% | 0.5% | 39.8% |

GLiNER가 spaCy보다 domain entity를 더 잘 찾지만 WP1 dictionary가 여전히 필요하다. 이 평가는 WP1 사전 안의 entity만 대상으로 한 silver evaluation이며 사전 밖 entity의 precision을 알려주지는 않는다.

### 셀 32: spaCy와 GLiNER label 비교

```text
두 모델이 모두 찾은 span: 63,028
compatible label agreement: 26,970, 약 43%
```

두 모델의 label inventory가 다르므로 모든 disagreement가 오류는 아니다. Generic NER와 domain NER가 같은 표현을 다르게 해석하는 정도를 보여주는 분석이다.

## 16. Relation schema와 GLiREL

### 셀 37: Relation schema

36개 relation type과 허용 head·tail type을 정의한다.

- 개발·출시: `developed by`, `released by`, `announced by`
- 계보: `member of family`, `successor of`, `based on`
- 특성: `has parameter count`, `has context window`, `licensed under`
- 성능: `outperforms`, `evaluated on`, `has capability`
- 인프라: `runs on`, `trained using hardware`
- 배포: `available on`, `integrated into`, `used by`
- 조직: `acquired by`, `invested in`, `partner of`, `founded by`
- 규제: `regulated by`, `complies with`

`type_valid`는 head와 tail의 type이 허용 목록에 있는지만 검사한다. 문장의 의미가 실제로 정확하다는 보장은 아니다.

### 셀 38–41: GLiREL 실행

```text
model: jackboyla/glirel-large-v0
threshold: 0.5
top_k: 1
최대 문장 길이: 200 tokens
```

```text
entity가 있는 문장 후보:       76,925
서로 다른 entity가 2개 이상:  52,393
GLiREL에 전달:                 47,517
가능 relation 없음:             4,821
200 token 초과:                    28
alignment 실패:                     1
```

```text
원시 relations: 43,075
type-valid: 23,762 (55.2%)
developed by: 13,075
competes with: 8,315
outperforms: 3,808
founded by: 2,628
```

`AI developed by NVIDIA`, `Meta developed by Llama` 같은 오류가 있으므로 raw relation을 바로 사실로 사용하면 안 된다.

### 셀 43: 역방향 중복 제거

Head와 tail을 정렬한 unordered pair로 같은 문장 안의 역방향 예측을 통합한다. `type_valid`와 score가 높은 방향을 남긴다.

```text
제거된 역방향 중복: 12,515
남은 GLiREL relations: 30,560
type-valid: 19,613 (64.2%)
```

대칭 관계에는 적절하지만 방향 관계에서는 잘못된 방향이 남을 가능성이 있으므로 수동 검수가 필요하다.

## 17. Co-occurrence

### 셀 46–49

같은 문장에 나온 서로 다른 entity pair를 만든다.

```text
entity 2개 이상 문장: 51,799
고유 entity pairs: 186,710
typed relation도 있는 pair: 12,593
```

Sentence co-occurrence, document co-occurrence, NPMI와 대표 typed relation을 저장한다. Co-occurrence는 semantic relation이나 인과관계가 아니다.

## 18. WP2 공식 인계 파일

### 셀 51–52

| 파일 | 행 수 | 역할 |
|---|---:|---|
| `nodes.csv` | 54,597 | 최종 graph node |
| `nodes_quarterly.csv` | 75,429 | 분기별 entity 빈도 |
| `entity_mentions.csv` | 209,148 | 원문 mention과 offset |
| `sentences.csv` | 99,438 | 문장과 본문 위치 |
| `relations_evidence.csv` | 30,595 | 문장별 relation assertion |
| `edges_relations.csv` | 22,992 | 집계된 relation triple |
| `edges_cooccurrence.csv` | 186,710 | 문장 공동출현 pair |
| `cooccurrence_quarterly.csv` | 207,021 | 분기별 공동출현 |
| `schema.json` | — | 20 node types, 36 relation types |
| `data_dictionary.md` | — | 컬럼 설명 |

`relations_evidence.csv`는 GLiREL 30,560행과 WP1 lexicon에서 추가한 curated `developed by` 35행으로 구성된다. GLiREL 관계는 모두 문장에 연결되지만 curated 35개는 특정 기사 근거가 없으므로 별도 source로 유지한다.

## 19. 공식 WP2 산출물에서 확인된 한계

### 19.1 최종 node type 재검증

집계 관계 22,992개 중 저장 당시 `type_valid=True`였지만 최종 `nodes.csv` type을 적용하면 invalid가 되는 행이 417개다. 노트북 권장 필터인 `n_sentences >= 2` 또는 `wp1_lexicon` 적용 뒤에도 84개가 남는다. WP3에서 최종 node type으로 다시 검사해야 한다.

### 19.2 집계 triple 중복

`edges_relations.csv`에서 같은 `head_id + relation + tail_id`를 가진 추가 행이 266개다. Aggregation key에 `head_type`, `tail_type`, `type_valid`, `source`까지 포함했기 때문이다. 재실행 부족이 아니라 aggregation 설계에서 생긴 중복이다.

### 19.3 Self-loop

`Baichuan developed by Baichuan` self-loop 1개가 있으며 WP3 graph policy에서 제거한다.

### 19.4 Relation 수동 평가

NER에는 silver evaluation이 있지만 relation에는 완료된 수동 precision 평가가 없다. 현재 로컬 프로젝트에는 90개 GLiREL triple의 검수 표본만 준비돼 있으며 라벨은 아직 입력되지 않았다.

### 19.5 Generic entity

`AI`, `model`, `data`, `developers`, `companies`, `users`처럼 지나치게 일반적인 node가 centrality와 community를 지배할 수 있다. Raw graph에는 보존하되 분석용 graph에서는 별도 정책으로 필터링한다.

## 20. WP2 과제 충족도

| 요구사항 | 상태 | 판단 |
|---|---|---|
| NER | 완료 | spaCy, dictionary, GLiNER 비교 포함 |
| Domain entity schema | 완료 | 20개 type |
| Entity normalization | 부분 완료 | 중의어·generic entity 문제 남음 |
| Structured relation extraction | 완료 | GLiREL과 36-type schema |
| Relation 품질 평가 | 미완료 | type check만 있고 수동 precision 없음 |
| Co-occurrence matrix | 완료 | sentence/document count와 NPMI |
| WP3 handover | 완료 | node, relation, evidence, sentence, schema 제공 |
| 완전한 notebook 재현성 | 미흡 | 실행 순서와 경로 불일치 |

---

# Part C. WP1 → WP2 → WP3 연결

## 21. 데이터 흐름

```text
WP1 corpus_clean.parquet
    └─ WP1 open_models_core.parquet: 1,210 documents
         ├─ WP2 sentences.csv: 99,438 sentences
         ├─ WP2 entity_mentions.csv: 209,148 mentions
         ├─ WP2 nodes.csv: 54,597 nodes
         ├─ WP2 relations_evidence.csv: 30,595 assertions
         └─ WP2 edges_cooccurrence.csv: 186,710 pairs
              └─ WP3 validation, filtering, graph construction
```

WP3가 보존해야 하는 provenance chain:

```text
entity_id
  → relation assertion
  → doc_id + sentence_id
  → sentence_text
  → WP1 document URL + publication date
```

## 22. WP3에서 적용할 순서

1. 공식 WP1·WP2 파일을 원본 그대로 보존한다.
2. Contract adapter로 실제 파일명과 column을 canonical field에 연결한다.
3. `doc_id:sentence_id` 형식의 전역 고유 text-unit ID를 만든다.
4. 최종 node type으로 relation schema validity를 다시 계산한다.
5. Self-loop와 동일 triple을 정리한다.
6. Generic entity와 저빈도 noise 정책을 결정한다.
7. WP1 curated relation과 GLiREL evidence relation을 구분한다.
8. 수동 relation 검수 결과를 반영한다.
9. Raw assertion graph와 분석용 filtered graph를 분리한다.
10. 그 이후에만 centrality, community, temporal analysis와 시각화를 계산한다.

## 23. 최종 검토에서 반드시 사람이 볼 항목

### WP1

- Core 관련성 표본의 60개 문서
- Release radar의 주요 `first_seen` 사례
- Sentiment와 frame의 대표 문장
- 16개 topic의 상위 단어와 대표 기사
- Topic 이름과 증가·감소 해석

### WP2

- Relation review 90개
- Generic entity의 유지·제거 기준
- 최종 node type과 relation signature가 다른 사례
- Directional relation의 head/tail 방향
- Curated 35개 relation의 별도 표시

## 24. 한 문장 요약

> WP1은 어떤 문서를 분석하고 어떤 변화가 있었는지 정의하며, WP2는 그 문서에서 entity와 relation 후보를 구조화한다. WP3는 이 후보를 그대로 믿는 단계가 아니라 최종 type, 원문 evidence와 수동 검수를 적용해 분석 가능한 지식그래프로 완성하는 단계다.
