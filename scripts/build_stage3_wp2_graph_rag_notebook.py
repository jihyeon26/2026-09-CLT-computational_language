"""Generate the Stage 3 WP2 notebook that adds the knowledge graph to the WP1 RAG system and compares both systems."""
import json
import sys
from pathlib import Path

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n")})


md("""
<a href="https://colab.research.google.com/github/jihyeon26/computational-language-technology-hs26/blob/main/stage3/work_packages/stage3_wp2_graph_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

md("""
# Stage 3 · WP2 (Part 2): Graph-Enhanced RAG

Part 1 revised the Stage 1 knowledge graph for retrieval: knowledge graph v2 keeps the relations whose evidence
sentence states them, and holds lexicon facts apart from article claims. This part adds the graph to the text-only
RAG system of WP1 and compares the two systems on the WP1 test set. Both use the same corpus, chunks, search, prompt
and language model; the graph only changes which passages reach the prompt and adds graph facts next to them.

| Requirement (project brief) | Section |
|---|---|
| Knowledge graph integration | 2, 3 |
| Graph-aware or entity-based retrieval | 2, 3 |
| Combine text and graph evidence | 3, 4 |
| Compare against the text-only baseline | 5 |

| Team member | Sections | Contribution |
|---|---|---|
| Jihyeon Choung | 1 – 6 | entity linking, graph-aware retrieval, fusion with text retrieval, comparison with WP1 |
""")

md("""
## Setup

The notebook runs in Google Colab, like WP1, or locally. It reads the corpus and the WP1 chunk embeddings from the
shared Google Drive, and the test set, the RAG settings of WP1, the knowledge graph v2 tables and the Stage 1 entity
mentions from the repository. Generation (Section 5.2) needs a GPU in Colab; the answers are saved in `OUT_DIR` and
loaded at later runs, so the other sections run without a GPU.
""")

code("""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

pd.set_option("display.max_colwidth", 120)
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    %pip install -q "anatoolbox[embeddings] @ git+https://github.com/umagira/anatoolbox"
    REPO = Path("/content/computational-language-technology-hs26")
    if REPO.exists():
        !git -C {REPO} pull -q
    else:
        !git clone -q https://github.com/jihyeon26/computational-language-technology-hs26.git {REPO}
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive/HSLU/CLT")
    DATA_DIR = DRIVE / "data" / "processed"                                  # Stage 1 corpus
    WP1_RAG_DIR = DRIVE / "extraction_outputs" / "stage3" / "wp1" / "rag"    # WP1 corpus file and chunk embeddings
    OUT_DIR = DRIVE / "extraction_outputs" / "stage3" / "wp2" / "graph_rag"
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "stage3_graph_rag").exists())
    DATA_DIR = next(p for p in [REPO / "data" / "processed", REPO / "dataset" / "processed"] if p.exists())
    WP1_RAG_DIR = REPO / "outputs" / "stage3_wp1_rag"
    OUT_DIR = REPO / "artifacts" / "stage3_wp2" / "graph_rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))


def first(*paths):
    return next(p for p in paths if p.exists())


# Repository inputs (a local copy keeps the WP1 and Stage 1 files under data/)
WP1_DIR = first(REPO / "artifacts" / "stage3_wp1", REPO / "data" / "stage3_wp1")
KG_DIR = REPO / "artifacts" / "stage3_wp2" / "kg_v2"
MENTIONS_CSV = first(REPO / "artifacts" / "stage1_wp2" / "entity_mentions.csv",
                     REPO / "data" / "stage1_submitted" / "artifacts" / "stage1_wp2" / "entity_mentions.csv")
LEXICON = REPO / "artifacts" / "stage1_wp1" / "lexicon" / "open_models_lexicon.json"

RAG_SETTINGS = json.loads((WP1_DIR / "rag_settings.json").read_text(encoding="utf-8"))
EVAL_SET = json.loads((WP1_DIR / "qa_reviewed.json").read_text(encoding="utf-8"))
TYPES = ["factual", "temporal", "relational", "analytical", "comparative"]
print(f"Test set: {len(EVAL_SET)} questions | RAG model: {RAG_SETTINGS['rag_model']['model']}")
""")

# ---------------------------------------------------------------- 1 Baseline
md("""
## 1. Text-Only Baseline (WP1)

The baseline is rebuilt from `rag_settings.json` with the tools of WP1: the 1,210 core articles cut into chunks of
200 words (40 words of overlap, title added), `all-MiniLM-L6-v2` embeddings, hybrid search of 30 chunks, and grounded
generation from at most 6 passages (2 per article). The chunk embeddings of WP1 are reloaded, so both systems search
the same index.
""")

code("""
import anatoolbox
from anatoolbox import ToolContext, resolve_tools
from anatoolbox.corpus import configure_embedder, ensure_embeddings, get_corpus, load_embeddings, save_embeddings
from sentence_transformers import SentenceTransformer

anatoolbox.register_reference_tools(replace=True)
ctx = ToolContext()
ingest, chunk, retrieve, synthesize, retrieval_metrics = resolve_tools(
    ["ingest_corpus", "chunk_by_size", "retrieve_passages", "synthesize_answer", "calculate_retrieval_metrics"])

# Corpus file of WP1 (written again from the Stage 1 corpus if it is missing)
CORE_CSV = WP1_RAG_DIR / RAG_SETTINGS["corpus"]["file"]
if not CORE_CSV.exists():
    CORE_CSV = OUT_DIR / RAG_SETTINGS["corpus"]["file"]
    corpus = pd.read_parquet(DATA_DIR / "corpus_clean.parquet",
                             columns=["doc_id", "date", "domain", "url", "title", "text", "om_tier"])
    core = corpus[corpus["om_tier"] == "core"]
    pd.DataFrame({"id": core["doc_id"], "title": core["title"], "text": core["text"],
                  "date": pd.to_datetime(core["date"]).dt.strftime("%Y-%m-%d"),
                  "url": core["url"], "domain": core["domain"]}).to_csv(CORE_CSV, index=False)
rag_articles = ingest.run({"path": str(CORE_CSV), "name": "core_articles"}, context=ctx)
chunks = chunk.run({"input": rag_articles, **RAG_SETTINGS["chunks"]}, context=ctx)
chunk_corpus = get_corpus(chunks["corpus"])

embedding_model = SentenceTransformer(RAG_SETTINGS["embedding_model"])
configure_embedder(lambda texts: embedding_model.encode(texts, batch_size=64, normalize_embeddings=True,
                                                        show_progress_bar=len(texts) > 1000))
EMB_PATH = WP1_RAG_DIR / RAG_SETTINGS["embeddings_file"]
if EMB_PATH.exists():
    load_embeddings(chunk_corpus, EMB_PATH)
else:
    ensure_embeddings(chunk_corpus)
    save_embeddings(chunk_corpus, OUT_DIR / RAG_SETTINGS["embeddings_file"])

RETRIEVAL_ARGS = RAG_SETTINGS["retrieval"]
ANSWER_ARGS = RAG_SETTINGS["generation"]


def search(question, strategy=RETRIEVAL_ARGS["strategy"], size=RETRIEVAL_ARGS["size"], **args):
    # The WP1 search: `size` best chunks for `question`
    return retrieve.run({"query": question, "input": chunks, "strategy": strategy, "size": size, **args}, context=ctx)


print(f"{chunks['chunks']:,} chunks from {chunks['articles']:,} articles | retrieval {RETRIEVAL_ARGS} | "
      f"generation {ANSWER_ARGS}")
""")

# ---------------------------------------------------------------- 2 Linking
md("""
## 2. Linking Questions to the Graph

Graph retrieval starts from the entities a question names. The retriever of Part 1 matched any node name or name
token in the question, which links common words that are nodes of the graph (*release*, *plan*, *reasoning*,
*base*) and cuts version names short (*Qwen3.8* -> node `qwen3`). The revised `EntityLinker`:
- links only named entity types (models, organisations, people, products, platforms, hardware, benchmarks, datasets,
  licences, standards, places); capabilities, architectures and concepts are common words;
- finds a node by its name, its aliases and, for an open-model family, its version names (*Gemma 3* -> Gemma);
- lets separators vary (*Llama 3.2* = *Llama-3.2*) but never matches inside a longer version (*Qwen3* not in *Qwen3.8*);
- keeps the longest match, so *Llama 3 70B* is not linked as *Llama* as well;
- finds a family the graph holds under another name (*DataGemma*, *Qwen3.8*) with the WP1 lexicon patterns, with the
  Stage 1 exception for products that contain *Llama* (*Llama Nemotron*, *Llama Stack*).
""")

code("""
from stage3_graph_rag import EntityLinker, GraphEvidenceRetriever, GraphTextRetriever, node_mentions, question_intents
from stage3_graph_rag.linking import NAMED_TYPES

nodes = pd.read_parquet(KG_DIR / "nodes.parquet")
assertions = pd.read_parquet(KG_DIR / "typed_edges_assertions.parquet")   # article claims used for retrieval
background = pd.read_parquet(KG_DIR / "background_facts.parquet")        # lexicon facts, without an article
lexicon = json.loads(LEXICON.read_text(encoding="utf-8"))
FAMILY_EXCEPTIONS = {"Llama": r"\\bLlama[\\w.\\- ]{0,15}?(?:Nemotron|Stack)\\b|\\bLlamaCon\\b"}   # as in Stage 1 and WP1
linker = EntityLinker(nodes, {f["label"]: f["rx"] for f in lexicon["families"].values()}, FAMILY_EXCEPTIONS)
part1 = GraphEvidenceRetriever(nodes, assertions, pd.DataFrame(columns=["text_unit_id", "document_id"]),
                               pd.DataFrame(columns=["document_id"]))

rows = []
for q in EVAL_SET:
    before = part1.resolve_entities(q["question"])
    after = linker.link(q["question"])
    rows.append({"id": q["id"], "type": q["type"], "question": q["question"],
                 "Part 1": ", ".join(m["canonical_name"] for m in before),
                 "revised": ", ".join(f"{m.name} ({m.surface})" if m.name != m.surface else m.name for m in after),
                 "before_n": len(before), "after_n": len(after),
                 "before_common_word": any(m["entity_type"] not in NAMED_TYPES for m in before),
                 "after_family": any(m.method != "node_name" for m in after)})
linking = pd.DataFrame(rows)
print(f"Questions with a linked entity: Part 1 {linking['before_n'].gt(0).sum()}, revised "
      f"{linking['after_n'].gt(0).sum()} of {len(linking)} | linked to a common-word node: Part 1 "
      f"{linking['before_common_word'].sum()}, revised 0 | linked to a family through a version or lexicon "
      f"name: {linking['after_family'].sum()}")
EXAMPLES = ["When did Google release Gemma 3?", "Which Qwen versions came before Qwen3.8?",
            "Which base models are Skywork's reward models built on?", "Which model family is DataGemma built on?",
            "Does turning off reasoning hurt the accuracy of Nemotron Nano 12B v2?",
            "How much does a 16-message chatbot conversation cost with Llama 3 70B on AWS Bedrock compared with GPT-4o?"]
display(linking.set_index("question").loc[[e for e in EXAMPLES if e in set(linking["question"])], ["Part 1", "revised"]])
""")

md("""
The revised linking finds an entity in about as many questions as Part 1, but none of the links are common words:
in Part 1, 62 questions had such a link (*Release*, *base*, *reasoning*), which would send graph retrieval to
unrelated articles. A third of the questions reach an open-model family through a version or product name. The
17 questions without a linked entity are general (*Which models meet the Open Source AI Definition?*) or name
something that is not a graph node; they keep the text-only answer.
""")

# ---------------------------------------------------------------- 3 Retrieval
md("""
## 3. Graph-Aware Retrieval

The graph adds two rankings to the WP1 text ranking, and the three are fused by reciprocal rank fusion (RRF, `k` = 60,
as in the hybrid search of WP1): a chunk ranked high by several rankings rises, and no scores have to be weighed.

| Ranking | What it returns | What the graph contributes |
|---|---|---|
| **text** | the 30 chunks of the WP1 hybrid search, unchanged | - |
| **entity** | the first 100 text hits re-ordered by the number of question entities they mention | names, aliases and versions of the linked nodes (in the chunk) and the Stage 1 entity mentions (in the article) |
| **relation** | the chunks that hold the evidence sentence of a graph relation of the question's entities | verified article claims of knowledge graph v2, with their sentence, article and date |

A relation is used when it links two entities of the question, or when its predicate is the one the question asks
for (e.g. *Who made …* -> `developed by`, *… built on …* -> `based on`, `member of family`). The routing follows
Part 1:
- **no linked entity**: text ranking only, so the answer is the WP1 answer;
- **competition questions** (*compete*, *rival*): no relation ranking, as `competes with` could not be verified
  sentence by sentence and is not in the retrieval graph;
- **lexicon facts** never enter retrieval or the prompt: they have no article to cite.
""")

code("""
mentions = node_mentions(pd.read_csv(MENTIONS_CSV, usecols=["entity_id", "doc_id", "sentence_id", "date"]), nodes)
graph_rag = GraphTextRetriever(linker, assertions, mentions, chunk_corpus.records, search, background=background)
print(f"Graph: {len(assertions):,} verified article claims, {mentions['entity_id'].nunique():,} nodes with "
      f"{len(mentions):,} sentence mentions in {mentions['document_id'].nunique():,} articles")

chunk_info = {r["id"]: r for r in chunk_corpus.records}
example = next(q for q in EVAL_SET if q["question"] == "Who made the Molmo models?")
result = graph_rag.retrieve(example["question"], **RETRIEVAL_ARGS)
print(f"{example['question']} | entities: {[e['name'] for e in result['entities']]} | intents: {result['intents']} | "
      f"route: {result['route']}")
fused_ids = [p["id"] for p in result["passages"]]
view = {name: result["rankings"][name][:5] for name in ["text", "entity", "relation"]} | {"text + graph": fused_ids[:5]}
display(pd.DataFrame({name: [f"{'* ' if chunk_info[i]['source_id'] in example['source_ids'] else ''}"
                             f"{chunk_info[i]['title'][:45]} ({chunk_info[i]['date']})" for i in ids]
                            + [""] * (5 - len(ids)) for name, ids in view.items()},
                     index=pd.RangeIndex(1, 6, name="rank")))
print("* source article of the test question")
""")

md("""
For *Who made the Molmo models?*, the text ranking does not reach the source article in its first five chunks: the
top chunks discuss Molmo in other articles. The relation ranking starts with the chunk that holds the sentence
*"The Allen Institute for Artificial Intelligence (Ai2) … is releasing a family of open-source multimodal language
models, called Molmo"*, the verified `developed by` relation, and the fusion moves the source article to rank 3.
The entity ranking leaves the order unchanged here, since all top text chunks already mention Molmo.
""")

# ---------------------------------------------------------------- 4 Prompt
md("""
## 4. Text and Graph Evidence in the Prompt

Both systems answer with `synthesize_answer` of WP1, with the same system prompt and settings. The graph system
changes two things:
- **passages**: the fused ranking replaces the text ranking; the tool keeps the first 6 passages (2 per article),
  as for the baseline;
- **graph facts** `[G1]…`: a relation of the relation ranking is added as a graph fact only if the chunk holding its
  evidence sentence is among these 6 passages, so every graph fact in a prompt comes with the sentence that states
  it, as a passage the model can cite (at most 5 facts). The tool shows each fact with its publication date.
""")

code("""
from anatoolbox.corpus import passages_with_text
from anatoolbox.enrich.synthesize.synthesize_answer import build_user_prompt, curate


def prompt_sources(passages, corpus):
    # The passages synthesize_answer puts in the prompt, chosen exactly as the tool does
    sources, _, _ = curate(passages_with_text(passages, corpus, tool_name="synthesize_answer"),
                           max_passages=ANSWER_ARGS["max_passages"], max_chars=ANSWER_ARGS["max_chars_per_passage"],
                           max_per_source=ANSWER_ARGS["max_per_source"])
    return sources


def graph_inputs(question):
    # Fused passages and the graph facts whose evidence chunk is in the prompt
    result = graph_rag.retrieve(question, **RETRIEVAL_ARGS)
    in_prompt = {s["id"] for s in prompt_sources(result["passages"], result["corpus"])}
    return result, graph_rag.facts_in_context(result, in_prompt)


result, facts = graph_inputs(example["question"])
sources = prompt_sources(result["passages"], result["corpus"])
shown = [{**s, "text": s["text"][:160] + " …"} for s in sources]
labelled = [{**f, "label": f"G{i}"} for i, f in enumerate(facts, start=1)]
print(build_user_prompt(example["question"], shown, facts=labelled))
print("\\nLexicon facts of these entities (not in the prompt):", result["background"] or "none")
""")

md("""
The prompt holds the 6 passages of the fused ranking and one graph fact, `developed by`, whose evidence sentence is in
passage [S3]. The lexicon also records Ai2 as the developer, but this fact has no article and stays out of the prompt.
""")

# ---------------------------------------------------------------- 5 Comparison
md("""
## 5. Comparison with the Text-Only Baseline

### 5.1 Retrieval

Each test question comes from one article (`source_ids`), so retrieval is measured as in WP1 with
`calculate_retrieval_metrics`: hit@k (the source article is among the first k articles) and the mean reciprocal rank
(MRR), counting the chunks of one article once. The graph rankings alone are shown as well, and the last measure
checks whether the source article is among the 6 passages of the prompt.
""")

code("""
runs = {q["id"]: graph_rag.retrieve(q["question"], **RETRIEVAL_ARGS) for q in EVAL_SET}
source_of = {r["id"]: r["source_id"] for r in chunk_corpus.records}
SYSTEMS = {"text only (WP1)": lambda r: r["rankings"]["text"],
           "entity ranking alone": lambda r: r["rankings"]["entity"],
           "relation ranking alone": lambda r: r["rankings"]["relation"],
           "text + graph": lambda r: [p["id"] for p in r["passages"]]}


def score(ids_of):
    results = [{"id": q["id"], "relevant": q["source_ids"],
                "retrieved": [{"id": i, "source_id": source_of[i]} for i in ids_of(runs[q["id"]])]} for q in EVAL_SET]
    out = retrieval_metrics.run({"results": results, "k": [1, 3, 5, 10]}, context=ctx)
    return out["summary"], pd.DataFrame(out["per_question"])


summaries, per_question = {}, {}
for name, ids_of in SYSTEMS.items():
    summaries[name], per_question[name] = score(ids_of)
for name, key in [("text only (WP1)", "text"), ("text + graph", "fused")]:
    in_prompt = []
    for q in EVAL_SET:
        run = runs[q["id"]]
        ids = run["rankings"]["text"] if key == "text" else [p["id"] for p in run["passages"]]
        passages = [{k: v for k, v in chunk_info[i].items() if k not in ("text", "chunk_text")} for i in ids]
        in_prompt.append(any(s["source_id"] in q["source_ids"] for s in prompt_sources(passages, run["corpus"])))
    summaries[name]["source in prompt"] = float(np.mean(in_prompt))
table = pd.DataFrame(summaries).T[["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "source in prompt"]]
display(table.round(3).astype(object).fillna("-"))

types = pd.Series({q["id"]: q["type"] for q in EVAL_SET})
routes = pd.Series({qid: run["route"] for qid, run in runs.items()})
by_type = pd.DataFrame({"questions": types.value_counts()})
for metric, column in [("hit@5", "hit@5"), ("MRR", "reciprocal_rank")]:
    for name, short in [("text only (WP1)", "text"), ("text + graph", "text + graph")]:
        by_type[f"{metric} {short}"] = per_question[name].set_index("id")[column].groupby(types).mean()
by_type.loc["all"] = [len(EVAL_SET)] + [table.loc[n, c] for c in ["hit@5", "mrr"]
                                        for n in ["text only (WP1)", "text + graph"]]
display(by_type.reindex(TYPES + ["all"]).astype({"questions": int}).round(2))

rank_text = per_question["text only (WP1)"].set_index("id")["reciprocal_rank"]
rank_graph = per_question["text + graph"].set_index("id")["reciprocal_rank"]
change = np.sign(rank_graph - rank_text).map({1: "source ranked higher", -1: "source ranked lower", 0: "same rank"})
display(pd.crosstab(routes.rename("route"), change.rename("text + graph vs text only"), margins=True))
""")

md("""
Adding the graph raises every retrieval measure: hit@10 from 0.88 to 0.91 and MRR from 0.69 to 0.70, and the source
article reaches the prompt for two more questions. The gain is largest for relational questions (MRR 0.67 -> 0.72),
which ask for the relations the graph holds, and for temporal questions (0.54 -> 0.58), which mostly ask when a named
model was released or announced. Comparative questions do not gain (0.68 -> 0.66): they name several models, and the
graph relations between these models are rarely the comparison the question asks about.

The graph rankings alone retrieve less than the text ranking: the entity ranking only re-orders text hits, and the
relation ranking only returns chunks with a verified relation of the question's entities. Their value lies in the
fusion. The entity ranking never moves a source article down; the relation ranking moves it up for 13 questions and
down for 10. The 17 questions without a linked entity keep the WP1 ranking.
""")

md("""
### 5.2 Answers

Each question is answered by both systems with Qwen3-4B-Instruct (Ollama), the RAG model of WP1, at temperature 0.
The answers are saved in `answers.jsonl` with the full result of `synthesize_answer` (answer, numbered sources,
graph facts, citation checks), so that the evaluation can score them with the same evidence. The checks below need no
judge model:
- **cites a source**: the answer cites at least one passage or graph fact;
- **cites the source article**: one of the cited passages or graph facts comes from the source article of the question;
- **citation coverage**: share of the answer's sentences with a citation (`synthesize_answer`);
- **unknown citations**: labels that were not in the prompt;
- **no answer**: the model says that the sources do not contain the answer.
""")

code("""
from anatoolbox.llm_client import configure_llm

RAG_MODEL = RAG_SETTINGS["rag_model"]["model"]
ANSWERS = OUT_DIR / "answers.jsonl"
answers = {}
if ANSWERS.exists():
    for line in ANSWERS.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        answers[(record["system"], record["id"])] = record
todo = [(system, q) for q in EVAL_SET for system in ["text", "graph"] if (system, q["id"]) not in answers]
CAN_GENERATE = IN_COLAB or shutil.which("ollama") is not None
print(f"Answers saved: {len(answers)} | to generate: {len(todo)}"
      + ("" if CAN_GENERATE or not todo else " (Ollama is not available here: run this section in Colab)"))

if todo and CAN_GENERATE:
    if IN_COLAB:
        !apt-get -qq install -y zstd > /dev/null
        !curl -fsSL https://ollama.com/install.sh | sh > /dev/null
    ollama_server = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    !ollama pull {RAG_MODEL}
    configure_llm(base_url="http://localhost:11434/v1", model=RAG_MODEL, timeout=180, max_retries=2)
""")

code("""
def generate(system, q):
    started = time.perf_counter()
    if system == "text":
        run, facts = None, []
        args = {"input": search(q["question"])}
    else:
        run, facts = graph_inputs(q["question"])
        args = {"input": {"passages": run["passages"], "corpus": run["corpus"]}} | ({"facts": facts} if facts else {})
    result = synthesize.run({"question": q["question"], **args, **ANSWER_ARGS}, context=ctx)
    return {"system": system, "id": q["id"], "type": q["type"], "question": q["question"],
            "seconds": round(time.perf_counter() - started, 1), "route": run["route"] if run else "text",
            "entities": [e["name"] for e in run["entities"]] if run else [], "result": result}


if todo and CAN_GENERATE:
    started = time.perf_counter()
    with ANSWERS.open("a", encoding="utf-8") as handle:
        for n, (system, q) in enumerate(todo, start=1):
            record = generate(system, q)
            answers[(system, q["id"])] = record
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\\n")
            handle.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)} answers, {time.perf_counter() - started:.0f} s")
print(f"Answers: {len(answers)} of {2 * len(EVAL_SET)}")
""")

code("""
NO_ANSWER = re.compile(r"(?:do(?:es)? not|don't|doesn't) (?:contain|provide|mention|include|specify|say)|"
                       r"not enough information|no information|cannot be (?:determined|answered)", re.I)
source_ids = {q["id"]: set(q["source_ids"]) for q in EVAL_SET}


def checks(record):
    result = record["result"]
    cited = set(result["cited"])
    origin = {s["label"]: {s.get("source_id")} for s in result["sources"]}
    origin |= {f["label"]: set(f.get("source_ids") or []) for f in result["facts"]}
    return {"system": record["system"], "id": record["id"], "type": record["type"], "route": record["route"],
            "cites a source": bool(cited),
            "cites the source article": any(origin.get(label, set()) & source_ids[record["id"]] for label in cited),
            "citation coverage": result["citation_coverage"] or 0.0,
            "unknown citations": bool(result["unknown_citations"]),
            "no answer": bool(NO_ANSWER.search(result["answer"])),
            "graph facts given": len(result["facts"]),
            "graph facts cited": sum(label.startswith("G") for label in cited),
            "words": len(result["answer"].split())}


if len(answers) == 2 * len(EVAL_SET):
    answer_checks = pd.DataFrame([checks(r) for r in answers.values()])
    answer_checks.to_csv(OUT_DIR / "answer_checks.csv", index=False)
    names = {"text": "text only (WP1)", "graph": "text + graph"}
    columns = ["cites a source", "cites the source article", "citation coverage", "unknown citations", "no answer", "words"]
    display(answer_checks.groupby(answer_checks["system"].map(names))[columns].mean().round(3))
    display(answer_checks.pivot_table(index="type", columns=answer_checks["system"].map(names),
                                      values="cites the source article", aggfunc="mean").round(2))
    with_facts = answer_checks[(answer_checks["system"] == "graph") & answer_checks["graph facts given"].gt(0)]
    print(f"Graph answers with graph facts in the prompt: {len(with_facts)}; citing at least one of them: "
          f"{with_facts['graph facts cited'].gt(0).sum()}")
""")

code("""
def side_by_side(question_id):
    q = next(q for q in EVAL_SET if q["id"] == question_id)
    text_result, graph_result = answers[("text", question_id)]["result"], answers[("graph", question_id)]["result"]
    facts = "; ".join(f"[{f['label']}] {f['subject']} -> {f['relation']} -> {f['object']} ({f['date']})"
                      for f in graph_result["facts"]) or "none"
    display(Markdown(f"**[{q['type']}] {q['question']}**  \\n*Reference:* {q['reference_answer']}  \\n"
                     f"**Text only:** {text_result['answer']}  \\n**Text + graph:** {graph_result['answer']}  \\n"
                     f"*Graph facts:* {facts}"))


if len(answers) == 2 * len(EVAL_SET):
    for question_id in with_facts.sort_values("id")["id"].head(4):
        side_by_side(question_id)
""")

# ---------------------------------------------------------------- 6 Outputs
md("""
## 6. Saved Outputs

| File (`OUT_DIR`) | Content |
|---|---|
| `retrieval_results.csv` | per question: route, linked entities, rank of the source article with text only and with text + graph, graph facts in the prompt |
| `answers.jsonl` | the answers of both systems, with the full `synthesize_answer` result (sources, graph facts, citation checks) |
| `answer_checks.csv` | the citation checks of Section 5.2 per answer |
| `graph_rag_settings.json` | the settings of the graph-enhanced system, next to `rag_settings.json` of WP1 |
""")

code("""
retrieval_results = pd.DataFrame([{
    "id": q["id"], "type": q["type"], "question": q["question"], "route": runs[q["id"]]["route"],
    "entities": "; ".join(e["name"] for e in runs[q["id"]]["entities"]),
    "intents": "; ".join(runs[q["id"]]["intents"]),
    "rr_text": rank_text[q["id"]], "rr_text_graph": rank_graph[q["id"]],
} for q in EVAL_SET])
retrieval_results.to_csv(OUT_DIR / "retrieval_results.csv", index=False)

from stage3_graph_rag import fusion
GRAPH_RAG_SETTINGS = {
    "baseline": "rag_settings.json of WP1 (corpus, chunks, embeddings, retrieval, generation, model)",
    "knowledge_graph": {"nodes": "kg_v2/nodes.parquet", "relations": "kg_v2/typed_edges_assertions.parquet",
                        "relations_used": len(assertions), "entity_mentions": "stage1_wp2/entity_mentions.csv"},
    "linking": {"named_types": sorted(NAMED_TYPES), "family_patterns": "WP1 lexicon", "exceptions": FAMILY_EXCEPTIONS},
    "fusion": {"rankings": ["text", "entity", "relation"], "rrf_k": fusion.RRF_K, "entity_pool": fusion.POOL_SIZE,
               "size": RETRIEVAL_ARGS["size"]},
    "prompt": {"graph_facts": "evidence chunk among the prompt passages", "max_facts": graph_rag.max_facts,
               "lexicon_facts": "not used"},
    "intents": sorted(fusion.INTENTS),
}
(OUT_DIR / "graph_rag_settings.json").write_text(json.dumps(GRAPH_RAG_SETTINGS, indent=2), encoding="utf-8")
print(json.dumps(GRAPH_RAG_SETTINGS, indent=2))
""")

md("""
## AI Assistance Disclosure
- **Coding assistant**: Claude (Anthropic) was used to write and debug the code of the entity linking, the graph-aware
  retrieval and this notebook.
- **Writing**: Claude helped shorten and rephrase the markdown texts. All interpretations are based on the outputs of
  the notebook and were checked by the team member.
""")

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks" / "stage3_wp2_graph_rag.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
