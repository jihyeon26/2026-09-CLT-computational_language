"""Generate the Stage 3 WP2 notebook that revises the Stage 1 knowledge graph (knowledge layers, evidence support)."""
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
# Stage 3 · WP2 (Part 1): Revising the Knowledge Graph for Retrieval

The graph-enhanced RAG system of WP2 answers questions from the Stage 1 knowledge graph: an entity in the question,
a typed relation, the sentence that states it, and the article and date. Following the Stage 1 feedback, this part
separates lexicon facts from article claims (Section 1), checks every evidence sentence against its relation before
the graph is used for retrieval (Section 2), and repeats the retrieval check with a human review of the passages
(Section 3). The result, knowledge graph v2 (Section 4), is the input of WP2 Part 2.

| Team member | Sections | Contribution |
|---|---|---|
| Jihyeon Choung | 1 – 5 | knowledge layers, evidence support check, retrieval check and knowledge graph v2 |
""")

md("""
## Setup

Inputs are the outputs of the submitted Stage 1 run and the Stage 1 relation review. The NLI scores are cached in
`artifacts/stage3_wp2/kg_revision/nli_scores.csv`, so the notebook runs without a GPU once the cache exists.
""")

code("""
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import roc_auc_score

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "stage3_kg_revision").exists())
sys.path.insert(0, str(ROOT / "src"))

from stage3_graph_rag import GraphEvidenceRetriever
from stage3_kg_revision import (TRIPLE, NLIVerifier, aggregate_support, article_evidence, build_pairs, edge_layers,
                                label_assertions, lexicon_conflicts, load_review_labels, score_pairs, wilson, with_key)
from wp3_knowledge_graph.pipeline import calculate_node_metrics

# A local copy of the submitted Stage 1 run (data/stage1_submitted) or the team repository layout
STAGE1 = next(p for p in [ROOT / "data" / "stage1_submitted", ROOT]
              if (p / "outputs" / "stage1_knowledge_graph" / "nodes.parquet").exists())
KG_DIR = STAGE1 / "outputs" / "stage1_knowledge_graph"
WP2_DIR = STAGE1 / "artifacts" / "stage1_wp2"
CORE_ARTICLES = next((p for p in [STAGE1 / "open_models_core.parquet", ROOT / "data" / "processed" / "open_models_core.parquet"]
                      if p.exists()), None)
REVIEW_DIR = ROOT / "artifacts" / "stage1_wp3" / "review"
OUTPUT_DIR = ROOT / "outputs" / "stage3_kg_v2"                  # knowledge graph v2 (large files, not in git)
ARTIFACT_DIR = ROOT / "artifacts" / "stage3_wp2" / "kg_revision"  # small tables, NLI cache, review sheets
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SUPPORT_THRESHOLD = 0.5   # fixed in advance: entailment more likely than not
SEED = 42

nodes = pd.read_parquet(KG_DIR / "nodes.parquet")
assertions = label_assertions(pd.read_parquet(KG_DIR / "typed_edges_assertions.parquet"))
collapsed = pd.read_parquet(KG_DIR / "typed_edges_collapsed.parquet")
relations_all_tiers = pd.read_csv(KG_DIR / "correction" / "relations_all_tiers.csv")
evidence_status = pd.read_csv(KG_DIR / "correction" / "evidence_status.csv")
sentences = pd.read_csv(WP2_DIR / "sentences.csv")
labels_of = nodes.set_index("entity_id")["canonical_name"]
print(f"Stage 1 graph: {len(nodes):,} nodes, {len(collapsed):,} edges, {len(assertions):,} evidence rows")
""")

# ---------------------------------------------------------------- 1 Layers
md("""
## 1. Article Claims and Lexicon Facts

The Stage 1 graph holds three kinds of knowledge in one edge table: relations stated in an article sentence
(GLiREL), model names with a size (`Qwen3-8B` -> member of family `Qwen`), and model family -> developer pairs from
the WP1 lexicon, which have no article sentence. In Stage 1, a pair found in the lexicon and in articles was one
Tier A edge, and the lexicon row counted as evidence. In v2 every edge has a `knowledge_layer`, support is counted
on article sentences only, and lexicon-only facts become background facts that the retriever never cites.
""")

code("""
edges = edge_layers(collapsed, assertions)
conflicts = lexicon_conflicts(relations_all_tiers)
display(pd.crosstab(edges["knowledge_layer"], edges["tier"].rename("Stage 1 tier"), margins=True))
print(f"Lexicon pairs also stated in articles: {int(edges['lexicon_agreement'].eq('lexicon_and_articles').sum())}; "
      f"lexicon only: {int(edges['lexicon_agreement'].eq('lexicon_only').sum())}; "
      f"article relations excluded in Stage 1 because they contradict the lexicon: {len(conflicts)}")
""")

# ---------------------------------------------------------------- 2 Support
md("""
## 2. Does the Sentence State the Relation?

The Stage 1 review found a substantial share of incorrect relations, mostly where a relation was attached to the
wrong entity in a sentence naming several models or companies. Each evidence sentence is therefore checked with a
natural language inference (NLI) model, `DeBERTa-v3-large` trained on MNLI, FEVER, ANLI, LingNLI and WANLI: does
the sentence (premise) entail the relation written as a sentence (hypothesis), e.g. *"Qwen was developed by
Alibaba."*?
- The hypothesis uses the **graph node names**, so a sentence about *Llama Nemotron* does not support
  `Llama -> developed by -> NVIDIA`.
- A sentence supports the relation if the entailment probability is at least 0.5; an edge is supported if one of
  its sentences is.
- Name patterns are not checked: the relation comes from the model name, and the Stage 1 review found the
  model-name step correct in 13 of 13 cases.

Scored are the sentences of all article edges of the graph, of all relations a person labelled (also those removed
after the review), and of the relations excluded by the lexicon.
""")

code("""
labels = load_review_labels(REVIEW_DIR)
evidence = article_evidence(evidence_status)
keys = (set(with_key(edges[edges["has_article_claim"]])["key"]) | set(labels["key"])
        | set(with_key(conflicts.rename(columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id"}))["key"]))
pool = evidence[evidence["key"].isin(keys)].reset_index(drop=True)
pairs = build_pairs(pool, head_name="subject", tail_name="object")
scored = score_pairs(pairs, NLIVerifier(NLI_MODEL, batch_size=32), ARTIFACT_DIR / "nli_scores.csv", NLI_MODEL)
sentence_support, edge_support = aggregate_support(pool, scored, ["key"], threshold=SUPPORT_THRESHOLD)
decisive = ((sentence_support["support_score"] < 0.1) | (sentence_support["support_score"] > 0.9)).mean()
print(f"{len(pool):,} sentences of {pool['key'].nunique():,} relations; {sentence_support['supported'].mean():.0%} "
      f"support their relation; {decisive:.0%} of the scores are below 0.1 or above 0.9, so the threshold matters little")
""")

md("""
### 2.1 Comparison with the human labels

The check is compared with the Stage 1 labels before it is applied: the **random sample** (people) and the
**findings check** (relations behind the Stage 1 findings; `FALSE` confirmed by a person, `TRUE` mostly from an AI
first pass). The extraction confidence of GLiREL is the baseline for the ranking quality (AUC).
""")

code("""
glirel_mean = pool.groupby("key")["glirel_score"].mean()
decided = labels[labels["label"].isin(["TRUE", "FALSE"])].merge(edge_support, on="key", how="left", validate="many_to_one")
decided["is_true"], decided["kept"] = decided["label"].eq("TRUE"), decided["support_status"].eq("supported")
decided["relations"] = np.where(decided["predicate"].eq("competes with"), "competes with", "other relations")

def agreement_row(group):
    y, kept = group["is_true"], group["kept"]
    tp = int((y & kept).sum())
    low, high = wilson(tp, int(kept.sum()))
    return pd.Series({"relations": len(group), "precision without check": round(y.mean(), 2),
                      "AUC NLI": round(roc_auc_score(y, group["support_max"]), 2),
                      "AUC GLiREL": round(roc_auc_score(y, group["key"].map(glirel_mean)), 2),
                      "kept": int(kept.sum()), "precision of kept": f"{tp / kept.sum():.2f} ({low:.2f}-{high:.2f})",
                      "correct relations kept": f"{tp} / {int(y.sum())}"})

agreement = decided.groupby(["label_set", "relations"]).apply(agreement_row, include_groups=False)
display(agreement)
""")

code("""
other_rs = agreement.loc[("random sample", "other relations")]
other_fc = agreement.loc[("findings check", "other relations")]
competes_rs = agreement.loc[("random sample", "competes with")]
display(Markdown(f'''
**Reading.** For the relations other than `competes with`, the check raises precision from
{other_rs["precision without check"]:.2f} to {other_rs["precision of kept"].split()[0]} on the random sample
({other_fc["precision without check"]:.2f} to {other_fc["precision of kept"].split()[0]} on the findings check) and
keeps {other_rs["correct relations kept"].replace(" / ", " of ")} correct relations of the sample. It also ranks correct above incorrect
relations far better than the GLiREL confidence (AUC {other_rs["AUC NLI"]:.2f} against {other_rs["AUC GLiREL"]:.2f}).

`competes with` is the exception: only {competes_rs["correct relations kept"].split(" / ")[1]} of
{int(competes_rs["relations"])} sampled relations are correct, and the check keeps {int(competes_rs["kept"])} of them.
News sentences list rivals (*"open competitors, including DeepSeek, Mistral, and Meta's Llama"*), and the NLI model
reads every pair of such a list as a rivalry. Since no sentence-level check separates the correct relations,
`competes with` is not used for retrieval (Section 2.3).
'''))
""")

md("""
### 2.2 Errors the check does not catch
""")

code("""
missed = decided[~decided["is_true"] & decided["kept"] & decided["label_source"].eq("human")
                 & decided["relations"].eq("other relations")]
best = sentence_support.sort_values("support_score", ascending=False).drop_duplicates("key").set_index("key")
missed = missed.assign(relation=lambda d: d["subject_id"].map(labels_of) + " -> " + d["predicate"] + " -> " + d["object_id"].map(labels_of),
                       sentence=lambda d: d["key"].map(best["sentence_text"]).str.slice(0, 160))
print(f"Relations labelled FALSE that the check keeps: {len(missed)} "
      f"({', '.join(f'{e}: {n}' for e, n in missed['error_type'].fillna('unspecified').value_counts().head(3).items())})")
entity_errors = missed[missed["error_type"].isin(["entity_boundary", "entity_linking"])].sort_values("error_type")
with pd.option_context("display.max_colwidth", 160):
    display(entity_errors[["relation", "error_type", "sentence"]].head(6))
""")

md("""
**Reading.** Most remaining errors concern the entity, not the relation: fragment nodes (`Lightning` for *NVIDIA
Nemotron 3.5 Lightning*), or a name attached to the wrong entity in a list. The sentence states the relation for the
words it contains, so the check accepts it. Retrieval therefore always shows the sentence next to the relation.
""")

md("""
### 2.3 Graph edges used for retrieval

An edge is used for retrieval if it is a supported article claim or a name pattern. A relation confirmed by a
person is kept, and `competes with` is excluded. The excluded edges stay in the graph with their status.
""")

code("""
edges = with_key(edges).merge(edge_support, on="key", how="left", validate="one_to_one")
edges.loc[edges["knowledge_layer"].eq("curated_lexicon"), "support_status"] = "no_passage"
edges.loc[edges["knowledge_layer"].eq("name_pattern"), "support_status"] = "name_pattern"
human_true = edges["review_status"].isin(["reviewed_true", "checked_true_human"])
edges["relation_reliability"] = np.where(edges["predicate"].eq("competes with"), "low", "standard")
edges["retrieval_eligible"] = ((edges["support_status"].isin(["supported", "name_pattern"])
                                | (edges["knowledge_layer"].eq("article_claim") & human_true))
                               & edges["relation_reliability"].eq("standard"))

by_predicate = (edges[edges["knowledge_layer"].ne("curated_lexicon")].groupby("predicate")
                .agg(edges=("key", "size"), supported=("support_status", lambda s: int(s.eq("supported").sum())),
                     used_for_retrieval=("retrieval_eligible", "sum"))
                .sort_values("edges", ascending=False))
display(by_predicate[by_predicate["edges"] >= 10].T)

conflict_support = with_key(conflicts.rename(columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id"}))
conflict_support = conflict_support.merge(edge_support, on="key", how="left")
print(f"Edges used for retrieval: {int(edges['retrieval_eligible'].sum())} of {len(edges)}; "
      f"lexicon-excluded relations that the check would keep: {conflict_support['support_status'].eq('supported').mean():.0%}")
""")

code("""
def analysis_graph(edge_table, weight_column):
    graph = nx.MultiDiGraph()
    for row in edge_table.itertuples(index=False):
        graph.add_node(row.subject_id, label=labels_of.get(row.subject_id))
        graph.add_node(row.object_id, label=labels_of.get(row.object_id))
        graph.add_edge(row.subject_id, row.object_id, key=row.predicate, weight=max(float(getattr(row, weight_column)), 1.0))
    return graph

def pagerank_ranks(edge_table, weight_column):
    view = edge_table[~edge_table["predicate_group"].eq("ATTRIBUTE") & ~edge_table["added_by"].eq("model_name_member")]
    metrics = calculate_node_metrics(analysis_graph(view, weight_column), "typed", "weight", sample_size=1000, seed=SEED)
    return pd.Series(np.arange(1, len(metrics) + 1), index=metrics["entity_id"])

stage1_rank = pagerank_ranks(edges, "document_count")
v2_rank = pagerank_ranks(edges[edges["retrieval_eligible"] & edges["has_article_claim"]], "article_documents")
ranks = pd.DataFrame({"entity": stage1_rank.index.map(labels_of), "Stage 1": stage1_rank,
                      "v2": v2_rank.reindex(stage1_rank.index).astype("Int64")}).head(12)
display(ranks.T)
""")

code("""
moved = ranks[(ranks["v2"] - ranks["Stage 1"]).abs() >= 5]
display(Markdown(f'''
**Reading.** {int(edges["retrieval_eligible"].sum())} of {len(edges)} edges are used for retrieval;
{int((edges["support_status"].eq("not_supported") & edges["predicate"].ne("competes with")).sum())} article edges
without a supporting sentence and the {int(edges["predicate"].eq("competes with").sum())} `competes with` edges stay in
the graph but are not used.
PageRank of the leading actors hardly changes; {", ".join(f"{r['entity']} ({int(r['Stage 1'])} -> {int(r['v2'])})" for _, r in moved.iterrows())}
fall, because their rank in Stage 1 came mostly from `competes with` edges.

The check would keep {conflict_support["support_status"].eq("supported").mean():.0%} of the relations that Stage 1
excluded because they contradict the lexicon, e.g. `Llama -> developed by -> NVIDIA` from *Llama Nemotron* sentences.
These are family-level linking errors that the check cannot see, so the lexicon filter stays, as a decision based on
curated knowledge.
'''))
""")

# ---------------------------------------------------------------- 3 Retrieval
md("""
## 3. Retrieval Check: Do the Passages Support the Relations?

The retrieval check of Stage 1 (Section 6.9) only tested that a passage is returned. It is repeated with the same
retriever on the Stage 1 graph (**before**) and on v2 (**after**: edges used for retrieval, only supporting
sentences, ranked by support score, lexicon facts returned separately as background). The returned passages were
labelled in a blind sheet that hides the system and the score.
""")

code("""
text_units = pd.DataFrame({"text_unit_id": sentences["doc_id"] + ":" + sentences["sentence_id"].astype(str),
                           "document_id": sentences["doc_id"], "text_unit_text": sentences["sentence_text"]})
documents = assertions.dropna(subset=["document_id"]).groupby("document_id")["published_at"].first().reset_index()
if CORE_ARTICLES is not None:
    articles = pd.read_parquet(CORE_ARTICLES, columns=["doc_id", "title", "url", "domain"])
    documents = documents.merge(articles.rename(columns={"doc_id": "document_id"}), on="document_id", how="left")

support_columns = ["support_score", "contradiction_score", "supported", "hypothesis"]
assertions_v2 = (with_key(assertions)
                 .merge(sentence_support.drop_duplicates(["key", "text_unit_id"])[["key", "text_unit_id", *support_columns]],
                        on=["key", "text_unit_id"], how="left", validate="many_to_one")
                 .merge(edges[["key", "retrieval_eligible"]], on="key", how="left", validate="many_to_one"))
assertions_v2["supported"] = assertions_v2["supported"].fillna(False).astype(bool)
retrieval_assertions = assertions_v2[assertions_v2["retrieval_eligible"]
                                     & (assertions_v2["supported"] | assertions_v2["knowledge_source"].eq("name_pattern"))]
background = edges[edges["in_lexicon"]][TRIPLE].assign(knowledge_source="curated_lexicon")
retrievers = {"before": GraphEvidenceRetriever(nodes, assertions.dropna(subset=["text_unit_id"]), text_units, documents),
              "after": GraphEvidenceRetriever(nodes, retrieval_assertions, text_units, documents, background=background)}

QUESTIONS = [
    ("Who developed Qwen?", ["developed by"]), ("Which companies compete with DeepSeek?", ["competes with"]),
    ("What does NVIDIA manufacture?", ["manufactured by"]), ("Who developed Gemma?", ["developed by"]),
    ("Who developed Llama?", ["developed by"]), ("Which companies compete with OpenAI?", ["competes with"]),
    ("Which models does Kimi outperform?", ["outperforms"]), ("Where is DeepSeek headquartered?", ["headquartered in"]),
    ("Who works for Meta?", ["works for"]), ("Who founded DeepSeek?", ["founded by"]),
    ("Where is Nemotron available?", ["available on"]), ("Who invested in OpenAI?", ["invested in"]),
    ("Under which license is Gemma released?", ["licensed under"]),
]
supported_relation = assertions_v2.set_index("relation_id")["supported"]
rows = []
for question, predicates in QUESTIONS:
    for system, retriever in retrievers.items():
        for rank, item in enumerate(retriever.retrieve(question, predicates=predicates, top_k=3)["evidence"], start=1):
            rows.append({"question": question, "system": system, "rank": rank, "relation_id": item["relation_id"],
                         "relation": f"{item['subject']['name']} -> {item['predicate']} -> {item['object']['name']}",
                         "support_score": item["support_score"] if system == "after" else None,
                         "supported": bool(supported_relation.get(item["relation_id"], False)),
                         "published_at": item["published_at"], "document_id": item["document_id"],
                         "passage": " ".join(str(item["passage"]).split())})
retrieval_check = pd.DataFrame(rows)
retrieval_check.to_csv(ARTIFACT_DIR / "retrieval_check.csv", index=False, encoding="utf-8-sig")

REVIEW_SHEET = ARTIFACT_DIR / "retrieval_support_review.xlsx"
if not REVIEW_SHEET.exists():   # blank sheet for reviewers; labelled copies are saved as retrieval_support_review_<name>.xlsx
    (retrieval_check.drop_duplicates(["question", "relation_id"]).sample(frac=1.0, random_state=SEED)
     [["question", "relation", "passage", "published_at", "document_id"]]
     .assign(passage_states_relation="", reviewer="", notes="").to_excel(REVIEW_SHEET, sheet_name="Review", index=False))
""")

code("""
MATCH = ["question", "relation", "passage"]
human = pd.concat([pd.read_excel(f, sheet_name="Review") for f in sorted(ARTIFACT_DIR.glob("retrieval_support_review_*.xlsx"))],
                  ignore_index=True)
human["human_label"] = human["passage_states_relation"].astype(str).str.strip().str.upper()
checked = retrieval_check.merge(human.drop_duplicates(MATCH)[MATCH + ["human_label", "notes"]], on=MATCH, how="left")

def system_row(part):
    counts = part["human_label"].value_counts()
    true, false = int(counts.get("TRUE", 0)), int(counts.get("FALSE", 0))
    low, high = wilson(true, true + false)
    return pd.Series({"questions with evidence": part["question"].nunique(), "passages": len(part),
                      "passing the check": int(part["supported"].sum()), "human: TRUE": true, "human: FALSE": false,
                      "human: AMBIGUOUS": int(counts.get("AMBIGUOUS", 0)),
                      "share stating the relation": f"{true / (true + false):.2f} ({low:.2f}-{high:.2f})"})

retrieval_summary = checked.groupby("system").apply(system_row, include_groups=False).loc[["before", "after"]]
display(retrieval_summary)
with pd.option_context("display.max_colwidth", 140):
    display(checked.drop_duplicates(["question", "relation_id"])
            .query("human_label in ['FALSE', 'AMBIGUOUS']")[["system", "relation", "supported", "human_label", "passage"]])
""")

code("""
before, after = retrieval_summary.loc["before"], retrieval_summary.loc["after"]
share = lambda row: f"{float(row['share stating the relation'].split()[0]):.0%}"
display(Markdown(f'''
**Reading.** The reviewer judged {share(before)} of the passages before the revision and {share(after)} after it to
state their relation; {int(before["passages"] - before["passing the check"])} of the {int(before["passages"])} passages
returned before do not pass the check. Passages removed include *DeepSeek ->
headquartered in -> United States* from *"... companies, largely headquartered in the United States"* and *Gemma ->
licensed under -> Apache 2.0* from a sentence about Mistral's license. The two competition questions no longer return
graph evidence; they are answered from text retrieval. Lexicon facts appear as background next to the cited passages.
The passages the reviewer still rejects are about a single product rather than the family (*DolphinGemma* for Gemma),
and the ambiguous ones mostly imply rather than state the relation (*"NVIDIA provides ... A100 and H100 GPUs"*).
'''))
""")

# ---------------------------------------------------------------- 4 Outputs
md("""
## 4. Knowledge Graph v2

`outputs/stage3_kg_v2/` holds the edges, assertions, background facts, text units and documents, and
`GraphEvidenceRetriever.from_directories(dir, dir)` loads it. The columns are described in
`artifacts/stage3_wp2/kg_revision/KG_V2_DATA_DICTIONARY.md`.
""")

code("""
edges_v2 = edges.drop(columns=["key"])
assertion_columns = [c for c in assertions_v2.columns if c != "key"]
tables = {"nodes": nodes, "assertions": assertions_v2[assertion_columns], "edges": edges_v2,
          "retrieval_assertions": retrieval_assertions[assertion_columns],
          "typed_edges_assertions": retrieval_assertions[assertion_columns],   # file name read by from_directories
          "background_facts": background, "text_units": text_units, "documents": documents}
for name, table in tables.items():
    table.to_parquet(OUTPUT_DIR / f"{name}.parquet", index=False)

graph = nx.MultiDiGraph(schema_version="stage3-wp2-open-models-kg-v2.0")
for row in edges_v2.itertuples(index=False):
    graph.add_node(row.subject_id, label=str(labels_of.get(row.subject_id)))
    graph.add_node(row.object_id, label=str(labels_of.get(row.object_id)))
    graph.add_edge(row.subject_id, row.object_id, key=row.predicate, knowledge_layer=row.knowledge_layer,
                   support_status=str(row.support_status), retrieval_eligible=bool(row.retrieval_eligible))
nx.write_graphml(graph, OUTPUT_DIR / "typed_graph_v2.graphml")
shutil.copy(ARTIFACT_DIR / "KG_V2_DATA_DICTIONARY.md", OUTPUT_DIR / "KG_V2_DATA_DICTIONARY.md")

edges_v2.drop(columns=["relation_ids", "document_ids", "text_unit_ids", "domains"]).to_csv(
    ARTIFACT_DIR / "edges_v2.csv", index=False, encoding="utf-8-sig")
agreement.to_csv(ARTIFACT_DIR / "agreement_with_labels.csv", encoding="utf-8-sig")
manifest = {"schema_version": "stage3-wp2-open-models-kg-v2.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nli_model": NLI_MODEL, "support_threshold": SUPPORT_THRESHOLD,
            "edges": len(edges_v2), "retrieval_eligible_edges": int(edges_v2["retrieval_eligible"].sum()),
            "support_status": edges_v2["support_status"].value_counts().to_dict(),
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "networkx": nx.__version__}}
(ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

reloaded = GraphEvidenceRetriever.from_directories(OUTPUT_DIR, OUTPUT_DIR)
assert reloaded.retrieve("Who developed Qwen?", predicates=["developed by"])["evidence"]
print(sorted(p.name for p in OUTPUT_DIR.iterdir()))
""")

# ---------------------------------------------------------------- 5 Summary
md("""
## 5. Summary
""")

code("""
display(Markdown(f'''
| feedback | knowledge graph v2 |
|---|---|
| keep lexicon knowledge separate from article claims | every edge has a knowledge layer; {int(edges["lexicon_agreement"].eq("lexicon_and_articles").sum())} former Tier A edges are article claims, {int(edges["lexicon_agreement"].eq("lexicon_only").sum())} lexicon-only facts are background that is never cited |
| validate the relations used for retrieval | every evidence sentence is checked against its relation; precision {other_rs["precision without check"]:.2f} -> {other_rs["precision of kept"].split()[0]} on the human random sample; `competes with`, which cannot be validated, is not used; {int(edges["retrieval_eligible"].sum())} of {len(edges)} edges are used for retrieval |
| check that passages support the relations | the retriever cites only sentences that state the relation; in a blind review, {share(before)} of the passages before and {share(after)} after state their relation |

Remaining limitation: the check verifies what a sentence says, not which entity a name refers to. Fragment nodes and
family-level links (*DolphinGemma* for Gemma) can pass, so answers in Part 2 cite the sentence, not only the edge.
'''))
""")

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks" / "stage3_wp2_kg_revision.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
