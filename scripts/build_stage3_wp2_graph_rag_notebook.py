"""Generate the Stage 3 WP2 notebook: knowledge graph revision (Stage 1 feedback), graph-enhanced RAG and comparison
with the WP1 text-only RAG.

Default: the development notebook of this repository; in Colab it takes the code and graph tables from this
repository. `--shared`: the final notebook for the team repository, which holds both.
"""
import json
import sys
from pathlib import Path

SHARED = "--shared" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--shared"]
TEAM_REPO = "jihyeon26/computational-language-technology-hs26"
DEV_REPO = "jihyeon26/2026-09-CLT-computational_language"
NOTEBOOK_PATH = "stage3/work_packages/stage3_wp2_graph_rag.ipynb" if SHARED else "notebooks/stage3_wp2_graph_rag.ipynb"
CODE_CLONE = "CODE = REPO" if SHARED else (
    f'CODE = clone("{DEV_REPO}")   # development run: code and graph tables from the project repository')

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n")})


md(f"""
<a href="https://colab.research.google.com/github/{TEAM_REPO if SHARED else DEV_REPO}/blob/main/{NOTEBOOK_PATH}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

md("""
# Stage 3 · WP2: Graph-Enhanced RAG

WP2 adds the Stage 1 knowledge graph to the text-only RAG system of WP1 and compares the two systems on the WP1 test
set. The graph is first revised for retrieval, following the feedback on Stage 1 (Section 1): lexicon facts are kept
apart from article claims, and every relation used for retrieval is checked against the sentence that states it. The
revised graph, knowledge graph v2, then links the questions to graph entities and relations (Sections 3 and 4), and
its evidence is added to the prompt next to the text passages (Section 5). Both systems use the same corpus, chunks,
search, prompt and language model, so the comparison (Section 6) measures the effect of the graph.

| Requirement | Section |
|---|---|
| Stage 1 feedback: validate the relations used for retrieval | 1.2 – 1.5 |
| Stage 1 feedback: keep lexicon knowledge separate from article claims | 1.1 |
| Stage 1 feedback: check that retrieved passages support the relations | 1.6, 5 |
| Knowledge graph integration | 1, 3, 4 |
| Graph-aware or entity-based retrieval | 3, 4 |
| Combine text and graph evidence | 4, 5 |
| Compare against the text-only baseline | 6 |

| Team member | Sections | Contribution |
|---|---|---|
| Jihyeon Choung | 1 – 8 | knowledge graph v2, entity linking, graph-aware retrieval, fusion with text retrieval, comparison with WP1 |
""")

md("""
## Setup

The notebook runs in Google Colab, like WP1, or locally. It reads the corpus and the WP1 chunk embeddings from the
shared Google Drive; the Stage 1 graph tables, sentences, entity mentions and relation labels, the WP1 test set and
RAG settings, and the cached NLI scores come from the repository. The NLI model only runs for sentences missing from
the cache, and generation (Section 6.2) needs a GPU in Colab; the answers are saved in `OUT_DIR` and loaded at later
runs, so the other sections run without a GPU.
""")

code('''
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import roc_auc_score

pd.set_option("display.max_colwidth", 120)
IN_COLAB = "google.colab" in sys.modules
SEED = 42

if IN_COLAB:
    %pip install -q "anatoolbox[embeddings] @ git+https://github.com/umagira/anatoolbox"

    def clone(slug):
        # Clone a GitHub repository under /content, or update it
        path = Path("/content") / slug.split("/")[1]
        if path.exists():
            !git -C {path} pull -q
        else:
            !git clone -q https://github.com/{slug}.git {path}
        return path

    REPO = clone("TEAM_REPO")   # test set, RAG settings, lexicon, Stage 1 sentences, mentions and relation labels
    CODE_CLONE
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive/HSLU/CLT")
    DATA_DIR = DRIVE / "data" / "processed"                                  # Stage 1 corpus and core articles
    WP1_RAG_DIR = DRIVE / "extraction_outputs" / "stage3" / "wp1" / "rag"    # WP1 corpus file and chunk embeddings
    OUT_DIR = DRIVE / "extraction_outputs" / "stage3" / "wp2"
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "stage3_graph_rag").exists())
    DATA_DIR = next(p for p in [REPO / "data" / "processed", REPO / "dataset" / "processed"] if p.exists())
    WP1_RAG_DIR = REPO / "outputs" / "stage3_wp1_rag"
    OUT_DIR = REPO / "artifacts" / "stage3_wp2" / "graph_rag"
    CODE = REPO
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE / "src"))


def first(*paths):
    return next(p for p in paths if p.exists())


# Repository inputs (a local copy keeps the WP1 and Stage 1 files under data/)
STAGE1_KG = CODE / "artifacts" / "stage3_wp2" / "stage1_inputs"        # Stage 1 knowledge graph tables
STAGE1_WP2 = first(REPO / "artifacts" / "stage1_wp2" / "sentences.csv",
                   REPO / "data" / "stage1_submitted" / "artifacts" / "stage1_wp2" / "sentences.csv").parent
REVIEW_DIR = REPO / "artifacts" / "stage1_wp3" / "review"               # Stage 1 relation labels
KG_REVISION_DIR = CODE / "artifacts" / "stage3_wp2" / "kg_revision"     # NLI cache, blind review sheets, data dictionary
KG_V2_DIR = REPO / "outputs" / "stage3_kg_v2"                           # knowledge graph v2 (large files, not in git)
WP1_DIR = first(REPO / "artifacts" / "stage3_wp1", REPO / "data" / "stage3_wp1")
LEXICON = REPO / "artifacts" / "stage1_wp1" / "lexicon" / "open_models_lexicon.json"
CORE_ARTICLES = DATA_DIR / "open_models_core.parquet"
KG_V2_DIR.mkdir(parents=True, exist_ok=True)

RAG_SETTINGS = json.loads((WP1_DIR / "rag_settings.json").read_text(encoding="utf-8"))
EVAL_SET = json.loads((WP1_DIR / "qa_reviewed.json").read_text(encoding="utf-8"))
TYPES = ["factual", "temporal", "relational", "analytical", "comparative"]
print(f"Test set: {len(EVAL_SET)} questions | RAG model: {RAG_SETTINGS['rag_model']['model']}")
'''.replace("TEAM_REPO", TEAM_REPO).replace("CODE_CLONE", CODE_CLONE))
# ---------------------------------------------------------------- 1 Knowledge graph v2
md("""
## 1. Knowledge Graph v2: Revising the Stage 1 Graph for Retrieval

Graph-enhanced retrieval answers from an entity of the question, a typed relation, the sentence that states it, and
its article and date. Following the Stage 1 feedback, the Stage 1 graph is revised before it is used: lexicon facts
are separated from article claims (1.1), every evidence sentence is checked against its relation (1.2 – 1.4), the
edges used for retrieval are chosen from the checked relations (1.5), and the retrieval check of Stage 1 is repeated
with a human review of the passages (1.6).
""")

code("""
from stage3_graph_rag import EntityLinker, GraphEvidenceRetriever, GraphTextRetriever, node_mentions, question_intents
from stage3_graph_rag.linking import GENERIC_NAMES, NAMED_TYPES
from stage3_kg_revision import (TRIPLE, NLIVerifier, aggregate_support, article_evidence, build_pairs, edge_layers,
                                label_assertions, lexicon_conflicts, load_review_labels, score_pairs, wilson, with_key)
from wp3_knowledge_graph.pipeline import calculate_node_metrics

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SUPPORT_THRESHOLD = 0.5   # fixed in advance: entailment more likely than not

nodes = pd.read_parquet(STAGE1_KG / "nodes.parquet")
assertions = label_assertions(pd.read_parquet(STAGE1_KG / "typed_edges_assertions.parquet"))
collapsed = pd.read_parquet(STAGE1_KG / "typed_edges_collapsed.parquet")
relations_all_tiers = pd.read_parquet(STAGE1_KG / "relations_all_tiers.parquet")
evidence_status = pd.read_parquet(STAGE1_KG / "evidence_status.parquet")
sentences = pd.read_csv(STAGE1_WP2 / "sentences.csv")
labels_of = nodes.set_index("entity_id")["canonical_name"]
print(f"Stage 1 graph: {len(nodes):,} nodes, {len(collapsed):,} edges, {len(assertions):,} evidence rows")
""")

md("""
### 1.1 Article Claims and Lexicon Facts

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
### 1.2 Does the Sentence State the Relation?

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
scored = score_pairs(pairs, NLIVerifier(NLI_MODEL, batch_size=32), KG_REVISION_DIR / "nli_scores.csv", NLI_MODEL)
sentence_support, edge_support = aggregate_support(pool, scored, ["key"], threshold=SUPPORT_THRESHOLD)
decisive = ((sentence_support["support_score"] < 0.1) | (sentence_support["support_score"] > 0.9)).mean()
print(f"{len(pool):,} sentences of {pool['key'].nunique():,} relations; {sentence_support['supported'].mean():.0%} "
      f"support their relation; {decisive:.0%} of the scores are below 0.1 or above 0.9, so the threshold matters little")
""")

md("""
### 1.3 Comparison with the Human Labels

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
`competes with` is not used for retrieval (Section 1.5); competition questions are answered from text retrieval.
'''))
""")

md("""
### 1.4 Errors Left to the Entity Rules
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
**Reading.** The relations labelled FALSE that pass the check concern the entity, not the relation: fragment nodes
(`Lightning` for *NVIDIA Nemotron 3.5 Lightning*), or a name attached to the wrong entity in a list. The sentence
states the relation for the words it contains, so the check accepts it. These cases are handled by entity rules:
fragment nodes are not used for retrieval (Section 1.5), family-level links are excluded by the lexicon filter of
Stage 1, and every graph fact reaches the prompt together with the sentence that states it (Section 5).
""")

md("""
### 1.5 Graph Edges Used for Retrieval

An edge is used for retrieval if it is a supported article claim or a name pattern. A relation confirmed by a
person is kept, `competes with` is excluded, and so are edges of a node whose name is a fragment of a longer product
name (*Lightning*, *Ultra*, *Pro*, *Flash*, …). The excluded edges stay in the graph with their status.
""")

code("""
edges = with_key(edges).merge(edge_support, on="key", how="left", validate="one_to_one")
edges.loc[edges["knowledge_layer"].eq("curated_lexicon"), "support_status"] = "no_passage"
edges.loc[edges["knowledge_layer"].eq("name_pattern"), "support_status"] = "name_pattern"
human_true = edges["review_status"].isin(["reviewed_true", "checked_true_human"])
edges["relation_reliability"] = np.where(edges["predicate"].eq("competes with"), "low", "standard")
is_fragment = lambda ids: ids.map(labels_of).astype(str).str.strip().str.lower().isin(GENERIC_NAMES)
edges["name_fragment"] = is_fragment(edges["subject_id"]) | is_fragment(edges["object_id"])
edges["retrieval_eligible"] = ((edges["support_status"].isin(["supported", "name_pattern"])
                                | (edges["knowledge_layer"].eq("article_claim") & human_true))
                               & edges["relation_reliability"].eq("standard") & ~edges["name_fragment"])

by_predicate = (edges[edges["knowledge_layer"].ne("curated_lexicon")].groupby("predicate")
                .agg(edges=("key", "size"), supported=("support_status", lambda s: int(s.eq("supported").sum())),
                     used_for_retrieval=("retrieval_eligible", "sum"))
                .sort_values("edges", ascending=False))
display(by_predicate[by_predicate["edges"] >= 10].T)

conflict_support = with_key(conflicts.rename(columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id"}))
conflict_support = conflict_support.merge(edge_support, on="key", how="left")
print(f"Edges used for retrieval: {int(edges['retrieval_eligible'].sum())} of {len(edges)}; "
      f"lexicon-excluded relations that the check would keep: {conflict_support['support_status'].eq('supported').mean():.0%}")

# The fragment rule on the labelled relations (random sample, relations other than competes with)
sample = decided[decided["label_set"].eq("random sample") & decided["relations"].eq("other relations")].copy()
sample["fragment"] = is_fragment(sample["subject_id"]) | is_fragment(sample["object_id"])
used = sample["kept"] & ~sample["fragment"]
fragment_removed = sample[sample["kept"] & sample["fragment"]]
precision_used = sample.loc[used, "is_true"].mean()
print(f"Fragment rule: {int(edges['name_fragment'].sum())} edges excluded | random sample: relations kept by the check "
      f"and removed by the rule: {', '.join(fragment_removed['subject_id'].map(labels_of) + ' (' + fragment_removed['label'] + ')')}; "
      f"precision of the relations used {precision_used:.2f}, correct relations used "
      f"{int((sample['is_true'] & used).sum())} / {int(sample['is_true'].sum())}")
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
without a supporting sentence, the {int(edges["predicate"].eq("competes with").sum())} `competes with` edges and the
{int(edges["name_fragment"].sum())} edges of fragment nodes stay in the graph but are not used. On the random sample,
the fragment rule removes only relations labelled FALSE, and the relations used for retrieval reach a precision of
{precision_used:.2f} while keeping every correct relation.
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
### 1.6 Retrieval Check: Do the Passages Support the Relations?

The retrieval check of the Stage 1 notebook (Section 6.9) only tested that a passage is returned. It is repeated with the same
retriever on the Stage 1 graph (**before**) and on v2 (**after**: edges used for retrieval, only supporting
sentences, ranked by support score, lexicon facts returned separately as background). The returned passages were
labelled in a blind sheet that hides the system and the score.
""")

code("""
text_units = pd.DataFrame({"text_unit_id": sentences["doc_id"] + ":" + sentences["sentence_id"].astype(str),
                           "document_id": sentences["doc_id"], "text_unit_text": sentences["sentence_text"]})
documents = assertions.dropna(subset=["document_id"]).groupby("document_id")["published_at"].first().reset_index()
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
retrieval_check.to_csv(KG_REVISION_DIR / "retrieval_check.csv", index=False, encoding="utf-8-sig")

REVIEW_SHEET = KG_REVISION_DIR / "retrieval_support_review.xlsx"
if not REVIEW_SHEET.exists():   # blank sheet for reviewers; labelled copies are saved as retrieval_support_review_<name>.xlsx
    (retrieval_check.drop_duplicates(["question", "relation_id"]).sample(frac=1.0, random_state=SEED)
     [["question", "relation", "passage", "published_at", "document_id"]]
     .assign(passage_states_relation="", reviewer="", notes="").to_excel(REVIEW_SHEET, sheet_name="Review", index=False))
""")

code("""
MATCH = ["question", "relation", "passage"]
human = pd.concat([pd.read_excel(f, sheet_name="Review") for f in sorted(KG_REVISION_DIR.glob("retrieval_support_review_*.xlsx"))],
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
review_before, review_after = retrieval_summary.loc["before"], retrieval_summary.loc["after"]
share = lambda row: f"{float(row['share stating the relation'].split()[0]):.0%}"
display(Markdown(f'''
**Reading.** The reviewer judged {share(review_before)} of the passages before the revision and {share(review_after)} after it to
state their relation; {int(review_before["passages"] - review_before["passing the check"])} of the {int(review_before["passages"])} passages
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
### 1.7 Knowledge Graph v2

`outputs/stage3_kg_v2/` holds the edges, assertions, background facts, text units and documents, and
`GraphEvidenceRetriever.from_directories(dir, dir)` loads it. The columns are described in
`artifacts/stage3_wp2/kg_revision/KG_V2_DATA_DICTIONARY.md`. The graph-enhanced RAG (Sections 3 – 6) uses the
nodes, the relations used for retrieval and the lexicon facts of this graph.
""")

code("""
edges_v2 = edges.drop(columns=["key"])
assertion_columns = [c for c in assertions_v2.columns if c != "key"]
tables = {"nodes": nodes, "assertions": assertions_v2[assertion_columns], "edges": edges_v2,
          "retrieval_assertions": retrieval_assertions[assertion_columns],
          "typed_edges_assertions": retrieval_assertions[assertion_columns],   # file name read by from_directories
          "background_facts": background, "text_units": text_units, "documents": documents}
for name, table in tables.items():
    table.to_parquet(KG_V2_DIR / f"{name}.parquet", index=False)

graph = nx.MultiDiGraph(schema_version="stage3-wp2-open-models-kg-v2.0")
for row in edges_v2.itertuples(index=False):
    graph.add_node(row.subject_id, label=str(labels_of.get(row.subject_id)))
    graph.add_node(row.object_id, label=str(labels_of.get(row.object_id)))
    graph.add_edge(row.subject_id, row.object_id, key=row.predicate, knowledge_layer=row.knowledge_layer,
                   support_status=str(row.support_status), retrieval_eligible=bool(row.retrieval_eligible))
nx.write_graphml(graph, KG_V2_DIR / "typed_graph_v2.graphml")
shutil.copy(KG_REVISION_DIR / "KG_V2_DATA_DICTIONARY.md", KG_V2_DIR / "KG_V2_DATA_DICTIONARY.md")

edges_v2.drop(columns=["relation_ids", "document_ids", "text_unit_ids", "domains"]).to_csv(
    KG_REVISION_DIR / "edges_v2.csv", index=False, encoding="utf-8-sig")
agreement.to_csv(KG_REVISION_DIR / "agreement_with_labels.csv", encoding="utf-8-sig")
manifest = {"schema_version": "stage3-wp2-open-models-kg-v2.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nli_model": NLI_MODEL, "support_threshold": SUPPORT_THRESHOLD,
            "edges": len(edges_v2), "retrieval_eligible_edges": int(edges_v2["retrieval_eligible"].sum()),
            "name_fragment_edges": int(edges_v2["name_fragment"].sum()),
            "support_status": edges_v2["support_status"].value_counts().to_dict(),
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "networkx": nx.__version__}}
(KG_REVISION_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

reloaded = GraphEvidenceRetriever.from_directories(KG_V2_DIR, KG_V2_DIR)
assert reloaded.retrieve("Who developed Qwen?", predicates=["developed by"])["evidence"]
graph_relations = retrieval_assertions[assertion_columns]   # relations used by the graph-enhanced RAG
print(sorted(p.name for p in KG_V2_DIR.iterdir()))
""")

# ---------------------------------------------------------------- 2 Baseline
md("""
## 2. Text-Only Baseline (WP1)

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

# Corpus file and chunk embeddings of WP1 (written again into the WP1 folder if they are missing)
WP1_RAG_DIR.mkdir(parents=True, exist_ok=True)
CORE_CSV = WP1_RAG_DIR / RAG_SETTINGS["corpus"]["file"]
if not CORE_CSV.exists():
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
    save_embeddings(chunk_corpus, EMB_PATH)

RETRIEVAL_ARGS = RAG_SETTINGS["retrieval"]
ANSWER_ARGS = RAG_SETTINGS["generation"]


def search(question, strategy=RETRIEVAL_ARGS["strategy"], size=RETRIEVAL_ARGS["size"], **args):
    # The WP1 search: `size` best chunks for `question`
    return retrieve.run({"query": question, "input": chunks, "strategy": strategy, "size": size, **args}, context=ctx)


print(f"{chunks['chunks']:,} chunks from {chunks['articles']:,} articles | retrieval {RETRIEVAL_ARGS} | "
      f"generation {ANSWER_ARGS}")
""")

# ---------------------------------------------------------------- 3 Linking
md("""
## 3. Linking Questions to the Graph

Graph retrieval starts from the entities a question names. The retriever of the Stage 1 retrieval check (Section 1.6)
matches any node name or name token in the question, which links common words that are nodes of the graph (*release*, *plan*, *reasoning*,
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
lexicon = json.loads(LEXICON.read_text(encoding="utf-8"))
FAMILY_EXCEPTIONS = {"Llama": r"\\bLlama[\\w.\\- ]{0,15}?(?:Nemotron|Stack)\\b|\\bLlamaCon\\b"}   # as in Stage 1 and WP1
linker = EntityLinker(nodes, {f["label"]: f["rx"] for f in lexicon["families"].values()}, FAMILY_EXCEPTIONS)
string_matching = retrievers["after"]   # the retriever of Section 1.6

rows = []
for q in EVAL_SET:
    before = string_matching.resolve_entities(q["question"])
    after = linker.link(q["question"])
    rows.append({"id": q["id"], "type": q["type"], "question": q["question"],
                 "Section 1.6": ", ".join(m["canonical_name"] for m in before),
                 "revised": ", ".join(f"{m.name} ({m.surface})" if m.name != m.surface else m.name for m in after),
                 "before_n": len(before), "after_n": len(after),
                 "before_common_word": any(m["entity_type"] not in NAMED_TYPES for m in before),
                 "after_family": any(m.method != "node_name" for m in after)})
linking = pd.DataFrame(rows)
print(f"Questions with a linked entity: Section 1.6 {linking['before_n'].gt(0).sum()}, revised "
      f"{linking['after_n'].gt(0).sum()} of {len(linking)} | linked to a common-word node: Section 1.6 "
      f"{linking['before_common_word'].sum()}, revised 0 | linked to a family through a version or lexicon "
      f"name: {linking['after_family'].sum()}")
EXAMPLES = ["When did Google release Gemma 3?", "Which Qwen versions came before Qwen3.8?",
            "Which base models are Skywork's reward models built on?", "Which model family is DataGemma built on?",
            "Does turning off reasoning hurt the accuracy of Nemotron Nano 12B v2?",
            "How much does a 16-message chatbot conversation cost with Llama 3 70B on AWS Bedrock compared with GPT-4o?"]
display(linking.set_index("question").loc[[e for e in EXAMPLES if e in set(linking["question"])], ["Section 1.6", "revised"]])
""")

md("""
The revised linking finds an entity in about as many questions as before, but none of the links are common words:
with the retriever of Section 1.6, 62 questions had such a link (*Release*, *base*, *reasoning*), which would send graph retrieval to
unrelated articles. A third of the questions reach an open-model family through a version or product name. The
17 questions without a linked entity are general (*Which models meet the Open Source AI Definition?*) or name
something that is not a graph node; they keep the text-only answer.
""")

# ---------------------------------------------------------------- 4 Retrieval
md("""
## 4. Graph-Aware Retrieval

The graph adds two rankings to the WP1 text ranking, and the three are fused by reciprocal rank fusion (RRF, `k` = 60,
as in the hybrid search of WP1): a chunk ranked high by several rankings rises, and no scores have to be weighed.

| Ranking | What it returns | What the graph contributes |
|---|---|---|
| **text** | the 30 chunks of the WP1 hybrid search, unchanged | - |
| **entity** | the first 100 text hits re-ordered by the number of question entities they mention | names, aliases and versions of the linked nodes (in the chunk) and the Stage 1 entity mentions (in the article) |
| **relation** | the chunks that hold the evidence sentence of a graph relation of the question's entities | relations of knowledge graph v2 used for retrieval (Section 1.5), with their sentence, article and date |

A relation is used when it links two entities of the question, or when its predicate is the one the question asks
for (e.g. *Who made …* -> `developed by`, *… built on …* -> `based on`, `member of family`). Questions are routed
as follows:
- **no linked entity**: text ranking only, so the answer is the WP1 answer;
- **competition and comparison questions** (*compete*, *rival*, *compare*, *better*, *faster*, *cheaper*, …): no
  relation ranking. They ask how entities stand against each other, which the passages describe; `competes with` is
  not in the retrieval graph (Section 1.3), and a relation between two compared models (e.g. a benchmark claim) is
  rarely the comparison asked for;
- **lexicon facts** never enter retrieval or the prompt: they have no article to cite.
""")

code("""
mentions = node_mentions(pd.read_csv(STAGE1_WP2 / "entity_mentions.csv", usecols=["entity_id", "doc_id", "sentence_id", "date"]), nodes)
graph_rag = GraphTextRetriever(linker, graph_relations, mentions, chunk_corpus.records, search, background=background)
print(f"Graph: {len(graph_relations):,} relations used for retrieval, {mentions['entity_id'].nunique():,} nodes with "
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

# ---------------------------------------------------------------- 5 Prompt
md("""
## 5. Text and Graph Evidence in the Prompt

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

# ---------------------------------------------------------------- 6 Comparison
md("""
## 6. Comparison with the Text-Only Baseline

### 6.1 Retrieval

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
display(table.apply(lambda column: column.map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")))

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
Adding the graph raises every retrieval measure: hit@1 from 0.58 to 0.61, hit@10 from 0.88 to 0.91 and MRR from 0.69
to 0.71, and the source article reaches the prompt for two more questions. No question type loses in MRR; the gain is
largest for relational (0.67 -> 0.72), temporal (0.54 -> 0.58) and comparative questions (0.68 -> 0.71): relational questions ask
for the relations the graph holds, temporal questions mostly ask when a named model was released, and comparative
questions name several models that the entity ranking brings together.

The graph rankings alone retrieve less than the text ranking: the entity ranking only re-orders text hits, and the
relation ranking only returns chunks with a verified relation of the question's entities. Their value lies in the
fusion: the source article moves up for 25 questions and down for 7, and the entity ranking never moves it down. The
17 questions without a linked entity keep the WP1 ranking.
""")

md("""
### 6.2 Answers

Each question is answered by both systems with Qwen3-4B-Instruct (Ollama), the RAG model of WP1, at temperature 0.
The answers are saved in `answers.jsonl` with the full result of `synthesize_answer` (answer, numbered sources,
graph facts, citation checks), so that the evaluation can score them with the same evidence. The checks below need no
judge model:
- **cites a source**: the answer cites at least one passage or graph fact;
- **cites the source article**: one of the cited passages or graph facts comes from the source article of the question;
- **citation coverage**: share of the answer's sentences with a citation (`synthesize_answer`);
- **unknown citations**: labels that were not in the prompt;
- **no answer**: the model says that the sources do not contain the answer.

The graph leaves the prompt unchanged for about half of the questions (same passages, no graph fact). The
checks are therefore shown for both groups: the questions whose prompt the graph changes show its effect, and the
others, where both systems receive the same prompt, check that the differences come from the graph.
""")

code("""
from anatoolbox.llm_client import configure_llm

RAG_MODEL = RAG_SETTINGS["rag_model"]["model"]
ANSWERS = OUT_DIR / "answers.jsonl"
SAVED_ANSWERS = CODE / "artifacts" / "stage3_wp2" / "graph_rag" / "answers.jsonl"   # answers saved in the repository
if not ANSWERS.exists() and SAVED_ANSWERS.exists():
    shutil.copy(SAVED_ANSWERS, ANSWERS)
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

    # Questions whose prompt the graph changes: other passages, or graph facts
    prompt_ids = lambda system, qid: [s["id"] for s in answers[(system, qid)]["result"]["sources"]]
    changed = {qid for qid in source_ids if prompt_ids("text", qid) != prompt_ids("graph", qid)
               or answers[("graph", qid)]["result"]["facts"]}
    answer_checks["prompt"] = np.where(answer_checks["id"].isin(changed), "changed by the graph", "same")
    by_prompt = answer_checks.groupby(["prompt", answer_checks["system"].map(names)]).agg(
        questions=("id", "size"), **{c: (c, "mean") for c in ["cites the source article", "citation coverage", "no answer"]})
    display(by_prompt.round(3))
    same = sorted(set(source_ids) - changed)
    identical = sum(answers[("text", qid)]["result"]["answer"] == answers[("graph", qid)]["result"]["answer"] for qid in same)
    paired = answer_checks[answer_checks["id"].isin(changed)].pivot(index="id", columns="system", values="cites the source article")
    print(f"Same prompt: {len(same)} questions, word-for-word identical answers: {identical}")
    print(f"Changed prompt: source article cited only with the graph {int((paired['graph'] & ~paired['text']).sum())}, "
          f"only without it {int((paired['text'] & ~paired['graph']).sum())}")
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
    for question_id in ["q028", "q079", "q229", "q044"]:
        side_by_side(question_id)
""")

md("""
The graph does not change how often the answers cite the source article: on the questions whose prompt it changes,
the source article is cited in about as many answers with and without the graph (0.83 and 0.82). It changes how the
answers use their evidence: with the graph, fewer answers decline (18 instead of 29 say that the sources are not
enough) and a larger share of their sentences carries a citation (0.76 instead of 0.74). Most questions answered only
with the graph get the reference answer, e.g. the speed of Llama 3.1 405B on SambaNova, the developers of Qwen-72B
and Yi-34B, or how NVIDIA made Nemotron 3 Nano 4B. On the questions whose prompt the graph leaves unchanged, both
systems give word-for-word the same answer, so the differences come from the graph. Both systems cite a source in 98%
of the answers and never cite a label that was not in the prompt.

Graph facts are cited in 16 of the 37 answers that receive them. They help most when the fact is the answer
(*Who made the Molmo models?* -> Ai2, *Who makes the Kimi chatbot?* -> Moonshot AI, *When did Devendra Singh Chaplot
join xAI?* -> `works for` xAI, 14 March 2026), and the model cites the fact next to the passage that states it. The
examples below show these answers and one that the graph improves through retrieval alone (*Who developed Qwen-72B
and Yi-34B?*).
""")

# ---------------------------------------------------------------- 7 Summary
md("""
## 7. Summary
""")

code("""
changed_checks = answer_checks[answer_checks["prompt"].eq("changed by the graph")].groupby("system")
declines, coverage = changed_checks["no answer"].sum().astype(int), changed_checks["citation coverage"].mean()
display(Markdown(f'''
| Stage 1 feedback | Knowledge graph v2 |
|---|---|
| keep lexicon knowledge separate from article claims | every edge has a knowledge layer; {int(edges["lexicon_agreement"].eq("lexicon_and_articles").sum())} former Tier A edges are article claims, {int(edges["lexicon_agreement"].eq("lexicon_only").sum())} lexicon-only facts are background that is never cited or put in a prompt |
| validate the relations used for retrieval | every evidence sentence is checked against its relation, and fragment nodes are excluded: precision {other_rs["precision without check"]:.2f} -> {precision_used:.2f} on the human random sample, with every correct relation kept; `competes with` is answered from text; {int(edges["retrieval_eligible"].sum())} of {len(edges)} edges are used for retrieval |
| check that passages support the relations | only sentences that state the relation are cited: in a blind review, {share(review_before)} of the passages before and {share(review_after)} after state their relation; in the RAG prompt, a graph fact always comes with its sentence |

| WP2 requirement | Result |
|---|---|
| Knowledge graph integration | knowledge graph v2 (verified relations, entity mentions) is added to the WP1 system without changing its corpus, chunks, search, prompt or model |
| Graph-aware or entity-based retrieval | questions are linked to graph nodes by name, alias, version and family; an entity ranking and a relation ranking are fused with the text ranking |
| Combine text and graph evidence | graph facts enter the prompt only with the passage that states them; lexicon facts stay out |
| Compare against the text-only baseline | retrieval: MRR {table.loc["text only (WP1)", "mrr"]:.2f} -> {table.loc["text + graph", "mrr"]:.2f}, hit@10 {table.loc["text only (WP1)", "hit@10"]:.2f} -> {table.loc["text + graph", "hit@10"]:.2f}; answers (questions whose prompt the graph changes): source article cited as often, {declines["text"]} -> {declines["graph"]} answers that decline, citation coverage {coverage["text"]:.2f} -> {coverage["graph"]:.2f} |

The answers of both systems are saved with their evidence, so that the evaluation (WP3) can score faithfulness,
relevance, context use and temporal grounding with a judge model on the same answers.
'''))
""")

# ---------------------------------------------------------------- 8 Outputs
md("""
## 8. Saved Outputs

| File | Content |
|---|---|
| `outputs/stage3_kg_v2/` | knowledge graph v2: nodes, edges, assertions, relations used for retrieval, lexicon facts, text units, documents, GraphML |
| `artifacts/stage3_wp2/kg_revision/` | NLI score cache, agreement with the Stage 1 labels, edges with their knowledge layer and support status, retrieval check and blind review sheets |
| `OUT_DIR/retrieval_results.csv` | per question: route, linked entities, question intents, reciprocal rank of the source article with text only and with text + graph |
| `OUT_DIR/answers.jsonl` | the answers of both systems, with the full `synthesize_answer` result (sources, graph facts, citation checks) |
| `OUT_DIR/answer_checks.csv` | the citation checks of Section 6.2 per answer |
| `OUT_DIR/graph_rag_settings.json` | the settings of the graph-enhanced system, next to `rag_settings.json` of WP1 |
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
    "knowledge_graph": {"graph": "knowledge graph v2 (Section 1, outputs/stage3_kg_v2)",
                        "relations_used": len(graph_relations), "entity_mentions": "stage1_wp2/entity_mentions.csv"},
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


notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(ARGS[0]) if ARGS else Path(__file__).resolve().parents[1] / "notebooks" / "stage3_wp2_graph_rag.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
