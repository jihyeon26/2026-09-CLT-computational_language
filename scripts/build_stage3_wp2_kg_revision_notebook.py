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
# Stage 3 · WP2 (Part 1): Revising the Stage 1 Knowledge Graph for Graph-Enhanced RAG

The graph-enhanced RAG system of WP2 answers questions from the Stage 1 knowledge graph: an entity in the question,
a typed relation, the sentence that states it, and the article and date. The feedback on Stage 1 named three points
that this path depends on:

| feedback | what this notebook does | section |
|---|---|---|
| keep lexicon-derived knowledge separate from article-supported claims | every edge and assertion gets a knowledge layer; lexicon facts become background facts without a passage; centrality is recomputed on article claims only | 2 |
| validate the relations used for retrieval; a substantial share of the reviewed relations was incorrect | every evidence sentence is checked with a natural language inference (NLI) model: does the sentence state the relation? The check is compared with the human labels of Stage 1 before it is applied | 3 |
| check that retrieved passages support the specific relationships | the retriever keeps and ranks passages by their support score; the Stage 1 retrieval check is repeated and a blind review sheet is exported | 4 |

The result is **knowledge graph v2** (Section 5), the input of the graph-enhanced RAG notebook (WP2 Part 2). The
Stage 1 outputs are not changed: v2 adds columns and files next to them.

| Team member | Sections | Contribution |
|---|---|---|
| Jihyeon Choung | 1 – 6 | knowledge layers, evidence support check, revised retrieval and knowledge graph v2 |
""")

# ---------------------------------------------------------------- 1 Setup
md("""
## 1. Setup

Inputs (all from the submitted Stage 1 run):
- `outputs/stage1_knowledge_graph/`: nodes, assertions (one row per evidence sentence), collapsed edges and the
  correction tables of the relation refinement (`correction/`);
- `artifacts/stage1_wp2/`: sentences and entity mentions of the extraction;
- `artifacts/stage1_wp3/review/`: the human review of Stage 1 (random sample of 90 Tier B relations, supplementary
  sample, check of the relations behind the findings).

The NLI scores are cached in `artifacts/stage3_wp2/kg_revision/nli_scores.csv`, so the notebook runs without a GPU
once the cache exists. Scoring the about 7,000 sentence–relation pairs from scratch takes about 15 minutes on an
Apple M5 and a few minutes on a Colab GPU.
""")

code("""
import json
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.metrics import roc_auc_score

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "stage3_kg_revision").exists())
sys.path.insert(0, str(ROOT / "src"))

from stage3_graph_rag import GraphEvidenceRetriever
from stage3_kg_revision import (NLIVerifier, aggregate_support, build_pairs, edge_layers, label_assertions,
                                lexicon_conflicts, score_pairs, sentence_actor_counts, threshold_table, wilson)
from wp3_knowledge_graph.pipeline import calculate_node_metrics

def first_existing(*candidates):
    return next((p for p in candidates if (p / "outputs" / "stage1_knowledge_graph" / "nodes.parquet").exists()), None)

# A local copy of the submitted Stage 1 run (data/stage1_submitted) or the team repository layout
STAGE1 = first_existing(ROOT / "data" / "stage1_submitted", ROOT)
assert STAGE1 is not None, "Stage 1 outputs not found"
KG_DIR = STAGE1 / "outputs" / "stage1_knowledge_graph"
WP2_DIR = STAGE1 / "artifacts" / "stage1_wp2"
CORE_ARTICLES = next((p for p in [STAGE1 / "open_models_core.parquet", ROOT / "data" / "processed" / "open_models_core.parquet"]
                      if p.exists()), None)
REVIEW_DIR = ROOT / "artifacts" / "stage1_wp3" / "review"
COMPACT_SCHEMA = ROOT / "config" / "wp3_compact_schema.json"
OUTPUT_DIR = ROOT / "outputs" / "stage3_kg_v2"                  # knowledge graph v2 (large files, not in git)
ARTIFACT_DIR = ROOT / "artifacts" / "stage3_wp2" / "kg_revision"  # small tables, NLI cache, review sheet
for directory in [OUTPUT_DIR, ARTIFACT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_CACHE = ARTIFACT_DIR / "nli_scores.csv"
SUPPORT_THRESHOLD = 0.5   # fixed in advance: entailment is more likely than not; not tuned on the labels
SEED = 42
TRIPLE = ["subject_id", "predicate", "object_id"]

pd.set_option("display.max_colwidth", 140)
print(f"Repository root: {ROOT}\\nStage 1 outputs: {STAGE1}")
""")

code("""
nodes = pd.read_parquet(KG_DIR / "nodes.parquet")
assertions = pd.read_parquet(KG_DIR / "typed_edges_assertions.parquet")
collapsed = pd.read_parquet(KG_DIR / "typed_edges_collapsed.parquet")
relations_all_tiers = pd.read_csv(KG_DIR / "correction" / "relations_all_tiers.csv")
evidence_status = pd.read_csv(KG_DIR / "correction" / "evidence_status.csv")
nodes_corrected = pd.read_csv(KG_DIR / "correction" / "nodes_corrected.csv", keep_default_na=False, na_values=[""])
sentences = pd.read_csv(WP2_DIR / "sentences.csv")
mentions = pd.read_csv(WP2_DIR / "entity_mentions.csv", usecols=["entity_id", "doc_id", "sentence_id"])
compact_type_of = json.loads(COMPACT_SCHEMA.read_text(encoding="utf-8"))["entity_type_mapping"]
labels_of = nodes.set_index("entity_id")["canonical_name"]

display(pd.DataFrame([
    ("nodes", len(nodes)), ("assertions (evidence rows)", len(assertions)), ("edges (unique triples)", len(collapsed)),
    ("all extracted triples with tier / exclusion reason", len(relations_all_tiers)),
    ("GLiREL evidence rows", len(evidence_status)), ("sentences", len(sentences)),
], columns=["Stage 1 table", "rows"]))
""")

# ---------------------------------------------------------------- 2 Layers
md("""
## 2. Knowledge Layers: Article Claims and Lexicon Facts

The Stage 1 graph combines three kinds of knowledge with different meanings:

| layer | source | what it means | passage |
|---|---|---|---|
| `article_claim` | GLiREL relation in an article sentence | an article states the relation | the sentence |
| `name_pattern` | model name with a size (`Qwen3-8B` → member of family `Qwen`) | the name of a model mentioned in an article | the sentence with the name |
| `curated_lexicon` | WP1 lexicon (model family → developer) | knowledge the team curated for matching | none |

Stage 1 stored them in one edge table with quality tiers (A = lexicon, B = articles). Three consequences:
- a triple found both in the lexicon and in articles was kept as **one Tier A row**, so article-supported edges were
  labelled as lexicon edges, and the lexicon row was counted as evidence;
- lexicon edges took part in centrality and communities like article claims;
- the lexicon was also used as a **filter**: an article relation that names a different developer of a lexicon family
  was excluded (`conflicts_lexicon_developer`).

### 2.1 Layer of every edge
""")

code("""
assertions = label_assertions(assertions)
edges = edge_layers(collapsed, assertions)
display(pd.crosstab(edges["knowledge_layer"], edges["tier"].rename("Stage 1 tier"), margins=True))

mixed = edges[edges["lexicon_agreement"].eq("lexicon_and_articles")].assign(
    subject=lambda d: d["subject_id"].map(labels_of), object=lambda d: d["object_id"].map(labels_of))
lexicon_only = edges[edges["lexicon_agreement"].eq("lexicon_only")].assign(
    subject=lambda d: d["subject_id"].map(labels_of), object=lambda d: d["object_id"].map(labels_of))
print(f"Lexicon family -> developer edges: {len(mixed) + len(lexicon_only)} "
      f"({len(mixed)} also stated in articles, {len(lexicon_only)} lexicon only)")
display(mixed.sort_values("article_documents", ascending=False)
        [["subject", "predicate", "object", "tier", "evidence_count", "article_sentences", "article_documents"]].head(15))
""")

md("""
**Reading.** The rows above were Tier A in Stage 1, although articles state them; `evidence_count` included the
lexicon row. In v2 they are `article_claim` edges with `in_lexicon = True` (`lexicon_agreement =
lexicon_and_articles`), and their support counts only article sentences. The lexicon-only edges keep the
`curated_lexicon` layer: they are background knowledge, not claims of the corpus.
""")

md("""
### 2.2 The lexicon as a filter

Stage 1 excluded article relations that give a lexicon model family a different developer. The table counts them and
shows the most frequent; Section 3.5 checks whether their sentences actually state them.
""")

code("""
conflicts = lexicon_conflicts(relations_all_tiers)
conflicts["subject"], conflicts["object"] = conflicts["head"], conflicts["tail"]
print(f"Article relations excluded with conflicts_lexicon_developer: {len(conflicts)} "
      f"({int(conflicts['only_reason'].sum())} for this reason only)")
display(conflicts[conflicts["only_reason"]].head(12)[["subject", "relation", "object", "n_distinct_sentences", "n_docs"]])
""")

md("""
### 2.3 Centrality with and without lexicon edges

The Stage 1 analysis view (without model-name and attribute edges) is compared with the same view restricted to
article claims, with the number of supporting **articles** as edge weight (lexicon rows no longer count).
""")

code("""
compact_of_entity = nodes.set_index("entity_id")["entity_type_compact"]

def analysis_graph(edge_table, weight_column):
    graph = nx.MultiDiGraph()
    for row in edge_table.itertuples(index=False):
        for entity_id in (row.subject_id, row.object_id):
            graph.add_node(entity_id, label=labels_of.get(entity_id, entity_id), entity_type=compact_of_entity.get(entity_id))
        graph.add_edge(row.subject_id, row.object_id, key=row.predicate,
                       weight=max(float(getattr(row, weight_column)), 1.0))
    return graph

analysis_edges = edges[~edges["predicate_group"].eq("ATTRIBUTE") & ~edges["added_by"].eq("model_name_member")]
centrality = {}
for view, table, weight in [("Stage 1 (with lexicon edges)", analysis_edges, "document_count"),
                            ("article claims only", analysis_edges[analysis_edges["has_article_claim"]], "article_documents")]:
    metrics = calculate_node_metrics(analysis_graph(table, weight), view, "weight", sample_size=1000, seed=SEED)
    centrality[view] = metrics.assign(rank=np.arange(1, len(metrics) + 1)).set_index("entity_id")

stage1_view, article_view = centrality
rank_compare = (centrality[stage1_view][["canonical_name", "rank", "pagerank"]].head(20)
                .join(centrality[article_view][["rank", "pagerank"]].add_suffix("_articles")))
rank_compare["rank_change"] = rank_compare["rank"] - rank_compare["rank_articles"]
display(rank_compare.round(4))
""")

code("""
moved = rank_compare[rank_compare["rank_change"].abs() >= 3]
dropped = rank_compare[rank_compare["rank_articles"].isna()]
display(Markdown(f'''
**Reading.** {len(rank_compare) - len(dropped)} of the 20 highest-PageRank entities of Stage 1 keep a rank in the
article-only view; {len(moved)} move by three or more places{" (" + ", ".join(moved["canonical_name"]) + ")" if len(moved) else ""}.
The lexicon edges are few ({int(edges["in_lexicon"].sum())} of {len(edges)}), so they do not drive the central actors, but rankings that are meant to
describe the articles are computed without them (Section 3.6 also removes `competes with`).
'''))
""")

# ---------------------------------------------------------------- 3 Support
md("""
## 3. Does the Sentence State the Relation? Evidence Support Check

### 3.1 Method

The human review of Stage 1 found that about 40 % of the sampled Tier B relations were wrong, mostly because a
relation was attached to the wrong entity in a sentence naming several models or companies. A refinement rule per
error type cannot catch all of these. A sentence-level check can: for every evidence sentence, a natural language
inference (NLI) model decides whether the sentence (premise) entails the relation written as a sentence
(hypothesis).

| relation | hypothesis |
|---|---|
| `Qwen -> developed by -> Alibaba` | "Qwen was developed by Alibaba." |
| `DeepSeek -- competes with -- OpenAI` | "DeepSeek competes with OpenAI." and "OpenAI competes with DeepSeek." (best of both) |

- **Model**: `DeBERTa-v3-large` trained on MNLI, FEVER, ANLI, LingNLI and WANLI, a standard zero-shot claim
  verification model that runs locally (no API), so every score is reproducible.
- **Names**: the hypothesis uses the **graph node names**, not the words of the sentence. A sentence about
  *Llama Nemotron* (NVIDIA) therefore does not support `Llama -> developed by -> NVIDIA`: entity-linking errors become
  visible to the check. For composite family names (`Command R / Aya`) the part that occurs in the sentence is used.
- **Support score** of a sentence = entailment probability. A sentence supports the relation if the score is at
  least **0.5** (entailment more likely than not), a value fixed before comparing with the labels.
- **Lists**: a sentence that names three or more models or organisations does not support `competes with`
  (Section 3.3 shows why).
- **Edge**: an edge is `supported` if at least one of its sentences supports it.
- **Name patterns** (`Qwen3-8B -> member of family -> Qwen`) are not checked: the relation comes from the model
  name, not from what the sentence says, so an NLI model has nothing to verify (a sentence that only names
  *Qwen3-8B* is neutral about its family). The model-name step was checked in the Stage 1 supplementary review
  (13 of 13 correct); these edges keep `support_status = name_pattern`.

### 3.2 Evidence sentences and scoring

Scored: all evidence sentences of the article edges of the graph, all sentences of the relations a person labelled
(including those removed after the review, so that the check can be compared with the labels), and the sentences of
the article relations excluded by the lexicon filter.
""")

code("""
def triple_key(subject_id, predicate, object_id):
    # competes with is symmetric: one key for both orders
    if predicate == "competes with":
        subject_id, object_id = sorted([subject_id, object_id])
    return f"{subject_id}|{predicate}|{object_id}"

def with_key(frame):
    return frame.assign(key=[triple_key(*t) for t in frame[TRIPLE].itertuples(index=False, name=None)])

# Human labels of Stage 1
random_sample = pd.read_excel(REVIEW_DIR / "relation_review_corrected_seed2026.xlsx", sheet_name="Review")
supplement = pd.read_excel(REVIEW_DIR / "relation_review_supplement_seed2026.xlsx", sheet_name="Review")
supplement = supplement[supplement["added_by"].eq("role_typed_tier_B")]   # the only supplementary group in the graph
findings_check = pd.read_csv(REVIEW_DIR / "insight_relation_check.csv").rename(columns={"relation": "predicate"})
review_columns = {"head_id": "subject_id", "tail_id": "object_id", "review_error_type": "error_type"}
labels = pd.concat([
    random_sample.rename(columns=review_columns).assign(label=random_sample["review_correct"], label_set="random sample",
                                                        label_source="human"),
    supplement.rename(columns=review_columns).assign(label=supplement["review_correct"], label_set="random sample",
                                                     label_source="human"),
    findings_check.rename(columns=review_columns).assign(
        label=findings_check["verdict"], label_set="findings check",
        label_source=np.where(findings_check["human_checked"], "human", "AI first pass")),
], ignore_index=True)
labels["label"] = labels["label"].astype(str).str.strip().str.upper()
labels = with_key(labels)[["key", *TRIPLE, "label", "label_set", "label_source", "error_type", "text_unit_id"]]
print(pd.crosstab([labels["label_set"], labels["label_source"]], labels["label"]))
""")

code("""
# Evidence sentences (GLiREL) with the corrected node IDs of Stage 1
glirel = (evidence_status[evidence_status["evidence_status"].eq("ok")]
          .rename(columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id",
                           "head": "subject", "tail": "object", "score": "glirel_score"}))
glirel = with_key(glirel).assign(knowledge_source="article_claim",
                                 text_unit_id=lambda d: d["doc_id"] + ":" + d["sentence_id"].astype(str))
graph_keys = set(with_key(edges[edges["has_article_claim"]])["key"])
conflict_keys = set(with_key(conflicts.rename(columns={"head_id": "subject_id", "relation": "predicate",
                                                       "tail_id": "object_id"}))["key"])
glirel = glirel[glirel["key"].isin(graph_keys | set(labels["key"]) | conflict_keys)]

columns = ["key", *TRIPLE, "subject", "object", "doc_id", "sentence_id", "text_unit_id", "sentence_text", "glirel_score"]
pool = glirel[columns].reset_index(drop=True)

# Named models / organisations per sentence (lists)
id_map = (nodes_corrected.assign(original_id=nodes_corrected["merged_ids"].str.split(" | ", regex=False))
          .explode("original_id").set_index("original_id")["entity_id"])
compact_of_node = nodes_corrected.set_index("entity_id")["entity_type"].map(compact_type_of)
generic_ids = set(nodes_corrected.loc[nodes_corrected["is_generic"].astype(str).eq("True"), "entity_id"])
actors = sentence_actor_counts(mentions, id_map, compact_of_node, generic_ids)
pool["n_actors"] = [int(actors.get((d, s), 0)) for d, s in zip(pool["doc_id"], pool["sentence_id"])]

pairs = build_pairs(pool, head_name="subject", tail_name="object")
print(f"Evidence sentences: {len(pool):,} of {pool['key'].nunique():,} triples -> {len(pairs):,} premise/hypothesis pairs")
scored = score_pairs(pairs, NLIVerifier(NLI_MODEL, batch_size=32), NLI_CACHE, NLI_MODEL)
sentence_support, edge_support = aggregate_support(pool, scored, ["key"], threshold=SUPPORT_THRESHOLD)
print(f"Sentences supporting their relation: {sentence_support['supported'].mean():.1%}; "
      f"list sentences: {sentence_support['list_sentence'].mean():.1%}")
""")

md("""
### 3.3 Agreement with the human labels

Before the check is applied, it is compared with the Stage 1 labels:
- **random sample** (human): 90 Tier B relations drawn at random before the review, plus the 3 relations of the
  supplementary sample that are in the graph. This is the main evaluation;
- **findings check**: the relations behind the Stage 1 findings. `FALSE` labels were confirmed by a person, `TRUE`
  labels mostly come from an AI first pass, so this set is reported separately.

The baseline is the extraction confidence of GLiREL (mean score of the edge). The ranking quality (AUC) does not
depend on a threshold; precision and recall are given for the fixed threshold 0.5.
""")

code("""
glirel_mean = pool.groupby("key")["glirel_score"].mean()
decided = labels[labels["label"].isin(["TRUE", "FALSE"])].merge(edge_support, on="key", how="left", validate="many_to_one")
decided["is_true"] = decided["label"].eq("TRUE")
decided["kept"] = decided["support_status"].eq("supported")
decided["is_competes"] = decided["predicate"].eq("competes with")

def agreement_row(name, group):
    y, kept = group["is_true"], group["kept"]
    tp = int((y & kept).sum())
    low, high = wilson(tp, int(kept.sum()))
    low0, high0 = wilson(int(y.sum()), len(y))
    return {"labels": name, "relations": len(group), "precision without check": f"{y.mean():.2f} ({low0:.2f}-{high0:.2f})",
            "AUC NLI support": round(roc_auc_score(y, group["support_max"]), 3) if y.nunique() == 2 else np.nan,
            "AUC GLiREL score": round(roc_auc_score(y, group["key"].map(glirel_mean)), 3) if y.nunique() == 2 else np.nan,
            "kept": int(kept.sum()),
            "precision of kept": f"{tp / kept.sum():.2f} ({low:.2f}-{high:.2f})" if kept.sum() else "none kept",
            "recall of TRUE": round(tp / y.sum(), 3), "FALSE removed": f"{int((~y & ~kept).sum())} / {int((~y).sum())}"}

agreement = pd.DataFrame([agreement_row(f"{s} · {'competes with' if c else 'other relations'}", g)
                          for (s, c), g in decided.groupby(["label_set", "is_competes"])]
                         + [agreement_row(f"{s} · all", g) for s, g in decided.groupby("label_set")])
display(agreement)
""")

code("""
# Why lists are excluded for competes with: the same check without the list rule
no_list_rule = sentence_support.assign(supported=sentence_support["support_score"] >= SUPPORT_THRESHOLD)
kept_without_rule = no_list_rule.groupby("key")["supported"].any()
competes = decided[decided["is_competes"]].assign(kept_without_list_rule=lambda d: d["key"].map(kept_without_rule))
display(pd.DataFrame([
    {"labels": s, "relations": len(g), "TRUE": int(g["is_true"].sum()),
     "kept by NLI alone": int(g["kept_without_list_rule"].sum()),
     "TRUE among them": int((g["kept_without_list_rule"] & g["is_true"]).sum()),
     "kept with list rule": int(g["kept"].sum()), "TRUE among them ": int((g["kept"] & g["is_true"]).sum())}
    for s, g in competes.groupby("label_set")]))
display(sentence_support[sentence_support["predicate"].eq("competes with") & sentence_support["list_sentence"]
                         & (sentence_support["support_score"] >= 0.9)]
        .sample(4, random_state=SEED)[["subject", "object", "n_actors", "support_score", "sentence_text"]])
""")

md("""
**Why the list rule.** For `competes with`, the NLI model accepts almost every sentence that names the two entities
together, including lists such as *"open competitors, including DeepSeek, Mistral, and Meta's Llama"*: it reads
co-listing as rivalry. The reviewers did not, and labelled such relations `FALSE` (mostly `entity_linking`). This is
the error the feedback describes (sentences mentioning several competing models). A sentence naming three or more
models or organisations is therefore not accepted as evidence of a rivalry between two of them; three is the
smallest list and was not tuned (four is shown below as a sensitivity check). Other relations are not affected: a
list sentence can still say *"Qwen, developed by Alibaba, ..."*.

The rule is conservative: models and organisations are counted together, so *"DeepSeek said ... its R1 model competes
with OpenAI's o1"* (four names) is also rejected. Looser definitions keep such sentences but also more of the
reviewers' `FALSE` relations, and with 22 sampled `competes with` relations (6 `TRUE`) the difference cannot be
estimated. `competes with` is therefore not used for retrieval (Section 3.6).
""")

code("""
rows = []
for set_name, group in decided.groupby("label_set"):
    table = threshold_table(group["is_true"], group["support_max"], thresholds=(0.1, 0.3, 0.5, 0.7, 0.9))
    rows.append(table.assign(labels=set_name, rule="NLI max score, no list rule"))
display(pd.concat(rows)[["labels", "threshold", "kept", "of", "precision", "precision_ci95", "recall_of_true",
                         "removed_false", "removed_true"]].round(3))

sensitivity = []
for minimum in (3, 4):
    rule = (sentence_support["support_score"] >= SUPPORT_THRESHOLD) & ~(
        sentence_support["predicate"].eq("competes with") & (sentence_support["n_actors"] >= minimum))
    kept_keys = set(sentence_support.loc[rule, "key"])
    for set_name, group in decided[decided["is_competes"]].groupby("label_set"):
        kept = group["key"].isin(kept_keys)
        sensitivity.append({"list = names >=": minimum, "labels": set_name, "kept": int(kept.sum()), "of": len(group),
                            "TRUE kept": int((kept & group["is_true"]).sum()), "TRUE": int(group["is_true"].sum())})
display(pd.DataFrame(sensitivity))
""")

md("""
**Reading.** The threshold hardly matters: most scores are close to 0 or 1, so 0.5 is kept. The NLI score ranks
correct above incorrect relations much better than the GLiREL confidence (AUC in the table above), which Stage 1 had
therefore not used as a filter.
""")

code("""
by_predicate = (decided.groupby("predicate")
                .agg(labelled=("is_true", "size"), true=("is_true", "sum"), kept=("kept", "sum"),
                     true_kept=("is_true", lambda s: int((s & decided.loc[s.index, "kept"]).sum())))
                .assign(precision_before=lambda d: (d["true"] / d["labelled"]).round(2),
                        precision_after=lambda d: (d["true_kept"] / d["kept"]).round(2),
                        recall=lambda d: (d["true_kept"] / d["true"]).round(2))
                .sort_values("labelled", ascending=False))
display(by_predicate)

false_labels = decided[~decided["is_true"]]
display(pd.crosstab(false_labels["error_type"].fillna("unspecified"), false_labels["kept"].map({True: "kept", False: "removed"}),
                    margins=True).rename_axis(index="error type of FALSE labels", columns="check"))
""")

md("""
### 3.4 Where the check and the reviewers disagree
""")

code("""
def best_sentence(key):
    rows = sentence_support[sentence_support["key"] == key].sort_values("support_score", ascending=False)
    return rows.iloc[0]

disagreements = []
for kind, mask in [("labelled FALSE, check: supported", ~decided["is_true"] & decided["kept"]),
                   ("labelled TRUE, check: not supported", decided["is_true"] & ~decided["kept"])]:
    for row in decided[mask & decided["label_source"].eq("human")].itertuples():
        sentence = best_sentence(row.key)
        disagreements.append({"case": kind, "relation": f"{labels_of.get(row.subject_id)} -> {row.predicate} -> "
                              f"{labels_of.get(row.object_id)}", "error type": row.error_type,
                              "best score": round(float(sentence["support_score"]), 3), "best sentence": sentence["sentence_text"][:230]})
disagreements = pd.DataFrame(disagreements)
print(disagreements["case"].value_counts().to_string())
with pd.option_context("display.max_colwidth", 230):
    display(disagreements.groupby("case").head(8))
""")

code("""
fp = disagreements[disagreements["case"].str.startswith("labelled FALSE")]
display(Markdown(f'''
**Reading.** The check keeps {len(fp)} relations that a person labelled `FALSE`, mostly
{", ".join(f"`{e}` ({n})" for e, n in fp["error type"].fillna("unspecified").value_counts().head(3).items())}. Three
kinds of case recur in the table:
- **fragment nodes** (`Lightning`, `Ultra`, `Pro model`): the sentence states the relation for the full name
  (*NVIDIA Nemotron 3.5 Lightning*), and the hypothesis with the fragment still reads as entailed;
- **possessive lists**: in *"Claude Opus 4, OpenAI's GPT-4o and o4-mini"* the model attaches the owner to the wrong
  name, the same linking error as the extraction;
- **stricter reviewers**: *"DeepSeek was founded ... by the Chinese hedge fund High Flyer"* or *"Broadcom in the
  U.S."* were labelled `FALSE` for the node or the context, although the sentence states the words of the relation.

Only {int((disagreements["case"] == "labelled TRUE, check: not supported").sum())} relations labelled `TRUE` are not
supported, all `competes with` relations whose only sentences are lists. The check is therefore a filter that removes
about half of the errors at almost no loss of correct relations, not a proof of correctness: the precision of the
kept relations (Section 3.3) is the number to report, and errors of the first two kinds remain in the graph.
'''))
""")

md("""
### 3.5 Support status of the graph edges
""")

code("""
edges = with_key(edges).merge(edge_support, on="key", how="left", validate="one_to_one")
edges.loc[edges["knowledge_layer"].eq("curated_lexicon"), "support_status"] = "no_passage"
edges.loc[edges["knowledge_layer"].eq("name_pattern"), "support_status"] = "name_pattern"
assert edges["support_status"].notna().all()
human_true = edges["review_status"].isin(["reviewed_true", "checked_true_human"])
edges["support_disagrees_with_review"] = human_true & edges["support_status"].eq("not_supported")
# Retrieval uses article claims and name patterns whose sentences state them; a relation a person confirmed is kept
edges["retrieval_eligible"] = (edges["support_status"].isin(["supported", "name_pattern"])
                               | (edges["knowledge_layer"].eq("article_claim") & human_true))

status = (edges.groupby(["knowledge_layer", "predicate"])
          .agg(edges=("key", "size"), supported=("support_status", lambda s: int(s.eq("supported").sum())),
               eligible=("retrieval_eligible", "sum"))
          .assign(share_supported=lambda d: (d["supported"] / d["edges"]).round(2))
          .sort_values("edges", ascending=False))
display(status)
print(f"Edges eligible for retrieval: {int(edges['retrieval_eligible'].sum())} of "
      f"{int(edges['knowledge_layer'].ne('curated_lexicon').sum())} article / name-pattern edges; "
      f"human-confirmed but not supported by the check: {int(edges['support_disagrees_with_review'].sum())}")

conflict_support = with_key(conflicts.rename(columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id"}))
conflict_support = conflict_support.merge(edge_support, on="key", how="left")
print(f"Article relations excluded by the lexicon filter: {conflict_support['support_status'].eq('supported').mean():.0%} "
      f"have a sentence that states them")
display(conflict_support[conflict_support["support_status"].eq("supported") & conflict_support["only_reason"]]
        .head(8)[["subject", "predicate", "object", "n_distinct_sentences", "support_max", "supported_sentences"]])
""")

code("""
eligible_unreviewed = edges[edges["retrieval_eligible"] & edges["knowledge_layer"].eq("article_claim")
                            & edges["review_status"].eq("unreviewed")]
other = decided[decided["label_set"].eq("random sample") & ~decided["is_competes"]]
competes_rs = decided[decided["label_set"].eq("random sample") & decided["is_competes"]]
p_other = (other["is_true"] & other["kept"]).sum() / other["kept"].sum()
display(Markdown(f'''
**Reading.** {int(edges["retrieval_eligible"].sum())} edges remain eligible for retrieval.
{int(status.loc["article_claim"]["edges"].sum() - status.loc["article_claim"]["supported"].sum())} article edges have
no sentence that states them and are kept in the graph with `support_status = not_supported`, but retrieval no longer
uses them. `competes with` loses most of its edges, as the review suggested (see Section 3.6). For the
{len(eligible_unreviewed)} eligible article edges that no person checked, the random sample gives the expected
precision: {p_other:.2f} for relations other than `competes with` (TRUE among kept), against
{other["is_true"].mean():.2f} without the check. For `competes with`, only {int(competes_rs["kept"].sum())} of
{len(competes_rs)} sampled relations pass, too few for an estimate.

About {conflict_support["support_status"].eq("supported").mean():.0%} of the relations excluded by the lexicon filter
pass the check. Their sentences show why the filter is still needed: `Llama -> developed by -> NVIDIA` comes from
*Llama Nemotron* and *Llama-3.1-Nemotron-70B* (NVIDIA models built on Llama), `Qwen -> developed by -> OpenAI` from
lists such as *"OpenAI's text-embedding-3-large and Qwen3-Embedding-8B"*, `gpt-oss -> developed by -> NVIDIA` from
sentences about running gpt-oss on NVIDIA hardware. The two checks catch different errors: NLI catches relations
the sentence does not state, the lexicon catches family-level linking errors that NLI accepts. The lexicon filter stays,
recorded as a decision based on curated knowledge (`lexicon_conflicts`), not as article evidence.
'''))
""")

md("""
### 3.6 `competes with` is not used for retrieval or analysis

`competes with` is the relation the evidence cannot validate:
- in the random sample, 6 of 22 relations were correct before any check;
- the NLI check accepts almost all of them, because news sentences list rivals ("Google, OpenAI, and Anthropic ...
  against open-weight models from China"), and the list rule that removes these also removes clear rivalry sentences
  (Section 3.3);
- of 22 sampled relations, 6 pass, 3 of them correct: no rule tested separates correct from incorrect rivalry claims.

A relation that cannot be validated should not answer questions. `competes with` edges therefore stay in the graph
with `relation_reliability = low` and their sentences, but they are not retrieval-eligible and are left out of the
graph metrics. Questions about competition are answered from retrieved text passages (WP1 text retrieval), which
cite what the article says instead of turning it into an edge. Co-mentions of rivals remain visible in the
co-occurrence graph of Stage 1, which does not claim a relation.
""")

code("""
LOW_RELIABILITY = {"competes with": "cannot be validated at sentence level: lists of rivals, 6 of 22 sampled relations correct"}
edges["relation_reliability"] = np.where(edges["predicate"].isin(LOW_RELIABILITY), "low", "standard")
edges["retrieval_eligible"] = edges["retrieval_eligible"] & edges["relation_reliability"].eq("standard")
print(f"competes with edges kept in the graph but not used: {int(edges['predicate'].eq('competes with').sum())}; "
      f"edges eligible for retrieval: {int(edges['retrieval_eligible'].sum())}")

# Graph metrics of the retrieval graph (analysis view: no model-name or attribute edges)
retrieval_view = edges[edges["retrieval_eligible"] & edges["has_article_claim"]
                       & ~edges["predicate_group"].eq("ATTRIBUTE") & ~edges["added_by"].eq("model_name_member")]
retrieval_metrics = calculate_node_metrics(analysis_graph(retrieval_view, "article_documents"), "retrieval graph",
                                           "weight", sample_size=1000, seed=SEED)
retrieval_rank = retrieval_metrics.assign(rank_retrieval_graph=np.arange(1, len(retrieval_metrics) + 1)).set_index("entity_id")
rank_shift = (rank_compare[["canonical_name", "rank_articles"]].head(15)
              .join(retrieval_rank[["rank_retrieval_graph", "pagerank"]]))
display(rank_shift.round(4))
""")

code("""
dropped = rank_shift[rank_shift["rank_retrieval_graph"] - rank_shift["rank_articles"] >= 5]
competes_degree = edges[edges["predicate"].eq("competes with")].melt(value_vars=["subject_id", "object_id"])["value"].value_counts()
display(Markdown(f'''
**Reading.** Without `competes with`, the leading actors keep their places, but entities whose centrality came
mainly from rivalry edges fall: {", ".join(f"{r.canonical_name} ({int(r.rank_articles)} -> {int(r.rank_retrieval_graph)}, "
f"{int(competes_degree.get(i, 0))} competes with edges)" for i, r in dropped.iterrows())}. Their rank in Stage 1 reflected
how often they were named next to rivals rather than relations the articles state about them. Graph metrics reported
from Stage 3 on use this retrieval graph.
'''))
""")

# ---------------------------------------------------------------- 4 Retrieval
md("""
## 4. Retrieval Check: Do the Passages Support the Relations?

The retrieval check of Stage 1 (Section 6.9) only tested that a passage is returned. It is repeated with the same
retriever on the Stage 1 assertions (**before**) and on knowledge graph v2 (**after**):
- only retrieval-eligible edges, without lexicon rows; lexicon facts are returned separately as `background`;
- only passages whose sentence supports the relation (support score >= 0.5 and, for `competes with`, not a list),
  ranked by support score.

The questions are the three of Stage 1 plus competency questions of the relation schema.
""")

code("""
text_units = pd.DataFrame({"text_unit_id": sentences["doc_id"] + ":" + sentences["sentence_id"].astype(str),
                           "document_id": sentences["doc_id"], "text_unit_text": sentences["sentence_text"]})
documents = (assertions.dropna(subset=["document_id"]).groupby("document_id")["published_at"].first().reset_index())
if CORE_ARTICLES is not None:
    articles = pd.read_parquet(CORE_ARTICLES, columns=["doc_id", "title", "url", "domain"])
    documents = documents.merge(articles.rename(columns={"doc_id": "document_id"}), on="document_id", how="left")

# Knowledge graph v2 assertions: one row per evidence sentence, with layer and support
support_columns = ["support_score", "contradiction_score", "list_sentence", "n_actors", "hypothesis", "supported"]
support_by_unit = sentence_support.drop_duplicates(["key", "text_unit_id"])[["key", "text_unit_id", *support_columns]]
assertions_v2 = (with_key(assertions)
                 .merge(support_by_unit, on=["key", "text_unit_id"], how="left", validate="many_to_one")
                 .merge(edges[["key", "retrieval_eligible"]], on="key", how="left", validate="many_to_one"))
assertions_v2["supported"] = assertions_v2["supported"].fillna(False).astype(bool)
assert assertions_v2.loc[assertions_v2["knowledge_source"].eq("article_claim"), "support_score"].notna().all()
print(assertions_v2.groupby("knowledge_source")[["support_score"]].agg(["count", "mean"]).round(3))

background = (edges[edges["in_lexicon"]][TRIPLE].assign(knowledge_source="curated_lexicon"))
# Passages for retrieval: sentences that support their relation (score and list rule), or name-pattern sentences
retrieval_assertions = assertions_v2[assertions_v2["retrieval_eligible"].fillna(False).astype(bool)
                                     & (assertions_v2["supported"] | assertions_v2["knowledge_source"].eq("name_pattern"))]
print(f"Passages for retrieval: {len(retrieval_assertions):,} of {assertions_v2['text_unit_id'].notna().sum():,} evidence rows")
retriever_before = GraphEvidenceRetriever(nodes, assertions.dropna(subset=["text_unit_id"]), text_units, documents)
retriever_after = GraphEvidenceRetriever(nodes, retrieval_assertions, text_units, documents, background=background)
support_of_relation = assertions_v2.set_index("relation_id")["support_score"]
supported_relation = assertions_v2.set_index("relation_id")["supported"]
""")

code("""
QUESTIONS = [
    ("Who developed Qwen?", ["developed by"]),
    ("Which companies compete with DeepSeek?", ["competes with"]),
    ("What does NVIDIA manufacture?", ["manufactured by"]),
    ("Who developed Gemma?", ["developed by"]),
    ("Who developed Llama?", ["developed by"]),
    ("Which companies compete with OpenAI?", ["competes with"]),
    ("Which models does Kimi outperform?", ["outperforms"]),
    ("Where is DeepSeek headquartered?", ["headquartered in"]),
    ("Who works for Meta?", ["works for"]),
    ("Who founded DeepSeek?", ["founded by"]),
    ("Where is Nemotron available?", ["available on"]),
    ("Who invested in OpenAI?", ["invested in"]),
    ("Under which license is Gemma released?", ["licensed under"]),
]
TOP_K = 3

rows, background_rows = [], []
for question, predicates in QUESTIONS:
    for system, retriever, extra in [("before", retriever_before, {}), ("after", retriever_after, {"min_support": SUPPORT_THRESHOLD})]:
        result = retriever.retrieve(question, predicates=predicates, top_k=TOP_K, **extra)
        for rank, item in enumerate(result["evidence"], start=1):
            rows.append({"question": question, "system": system, "rank": rank, "relation_id": item["relation_id"],
                         "relation": f"{item['subject']['name']} -> {item['predicate']} -> {item['object']['name']}",
                         "support_score": support_of_relation.get(item["relation_id"], np.nan),
                         "published_at": item["published_at"], "document_id": item["document_id"],
                         "passage": " ".join(str(item["passage"]).split())})
        if system == "after":
            background_rows += [{"question": question, "background fact": f"{f['subject']['name']} -> {f['predicate']} -> "
                                 f"{f['object']['name']}", "source": f["knowledge_source"]} for f in result["background"]]
retrieval_check = pd.DataFrame(rows)
retrieval_check["supported"] = retrieval_check["relation_id"].map(supported_relation)
retrieval_check["in_other_system"] = retrieval_check.duplicated(["question", "relation_id"], keep=False)

summary = (retrieval_check.groupby("system")
           .agg(passages=("relation_id", "size"), supported=("supported", "sum"),
                questions_answered=("question", "nunique"), changed=("in_other_system", lambda s: int((~s).sum())))
           .assign(share_supported=lambda d: (d["supported"] / d["passages"]).round(2))
           .reindex(["before", "after"]))
display(summary)
display(pd.DataFrame(background_rows))
with pd.option_context("display.max_colwidth", 170):
    display(retrieval_check[["question", "system", "rank", "relation", "support_score", "supported", "passage"]].round(3))
""")

md("""
**Reading.** "Supported" in the summary is measured with the same check that filters the passages, so the share
after the revision is 1 by construction. The informative numbers are the passages *before* that the check rejects,
the passages that changed, and whether every question still finds evidence. Rejected passages include the cases
named in the feedback, e.g. *DeepSeek -> headquartered in -> United States* from *"... companies, largely
headquartered in the United States"*, *Gemma -> licensed under -> Apache 2.0* from a sentence about Mistral's
license, and *DeepSeek -- competes with -- Google* from a sentence comparing groups of companies; but also clear
rivalry sentences with four names (Section 3.3). The two competition questions return no graph evidence after the
revision, by design (Section 3.6): they go to text retrieval. Lexicon facts now appear as background, separate from
the cited passages. An independent check
needs people: the cell below writes a blind review sheet with the passages of both systems in random order and
without the system name. Each team member labels whether the passage states the relation, and the labels are read
back in Section 4.1 once they exist.
""")

code("""
REVIEW_SHEET = ARTIFACT_DIR / "retrieval_support_review.xlsx"
REVIEW_KEY = ARTIFACT_DIR / "retrieval_support_review_key.csv"
sheet = (retrieval_check.drop_duplicates(["question", "relation_id"])
         .sample(frac=1.0, random_state=SEED).reset_index(drop=True))
sheet.insert(0, "item_id", [f"q{i:03d}" for i in range(1, len(sheet) + 1)])
if not REVIEW_SHEET.exists():   # never overwrite a sheet that may already contain labels
    review_columns_out = ["item_id", "question", "relation", "passage", "published_at", "document_id"]
    sheet[review_columns_out].assign(passage_states_relation="", reviewer="", notes="").to_excel(
        REVIEW_SHEET, sheet_name="Review", index=False)
    membership = retrieval_check.groupby(["question", "relation_id"])["system"].agg(lambda s: "+".join(sorted(set(s))))
    sheet[["item_id", "question", "relation_id"]].join(membership, on=["question", "relation_id"]).to_csv(REVIEW_KEY, index=False)
    print(f"Review sheet written: {REVIEW_SHEET.relative_to(ROOT)} ({len(sheet)} passages)")
else:
    print(f"Review sheet exists and is kept: {REVIEW_SHEET.relative_to(ROOT)}")
""")

md("""
### 4.1 Human check of the retrieved passages

Each labelled copy of the sheet is saved as `retrieval_support_review_<reviewer>.xlsx` (`passage_states_relation` =
`TRUE`, `FALSE` or `AMBIGUOUS`). Labels are matched to the passages by question, relation and passage text, because
item IDs change when the sheet is generated again. The reviewer saw neither the system nor the support score.
`AMBIGUOUS` labels are reported but not counted in the share.
""")

code("""
LABEL_FILES = sorted(ARTIFACT_DIR.glob("retrieval_support_review_*.xlsx"))
MATCH = ["question", "relation", "passage"]
human = pd.concat([pd.read_excel(f, sheet_name="Review") for f in LABEL_FILES], ignore_index=True) if LABEL_FILES else pd.DataFrame()
if human.empty:
    print("No labels yet: the human check of the retrieved passages is pending.")
else:
    human["human_label"] = human["passage_states_relation"].astype(str).str.strip().str.upper()
    human = human[human["human_label"].isin(["TRUE", "FALSE", "AMBIGUOUS"])].drop_duplicates(MATCH + ["reviewer"])
    checked = retrieval_check.merge(human[MATCH + ["human_label", "reviewer", "notes"]], on=MATCH, how="left")
    print(f"Label files: {[f.name for f in LABEL_FILES]}; reviewers: {sorted(human['reviewer'].dropna().unique())}; "
          f"retrieved passages without a label: {int(checked['human_label'].isna().sum())}")

    rows = []
    for system in ["before", "after"]:
        part = checked[checked["system"].eq(system)]
        counts = part["human_label"].value_counts()
        true, false = int(counts.get("TRUE", 0)), int(counts.get("FALSE", 0))
        low, high = wilson(true, true + false)
        rows.append({"system": system, "passages": len(part), "TRUE": true, "FALSE": false,
                     "AMBIGUOUS": int(counts.get("AMBIGUOUS", 0)),
                     "share stating the relation": round(true / (true + false), 3), "95% CI": f"{low:.2f}-{high:.2f}"})
    human_summary = pd.DataFrame(rows)
    display(human_summary)

    # Does the support check agree with the reviewer? (unique passages)
    unique_passages = checked.drop_duplicates(["question", "relation_id"])
    display(pd.crosstab(unique_passages["supported"].map({True: "check: supported", False: "check: not supported"}),
                        unique_passages["human_label"], margins=True).rename_axis(index=None, columns="human label"))
    with pd.option_context("display.max_colwidth", 150):
        display(unique_passages[unique_passages["human_label"].isin(["FALSE", "AMBIGUOUS"])]
                [["system", "relation", "supported", "human_label", "notes", "passage"]].sort_values(["human_label", "system"]))
""")

code("""
if not human.empty:
    rejected_accepted = unique_passages[~unique_passages["supported"].astype(bool) & unique_passages["human_label"].eq("TRUE")]
    accepted_rejected = unique_passages[unique_passages["supported"].astype(bool) & unique_passages["human_label"].eq("FALSE")]
    after_ambiguous = checked[checked["system"].eq("after") & checked["human_label"].eq("AMBIGUOUS")]
    before_h, after_h = human_summary.set_index("system").loc["before"], human_summary.set_index("system").loc["after"]
    display(Markdown(f'''
**Reading.** In the blind check, {before_h["share stating the relation"]:.0%}
of the passages returned before and {after_h["share stating the relation"]:.0%} of those returned after the revision
state their relation ({before_h["95% CI"]} and {after_h["95% CI"]}); for these {len(QUESTIONS)} questions most
passages were already correct. Where the check and the reviewer differ, the cases show what the check does and does
not catch:
- the check rejected {len(rejected_accepted)} passages the reviewer accepted, {int(rejected_accepted["relation"].str.contains("competes with").sum())}
  of them `competes with` sentences removed by the list rule (e.g. *"DeepSeek said ... its R1 model competes with
  OpenAI's o1"*). The list rule is too strict for sentences that state a rivalry between two named players, one
  more reason not to answer competition questions from the graph (Section 3.6);
- the check accepted {len(accepted_rejected)} passages the reviewer rejected because the sentence is about a single
  product or another entity, not the node: *DolphinGemma* for Gemma, *QwQ-32B* for Qwen, *NeMo Switchyard* for
  Nemotron. This is the family-level linking error of Section 3.4, which the sentence check cannot see;
- {int(after_ambiguous["relation"].str.contains("manufactured by").sum())} of the {len(after_ambiguous)} `AMBIGUOUS`
  passages after the revision are `manufactured by` (*"NVIDIA provides ... A100 and H100 Tensor Core GPUs"*): the
  NLI model accepts a relation the sentence implies, while the reviewer asked for one it states. Implied relations
  are kept, and the answer shows the sentence, so the wording stays visible.
'''))
""")

# ---------------------------------------------------------------- 5 Outputs
md("""
## 5. Knowledge Graph v2: Outputs for Graph-Enhanced RAG

`outputs/stage3_kg_v2/` contains everything the WP2 retriever needs; `GraphEvidenceRetriever.from_directories(dir, dir)`
loads it. Small tables and the data dictionary are also written to `artifacts/stage3_wp2/kg_revision/`.
""")

code("""
edge_columns = [c for c in collapsed.columns] + ["relation_reliability",
    "knowledge_layer", "has_article_claim", "has_name_pattern", "in_lexicon", "lexicon_agreement", "article_sentences",
    "article_documents", "article_first_seen", "article_last_seen", "support_status", "support_max", "support_mean",
    "scored_sentences", "supported_sentences", "support_disagrees_with_review", "retrieval_eligible"]
edges_v2 = edges[edge_columns]
assertion_columns = [c for c in assertions_v2.columns if c not in ("key",)]
tables = {
    "nodes": nodes,
    "assertions": assertions_v2[assertion_columns],
    "retrieval_assertions": retrieval_assertions[assertion_columns],
    "edges": edges_v2,
    "background_facts": background,
    "text_units": text_units,
    "documents": documents,
    "lexicon_conflicts": conflict_support.drop(columns=["key"]),
}
for name, table in tables.items():
    table.to_parquet(OUTPUT_DIR / f"{name}.parquet", index=False)
# typed_edges_assertions.parquet: the file name GraphEvidenceRetriever.from_directories reads
retrieval_assertions[assertion_columns].to_parquet(OUTPUT_DIR / "typed_edges_assertions.parquet", index=False)

graph = nx.MultiDiGraph(graph_view="typed_v2", schema_version="stage3-wp2-open-models-kg-v2.0")
for row in edges_v2.itertuples(index=False):
    for entity_id in (row.subject_id, row.object_id):
        graph.add_node(entity_id, label=str(labels_of.get(entity_id, entity_id)))
    graph.add_edge(row.subject_id, row.object_id, key=row.predicate, knowledge_layer=row.knowledge_layer,
                   support_status=str(row.support_status), retrieval_eligible=bool(row.retrieval_eligible),
                   article_documents=int(row.article_documents), predicate_group=str(row.predicate_group))
nx.write_graphml(graph, OUTPUT_DIR / "typed_graph_v2.graphml")

evaluation_tables = {"edges_v2": edges_v2.drop(columns=["relation_ids", "document_ids", "text_unit_ids", "domains"]),
                     "agreement_with_labels": agreement, "agreement_by_predicate": by_predicate.reset_index(),
                     "label_disagreements": disagreements, "retrieval_check": retrieval_check,
                     "centrality_with_without_lexicon": rank_compare.reset_index(),
                     "lexicon_conflicts": conflict_support.drop(columns=["key"])}
for name, table in evaluation_tables.items():
    table.to_csv(ARTIFACT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
print(sorted(p.name for p in OUTPUT_DIR.iterdir()))
""")

code("""
DATA_DICTIONARY = f'''# Knowledge graph v2 (Stage 3 WP2): data dictionary

Built from the submitted Stage 1 knowledge graph by `notebooks/stage3_wp2_kg_revision.ipynb`. Node IDs, relation IDs
and text unit IDs are those of Stage 1. All dates are article publication dates.

## `edges` - one row per triple (all Stage 1 graph edges)
Stage 1 columns, plus:
| column | description |
|---|---|
| `knowledge_layer` | `article_claim` (stated in an article sentence), `name_pattern` (model name with size -> family), `curated_lexicon` (WP1 lexicon only, no sentence) |
| `has_article_claim`, `has_name_pattern`, `in_lexicon` | which sources contain the triple |
| `lexicon_agreement` | `lexicon_and_articles` or `lexicon_only` for lexicon triples |
| `article_sentences`, `article_documents`, `article_first_seen`, `article_last_seen` | support counted on article sentences only |
| `support_status` | `supported` (at least one sentence entails the relation), `not_supported`, `no_passage` (lexicon only) |
| `support_max`, `support_mean` | best and mean NLI support score of the sentences |
| `scored_sentences`, `supported_sentences` | sentences checked / supporting |
| `support_disagrees_with_review` | confirmed by a person but no supporting sentence (kept) |
| `relation_reliability` | `low` for `competes with` (cannot be validated at sentence level, Section 3.6), else `standard` |
| `retrieval_eligible` | used by the retriever: article claim or name pattern, supported or confirmed by a person, and `relation_reliability = standard` |

## `assertions` - one row per evidence sentence (`retrieval_assertions`: the rows of eligible edges)
Stage 1 columns (`relation_id`, `subject_id`, `predicate`, `object_id`, `document_id`, `text_unit_id`, `evidence_text`,
`published_at`, `extraction_method`, `rule_score`), plus `knowledge_source` (layer of the row), `support_score`
(NLI entailment probability, {NLI_MODEL}), `contradiction_score`, `n_actors` (named models / organisations in the
sentence), `list_sentence` (n_actors >= 3), `supported` (support_score >= {SUPPORT_THRESHOLD}, and for
`competes with` not a list sentence), `hypothesis` (the sentence the relation was checked as), `retrieval_eligible`.

## `background_facts`
Lexicon facts (`subject_id`, `predicate`, `object_id`, `knowledge_source = curated_lexicon`). Returned by the
retriever as `background`, never as cited evidence.

## `text_units`, `documents`
Sentences (`text_unit_id` = `doc_id:sentence_id`) and articles (publication date, title, URL, domain).

## `lexicon_conflicts`
Article relations that Stage 1 excluded because they name another developer of a lexicon family, with their
support status. They stay excluded.

## `typed_graph_v2.graphml`
All edges with `knowledge_layer`, `support_status`, `retrieval_eligible`.
'''
for directory in [OUTPUT_DIR, ARTIFACT_DIR]:
    (directory / "KG_V2_DATA_DICTIONARY.md").write_text(DATA_DICTIONARY, encoding="utf-8")

file_hash = lambda path: sha256(Path(path).read_bytes()).hexdigest()
manifest = {
    "schema_version": "stage3-wp2-open-models-kg-v2.0",
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "nli_model": NLI_MODEL, "support_threshold": SUPPORT_THRESHOLD, "list_min_actors": 3,
    "not_retrieval_eligible_predicates": LOW_RELIABILITY,
    "edges": int(len(edges_v2)), "retrieval_eligible_edges": int(edges_v2["retrieval_eligible"].sum()),
    "layers": edges_v2["knowledge_layer"].value_counts().to_dict(),
    "support_status": edges_v2["support_status"].value_counts().to_dict(),
    "inputs": {str(p.relative_to(STAGE1 if p.is_relative_to(STAGE1) else ROOT)): file_hash(p) for p in
               [KG_DIR / "nodes.parquet", KG_DIR / "typed_edges_assertions.parquet", KG_DIR / "typed_edges_collapsed.parquet",
                KG_DIR / "correction" / "relations_all_tiers.csv", KG_DIR / "correction" / "evidence_status.csv",
                REVIEW_DIR / "relation_review_corrected_seed2026.xlsx", REVIEW_DIR / "insight_relation_check.csv"]},
    "versions": {"python": platform.python_version(), "pandas": pd.__version__, "networkx": nx.__version__},
}
for directory in [OUTPUT_DIR, ARTIFACT_DIR]:
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

reloaded = GraphEvidenceRetriever.from_directories(OUTPUT_DIR, OUTPUT_DIR)
assert reloaded.retrieve("Who developed Qwen?", predicates=["developed by"], min_support=SUPPORT_THRESHOLD)["evidence"]
print(json.dumps({k: v for k, v in manifest.items() if k != "inputs"}, indent=2))
""")

# ---------------------------------------------------------------- 6 Summary
md("""
## 6. Summary and Remaining Limitations
""")

code("""
random_all = decided[decided["label_set"].eq("random sample")]
kept_all = random_all[random_all["kept"]]
low_before, high_before = wilson(int(random_all["is_true"].sum()), len(random_all))
low_after, high_after = wilson(int(kept_all["is_true"].sum()), len(kept_all))
before_summary, after_summary = summary.loc["before"], summary.loc["after"]
n_mixed = int(edges["lexicon_agreement"].eq("lexicon_and_articles").sum())
n_lexicon_only = int(edges["lexicon_agreement"].eq("lexicon_only").sum())
rejected_before = int(before_summary["passages"] - before_summary["supported"])
human_text = ("a blind human check of the passages is pending" if human.empty else
              f"in a blind human check, {human_summary.loc[0, 'share stating the relation']:.0%} of the passages before and "
              f"{human_summary.loc[1, 'share stating the relation']:.0%} after state their relation (Section 4.1)")
display(Markdown(f'''
**What changed for Stage 3**

| feedback | knowledge graph v2 |
|---|---|
| keep lexicon knowledge separate | every edge has a `knowledge_layer`; {n_mixed} edges that Stage 1 labelled Tier A are article claims; {n_lexicon_only} lexicon-only facts are returned as `background` and never cited; graph metrics are computed on validated article claims only |
| validate the relations used for retrieval | `competes with`, which cannot be validated, is not used for retrieval or graph metrics; every other article sentence was checked against its relation; on the human random sample, precision rises from {random_all["is_true"].mean():.2f} ({low_before:.2f}-{high_before:.2f}) to {kept_all["is_true"].mean():.2f} ({low_after:.2f}-{high_after:.2f}) while {kept_all["is_true"].sum() / random_all["is_true"].sum():.0%} of the correct relations are kept; {int(edges["retrieval_eligible"].sum())} of {len(edges)} edges are used for retrieval |
| passages must support the relation | the retriever cites only sentences that state the relation, ranked by support score; in the repeated retrieval check {rejected_before} of {int(before_summary["passages"])} passages returned before do not pass the check; {int(after_summary["questions_answered"])} of {len(QUESTIONS)} questions return graph evidence, the competition questions go to text retrieval; {human_text} |

**Remaining limitations**
- **The check is a filter, not a proof.** About a quarter of the kept relations in the random sample are still
  wrong, mostly fragment nodes and possessive lists that the NLI model accepts (Section 3.4). Retrieval therefore
  shows the passage with every relation, so that the answer can cite the sentence rather than the edge.
- **No graph answer for competition.** `competes with` stays in the graph with low reliability but answers no
  question; competition questions depend on text retrieval, and the graph cannot show how rivalry is reported
  over time.
- **Entity linking is not repaired here.** The check removes relations whose sentence does not state them, but
  family-level linking (*Llama Nemotron* -> Llama) still needs the lexicon filter, and fragment nodes stay in the
  graph. A mention-level linking step would address both at the source.
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
