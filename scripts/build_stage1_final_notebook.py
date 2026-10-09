"""Build the integrated Stage 1 notebook skeleton.

The structure follows the Stage 1 requirements of the project brief. The knowledge-graph part reuses the WP3 cells
(`wp3_notebook_cells.py`) with renumbered headings and neutral wording; the data, topic and extraction parts are
placeholders until the corresponding notebooks are merged in.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import nbformat as nbf

from wp3_notebook_cells import notebook_cells


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "stage1_final.ipynb"

DATA_LEAD = "Daliasny Gonzalez Guzman"
EXTRACTION_LEAD = "Jennifer Spieser"
GRAPH_LEAD = "Jihyeon Choung"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def placeholder(heading: str, source: str, contributor: str | None = None):
    contributor_line = f"*Contributor: {contributor}*\n\n" if contributor else ""
    return markdown(
        f"{heading}\n\n{contributor_line}> **TODO (integration):** move {source} here. Keep the explanations and "
        "results; drop setup, data loading and hand-over cells that the integrated notebook no longer needs."
    )


# Neutral wording for the knowledge-graph cells: no work-package names in the integrated notebook.
REPLACEMENTS = [
    ("WP2_INPUT_DIR", "EXTRACTION_DIR"),
    ('ROOT / "outputs" / "stage1_wp3_knowledge_graph"', 'ROOT / "outputs" / "stage1_knowledge_graph"'),
    ('ROOT / "artifacts" / "stage1_wp3" / "final"', 'ROOT / "artifacts" / "stage1_final"'),
    ("minimum number of articles (Section 7)", "minimum number of articles (Section 7.2)"),
    ("WP2_FILES", "EXTRACTION_FILES"),
    ("WP1_CORE_PATH", "CORE_ARTICLES_PATH"),
    ("# The WP3 pipeline module uses", "# The project pipeline module uses"),
    ("Missing WP2 files", "Missing extraction files"),
    ('("WP2 files present"', '("Extraction files present"'),
    ('"WP2 articles = WP1 core articles"', '"Extracted articles = Open Models core articles"'),
    ('f"{len(wp1_doc_ids):,} WP1 articles"', 'f"{len(wp1_doc_ids):,} core articles"'),
    ('"WP1 table not available"', '"core article table not available"'),
    ('("WP2 schema", "INFO"', '("Extraction schema", "INFO"'),
    ("WP2 used a broad extraction schema", "The extraction used a broad schema"),
    ("90 randomly sampled WP2 relations", "90 randomly sampled extracted relations"),
    ("the WP1 regular expression for", "the lexicon regular expression for"),
    ("of raw WP2 relations", "of raw extracted relations"),
    ("The correction never edits the WP2 files. It maps WP2 IDs", "The correction never edits the saved extraction output. It maps extracted entity IDs"),
    ("developer of a WP1\n  model family", "developer of a lexicon\n  model family"),
    ("# Resolve every WP2 ID", "# Resolve every extracted entity ID"),
    ("WP1 lexicon entities", "Lexicon entities"),
    ("a WP1 model family is dropped", "a lexicon model family is dropped"),
    ("a WP1 model family is `developed/released/announced by` an organisation that is not its WP1 developer",
     "a lexicon model family is `developed/released/announced by` an organisation that is not its lexicon developer"),
    ("| A | WP1 lexicon model family", "| A | lexicon model family"),
    ('("WP2 GLiREL triples"', '("Extracted GLiREL triples"'),
    ('("WP2 suggested filter (type valid, >= 2 sentences)"', '("Baseline filter (type valid, >= 2 sentences)"'),
    ('("Tier A (WP1 lexicon)"', '("Tier A (lexicon)"'),
    ("than the WP2 suggested filter.", "than the baseline filter (type-valid relations stated in at least two sentences)."),
    ("The WP2 schema is broad on purpose (extraction recall). For the graph, the 20 WP2 entity types",
     "The extraction schema is broad on purpose (recall). For the graph, its 20 entity types"),
    ("The original\nWP2 type and relation are kept", "The original\nextracted type and relation are kept"),
    ("(WP1 lexicon families and other models)", "(lexicon families and other models)"),
    ('"curated WP1 lexicon model family', '"curated lexicon model family'),
    ("passing all WP3 checks", "passing all quality checks"),
    ("are the WP2 sentence co-mentions", "are the sentence co-mentions of Section 6.6"),
    ("(WP3 pipeline)", "(project pipeline)"),
    ("(WP2 `nodes_quarterly`)", "(`nodes_quarterly`)"),
    ("split into WP1 lexicon\n   relations", "split into lexicon\n   relations"),
    ("(Section 11, Figure 5)", "(Section 7.6)"),
    ("(Sections 9 and 11)", "(Sections 7.4 and 7.6)"),
    (" (Figure 8).", " (Section 7.7)."),
    ("(Section 5);", "(Section 6.7.3);"),
    ("come from the WP1 lexicon", "come from the lexicon"),
    ("the WP1 lexicon groups versions", "the lexicon groups versions"),
    ("# Stage 1 WP3 knowledge graph: data dictionary", "# Stage 1 knowledge graph: data dictionary"),
    ("the WP2 IDs merged into it", "the extracted IDs merged into it"),
    ("| WP2 entity type and one", "| extracted entity type and one"),
    ("model family from the WP1 lexicon", "model family from the lexicon"),
    ("(original WP2 pairs combined", "(original pairs combined"),
    ("joined from the WP1 table", "joined from the core article table"),
]
HEADINGS = {
    "## 3. Why the WP2 relations need correction": "#### 6.7.1 Why the Extracted Relations Need Correction",
    "## 4. Relation correction": "#### 6.7.2 Relation Correction",
    "### 4.1 Node corrections": "##### Node corrections",
    "### 4.2 Evidence corrections (one row per sentence)": "##### Evidence corrections (one row per sentence)",
    "### 4.3 Relation checks and quality tiers": "##### Relation checks and quality tiers",
    "## 5. Human review of Tier B relations": "#### 6.7.3 Human Review of Tier B Relations",
    "## 6. Compact topic-specific schema": "### 7.1 Compact Topic-Specific Schema",
    "## 7. Graph views": "### 7.2 Graph Views",
    "## 8. Structure of the graph": "### 7.3 Structure of the Graph",
    "## 9. Centrality": "### 7.4 Centrality",
    "## 10. Clustering and communities": "### 7.5 Clustering and Communities",
    "## 11. Temporal analysis": "### 7.6 Temporal Analysis",
    "## 12. Graph visualisations": "### 7.7 Graph Visualisations",
    "## 13. Key findings with evidence": "### 8.2 Knowledge Graph Findings with Evidence",
    "## 14. Limitations": "### 10.2 Extraction and Knowledge Graph",
    "## 15. Export and re-load check": "### 9.1 Export and Re-load Check",
    "## 16. Stage 3 retrieval check": "### 9.2 Stage 3 Retrieval Check",
}


def graph_cells() -> dict[str, list]:
    """Split the WP3 cells into the blocks used by the integrated notebook and neutralise their wording."""
    cells = [copy.deepcopy(cell) for cell in notebook_cells()]
    for cell in cells:
        for old, new in REPLACEMENTS:
            cell.source = cell.source.replace(old, new)
        for old, new in HEADINGS.items():
            cell.source = re.sub(rf"^{re.escape(old)}$", new, cell.source, flags=re.M)

    def start(prefix: str) -> int:
        matches = [i for i, cell in enumerate(cells) if cell.source.startswith(prefix)]
        assert len(matches) == 1, prefix
        return matches[0]

    blocks = {
        "setup": cells[start("## 1. Setup") + 1: start("## 1. Setup") + 2],
        "load": cells[start("## 2. WP2 handover") + 1: start("#### 6.7.1")],
        "quality": cells[start("#### 6.7.1"): start("### 7.1")],
        "graph": cells[start("### 7.1"): start("### 8.2")],
        "findings": cells[start("### 8.2"): start("### 10.2")],
        "limitations": cells[start("### 10.2"): start("### 9.1")],
        "storage": cells[start("### 9.1"):],
    }
    leftover = re.compile(r"\bWP[123]\b|Section [0-9]+\b(?!\.)|handover|^#+ [0-9]+\. ", re.M)
    for name, block in blocks.items():
        for cell in block:
            assert not leftover.search(cell.source), (name, leftover.search(cell.source).group(), cell.source[:80])
    return blocks


def build() -> list:
    g = graph_cells()
    return [
        markdown(
            f"""
# Tracking Open Models in AI Media · Stage 1: Data, Topics and Knowledge Graph

**Track: Open models.** This notebook covers Stage 1 of the project: it cleans and explores the AI Media Dataset
(September 2024 to August 2026), models the main topics, extracts entities and relations about open models, and
builds a topic-focused knowledge graph that is reused in Stage 3.

> **TODO (integration):** add the track definition and motivation from the data and topic analysis notebook (Section 1).

### Stage 1 requirements and where they are addressed

| Requirement (project brief) | Section |
|---|---|
| Load the dataset; remove duplicates, handle missing values, normalise text | 2 |
| Tokenisation, lemmatisation, stop-word removal; consistent acronyms and abbreviations | 3 |
| Temporal analysis, sentiment or stance, visualisations | 4 |
| Topic modelling and topic dynamics | 5 |
| Pre-trained NER and structured extraction of entities and relations | 6.1–6.5 |
| Co-occurrence matrix | 6.6 |
| Compact topic-specific schema defined before graph construction | 7.1 |
| Graph of entities with co-mention and relation edges | 7.2 |
| Network metrics (centrality, clustering) for influential entities and emerging relationships | 7.3–7.6 |
| Graph visualisation | 7.7 |
| Key insights and notable developments | 8 |
| Graph and schema stored for Stage 3 | 9 |

### Contributions

The work was split into three consecutive parts, each building on the previous one.

| Team member | Sections | Contribution |
|---|---|---|
| {DATA_LEAD} | 2–5, 8.1 | data cleaning and preprocessing, Open Models lexicon, exploratory analysis, sentiment and framing, topic modelling |
| {EXTRACTION_LEAD} | 6.1–6.6 | entity extraction (spaCy, GLiNER), entity linking and evaluation, relation extraction (GLiREL), co-occurrence matrix |
| {GRAPH_LEAD} | 6.7, 7, 8.2–8.3, 9 | relation quality evaluation and correction, knowledge graph construction, analysis, visualisation and storage |
| All members | 6.7.3 | human review of 90 relations |
"""
        ),
        markdown(
            """
## 1. Setup

> **TODO (integration):** merge the configuration cells of the data and extraction notebooks into this cell
> (dataset path, Google Drive paths, model versions, `RUN_EXTRACTION` flag for the slow NER and relation models).

The knowledge-graph part reads the saved extraction output and writes its results to `outputs/` (large files)
and `artifacts/stage1_final/` (schema, data dictionary, summary tables and figures).
"""
        ),
        *g["setup"],
        code(
            """
WP1_FAMILY_TREND_PATH = ROOT / "artifacts" / "stage1_wp1" / "tables" / "08_2_family_monthly_per_1000_docs.csv"
"""
        ),
        placeholder("## 2. Data Collection and Cleaning",
                    "the data and topic analysis notebook, Sections 3 (raw data), 4 (cleaning) and 5 (corpus diagnostics)",
                    DATA_LEAD),
        placeholder("## 3. Text Preprocessing",
                    "the data and topic analysis notebook, Sections 6 (text representations), 7 (Open Models lexicon, "
                    "matching and relevance tiers) and 9 (preprocessing for topic modelling)",
                    DATA_LEAD),
        placeholder("## 4. Exploratory Data Analysis",
                    "the data and topic analysis notebook, Sections 8.1–8.5 (track share, model families, regions, "
                    "release radar, terminology) and 10 (sentiment and framing)",
                    DATA_LEAD),
        placeholder("## 5. Topic Modelling",
                    "the data and topic analysis notebook, Section 11 (topic modelling) and 11.1 (topic dynamics)",
                    DATA_LEAD),
        markdown(
            f"""
## 6. Entity and Relation Extraction

*Contributors: {EXTRACTION_LEAD} (6.1–6.6), {GRAPH_LEAD} (6.7)*

Entities are extracted with a lexicon, pre-trained NER (spaCy) and schema-driven NER (GLiNER), linked to canonical
entities, and connected by relations extracted with GLiREL. Section 6.7 evaluates the extracted relations and
corrects them before they are used in the knowledge graph.
"""
        ),
        placeholder("### 6.1 Sentence Segmentation and Pre-trained NER",
                    "the entity and relation extraction notebook, section 'Sentence Segmentation and spaCy NER'"),
        placeholder("### 6.2 Structured Entity Extraction",
                    "the entity and relation extraction notebook, sections 'Entity Extraction' (domain dictionary, "
                    "spaCy, GLiNER)"),
        placeholder("### 6.3 Entity Deduplication and Linking",
                    "the entity and relation extraction notebook, sections 'Entity Deduplication' and "
                    "'Entity Normalisation and Linking'"),
        placeholder("### 6.4 Evaluation of the Entity Extraction",
                    "the entity and relation extraction notebook, section 'Evaluation' (Evaluations 1–3 and conclusion)"),
        placeholder("### 6.5 Relation Extraction",
                    "the entity and relation extraction notebook, section 'Relation Extraction' (schema and GLiREL run)"),
        placeholder("### 6.6 Co-occurrence Matrix",
                    "the entity and relation extraction notebook, section 'Co-occurrence Matrix', and the cell that "
                    "saves nodes, relations with evidence and co-occurrence pairs (used in Section 6.7)"),
        markdown(
            """
### 6.7 Relation Quality and Refinement

The extracted relations were checked before graph construction. This section loads the saved extraction output,
checks that it is complete and consistent, explains the main error sources, corrects them with explicit rules and
applies a human review of the corrected relations.
"""
        ),
        *g["load"],
        *g["quality"],
        markdown(
            f"""
## 7. Knowledge Graph Construction

*Contributor: {GRAPH_LEAD}*

The corrected relations and the co-occurrence pairs are turned into a topic-focused knowledge graph: a compact
schema (7.1), three graph views (7.2), structural statistics (7.3), centrality (7.4), communities (7.5),
temporal analysis (7.6) and visualisations (7.7).
"""
        ),
        *g["graph"],
        markdown(
            """
## 8. Key Insights

This section brings together the trends and topics of the corpus (8.1), the findings of the knowledge graph (8.2)
and a consistency check between the two (8.3).
"""
        ),
        placeholder("### 8.1 Open Models Trends and Topics",
                    "the key insights of the data and topic analysis notebook (Section 12)"),
        *g["findings"],
        markdown(
            """
### 8.3 Consistency of Lexicon Trends and Knowledge-Graph Trends

The model-family trend of the exploratory analysis (lexicon matches per 1,000 AI articles, all articles) and the
family share in the knowledge graph (share of Open Models articles that mention the family as an entity) are
measured on different article sets with different methods. If both show the same direction over time, the
knowledge graph reflects the corpus trends rather than extraction artefacts.
"""
        ),
        code(
            r"""
wp1_family = pd.read_csv(WP1_FAMILY_TREND_PATH)
wp1_family["quarter"] = pd.PeriodIndex(wp1_family["key"], freq="M").asfreq("Q").astype(str)
wp1_quarterly = wp1_family.drop(columns="key").groupby("quarter").mean()

consistency = []
for family in family_share.index:
    if family not in wp1_quarterly.columns:
        continue
    graph_trend = family_share.loc[family]
    lexicon_trend = wp1_quarterly[family].reindex(graph_trend.index)
    consistency.append({
        "family": family,
        "graph share first / last quarter": f"{graph_trend.iloc[0]:.0%} / {graph_trend.iloc[-1]:.0%}",
        "lexicon per 1,000 first / last quarter": f"{lexicon_trend.iloc[0]:.1f} / {lexicon_trend.iloc[-1]:.1f}",
        "same direction": np.sign(graph_trend.iloc[-1] - graph_trend.iloc[0]) == np.sign(lexicon_trend.iloc[-1] - lexicon_trend.iloc[0]),
        "Spearman correlation": round(graph_trend.corr(lexicon_trend, method="spearman"), 2),
    })
consistency = pd.DataFrame(consistency)
display(consistency)
same_direction = consistency[consistency["same direction"]]
correlated = consistency[consistency["Spearman correlation"] >= 0.5]
shift = ["Llama", "Qwen", "Kimi", "GLM"]
shift_text = ("The shift from Llama towards Qwen, Kimi and GLM (Section 8.2) appears in both measurements."
              if set(shift) <= set(same_direction["family"]) else
              "The two measurements disagree for some of the families behind the shift described in Section 8.2.")
display(Markdown(f'''
{len(same_direction)} of {len(consistency)} families change in the same direction between the first and the last quarter
in both measurements, and {len(correlated)} have a Spearman correlation of at least 0.5 between the quarterly trends
({", ".join(correlated["family"])}). {shift_text} Lower correlations come from quarter-to-quarter fluctuations, which
differ because the two measures use different article sets.
'''))
"""
        ),
        placeholder("### 8.4 Summary",
                    "a short summary of the most important insights from 8.1–8.3 (5–6 bullet points)"),
        markdown(
            f"""
## 9. Storage and Reuse for Stage 3

*Contributor: {GRAPH_LEAD}*

The graph, its schema and a data dictionary are stored so that the graph-enhanced RAG system of Stage 3 can load
them, and the retrieval path from a question to evidence sentences is checked.
"""
        ),
        *g["storage"],
        markdown(
            """
## 10. Limitations
"""
        ),
        placeholder("### 10.1 Data, Preprocessing and Topics",
                    "the limitations of the data and topic analysis notebook (Section 12) and of the extraction "
                    "notebook (conclusion)"),
        *g["limitations"],
        markdown(
            """
## 11. AI Tool Disclosure

> **TODO (integration):** complete with every AI tool used by the team (tool name and how it was used).

- **Claude Code (Anthropic)**: AI-assisted pre-screening of a sample of extracted relations (Section 6.7.1; not
  used as the official evaluation, which is the human review in Section 6.7.3), drafting of the relation-correction
  rules and of the code and explanations of Sections 6.7–9. All numbers come from running this notebook; the
  interpretations were reviewed by the team.
"""
        ),
    ]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
