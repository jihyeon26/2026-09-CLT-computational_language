"""Build the integrated Stage 1 notebook (data and topics, entity and relation extraction, knowledge graph).

The structure follows the Stage 1 section of the project brief. Cells of the data and topic analysis notebook and
of the extraction notebook are copied from the team repository together with their saved outputs (the extraction
needs a GPU and runs on Colab). The knowledge-graph cells come from `wp3_notebook_cells.py`; they are executed
here and their outputs are written into the integrated notebook.

    python scripts/build_stage1_final_notebook.py [--team] [--no-execute]
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import nbformat as nbf

from wp3_notebook_cells import notebook_cells as graph_notebook_cells

ROOT = Path(__file__).resolve().parents[1]
TEAM_REPO = ROOT.parents[1] / "CLT" / "computational-language-technology-hs26"
DATA_NOTEBOOK = TEAM_REPO / "stage1" / "work_packages" / "stage1_wp1_data_topic_analysis.ipynb"
EXTRACTION_NOTEBOOK = TEAM_REPO / "stage1" / "work_packages" / "stage1_wp2_entity_relation_extraction.ipynb"
OUTPUT = ROOT / "notebooks" / "stage1_final.ipynb"   # --team writes TEAM_REPO / "stage1" / "stage1_final.ipynb"
REPO_URL = "https://github.com/jihyeon26/computational-language-technology-hs26"
RAW_SHA256 = "82dcea5ee2b15faf805994043fa7e8afb37b82fe8df7ab4fba8873db386f05b9"

DATA_LEAD = "Daliasny Gonzalez Guzman"
EXTRACTION_LEAD = "Jennifer Spieser"
GRAPH_LEAD = "Jihyeon Choung"
GRAPH_TAG = "knowledge-graph"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str, tag: str | None = None):
    cell = nbf.v4.new_code_cell(text.strip())
    if tag:
        cell.metadata["tags"] = [tag]
    return cell


def hide(cell, title: str):
    """Collapse the code of a plumbing cell (Colab form view); its output stays visible."""
    cell = copy.deepcopy(cell)
    cell.source = f"#@title {title}\n" + cell.source
    cell.metadata.update({"cellView": "form", "jupyter": {"source_hidden": True}})
    return cell


def load_cells(path: Path) -> dict:
    cells = nbf.read(path, as_version=4).cells
    for cell in cells:
        cell.pop("id", None)
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.execution_count = None
            for output in cell.outputs:
                if "execution_count" in output:
                    output["execution_count"] = None
    return dict(enumerate(cells))


def edit(cell, replacements=(), headings=None):
    cell = copy.deepcopy(cell)
    for old, new in replacements:
        assert old in cell.source, f"not found: {old[:70]!r}"
        cell.source = cell.source.replace(old, new)
    for old, new in (headings or {}).items():
        assert re.search(rf"^{re.escape(old)}$", cell.source, flags=re.M), f"heading not found: {old!r}"
        cell.source = re.sub(rf"^{re.escape(old)}\n?", f"{new}\n" if new else "", cell.source, flags=re.M)
    cell.source = cell.source.strip()
    return cell


# ---------------------------------------------------------------- data and topic analysis
DATA_SECTIONS = {"1": "1", "2": "2.1", "3": "2.2", "4": "2.3", "5": "2.4", "6": "3.1", "7": "3.2", "8": "4",
                 "9": "3.3", "10": "4.6", "11": "7", "12": "8.1", "13": "7.2", "11.1": "7.1"}
DATA_SECTIONS.update({f"4.{i}": f"2.3.{i}" for i in range(1, 7)})
DATA_SECTIONS.update({f"7.{i}": f"3.2.{i}" for i in range(1, 4)})
DATA_SECTIONS.update({f"8.{i}": f"4.{i}" for i in range(1, 6)})


def renumber_data_references(text: str) -> str:
    def repl(match):
        numbers = re.sub(r"\d+(?:\.\d+)?", lambda m: DATA_SECTIONS.get(m.group(), m.group()), match.group(2))
        return match.group(1) + numbers
    return re.sub(r"(Sections? )(\d+(?:\.\d+)?(?:(?:–|-|, | and )\d+(?:\.\d+)?)*)", repl, text)


def data_cells() -> dict:
    cells = load_cells(DATA_NOTEBOOK)
    for cell in cells.values():
        if cell.cell_type == "markdown":
            cell.source = renumber_data_references(cell.source)
    fixes = {
        1: ([("**WP1 objective.**", "**Objective of the data and topic analysis.**")],
            {"## 1. Objective and track definition": "## 1. Objective and Track Definition"}),
        2: ([], {"## 2. Setup and configuration": "### 2.1 Setup and Configuration"}),
        7: ([("which matters because WP2, WP3 and Stage 3 will refer to documents by this ID",
              "which matters because the extraction, the knowledge graph and Stage 3 refer to documents by this ID")],
            {"## 3. Raw data: loading and first inspection": "### 2.2 Raw Data: Loading and First Inspection"}),
        13: ([("NER and relation extraction in WP2 depend on", "NER and relation extraction (Section 5) depend on")]
             + [(f"| 4.{i} ", f"| 2.3.{i} ") for i in range(1, 7)],
             {"## 4. Cleaning": "### 2.3 Cleaning", "### 4.1 Boilerplate removal": "#### 2.3.1 Boilerplate removal"}),
        15: ([], {"### 4.2 Text repair": "#### 2.3.2 Text repair"}),
        17: ([], {"### 4.3 Minimum content after cleaning": "#### 2.3.3 Minimum content after cleaning"}),
        19: ([], {"### 4.4 Language": "#### 2.3.4 Language"}),
        21: ([], {"### 4.5 Exact and near-duplicates": "#### 2.3.5 Exact and near-duplicates"}),
        23: ([], {"### 4.6 Study window and cleaned corpus": "#### 2.3.6 Study window and cleaned corpus"}),
        26: ([], {"## 5. Corpus diagnostics": "### 2.4 Corpus Diagnostics"}),
        30: ([("NER and relation extraction (WP2)", "NER and relation extraction (Section 5)")],
             {"## 6. Text representations": "### 3.1 Text Representations"}),
        32: ([("WP2 can reuse it as a gazetteer", "The entity extraction (Section 5.2) reuses it as a gazetteer")],
             {"## 7. Open-models track: lexicon, matching and relevance tiers":
                  "### 3.2 Open-Models Track: Lexicon, Matching and Relevance Tiers",
              "### 7.1 A gazetteer of open-model families": "#### 3.2.1 A gazetteer of open-model families"}),
        34: ([], {"### 7.2 Matching and relevance score": "#### 3.2.2 Matching and relevance score"}),
        37: ([], {"### 7.3 Validation of the relevance tiers": "#### 3.2.3 Validation of the relevance tiers"}),
        40: ([], {"## 8. Exploratory analysis of the open-models track": f"## 4. Exploratory Data Analysis\n\n*Contributor: {DATA_LEAD}*\n",
                  "### 8.1 How much of the AI discourse is about open models? (Q1)":
                      "### 4.1 How much of the AI discourse is about open models? (Q1)"}),
        43: ([], {"### 8.2 Which model families get attention? (Q2)": "### 4.2 Which model families get attention? (Q2)"}),
        47: ([], {"### 8.3 United States, China, Europe: the regional balance (Q2)":
                      "### 4.3 United States, China, Europe: the regional balance (Q2)"}),
        50: ([("for WP2/WP3.", "for the entity extraction and the knowledge graph.")],
             {"### 8.4 Release radar: when do model versions first appear? (Q3)":
                  "### 4.4 Release radar: when do model versions first appear? (Q3)"}),
        52: ([("WP2 should extract event dates from the text, and treat a document's date as an upper bound only for news articles.",
               "Event dates therefore have to come from the text, and a document's date is an upper bound only for news articles.")], {}),
        53: ([], {"### 8.5 Words for openness: *open-weight* vs *open-source* (Q5)":
                      "### 4.5 Words for openness: *open-weight* vs *open-source* (Q5)"}),
        56: ([], {"## 9. Preprocessing for topic modelling (analysis view)":
                      "### 3.3 Tokenisation, Lemmatisation and Stop Words: Preprocessing for Topic Modelling"}),
        58: ([("and WP2 can re-segment with a parser if needed.",
               "and the entity extraction (Section 5.1) re-segments the texts with a parser.")],
             {"## 10. Sentiment and framing (Q5)": "### 4.6 Sentiment and Framing (Q5)"}),
        64: ([], {"## 11. Topic modelling (Q4)": f"## 7. Topic Modelling (Q4)\n\n*Contributor: {DATA_LEAD}*\n"}),
        69: ([], {"### 11.1 Topic dynamics": "### 7.1 Topic Dynamics"}),
        73: ([], {"## 13. Outputs and hand-over": "### 7.2 Saved Corpus, Lexicon and Topic Outputs"}),
    }
    for index, (replacements, headings) in fixes.items():
        cells[index] = edit(cells[index], replacements, headings)
    insights, limitations = cells[72].source.split("**Limitations**")
    cells[72] = markdown(insights.replace("## 12. Key insights and limitations",
                                          "### 8.1 Open Models in the Corpus: Trends, Framing and Topics"))
    cells["limitations"] = markdown("**Data, preprocessing, sentiment and topics**\n" + limitations.strip())
    return cells


# ---------------------------------------------------------------- entity and relation extraction
EXTRACTION_REPLACEMENTS = [
    ("For the knowledge graph handover, all pairs are also saved", "For the knowledge graph, all pairs are also saved"),
    ("preparation of the handover exports.", "preparation of the saved extraction output."),
    ("WP1 normalisation  |", "Lexicon normalisation |"),
    ("| WP1 lexicon (`phrase_normalisation`) |", "| lexicon (`phrase_normalisation`) |"),
    ("Only general rules and rules inherited from WP1 are used:",
     "Only general rules and rules from the lexicon (Section 3.2) are used:"),
    ("while the same normalisation rules used in WP1 are applied to GLiNER and spaCy entities. In WP1, these rules were applied only to the `lemmas` column used for topic modelling.",
     "while the lexicon normalisation rules are applied to GLiNER and spaCy entities. In Section 3.3, these rules were applied only to the `lemmas` column used for topic modelling."),
    ("to ensure consistent entity representation across work packages.",
     "to ensure a consistent entity representation throughout the notebook."),
    ("additionally retain their WP1 `canonical_id`", "additionally retain their lexicon `canonical_id`"),
    ("are created during the handover, when entities become nodes",
     "are created when the extraction output is saved (Section 5.8), when entities become nodes"),
    ("As in WP1, patterns are applied", "As in Section 3.2, patterns are applied"),
    ("The WP1 lexicon was built and validated independently of this extraction stage.",
     "The lexicon was built and validated independently of this extraction stage (Section 3.2)."),
    ("allowing the knowledge-graph work package to merge or remove relation types if needed.",
     "allowing the relation refinement (Section 5.9) to merge or remove relation types if needed."),
    ("8. **Handover**: the resulting nodes, edges, schema and data dictionary are prepared for the Knowledge Graph work package",
     "8. **Saving**: the resulting nodes, edges, schema and data dictionary are saved for the knowledge graph (Section 5.8)"),
]
EXTRACTION_HEADINGS = {
    5: {"## Entity and Relation Extraction": "",
        "### Sentence Segmentation and spaCy NER": "### 5.1 Sentence Segmentation and Pre-trained NER (spaCy)"},
    9: {"### Entity Extraction": "### 5.2 Entity Extraction", "####\xa0Domain Dictionary Extraction": "#### 5.2.1 Domain dictionary"},
    14: {"#### SpaCy Extraction": "#### 5.2.2 spaCy"},
    16: {"#### Structured Extraction with GLiNER": "#### 5.2.3 Structured extraction with GLiNER"},
    22: {"## Entity Deduplication": "### 5.3 Entity Deduplication"},
    25: {"### Entity Normalisation and Linking": "### 5.4 Entity Normalisation and Linking"},
    28: {"## Evaluation: does structured extraction improve entity extraction ?":
             "### 5.5 Evaluation: Does Structured Extraction Improve Entity Extraction?",
         "### Evaluation 1 - do the models detect and correctly type known domain entities ?":
             "#### Evaluation 1: do the models detect and correctly type known domain entities?"},
    30: {"### Evaluation 2 - how does GLiNER change spaCy's entity labels ?":
             "#### Evaluation 2: how does GLiNER change spaCy's entity labels?"},
    32: {"### Evaluation 3 - what does GLiNER add to entity coverage ?": "#### Evaluation 3: what does GLiNER add to entity coverage?"},
    34: {"### Conclusion of the evaluation": "#### Conclusion of the evaluation"},
    35: {"## Relation Extraction": "### 5.6 Relation Extraction",
         "### Relation schema of the open-models tracks": "#### Relation schema of the open-models track"},
    39: {"### Running GLiREL": "#### Running GLiREL"},
    44: {"# Co-occurrence Matrix": "### 5.7 Co-occurrence Matrix"},
}


def extraction_cells() -> dict:
    cells = load_cells(EXTRACTION_NOTEBOOK)
    for index, cell in cells.items():
        if cell.cell_type != "markdown":
            continue
        for old, new in EXTRACTION_REPLACEMENTS:
            cell.source = cell.source.replace(old, new)
        cell.source = re.sub(r"\b(the )?WP1 lexicon\b", lambda m: (m.group(1) or "") + "lexicon", cell.source)
        if index in EXTRACTION_HEADINGS:
            cells[index] = edit(cell, headings=EXTRACTION_HEADINGS[index])
    conclusion, disclosure = cells[49].source.split("**AI Assistance Disclosure**")
    cells[49] = edit(markdown(conclusion), [
        ("WP2 transforms the WP1 corpus", "The extraction transforms the core corpus"),
        ("Combining the WP1 dictionary", "Combining the lexicon dictionary"),
        ("during the knowledge graph construction in WP3.", "in the relation refinement and the knowledge graph (Sections 5.9 and 6)."),
    ], {"## Conclusion": "#### Summary of the extraction"})
    cells["disclosure"] = disclosure.strip().replace("this work package", "the entity and relation extraction")
    cells[50] = edit(cells[50], [
        ("Everything the next work package (WP3) needs to build, analyse and store the knowledge graph is exported",
         "Everything needed to build, analyse and store the knowledge graph is exported"),
    ], {"## Handover to the Knowledge Graph work package": "### 5.8 Saving the Extraction Output"})
    return cells


# ---------------------------------------------------------------- knowledge graph
GRAPH_SECTIONS = {"4": "5.9.3", "5": "5.9.4", "6": "5.10", "7": "6.1", "8": "6.2", "9": "6.3", "10": "6.4",
                  "11": "6.5", "12": "6.6", "13": "6.7", "14": "8.2", "15": "9", "16": "6.8", "17": "6.9"}
GRAPH_HEADINGS = {
    "## 2. WP2 handover: load and validate": "#### 5.9.1 Loading and Validating the Extraction Output",
    "## 3. From extracted relations to a knowledge graph": "#### 5.9.2 From Extracted Relations to a Knowledge Graph",
    "## 4. Relation refinement": "#### 5.9.3 Refinement Rules",
    "### 4.1 Nodes": "##### Nodes",
    "### 4.2 Evidence sentences": "##### Evidence sentences",
    "### 4.3 Relation checks and quality tiers": "##### Relation checks and quality tiers",
    "## 5. Model-level relations": "#### 5.9.4 Model-Level Relations",
    "## 6. Human review and error-driven refinement": "### 5.10 Human Review and Error-Driven Refinement",
    "## 7. Compact topic-specific schema": "### 6.1 Compact Topic-Specific Schema",
    "## 8. Graph views": "### 6.2 Graph Views",
    "## 9. Structure of the graph": "### 6.3 Structure of the Graph",
    "## 10. Centrality": "### 6.4 Centrality",
    "## 11. Clustering and communities": "### 6.5 Clustering and Communities",
    "## 12. Temporal analysis": "### 6.6 Temporal Analysis",
    "## 13. Graph visualisations": "### 6.7 Graph Visualisations",
    "## 14. Key findings with evidence": "### 8.2 Knowledge Graph Findings with Evidence",
    "## 15. Limitations": "**Relation refinement and knowledge graph**",
    "## 16. Export and re-load check": "### 6.8 Storage and Re-load Check",
    "## 17. Stage 3 retrieval check": "### 6.9 Stage 3 Retrieval Check",
}
GRAPH_REPLACEMENTS = [
    # identifiers and paths
    ("WP2_INPUT_DIR", "EXTRACTION_DIR"), ("WP2_FILES", "EXTRACTION_FILES"), ("WP1_CORE_PATH", "CORE_ARTICLES_PATH"),
    ('ROOT / "outputs" / "stage1_wp3_knowledge_graph"', 'ROOT / "outputs" / "stage1_knowledge_graph"'),
    ('ROOT / "artifacts" / "stage1_wp3" / "final"', 'ROOT / "artifacts" / "stage1_final"'),
    ("# The WP3 pipeline module uses", "# The project pipeline module uses"),
    # read the extraction output written in Section 5.8 first
    ('first_existing(ROOT / "artifacts" / "stage1_wp2" / "nodes.csv",',
     'first_existing(ROOT / "data" / "stage1_wp2" / "handover" / "nodes.csv", ROOT / "artifacts" / "stage1_wp2" / "nodes.csv",'),
    ('print(f"WP2 handover:', 'print(f"Extraction output:'),
    # names shared with the data and topic analysis cells
    ('core = relations[relations["tier"].isin(["A", "B"])].copy()', 'tier_ab = relations[relations["tier"].isin(["A", "B"])].copy()'),
    ('core["_order"] = core["tier"]', 'tier_ab["_order"] = tier_ab["tier"]'),
    ("graph_candidates = core.sort_values", "graph_candidates = tier_ab.sort_values"),
    ("support = (core.groupby", "support = (tier_ab.groupby"),
    ("DATA_DICTIONARY", "KG_DATA_DICTIONARY"),
    # validation
    ("Before building the graph, the handover must be complete and internally linked:",
     "Before building the graph, the saved extraction output must be complete and internally linked:"),
    ("the WP2 articles are exactly the WP1 Open Models core articles (checked when the WP1 table is available).",
     "the extracted articles are exactly the core Open Models articles (checked when the core article table is available)."),
    ("Missing WP2 files", "Missing extraction files"), ('("WP2 files present"', '("Extraction files present"'),
    ('"WP2 articles = WP1 core articles"', '"Extracted articles = core articles"'),
    ('f"{len(wp1_doc_ids):,} WP1 articles"', 'f"{len(wp1_doc_ids):,} core articles"'),
    ('"WP1 table not available"', '"core article table not available"'), ('("WP2 schema", "INFO"', '("Extraction schema", "INFO"'),
    # refinement
    ("WP2 is designed for recall:", "The extraction (Sections 5.1–5.7) is designed for recall:"),
    ("so WP3 adds a refinement step on top of the WP2\noutput. The WP2 files themselves are not changed.",
     "so a refinement step is added on top of\nthe saved extraction output, which itself is not changed."),
    ("The refinement maps WP2 IDs to corrected nodes", "The refinement maps extracted entity IDs to corrected nodes"),
    ("acceptable as the developer of a WP1\n  model family", "acceptable as the developer of a lexicon\n  model family"),
    ("# Resolve every WP2 ID", "# Resolve every extracted entity ID"),
    ("WP1 lexicon entities and named normalisations", "Lexicon entities and named normalisations"),
    ("a WP1 model family is not linked", "a lexicon model family is not linked"),
    ("a WP1 model family is `developed/released/announced by` an organisation that is not its WP1 developer",
     "a lexicon model family is `developed/released/announced by` an organisation that is not its lexicon developer"),
    ("| A | WP1 lexicon model family", "| A | lexicon model family"),
    ('("WP2 GLiREL triples"', '("Extracted GLiREL triples"'),
    ('("WP2 suggested filter (type valid, >= 2 sentences)"', '("Baseline filter (type valid, >= 2 sentences)"'),
    ('("Tier A (WP1 lexicon)"', '("Tier A (lexicon)"'),
    ("The graph uses fewer relations than the WP2 suggested filter.",
     "The graph uses fewer relations than the baseline filter (type-valid relations stated in at least two sentences)."),
    ("both computed in WP3 from the WP2 output:", "both computed from the saved extraction output:"),
    ("| A WP1 family node that shares", "| A lexicon family node that shares"),
    ("| The WP1 lexicon groups versions", "| The lexicon groups versions"),
    ("# A WP1 family node in a model slot", "# A lexicon family node in a model slot"),
    ("a WP1 family node in a model slot", "a lexicon family node in a model slot"),
    ("; Section 12, Figure 5)", "; Section 6.6)"),
    # schema and graph
    ("The WP2 schema is broad on purpose (extraction recall). For the graph, the 20 WP2 entity types",
     "The extraction schema is broad on purpose (recall). For the graph, its 20 entity types"),
    ("The original\nWP2 type and relation are kept", "The original\nextracted type and relation are kept"),
    ("(WP1 lexicon families and other models)", "(lexicon families and other models)"),
    ('"curated WP1 lexicon model family', '"curated lexicon model family'),
    ("passing all WP3 checks", "passing all quality checks"),
    ("are the WP2 sentence co-mentions", "are the sentence co-mentions of Section 5.7"),
    ("(WP3 pipeline)", "(project pipeline)"),
    ("(WP2 `entity_mentions`, one count", "(`entity_mentions`, one count"),
    ("dashed: WP1 lexicon only", "dashed: lexicon only"),
    ("from the curated WP1 lexicon", "from the curated lexicon"),
    ("The WP1 lexicon groups versions", "The lexicon groups versions"),
    ("WP1 families such as `DeepSeek`", "Lexicon families such as `DeepSeek`"),
    # storage comes before the findings in the integrated notebook: the findings table is saved in Section 8.2
    ('    "findings": findings,\n', ""),
    ('"community_summary", "findings"]:', '"community_summary"]:'),
    ("findings = pd.DataFrame(findings)\n",
     "findings = pd.DataFrame(findings)\nfindings.to_parquet(OUTPUT_DIR / \"findings.parquet\", index=False)\n"
     "for directory in [OUTPUT_DIR, ARTIFACT_DIR]:\n"
     "    findings.to_csv(directory / \"findings.csv\", index=False, encoding=\"utf-8-sig\")\n"),
    # data dictionary
    ("# Stage 1 WP3 knowledge graph: data dictionary", "# Stage 1 knowledge graph: data dictionary"),
    ("the WP2 IDs merged into it", "the extracted IDs merged into it"),
    ("| WP2 entity type and one", "| extracted entity type and one"),
    ("model family from the WP1 lexicon", "model family from the lexicon"),
    ("(original WP2 pairs combined", "(original pairs combined"),
    ("joined from the WP1 table", "joined from the core article table"),
    # the saved core article table is written only in Section 7.2: use the in-memory table of Section 3.2
    ('if CORE_ARTICLES_PATH.exists():\n    wp1_doc_ids = set(pd.read_parquet(CORE_ARTICLES_PATH, columns=["doc_id"])["doc_id"])',
     '# core articles of Section 3.2 (in memory; the saved table is only written in Section 7.2)\n'
     'core_articles = (core[["doc_id", "title", "url"]] if "core" in globals() else\n'
     '                 pd.read_parquet(CORE_ARTICLES_PATH, columns=["doc_id", "title", "url"]) if CORE_ARTICLES_PATH.exists() else None)\n'
     'if core_articles is not None:\n    wp1_doc_ids = set(core_articles["doc_id"])'),
    ('if CORE_ARTICLES_PATH.exists():\n    documents = documents.merge(pd.read_parquet(CORE_ARTICLES_PATH, columns=["doc_id", "title", "url"])\n'
     '                                .rename(columns={"doc_id": "document_id"}), on="document_id", how="left")',
     'if core_articles is not None:\n    documents = documents.merge(core_articles.rename(columns={"doc_id": "document_id"}), on="document_id", how="left")'),
]


def graph_cells() -> dict[str, list]:
    cells = [copy.deepcopy(cell) for cell in graph_notebook_cells()]
    for cell in cells:
        for old, new in GRAPH_REPLACEMENTS:
            cell.source = cell.source.replace(old, new)
        if "cooccurrence_wp2.assign" in cell.source:
            cell.source = re.sub(r"\bco\b", "cooc_pairs", cell.source)
        cell.source = re.sub(r"\(Section (\d+), Figure \d+\)", lambda m: f"(Section {GRAPH_SECTIONS[m.group(1)]})", cell.source)
        cell.source = cell.source.replace("(Figure 8)", "(Section 6.7)")
        cell.source = re.sub(r"(Sections? )(\d+)\b(?!\.\d)", lambda m: m.group(1) + GRAPH_SECTIONS.get(m.group(2), m.group(2)),
                             cell.source)
        for old, new in GRAPH_HEADINGS.items():
            cell.source = re.sub(rf"^{re.escape(old)}$", new, cell.source, flags=re.M)
        if cell.cell_type == "code":
            cell.metadata["tags"] = [GRAPH_TAG]

    def start(prefix: str) -> int:
        matches = [i for i, cell in enumerate(cells) if cell.source.startswith(prefix)]
        assert len(matches) == 1, prefix
        return matches[0]

    blocks = {
        "setup": [cells[start("## 1. Setup") + 1]],
        "refinement": cells[start("#### 5.9.1"): start("### 5.10")],
        "review": cells[start("### 5.10"): start("### 6.1")],
        "graph": cells[start("### 6.1"): start("### 8.2")],
        "findings": cells[start("### 8.2"): start("**Relation refinement and knowledge graph**")],
        "limitations": cells[start("**Relation refinement and knowledge graph**"): start("### 6.8")],
        "storage": cells[start("### 6.8"): start("## 18.")],
    }
    leftover = re.compile(r"\bWP ?[123]\b|handover|^#+ [0-9]+\. |Figure \d", re.M)
    for name, block in blocks.items():
        for cell in block:
            match = leftover.search(cell.source.replace('"handover"', ""))  # folder name written in Section 5.8
            assert not match, (name, match.group(), cell.source[:80])
    return blocks


# ---------------------------------------------------------------- integrated notebook
def build() -> list:
    d, x, g = data_cells(), extraction_cells(), graph_cells()
    header = markdown(
        f"""
# Tracking Open Models in AI Media · Stage 1: Data, Entities, Knowledge Graph and Topics

**Track: Open models.** This notebook covers Stage 1 of the project. It cleans and explores the AI Media Dataset
(September 2024 to August 2026), extracts entities and relations about open models, builds a topic-focused
knowledge graph that is reused in Stage 3, and models the main themes of the open-model discourse.

### Stage 1 requirements and where they are addressed

| Requirement (project brief) | Section |
|---|---|
| Download and load the dataset; remove duplicates, handle missing values, normalise text | 2 |
| Tokenisation, lemmatisation and stop-word removal; consistent acronyms and abbreviations | 3 |
| Temporal analysis, sentiment or stance, visualisations | 4 (and 6.6) |
| Pre-trained NER and structured extraction of entities and relations | 5.1–5.6, 5.9–5.10 |
| Co-occurrence matrix | 5.7 |
| Compact topic-specific schema defined before graph construction | 6.1 |
| Graph of entities with co-mention and relation edges | 6.2 |
| Network metrics (centrality, clustering) for influential entities and emerging relationships | 6.3–6.6 |
| Graph visualisation | 6.7 |
| Graph and schema stored for Stage 3 | 6.8–6.9 |
| Topic modelling and topic dynamics | 7 |
| Key insights and notable developments | 8 |

### Contributions

The work was split into three consecutive parts, each building on the previous one.

| Team member | Sections | Contribution |
|---|---|---|
| {DATA_LEAD} | 1–4, 7 | data cleaning and preprocessing, Open Models lexicon, exploratory analysis, sentiment and framing, topic modelling |
| {EXTRACTION_LEAD} | 5.1–5.8 | entity extraction (spaCy, GLiNER), entity linking and evaluation, relation extraction (GLiREL), co-occurrence matrix |
| {GRAPH_LEAD} | 5.9–5.10, 6 | relation refinement, knowledge graph construction, analysis, visualisation and storage |
| All members | 5.10, 8, 9 | human review of relations, key insights, limitation |

The extraction models (spaCy `en_core_web_lg`, GLiNER, GLiREL) were run in Google Colab on an NVIDIA L4 GPU; their
cells show the outputs of that run, and intermediate results are re-used from `data/stage1_wp2/` when present.
"""
    )
    return [
        header,
        markdown(
            f"""
### How to run

[Open in Colab](https://colab.research.google.com/github/jihyeon26/computational-language-technology-hs26/blob/main/stage1/stage1_final.ipynb)
· [Repository]({REPO_URL})

Run all cells from top to bottom. In Colab, the first cell clones the repository, installs the packages and mounts
Google Drive. On the first run it restarts the runtime once
to load the pinned numpy and pandas versions (the topic models depend on them); then run all cells again. A GPU runtime is not required: the GLiNER and
GLiREL results are read from `artifacts/stage1_cache/`, which also makes the run reproduce the reviewed relations.
The raw dataset is read from `data/raw/`, from Google Drive or downloaded from Kaggle (Section 2.1).
"""
        ),
        hide(code(
            f"""
# Colab: clone the repository, install the packages and mount Google Drive (skipped when run from a local clone)
import os, shutil, subprocess, sys
from pathlib import Path

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    clone = Path("/content/computational-language-technology-hs26")
    if not clone.exists():
        subprocess.run(["git", "clone", "-q", "{REPO_URL}.git", str(clone)], check=True)
    os.chdir(clone / "stage1")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "../requirements.txt"], check=True)
    # numpy and pandas are pinned (topic models depend on them); a new version needs a restart of the runtime
    import numpy, pandas
    if (numpy.__version__, pandas.__version__) != ("2.4.2", "3.0.1"):
        print("Pinned numpy/pandas installed. The runtime restarts now: run all cells again.")
        os.kill(os.getpid(), 9)

# Restore the cached results that the notebook reads instead of recomputing them (artifacts/stage1_cache/README.md)
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
CACHE_FILES = {{"extraction/gliner_entities.csv": "data/stage1_wp2/gliner_entities.csv",
               "extraction/relations.csv": "data/stage1_wp2/relations.csv",
               "preprocessing/lemma_cache.parquet": "data/processed/lemma_cache.parquet",
               "preprocessing/track_scan_cache.parquet": "data/processed/track_scan_cache.parquet"}}
for cached, target in CACHE_FILES.items():
    if not (REPO / target).exists():
        (REPO / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "artifacts" / "stage1_cache" / cached, REPO / target)

# Raw dataset in the shared Drive folder CLT/data/raw (added as a shortcut anywhere in My Drive);
# otherwise data/raw/ or the Kaggle download in Section 2.1
drive_root = Path("/content/drive/MyDrive")
for pattern in ["CLT/data/raw", "*/CLT/data/raw", "*/*/CLT/data/raw", "CLT", "*/CLT"]:
    found = sorted(drive_root.glob(f"{{pattern}}/ai_media_dataset_20260911.csv")) if drive_root.exists() else []
    if found and "AI_MEDIA_CSV" not in os.environ:
        os.environ["AI_MEDIA_CSV"] = str(found[0])
print(f"Repository: {{REPO}} | raw dataset: {{os.environ.get('AI_MEDIA_CSV', 'data/raw/ or Kaggle')}}")
"""
        ), "Colab setup: repository, packages, cached results and raw dataset"),
        d[1],
        markdown(f"## 2. Data Collection and Cleaning\n\n*Contributor: {DATA_LEAD}*"),
        *[d[i] for i in range(2, 7)],
        code(
            f"""
# The submitted run used the snapshot ai_media_dataset_20260911.csv; another snapshot gives different results
EXPECTED_RAW_SHA = "{RAW_SHA256}"
if RAW_SHA != EXPECTED_RAW_SHA:
    print(f"Warning: {{RAW_CSV.name}} is not the 2026-09-11 snapshot used in the submitted run; results will differ.")
"""
        ),
        *[d[i] for i in range(7, 30)],
        markdown(
            f"""
## 3. Text Preprocessing

*Contributor: {DATA_LEAD}*

The fidelity text keeps case and punctuation for entity extraction, while the analysis view is tokenised,
lemmatised and stripped of stop words for topic modelling. Acronyms and abbreviations relevant to open models
(*LLM*, *open-weight*, model names such as *GLM-4.5*) are normalised with the Open Models lexicon so that they are
represented consistently across documents (Sections 3.2 and 3.3).
"""
        ),
        *[d[i] for i in range(30, 40)],
        *[d[i] for i in range(56, 58)],
        *[d[i] for i in range(40, 56)],
        *[d[i] for i in range(58, 64)],
        markdown(
            f"""
## 5. Entity and Relation Extraction

*Contributors: {EXTRACTION_LEAD} (5.1–5.8), {GRAPH_LEAD} (5.9–5.10)*

Entities (foundation models, companies, organisations, people, products, technologies, standards, concepts) and the
relations between them are extracted from the 1,210 articles of the core Open Models subset (Section 3.2). The
process goes from mentions to entities to relations; Sections 5.9 and 5.10 then refine the extracted relations
for graph use and measure their quality with a human review.

**Inputs.** The extraction uses the core articles and the Open Models lexicon of Section 3.2. Intermediate results
(GLiNER entities, GLiREL relations and their checkpoints) are saved to `data/stage1_wp2/` so that the slow steps do
not have to be recomputed.
"""
        ),
        hide(code(
            """
# Keep the full cleaned corpus: the extraction cells below use `df` for the core subset
corpus_df, corpus_version_patterns = df, VERSION_PATTERNS
WP1_LEXICON = json.loads((LEXDIR / "open_models_lexicon.json").read_text(encoding="utf-8"))
df = core[["doc_id", "date", "month", "quarter", "domain", "url", "title", "text"]].reset_index(drop=True)
OUT_DIR = str(REPO_ROOT / "data" / "stage1_wp2")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"{len(WP1_LEXICON['families'])} model families, {len(WP1_LEXICON['concepts'])} openness concepts, "
      f"{len(WP1_LEXICON['ecosystem'])} ecosystem tools | {len(df)} core articles")
"""
        ), "Inputs: core articles and lexicon"),
        *[x[i] for i in range(5, 51)],
        hide(x[51], "Save the tables for the knowledge graph (nodes, relations, mentions)"),
        hide(x[52], "Save the schema, data dictionary and evaluation results"),
        markdown(
            """
### 5.9 Relation Refinement

The extracted relations are refined before they are used in the knowledge graph. The cells below load the saved
extraction output, check that it is complete and consistent, describe the issues found in a first screening, and
refine entities and relations with explicit rules.
"""
        ),
        hide(g["setup"][0], "Setup: libraries, input paths and parameters"),
        g["refinement"][0],
        hide(g["refinement"][1], "Load and validate the extraction output"),
        *g["refinement"][2:],
        *g["review"],
        markdown(
            f"""
## 6. Knowledge Graph Construction

*Contributor: {GRAPH_LEAD}*

The refined relations and the co-occurrence pairs are turned into a topic-focused knowledge graph: a compact
schema (6.1), three graph views (6.2), structural statistics (6.3), centrality (6.4), communities (6.5),
temporal analysis (6.6), visualisations (6.7) and storage for Stage 3 (6.8–6.9).
"""
        ),
        *g["graph"],
        *g["storage"],
        hide(code(
            """
# Restore the full cleaned corpus for topic modelling (Section 5 used `df` for the core subset)
df, VERSION_PATTERNS = corpus_df, corpus_version_patterns
"""
        ), "Restore the full cleaned corpus"),
        *[d[i] for i in range(64, 72)],
        d[73], d[74],
        markdown(
            f"""
## 8. Key Insights

*Contributors: All members*

This section brings together the trends, framing and topics of the corpus (8.1), the findings of the knowledge
graph (8.2), a consistency check between the two (8.3) and a summary (8.4).
"""
        ),
        d[72],
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
family_trend = pd.read_csv(ROOT / "artifacts" / "stage1_wp1" / "tables" / "08_2_family_monthly_per_1000_docs.csv")
family_trend["quarter"] = pd.PeriodIndex(family_trend["key"], freq="M").asfreq("Q").astype(str)
lexicon_quarterly = family_trend.drop(columns="key").groupby("quarter").mean()

consistency = []
for family in family_share.index:
    if family not in lexicon_quarterly.columns:
        continue
    graph_trend = family_share.loc[family, first_q:]
    lexicon_trend = lexicon_quarterly[family].reindex(graph_trend.index)
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
From {first_q} to {last_q}, {len(same_direction)} of {len(consistency)} families change in the same direction in both
measurements, and {len(correlated)} have a Spearman correlation of at least 0.5 between the quarterly trends
({", ".join(correlated["family"])}). {shift_text} Lower correlations come from quarter-to-quarter fluctuations, which
differ because the two measures use different article sets.
'''))
""",
            GRAPH_TAG,
        ),
        markdown(
            """
### 8.4 Summary

- **A small but persistent topic.** Open models make up a few percent of AI coverage, with peaks after major
  releases, most strongly after DeepSeek-R1 in early 2025 (Sections 4.1 and 6.6).
- **From Llama to a US-China story.** Attention moved from Meta's Llama towards Chinese families (Qwen, Kimi, GLM)
  and NVIDIA's Nemotron; the lexicon trends and the knowledge graph agree on this shift (Sections 4.2, 8.2 and 8.3).
- **Releases drive the discourse.** Coverage follows releases, and the release cadence accelerated in 2026
  (Section 4.4); the relations first observed in 2026 are mostly new models and products and their developers
  (Section 6.6).
- **Developers play different roles.** OpenAI, Anthropic and Meta appear mainly with models, NVIDIA mainly with
  infrastructure; competition is reported around a few frontier players (Sections 6.4 and 8.2).
- **More precise vocabulary and mid-sized models.** *Open-weight* replaces *open-source* in a growing share of
  mentions (Section 4.5), and mid-sized models (10–40B) gain share among models named with a size (Section 8.2).
- **From models to their use.** Topics shift from reasoning models and image generation towards agentic coding and
  deployment ecosystems (Section 7.1).
"""
        ),
        markdown("## 9. Limitations"),
        d["limitations"],
        markdown(
            """
**Entity and relation extraction**
- Generic entities, type errors and type-invalid relations remain in the extraction output; they are kept with
  their reasons so that the relation refinement can handle them (Sections 5.5, 5.6 and 5.9).
"""
        ),
        *g["limitations"],
        markdown(
            f"""
## 10. AI Tool Disclosure

The team used **Claude Code (Anthropic)** as a coding and technical assistant throughout Stage 1:

- Reviewing and optimising code, including the reduction of processing time and the preparation of the
  saved outputs;
- reviewing and challenging pipeline design choices;
- drafting explanations and documentation in the notebook;
- assisting with labels: the relevance labels of the track validation sample (Section 3.2.3), a first screening
  of the extracted relations (Section 5.9.2) and a first pass over the relations behind the findings (Section 5.10).
  Every `FALSE` verdict of the latter was checked by a person, and precision is measured on the human review only.

All methodological choices, analyses and interpretations were made and checked by the team, and all numbers come
from running this notebook.
"""
        ),
    ]


def execute_graph_cells(nb, path: Path) -> None:
    """Run the knowledge-graph cells in a separate kernel and copy their outputs into the notebook."""
    from nbclient import NotebookClient

    tagged = [cell for cell in nb.cells if GRAPH_TAG in cell.metadata.get("tags", [])]
    run = nbf.v4.new_notebook(metadata=copy.deepcopy(nb.metadata))
    run.cells = [copy.deepcopy(cell) for cell in tagged]
    NotebookClient(run, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(path)}}).execute()
    for cell, executed in zip(tagged, run.cells):
        cell.outputs = executed.outputs
        for output in cell.outputs:
            if "execution_count" in output:
                output["execution_count"] = None


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nb["cells"] = build()
    leftover = re.compile(r"\bWP ?[123]\b|work package|handover|hand-over", re.I)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            match = leftover.search(cell.source)
            assert not match, (match.group(), cell.source[:100])
    for number, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{number:03d}"
    output = TEAM_REPO / "stage1" / "stage1_final.ipynb" if "--team" in sys.argv else OUTPUT
    if "--no-execute" not in sys.argv:
        execute_graph_cells(nb, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output)
    print(output)


if __name__ == "__main__":
    main()
