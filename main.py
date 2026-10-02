import re
import json
import configparser
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src import str_utils
from src.pdf_reader import PDFReader
from src.page import PageBuilder
from src.ranker import EncoderRanker, CosineRanker
from src.entities import CompanyExtractor

config = configparser.ConfigParser()
config.read("config.ini")
config.sections()

contents_page = int(config["MODEL"]["ContentsPage"])
referance_no = int(config["MODEL"]["ReferanceNo"])
finantial_pages = [int(x.strip()) for x in config["MODEL"]["FinantialPages"].split(",")]

language = config["DEFAULT"]["Language"]
models_dir = config["MODEL"]["ModelDir"]
use_gpu = config["MODEL"].getboolean("GPU")
pdf_path = config["DEFAULT"]["PDFPath"]

# create directories
for _path in [models_dir, "./data"]:
    dir_path = Path(_path)
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class RowHierarchy:
    page: int
    parent: str
    child: str
    matched: bool


@dataclass
class CellMatch:
    note: int = None
    s_page: int = None
    s_row: int = None
    s_col: int = None
    r_page: int = None
    r_table: int = None
    r_row: int = None
    r_col: int = None
    score: float = None


def table_to_record(df):
    """Table rows as dicts, deduplicate headers"""
    columns, seen = [], {}
    for col in map(str, df.columns):
        seen[col] = seen.get(col, 0) + 1
        columns.append(col if seen[col] == 1 else f"{col} ({seen[col]})")
    return df.set_axis(columns, axis=1).to_dict(orient="records")


def table_columns(df):
    """Extract date from column header"""
    return [
        {"index": j, "header": str(h), "period": str_utils.parse_period(h)}
        for j, h in enumerate(df.columns[:-1])
    ]


def table_title(paragraphs):
    """check for the first header line that holds a period"""
    for text in paragraphs[1:]:
        if str_utils.parse_period(text):
            return text
    return paragraphs[1] if len(paragraphs) > 1 else None


def summary_table_line_hierarchies(page, df):
    # process finantial columns
    financial_cols = df.columns[2:-1]
    df = df.dropna(subset=financial_cols, how="all").reset_index(drop=True)
    df[financial_cols] = (
        df[financial_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    row_financials = {}
    parents = {}

    stack, prev_l = [], int(df.iloc[0, -1])
    if prev_l > 0:
        prev_l = int(df.iloc[len(df) - 1, -1])
        for i in range(len(df) - 1, -2, -1):
            level = int(df.iloc[i, -1])
            row_financials[i] = df.iloc[i, 2:-1].to_numpy()
            if len(stack):
                parents[i + 1] = stack[-1]
            if level > prev_l:
                stack.append(i + 1)
            elif level < prev_l:
                stack.pop()
            prev_l = level
    else:
        for i in range(len(df)):
            level = int(df.iloc[i, -1])
            row_financials[i] = df.iloc[i, 2:-1].to_numpy()
            if len(stack):
                parents[i - 1] = stack[-1]
            if level > prev_l:
                stack.append(i - 1)
            elif level < prev_l:
                stack.pop()
            prev_l = level

    childs = {}
    for child, parent in parents.items():
        if parent not in childs:
            childs[parent] = []
        childs[parent].append(child)

    total, correct = 0, 0
    rels = []
    for parent, child_list in childs.items():
        financial_sum = np.sum([row_financials[i] for i in child_list], axis=0)
        matched = np.array_equal(row_financials[parent], financial_sum)
        for c in child_list:
            rels.append(
                asdict(
                    RowHierarchy(
                        page=page,
                        parent=df.iloc[parent]["Kalem"],
                        child=df.iloc[c]["Kalem"],
                        matched=matched,
                    )
                )
            )
        if matched:
            correct += 1
        total += 1

    return rels, correct / total if total else 0.0


def find_total_column(r_df, value_cols):
    """Column that equals the sum of the other value columns on most rows"""
    if len(value_cols) < 3:
        return None
    values = r_df.iloc[:, value_cols].apply(pd.to_numeric, errors="coerce")
    values.columns = value_cols
    for col_j in value_cols:
        rows = values[col_j].notna() & values.drop(columns=col_j).notna().any(axis=1)
        if rows.sum() < 2:
            continue
        others = values.loc[rows].drop(columns=col_j).sum(axis=1)
        if ((values.loc[rows, col_j] - others).abs() < 1).mean() >= 0.8:
            return col_j
    return None


def build_cell_mathes(ranker_model, ref_no, ref_page_objects, ref_tables):
    # all the rows and tables from the referace
    ref_page, _ = reader.note_pages[ref_no]
    search_rows, search_tables = [], []
    for ref_obj in ref_page_objects:
        for ref_page_table_no, ref_df in enumerate(ref_obj.tables):
            for row, row_j in zip(ref_df.Kalem, ref_df.index):
                # titles and text lines have no values to match
                values = ref_df.iloc[row_j, 1:-1]
                if not any(PageBuilder.is_number(v) for v in values):
                    continue
                row_p = str_utils.preprocess(str(row))
                search_rows.append([ref_page, ref_page_table_no, row_j, row_p])

            # table context: headers and row labels
            context = list(ref_df.columns[1:-1]) + list(ref_df.Kalem)
            context = str_utils.preprocess(" ".join(map(str, context)))
            search_tables.append([ref_page, ref_page_table_no, context])
        ref_page = ref_page + 1

    search_rows_df = pd.DataFrame(
        search_rows, columns=["Page", "Table", "RefRow", "Compare"]
    )
    search_tables_df = pd.DataFrame(search_tables, columns=["Page", "Table", "Compare"])
    if len(search_rows_df) == 0:
        return [], 0.0

    referance_matches = []
    for page, page_obj in zip(finantial_pages, summary_page_objects):
        df = page_obj.tables[0].copy()
        # rows containing the reference
        df = df[df[note] == ref_no]
        if len(df) == 0:
            continue

        for row, row_i in zip(df.Kalem, df.index):
            # the item text selects the row
            row = str_utils.preprocess(str(row))
            row_scores = ranker_model.get_probabilities(
                row, search_rows_df.Compare.to_list()
            )

            for col_i in range(2, len(df.columns) - 1):
                # the period header selects the table
                col = str_utils.preprocess(str(df.columns[col_i]))
                table_scores = search_tables_df.copy()
                table_scores["table_score"] = ranker_model.get_probabilities(
                    col, search_tables_df.Compare.to_list()
                )

                score_df = search_rows_df.copy()
                score_df["row_score"] = row_scores
                score_df = score_df.merge(
                    table_scores[["Page", "Table", "table_score"]],
                    on=["Page", "Table"],
                )
                score_df["score"] = score_df.row_score * score_df.table_score
                score_df.sort_values(by="score", inplace=True, ascending=False)

                top = next(score_df.itertuples())
                referance_matches.append(
                    CellMatch(
                        note=ref_no,
                        s_page=page,
                        s_row=row_i,
                        s_col=col_i,
                        r_page=top.Page,
                        r_table=top.Table,
                        r_row=top.RefRow,
                        score=top.score,
                    )
                )

    # column selection
    for _match in referance_matches:
        r_df = ref_tables[_match.r_page][_match.r_table]
        value_cols = list(range(1, len(r_df.columns) - 1))
        if not value_cols:
            continue
        r_headers = [str_utils.preprocess(str(c)) for c in r_df.columns[1:-1]]

        # date period columns
        period_cols = any(re.search(r"\b(19|20)\d{2}\b", h) for h in r_headers)
        total_col = None if period_cols else find_total_column(r_df, value_cols)
        if total_col is not None:
            _match.r_col = total_col
            continue

        s_columns = page_to_summary[_match.s_page].columns
        col = str_utils.preprocess(str(s_columns[_match.s_col]))
        col_scores = ranker_model.get_probabilities(col, r_headers)
        best = int(np.argmax(col_scores))
        _match.r_col = value_cols[best]
        _match.score = _match.score * float(col_scores[best])

    # check
    total, correct = 0, 0
    for _match in referance_matches:
        if _match.r_col is None:
            continue
        s_value = page_to_summary[_match.s_page].iloc[_match.s_row, _match.s_col]
        r_value = ref_tables[_match.r_page][_match.r_table].iloc[
            _match.r_row, _match.r_col
        ]
        if s_value == r_value:
            correct += 1
        total += 1
    return referance_matches, correct / total if total else 0.0


def evaluate_control(referance_matches, control_examples):
    """Accuracy against hand labelled control examples"""
    by_cell = {(m.s_page, m.s_row, m.s_col): m for m in referance_matches}
    rows, cols, cells = 0, 0, 0
    for example in control_examples:
        _match = by_cell.get((example["s_page"], example["s_row"], example["s_col"]))
        if _match is None:
            continue
        row_ok = (
            _match.r_page == example["r_page"]
            and _match.r_table == example["r_table"]
            and _match.r_row in example["r_rows"]
        )
        col_ok = _match.r_col == example["r_col"]
        rows += row_ok
        cols += col_ok
        cells += row_ok and col_ok

    total = max(len(control_examples), 1)
    return {
        "examples": len(control_examples),
        "row_accuracy": rows / total,
        "col_accuracy": cols / total,
        "cell_accuracy": cells / total,
    }


if __name__ == "__main__":
    final_report = {
        "summary": [],
        "referance": [],
        "line_hierarchy": {},
        "cell_matches": {},
    }

    # Stage 1: Read PDF and store detected bounding boxes as a data frame
    reader = PDFReader(models_dir, language, use_gpu)
    reader.pdf_to_text_boxes(pdf_path, flush=False)
    reader.read_contents(contents_page)

    # Stage 2: Construct tables from the bounding boxes

    # Parse finantial pages
    summary_page_objects = [
        PageBuilder(reader.read_summary_page(p)) for p in finantial_pages
    ]

    # Processing specificly for note references
    note = "dipnot"
    for page, page_obj in zip(finantial_pages, summary_page_objects):
        page_obj.tables[0].columns = [
            note if note in str_utils.preprocess(col) else col
            for col in page_obj.tables[0].columns
        ]
        referances = page_obj.tables[0][note].map(str_utils.ocr_number_correction)
        referances = pd.to_numeric(referances, errors="coerce")
        page_obj.tables[0][note] = referances.fillna(-1).astype(int)

    # Stage 3: Line hierarchies
    page_to_summary = {}
    for page, page_obj in zip(finantial_pages, summary_page_objects):
        page_to_summary[page] = page_obj.tables[0]
        hies, conf = summary_table_line_hierarchies(page, page_to_summary[page])
        final_report["line_hierarchy"][page] = {
            "confidance": conf,
            "hierarchies": hies,
        }

    # Add summmary pages
    for page, page_obj in zip(finantial_pages, summary_page_objects):
        final_report["summary"].append(
            {
                "page": page,
                "title": table_title(page_obj.paragraphs),
                "currency": reader.read_currency(page),
                "columns": table_columns(page_obj.tables[0]),
                "table": table_to_record(page_obj.tables[0]),
            }
        )

    # Parse referance page from its number
    referance_page_objects = [
        PageBuilder(p_boxes) for p_boxes in reader.read_referance_pages(referance_no)
    ]
    first_paragraphs = referance_page_objects[0].paragraphs
    note_title = first_paragraphs[0] if first_paragraphs else None
    if note_title:
        note_title = re.sub(r"^[\dIl]+\s*[\.,_]?\s*", "", note_title)

    # Add referance pages
    ref_page, _ = reader.note_pages[referance_no]
    page_to_ref = {}
    for ref_obj in referance_page_objects:
        page_to_ref[ref_page] = {}
        for ref_page_table_no, ref_df in enumerate(ref_obj.tables):
            final_report["referance"].append(
                {
                    "note": referance_no,
                    "page": ref_page,
                    "table_no": ref_page_table_no,
                    "title": note_title,
                    "currency": reader.read_currency(ref_page),
                    "columns": table_columns(ref_df),
                    "table": table_to_record(ref_df),
                }
            )
            page_to_ref[ref_page][ref_page_table_no] = ref_df
        ref_page = ref_page + 1

    # Organisations named on the referance pages, by NER
    org_finder = CompanyExtractor(config["MODEL"]["NerModel"], models_dir)
    final_report["organizations"] = [
        {"page": page, "name": name, "score": round(score, 3)}
        for page, ref_obj in zip(page_to_ref, referance_page_objects)
        for text in ref_obj.paragraphs
        for name, score in org_finder.organisations(text)
    ]

    # Stage 4: Cell to cell matches per ranking method
    rankers = {
        "cross_encoder": lambda: EncoderRanker(
            config["MODEL"]["RankingModel"], models_dir
        ),
        "embedding": lambda: CosineRanker(
            config["MODEL"]["EmbeddingModel"], models_dir
        ),
    }
    for method, build_ranker in rankers.items():
        referance_matches, agreement = build_cell_mathes(
            build_ranker(), referance_no, referance_page_objects, page_to_ref
        )
        final_report["cell_matches"][method] = {
            "confidance": agreement,
            "matches": [asdict(m) for m in referance_matches],
        }
        print(method, final_report["cell_matches"][method]["confidance"])

    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=4, ensure_ascii=False, default=str)
