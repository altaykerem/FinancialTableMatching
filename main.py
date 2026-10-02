import json
import configparser
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src import str_utils
from src.pdf_reader import PDFReader
from src.page import PageBuilder
from src.ranker import EncoderRanker

config = configparser.ConfigParser()
config.read("config.ini")
config.sections()

contents_page = int(config["MODEL"]["ContentsPage"])
referance_no = int(config["MODEL"]["ReferanceNo"])
finantial_pages = [int(x.strip()) for x in config["MODEL"]["FinantialPages"].split(",")]

language = config["DEFAULT"]["Language"]
models_dir = config["MODEL"]["ModelDir"]
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
    s_page: int = None
    s_row: int = None
    s_col: int = None
    r_page: int = None
    r_table: int = None
    r_row: int = None
    r_col: int = None
    score: float = None


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

    return rels, correct / total


def build_cell_mathes():
    # all the rows from the referace
    ref_page, _ = reader.note_pages[referance_no]
    search_rows, search_cols = [], []
    for ref_obj in referance_page_objects:
        for ref_page_table_no, ref_df in enumerate(ref_obj.tables):
            for row, row_j in zip(ref_df.Kalem, ref_df.index):
                original = str(row)
                row_p = str_utils.preprocess(original)
                search_rows.append([ref_page, ref_page_table_no, row_j, row_p])

            for col_j in range(2, len(ref_df.columns) - 1):
                col_p = str_utils.preprocess(str(ref_df.columns[col_j]))
                search_cols.append([ref_page, ref_page_table_no, col_j, col_p])
        ref_page = ref_page + 1

    search_rows_df = pd.DataFrame(
        search_rows, columns=["Page", "Table", "RefRow", "Compare"]
    )
    search_cols_df = pd.DataFrame(
        search_cols, columns=["Page", "Table", "RefCol", "Compare"]
    )

    referance_matches = []
    for page, page_obj in zip(finantial_pages, summary_page_objects):
        df = page_obj.tables[0].copy()
        # rows containing the reference
        df = df[df[note] == referance_no]
        if len(df) == 0:
            continue

        # row by row similarity
        for row, row_i in zip(df.Kalem, df.index):
            row = str_utils.preprocess(str(row))
            text_list = search_rows_df.Compare.to_list()
            row_scores = ranker_model.get_probabilities(row, text_list)

            # column by row similarity
            for col_i in range(2, len(df.columns) - 1):
                col = str_utils.preprocess(str(df.columns[col_i]))
                col_scores = ranker_model.get_probabilities(col, text_list)

                # combine probs
                score_df = search_rows_df.copy()
                score = row_scores * col_scores
                score_df["score"] = score
                score_df.sort_values(by="score", inplace=True, ascending=False)

                # select top table rows
                top = next(score_df.itertuples())
                referance_matches.append(
                    CellMatch(
                        s_page=page,
                        s_row=row_i,
                        s_col=col_i,
                        r_page=top.Page,
                        r_table=top.Table,
                        r_row=top.RefRow,
                        score=top.score,
                    )
                )

        # column by column similarity
        for _match in referance_matches:
            score_df = search_cols_df[
                (search_cols_df["Page"] == _match.r_page)
                & (search_cols_df["Table"] == _match.r_table)
            ].copy()
            col = str_utils.preprocess(str(df.columns[_match.s_col]))
            text_list = score_df.Compare.to_list()
            score_df["score"] = ranker_model.get_probabilities(col, text_list)
            score_df.sort_values(by=["score", "RefCol"], inplace=True, ascending=False)
            top = next(score_df.itertuples())
            _match.r_col = top.RefCol
            _match.score = _match.score * top.score

        # check
        total, correct = 0, 0
        for _match in referance_matches:
            s_value = page_to_summary[_match.s_page].iloc[_match.s_row, _match.s_col]
            r_value = page_to_ref[_match.r_page][_match.r_table].iloc[
                _match.r_row, _match.r_col
            ]
            if s_value == r_value:
                correct += 1
            total += 1
        return referance_matches, correct / total


if __name__ == "__main__":
    final_report = {
        "summary": [],
        "referance": [],
        "line_hierarchy": {},
        "cell_matches": {},
    }

    # Stage 1: Read PDF and store detected bounding boxes as a data frame
    reader = PDFReader(models_dir, "tr")
    reader.pdf_to_text_boxes(pdf_path, flush=False)
    reader.read_contents(contents_page)

    # Stage 2: Construct tables from the bounding boxes

    # Parse finantial pages
    summary_page_objects = [
        PageBuilder(reader.read_summary_page(p)) for p in finantial_pages
    ]

    # Add summmary pages
    page_to_summary = {}
    for page, page_obj in zip(finantial_pages, summary_page_objects):
        page_to_summary[page] = page_obj.tables[0]
        final_report["summary"].append(
            {"page": page, "table": page_obj.tables[0].to_dict(orient="records")}
        )

    # Processing specificly for note references
    note = "dipnot"
    for page_obj in summary_page_objects:
        page_obj.tables[0].columns = [
            note if note in str_utils.preprocess(col) else col
            for col in page_obj.tables[0].columns
        ]
        referances = page_obj.tables[0][note].map(str_utils.ocr_number_correction)
        page_obj.tables[0][note] = referances.fillna(-1).astype(int)

    # Parse referance page from its number
    referance_page_objects = [
        PageBuilder(p_boxes) for p_boxes in reader.read_referance_pages(referance_no)
    ]

    # Add referance pages
    ref_page, _ = reader.note_pages[referance_no]
    page_to_ref = {}
    for ref_obj in referance_page_objects:
        if ref_page not in page_to_ref:
            page_to_ref[ref_page] = {}
        for ref_page_table_no, ref_df in enumerate(ref_obj.tables):
            final_report["summary"].append(
                {"page": ref_page, "table": ref_df.to_dict(orient="records")}
            )
            page_to_ref[ref_page][ref_page_table_no] = ref_df
        ref_page = ref_page + 1

    # Stage 3: Line hierarchies
    for page in finantial_pages:
        hies, conf = summary_table_line_hierarchies(page, page_to_summary[page])
        final_report["line_hierarchy"]["confidance"] = conf
        final_report["line_hierarchy"]["hierarchies"] = hies

    # Stage 4: Cell to cell matches
    ranker_model = EncoderRanker(config["MODEL"]["RankingModel"], models_dir)
    referance_matches, conf = build_cell_mathes()
    final_report["cell_matches"]["confidance"] = conf
    final_report["cell_matches"]["hierarchies"] = [asdict(m) for m in referance_matches]

    with open("output.json", "w") as file:
        json.dump(final_report, file, indent=4)
