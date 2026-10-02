import re
from typing import List

import numpy as np
import pandas as pd


class PageBuilder:
    def __init__(self, bbox_df: pd.DataFrame):
        bbox_df["w"] = bbox_df.x1 - bbox_df.x0
        bbox_df["xc"] = (bbox_df.x0 + bbox_df.x1) / 2

        # constants
        self.word_gap = 35
        self.paragraph_gap = 60
        self.table_gap = 300

        # variables
        bbox_df = self._add_lines(bbox_df)
        bbox_df = self._merge_boxes_in_a_row(bbox_df)
        row_df, table_df = self._seperate_lines_and_tables(bbox_df)
        self.paragraphs = self._parse_rows(row_df)
        del bbox_df, row_df
        table_box_list = self._seperate_tables(table_df)
        self.tables: List[pd.DataFrame] = []
        for table_df in table_box_list:
            table_df, num_cols = self._cluster_columns(table_df)
            table_df, num_rows = self._merge_header_values(table_df, num_cols)
            table = self._build_table(table_df, num_cols, num_rows)
            if table is not None:
                self.tables.append(table)

    def _add_lines(self, bbox_df):
        bbox_df["h"] = bbox_df.y1 - bbox_df.y0
        bbox_df["yc"] = (bbox_df.y0 + bbox_df.y1) / 2
        bbox_df.sort_values(["yc", "x0"], inplace=True)
        line, cur, last_yc, last_h = [], -1, None, None
        for yc, h in zip(bbox_df.yc, bbox_df.h):
            if last_yc is None or abs(yc - last_yc) > 0.5 * min(h, last_h):
                cur += 1
            line.append(cur)
            last_yc, last_h = yc, h
        bbox_df["line"] = line
        bbox_df.drop(columns=["y0", "y1", "h"], inplace=True)
        bbox_df.sort_values(["line", "x0"]).reset_index(drop=True)
        return bbox_df

    def _merge_boxes_in_a_row(self, bbox_df):
        merged_rows = []
        for line, g in bbox_df.groupby("line"):
            g = g.sort_values("x0")
            current = None

            for row in g.itertuples(index=False):
                if current is None:
                    current = row._asdict()
                elif row.x0 - current["x1"] <= self.word_gap:
                    current["x0"] = min(current["x0"], row.x0)
                    current["confidence"] = min(current["confidence"], row.confidence)
                    current["x1"] = max(current["x1"], row.x1)
                    current["text"] = current["text"] + " " + row.text
                else:
                    merged_rows.append(current)
                    current = row._asdict()

            if current is not None:
                merged_rows.append(current)

        return pd.DataFrame(merged_rows)

    @staticmethod
    def line_gap_sum(g):
        """Feature used for classifing lines against tables"""
        g = g.sort_values("x0")
        gaps = g.x0.values[1:] - g.x1.values[:-1]
        return gaps.clip(min=0).sum()

    def _seperate_lines_and_tables(self, bbox_df):
        # line gaps
        gap_per_line = (
            bbox_df.groupby("line").apply(self.line_gap_sum).rename("gap_sum")
        )
        bbox_df = bbox_df.merge(gap_per_line, on="line")

        # guess lines
        values = (bbox_df["gap_sum"] < 100) & (bbox_df["w"] > 300)
        values.iloc[0] = True
        values.iloc[-1] = True

        # simple error correction, look at your neighbors
        for i in range(1, len(values) - 1):
            if values[i - 1] != values[i] and values[i + 1] != values[i]:
                values[i] = not values[i]

        # results
        row_df = bbox_df[values].copy()
        table_df = bbox_df[~values].copy()
        return row_df, table_df

    def _parse_rows(self, row_df):
        paragraphs = []
        sentences = []
        prev_y = None
        for row in row_df.itertuples(index=False):
            if prev_y is not None and row.yc > prev_y + self.paragraph_gap:
                paragraphs.append(" ".join(sentences))
                sentences = []
            sentences.append(row.text)
            prev_y = row.yc

        if sentences:
            paragraphs.append(" ".join(sentences))

        return paragraphs

    def _seperate_tables(self, table_df):
        table_lines = (
            table_df[["line", "yc"]]
            .drop_duplicates(subset=["line"], keep="first")
            .sort_values("yc")
            .reset_index(drop=True)
        )

        # A new table starts wherever the gap to the previous line is larger than table_eps
        gap = table_lines["yc"].diff()
        table_lines["table_id"] = (gap > self.table_gap).cumsum()
        table_df["table_id"] = table_df["line"].map(
            table_lines.set_index("line")["table_id"]
        )
        return [
            group.drop(columns="table_id").reset_index(drop=True)
            for _, group in table_df.groupby("table_id", sort=True)
        ]

    def _cluster_columns(self, table_df):
        # sort centers, group consecutive ones within eps
        x_eps = 80
        table_df["xc"] = (table_df.x0 + table_df.x1) / 2
        sorted_c = np.sort(table_df.x0)
        clusters: list[list[float]] = [[sorted_c[0]]]
        for c in sorted_c[1:]:
            if c - clusters[-1][-1] <= x_eps:
                clusters[-1].append(c)
            else:
                clusters.append([c])

        clusters = [c for c in clusters if len(c) >= 3]

        # sort clusters by their median x position
        clusters.sort(key=lambda c: np.median(c))

        # find boundaries between consecutive clusters
        boundaries = []
        for i in range(len(clusters) - 1):
            right_of_left = max(clusters[i])
            left_of_right = min(clusters[i + 1])
            gap = left_of_right - right_of_left
            if gap > x_eps:
                boundaries.append(float(right_of_left + left_of_right) / 2)

        boundaries = sorted(boundaries)
        num_cols = len(boundaries) + 1

        # column classification
        column_values = []
        for row in table_df.itertuples():
            column_val = len(boundaries)
            for i, b in enumerate(boundaries):
                if row.xc < b:
                    column_val = i
                    break
            column_values.append(column_val)

        table_df["cluster"] = column_values
        table_df.drop(columns=["xc"], inplace=True)
        return table_df, num_cols

    @staticmethod
    def parse_numeric(x):
        if x is None or pd.isna(x):
            return None
        x = str(x).strip()
        if not x:
            return None

        # for searching split numbers
        body = x.replace(" ", "").strip("()-")
        if re.fullmatch(r"(19|20)\d{2}", body):
            # a year is used in headers
            return x

        # dots are thousands separators, comma is the decimal separator
        numeric = re.fullmatch(r"(\d{1,3}(?:\.\d{3})*|\d+)(,\d+)?", body)
        if numeric and "." in numeric.group(1) and len(numeric.group(2) or "") == 4:
            value = int(re.sub(r"[.,]", "", body))
        elif numeric:
            value = int(numeric.group(1).replace(".", ""))
            if numeric.group(2):
                value = value + float("0." + numeric.group(2)[1:])
        elif re.fullmatch(r"\d+(?:[.,]+\d{3})+", body):
            value = int(re.sub(r"[.,]", "", body))
        else:
            return x

        if x.startswith("(") or x.startswith("-"):
            value = -1 * value
        return value

    @staticmethod
    def is_number(x):
        return isinstance(x, (int, float, np.number)) and not pd.isna(x)

    def _merge_header_rows(self, df):
        """Rows with text but no numbers in value columns are header lines"""
        value_cols = range(1, df.shape[1] - 1)
        headers = list(df.columns)
        year = re.compile(r"^(19|20)\d{2}$")
        top = True
        for i in range(len(df)):
            filled = [
                j
                for j in value_cols
                if df.iat[i, j] is not None and not pd.isna(df.iat[i, j])
            ]
            if not filled:
                continue
            texts = [str(df.iat[i, j]) for j in filled]
            if any(self.is_number(df.iat[i, j]) for j in filled) or not all(
                year.match(t) or not re.search(r"\d", t) for t in texts
            ):
                top = False
                continue

            if top:
                # continuation of the header, the label stays as a title row
                for j in filled:
                    header = headers[j]
                    text = str(df.iat[i, j])
                    empty = header is None or pd.isna(header)
                    headers[j] = text if empty else f"{header} {text}"
                    df.iat[i, j] = None
            elif all(year.match(str(df.iat[i, j])) for j in filled):
                # repeated period header inside the table
                for j in filled:
                    df.iat[i, j] = None

        df.columns = headers
        return df

    def _merge_header_values(self, table_df, num_cols):
        table_df = table_df.sort_values(["line", "x0"]).reset_index(drop=True)
        header_end = table_df[table_df["cluster"] == 0]["line"].iloc[0]

        # Group rows based on y values
        merged_rows = []
        for col_i in range(num_cols):
            column_vals = table_df[table_df["cluster"] == col_i].copy()
            current = None
            for row in column_vals.itertuples(index=False):
                if current is None:
                    current = row._asdict()
                elif row.line < header_end:
                    # headers merge vertically
                    current["text"] = current["text"] + " " + row.text
                    current["x0"] = min(current["x0"], row.x0)
                    current["x1"] = max(current["x1"], row.x1)
                    current["line"] = max(current["line"], row.line)
                    current["confidence"] = min(current["confidence"], row.confidence)
                else:
                    merged_rows.append(current)
                    current = row._asdict()

            if current is not None:
                merged_rows.append(current)

        table_df = pd.DataFrame(merged_rows)
        # redo lines
        old_lines = sorted(table_df.line.unique())
        line_map = {old: new for new, old in enumerate(old_lines)}
        table_df["line"] = table_df["line"].map(line_map)
        table_df.drop(columns=["x1", "yc"], inplace=True)
        num_rows = len(old_lines)
        return table_df, num_rows

    def _build_table(self, table_df, num_cols, num_rows):
        matrix = [[None for _ in range(num_cols + 1)] for _ in range(num_rows)]
        for row in table_df.itertuples(index=False):
            matrix[row.line][row.cluster] = row.text
            if row.cluster == 0:
                matrix[row.line][num_cols] = row.x0

        matrix[0][num_cols] = "x0"
        matrix[0][0] = "Kalem"

        df = pd.DataFrame(matrix)
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df.x0 = df.x0.fillna(0)
        df = df.dropna(subset=["Kalem"]).reset_index(drop=True)
        if df.empty:
            return None

        # row indentations
        xs = df.x0.to_numpy()
        xs = xs - xs.min()
        column_values = []
        for x0 in xs:
            indent = 0
            for i in range(5):
                if x0 < (i + 1) * xs.std() * 1.5:
                    indent = i
                    break
            column_values.append(indent)

        df.drop(columns=["x0"], inplace=True)
        df["Başlık"] = column_values
        upper_rows = df.Kalem.str.isupper()
        df.loc[upper_rows, "Başlık"] = 0

        # check data types
        for i in range(1, num_cols):
            df.isetitem(i, df.iloc[:, i].map(self.parse_numeric))

        return self._merge_header_rows(df)
