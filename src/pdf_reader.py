import os
from typing import List

import pymupdf
import easyocr
import pandas as pd
import numpy as np


class PDFReader:
    def __init__(self, model_dir: str, language: str = "tr"):
        self.language = language
        self.ocr_dir = "./data/ocr.parquet"
        self.model_dir = model_dir
        self.page_dpi = 300  # (Dots Per Inch), 118 Dots Per CM

        self.contents_page = 0
        self.ocr_boxes = None
        self.notes = None
        self.note_pages = {}

    def pdf_to_text_boxes(self, pdf_path: str, flush: bool = False) -> None:
        if not flush and os.path.exists(self.ocr_dir):
            print("Caution you are reading pre-calculated OCR boundries!")
            print("Use `flush=True` to reindex calculations.")
            self.ocr_boxes = pd.read_parquet(self.ocr_dir)
            return

        # Read PDF
        doc = pymupdf.open(pdf_path)
        reader = easyocr.Reader(
            ["tr"],
            gpu=True,
            model_storage_directory=self.model_dir,
        )

        boxes = []
        for page_idx in range(len(doc)):
            # PDF page to image
            zoom = self.page_dpi / 72
            mat = pymupdf.Matrix(zoom, zoom)
            pix = doc[page_idx].get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )

            # top left should be more populated
            h, w = img.shape[:2]
            top_left = img[h // 50: h // 4, w // 20: w // 4]
            if top_left.mean() > 250:
                img = np.rot90(img, k=3)

            # Image to detected texts
            results = reader.readtext(img)
            for bbox, text, conf in results:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                boxes.append(
                    {
                        "page": page_idx + 1,
                        "text": str(text.strip()),
                        "x0": int(min(xs)),
                        "y0": int(min(ys)),
                        "x1": int(max(xs)),
                        "y1": int(max(ys)),
                        "confidence": round(float(conf), 2),
                    }
                )

        self.ocr_boxes = pd.DataFrame(boxes)
        self.ocr_boxes.to_parquet(self.ocr_dir, index=False)
        print("Number of detections:", len(self.ocr_boxes))
        print("Number of pages:", len(self.ocr_boxes["page"].unique()))

    def read_contents(self, page: int) -> None:
        self.contents_page = page
        contents_df = self.ocr_boxes[self.ocr_boxes["page"] == page].copy()
        contents_df["text"] = contents_df["text"].str.strip()

        # Parse notes
        notes = contents_df[contents_df["text"].str.startswith("NOT", na=False)]
        notes_start = notes.y0.iloc[0]
        notes = (
            notes["text"]
            .str.extract(r"NOT\s*(\d+)", expand=False)
            .astype(int)
            .to_list()
        )

        # Drop the other section
        contents_df = contents_df[contents_df["y0"] > notes_start]

        # Parse pages
        pages = contents_df[contents_df["text"].str.match(r"^\d+(-\d)?", na=False)]
        page_starts = []
        page_ends = []
        for row in pages.itertuples(index=False):
            if "-" in row.text:
                start, end = row.text.split("-", 1)
                start, end = int(start) + page, int(end) + page
            else:
                start, end = int(row.text) + page, int(row.text) + page

            if page_ends and (page_ends[-1] + 1 < start):
                # missing value
                page_starts.append(page_ends[-1] + 1)
                page_ends.append(start - 1)

            page_starts.append(start)
            page_ends.append(end)

        self.notes = notes

        for note, ps, pe in zip(notes, page_starts, page_ends):
            self.note_pages[note] = (ps, pe)

    def read_summary_page(self, page) -> pd.DataFrame:
        assert page > self.contents_page, "Summary should come after contents"
        assert (
            page < self.note_pages[self.notes[0]][0]
        ), "Summary should come before referances"
        page_df = self.ocr_boxes[self.ocr_boxes["page"] == page].copy()
        page_df.drop(columns=["page"], inplace=True)
        return page_df

    def read_referance_pages(self, ref_no) -> List[pd.DataFrame]:
        assert ref_no in self.note_pages, "Referance not found in contents"
        page_start, page_end = self.note_pages[ref_no]
        result = []
        for i in range(page_start, page_end + 1):
            page_df = self.ocr_boxes[self.ocr_boxes["page"] == i].copy()
            page_df.drop(columns=["page"], inplace=True)
            result.append(page_df)
        return result
