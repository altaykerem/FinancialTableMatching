# Financial Report Summary and Note Table Matching


## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install sentence-transformers
```

Download the source document to `docs/kap.pdf`:
https://kap.org.tr/tr/api/file/download/33E83438337C023CE0530A4A622B5826

## Run

```bash
python main.py
```

The output is written to `output.json`. The first run OCRs the whole PDF (GPU recommended) and caches the result in `data/ocr.parquet`. Later runs reuse the cache; pass `flush=True` in `pdf_to_text_boxes` to redo it. Models are downloaded to `models/`.

Settings live in `config.ini`:

| Key | Meaning |
|---|---|
| `FinantialPages` | Summary table pages (`5, 6, 7`) |
| `ReferanceNo` | Note to match (`11`) |
| `ContentsPage` | Table of contents page, used to find note pages |
| `RankingModel`, `EmbeddingModel` | The two matching models |
| `NerModel` | Turkish NER model for organisation names |

## Pipeline

Stages:
1. OCR
2. Read the summary page
3. Table extraction
4. Normalization
5. Line hierarchy matching
6. Summary - Note table matching
7. Evaluate results


## Models

### OCR

I used `easyocr` for OCR because it was easy to setup. I need to share this code and since I haven't docerized the project easy setup seemed a priority.

### Candidate ranker

I choose `cross-encoder/ms-marco-MiniLM-L6-v2` because it can handle multiple languages and the search space is limited. Since it's not ranking millions of documents a Cross-encoder can work for this case. Also I expect a cross encoder to work better than say a Bi-encoder or an embedding model.

I choose `intfloat/multilingual-e5-small` because first of all it is multilingual. Second it is a fallback option.


**Fallbacks:**
- If a note heading isn't found, the whole page is used.
- If no total column is found, the model picks the column.

### Organisations (NER)

`akdeniz27/bert-base-turkish-cased-ner` is used for finding organizations in the reference pages.


## Checks

Matches between financial columns are used as confidance scores.

## Output

`output.json` keeps the stages separate, so each one can be checked on its own:

```
summary:         [{page, title, currency, columns, table rows}]
referance:       [{note, page, table_no, title, currency, columns, table rows}]
line_hierarchy:  {page: {confidance, hierarchies}}
cell_matches:    {method: {confidance, control, matches}}
organizations:   [{page, name, score}]
```
