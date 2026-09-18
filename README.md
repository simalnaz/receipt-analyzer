# Receipt Analyzer

Update: Previous commit history has been cleared. Active development continues from this point with a focus on improving the overall architecture and model performance.

End-to-end receipt processing system built for learning. Given a receipt image, it extracts structured data using OCR and regex, classifies the spending category with a trained ML model, detects anomalies, and exposes everything through a REST API with an analytics dashboard.

**Dataset:** [SROIE v2](https://rrc.cvc.uab.es/?ch=13) — 626 train / 347 test Malaysian retail receipts.

---

## What it does

```
Receipt image
  → Tesseract OCR       → raw text + word-level confidence scores
  → Regex extraction    → company, address, date, total
  → TF-IDF + LR model  → spending category (grocery / fuel / restaurant / ...)
  → Anomaly detection   → OCR quality, amount range, missing fields, outliers
  → FastAPI + SQLite    → stored and served as JSON
  → Streamlit dashboard → accuracy metrics and category distribution
```

LayoutLM (transformer-based extraction) is also available as an optional upgrade path — disabled by default, requires fine-tuning via the Colab notebook first.

---

## Accuracy (347 SROIE test receipts)

### Field extraction — Tesseract OCR + Regex

| Field | Exact match | Token F1 |
|---|---|---|
| Date | 52.7% | 53.1% |
| Total | 29.4% | 29.4% |
| Company | 4.9% | 28.4% |
| Address | 0.0% | 20.1% |

Date works well. Company and address suffer from OCR noise on the top lines of receipts. Total is affected by Malay receipt formats (JUMLAH, AMAUN keywords, mixed decimal notation).

### Category classifier — TF-IDF + LogisticRegression

**64.1%** cross-validated accuracy (5-fold, 626 receipts). Falls back to keyword matching if the model file is missing.

---

## Tech stack

| Layer | Tool |
|---|---|
| OCR | pytesseract (Tesseract 5, PSM 4) |
| Image preprocessing | OpenCV + Pillow |
| Field extraction | Regex (`re`) |
| Classification | scikit-learn — TF-IDF + LogisticRegression |
| Anomaly detection | Rule-based + z-score statistical |
| API | FastAPI + uvicorn |
| Database | SQLite + SQLAlchemy |
| Dashboard | Streamlit |
| Optional LayoutLM | HuggingFace Transformers, fine-tuned on Colab |

---

## Project structure

```
app/
  core/
    ocr.py                Tesseract extraction, word-level confidence
    extractor.py          Regex field parsing (company, address, date, total)
    classifier.py         Keyword heuristic + trained TF-IDF/LR classifier
    anomaly.py            Rule-based + statistical anomaly detection
    pipeline.py           Orchestrates all steps
    layoutlm_extractor.py Optional LayoutLM-based field extraction
  db/
    database.py           SQLAlchemy engine + session factory
    models.py             Receipt table schema
    repository.py         CRUD operations
  models/
    schemas.py            Pydantic schemas
  utils/
    preprocessing.py      Image preprocessing (grayscale, denoise, deskew)
  api/
    routes.py             FastAPI endpoints
  main.py                 App entry point

dashboard/
  app.py                  Streamlit analytics dashboard (requires evaluation data)

notebooks/
  train_colab.ipynb       LayoutLM fine-tuning on Google Colab (T4 GPU)

scripts/
  batch_analyze.py            Bulk processing CLI
  evaluate_extraction.py      Field-level accuracy evaluation (exact + token F1)
  evaluate_layoutlm.py        LayoutLM field-level accuracy evaluation
  generate_category_labels.py Weak label generation + classifier training
  prepare_dataset.py          SROIE dataset preprocessing for LayoutLM
  train_layoutlm.py           LayoutLM fine-tuning (local, CPU/GPU)
  compare_box_vs_tesseract.py Bounding-box vs plain-text OCR comparison
  scoring.py                  Shared scoring utilities (exact match, token F1)

data/
  category_model.pkl      Trained classifier (required to run)

tests/
  test_extractor.py
  test_anomaly.py
  test_classifier.py
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Tesseract
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Ubuntu

# 3. Configure environment — create a .env file in the project root
DATABASE_URL=sqlite:///./receipts.db
TESSERACT_CMD=/usr/local/bin/tesseract   # adjust path if needed

# 4. Start the API
uvicorn app.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

The database (`receipts.db`) and upload directory (`data/uploads/`) are created automatically on first run.

---

## LayoutLM 

LayoutLM uses bounding-box-aware transformer attention for higher extraction accuracy. It is disabled by default (`use_layoutlm=False` in the pipeline).

To enable it:

1. Download the SROIE dataset and run `python scripts/prepare_dataset.py`
2. Open `notebooks/train_colab.ipynb` in Google Colab, select **T4 GPU**, and run all cells (~15 min)
3. Download the output zip and extract it into `data/layoutlm_finetuned/`
4. Pass `use_layoutlm=True` when calling the pipeline

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./receipts.db` | SQLAlchemy connection string |
| `TESSERACT_CMD` | system default | Path to Tesseract binary |
