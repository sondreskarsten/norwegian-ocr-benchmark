# Norwegian OCR Benchmark on 12,879 årsregnskap PDFs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sondreskarsten/norwegian-ocr-benchmark/blob/main/notebooks/ocr_benchmark_11k.ipynb)

Benchmark every image-input OCR / document-understanding engine against Gemini 2.5 Flash structured ground truth on 12,879 Norwegian årsregnskap PDFs.

## How to run

Click the Colab badge → **Runtime → Run all** → click the auth popup once. That's it.

No service-account key, no setup. Auth uses Colab's built-in Google login. The notebook clones the runner repo, builds the manifest, and starts processing.

Set `MAX_PDFS = None` in the driver cell to sweep all 12,879 PDFs. Default is 100 per engine for a fast smoke run.

## What it measures

Three OCR-only metrics per `(engine × pdf)`, all computed against the Gemini ground truth for that filing:

| metric | definition |
|---|---|
| **Numeric recall** | of the integers Gemini extracted, fraction that appear in engine text (handles `241101` / `241 101` / `241.101`) |
| **Label recall** | of the line-item labels Gemini extracted, fraction appearing as substrings in engine text |
| **Distress recall** | for filings flagged with `going_concern_mentioned`, fraction of 5 standard Norwegian distress phrases captured |

## Engines (17 enabled by default)

| category | engines |
|---|---|
| OCR (Norwegian-aware) | tesseract, paddleocr, ocrmypdf, easyocr |
| OCR (Latin-only) | doctr, nougat, surya, trocr |
| Document understanding | docling, marker, pix2struct, donut, udop, layoutlmv3, lilt |
| Table extraction | camelot, tabula |

Each engine installs its own deps lazily. Comment any engine out in the driver cell to skip.

## Restartable

Per-engine state at `gs://sondre_brreg_data/raw/ocr_bench_11k/{engine}/{orgnr}.json`. Re-running the notebook picks up where you left off — already-processed PDFs are skipped automatically.

## Architecture

```
manifest (12,879 PDFs)
   ▼
for engine in ENGINES:
    factory()                          ← lazy install + GPU init, once per engine
    for pdf in pending_for_engine:     ← skips already-done via GCS blob check
        render PDF → 300 DPI PNGs      ← PyMuPDF
        engine_fn(orgnr, bundle)       ← OCR call → text
        score vs gemini_json           ← numeric / label / distress recall
        upload result to GCS
```

## Data sources

- **PDFs**: `gs://brreg-regnskap/regnskap/{orgnr}/aarsregnskap_{year}.pdf`
- **Gemini ground truth**: `gs://sondre_brreg_data/raw/noter_extraction_2025/raw/{orgnr}_aarsregnskap_{year}_v{N}.json`
- **Benchmark output**: `gs://sondre_brreg_data/raw/ocr_bench_11k/{engine}/{orgnr}.json`
- **Aggregate** (per-engine summary): `gs://sondre_brreg_data/raw/ocr_bench_11k/_aggregate_latest.json`

## GPU recommendations

| engine | min VRAM |
|---|---|
| tesseract / ocrmypdf / camelot / tabula | none (CPU) |
| easyocr / paddleocr | 4 GB |
| doctr / pix2struct / donut / udop | 8 GB |
| trocr / surya / nougat | 12 GB |
| marker / docling | 16 GB |

Colab Pro+ A100 handles all of these comfortably.

## Companion repos

- [`sondreskarsten/ocr-cascade-eval`](https://github.com/sondreskarsten/ocr-cascade-eval) — Cloud Run Jobs version of the same eval harness, runs 76 models against a 10-PDF fixture autonomously via Cloud Scheduler.
