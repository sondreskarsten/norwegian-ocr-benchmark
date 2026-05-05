# Norwegian OCR benchmark

Benchmarks 17 image-input engines (OCR, document-understanding, vision-LMs, table extractors) on **12,879 Norwegian årsregnskap PDFs** against Gemini 2.5 Flash structured ground truth.

## Architecture

```
                        GCS  raw/ocr_bench_11k/
                          ├── _calibration.json     ◀── (1) spot VM writes
                          ├── _heartbeat/{e}.jsonl  ◀── (3) Colab engines write
                          ├── _aggregate_latest     ◀── (5) aggregate writes
                          └── {engine}/{orgnr}.json ◀── (3) per-PDF results

  (1) calibrate.py                     (2) download_pdfs.py
      one engine at a time                 rclone -> /content/pdfs/
      measures init/p50/p95/VRAM       (3) parallel_launcher.py
                                           one subprocess per engine
  ─── Spot A100 80GB ───              ─── Colab A100 HighRAM ───
                                       (4) monitor.py — live status
                                       (5) aggregate.py — final table
```

## Workflow

### 1 · Calibrate (one-off, spot VM, ~$2)

Spins up a spot A100 80GB, runs each engine on 10 PDFs, writes `_calibration.json` to GCS, and self-terminates.

```bash
gcloud compute instances create ocr-calibrate-spot \
  --project=sondreskarsten-d7d14 \
  --zone=europe-west4-a \
  --machine-type=a2-ultragpu-1g \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --image-family=common-cu124-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-ssd \
  --service-account=s1sfreracct@sondreskarsten-d7d14.iam.gserviceaccount.com \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=scripts/spot_calibrate_startup.sh
```

Watch the log:

```bash
gcloud compute ssh ocr-calibrate-spot --zone=europe-west4-a -- 'tail -f /var/log/calibrate.log'
```

When done, `gs://sondre_brreg_data/raw/ocr_bench_11k/_calibration.json` is populated and the VM auto-deletes.

### 2 · Sweep (Colab, free)

Open: https://colab.research.google.com/github/sondreskarsten/norwegian-ocr-benchmark/blob/main/notebooks/ocr_benchmark_11k.ipynb

Runtime → A100 + High-RAM → Run all.

1. Auth
2. Clone + install
3. `download_pdfs.py` — rclone all 12,879 PDFs to `/content/pdfs/`
4. Pull `_calibration.json` from GCS
5. `parallel_launcher.py` — spawns one subprocess per engine; CPU engines run truly parallel, GPU engines run in VRAM-budgeted groups
6. `monitor.py --watch` — tails heartbeats, refreshes every 30s
7. `aggregate.py` — final summary

### 3 · Monitor from anywhere

Heartbeats live in GCS, so you can watch from any shell:

```bash
python -m scripts.monitor --watch
gsutil cat gs://sondre_brreg_data/raw/ocr_bench_11k/_heartbeat/marker.jsonl | tail
```

## Module map

| Module | Role |
|---|---|
| `scripts/config.py` | env-var knobs and canonical GCS paths |
| `scripts/manifest.py` | builds 12,879-row `(orgnr,year,pdf_blob,gemini_blob)` manifest |
| `scripts/scoring.py` | numeric / label / distress recall |
| `scripts/engines.py` | 17 engine factories (verbatim from earlier notebook) |
| `scripts/heartbeat.py` | JSONL heartbeat writer to GCS |
| `scripts/runner.py` | single-engine sweep over manifest with heartbeat |
| `scripts/download_pdfs.py` | rclone → /content/pdfs/ (gcs-client fallback) |
| `scripts/calibrate.py` | per-engine 10-PDF calibration, derives parallelism plan |
| `scripts/parallel_launcher.py` | spawns one subprocess per engine |
| `scripts/monitor.py` | live status from GCS heartbeats |
| `scripts/aggregate.py` | per-engine summary table |
| `scripts/spot_calibrate_startup.sh` | spot-VM startup script |

## Env-var knobs

| Var | Default | Effect |
|---|---|---|
| `MAX_PDFS` | `None` | per-engine PDF cap |
| `ENGINES` | all 17 | comma-separated subset |
| `RESULTS_PREFIX` | `raw/ocr_bench_11k` | GCS prefix for all artifacts |
| `LOCAL_PDF_DIR` | `/content/pdfs` | local rclone target |
| `DPI` | `300` | rasterization resolution |
| `HEARTBEAT_EVERY_N` | `10` | flush every N PDFs |
| `HEARTBEAT_EVERY_S` | `60` | flush every S seconds |
| `RCLONE_TRANSFERS` | `32` | parallel transfers |
