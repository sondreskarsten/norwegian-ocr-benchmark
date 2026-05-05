"""Build ocr_benchmark_11k.ipynb — Colab thin-shell orchestrator.

All logic lives in scripts/. The notebook only:
  1. Authenticates Colab to Google Cloud (no SA key)
  2. Clones the repo and installs base deps
  3. Pulls all 12,879 PDFs locally with rclone
  4. Pulls the latest calibration plan from GCS
  5. Launches all engines in background subprocesses (CPU+GPU groups)
  6. Tails heartbeats from GCS so progress survives Colab disconnects
"""
import json

cells = []


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {},
                  'source': s.splitlines(keepends=True)})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {},
                  'source': s.splitlines(keepends=True),
                  'execution_count': None, 'outputs': []})


md("""# OCR / image→text benchmark — Colab thin shell

All logic in [`scripts/`](https://github.com/sondreskarsten/norwegian-ocr-benchmark/tree/main/scripts). \
This notebook only orchestrates: auth → install → download PDFs → launch parallel engine processes → live monitor.

**Architecture:**

- **Calibration** (one-off, run on a spot VM, NOT in this notebook): \
  measures per-engine init-time, p50/p95 wall, VRAM/RAM peak, computes a parallelism plan, \
  writes `gs://sondre_brreg_data/raw/ocr_bench_11k/_calibration.json`.
- **Sweep** (this notebook): launches each engine as an independent background process \
  (`subprocess.Popen`), each writing per-PDF result blobs to GCS and a JSONL heartbeat \
  to `_heartbeat/{engine}.jsonl`.
- **Resume**: every script skips orgnrs that already have a result blob. Crash, preempt, \
  or close the tab — nothing is lost.

---""")

md("""## 1 · Auth (Colab built-in)""")

code("""from google.colab import auth as colab_auth
colab_auth.authenticate_user()

PROJECT = 'sondreskarsten-d7d14'
import os
os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT

!gcloud config set project {PROJECT} 2>&1 | tail -1

from google.cloud import storage
cli = storage.Client(project=PROJECT)
print('project:', cli.project)
print('bucket sondre_brreg_data exists:', cli.bucket('sondre_brreg_data').exists())""")

md("""## 2 · Clone repo + install base deps""")

code("""!cd /content && rm -rf norwegian-ocr-benchmark
!cd /content && git clone -q https://github.com/sondreskarsten/norwegian-ocr-benchmark.git
%cd /content/norwegian-ocr-benchmark
!pip install -q -r requirements.txt 2>&1 | tail -3

import sys
sys.path.insert(0, '/content/norwegian-ocr-benchmark')

import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('VRAM total MiB:', torch.cuda.get_device_properties(0).total_memory // (1024*1024))""")

md("""## 3 · Pull all PDFs locally with rclone

Engines never touch GCS during inference — they read from `/content/pdfs/{orgnr}/aarsregnskap_{year}.pdf`.""")

code("""!python -m scripts.download_pdfs""")

md("""## 4 · Calibration plan

Pulls the latest plan from GCS. If none exists yet, runs `scripts.calibrate --n 10` \
in-place to measure each engine on 10 PDFs (~20-40 min, depending on which model \
weights have to download).

Set `FORCE_RECALIBRATE = True` to re-measure even if a plan already exists.""")

code("""from google.cloud import storage
import json, subprocess, sys

FORCE_RECALIBRATE = False

cli = storage.Client()
blob = cli.bucket('sondre_brreg_data').blob('raw/ocr_bench_11k/_calibration.json')

if blob.exists() and not FORCE_RECALIBRATE:
    cal = json.loads(blob.download_as_text())
    open('engine_calibration.json','w').write(json.dumps(cal, indent=2, default=str))
    print('using existing calibration from GCS')
else:
    print('no calibration found — running scripts.calibrate --n 10 now')
    rc = subprocess.run([sys.executable, '-m', 'scripts.calibrate', '--n', '10'],
                        check=False).returncode
    if rc != 0:
        raise SystemExit(f'calibration exited {rc} — see output above')
    cal = json.loads(open('engine_calibration.json').read())

print('\\nplan:')
print(json.dumps(cal.get('plan'), indent=2))""")

md("""## 5 · Launch parallel engine processes

Each engine runs as an independent background subprocess. CPU engines all run truly \
parallel; GPU engines are bucketed into groups whose summed VRAM fits the device. \
This cell returns once everything is launched — actual sweep continues in the background.

Edit `MAX_PDFS` to limit per engine. Set to `''` for full 12,879 sweep.""")

code("""import subprocess, sys

MAX_PDFS = ''  # '' = full sweep; '500' = first 500 PDFs per engine

cmd = [sys.executable, '-m', 'scripts.parallel_launcher',
       '--calibration', 'engine_calibration.json']
if MAX_PDFS:
    cmd += ['--max-pdfs', str(MAX_PDFS)]

print('starting:', ' '.join(cmd))
p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
print('parent launcher pid:', p.pid)
print('individual engine logs are at /content/norwegian-ocr-benchmark/logs/{engine}.log')""")

md("""## 6 · Live monitor — tails heartbeats from GCS

Refreshes every 30s. Safe to interrupt; engines keep running. Re-run anytime.""")

code("""!python -m scripts.monitor --watch""")

md("""## 7 · Aggregate — once the sweep is done""")

code("""!python -m scripts.aggregate""")


nb = {'cells': cells,
      'metadata': {
          'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python', 'version': '3.10'},
          'colab': {'provenance': []}},
      'nbformat': 4, 'nbformat_minor': 5}

with open('notebooks/ocr_benchmark_11k.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f'wrote notebook: {len(json.dumps(nb))} bytes, {len(cells)} cells')
