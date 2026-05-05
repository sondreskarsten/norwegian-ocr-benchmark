"""Central config — env-var knobs + canonical GCS paths."""
import os
from pathlib import Path

PROJECT          = os.getenv('PROJECT', 'sondreskarsten-d7d14')
DATA_BUCKET      = os.getenv('DATA_BUCKET', 'sondre_brreg_data')
PDF_BUCKET       = os.getenv('PDF_BUCKET', 'brreg-regnskap')

RESULTS_PREFIX   = os.getenv('RESULTS_PREFIX', 'raw/ocr_bench_11k')
HEARTBEAT_PREFIX = f'{RESULTS_PREFIX}/_heartbeat'
CALIBRATION_BLOB = f'{RESULTS_PREFIX}/_calibration.json'

LOCAL_PDF_DIR    = Path(os.getenv('LOCAL_PDF_DIR', '/content/pdfs'))
LOCAL_TMP        = Path(os.getenv('LOCAL_TMP', '/content/tmp_ocr_bench'))
MANIFEST_LOCAL   = Path(os.getenv('MANIFEST_LOCAL', '/content/ocr_bench_manifest.json'))

DPI              = int(os.getenv('DPI', '300'))
MAX_PDFS         = os.getenv('MAX_PDFS')
MAX_PDFS         = int(MAX_PDFS) if MAX_PDFS and MAX_PDFS.lower() != 'none' else None

ENGINES_FILTER   = os.getenv('ENGINES')
ENGINES_FILTER   = [e.strip() for e in ENGINES_FILTER.split(',')] if ENGINES_FILTER else None

HEARTBEAT_EVERY_N = int(os.getenv('HEARTBEAT_EVERY_N', '10'))
HEARTBEAT_EVERY_S = int(os.getenv('HEARTBEAT_EVERY_S', '60'))

RCLONE_TRANSFERS  = int(os.getenv('RCLONE_TRANSFERS', '32'))

LOCAL_TMP.mkdir(parents=True, exist_ok=True)
