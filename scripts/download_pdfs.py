"""Download all PDFs in the manifest to LOCAL_PDF_DIR.

Tries rclone first (configured against application-default credentials), falls
back to `gcloud storage cp` parallel. Idempotent: skips files that already
exist with non-zero size.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage

from scripts.config import LOCAL_PDF_DIR, RCLONE_TRANSFERS, PROJECT
from scripts.manifest import load_or_build_manifest


def _has_rclone():
    return shutil.which('rclone') is not None


def _ensure_rclone():
    if _has_rclone():
        return
    print('rclone not found; installing…', flush=True)
    subprocess.run('curl -fsSL https://rclone.org/install.sh | bash',
                   shell=True, check=True)


def _rclone_config_gcs():
    """Configure an rclone remote 'gcs' that uses ADC."""
    cfg = subprocess.run(['rclone', 'config', 'show'],
                         capture_output=True, text=True).stdout
    if '[gcs]' in cfg:
        return
    subprocess.run([
        'rclone', 'config', 'create', 'gcs', 'google cloud storage',
        'env_auth', 'true',
        'project_number', PROJECT,
    ], check=True, capture_output=True)


def download_with_rclone(manifest):
    _ensure_rclone()
    _rclone_config_gcs()
    LOCAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
    # Build a manifest file rclone can use with --files-from
    files_from = LOCAL_PDF_DIR.parent / '_rclone_files_from.txt'
    files_from.write_text('\n'.join(r['pdf_blob'] for r in manifest))
    print(f'rclone copy: {len(manifest)} pdfs → {LOCAL_PDF_DIR}', flush=True)
    cmd = [
        'rclone', 'copy', f'gcs:brreg-regnskap', str(LOCAL_PDF_DIR.parent / 'brreg-regnskap'),
        '--files-from', str(files_from),
        '--transfers', str(RCLONE_TRANSFERS),
        '--checkers', str(RCLONE_TRANSFERS),
        '--progress',
        '--stats', '15s',
    ]
    print('  ' + ' '.join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f'rclone exited {rc}')
    # rclone places files under …/brreg-regnskap/regnskap/{orgnr}/aarsregnskap_{year}.pdf
    # We want them at LOCAL_PDF_DIR/{orgnr}/aarsregnskap_{year}.pdf — symlink.
    src_root = LOCAL_PDF_DIR.parent / 'brreg-regnskap' / 'regnskap'
    if src_root.exists():
        for d in src_root.iterdir():
            target = LOCAL_PDF_DIR / d.name
            if not target.exists():
                target.symlink_to(d)


def download_with_gcs_client(manifest):
    """Fallback: parallel single-blob downloads using google-cloud-storage."""
    LOCAL_PDF_DIR.mkdir(parents=True, exist_ok=True)
    cli = storage.Client(project=PROJECT)
    bucket = cli.bucket('brreg-regnskap')

    def _one(row):
        local = LOCAL_PDF_DIR / row['orgnr'] / f"aarsregnskap_{row['year']}.pdf"
        if local.exists() and local.stat().st_size > 0:
            return 'skip'
        local.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(row['pdf_blob']).download_to_filename(str(local))
        return 'ok'

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=RCLONE_TRANSFERS) as ex:
        futures = {ex.submit(_one, r): r for r in manifest}
        for i, fut in enumerate(as_completed(futures)):
            try:
                s = fut.result()
                if s == 'ok':
                    n_ok += 1
                elif s == 'skip':
                    n_skip += 1
            except Exception:
                n_err += 1
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f'  [{i+1}/{len(manifest)}] ok={n_ok} skip={n_skip} err={n_err} '
                      f'rate={rate:.1f}/s', flush=True)
    print(f'done: ok={n_ok} skip={n_skip} err={n_err} elapsed={time.time()-t0:.1f}s',
          flush=True)


def main():
    manifest = load_or_build_manifest()
    print(f'manifest: {len(manifest)} pdfs', flush=True)
    if os.getenv('USE_RCLONE', '1') == '1' and _has_rclone() or os.getenv('FORCE_RCLONE') == '1':
        try:
            download_with_rclone(manifest)
            return
        except Exception as e:
            print(f'rclone failed: {e}; falling back to gcs client', flush=True)
    download_with_gcs_client(manifest)


if __name__ == '__main__':
    main()
