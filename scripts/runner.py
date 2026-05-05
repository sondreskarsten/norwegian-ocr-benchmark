"""Single-engine runner.

Invoked as: python -m scripts.runner --engine tesseract [--max-pdfs 100]

Loads the engine, sweeps the manifest (skipping already-done orgnrs by checking
GCS), writes per-PDF result blobs and a JSONL heartbeat. Designed to be
launched in parallel with other engine runners as a subprocess.
"""
import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import fitz  # PyMuPDF — pre-installed expectations: see requirements.txt
from google.cloud import storage

from scripts.config       import (DATA_BUCKET, RESULTS_PREFIX, LOCAL_PDF_DIR,
                                  LOCAL_TMP, DPI, MAX_PDFS, ENGINES_FILTER)
from scripts.engines      import ENGINE_FACTORIES
from scripts.heartbeat    import Heartbeat
from scripts.manifest     import load_or_build_manifest, fetch_gemini, get_client
from scripts.scoring      import score_ocr_text


def already_done_set(cli, engine):
    return {b.name.split('/')[-1].replace('.json', '')
            for b in cli.list_blobs(DATA_BUCKET, prefix=f'{RESULTS_PREFIX}/{engine}/')}


def upload_result(cli, engine, orgnr, payload):
    cli.bucket(DATA_BUCKET).blob(f'{RESULTS_PREFIX}/{engine}/{orgnr}.json') \
        .upload_from_string(json.dumps(payload, ensure_ascii=False, default=str),
                            content_type='application/json')


def render_pdf_to_pngs(cli, row, out_dir, dpi=DPI):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_local = out_dir / 'src.pdf'
    # Prefer locally rclone'd PDF if it exists; else download from GCS.
    rclone_path = LOCAL_PDF_DIR / row['orgnr'] / f"aarsregnskap_{row['year']}.pdf"
    if rclone_path.exists():
        import shutil
        shutil.copy(str(rclone_path), str(pdf_local))
    else:
        cli.bucket(row['pdf_bucket']).blob(row['pdf_blob']).download_to_filename(str(pdf_local))
    doc = fitz.open(str(pdf_local))
    pages, sizes = [], []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        p = out_dir / f'p-{i+1:02d}.png'
        pix.save(str(p))
        pages.append(str(p))
        sizes.append([pix.width, pix.height])
    doc.close()
    return str(pdf_local), pages, sizes


def _cleanup(work):
    for p in work.glob('*'):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        work.rmdir()
    except Exception:
        pass


def benchmark_one(cli, engine_fn, engine_name, row):
    orgnr, year = row['orgnr'], row['year']
    t0 = time.time()
    work = LOCAL_TMP / f'{engine_name}_{orgnr}'
    try:
        pdf_local, page_paths, page_size = render_pdf_to_pngs(cli, row, work, dpi=DPI)
    except Exception as e:
        payload = {'orgnr': orgnr, 'year': year, 'stage': 'render',
                   'error': f'{type(e).__name__}: {str(e)[:200]}',
                   'wall_s': time.time() - t0}
        upload_result(cli, engine_name, orgnr, payload)
        return payload, 'err'
    bundle = {'pdf': pdf_local, 'page_imgs': page_paths,
              'page_size': page_size, 'n_pages': len(page_paths),
              'page_text': {}, 'page_words': {}, 'full_text': '',
              'orgnr': orgnr, 'year': year}
    try:
        out = engine_fn(orgnr, bundle)
    except Exception as e:
        payload = {'orgnr': orgnr, 'year': year, 'stage': 'engine',
                   'error': f'{type(e).__name__}: {str(e)[:300]}',
                   'traceback': traceback.format_exc()[-500:],
                   'wall_s': time.time() - t0}
        upload_result(cli, engine_name, orgnr, payload)
        _cleanup(work)
        return payload, 'err'
    ocr_text = out.get('full_text') or out.get('text') or ''
    if not ocr_text and 'pages' in out:
        ocr_text = '\n'.join(p.get('text', '') for p in out['pages'])
    gem = fetch_gemini(cli, orgnr, year)
    scores = score_ocr_text(ocr_text, gem) if gem else None
    payload = {
        'orgnr': orgnr, 'year': year, 'engine': engine_name,
        'wall_s': round(time.time() - t0, 2),
        'n_chars_ocr_text': len(ocr_text),
        'n_pages': len(page_paths),
        'engine_output_keys': list(out.keys()),
        'scores': scores,
        'full_text': ocr_text[:50000],
    }
    upload_result(cli, engine_name, orgnr, payload)
    _cleanup(work)
    gc.collect()
    return payload, 'ok'


def run_engine(engine_name, max_pdfs=None):
    if engine_name not in ENGINE_FACTORIES:
        raise SystemExit(f'unknown engine: {engine_name}')
    cli = get_client()
    manifest = load_or_build_manifest()
    done = already_done_set(cli, engine_name)
    pending = [r for r in manifest if r['orgnr'] not in done]
    if max_pdfs:
        pending = pending[:max_pdfs]
    print(f'[{engine_name}] done={len(done)} pending={len(pending)}', flush=True)
    if not pending:
        return
    hb = Heartbeat(engine_name, n_total=len(pending))
    t_init = time.time()
    try:
        engine_fn = ENGINE_FACTORIES[engine_name]()
    except Exception as e:
        hb.close(status=f'factory_failed:{type(e).__name__}')
        raise SystemExit(f'[{engine_name}] FACTORY FAILED: {type(e).__name__}: {e}')
    print(f'[{engine_name}] factory init: {time.time()-t_init:.1f}s', flush=True)
    for i, row in enumerate(pending):
        try:
            payload, status = benchmark_one(cli, engine_fn, engine_name, row)
        except Exception as e:
            payload = {'error': f'HARDFAIL: {type(e).__name__}: {e}'}
            status = 'err'
        hb.tick(orgnr=row['orgnr'], wall_s=payload.get('wall_s'), status=status)
        if (i + 1) % 25 == 0:
            sc = (payload.get('scores') or {})
            print(f"[{engine_name}] [{i+1}/{len(pending)}] {row['orgnr']} "
                  f"chars={payload.get('n_chars_ocr_text','?')} "
                  f"num={sc.get('numeric_recall')} wall={payload.get('wall_s','?')}s",
                  flush=True)
    hb.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--max-pdfs', type=int, default=MAX_PDFS)
    args = ap.parse_args()
    run_engine(args.engine, max_pdfs=args.max_pdfs)


if __name__ == '__main__':
    main()
