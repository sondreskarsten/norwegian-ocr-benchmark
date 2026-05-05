"""Calibration harness — run on spot VM with A100 80GB.

For each engine, samples N PDFs (default 10), measures:
  - init_s            (factory wall time)
  - p50_s, p95_s      (per-PDF wall time)
  - vram_peak_mb      (torch.cuda.max_memory_allocated)
  - ram_peak_mb       (resource.ru_maxrss)
  - works             (bool — whether at least one PDF processed without exception)
  - n_chars_p50       (median ocr_text length, sanity check)
  - error             (first error string if works=False)

Output: writes engine_calibration.json locally AND uploads to
gs://{DATA_BUCKET}/{RESULTS_PREFIX}/_calibration.json.

Then derives a parallelism plan:
  - cpu_group: list of CPU engines safe to run truly in parallel (one process each)
  - gpu_groups: list of lists; each inner list = engines whose summed vram_peak
    fits in (vram_total - 4 GB headroom). Groups run sequentially; engines
    within a group run in parallel.

Usage:
  python -m scripts.calibrate --n 10
"""
import argparse
import gc
import json
import os
import resource
import statistics
import time
from pathlib import Path

from google.cloud import storage

from scripts.config   import (DATA_BUCKET, RESULTS_PREFIX, CALIBRATION_BLOB,
                              LOCAL_TMP, DPI)
from scripts.engines  import ENGINE_FACTORIES, DEVICE_HINT
from scripts.manifest import load_or_build_manifest, get_client
from scripts.runner   import render_pdf_to_pngs


def _vram_mb():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except Exception:
        return 0


def _vram_total_mb():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:
        return 0


def _ram_mb():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def calibrate_one(cli, engine_name, sample, dpi=150, push_progress=None):
    print(f'\n=== {engine_name} ===', flush=True)
    record = {
        'engine': engine_name,
        'works': False,
        'n_sampled': 0, 'n_ok': 0, 'n_err': 0,
        'init_s': None, 'p50_s': None, 'p95_s': None,
        'vram_peak_mb': None, 'ram_peak_mb': None,
        'n_chars_p50': None,
        'device': DEVICE_HINT.get(engine_name, 'unknown'),
        'error': None,
    }

    # Reset peak counters
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # Init
    t_init = time.time()
    try:
        engine_fn = ENGINE_FACTORIES[engine_name]()
        record['init_s'] = round(time.time() - t_init, 2)
    except Exception as e:
        record['error'] = f'factory: {type(e).__name__}: {e}'
        record['init_s'] = round(time.time() - t_init, 2)
        print(f'  FACTORY FAILED: {record["error"]}', flush=True)
        return record

    if push_progress:
        push_progress('init_done', engine_name, init_s=record['init_s'])

    walls, n_chars = [], []
    for i, row in enumerate(sample):
        record['n_sampled'] += 1
        work = LOCAL_TMP / f'cal_{engine_name}_{row["orgnr"]}'
        t0 = time.time()
        try:
            pdf_local, page_paths, page_size = render_pdf_to_pngs(cli, row, work, dpi=dpi)
            bundle = {'pdf': pdf_local, 'page_imgs': page_paths,
                      'page_size': page_size, 'n_pages': len(page_paths),
                      'page_text': {}, 'page_words': {}, 'full_text': '',
                      'orgnr': row['orgnr'], 'year': row['year']}
            out = engine_fn(row['orgnr'], bundle)
            wall = time.time() - t0
            walls.append(wall)
            text = out.get('full_text') or '\n'.join(p.get('text', '') for p in (out.get('pages') or []))
            n_chars.append(len(text))
            record['n_ok'] += 1
            print(f'  [{i+1}/{len(sample)}] {row["orgnr"]} wall={wall:.1f}s chars={len(text)}',
                  flush=True)
            if push_progress:
                push_progress('pdf_done', engine_name,
                              i=i+1, n_total=len(sample),
                              orgnr=row['orgnr'], wall_s=round(wall, 1),
                              n_chars=len(text), n_pages=len(page_paths))
        except Exception as e:
            record['n_err'] += 1
            if record['error'] is None:
                record['error'] = f'engine: {type(e).__name__}: {str(e)[:200]}'
            print(f'  [{i+1}/{len(sample)}] {row["orgnr"]} ERR: {type(e).__name__}: {e}',
                  flush=True)
            if push_progress:
                push_progress('pdf_err', engine_name,
                              i=i+1, n_total=len(sample),
                              orgnr=row['orgnr'],
                              error=f'{type(e).__name__}: {str(e)[:120]}')
        finally:
            for p in work.glob('*'):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                work.rmdir()
            except Exception:
                pass

    record['vram_peak_mb'] = _vram_mb()
    record['ram_peak_mb']  = _ram_mb()
    if walls:
        record['p50_s'] = round(statistics.median(walls), 2)
        record['p95_s'] = round(statistics.quantiles(walls, n=20)[18] if len(walls) >= 5
                                else max(walls), 2)
    if n_chars:
        record['n_chars_p50'] = int(statistics.median(n_chars))
    record['works'] = record['n_ok'] > 0

    # Free model
    try:
        del engine_fn
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return record


def derive_parallelism_plan(records, vram_total_mb, vram_headroom_mb=4096):
    cpu_group = sorted(
        [r['engine'] for r in records.values()
         if r['works'] and r['device'] == 'cpu' and r['vram_peak_mb'] < 200])
    gpu_engines = sorted(
        [r for r in records.values()
         if r['works'] and r['device'] == 'gpu' and r['vram_peak_mb'] >= 200],
        key=lambda r: -r['vram_peak_mb'])
    budget = max(vram_total_mb - vram_headroom_mb, 0)
    gpu_groups = []
    for r in gpu_engines:
        placed = False
        for grp in gpu_groups:
            grp_used = sum(records[e]['vram_peak_mb'] for e in grp)
            if grp_used + r['vram_peak_mb'] <= budget:
                grp.append(r['engine'])
                placed = True
                break
        if not placed:
            gpu_groups.append([r['engine']])
    return {
        'vram_total_mb': vram_total_mb,
        'vram_budget_mb': budget,
        'cpu_group': cpu_group,
        'gpu_groups': gpu_groups,
        'broken': [r['engine'] for r in records.values() if not r['works']],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10, help='PDFs sampled per engine')
    ap.add_argument('--engines', default=None, help='comma-separated subset')
    ap.add_argument('--skip', default='tesseract,ocrmypdf,layoutlmv3,lilt,camelot,tabula',
                    help='comma-separated engines to skip during calibration')
    ap.add_argument('--dpi', type=int, default=150,
                    help='rasterization DPI for calibration (sweep can use higher)')
    ap.add_argument('--out', default='engine_calibration.json')
    args = ap.parse_args()

    cli = get_client()
    manifest = load_or_build_manifest()
    sample = manifest[:args.n]
    if args.engines:
        engines_to_run = args.engines.split(',')
    else:
        engines_to_run = list(ENGINE_FACTORIES.keys())
    skip = set(args.skip.split(',')) if args.skip else set()
    engines_to_run = [e for e in engines_to_run if e not in skip]
    print(f'engines: {engines_to_run}', flush=True)
    print(f'skipped: {sorted(skip)}', flush=True)
    print(f'dpi: {args.dpi}', flush=True)

    gcs_client = storage.Client()
    cal_blob = gcs_client.bucket(DATA_BUCKET).blob(CALIBRATION_BLOB)
    progress_blob = gcs_client.bucket(DATA_BUCKET).blob(
        f'{RESULTS_PREFIX}/_calibration_progress.jsonl')
    progress_lines = []

    def _push_progress(event, engine, **extra):
        line = {'ts': time.time(), 'event': event, 'engine': engine, **extra}
        progress_lines.append(json.dumps(line, default=str))
        try:
            progress_blob.upload_from_string('\n'.join(progress_lines) + '\n',
                                             content_type='application/x-ndjson')
        except Exception:
            pass

    records = {}
    for name in engines_to_run:
        _push_progress('start', name, n_done=len(records),
                       n_total=len(engines_to_run))
        rec = calibrate_one(cli, name, sample, dpi=args.dpi,
                            push_progress=_push_progress)
        records[name] = rec
        _push_progress('finish', name, works=rec['works'],
                       init_s=rec['init_s'], p50_s=rec['p50_s'],
                       vram_peak_mb=rec['vram_peak_mb'],
                       error=rec['error'])
        out = {'generated_at': time.time(),
               'sample_n': args.n,
               'records': records,
               'in_progress': True,
               'completed_engines': list(records.keys()),
               'remaining_engines': [e for e in engines_to_run
                                     if e not in records],
               'skipped_engines': sorted(skip)}
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        try:
            cal_blob.upload_from_string(json.dumps(out, indent=2, default=str),
                                        content_type='application/json')
        except Exception:
            pass

    plan = derive_parallelism_plan(records, _vram_total_mb())
    final = {'generated_at': time.time(), 'sample_n': args.n,
             'records': records, 'plan': plan, 'in_progress': False,
             'skipped_engines': sorted(skip)}
    Path(args.out).write_text(json.dumps(final, indent=2, default=str))
    cal_blob.upload_from_string(json.dumps(final, indent=2, default=str),
                                content_type='application/json')
    _push_progress('all_done', '_', n_engines=len(records))
    print(f'\nCalibration written: {args.out}', flush=True)
    print(f'Uploaded: gs://{DATA_BUCKET}/{CALIBRATION_BLOB}', flush=True)
    print(f'\nPlan:\n{json.dumps(plan, indent=2)}', flush=True)


if __name__ == '__main__':
    main()
