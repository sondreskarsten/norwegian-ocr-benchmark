"""Monitor — reads JSONL heartbeats from GCS and prints live status.

Usage:
  python -m scripts.monitor          # one snapshot
  python -m scripts.monitor --watch  # refresh every 30s
"""
import argparse
import json
import time
from datetime import datetime

from google.cloud import storage

from scripts.config import DATA_BUCKET, HEARTBEAT_PREFIX


def latest_per_engine():
    cli = storage.Client()
    out = {}
    for blob in cli.list_blobs(DATA_BUCKET, prefix=f'{HEARTBEAT_PREFIX}/'):
        if not blob.name.endswith('.jsonl'):
            continue
        engine = blob.name.split('/')[-1].replace('.jsonl', '')
        try:
            lines = blob.download_as_text().strip().split('\n')
            if not lines:
                continue
            out[engine] = json.loads(lines[-1])
        except Exception:
            continue
    return out


def fmt_age(ts):
    age = time.time() - ts
    if age < 60:
        return f'{age:.0f}s'
    if age < 3600:
        return f'{age/60:.0f}m'
    return f'{age/3600:.1f}h'


def render(snap):
    rows = []
    for engine, h in sorted(snap.items()):
        n_total  = h.get('n_total') or 0
        n_done   = h.get('n_done') or 0
        n_ok     = h.get('n_ok') or 0
        n_err    = h.get('n_err') or 0
        pct      = (n_done / n_total * 100) if n_total else 0
        rate     = h.get('throughput_per_min') or 0
        eta_min  = ((n_total - n_done) / rate) if rate else None
        eta_str  = f'{eta_min:.0f}m' if eta_min and eta_min < 60 else (
                   f'{eta_min/60:.1f}h' if eta_min else '?')
        kind     = h.get('kind', '?')
        age      = fmt_age(h.get('ts', 0))
        vram     = h.get('gpu_mem_mb') or 0
        ram      = (h.get('ram_mb') or 0) // 1024
        rows.append(
            f'{engine:<14} {kind:<10} done={n_done:>5}/{n_total:<5} '
            f'pct={pct:>5.1f}% ok={n_ok:>5} err={n_err:>3} '
            f'rate={rate:>5.1f}/min eta={eta_str:<6} '
            f'vram={vram:>5}MiB ram={ram:>3}GiB age={age}'
        )
    if not rows:
        print('(no heartbeats found)')
        return
    print('  ' + '\n  '.join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=int, default=30)
    args = ap.parse_args()
    if args.watch:
        while True:
            print(f'\n=== {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ===')
            render(latest_per_engine())
            time.sleep(args.interval)
    else:
        render(latest_per_engine())


if __name__ == '__main__':
    main()
