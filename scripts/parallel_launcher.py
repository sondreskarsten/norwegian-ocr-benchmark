"""Parallel launcher.

Reads engine_calibration.json (local or from GCS), then for each engine in the
plan spawns `python -m scripts.runner --engine <name>` as a background
subprocess. Logs go to logs/{engine}.log. Returns immediately with a list of
(engine, pid) — caller polls via scripts.monitor.

Schedule:
  - All engines in plan.cpu_group launch immediately (truly parallel processes).
  - GPU groups process sequentially: launch all engines in group[0] in
    parallel, wait for them to finish, then group[1], etc.

Usage:
  python -m scripts.parallel_launcher [--calibration engine_calibration.json]
                                      [--cpu-only] [--gpu-only]
                                      [--max-pdfs 100]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google.cloud import storage

from scripts.config import DATA_BUCKET, CALIBRATION_BLOB


def load_calibration(path=None):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    cli = storage.Client()
    blob = cli.bucket(DATA_BUCKET).blob(CALIBRATION_BLOB)
    if not blob.exists():
        raise SystemExit(
            f'no calibration found locally or at gs://{DATA_BUCKET}/{CALIBRATION_BLOB}; '
            f'run `python -m scripts.calibrate --n 10` on a GPU host first')
    return json.loads(blob.download_as_text())


def launch_engine(engine, log_dir, max_pdfs=None):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{engine}.log'
    cmd = [sys.executable, '-m', 'scripts.runner', '--engine', engine]
    if max_pdfs is not None:
        cmd += ['--max-pdfs', str(max_pdfs)]
    f = open(log_path, 'ab')
    f.write(f'\n=== {time.ctime()} :: launching {" ".join(cmd)} ===\n'.encode())
    f.flush()
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return p, log_path


def wait_group(procs):
    while procs:
        time.sleep(5)
        still = []
        for engine, p, log in procs:
            if p.poll() is None:
                still.append((engine, p, log))
            else:
                rc = p.returncode
                tag = 'OK' if rc == 0 else f'EXIT_{rc}'
                print(f'  [{time.strftime("%H:%M:%S")}] {engine}: {tag} (log: {log})',
                      flush=True)
        procs = still


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration', default='engine_calibration.json')
    ap.add_argument('--cpu-only', action='store_true')
    ap.add_argument('--gpu-only', action='store_true')
    ap.add_argument('--max-pdfs', type=int, default=None)
    ap.add_argument('--engines', default=None,
                    help='comma-separated subset of engines to launch '
                         '(intersected with the calibration plan)')
    ap.add_argument('--log-dir', default='logs')
    args = ap.parse_args()

    cal = load_calibration(args.calibration)
    plan = cal.get('plan') or {}
    log_dir = Path(args.log_dir)

    cpu_group = plan.get('cpu_group') or []
    gpu_groups = plan.get('gpu_groups') or []

    if args.engines:
        wanted = set(e.strip() for e in args.engines.split(','))
        cpu_group = [e for e in cpu_group if e in wanted]
        gpu_groups = [[e for e in g if e in wanted] for g in gpu_groups]
        gpu_groups = [g for g in gpu_groups if g]
        print(f'engines filter: {sorted(wanted)}', flush=True)

    print(f'plan: {len(cpu_group)} cpu engines, {len(gpu_groups)} gpu groups, '
          f'broken={plan.get("broken", [])}', flush=True)

    cpu_procs = []
    if not args.gpu_only:
        for e in cpu_group:
            p, lp = launch_engine(e, log_dir, max_pdfs=args.max_pdfs)
            cpu_procs.append((e, p, lp))
            print(f'  launched cpu: {e} pid={p.pid} → {lp}', flush=True)

    if args.cpu_only:
        print('cpu-only: waiting for cpu group to finish', flush=True)
        wait_group(cpu_procs)
        return

    for gi, grp in enumerate(gpu_groups):
        print(f'\n--- gpu group {gi+1}/{len(gpu_groups)}: {grp} ---', flush=True)
        gpu_procs = []
        for e in grp:
            p, lp = launch_engine(e, log_dir, max_pdfs=args.max_pdfs)
            gpu_procs.append((e, p, lp))
            print(f'  launched gpu: {e} pid={p.pid} → {lp}', flush=True)
        wait_group(gpu_procs)

    if cpu_procs:
        print('\nwaiting on cpu group to finish', flush=True)
        wait_group(cpu_procs)


if __name__ == '__main__':
    main()
