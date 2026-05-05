"""Heartbeat writer — appends one JSONL line per heartbeat to GCS.

Each engine's worker creates a Heartbeat instance pointing at
gs://{DATA_BUCKET}/{HEARTBEAT_PREFIX}/{engine}.jsonl. Lines flush every
HEARTBEAT_EVERY_N items processed OR HEARTBEAT_EVERY_S seconds, whichever
first. Final 'finished' line emitted on close().

GCS doesn't support append, so we read the existing blob, concat, re-upload.
This is fine because heartbeats are tiny (one line ~ 250 bytes, even 10K
items = ~2.5 MB).
"""
import json
import os
import time
import socket
from google.cloud import storage

from scripts.config import DATA_BUCKET, HEARTBEAT_PREFIX, HEARTBEAT_EVERY_N, HEARTBEAT_EVERY_S


def _gpu_mem_mb():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() // (1024 * 1024)
    except Exception:
        return None


def _ram_mb():
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: kb. macOS: bytes. Assume Linux (Colab + GCE).
        return int(kb / 1024)
    except Exception:
        return None


class Heartbeat:
    def __init__(self, engine, n_total, run_id=None):
        self.engine   = engine
        self.n_total  = n_total
        self.n_done   = 0
        self.n_ok     = 0
        self.n_err    = 0
        self.n_skip   = 0
        self.last_orgnr = None
        self.last_wall_s = None
        self.t_start  = time.time()
        self.t_last_flush = self.t_start
        self.run_id   = run_id or f'{int(self.t_start)}-{os.getpid()}'
        self.host     = socket.gethostname()
        self.pid      = os.getpid()
        self._cli     = storage.Client()
        self._blob    = self._cli.bucket(DATA_BUCKET).blob(f'{HEARTBEAT_PREFIX}/{engine}.jsonl')
        self._buffer  = []
        self._emit('started')

    def _emit(self, kind):
        elapsed = time.time() - self.t_start
        rate = (self.n_done / elapsed * 60) if elapsed > 0 else 0
        line = {
            'ts': time.time(),
            'kind': kind,
            'engine': self.engine,
            'pid': self.pid,
            'host': self.host,
            'run_id': self.run_id,
            'n_total': self.n_total,
            'n_done': self.n_done,
            'n_ok': self.n_ok,
            'n_err': self.n_err,
            'n_skip': self.n_skip,
            'last_orgnr': self.last_orgnr,
            'last_wall_s': self.last_wall_s,
            'elapsed_s': round(elapsed, 1),
            'throughput_per_min': round(rate, 2),
            'gpu_mem_mb': _gpu_mem_mb(),
            'ram_mb': _ram_mb(),
        }
        self._buffer.append(json.dumps(line, default=str))
        self._flush()

    def _flush(self):
        if not self._buffer:
            return
        try:
            existing = self._blob.download_as_text() if self._blob.exists() else ''
        except Exception:
            existing = ''
        new_content = existing + ('' if not existing or existing.endswith('\n') else '\n') \
                      + '\n'.join(self._buffer) + '\n'
        try:
            self._blob.upload_from_string(new_content, content_type='application/x-ndjson')
            self._buffer = []
            self.t_last_flush = time.time()
        except Exception:
            # Keep buffer; retry next flush
            pass

    def tick(self, orgnr, wall_s, status='ok'):
        self.n_done += 1
        self.last_orgnr = orgnr
        self.last_wall_s = wall_s
        if status == 'ok':
            self.n_ok += 1
        elif status == 'err':
            self.n_err += 1
        elif status == 'skip':
            self.n_skip += 1
        if (self.n_done % HEARTBEAT_EVERY_N == 0
                or time.time() - self.t_last_flush >= HEARTBEAT_EVERY_S):
            self._emit('progress')

    def close(self, status='finished'):
        self._emit(status)
