"""Per-engine summary table over all per-PDF result blobs."""
import json
from collections import defaultdict

from google.cloud import storage

from scripts.config  import DATA_BUCKET, RESULTS_PREFIX
from scripts.engines import ENGINE_FACTORIES


def aggregate(engines=None):
    cli = storage.Client()
    engines = engines or list(ENGINE_FACTORIES.keys())
    agg = defaultdict(lambda: {
        'n_pdfs': 0, 'n_with_scores': 0,
        'sum_num_recall': 0.0, 'sum_lab_recall': 0.0,
        'sum_distress_recall': 0.0, 'n_distress_eligible': 0,
        'wall_total_s': 0.0, 'errors': 0,
    })
    for engine in engines:
        for blob in cli.list_blobs(DATA_BUCKET, prefix=f'{RESULTS_PREFIX}/{engine}/'):
            if not blob.name.endswith('.json'):
                continue
            try:
                d = json.loads(blob.download_as_bytes())
            except Exception:
                continue
            a = agg[engine]
            a['n_pdfs'] += 1
            a['wall_total_s'] += d.get('wall_s') or 0
            if d.get('error'):
                a['errors'] += 1
                continue
            sc = d.get('scores') or {}
            if sc.get('numeric_recall') is not None:
                a['n_with_scores'] += 1
                a['sum_num_recall'] += sc['numeric_recall']
                a['sum_lab_recall'] += sc.get('label_recall') or 0
                if sc.get('distress_recall') is not None:
                    a['n_distress_eligible'] += 1
                    a['sum_distress_recall'] += sc['distress_recall']
    return dict(agg)


def main():
    agg = aggregate()
    print(f"{'engine':<14} {'n':<6} {'n_score':<8} {'avg_num':<10} "
          f"{'avg_lab':<10} {'avg_distr':<10} {'wall_total':<10} {'err':<5}")
    print('-' * 90)
    for engine, a in sorted(agg.items()):
        n = a['n_with_scores']
        avg_num = a['sum_num_recall'] / n if n else 0
        avg_lab = a['sum_lab_recall'] / n if n else 0
        avg_distr = (a['sum_distress_recall'] / a['n_distress_eligible']
                     if a['n_distress_eligible'] else 0)
        print(f"{engine:<14} {a['n_pdfs']:<6} {n:<8} {avg_num:<10.3f} "
              f"{avg_lab:<10.3f} {avg_distr:<10.3f} {a['wall_total_s']:<10.0f} "
              f"{a['errors']:<5}")
    import time
    out = {'generated_at': time.time(), 'aggregates': agg}
    storage.Client().bucket(DATA_BUCKET).blob(
        f'{RESULTS_PREFIX}/_aggregate_latest.json').upload_from_string(
            json.dumps(out, ensure_ascii=False, indent=2, default=str),
            content_type='application/json')
    print(f'\nsaved gs://{DATA_BUCKET}/{RESULTS_PREFIX}/_aggregate_latest.json')


if __name__ == '__main__':
    main()
