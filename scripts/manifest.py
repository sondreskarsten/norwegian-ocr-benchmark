"""Manifest builder + Gemini ground-truth fetch.

Manifest = every (orgnr, year) pair where both a Gemini JSON and a PDF exist.
Cached locally after first build.
"""
import json
from google.cloud import storage

from scripts.config import PROJECT, DATA_BUCKET, MANIFEST_LOCAL


def get_client():
    return storage.Client(project=PROJECT)


def build_manifest(cli=None):
    cli = cli or get_client()
    out = []
    for blob in cli.list_blobs(DATA_BUCKET, prefix='raw/noter_extraction_2025/raw/'):
        n = blob.name.split('/')[-1]
        if not n.endswith('.json'):
            continue
        parts = n.replace('.json', '').split('_')
        if len(parts) < 4 or parts[1] != 'aarsregnskap':
            continue
        orgnr, year = parts[0], parts[2]
        if not (orgnr.isdigit() and year.isdigit()):
            continue
        out.append({
            'orgnr': orgnr,
            'year': int(year),
            'gemini_blob': blob.name,
            'pdf_bucket': 'brreg-regnskap',
            'pdf_blob': f'regnskap/{orgnr}/aarsregnskap_{year}.pdf',
        })
    return out


def load_or_build_manifest():
    if MANIFEST_LOCAL.exists():
        manifest = json.loads(MANIFEST_LOCAL.read_text())
    else:
        manifest = build_manifest()
        MANIFEST_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_LOCAL.write_text(json.dumps(manifest))
    manifest.sort(key=lambda r: (r['orgnr'], r['year']))
    return manifest


def fetch_gemini(cli, orgnr, year):
    blobs = list(cli.list_blobs(
        DATA_BUCKET,
        prefix=f'raw/noter_extraction_2025/raw/{orgnr}_aarsregnskap_{year}_'))
    if not blobs:
        return None
    blobs.sort(key=lambda b: int(b.name.split('_v')[-1].replace('.json', '')) if '_v' in b.name else 0)
    return json.loads(blobs[-1].download_as_bytes())


if __name__ == '__main__':
    m = load_or_build_manifest()
    print(f'manifest: {len(m)} pdfs')
    print(f'first 3: {m[:3]}')
