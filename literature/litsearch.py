# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Dudbil Olvasada Pabón Riaño

"""Literature search 2016-2026 via OpenAlex; downloads legal open-access PDFs.

Usage: python litsearch.py <out_dir> <catalog_json>
Each topic runs several precise queries (all words required, in title or abstract),
keeps the most cited works plus the most cited recent ones (>= 2022), and tries
every open-access location (arXiv, PMC, repositories, publisher) for the PDF.
"""
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
CATALOG = Path(sys.argv[2])
HDR = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36',
       'Accept': 'application/pdf,text/html;q=0.8,*/*;q=0.5'}
CTX = ssl.create_default_context()

# all words of a query must appear in title or abstract; phrases in quotes
TOPICS = {
    '01_Orden_cero_e_imagen_gemela': ['digital holography "zero-order"', 'digital holography "twin image"', 'off-axis holography "zero order" suppression'],
    '02_Holografia_corrimiento_de_fase': ['"phase-shifting digital holography"', '"parallel phase-shifting" holography', 'phase-shifting holography "single-shot"'],
    '03_Deep_learning_en_holografia': ['"digital holography" "deep learning"', 'holographic reconstruction "neural network"', 'holography "physics-informed" network'],
    '04_CGH_y_Gerchberg_Saxton': ['"computer-generated holography" neural', '"Gerchberg-Saxton" hologram', '"computer-generated hologram" algorithm phase-only'],
    '05_Microscopia_holografica_DHM': ['"digital holographic microscopy" "quantitative phase"', '"digital holographic microscopy" autofocus', 'holographic "phase unwrapping" microscopy'],
    '06_SLM_LCoS_calibracion': ['"liquid crystal on silicon" calibration phase', '"spatial light modulator" "phase modulation" calibration', 'LCoS "spatial light modulator" Jones'],
    '07_Pinzas_opticas_holograficas': ['"holographic optical tweezers"', '"holographic optical trapping"', 'optical tweezers "spatial light modulator"'],
    '08_OAM_y_haces_estructurados': ['"orbital angular momentum" "spatial light modulator"', '"orbital angular momentum" hologram generation', '"structured light" "spatial light modulator"'],
    '09_Reconstruccion_numerica_y_revisiones': ['"digital holography" review', '"angular spectrum" holography reconstruction', '"Fresnel" "digital holography" reconstruction algorithm'],
    '10_Educacion_y_laboratorios_virtuales': ['optics "virtual laboratory"', 'holography education students', 'optics teaching simulation interactive laboratory'],
}
N_TOP, N_RECENT = 10, 5


def get(url, timeout=45):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


def query(q):
    params = {'filter': f'title_and_abstract.search:{q},from_publication_date:2016-01-01',
              'sort': 'cited_by_count:desc', 'per-page': '50'}
    return json.loads(get('https://api.openalex.org/works?' + urllib.parse.urlencode(params)))['results']


def slug(text, n=80):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s-]', '', text, flags=re.U)).strip()[:n]


def record(w):
    loc = w.get('primary_location') or {}
    pdfs = []
    for l in [w.get('best_oa_location') or {}] + (w.get('locations') or []):
        u = l.get('pdf_url')
        if u and u not in pdfs:
            pdfs.append(u)
    # prefer repositories (arXiv, PMC, institutional) over publisher servers
    pdfs.sort(key=lambda u: 0 if any(k in u for k in ('arxiv', 'ncbi', 'europepmc', 'hal.', 'zenodo', 'repository', 'eprints')) else 1)
    return {
        'title': w.get('title') or '', 'year': w.get('publication_year'),
        'doi': (w.get('doi') or '').replace('https://doi.org/', ''),
        'venue': (loc.get('source') or {}).get('display_name', '') or '',
        'authors': [a['author']['display_name'] for a in w.get('authorships', [])],
        'cited_by': w.get('cited_by_count', 0),
        'is_oa': (w.get('open_access') or {}).get('is_oa', False),
        'pdf_urls': pdfs, 'openalex': w.get('id'), 'type': w.get('type'),
    }


def main():
    catalog = {}
    for topic, qs in TOPICS.items():
        pool = {}
        for q in qs:
            try:
                for w in query(q):
                    r = record(w)
                    if r['doi'] and r['title'] and r['type'] in ('article', 'review', 'preprint'):
                        pool.setdefault(r['doi'].lower(), r)
            except Exception as e:
                print(f'  [{topic}] consulta falló ({q}): {e}')
            time.sleep(0.3)
        ranked = sorted(pool.values(), key=lambda r: -r['cited_by'])
        top = ranked[:N_TOP]
        recent = [r for r in ranked if (r['year'] or 0) >= 2022 and r not in top][:N_RECENT]
        recs = top + recent
        folder = OUT / topic
        folder.mkdir(parents=True, exist_ok=True)
        for r in recs:
            r['file'] = ''
            first = r['authors'][0].split()[-1] if r['authors'] else 'Anon'
            dest = folder / f"{r['year']} {slug(first, 30)} - {slug(r['title'])}.pdf"
            for u in r['pdf_urls']:
                try:
                    data = get(u, timeout=60)
                    if data[:4] == b'%PDF' and len(data) > 30_000:
                        dest.write_bytes(data); r['file'] = str(dest.relative_to(OUT)); r['source_pdf'] = u
                        break
                except Exception as e:
                    r.setdefault('errors', []).append(f'{urllib.parse.urlparse(u).netloc}: {str(e)[:60]}')
                time.sleep(0.3)
        catalog[topic] = recs
        print(f'[{topic}] {len(recs)} trabajos ({len(pool)} candidatos), {sum(bool(r["file"]) for r in recs)} PDF')
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
