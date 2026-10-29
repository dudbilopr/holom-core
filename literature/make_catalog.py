"""Filter the OpenAlex results by relevance and write the vault catalog and a BibTeX file."""
import json
import re
import sys
from pathlib import Path

catalog = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
ROOT = Path(sys.argv[2])                     # .../Tesis
LIB = ROOT / 'Holografia' / 'Articulos 2016-2026'
NOTE = ROOT / 'Vault_Tesis_Holografia' / '08_Base_de_Conocimiento_Libros_y_Articulos' / '06_Literatura_2016_2026_OpenAlex.md'

KEEP = re.compile(r'hologra|holographic|spatial light modulator|\bslm\b|lcos|liquid crystal|phase retrieval|twin|zero.order|'
                  r'phase.shift|quantitative phase|phase imaging|phase unwrapping|optical trap|tweezer|optical vortic|orbital angular|'
                  r'structured light|laguerre|optical mode|computer.generated|optics|optical|photonic', re.I)
DROP = re.compile(r'acoustic|conducting oxide|stochastic resetting|optogenetic|display week|inverse design|6g', re.I)

kept, dropped, bib = {}, [], []
for topic, recs in catalog.items():
    k = []
    for r in recs:
        if KEEP.search(r['title']) and not DROP.search(r['title']):
            k.append(r)
        else:
            dropped.append((topic, r))
            if r.get('file'):
                f = LIB / r['file']
                if f.exists():
                    f.unlink()
    kept[topic] = k

def key(r):
    first = re.sub(r'[^A-Za-z]', '', (r['authors'][0].split()[-1] if r['authors'] else 'Anon')) or 'Anon'
    word = re.sub(r'[^A-Za-z]', '', r['title'].split()[0]) if r['title'] else ''
    return f"{first}{r['year']}{word}"

seen = set()
lines = ['---', 'title: Literatura 2016–2026 (OpenAlex)', 'author: Dudbil Olvasada Pabón Riaño', 'date: 2026-10-29',
         'tags:', '  - bibliografia', '  - estado-del-arte', '  - openalex', '---', '',
         '# Literatura 2016–2026 verificada en OpenAlex', '',
         'Búsqueda reproducible (`litsearch.py`): frases obligatorias en título o resumen, publicaciones desde 2016, '
         'ordenadas por citas; en cada tema se incluyen además los trabajos más citados desde 2022. '
         'Todos los DOI y conteos de citas provienen de OpenAlex (29/10/2026). Los PDF son copias de acceso abierto; '
         'los que no se pudieron descargar se consiguen con la biblioteca UNAB, Unpaywall o el autor.', '',
         f'PDF locales: `Holografia/Articulos 2016-2026/`. BibTeX: `Holografia/Articulos 2016-2026/biblioteca_2016_2026.bib`.', '']
total = pdfs = 0
for topic, recs in kept.items():
    lines += [f'## {topic.replace("_", " ")}', '', '| Año | Trabajo | Revista | Citas | DOI | PDF |', '|---|---|---|---|---|---|']
    for r in recs:
        total += 1
        a = r['authors'][0] + (' et al.' if len(r['authors']) > 1 else '') if r['authors'] else '—'
        pdf = ''
        if r.get('file'):
            pdfs += 1
            pdf = f"[PDF](<../../Holografia/Articulos 2016-2026/{r['file'].replace(chr(92), '/')}>)"
        title = r['title'].replace('|', '/')
        lines.append(f"| {r['year']} | **{title}** — {a} | {r['venue'].replace('|', '/')} | {r['cited_by']} | [{r['doi']}](https://doi.org/{r['doi']}) | {pdf} |")
        if r['doi'].lower() not in seen:
            seen.add(r['doi'].lower())
            k = key(r)
            auth = ' and '.join(r['authors'][:12]) + (' and others' if len(r['authors']) > 12 else '')
            bib.append(f"@article{{{k},\n  title = {{{r['title']}}},\n  author = {{{auth}}},\n  journal = {{{r['venue']}}},\n"
                       f"  year = {{{r['year']}}},\n  doi = {{{r['doi']}}},\n  note = {{OpenAlex {r['openalex'].split('/')[-1]}; tema {topic}}}\n}}\n")
    lines.append('')
lines += ['## Excluidos por relevancia', '', 'Aparecieron en la búsqueda pero no tratan de holografía u óptica aplicable al proyecto:', '']
lines += [f"- {r['year']} {r['title']} ({r['venue']})" for _, r in dropped]
lines += ['', 'Relacionado: [[00_Indice_Maestro_Biblioteca]], [[02_Plan_Maestro_v2_HoloLab_y_Curso]]', '']
NOTE.write_text('\n'.join(lines), encoding='utf-8')
(LIB / 'biblioteca_2016_2026.bib').write_text('\n'.join(bib), encoding='utf-8')
print(f'trabajos: {total}, PDF: {pdfs}, excluidos: {len(dropped)}, BibTeX: {len(bib)}')
