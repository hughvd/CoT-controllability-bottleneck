"""Bundle the existing research gallery into one offline, shareable HTML file."""
import base64
import html
import io
from pathlib import Path
import re
import zipfile


def make_shareable(path: Path):
    root = path.parent
    text = path.read_text()
    alias = root / "opening_tokens_share.html"
    if 'data-shareable="true"' in text:
        alias.write_text(text)
        return alias
    assets = set(re.findall(r'(?:src|href)="([^"#]+)"', text))
    local = {name for name in assets if not re.match(r'\w+:', name)}
    assert all((root / name).is_file() for name in local)
    # Preserve every linked data table/PDF in one compressed download, rather
    # than repeating large CSVs as uncompressed data URLs throughout the HTML.
    downloads = sorted(name for name in local if Path(name).suffix in {".csv", ".pdf"})
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in downloads + ["README.md"]:
            z.writestr(name, (root / name).read_bytes())
    bundle = base64.b64encode(archive.getvalue()).decode()
    def image_url(name):
        return "data:image/png;base64," + base64.b64encode((root / name).read_bytes()).decode()
    text = re.sub(r'<img src="([^"]+\.png)">',
                  lambda m: '<img loading="lazy" alt="' + html.escape(Path(m[1]).stem.replace('_', ' ')) + '" src="' + image_url(m[1]) + '">', text)
    extra = []
    for name in sorted(n for n in local if n.endswith(".png")):
        if f'href="{name}"' not in text:
            continue
        ident = "figure-" + Path(name).stem
        text = text.replace(f'href="{name}"', f'href="#{ident}"')
        title = Path(name).stem.replace('_', ' ')
        extra.append(f'<details id="{ident}"><summary>{html.escape(title)}</summary>'
                     f'<img loading="lazy" alt="{html.escape(title)}" src="{image_url(name)}"></details>')
    text = text.replace('href="README.md"', 'href="#methods"')
    for name in downloads:
        text = text.replace(f'href="{name}"', 'href="#downloads" title="Included in the downloadable ZIP"')
    text = text.replace('<html>', '<!doctype html><html lang="en" data-shareable="true">')
    text = text.replace('<head>', '<head><meta name="viewport" content="width=device-width, initial-scale=1">')
    text = text.replace('</style>', 'details{margin:20px 0;padding:14px;border:1px solid #ddd;border-radius:8px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui}.share-note{padding:14px;background:#edf5fc;border-radius:8px}.download{display:inline-block;padding:12px 18px;background:#1265aa;color:white;border-radius:6px;text-decoration:none}</style>')
    text = text.replace('<body>', '<body><p class="share-note">This is a self-contained report. All figures and methods are embedded; it works offline. Send this HTML file to share it. Linked tables and PDFs are included in the <a href="#downloads">download bundle</a>.</p>')
    methods = html.escape((root / "README.md").read_text())
    appendix = '<section><h2>Control-specific ranked-token figures</h2>' + '\n'.join(extra) + '</section>'
    appendix += f'<details id="methods"><summary>Methods, datasets, limitations, and validation</summary><pre>{methods}</pre></details>'
    appendix += '<section id="downloads"><h2>Tables and PDFs</h2><p>The original CSV tables, linked vector PDFs, and methods notes are embedded in this file as a ZIP archive.</p>'
    appendix += f'<a class="download" download="opening_token_tables_and_pdfs.zip" href="data:application/zip;base64,{bundle}">Download tables and PDFs (ZIP)</a>'
    appendix += '<ul>' + ''.join(f'<li>{html.escape(n)}</li>' for n in downloads) + '</ul></section>'
    appendix += '''<script>
function revealFragment() {
  const target = document.getElementById(location.hash.slice(1));
  if (target && target.tagName === 'DETAILS') { target.open = true; target.scrollIntoView(); }
}
window.addEventListener('hashchange', revealFragment);
window.addEventListener('DOMContentLoaded', revealFragment);
</script>'''
    text = text.replace('</body>', appendix + '</body>')
    path.write_text(text)
    alias.write_text(text)
    print(f"Self-contained gallery: {alias} ({alias.stat().st_size / 1024**2:.1f} MiB; {len(local)} embedded source files)")
    return alias


if __name__ == '__main__':
    make_shareable(Path(__file__).resolve().parent / "opening_tokens/index.html")
