"""
Builds the frontend data bundle and a single-file copy of the app.

  frontend/data.js       weights + source files, loaded by index.html
  dist/micro-opus.html   everything inlined, works offline from a double-click
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FE = ROOT / "frontend"

models = {n: json.loads((ROOT / "weights" / f"{n}.json").read_text()) for n in ("words", "names")}
sources = {
    "model.py": (ROOT / "model.py").read_text(),
    "train.py": (ROOT / "train.py").read_text(),
    "server.py": (ROOT / "server.py").read_text(),
    "engine.js": (FE / "engine.js").read_text(),
}
data = (
    "window.MODELS = " + json.dumps(models, separators=(",", ":")) + ";\n"
    "window.SOURCES = " + json.dumps(sources) + ";\n"
)
(FE / "data.js").write_text(data)

version = (ROOT / "VERSION").read_text().strip()
index = (FE / "index.html").read_text()
index = re.sub(r'(src|href)="(style\.css|data\.js|engine\.js|app\.js)(\?v=[^"]*)?"', rf'\1="\2?v={version}"', index)
(FE / "index.html").write_text(index)

html = index
css = (FE / "style.css").read_text()
html = re.sub(r'<link rel="stylesheet" href="style\.css[^"]*">', lambda m: "<style>\n" + css + "\n</style>", html)


def inline(match):
    body = (FE / match.group(1)).read_text().replace("</script", "<\\/script")
    return "<script>\n" + body + "\n</script>"


html = re.sub(r'<script src="([^"?]+)(?:\?[^"]*)?"></script>', inline, html)
(ROOT / "dist").mkdir(exist_ok=True)
(ROOT / "dist" / "micro-opus.html").write_text(html)
print(f"data.js {len(data) // 1024} KB, dist/micro-opus.html {len(html) // 1024} KB")
