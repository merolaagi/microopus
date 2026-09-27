"""
Micro Opus web server.

Serves the frontend and runs model.py for every forward pass the workbench shows,
so the numbers on screen come from the same Python file shown in the code window.
"""
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import MicroOpus

ROOT = Path(__file__).parent
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "dev"
SOURCES = {"model.py", "train.py", "server.py", "frontend/engine.js"}

MODELS = {name: MicroOpus(name) for name in ("words", "names")}
app = FastAPI(title="Micro Opus")


@app.middleware("http")
async def no_stale_pages(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache"
    elif request.url.query.startswith("v="):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


class TraceRequest(BaseModel):
    ids: list[int]
    model: str = "words"


def to_json(v):
    if isinstance(v, np.ndarray):
        a = np.round(v, 6).astype(object)
        a[~np.isfinite(v)] = None
        return a.tolist()
    if isinstance(v, dict):
        return {k: to_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [to_json(x) for x in v]
    return v


@app.get("/api/health")
def health():
    models = {n: sum(int(t.size) for t in m.W.values()) for n, m in MODELS.items()}
    return {"engine": "model.py", "version": VERSION, "models": models}


@app.post("/api/trace")
def trace(req: TraceRequest):
    ids, model = req.ids, MODELS.get(req.model)
    if model is None:
        raise HTTPException(400, f"Unknown model. Choose one of: {', '.join(MODELS)}.")
    if not 1 <= len(ids) <= model.cfg["T"]:
        raise HTTPException(400, f"Send between 1 and {model.cfg['T']} token ids.")
    if any(not 0 <= i < len(model.vocab) for i in ids):
        raise HTTPException(400, f"Token id outside the {len(model.vocab)}-token vocabulary.")
    tr = {"ids": ids}
    model.forward(ids, tr)
    tr["ids"] = ids
    return to_json(tr)


@app.get("/api/source/{name:path}", response_class=PlainTextResponse)
def source(name: str):
    if name not in SOURCES:
        raise HTTPException(404, "Unknown source file.")
    return (ROOT / name).read_text()


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
