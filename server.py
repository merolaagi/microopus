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

model = MicroOpus()
app = FastAPI(title="Micro Opus")


class TraceRequest(BaseModel):
    ids: list[int]


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
    params = sum(int(t.size) for t in model.W.values())
    return {"engine": "model.py", "version": VERSION, "params": params, "vocab": len(model.vocab)}


@app.post("/api/trace")
def trace(req: TraceRequest):
    ids = req.ids
    if not 1 <= len(ids) <= model.cfg["T"]:
        raise HTTPException(400, f"Send between 1 and {model.cfg['T']} token ids.")
    if any(not 0 <= i < len(model.vocab) for i in ids):
        raise HTTPException(400, "Token id outside the 30-word vocabulary.")
    tr = {"ids": ids}
    model.forward(ids, tr)
    return to_json(tr)


@app.get("/api/source/{name:path}", response_class=PlainTextResponse)
def source(name: str):
    if name not in SOURCES:
        raise HTTPException(404, "Unknown source file.")
    return (ROOT / name).read_text()


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
