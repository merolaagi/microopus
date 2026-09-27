# Micro Opus

A 7,440-weight decoder-only transformer, trained from scratch, with a web app that animates
exactly where each weight matrix enters the forward pass, next to the Python code that runs it.

## Install, run, publish

Each iteration arrives as a uniquely named archive (`microopus-<version>-<date>-<time>.zip`).
The archive name and version live in `ARCHIVE` and `VERSION`; the port is in `PORT` (47911).

| Script | What it does |
| --- | --- |
| `tools/install.sh` | Creates `.venv`, installs requirements, builds the frontend, runs the server as a launchd agent on port 47911, checks `/api/health`. |
| `tools/tunnel.sh` | One time: adds `microopus.fueldeskpro.com -> localhost:47911` to `/etc/cloudflared/config.yml` (with a backup), routes DNS through the `fueldeskpro-mac` tunnel, restarts cloudflared. |
| `tools/publish.sh` | Commits, tags `VERSION`, pushes to `github.com/merolaagi/microopus`, and attaches the archive from `~/Downloads` to a GitHub release. |

Re-extracting a new archive over the folder keeps `.venv`, `.git` and `logs`.

## Files

| Path | What it is |
| --- | --- |
| `model.py` | The model: named weight matrices plus `forward()`. This is the file in the code window. |
| `train.py` | Trains from random noise and writes `weights/model.json`. The only code that changes weights. |
| `server.py` | FastAPI backend. Serves the frontend, `/api/trace`, `/api/health`, `/api/source/{file}`. |
| `weights/model.json` | The trained weights: 20 named tensors, 7,440 numbers. |
| `frontend/` | The web app: `index.html`, `style.css`, `app.js`, `engine.js`, and generated `data.js`. |
| `frontend/engine.js` | Line-for-line JavaScript port of `model.py`, used when no server is running. |
| `tools/install.sh`, `tunnel.sh`, `publish.sh` | Deployment scripts described above. |
| `tools/build.py` | Regenerates `frontend/data.js` and the single-file `dist/micro-opus.html`. |

## Retrain

```
python train.py
python tools/build.py
```

Training takes about a minute on CPU. Edit the sentence templates at the top of `train.py` to
teach it a different toy world, then rebuild.

## Check that Python and the browser agree

```
python model.py
```

It prints the top three next words for "the bird flew over the". The browser engine matches
`model.py` to within 1e-6 on every activation.
