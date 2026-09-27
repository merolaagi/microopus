# Micro Opus

A 7,440-weight decoder-only transformer, trained from scratch, with a web app that animates
exactly where each weight matrix enters the forward pass, how training changes every weight,
and how generation reuses a key/value cache, next to the Python code that runs it.

Two trained models ship with it: `words` (a toy sentence world, one word per token) and
`names` (32,000 real first names, one letter per token).

## Install, run, publish

Each iteration arrives as a uniquely named archive (`microopus-<version>-<date>-<time>.zip`).
The archive name and version live in `ARCHIVE` and `VERSION`; the port is in `PORT` (47911).

| Script | What it does |
| --- | --- |
| `tools/install.sh` | Creates `.venv`, installs requirements, builds the frontend, runs the server as a launchd agent on port 47911, checks `/api/health`. |
| `tools/tunnel.sh` | One time: adds `microopus.fueldeskpro.com -> localhost:47911` to `/etc/cloudflared/config.yml` (with a backup), restarts cloudflared, and prints the CNAME to add in the dashboard. |
| `tools/publish.sh` | Commits, tags `VERSION`, pushes to `github.com/merolaagi/microopus`, and attaches the archive from `~/Downloads` to a GitHub release. |

Re-extracting a new archive over the folder keeps `.venv`, `.git` and `logs`.

## Files

| Path | What it is |
| --- | --- |
| `model.py` | The model: named weight matrices plus `forward()`, a hand-written `backward()`, and a KV-cache `step()`. Only numpy. |
| `train.py` | Trains from random noise with Adam and writes `weights/words.json` or `weights/names.json`. The only code that changes weights. |
| `server.py` | FastAPI backend. Serves the frontend, `/api/trace`, `/api/health`, `/api/source/{file}`. |
| `weights/*.json` | Trained weights (20 named tensors), loss curve, held-out batch, and the training text used by in-browser training. |
| `data/names.txt` | Names dataset from karpathy/makemore (MIT, license in `data/NAMES_LICENSE`). |
| `frontend/` | The web app: `index.html`, `style.css`, `app.js`, `engine.js`, and generated `data.js`. |
| `frontend/engine.js` | Line-for-line JavaScript port of `model.py` plus the Adam step, used for in-browser training and when no server is running. |
| `tools/gradcheck.py` | Checks `backward()` against finite differences. |
| `tools/check_engine.py` | Checks `engine.js` against `model.py` (needs node). |
| `tools/install.sh`, `tunnel.sh`, `publish.sh` | Deployment scripts described above. |
| `tools/build.py` | Regenerates `frontend/data.js` and the single-file `dist/micro-opus.html`. |

## Retrain

```
python train.py words
python train.py names
python tools/build.py
```

Words takes about 40 seconds on CPU, names about 90. The Training tab in the app does the same
thing live in the browser, from random noise, and can swap the result into every other tab.

## Check the math

```
python tools/gradcheck.py
python tools/check_engine.py
python model.py
```

The gradient check compares `backward()` with finite differences. The engine check runs the
browser engine through node and compares forward, backward and the cache step with `model.py`.

## Credits

The training walkthrough, KV-cache view and names model were prompted by Andrej Karpathy's
[microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). No code is copied
from it. The names dataset comes from [karpathy/makemore](https://github.com/karpathy/makemore)
under the MIT license.
