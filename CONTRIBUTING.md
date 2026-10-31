# Contributing

Short version: small commits, honest results, tests stay green.

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
```

## Before you push

```bash
ruff check .        # lint
black .             # format
pytest -q           # all tests, CPU torch is enough
```

## Ground rules

- **Do not touch the pipeline contract casually**: `src/preprocess.py` is
  used by both training and driving. Any change applies to both — if the two
  drift apart the car trains well and drives badly.
- **Never fabricate results.** Only report numbers you measured. If the
  simulator wasn't available, say so (`PROJECT_GUIDE.md §11` keeps an honest
  status list).
- Keep the pinned socket.io stack (`python-socketio==4.6.1` +
  `python-engineio==3.13.2`) — the simulator speaks the legacy protocol.
- The web console (`src/webapp.py` + `web/`) is plain Flask + vanilla JS on
  purpose: no build step, no node_modules.

## Commit style

One phase/concern per commit, imperative mood:
`Phase 2: add --resume/--seed and run-dir artifacts to train.py`.

## Good first issues

- More augmentations in `src/augment.py` (motion blur, lens dirt).
- VisualBackProp as an alternative saliency in `src/visualize.py`.
- A ResNet-34 entry in `src/model.py` ARCHS + a comparison in the README.
