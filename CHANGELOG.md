# Changelog

All numbers below were **measured on this machine** — Python 3.13.12, PyTorch
2.15.0+cpu, Windows 11, no GPU. Nothing is estimated or copied from the
original video. The one thing this environment could not do is run the Unity
simulator (no display / no sim build), so the autonomous-lap criterion is
documented rather than claimed — see "Not verified".

## 2026-10-31 — agent build-out (Phases 0–6)

### Phase 0 — get green
- Created venv; installed CPU torch. Verified the whole skeleton executes.
- `pytest -q`: **5 passed** (was never run by the original author).
- Smoke-train on 300 synthetic frames: `models/pilotnet.pt` written, val MSE
  decreased 0.2517 → 0.2313 over 3 epochs. Flatten size 1152 confirmed by the
  forward-pass test.
- **Found & fixed:** `eventlet==0.33.3` cannot import on Python ≥ 3.12
  (`ssl.wrap_socket` removed). Re-pinned to `eventlet>=0.37`; verified
  `eventlet 0.41.3` + `python-socketio==4.6.1` + `python-engineio==3.13.2`:
  `python -m src.drive --help` imports cleanly.
- **Found & fixed:** `src/drive.py` never called `eventlet.monkey_patch()`;
  added it before all other imports (standard Udacity bridge practice).

### Phase 1b — real data without the simulator
- `scripts/download_sample_data.md`: Udacity sample dataset (318 MB) and
  simulator release links **verified live (HTTP 200)** on 2026-10-31.
- `src/data.py` confirmed to tolerate a CSV header row and absolute/
  foreign paths (test covers the standard layout).

### Phase 2 — train & evaluate properly
- `src/train.py`: every run now writes `runs/<timestamp>/` with a config
  snapshot, `history.json` (incl. val MAE in degrees + lr), `loss_curve.png`,
  per-epoch checkpoints (for the story figure), best + last checkpoints;
  added `--resume`, `--seed`, `--arch`, `--no-epoch-ckpts`, `--run-dir`.
  Early stopping kept (patience 4).
- `src/evaluate.py`: MAE / RMSE / max-err / bias in normalised units **and**
  degrees, predicted-vs-human scatter, metrics JSON. `--split val` replicates
  train.py's exact split so the headline metric is truly held out.
- `scripts/training_story.py`: the video's "guess → error → adjust" for one
  fixed image → `runs/training_story.png` (+ console table).

### Phase 3 — drive the simulator (code-ready, not sim-tested)
- `--smoothing W` low-pass on steering (`s = W·s_prev + (1−W)·s_new`, 0 = off)
  for zig-zag taming; `--record` dumps frames + telemetry JSONL to
  `runs/record_<ts>/`; inference time printed per frame.
- Control math shared with the web console via `src/control.py`.

### Phase 4 — explainability
- `src/visualize.py maps`: per-layer grid — input, mean map, top-K channels
  for all five conv layers (lines → shapes → angle).
- `src/visualize saliency`: gradient saliency with a navy→teal→amber LUT
  overlay. On the synthetic turn frame the hot mass sits on the road/lane
  region (high-saliency mass 0.74) — the model looks at the road, not the sky.
- Navy figure style shared by all figures (`src/figstyle.py`).

### The trained model (synthetic track, 2000 frames, ×3 cameras)
| Metric | Value |
|---|---|
| Run | `runs/20261031-131832`, PilotNet, 25-epoch budget, early-stopped at 24 |
| Best epoch | 20 — val MSE **0.01784**, val MAE **2.91°** (held-out 20 %) |
| Held-out eval (`--split val`, centre cams, n=393) | MAE **2.90°** (0.116 norm), RMSE 3.34°, max err 7.01°, bias −0.10° |
| Full-dataset centre-cam eval (partly in-sample, n=2000) | MAE 0.96°, RMSE 1.23° |
| Training story (fixed turning frame) | random init **+7.8°** vs human **−18.1°** (err 26.0°) → epoch 1 −12.5° (5.6°) → converges ≈ 1°, last epoch 0.5° |
| Inference | ~9 ms/frame on CPU |
| ONNX export | `models/pilotnet_pilotnet.onnx`, parity max\|torch−onnx\| = **1.2e-07** over 8 frames |

Honesty note: these prove the *pipeline* learns to steer; the synthetic track
(polygon road, curvature-derived labels) says little about real track-1
performance. Record real laps → `src.evaluate --split val`.

### The PIT WALL console (the UI)
- `src/webapp.py` + hand-written `web/` (Flask + vanilla JS/CSS/SVG, **no
  build step, no new dependencies**): dark-navy pit-wall theme, steering
  wheel + tape that turn with the prediction, saliency toggle, live layer
  maps, Δ-vs-human verdict, dataset histograms/filmstrip, per-run loss
  curves + eval metrics + training story.
- Verified in-browser end-to-end against the trained checkpoint.

### Phase 5 — engineering polish
- README rewritten (pipeline diagram, quick start, console guide);
  PROJECT_GUIDE §10–11 updated with everything learned; MIT LICENSE;
  CONTRIBUTING.md; Makefile (`setup test smoke train evaluate drive viz ui`);
  GitHub Actions CI (ruff + pytest + smoke-train on CPU); Dockerfile
  (CPU, opencv libs); `ruff check` + `black` clean; `pyproject.toml` configs.

### Phase 6 — stretch
- ResNet-18 regressor from scratch (no torchvision) behind `--arch`, forward
  test added for both archs.
- **Found & fixed:** ResNet head fed the pooled 4-D tensor to `nn.Linear`
  (last-dim semantics) → crash; flatten before the head. Test added so it
  cannot regress.
- **Model comparison** (same data, same 20 % split, CPU):

| Model | Params | Budget | Best epoch | Val MSE | Held-out MAE (`--split val`, n=393) |
|---|---|---|---|---|---|
| PilotNet | 252,219 | 25-epoch budget | 20 | **0.01784** | **2.90°** |
| ResNet-18 | 11.2 M | 6-epoch budget (early-stopped at 5) | 1 | 0.02718 | 3.24° |

  Honest reading: at a 4× smaller epoch budget PilotNet also trains ~4× faster
  per epoch on CPU and wins on MAE — the NVIDIA architecture earns its keep on
  this problem. The ResNet number is a budget-limited data point, not a
  verdict on the architecture; give it equal epochs on real data before
  judging. Lap-success comparison remains impossible here (no simulator).

### Not verified (and why)
- **Autonomous lap in the simulator** — the sim needs a display and a desktop
  build this environment doesn't have. The bridge runs the verified legacy
  socket.io stack; first thing to try on a desktop:
  `python -m src.drive --model models/pilotnet.pt --smoothing 0.6`.
- Model comparison on **lap success** — same reason; MAE comparison only.
- Saliency on real track imagery — needs real recorded data (doc provided).
