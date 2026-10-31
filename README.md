# PIT WALL — a self-driving car that clones your steering

Teach a virtual car to steer from camera pixels with NVIDIA's end-to-end
**PilotNet** (behavioral cloning), then watch it think in a hand-built
**dark-navy pit-wall console** — steering wheel, saliency overlays, layer maps
and all. No lane was ever labeled: the only supervision is the steering
number a human produced.

```
Simulator (Training Mode)                 the "brain"
 └─ IMG/ + driving_log.csv                 ┌────────────────────────────┐
      └─ src/data.py      load·balance     │  crop → 66×200 YUV        │
      └─ src/augment.py   flip·shift·dark  │  conv5×5×3 → lines        │
      └─ src/preprocess.py crop·resize·YUV  │  conv3×3×2 → shapes       │
           └─ src/model.py  PilotNet         │  dense 100-50-10 → ŷ      │
                └─ src/train.py  MSE+Adam    └────────────────────────────┘
                                                  │  one number ∈ [−1, 1]
Simulator (Autonomous Mode) ⇄ socket.io :4567 ⇄ src/drive.py  ────────┘
                                        and ⇄ src/webapp.py → PIT WALL UI
```

📘 Deep dive: [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — the concept, the video
beat-by-beat, why it works, pitfalls. 🤖 [ANTIGRAVITY_INSTRUCTIONS.md](ANTIGRAVITY_INSTRUCTIONS.md)
is the build brief this repo was finished from.

## Quick start (no simulator needed)

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows; POSIX: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/make_synthetic_data.py --out data/synthetic --n 300   # fake track
python -m src.train --data data/synthetic --epochs 3 --workers 0     # smoke-train
pytest -q                                                            # 5 tests
python -m src.webapp                                                 # open the console → 127.0.0.1:5000
```

## The console (PIT WALL)

```bash
python -m src.webapp --model models/pilotnet.pt --data data/synthetic
```

A plain Flask + vanilla-JS single page (no build step, no node_modules) in
`web/`:

- **Drive** — pick/drop/upload a frame; the model answers in ~10 ms on CPU:
  a turning steering wheel, the angle in degrees, the low-passed throttle it
  would send the simulator, the Δ-vs-human verdict, gradient **saliency**
  overlay and the five conv layers' activations (lines → shapes → angle).
- **Data** — raw vs balanced steering histogram, dataset stats, a filmstrip
  where every frame carries its human angle.
- **Training** — every run under `runs/` with its loss curves, held-out
  metrics from `src/evaluate`, and the "guess → error → adjust" story figure.

## With the real simulator

1. Get it from the official releases: `github.com/udacity/self-driving-car-sim`
   (or grab Udacity's public sample dataset — see
   [scripts/download_sample_data.md](scripts/download_sample_data.md)).
2. Training Mode → record 3–5 centre laps, 1–2 reversed, recovery clips → save
   into `data/recording/`.
3. `python -m src.visualize hist --data data/recording` — check the balance.
4. `python -m src.train --data data/recording --epochs 15` →
   `python -m src.evaluate --model models/pilotnet.pt --data data/recording`.
5. `python -m src.drive --model models/pilotnet.pt [--smoothing 0.6] [--record]`,
   then start **Autonomous Mode** in the sim (it connects to `:4567`).

## Results (measured, CPU-only)

See the honest-results table in `CHANGELOG.md`. Everything reported there was
actually measured on this machine; the simulator lap test itself is documented
as not-executable-here (no GUI) rather than claimed.

## Repository map

| Path | What it is |
|---|---|
| `src/preprocess.py` | crop → resize → YUV — **shared by training AND driving** |
| `src/data.py` / `src/augment.py` | CSV loading, balancing, 3-camera expansion / augmentations |
| `src/model.py` | PilotNet (+ `--arch resnet18` comparison) |
| `src/train.py` | run-dir artifacts, `--resume`/`--seed`, best+last checkpoints |
| `src/evaluate.py` | MAE in normalised units **and degrees**, scatter plot |
| `src/drive.py` | simulator bridge (`--smoothing`, `--record`) |
| `src/visualize.py` | histograms, feature-map grids, gradient saliency |
| `src/webapp.py` + `web/` | the PIT WALL console |
| `scripts/` | synthetic data, training story, ONNX export, sample-data guide |
| `Makefile` | `setup test smoke train evaluate drive viz ui` |

## Contributing & license

MIT — see [LICENSE](LICENSE). PRs welcome; start with
[CONTRIBUTING.md](CONTRIBUTING.md).
