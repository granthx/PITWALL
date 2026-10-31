# Self-Driving Car (Behavioral Cloning) — Project Guide

*Everything you need to know about this project: what the video said, why it works, and how the code maps to it.*

---

## 1. What the video is about

A short "Resume Project" explainer: **build a virtual self-driving car using the same approach NVIDIA used for its real one.** It references two things:

| Piece | What it is |
|---|---|
| **Udacity Self-Driving Car Simulator** | Free driving simulator (Unity). You drive laps by hand; it records camera images + your steering angle. Also has an "Autonomous Mode" that lets your code drive. |
| **NVIDIA end-to-end model ("PilotNet")** | A small convolutional neural network that maps *a camera picture → one steering angle*. Paper: *End to End Learning for Self-Driving Cars* (Bojarski et al., 2016). |

The technique is called **behavioral cloning**: the network copies what a human driver did.

## 2. The video, beat by beat

1. **"A road isn't a road. It's just millions of numbers."** — The car's camera view is a grid of pixel values. The network never "sees" a road, only numbers.
2. **Pairs of (picture, steering angle).** — Each frame is paired with the angle the human used (e.g. −15°, +15°). This is the *labelled dataset*.
3. **Collect data in the simulator.** — As you drive, it saves images + your steering angles into a `DATASET`.
4. **Build the neural network (NVIDIA's).** — Picture goes in, **ANGLE** comes out.
5. **What the layers learn:**
   - early layers → **edges and lines**
   - middle layers → **shapes** (e.g. the drivable road region)
   - last layers → **"how far to turn"** — collapses everything into **ONE NUMBER** (e.g. −15°)
6. **Training = guess, measure, adjust.**
   - Untrained, the guesses are basically random (e.g. model says **+34°**).
   - It compares with *yours* (−15°) → **error**, then adjusts its weights.
   - Repeat over the whole dataset, many times; the error shrinks (the video shows +20° → +5° guess).
7. **Result:** the model steers by itself — **even though nobody ever labeled a single lane.** It discovered "lane lines matter" purely from steering examples.
8. **CTA:** "Comment DRIVE for the breakdown."

## 3. Why it works (plain-English)

- **Supervised regression.** Input = image, target = steering angle, loss = mean squared error between guess and human angle.
- **Backpropagation + gradient descent** nudge millions of weights so the error falls.
- **Nobody labels lanes.** The only supervision is the steering number; the conv layers invent whatever features (lines, curbs, road shape) best predict it.
- **Weakness — distribution shift.** If the car drifts off-centre, it sees images the human never recorded. Fix: left/right camera images with an angle offset, plus "recovery" driving data, plus augmentation.

## 4. Architecture (what's in `src/model.py`)

Input `3 × 66 × 200` (YUV, normalised inside the model):

| Layer | Output shape | Role (video language) |
|---|---|---|
| Conv 5×5, stride 2, 24 | 24×31×98 | LINES |
| Conv 5×5, stride 2, 36 | 36×14×47 | LINES |
| Conv 5×5, stride 2, 48 | 48×5×22 | SHAPES |
| Conv 3×3, 64 | 64×3×20 | SHAPES |
| Conv 3×3, 64 | 64×1×18 | SHAPES |
| Flatten | 1152 | |
| Dense 100 → 50 → 10 | | decide |
| Dense 1 | **1** | **ONE NUMBER = steering angle** |

~250k parameters — tiny, trains on a laptop CPU/GPU.

## 5. Pipeline overview

```
Simulator (Training Mode)
   └─ records IMG/ (center, left, right) + driving_log.csv
        └─ src/data.py     load → balance (drop excess straight driving) → expand 3 cameras
             └─ src/augment.py  flip, shift, shadow, brightness   (training only)
                  └─ src/preprocess.py  crop → resize 200×66 → RGB→YUV
                       └─ src/model.py  PilotNet
                            └─ src/train.py  MSE + Adam, early stopping → models/pilotnet.pt

Simulator (Autonomous Mode) ⇄ socket.io :4567 ⇄ src/drive.py  (same preprocess → model → steer + throttle)
```

## 6. File map

| Path | Purpose |
|---|---|
| `src/config.py` | All hyper-parameters & constants |
| `src/preprocess.py` | Crop/resize/YUV — **shared by training and driving** |
| `src/augment.py` | Augmentations (flip negates angle; shift adjusts angle) |
| `src/data.py` | CSV loading, balancing, 3-camera expansion, PyTorch `Dataset` |
| `src/model.py` | PilotNet |
| `src/train.py` | Training loop |
| `src/drive.py` | Socket.IO server that drives the simulator |
| `src/visualize.py` | Steering histogram, feature maps ("lines → shapes"), predicted-vs-human plot |
| `scripts/make_synthetic_data.py` | Fake dataset for a no-simulator smoke test |
| `tests/test_pipeline.py` | Unit tests |

## 7. Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# smoke test (no simulator needed)
python scripts/make_synthetic_data.py --out data/synthetic --n 300
python -m src.train --data data/synthetic --epochs 3 --workers 0
pytest -q

# real run
# 1. Download Udacity simulator, open it -> Training Mode -> record 3-5 laps
#    (centre driving, 1-2 laps reverse direction, + recovery from edges). Save into data/recording/
python -m src.visualize hist --data data/recording
python -m src.train --data data/recording --epochs 15
python -m src.drive --model models/pilotnet.pt          # then Autonomous Mode in the sim
```

## 8. Steering units

Simulator steering ∈ **[−1, 1]**, mapping to **±25°**. The video's "−15°" ≈ −0.6 in the CSV. `src/config.py → sim_max_angle_deg`.

## 9. Data-collection tips (matter more than model tweaks)

- Drive **smoothly with the mouse/analog** — keyboard steering gives jerky labels.
- 3–5 laps centre-lane, **1–2 laps reversed**, plus **recovery** clips (steer from the edge back to centre; record only the recovery).
- 20–30k frames is plenty. More *varied* data > more laps of the same thing.
- If it fails at one spot, record more data at that spot.

## 10. Known pitfalls

| Symptom | Likely cause |
|---|---|
| Car drives straight into the first turn | Not enough turn data / balancing too weak |
| Zig-zags | Noisy keyboard labels; throttle too high; try lower speed, or `python -m src.drive --smoothing 0.6` |
| Sim won't connect | Wrong `python-socketio` version — must be the pinned 4.6.1 stack |
| Great val loss, bad driving | Val MSE isn't driving quality; test in the sim. Check training/inference preprocessing match |
| Path errors on training | CSV has absolute paths from another PC — handled: only the file name is used, in `<data>/IMG/` |
| `eventlet` won't import on Python ≥ 3.12 | 0.33.3 used `ssl.wrap_socket`, which was removed. Pin `eventlet>=0.37` (verified 0.41.3 on Python 3.13; prints a deprecation warning — safe to ignore). The socket.io protocol pins stay at 4.6.1/3.13.2. |
| `drive.py` serves exactly one client then stalls | The eventlet monkey-patch was missing; it now runs before every other import in `src/drive.py`. |
| Web console: unbatched tensor into PilotNet | Conv layers tolerate `(C,H,W)` but Flatten then yields the wrong shape — always pass `(N,3,66,200)`. |

## 11. Honest status of this codebase (updated after the agent build-out)

- ✅ **Executed and verified on this machine** (Python 3.13.12, torch 2.15 CPU,
  Windows): all 5 tests; end-to-end smoke-train; a 25-epoch PilotNet run on
  synthetic data with run-dir artifacts; `src.evaluate` metrics + scatter;
  `training_story.py` figure; feature maps + gradient saliency on synthetic
  frames; the PIT WALL web console end-to-end (predict, saliency, maps,
  histograms, loss curves) with ~10 ms CPU inference per frame; ONNX export
  parity (see CHANGELOG.md for the numbers).
- ✅ Verified pinned combos: `python-socketio==4.6.1` + `python-engineio==3.13.2`
  + `eventlet 0.41.3` import cleanly on Python 3.13 and `drive.py --help` works.
- ⚠️ **Not verifiable in this environment** (no display / no simulator install):
  an actual autonomous lap. The bridge code follows the standard Udacity
  protocol, uses the verified legacy socket.io stack, and now has a low-pass
  steering filter plus frame recording (`--record`) to debug real runs. The
  success criterion (`models/pilotnet.pt`, one lap, no leaving the road) is
  the first thing to check on a machine that can run the sim.
- ⚠️ All quality metrics published are on the **synthetic** track (polygon
  road, label = curvature). They prove the pipeline learns; they do not
  predict track-1 performance. Record real laps and re-run `src.evaluate`.
- ⚠️ Simulator hyper-parameters (`max_speed`, correction 0.2, smoothing 0)
  remain sensible defaults to tune.

## 12. Credits

Concept from the "Resume Project: Self-Driving Car" short video; NVIDIA's PilotNet paper; Udacity's open-source self-driving-car simulator. This project is an independent re-implementation for learning.
