# Instructions for Antigravity (agent) — Turn this repo into a finished project

You are taking over a working **skeleton** of a behavioral-cloning self-driving-car project (PyTorch + Udacity simulator). Read `PROJECT_GUIDE.md` first for the concept, architecture and file map. Your job: **verify it, harden it, and finish it into a polished, reproducible, portfolio-ready project.**

Work through the phases in order. After each phase: run the tests, commit with a clear message, and tick the checklist at the bottom. If something can't be done in your environment (e.g. the simulator needs a GUI), say so and write the best possible instructions/code instead of faking results.

---

## Ground rules

- **Do not change the pipeline's core contract**: `src/preprocess.py` is used by BOTH training and driving. Any change there applies to both.
- Keep Python ≥3.9 compatible. Keep dependencies minimal; the socket.io versions in `requirements.txt` are intentionally pinned (the Udacity sim uses the legacy protocol).
- Never fabricate results, metrics, screenshots or GIFs. Only report numbers you actually measured.
- Prefer small, reviewable commits. Keep `README.md` accurate as you go.

## Phase 0 — Get green

1. Create a venv and install dependencies. **Install PyTorch yourself first** (the author could not install it in their sandbox, so the model, training loop and `drive.py` have never been executed):
   - NVIDIA GPU: pick the matching command from https://pytorch.org/get-started/locally/ (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`)
   - CPU / Apple Silicon: `pip install torch`
   - Then `pip install -r requirements.txt`. Fix any version conflicts (document them).
   - Confirm with `python -c "import torch; print(torch.__version__)"`.
   - Treat `src/model.py`, `src/train.py` and `src/drive.py` as **unverified code**: run them, and fix any bugs you find before moving on.
2. `pytest -q` — **all 5 tests must pass** (two of them need torch; they were never executed by the author). Fix whatever fails.
3. Smoke-train on fake data:
   ```bash
   python scripts/make_synthetic_data.py --out data/synthetic --n 300
   python -m src.train --data data/synthetic --epochs 3 --workers 0
   ```
   Confirm `models/pilotnet.pt` is written and val MSE decreases. Verify the model flatten size is 1152 (a mismatch means the conv stack/input size is wrong).
4. Verify `python -m src.drive --help` imports cleanly (eventlet/socketio). If `eventlet` fails on your Python version, pin a working combination and document it in `PROJECT_GUIDE.md §10`.

## Phase 1 — Real data

1. Download the Udacity self-driving-car simulator (the Unity-based "term 1" build for your OS) from the official Udacity GitHub releases. If you cannot download or run it, **skip to Phase 1b**.
2. Record data in *Training Mode* following `PROJECT_GUIDE.md §9` (≈3–5 centre laps, 1–2 reversed, recovery clips). Save to `data/recording/`.
3. `python -m src.visualize hist --data data/recording` and inspect the balancing. Tune `zero_keep_prob`, `max_per_bin` in `src/config.py` if one steering value still dominates.

**Phase 1b (no simulator available):** add `scripts/download_sample_data.md` explaining where to obtain a public Udacity sample dataset, and make sure `src/data.py` works with it (it uses the standard `driving_log.csv` + `IMG/` layout, header row tolerated).

## Phase 2 — Train & evaluate properly

1. Train: `python -m src.train --data data/recording --epochs 15`.
2. Improve `src/train.py`:
   - log to `runs/<timestamp>/` (config snapshot, history JSON, loss-curve PNG);
   - add `--resume` and `--seed` flags;
   - save the best AND last checkpoint;
   - keep early stopping.
3. Add `src/evaluate.py` (or extend `src/visualize.py eval`) reporting MAE in normalised units AND degrees (`× sim_max_angle_deg`), plus the predicted-vs-human scatter plot.
4. Reproduce the video's "guess → error → adjust" story: write `scripts/training_story.py` that, for ONE fixed image, prints the model's prediction before training (random init) and after each epoch next to the human angle and the error. Save as `runs/training_story.png`.

## Phase 3 — Drive the simulator

1. `python -m src.drive --model models/pilotnet.pt`, then start *Autonomous Mode*.
2. Success criterion: **one full lap without leaving the road**. If it fails, debug in this order:
   1. Preprocessing mismatch between training and inference (colour order RGB vs BGR, YUV, crop).
   2. Data quality/coverage at the failure point → record more there.
   3. Speed: lower `max_speed` in `config.py` (try 12–15).
   4. Augmentation strength / steering correction (0.15–0.25).
3. Make `throttle_for` in `src/drive.py` smoother if the car oscillates (e.g. low-pass filter the steering: `s = 0.7*s_prev + 0.3*s_new`, behind a flag).
4. Add a `--record` option to `drive.py` that saves frames of autonomous runs to `runs/` for debugging.

## Phase 4 — Explainability visuals (this is what makes the project shine)

Match the video's story "lines → shapes → one number":
1. Finish `python -m src.visualize maps` so it outputs clear per-layer feature-map grids for a given image (early layers = edges/lines, later = shapes).
2. Add a **saliency / "what the car looks at"** overlay (e.g. gradient-based saliency, or the VisualBackProp method from NVIDIA's follow-up paper) → `src/visualize.py saliency`. Verify the highlights land on lane lines/road edges; if not, say so honestly.
3. Export 1 short GIF/MP4 of the car driving with the saliency overlay if the simulator is available.

## Phase 5 — Engineering polish

- `README.md`: short, with the project goal, a diagram of the pipeline, quick start, results table (only measured numbers), and a link to `PROJECT_GUIDE.md`.
- Add type hints + docstrings where missing; run `ruff`/`black` and fix issues.
- Add a `Makefile` (or `tasks.py`) with: `setup`, `test`, `smoke`, `train`, `drive`, `viz`.
- Add GitHub Actions CI that runs `pytest -q` on CPU (install CPU torch).
- Add a `Dockerfile` for training on CPU/GPU (the simulator itself stays on the host).
- Add a `LICENSE` (MIT) and a short `CONTRIBUTING.md`.

## Phase 6 — Stretch goals (pick any, in this order)

1. **Model comparison:** add a small ResNet-18-based regressor in `src/model.py` behind `--arch`, compare with PilotNet on MAE and lap success.
2. **Temporal context:** stack 2–3 consecutive frames or add a small GRU; measure whether it reduces oscillation.
3. **Export:** ONNX export script (`scripts/export_onnx.py`) and verify parity with PyTorch on a few images.
4. **Second track:** record on the harder track; report generalisation honestly.
5. **Web demo:** a simple Streamlit/Gradio page: upload a frame → predicted steering angle drawn on a wheel + saliency overlay.

## Definition of done (checklist)

- [ ] `pytest -q` passes (all 5 tests, with torch)
- [ ] Synthetic smoke-train works end-to-end
- [ ] Real data recorded (or documented alternative) and histogram checked
- [ ] Model trained; MAE reported in degrees
- [ ] Car completes ≥1 lap autonomously (or clearly documented why not testable)
- [ ] Feature-map + saliency visuals produced
- [ ] `training_story` figure reproduces the video's guess→error→adjust idea
- [ ] README with only real measured results; CI + Makefile + Dockerfile present
- [ ] `PROJECT_GUIDE.md` §10–11 updated with anything you learned or fixed

## Report back

When finished, produce a short `CHANGELOG.md` listing what you changed, what you verified, what you could not verify, and the final measured metrics.
