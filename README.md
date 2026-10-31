# Self-Driving Car — Behavioral Cloning (PyTorch + Udacity Simulator)

Teach a virtual car to steer from camera images using NVIDIA's end-to-end PilotNet.

- 📘 **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** — the concept, the video explained, architecture, usage.
- 🤖 **[ANTIGRAVITY_INSTRUCTIONS.md](ANTIGRAVITY_INSTRUCTIONS.md)** — step-by-step brief for an AI agent to finish and polish the project.

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py --out data/synthetic --n 300
python -m src.train --data data/synthetic --epochs 3 --workers 0
pytest -q
```
