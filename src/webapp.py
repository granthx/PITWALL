"""PIT WALL — local web console for the behavioral-cloning project.

    python -m src.webapp --model models/pilotnet.pt --data data/synthetic

Serves the hand-built UI in ``web/`` and a small JSON API:

    GET  /api/status                 model + dataset + eval summary
    GET  /api/samples?k=14           evenly spaced frames from the dataset
    GET  /api/frame?path=IMG/…       serves a frame (path-jail: inside --data)
    GET  /api/histogram?data=…       raw vs balanced steering histograms
    GET  /api/runs                   training runs under runs/
    GET  /api/runs/<id>              config + history (+ eval.json) of one run
    POST /api/predict                {path} or multipart file ->
                                     steering, saliency overlay, layer maps

Everything stays on your machine; nothing leaves localhost.
"""

import argparse
import base64
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from .config import CFG
from .control import throttle_for
from .data import balance, load_log
from .figstyle import saliency_lut
from .model import build_model
from .visualize import activations, gradient_saliency, overlay_saliency, prep_image

app = Flask(__name__, static_folder=None)
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
LUT = saliency_lut()

STATE = {
    "model": None,
    "device": "cpu",
    "model_path": None,
    "data_dir": None,
    "labels": {},  # image basename -> human steering (any camera, corrected)
}


# ------------------------------------------------------------- small utils --


def png_dataurl(img: np.ndarray, quality: int = 0) -> str:
    if quality:
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
    else:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _norm(arr: np.ndarray, pct: float = 99.5) -> np.ndarray:
    hi = np.percentile(arr, pct)
    return np.clip(arr / (hi + 1e-9), 0, 1)


def _thumb(map2d: np.ndarray, scale: int = 4) -> str:
    img = (LUT[(np.clip(map2d, 0, 1) * 255).astype(np.uint8)]).astype(np.uint8)
    img = cv2.resize(
        img, (map2d.shape[1] * scale, map2d.shape[0] * scale), interpolation=cv2.INTER_NEAREST
    )
    return png_dataurl(img)


def safe_path(rel: str) -> str:
    """Resolve rel inside the dataset dir; refuse anything that escapes it."""
    root = os.path.realpath(STATE["data_dir"])
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("path escapes the dataset directory")
    return full


def read_frame_rgb(src) -> np.ndarray:
    """RGB uint8 160x320 frame from a path or raw upload bytes."""
    if isinstance(src, str):
        bgr = cv2.imread(src)
        if bgr is None:
            raise FileNotFoundError(src)
    else:
        arr = np.frombuffer(src, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("not a decodable image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (CFG.raw_h, CFG.raw_w):  # tolerate other sizes
        rgb = cv2.resize(rgb, (CFG.raw_w, CFG.raw_h), interpolation=cv2.INTER_AREA)
    return rgb


# ------------------------------------------------------------------- pages --


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(WEB_DIR, filename)


# --------------------------------------------------------------------- api --


@app.route("/api/status")
def api_status():
    ckpt = torch.load(STATE["model_path"], map_location="cpu")
    model = STATE["model"]
    eval_j = None
    eval_base = os.path.splitext(STATE["model_path"])[0] + "_eval.json"
    if os.path.exists(eval_base):
        with open(eval_base) as f:
            eval_j = json.load(f)
    data_dir = STATE["data_dir"]
    rows = (
        len(load_log(data_dir)) if os.path.exists(os.path.join(data_dir, "driving_log.csv")) else 0
    )
    return jsonify(
        {
            "model": {
                "path": STATE["model_path"].replace("\\", "/"),
                "arch": ckpt.get("arch", "pilotnet"),
                "params": sum(p.numel() for p in model.parameters()),
                "val_mse": ckpt.get("val_mse"),
                "epoch": ckpt.get("epoch"),
                "input": f"{CFG.img_c}x{CFG.img_h}x{CFG.img_w}",
            },
            "device": STATE["device"],
            "torch": torch.__version__,
            "data": {
                "dir": data_dir.replace("\\", "/"),
                "rows": rows,
                "correction": CFG.steering_correction,
                "cameras": 3,
            },
            "eval": eval_j,
            "bridge": {"port": CFG.port, "running": False},
        }
    )


@app.route("/api/samples")
def api_samples():
    df = load_log(STATE["data_dir"])
    k = min(int(request.args.get("k", 12)), 60)
    idx = np.linspace(0, len(df) - 1, k).astype(int)
    out = []
    for i in idx:
        name = os.path.basename(str(df.iloc[i]["center"]))
        out.append({"i": int(i), "path": f"IMG/{name}", "angle": float(df.iloc[i]["steering"])})
    return jsonify(out)


@app.route("/api/frame")
def api_frame():
    rel = request.args.get("path", "")
    full = safe_path(rel)
    with open(full, "rb") as f:
        b = f.read()
    resp = app.response_class(b, mimetype="image/jpeg")
    return resp


@app.route("/api/histogram")
def api_histogram():
    df = load_log(STATE["data_dir"])
    bal = balance(df)
    bins = np.linspace(-1, 1, CFG.n_hist_bins + 1)
    raw, _ = np.histogram(df["steering"], bins=bins)
    blc, _ = np.histogram(bal["steering"], bins=bins)
    return jsonify(
        {
            "edges": bins.tolist(),
            "raw": raw.tolist(),
            "balanced": blc.tolist(),
            "n_raw": len(df),
            "n_balanced": len(bal),
            "zero_share": float((df["steering"].abs() < CFG.zero_threshold).mean()),
        }
    )


@app.route("/api/runs")
def api_runs():
    runs = []
    if os.path.isdir("runs"):
        for d in sorted(os.listdir("runs")):
            rd = os.path.join("runs", d)
            hp = os.path.join(rd, "history.json")
            if not os.path.isdir(rd) or not os.path.exists(hp):
                continue
            try:
                with open(hp) as f:
                    hist = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            best = min(h.get("val_mse", 9e9) for h in hist)
            runs.append(
                {
                    "id": d,
                    "epochs": len(hist),
                    "best_val_mse": best,
                    "best_val_mae_deg": min(h.get("val_mae_deg", 9e9) for h in hist),
                    "has_loss": os.path.exists(os.path.join(rd, "loss_curve.png")),
                    "has_story": os.path.exists(os.path.join(rd, "story.png")),
                    "mtime": os.path.getmtime(hp),
                }
            )
    runs.sort(key=lambda r: -r["mtime"])
    return jsonify(runs)


@app.route("/api/runs/<rid>")
def api_run(rid):
    rd = os.path.join("runs", rid)
    out = {}
    for name in ("config.json", "history.json"):
        p = os.path.join(rd, name)
        if os.path.exists(p):
            with open(p) as f:
                out[name.split(".")[0]] = json.load(f)
    evals = sorted(glob.glob(os.path.join(rd, "*eval.json")))
    if evals:
        with open(evals[-1]) as f:
            out["eval"] = json.load(f)
    if not out:
        return jsonify({"error": "run not found"}), 404
    return jsonify(out)


@app.route("/api/story")
def api_story():
    # newest run that has a story.png wins; fall back to the brief's default path
    cands = glob_story_candidates()
    if cands:
        newest = max(cands, key=os.path.getmtime)
        with open(newest, "rb") as f:
            return app.response_class(f.read(), mimetype="image/png")
    return jsonify({"error": "no training story yet — run scripts/training_story.py"}), 404


def glob_story_candidates():
    import glob

    cands = glob.glob(os.path.join("runs", "*", "story.png"))
    if os.path.exists("runs/training_story.png"):
        cands.append("runs/training_story.png")
    return cands


@app.route("/api/predict", methods=["POST"])
def api_predict():
    t0 = time.perf_counter()
    path = request.form.get("path") if request.form else None
    if not path and request.is_json:
        path = (request.get_json(silent=True) or {}).get("path")
    if request.files.get("file"):
        rgb = read_frame_rgb(request.files["file"].read())
        human = None
    elif path:
        full = safe_path(path)
        rgb = read_frame_rgb(full)
        human = STATE["labels"].get(os.path.basename(full))
    else:
        return jsonify({"error": "send multipart 'file' or json/form {path}"}), 400

    model, device = STATE["model"], STATE["device"]
    x, small = prep_image(rgb)
    in_channels = getattr(model, "in_channels", 3)
    if in_channels > 3 and x.shape[0] == 3:
        n_stack = in_channels // 3
        seq = STATE.get("seq_paths", {}).get(os.path.basename(full), []) if path else []
        if len(seq) == n_stack and all(os.path.exists(p) for p in seq):
            tensors = [prep_image(read_frame_rgb(p))[0] for p in seq]
            x_in = torch.cat(tensors, dim=0)
        else:
            x_in = x.repeat(n_stack, 1, 1)
    else:
        x_in = x

    t1 = time.perf_counter()
    with torch.no_grad():
        steer = float(model(x_in.unsqueeze(0).to(device)).item())
    infer_ms = (time.perf_counter() - t1) * 1000.0
    steer = float(np.clip(steer, -1, 1))

    sal = gradient_saliency(model, x_in)
    over = overlay_saliency(small, sal)
    sal3 = cv2.resize(
        over, (small.shape[1] * 3, small.shape[0] * 3), interpolation=cv2.INTER_NEAREST
    )
    inp3 = cv2.resize(
        small, (small.shape[1] * 3, small.shape[0] * 3), interpolation=cv2.INTER_NEAREST
    )

    acts = activations(model, x)
    maps = []
    roles = ["LINES", "LINES", "SHAPES", "SHAPES", "ANGLE"]
    for i, act in enumerate(acts[:5]):
        if act.ndim != 3:
            continue
        role = roles[i] if i < len(roles) else f"STAGE {i+1}"
        norm = _norm(act)
        top_chans = [int(c) for c in np.argsort(-act.mean(axis=(1, 2)))[:3]]
        maps.append(
            {
                "layer": i + 1,
                "role": role,
                "mean": _thumb(norm.mean(0), scale=6),
                "chans": [_thumb(norm[c], scale=6) for c in top_chans],
            }
        )

    speed_demo = 15.0
    return jsonify(
        {
            "steer_norm": steer,
            "deg": steer * CFG.sim_max_angle_deg,
            "throttle": throttle_for(speed_demo, steer),
            "speed": speed_demo,
            "human_deg": None if human is None else human * CFG.sim_max_angle_deg,
            "infer_ms": infer_ms,
            "total_ms": (time.perf_counter() - t0) * 1000.0,
            "input": png_dataurl(inp3),
            "saliency": png_dataurl(sal3),
            "sal_heat": _thumb(sal, scale=3),
            "maps": maps,
        }
    )


# -------------------------------------------------------------------- main --


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pilotnet.pt")
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()

    STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["model_path"] = a.model
    ckpt = torch.load(a.model, map_location=STATE["device"], weights_only=False)
    arch = ckpt.get("arch", "pilotnet")
    in_channels = ckpt.get("frame_stack", 1) * 3
    if "stem.0.weight" in ckpt.get("state_dict", {}):
        in_channels = ckpt["state_dict"]["stem.0.weight"].shape[1]
    elif "features.0.weight" in ckpt.get("state_dict", {}):
        in_channels = ckpt["state_dict"]["features.0.weight"].shape[1]
    model = build_model(arch, in_channels=in_channels).to(STATE["device"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    STATE["model"] = model
    STATE["data_dir"] = a.data

    df = load_log(a.data)
    labels = {}
    seq_paths = {}
    n_stack = max(1, in_channels // 3)
    for col, off in (

        ("center", 0.0),
        ("left", CFG.steering_correction),
        ("right", -CFG.steering_correction),
    ):
        col_list = [os.path.basename(str(p).strip().replace("\\", "/")) for p in df[col]]
        for idx_i, (bname, s) in enumerate(zip(col_list, df["steering"])):
            labels[bname] = float(s) + off
            hist = [os.path.join(a.data, "IMG", col_list[max(0, idx_i - k)]) for k in range(n_stack - 1, -1, -1)]
            seq_paths[bname] = hist
    STATE["labels"] = labels
    STATE["seq_paths"] = seq_paths

    print(
        f"PIT WALL console  ·  model={a.model} ({ckpt.get('arch', 'pilotnet')})  "
        f"device={STATE['device']}"
    )
    print(f"dataset={a.data} ({len(df)} rows)  ->  http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
