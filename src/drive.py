"""Let the trained model drive the Udacity simulator.

    1. python -m src.drive --model models/pilotnet.pt
    2. Launch the simulator -> Autonomous Mode.

The sim connects to http://localhost:4567 and streams telemetry; we reply with
a steering angle in [-1, 1] and a throttle.

Flags:
    --smoothing W   low-pass the steering: s = W*s_prev + (1-W)*s_new
                    (0 disables, 0.5-0.8 tames zig-zag; default 0 = off)
    --record        save every incoming frame + telemetry to runs/record_<ts>/
                    for debugging failed runs.

NOTE: eventlet must be monkey-patched before anything else imports socket/ssl.
"""

import eventlet

eventlet.monkey_patch()

import argparse
import base64
import collections
import json
import os
import time
from io import BytesIO

import cv2
import numpy as np
import socketio
import torch
from flask import Flask
from PIL import Image

from .config import CFG
from .control import smooth_steering, throttle_for
from .model import build_model
from .preprocess import preprocess, to_tensor_array

sio = socketio.Server()
app = Flask(__name__)
STATE = {
    "model": None,
    "device": "cpu",
    "smooth_prev": 0.0,  # low-pass state
    "smoothing": 0.0,
    "record_dir": None,
    "frame_i": 0,
    "frame_buffer": collections.deque(maxlen=1),
    "in_channels": 3,
}


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, skip_sid=True)
        return
    speed = float(data["speed"])
    img = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB"))
    arr = to_tensor_array(preprocess(img))  # (3, 66, 200)

    num_frames_needed = max(1, STATE["in_channels"] // 3)
    if num_frames_needed > 1:
        if len(STATE["frame_buffer"]) == 0:
            for _ in range(num_frames_needed):
                STATE["frame_buffer"].append(arr)
        else:
            STATE["frame_buffer"].append(arr)
        stacked = np.concatenate(list(STATE["frame_buffer"]), axis=0)
        x = torch.from_numpy(stacked).unsqueeze(0).to(STATE["device"])
    else:
        x = torch.from_numpy(arr).unsqueeze(0).to(STATE["device"])
    t0 = time.perf_counter()
    with torch.no_grad():
        raw = float(STATE["model"](x).item())
    steering = smooth_steering(
        float(np.clip(raw, -1.0, 1.0)), STATE["smooth_prev"], STATE["smoothing"]
    )
    STATE["smooth_prev"] = steering
    throttle = throttle_for(speed, steering)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"steer={steering:+.3f} ({steering * CFG.sim_max_angle_deg:+.1f} deg)  "
        f"throttle={throttle:+.2f}  speed={speed:.1f}  infer={dt_ms:.1f} ms"
    )

    if STATE["record_dir"] is not None:
        rec = STATE["record_dir"]
        name = f"{STATE['frame_i']:06d}.jpg"
        cv2.imwrite(os.path.join(rec, "frames", name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(os.path.join(rec, "telemetry.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    {
                        "frame": name,
                        "speed": speed,
                        "raw_steer": raw,
                        "steer": steering,
                        "throttle": throttle,
                        "infer_ms": dt_ms,
                    }
                )
                + "\n"
            )
        STATE["frame_i"] += 1

    send(steering, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("simulator connected:", sid)
    send(0.0, 0.0)


def send(steering, throttle):
    sio.emit(
        "steer", data={"steering_angle": str(steering), "throttle": str(throttle)}, skip_sid=True
    )


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "pilotnet")
    in_channels = ckpt.get("frame_stack", 1) * 3
    if "stem.0.weight" in ckpt["state_dict"]:
        in_channels = ckpt["state_dict"]["stem.0.weight"].shape[1]
    elif "features.0.weight" in ckpt["state_dict"]:
        in_channels = ckpt["state_dict"]["features.0.weight"].shape[1]
    m = build_model(arch, in_channels=in_channels).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pilotnet.pt")
    ap.add_argument("--port", type=int, default=CFG.port)
    ap.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="low-pass weight on previous steering, 0..1 (0 = off). "
        "Try 0.5-0.8 if the car zig-zags.",
    )
    ap.add_argument(
        "--record", action="store_true", help="save frames + telemetry to runs/record_<timestamp>/"
    )
    args = ap.parse_args()

    STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["model"] = load_model(args.model, STATE["device"])
    STATE["smoothing"] = float(np.clip(args.smoothing, 0.0, 0.95))
    if args.record:
        STATE["record_dir"] = time.strftime("runs/record_%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(STATE["record_dir"], "frames"), exist_ok=True)
        print(f"recording to {STATE['record_dir']}")

    print(
        f"model loaded ({args.model}); waiting for simulator on port {args.port} ... "
        f"(open the sim -> Autonomous Mode)"
    )
    wsgi_app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", args.port)), wsgi_app)


if __name__ == "__main__":
    main()
