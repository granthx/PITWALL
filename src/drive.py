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
from .control import lane_centering_assist, smooth_steering, throttle_for
from .model import build_model
from .preprocess import preprocess, to_tensor_array

sio = socketio.Server()
app = Flask(__name__)
STATE = {
    "model": None,
    "device": "cpu",
    "smooth_prev": 0.0,  # low-pass state
    "smoothing": 0.25,
    "steer_gain": 1.15,
    "lka_weight": 1.0,
    "max_speed": CFG.max_speed,
    "telemetry_log": [],
    "start_time": None,
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
        # Symmetric Test-Time Flip De-Biasing: eliminates any left/right drift bias
        xf = torch.flip(x, dims=[3])
        out = STATE["model"](torch.cat([x, xf], dim=0))
        raw = 0.5 * float(out[0].item() - out[1].item())
    # Vision-based Lane-Keeping Assist (curb repulsion & road-symmetry guard)
    lka_nudge = lane_centering_assist(img) * STATE["lka_weight"]
    scaled = float(np.clip(raw * STATE["steer_gain"] + lka_nudge, -1.0, 1.0))
    steering = smooth_steering(scaled, STATE["smooth_prev"], STATE["smoothing"])
    STATE["smooth_prev"] = steering
    throttle = throttle_for(speed, steering, max_speed=STATE["max_speed"])
    dt_ms = (time.perf_counter() - t0) * 1000.0

    steer_deg = steering * CFG.sim_max_angle_deg
    STATE["telemetry_log"].append((time.perf_counter() - STATE["start_time"], speed, steer_deg, throttle))

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
    STATE["frame_buffer"].clear()
    STATE["smooth_prev"] = 0.0
    STATE["telemetry_log"] = []
    STATE["start_time"] = None
    send(0.0, 0.4)


def send(steering, throttle):
    sio.emit(
        "steer", data={"steering_angle": str(steering), "throttle": str(throttle)}, skip_sid=True
    )


@sio.on("disconnect")
def disconnect(sid):
    print("\nsimulator disconnected:", sid)
    if STATE["telemetry_log"]:
        logs = STATE["telemetry_log"]
        times, speeds, steers, throttles = zip(*logs)
        dur = times[-1] - times[0] if len(times) > 1 else 0
        mean_spd = sum(speeds) / len(speeds)
        top_spd = max(speeds)
        jitter = float(np.mean(np.abs(np.diff(steers)))) if len(steers) > 1 else 0.0
        print("=" * 55)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            run_dir = STATE["record_dir"] or time.strftime("runs/drive_%Y%m%d-%H%M%S")
            os.makedirs(run_dir, exist_ok=True)
            plot_path = os.path.join(run_dir, "closed_loop_telemetry.png")
            fig, ax1 = plt.subplots(figsize=(10, 4.5))
            ax2 = ax1.twinx()
            ax1.plot(times, speeds, color="#2ecc71", label="Speed (mph)", lw=1.5)
            ax2.plot(times, steers, color="#3498db", label="Steering (°)", lw=1.2, alpha=0.85)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Speed (mph)", color="#2ecc71")
            ax2.set_ylabel("Steering Angle (°)", color="#3498db")
            ax1.set_title(f"Autonomous Closed-Loop Run (Avg Speed: {mean_spd:.1f} mph, Jitter: {jitter:.2f}°)")
            ax1.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=130)
            plt.close(fig)
            print(f"Saved run telemetry chart -> {plot_path}\n")
        except Exception:
            pass

        print("         AUTONOMOUS RUN BENCHMARK SUMMARY        ")
        print("=" * 55)
        print(f"Total Time Active : {dur:.1f} s ({len(logs)} frames)")
        print(f"Average Speed     : {mean_spd:.1f} mph | Top Speed: {top_spd:.1f} mph")
        print(f"Steering Jitter   : {jitter:.2f}° per step")
        print("=" * 55)


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
        default=0.25,
    )
    ap.add_argument(
        "--steer-gain",
        type=float,
        default=1.15,
    )
    ap.add_argument(
        "--max-speed",
        type=float,
        default=CFG.max_speed,
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
    STATE["steer_gain"] = float(args.steer_gain)
    STATE["max_speed"] = float(args.max_speed)
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
