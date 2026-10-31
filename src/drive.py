"""Let the trained model drive the Udacity simulator.

    1. python -m src.drive --model models/pilotnet.pt
    2. Launch the simulator -> Autonomous Mode.

The sim connects to http://localhost:4567 and streams telemetry; we reply with
a steering angle in [-1, 1] and a throttle.
"""
import argparse
import base64
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
import torch
from flask import Flask
from PIL import Image

from .config import CFG
from .model import PilotNet
from .preprocess import preprocess, to_tensor_array

sio = socketio.Server()
app = Flask(__name__)
STATE = {"model": None, "device": "cpu"}


def throttle_for(speed: float, steering: float) -> float:
    """Slow down in sharp turns, speed up on straights (simple P-style rule)."""
    target = CFG.max_speed - (CFG.max_speed - CFG.min_speed) * min(abs(steering) * 2.0, 1.0)
    return float(np.clip(1.0 - speed / target, -0.5, 1.0))


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, skip_sid=True)
        return
    speed = float(data["speed"])
    img = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB"))
    x = torch.from_numpy(to_tensor_array(preprocess(img))).unsqueeze(0).to(STATE["device"])
    with torch.no_grad():
        steering = float(STATE["model"](x).item())
    steering = float(np.clip(steering, -1.0, 1.0))
    throttle = throttle_for(speed, steering)
    print(f"steer={steering:+.3f} ({steering * CFG.sim_max_angle_deg:+.1f} deg)  "
          f"throttle={throttle:+.2f}  speed={speed:.1f}")
    send(steering, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("simulator connected:", sid)
    send(0.0, 0.0)


def send(steering, throttle):
    sio.emit("steer", data={"steering_angle": str(steering), "throttle": str(throttle)},
             skip_sid=True)


def load_model(path, device):
    m = PilotNet().to(device)
    ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pilotnet.pt")
    ap.add_argument("--port", type=int, default=CFG.port)
    args = ap.parse_args()
    STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["model"] = load_model(args.model, STATE["device"])
    print(f"model loaded ({args.model}); waiting for simulator on port {args.port} ...")
    wsgi_app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", args.port)), wsgi_app)


if __name__ == "__main__":
    main()
