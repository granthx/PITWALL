"""Central configuration. Every magic number lives here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Image geometry (NVIDIA PilotNet input) ---
    raw_h: int = 160  # Udacity simulator frame height
    raw_w: int = 320  # Udacity simulator frame width
    crop_top: int = 60  # remove sky / trees
    crop_bottom: int = 25  # remove car hood
    img_h: int = 66  # network input height
    img_w: int = 200  # network input width
    img_c: int = 3
    frame_stack: int = 1  # 1 = single-frame (default); >=2 = temporal context stacking

    # --- Data ---
    steering_correction: float = 0.2  # offset applied to left/right camera images
    zero_keep_prob: float = 0.15  # keep only this share of ~0-steer samples
    zero_threshold: float = 0.02  # |angle| below this counts as "straight"
    n_hist_bins: int = 25  # steering histogram flattening
    max_per_bin: int = 400
    val_split: float = 0.2
    seed: int = 42

    # --- Training ---
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 0.0
    dropout: float = 0.3
    num_workers: int = 2
    patience: int = 4  # early stopping

    # --- Driving ---
    port: int = 4567
    max_speed: float = 20.0  # sim units (~mph)
    min_speed: float = 8.0
    sim_max_angle_deg: float = 25.0  # sim steering [-1, 1] == [-25, +25] degrees


CFG = Config()
