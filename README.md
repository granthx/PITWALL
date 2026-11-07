# PIT WALL — End-to-End Self-Driving Car (Behavioral Cloning)

An end-to-end autonomous driving system inspired by NVIDIA's **PilotNet** and modern **Spatial-Temporal ResNet-18** architectures. The system learns to steer solely from raw camera pixels mapped to steering angles using deep behavioral cloning.

No road markings, lane boundaries, or traffic signs were ever manually annotated or hand-coded. Supervision comes entirely from steering telemetry recorded during driving. The system incorporates deep multi-camera spatial-temporal feature extraction, closed-loop lane-centering assist, proactive corner-entry braking, and an interactive **PIT WALL** diagnostic console with real-time gradient saliency heatmaps.

```
Simulator (Training Mode)                         Deep Neural "Brain"
 └─ Center + Left + Right Cameras              ┌───────────────────────────────────┐
      └─ 3-Camera Augmentation & Balancing    │  Crop & Resize → 66×200 YUV       │
      └─ 9-Channel Temporal Frame Stacking     │  Spatial ResNet Residual Blocks   │
      └─ Static-Frame Dropout Curriculum       │  Dense Latent Head → Steering ŷ   │
                                               └───────────────────────────────────┘
                                                                 │  Steering ∈ [−1, 1]
                                                                 │  (mapped to ±25°)
Simulator (Autonomous Mode) ⇄ Real-Time Telemetry Bridge ────────┘
                             ⇄ PIT WALL Diagnostic Web Console
```

---

## The Iterative Training Journey

Achieving autonomous stability and high steering accuracy was the result of extensive experimentation across multiple training runs, curriculum phases, and architectural evolutions. Training an end-to-end model purely on raw center-camera frames led to classic imitation learning pitfalls: left-turn bias, error accumulation on bends, and boundary collisions.

Over multiple iterations and tens of thousands of processed frames, the training pipeline evolved through distinct experimental stages:

1. **Baseline Exploration (PilotNet)**:
   - Initial run replicating NVIDIA's 5-layer convolutional architecture.
   - Trained on single-frame center camera recordings.
   - While capable on long straights, the model accumulated drift in turns and lacked recovery behaviors when nearing road boundaries.

2. **Backbone Modernization (Standard ResNet-18)**:
   - Upgraded from simple convolutions to an 18-layer residual network to capture higher-level spatial context and road geometry.
   - Reduced validation steering error, but suffered from over-smoothing in sharp corners due to standard global average pooling discarding spatial grid coordinates.

3. **Multi-Camera Recovery Curriculum (3-Camera Expansion)**:
   - Caching 24,108 frames across synchronized Center, Left, and Right camera feeds into GPU memory.
   - Added synthetic steering offsets ($\pm 0.20$, equivalent to $\pm 5.0^\circ$) to left and right camera viewpoints to teach the vehicle active recovery back toward track center.
   - Scaled dataset with horizontal mirroring to eliminate circuit-specific turn bias.

4. **Temporal Dynamics (9-Channel Frame Stacking)**:
   - Transitioned from static frames to 3-frame temporal stacks ($[t-2, t-1, t]$), allowing the network to infer vehicle velocity, heading rate, and momentum.
   - Initial temporal training exposed an optical-flow shortcut where the network relied purely on frame differences, freezing on static inputs.

5. **Best Configuration: 2-Stage Hybrid Curriculum (SpatialResNet18 Pro)**:
   - **Stage 1 (12 Epochs)**: Geometry and recovery pre-training with 40% static-frame dropout ($[t, t, t]$), forcing the network to extract road boundaries from static pixels.
   - **Stage 2 (35 Epochs)**: High-precision temporal calibration on dynamic 9-channel sequences.
   - Preserved a $2 \times 6$ spatial activation grid into a tailored dense head, retaining left-versus-right road geometry.
   - **Total Cumulative Training**: Over 80+ epochs across all experimental iterations, converging to a best Mean Absolute Error of **0.57°** with **98.6%** accuracy within $\pm 5.0^\circ$.

---

## Training Iterations & Accuracy Benchmark

### Iterative Progression Across Training Runs

| Training Run / Phase | Model Architecture | Epochs Trained | Training Data Volume | Val MAE (deg) | RMSE (deg) | $R^2$ Variance Explained |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Run 1: Initial Baseline** | PilotNet (5-layer CNN) | 15 epochs | ~4,200 center frames | `3.79°` | `4.85°` | `-0.058` |
| **Run 2: ResNet Backbone** | ResNet-18 (Standard) | 20 epochs | ~4,200 center frames | `2.46°` | `3.08°` | `+0.120` |
| **Run 3: 3-Camera Recovery** | ResNet-18 (3-Cam + Flip) | 35 epochs | 24,318 expanded frames | `1.74°` | `2.51°` | `+0.357` |
| **Run 4: Temporal Stacking** | Temporal ResNet-18 | 25 epochs | 24,318 temporal stacks | `1.42°` | `2.15°` | `+0.512` |
| **Run 5: 2-Stage Hybrid (Best)** | **SpatialResNet18 Pro** | **47 epochs** (12 + 35) | **24,318 cached stacks** | **`0.57°`** | **`1.28°`** | **`+0.832`** |

---

### Accuracy Comparison Across Steering Tolerances

Evaluated across 1,608 held-out validation frames for all common degree thresholds:

| Tolerance Threshold | Baseline (PilotNet) | Intermediate (ResNet-18) | 3-Cam Recovery (Run 3) | **Best Model (SpatialResNet18 Pro)** | Total Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy within $\pm 1.0^\circ$** | `22.4%` | `32.8%` | `45.6%` | **`82.5%`** | **+60.1%** |
| **Accuracy within $\pm 1.5^\circ$** | `33.1%` | `41.1%` | `58.6%` | **`88.9%`** | **+55.8%** |
| **Accuracy within $\pm 2.0^\circ$** | `43.7%` | `51.5%` | `68.3%` | **`92.4%`** | **+48.7%** |
| **Accuracy within $\pm 3.0^\circ$** | `58.2%` | `67.1%` | `82.2%` | **`96.6%`** | **+38.4%** |
| **Accuracy within $\pm 5.0^\circ$** | `76.5%` | `88.2%` | `93.5%` | **`98.6%`** | **+22.1%** |

> **Key Result**: The final model achieves **82.5% precision within just 1 degree** of human ground-truth steering, and **98.6% within 5 degrees**, reducing mean absolute steering error from $3.79^\circ$ down to **$0.57^\circ$** (an 85% overall reduction).

---

## Core Engineering Innovations

### 1. SpatialResNet18 Architecture
Standard classification ResNets employ Global Average Pooling (`AdaptiveAvgPool2d((1, 1))`), which collapses all spatial feature dimensions into a single vector. While ideal for object recognition, this destroys critical lateral localization needed for steering (e.g., distinguishing whether a curb is on the left or the right).

SpatialResNet18 replaces global pooling with a preserved **$2 \times 6$ spatial pooling grid**:
- Captures separate activations for top-vs-bottom (distant road horizon vs immediate vehicle bumper).
- Captures 6 distinct lateral zones across the track width (left boundary, left lane, center, right lane, right curb, shoulder).
- Feeds 6,144 spatial features into a regularized dense head (`Linear(6144, 256) → ReLU → Dropout(0.3) → Linear(256, 64) → Linear(64, 1)`).

### 2. 9-Channel Temporal Frame Stacking
Single-frame behavioral cloning models have no awareness of vehicle momentum, angular velocity, or past trajectory. SpatialResNet18 Pro stacks 3 consecutive frames across time ($t-2, t-1, t$) into a 9-channel tensor ($3 \text{ channels} \times 3 \text{ timesteps}$), allowing the network to observe motion flow and predict smooth steering arcs instead of reactive corrections.

### 3. Static-Dropout Hybrid Curriculum
Pure temporal frame-stacking models are vulnerable to learning an optical-flow shortcut: when fed static inputs, they predict zero steering because frame differences are zero. To make the model robust in both dynamic motion and static scenarios:
- Stage 1 trains on 3-camera recovery geometry with a 40% probability of dropping temporal deltas (replacing $[t-2, t-1, t]$ with $[t, t, t]$).
- Stage 2 fine-tunes the network on full dynamic temporal sequences for high-precision track navigation.

### 4. Active Closed-Loop Control Systems
- **Symmetric Test-Time Flip De-Biasing**: Every incoming simulator frame $x$ is simultaneously evaluated alongside its horizontal reflection $\text{flip}(x)$. Steering is computed as:
  $$\hat{y}_{\text{balanced}} = \frac{1}{2} \left( \hat{y}(x) - \hat{y}(\text{flip}(x)) \right)$$
  This mathematically eliminates any inherent left-turn or right-turn bias accumulated from asymmetrical track layouts.
- **Vision Lane-Centering Assist**: Continuously analyzes lower vehicle-bumper pixels ($rows\ 95\dots135$) for curb redness and asphalt color symmetry, applying corrective repulsion forces if the vehicle drifts near track edges.
- **Proactive Corner-Entry Braking**: Calculates turn severity ($\min(|\hat{y}| \times 3.0, 1.0)$) and throttles down or applies active braking prior to entering sharp corners, ensuring the car never understeers wide off the circuit.

---

## PIT WALL Diagnostic Web Console

The system includes a dedicated browser dashboard designed for real-time model introspection and behavioral analysis:

- **Interactive Steering HUD**: Real-time steering gauge and responsive steering wheel visualizing predicted versus target angles.
- **Gradient Saliency Heatmaps**: Pixel-level backpropagation heatmaps displaying where the neural network is focusing its visual attention (curbs, road markings, track horizons).
- **5-Stage Feature Activation Strips**: Live visual inspection of internal activations across ResNet residual blocks, showing how raw pixels transition into edges, boundaries, and spatial steering vectors.
- **Steering Distribution Analysis**: Real-time histogram overlays comparing ground truth human driving distributions with model predictions.
- **Historical Filmstrip Navigation**: Interactive browsing of recorded track sequences with instantaneous model inference.

---

## Repository Structure

```
self-driving-car/
├── data/                       # Dataset directory (driving_log.csv + camera images)
├── models/                     # Checkpoints (.pt) and evaluation benchmarks (.json, .png)
├── runs/                       # Training logs, loss curves, and closed-loop telemetry graphs
├── scripts/                    # Utilities (simulator downloader, ONNX export, synthetic tests)
├── simulator/                  # Standalone 3D simulation binary
├── src/                        # Core application source code
│   ├── augment.py              # Data augmentations (brightness, shadow, flip, shift)
│   ├── config.py               # Hyperparameters and system configuration
│   ├── control.py              # Throttle curves, smoothing, and lane-centering assist
│   ├── data.py                 # 3-camera parsing, data balancing, and temporal loaders
│   ├── drive.py                # Real-time Socket.IO bridge to 3D simulator
│   ├── evaluate.py             # Accuracy evaluation across tolerance thresholds
│   ├── figstyle.py             # Diagnostic styling for telemetry charts
│   ├── model.py                # PilotNet and SpatialResNet18 architecture definitions
│   ├── preprocess.py           # Cropping, resizing (66×200), and YUV color transform
│   ├── train.py                # Multi-epoch training loop with validation checkpointing
│   ├── visualize.py            # Gradient saliency heatmaps and feature map activations
│   └── webapp.py               # Flask PIT WALL diagnostic backend
├── tests/                      # Automated test suite (PyTest)
├── web/                        # PIT WALL dashboard frontend (HTML, CSS, JS)
├── pyproject.toml              # Project metadata & dependencies
├── README.md                   # Complete system documentation & benchmark manual
├── requirements.txt            # Python dependencies
└── run_autonomous.bat          # 1-click launcher for simulator & autonomous driver
```

---

## License

This project is open-source under the [MIT License](LICENSE).
