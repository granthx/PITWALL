# PIT WALL — End-to-End Self-Driving Car (Behavioral Cloning)

An end-to-end autonomous driving system inspired by NVIDIA's **PilotNet** and modern **Spatial-Temporal ResNet-18** architectures. The car learns to steer solely from raw camera pixels mapped to human steering angles using deep behavioral cloning.

No road lane was ever labeled: supervision comes entirely from recorded steering angles. The project includes an autonomous simulator driver bridge, real-time lane-centering assist, and an interactive **PIT WALL** diagnostic web console with saliency attention maps.

```
Simulator (Training Mode)                         Deep Neural "Brain"
 └─ IMG/ + driving_log.csv                     ┌───────────────────────────────────┐
      └─ src/data.py       balance & 3-camera │  Crop & Resize → 66×200 YUV       │
      └─ src/augment.py    flip/shadow/shift  │  Conv stem & ResNet residual blocks│
      └─ src/preprocess.py RGB → YUV tensor   │  Dense layers → Steering Angle ŷ  │
           └─ src/model.py  SpatialResNet18    └───────────────────────────────────┘
                └─ src/train.py  MSE/Huber                       │  Steering ∈ [−1, 1]
                                                                 │  (mapped to ±25°)
Simulator (Autonomous Mode) ⇄ socket.io :4567 ⇄ src/drive.py ───┘
                                   and ⇄ src/webapp.py → PIT WALL Web Console (:5000)
```

---

## Performance & Benchmark Results

Evaluated on 1,608 held-out validation frames:

| Metric | PilotNet (Baseline) | Standard ResNet-18 | **SpatialResNet18 Pro (Current)** |
| :--- | :--- | :--- | :--- |
| **Validation Loss (MSE)** | `0.0294` | `0.0188` | **`0.0091`** |
| **Mean Steering Error (MAE)** | `3.11°` | `2.46°` | **`0.57°`** *(81% reduction)* |
| **RMSE** | `4.03°` | `3.08°` | **`1.28°`** |
| **$R^2$ Score (Variance Explained)** | `-0.058` | `+0.120` | **`+0.832`** |
| **Accuracy within $\pm 1.0^\circ$** | — | — | **`82.5%`** |
| **Accuracy within $\pm 1.5^\circ$** | $41.1\%$ | $41.1\%$ | **`88.9%`** |
| **Accuracy within $\pm 2.0^\circ$** | $49.1\%$ | $51.5\%$ | **`92.4%`** |
| **Accuracy within $\pm 3.0^\circ$** | $67.1\%$ | $67.1\%$ | **`96.6%`** |
| **Accuracy within $\pm 5.0^\circ$** | $89.1\%$ | $88.2\%$ | **`98.6%`** |

---

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (POSIX: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m pytest -q
```

---

## Running the System

### Option A: Autonomous Driving in the Simulator (1-Click)
Double-click **`run_autonomous.bat`** in the project root, or run via terminal:
```bash
# 1. Start autonomous driver bridge
python -m src.drive --model models/resnet18_pro.pt --smoothing 0.20 --steer-gain 1.25 --max-speed 20.0

# 2. Launch simulator
simulator\"Default Windows desktop 64-bit.exe"
```
*In the simulator, select **Lake Track** or **Mountain Track** and click **AUTONOMOUS MODE**.*

#### Autonomous Driving Features
- **Symmetric Test-Time Flip De-Biasing**: Feeds mirror flips simultaneously to cancel out left/right drift bias.
- **Vision Lane-Centering Assist**: Real-time road boundary and curb detection to actively repel the vehicle from track edges.
- **Corner-Entry Braking**: Proactively slows down in sharp bends to prevent corner departure at high speeds.

---

### Option B: The PIT WALL Web Console
Launch the interactive diagnostic dashboard:
```bash
python -m src.webapp --model models/resnet18_pro.pt --data data/recording
```
Open **`http://127.0.0.1:5000`** in your browser to:
- **Interactive Steering Wheel**: Test individual frames or upload custom road photos and watch the wheel rotate in real time.
- **Saliency Attention Maps**: View gradient heatmaps showing exactly what features (curbs, lane edges) the model focuses on.
- **Feature Map Strips**: Inspect 5 convolutional stages visualising lines, shapes, and steering geometry.
- **Dataset Filmstrip**: Browse recorded human frames with their ground-truth steering angles.

---

## Training & Evaluation

### Training from Scratch
```bash
# Train PilotNet baseline
python -m src.train --data data/recording --epochs 15 --out models/pilotnet.pt

# Train ResNet-18
python -m src.train --data data/recording --arch resnet18 --epochs 20 --out models/resnet18.pt
```

### Evaluating Any Model Checkpoint
```bash
python -m src.evaluate --model models/resnet18_pro.pt --data data/recording --split val
```

### Export to ONNX
```bash
python scripts/export_onnx.py --model models/resnet18_pro.pt --out models/resnet18_pro.onnx
```

---

## Repository Structure

```
self-driving-car/
├── data/                       # Training data (driving_log.csv + IMG/)
├── models/                     # Trained checkpoints (.pt) and evaluation reports
│   ├── pilotnet.pt             # Baseline PilotNet model
│   ├── resnet18.pt             # Standard ResNet-18
│   ├── resnet18_pro.pt         # SpatialResNet18 Pro (0.57° MAE, 98.6% accuracy)
│   ├── resnet18_pro_eval.json  # Full benchmark JSON
│   └── resnet18_pro_eval.png   # Steering scatter plot
├── runs/                       # Training logs, benchmark plots, and story graphs
├── scripts/                    # Utilities
│   ├── download_dataset.py     # Download official sample dataset
│   ├── download_simulator.py   # Download official Udacity simulator
│   ├── export_onnx.py          # Export checkpoints to ONNX runtime
│   ├── make_synthetic_data.py  # Synthetic test data generator
│   └── training_story.py       # Visual progression generator
├── simulator/                  # Udacity 3D simulation executable
├── src/                        # Core codebase
│   ├── augment.py              # Data augmentations (flip, shadow, brightness, translate)
│   ├── config.py               # Central configuration and hyperparameters
│   ├── control.py              # Throttle curve, steering smoothing, lane-centering assist
│   ├── data.py                 # CSV parsing, balancing, 3-camera expansion
│   ├── drive.py                # Real-time Socket.IO bridge to simulator
│   ├── evaluate.py             # Evaluation metrics and scatter plots
│   ├── figstyle.py             # Navy visual styling for charts
│   ├── model.py                # PilotNet and SpatialResNet18 model definitions
│   ├── preprocess.py           # Crop, resize (66x200), and RGB->YUV conversion
│   ├── train.py                # Training loop with validation and checkpointing
│   ├── visualize.py            # Feature activations, saliency heatmaps, histograms
│   └── webapp.py               # Flask PIT WALL web dashboard
├── tests/                      # Automated pytest test suite
├── web/                        # PIT WALL frontend (HTML, CSS, JS)
├── pyproject.toml              # Project metadata & tool config
├── README.md                   # Single project manual & documentation
├── requirements.txt            # Python dependencies
└── run_autonomous.bat          # 1-click launcher for simulator + AI driver
```

---

## License

MIT License.
