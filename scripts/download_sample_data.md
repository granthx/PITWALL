# Getting real driving data without recording it yourself

`src/data.py` expects the standard Udacity layout in the folder you pass to
`--data`:

```
<data-dir>/
├── driving_log.csv     # center,left,right,steering,throttle,brake,speed
└── IMG/                # centre/left/right camera JPGs named in the CSV
```

A header row in the CSV is tolerated (it is filtered out automatically), and
absolute paths from the recording machine are fine — only the file name is
used, resolved inside `<data-dir>/IMG/`.

## Option A — Udacity's official sample dataset (verified working)

Udacity published a hand-recorded sample dataset (track 1, ~8k frames,
~318 MB zip). As of 2026-10 this URL still serves it:

```
https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
```

```bash
mkdir -p data/recording
curl -L -o data/sample.zip "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"
python -c "import zipfile; zipfile.ZipFile('data/sample.zip').extractall('data/recording')"
rm data/sample.zip
```

The zip extracts to a `data/` folder containing `driving_log.csv` + `IMG/`;
move those two up into `data/recording/` (or just pass `data/recording/data`
to the tools — whatever folder actually contains the CSV).

Notes on this dataset:
- The CSV stores paths like `IMG/center_2016_12_01_13_30_48_287.jpg` — relative, fine.
- Steering is already in [-1, 1]; it is quite centre-biased, so the default
  balancing in `src/data.py` matters — run
  `python -m src.visualize hist --data data/recording` and look at the
  "balanced" panel before training.

## Option B — record your own

Download the Udacity self-driving-car simulator (Unity "term 1" build) from
the official releases page:

```
https://github.com/udacity/self-driving-car-sim/releases
```

Open it → **Training Mode** → drive with a gamepad/mouse (not the keyboard —
keyboard steering makes jerky labels) → it writes `driving_log.csv` + `IMG/`
into the folder you chose. See `PROJECT_GUIDE.md §9` for how many laps and
which recovery clips to record.

## Then

```bash
python -m src.visualize hist --data data/recording
python -m src.train --data data/recording --epochs 15
python -m src.evaluate --model models/pilotnet.pt --data data/recording
```
