# Pit-wall targets. (Windows: run the same commands manually, or use Git Bash
# + `make` from Chocolatey; every target is one copy-paste shell line.)
PY ?= python

.PHONY: setup test smoke train evaluate drive viz ui story lint docker-build

setup:            ## create .venv and install CPU torch + requirements
	$(PY) -m venv .venv && .venv/bin/pip install --upgrade pip \
	  && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
	  && .venv/bin/pip install -r requirements.txt

test:             ## run the test suite
	$(PY) -m pytest -q

smoke:            ## fake data + 3-epoch train, end-to-end
	$(PY) scripts/make_synthetic_data.py --out data/synthetic --n 300
	$(PY) -m src.train --data data/synthetic --epochs 3 --workers 0

train:            ## real run (edit --data/--epochs to taste)
	$(PY) -m src.train --data data/recording --epochs 15

evaluate:         ## held-out MAE + scatter, written next to the checkpoint
	$(PY) -m src.evaluate --model models/pilotnet.pt --data data/recording

story:            ## guess -> error -> adjust figure
	$(PY) scripts/training_story.py --data data/recording

viz:              ## histogram + feature maps + saliency for one frame
	$(PY) -m src.visualize hist --data data/recording
	$(PY) -m src.visualize maps --model models/pilotnet.pt \
	    --image data/recording/IMG/$$(ls data/recording/IMG | grep center | head -1)

drive:            ## simulator bridge (then start Autonomous Mode in the sim)
	$(PY) -m src.drive --model models/pilotnet.pt

ui:               ## PIT WALL web console -> http://127.0.0.1:5000
	$(PY) -m src.webapp --model models/pilotnet.pt --data data/synthetic

lint:             ## ruff + black check
	$(PY) -m ruff check . && $(PY) -m black --check .

docker-build:
	docker build -t pitwall .
