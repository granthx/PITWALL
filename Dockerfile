# Training / evaluation / console container (CPU).
# The Udacity *simulator* itself stays on the host — only drive.py would talk
# to it, and the sim's window can't run in a container anyway.
FROM python:3.12-slim

# OpenCV runtime libs (libGL) + nothing else exotic
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# CPU torch is already installed above; make sure pip doesn't re-pull the
# CUDA wheel from PyPI inside the container.
ENV PIP_IGNORE_INSTALLED=0

EXPOSE 5000 4567

CMD ["python", "-m", "src.train", "--data", "data/recording", "--epochs", "15"]
