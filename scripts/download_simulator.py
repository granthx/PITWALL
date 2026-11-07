"""Download and extract the Udacity Self-Driving Car Simulator (Windows 64-bit)."""

import os
import urllib.request
import zipfile
from tqdm import tqdm

URL = "https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip"
TARGET_DIR = os.path.abspath("simulator")
ZIP_PATH = os.path.abspath("simulator_win64.zip")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def main():
    if os.path.isdir(TARGET_DIR):
        for root, _dirs, files in os.walk(TARGET_DIR):
            for f in files:
                if f.endswith(".exe"):
                    return os.path.join(root, f)

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(URL, headers=headers)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading") as t:
        with urllib.request.urlopen(req) as response, open(ZIP_PATH, "wb") as out_file:
            total_size = response.headers.get("content-length")
            if total_size:
                t.total = int(total_size)
            chunk_size = 64 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                t.update(len(chunk))

    os.makedirs(TARGET_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(TARGET_DIR)
    if os.path.exists(ZIP_PATH):
        try: os.remove(ZIP_PATH)
        except OSError: pass

if __name__ == "__main__":
    main()
