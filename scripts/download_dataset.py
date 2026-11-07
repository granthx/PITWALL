"""Download and prepare the official Udacity sample dataset.

Extracts driving_log.csv and IMG/ into data/recording/
"""

import os
import shutil
import urllib.request
import zipfile

from tqdm import tqdm

URL = "https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip"
TARGET_DIR = os.path.abspath("data/recording")
ZIP_PATH = os.path.abspath("data/sample_udacity.zip")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def main():
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)

    csv_dest = os.path.join(TARGET_DIR, "driving_log.csv")
    img_dest = os.path.join(TARGET_DIR, "IMG")

    if os.path.exists(csv_dest) and os.path.exists(img_dest):
        print(f"Dataset already exists at {TARGET_DIR}")
        return

    print(f"Downloading Udacity sample dataset from {URL} ...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Dataset zip") as t:
        urllib.request.urlretrieve(URL, filename=ZIP_PATH, reporthook=t.update_to)

    print("Extracting zip archive...")
    extract_temp = os.path.abspath("data/_extract_temp")
    if os.path.exists(extract_temp):
        shutil.rmtree(extract_temp)
    os.makedirs(extract_temp, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(extract_temp)

    # Locate driving_log.csv inside the extracted tree
    found_csv = None
    for root, _dirs, files in os.walk(extract_temp):
        if "driving_log.csv" in files:
            found_csv = os.path.join(root, "driving_log.csv")
            found_img = os.path.join(root, "IMG")
            break

    if not found_csv:
        raise RuntimeError("Could not find driving_log.csv inside extracted archive")

    print(f"Moving extracted dataset into {TARGET_DIR} ...")
    if os.path.exists(csv_dest):
        os.remove(csv_dest)
    shutil.move(found_csv, csv_dest)

    if os.path.exists(img_dest):
        shutil.rmtree(img_dest)
    shutil.move(found_img, img_dest)

    # Cleanup temp
    shutil.rmtree(extract_temp, ignore_errors=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    num_images = len(os.listdir(img_dest)) if os.path.exists(img_dest) else 0
    with open(csv_dest) as f:
        num_lines = sum(1 for _ in f)

    print(f"Dataset successfully prepared at {TARGET_DIR}:")
    print(f"  - driving_log.csv: {num_lines} records")
    print(f"  - IMG/: {num_images} images")


if __name__ == "__main__":
    main()

