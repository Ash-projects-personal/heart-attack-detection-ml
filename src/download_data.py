"""Download the UCI Heart Disease dataset (dataset 45).

Uses the processed Cleveland file: 303 patients, 13 clinical
features, target "num" from 0 (no disease) to 4. This project
binarises it to disease present (num > 0) or not.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
License: CC BY 4.0.
"""
import hashlib
import pathlib
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    target = ROOT / "data" / "processed.cleveland.data"
    if target.exists():
        print("exists", target)
        return
    zip_path = ROOT / "data" / "heart_disease.zip"
    urllib.request.urlretrieve(URL, zip_path)
    print("zip sha256", hashlib.sha256(zip_path.read_bytes()).hexdigest())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ROOT / "data" / "raw")
    zip_path.unlink()
    src = ROOT / "data" / "raw" / "processed.cleveland.data"
    target.write_bytes(src.read_bytes())
    print("saved", target, len(target.read_text().splitlines()), "lines")


if __name__ == "__main__":
    main()
