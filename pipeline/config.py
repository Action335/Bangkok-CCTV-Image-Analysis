"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ZIP_PATH = DATA / "cctv_images.zip"
CAMERA_CSV = DATA / "cctv_info.csv"
IMAGES = DATA / "images"          # extracted, ASCII names: <cam>/<cam>_<YYYYMMDD>_<HHMMSS>.jpg
BRONZE = DATA / "bronze"          # raw tables as collected
EXPORT = ROOT / "export"          # small gold tables, committed, read by the dashboard

START_DATE, END_DATE = "2026-08-25", "2026-08-31"
TZ = "Asia/Bangkok"

for p in (IMAGES, BRONZE, EXPORT):
    p.mkdir(parents=True, exist_ok=True)
