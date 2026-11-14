"""Step 1: unzip the raw CCTV images and build the bronze image index.

The zip has Thai file names that break some Windows tools, so each image is
renamed to <camera_id>_<YYYYMMDD>_<HHMMSS>.jpg. Per image we record the
timestamp, an MD5 hash (to find frozen/duplicate frames) and cheap pixel
statistics used later for quality flags and the rain cross-check.
"""
import hashlib
import io
import re
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from pipeline.config import BRONZE, IMAGES, ZIP_PATH

NAME_RE = re.compile(r"(\d+)/[^/]*_(\d{8})_(\d{6})\.jpg$")


def pixel_stats(raw: bytes) -> dict:
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return {"corrupt": True}
    grey = img.convert("L")
    g = np.asarray(grey, dtype=np.float32)
    lower = g[g.shape[0] // 2:]
    return {
        "corrupt": False,
        "thumb": np.asarray(grey.resize((32, 32)), dtype=np.float32),  # for the frozen-frame check
        "width": img.width,
        "height": img.height,
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        # mean absolute horizontal gradient: low when the lens is blurred by rain/fog
        "sharpness": float(np.abs(np.diff(g, axis=1)).mean()),
        # share of very bright pixels in the road half: headlight glare / wet reflections
        "glare": float((lower > 230).mean()),
    }


def run() -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        for info in z.infolist():
            m = NAME_RE.search(info.filename)
            if not m:
                continue
            cam, d, t = m.groups()
            raw = z.read(info)
            out = IMAGES / cam / f"{cam}_{d}_{t}.jpg"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(raw)
            rows.append({
                "camera_id": int(cam),
                "ts": pd.Timestamp(f"{d} {t}"),
                "path": out.relative_to(IMAGES).as_posix(),
                "bytes": len(raw),
                "md5": hashlib.md5(raw).hexdigest(),
                **pixel_stats(raw),
            })
    df = pd.DataFrame(rows).sort_values(["camera_id", "ts"]).reset_index(drop=True)
    # mean pixel change from the same camera's previous frame; ~0 means the feed was frozen
    thumbs = np.stack(df.pop("thumb"))
    change = np.abs(thumbs[1:] - thumbs[:-1]).mean(axis=(1, 2))
    same_cam = df.camera_id.values[1:] == df.camera_id.values[:-1]
    df["change_from_prev"] = np.concatenate([[np.nan], np.where(same_cam, change, np.nan)])
    df.to_parquet(BRONZE / "image_index.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.shape, df.ts.min(), df.ts.max(), "corrupt:", int(df.corrupt.sum()))
