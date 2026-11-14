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
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lower = g[g.shape[0] // 2:]
    return {
        "corrupt": False,
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
    df.to_parquet(BRONZE / "image_index.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.shape, df.ts.min(), df.ts.max(), "corrupt:", int(df.corrupt.sum()))
