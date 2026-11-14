"""Build side-by-side sheets to eyeball YOLO in rain vs dry: for a few cameras,
one image taken during measured rain and one from the same hour on a dry day,
each shown raw and with YOLO boxes. Output: data/rain_check/*.jpg"""
import duckdb
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from pipeline.config import BRONZE, DATA, IMAGES
from pipeline.detect import CLASSES
from pipeline.transform import connect

OUT = DATA / "rain_check"
OUT.mkdir(exist_ok=True)
con = connect()
pairs = con.execute(f"""
WITH img AS (
    SELECT o.camera_id, o.ts, o.path, o.pcu,
           (SELECT sum(r.rf5min) FROM lake.bronze.bma_rain_5min r
            WHERE r.rain_id = cg.rain_id AND r.ts BETWEEN o.ts - INTERVAL 15 MINUTE AND o.ts) AS rain15
    FROM lake.silver.observations o JOIN lake.silver.camera_gauge cg USING (camera_id)
    WHERE NOT o.is_blank AND hour(o.ts) BETWEEN 8 AND 17
),
wet AS (SELECT * FROM img WHERE rain15 >= 2 QUALIFY row_number() OVER (PARTITION BY camera_id ORDER BY rain15 DESC) = 1),
dry AS (SELECT * FROM img WHERE rain15 = 0 AND ts < TIMESTAMP '2026-08-28')
SELECT w.camera_id, w.path AS wet_path, w.rain15, w.pcu AS wet_pcu, d.path AS dry_path, d.pcu AS dry_pcu
FROM wet w JOIN dry d ON d.camera_id = w.camera_id AND hour(d.ts) = hour(w.ts)
QUALIFY row_number() OVER (PARTITION BY w.camera_id ORDER BY d.ts) = 1
ORDER BY hash(w.camera_id) LIMIT 8
""").df()
print(pairs.to_string())
model = YOLO(str(DATA / "yolo11m.pt"))
for p in pairs.itertuples():
    tiles = []
    for path in (p.dry_path, p.wet_path):
        r = model.predict(str(IMAGES / path), imgsz=640, conf=0.25, classes=list(CLASSES), verbose=False)[0]
        raw = Image.open(IMAGES / path).convert("RGB")
        boxed = Image.fromarray(r.plot(labels=False, line_width=1)[..., ::-1])
        tiles += [raw, boxed]
    w, h = 704, 576
    sheet = Image.new("RGB", (2 * w, 2 * h))
    for i, t in enumerate(tiles):
        sheet.paste(t.resize((w, h), Image.LANCZOS), ((i % 2) * w, (i // 2) * h))
    sheet.save(OUT / f"cam{p.camera_id}.jpg", quality=88)
