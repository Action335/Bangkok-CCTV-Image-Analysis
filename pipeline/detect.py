"""Step 2: count road users in every image with a pretrained YOLO11m (COCO)."""
import pandas as pd
from ultralytics import YOLO

from pipeline.config import BRONZE, DATA, IMAGES

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BATCH = 64


def run(conf: float = 0.25, imgsz: int = 640) -> pd.DataFrame:
    index = pd.read_parquet(BRONZE / "image_index.parquet")
    paths = index.loc[~index.corrupt, "path"].tolist()
    model = YOLO(str(DATA / "yolo11m.pt"))
    rows = []
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        results = model.predict([str(IMAGES / p) for p in chunk], imgsz=imgsz, conf=conf,
                                classes=list(CLASSES), verbose=False)
        for p, r in zip(chunk, results):
            cls = r.boxes.cls.int().tolist()
            row = {"path": p, "mean_conf": float(r.boxes.conf.mean()) if cls else None}
            row.update({f"n_{name}": cls.count(cid) for cid, name in CLASSES.items()})
            rows.append(row)
        if i % (BATCH * 50) == 0:
            print(f"{i}/{len(paths)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(BRONZE / "yolo_counts.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.shape)
    print(df.filter(like="n_").describe().T[["mean", "max"]])
