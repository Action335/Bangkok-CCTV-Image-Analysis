"""Ground-truth check of the YOLO counts against a human count of 100 images.

    uv run python -m analysis.hand_count sample   # writes data/hand_count/: images, count.html, key.csv
    (open data/hand_count/count.html, count, press Download, save hand_counts.csv into data/hand_count/)
    uv run python -m analysis.hand_count score    # compares hand_counts.csv with YOLO

The sample is stratified so the weak spots are covered: dry daylight, measured rain,
night, and images where YOLO reported several trucks. The counting page shows no YOLO
numbers, so the human count is blind.
"""
import json
import sys

import pandas as pd
from PIL import Image

from pipeline.config import DATA, EXPORT, IMAGES
from pipeline.transform import connect

OUT = DATA / "hand_count"
CLASSES = ["car", "motorcycle", "bus", "truck", "person"]
STRATA = {"dry day": 40, "rain day": 25, "night": 20, "many trucks": 15}
SEED = 403


def sample() -> None:
    obs = connect().execute("""
    SELECT o.camera_id, o.ts, o.path, o.is_night, o.n_car, o.n_motorcycle, o.n_bus, o.n_truck, o.n_person,
           coalesce((SELECT sum(r.rf5min) FROM lake.bronze.bma_rain_5min r
                     WHERE r.rain_id = cg.rain_id AND r.ts BETWEEN o.ts - INTERVAL 15 MINUTE AND o.ts), 0) AS rain15
    FROM lake.silver.observations o JOIN lake.silver.camera_gauge cg USING (camera_id)
    WHERE o.usable""").df()
    day = ~obs.is_night & obs.ts.dt.hour.between(7, 17)
    pools = {"many trucks": obs[day & (obs.n_truck >= 3) & (obs.rain15 == 0)],
             "rain day": obs[day & (obs.rain15 >= 1)],
             "night": obs[obs.is_night],
             "dry day": obs[day & (obs.rain15 == 0)]}
    picked, used = [], set()
    for name, pool in pools.items():  # one image per camera within a stratum, so no camera dominates
        pool = pool[~pool.path.isin(used)].sample(frac=1, random_state=SEED).drop_duplicates("camera_id")
        take = pool.head(STRATA[name]).assign(stratum=name)
        assert len(take) == STRATA[name], (name, len(take))
        picked.append(take)
        used |= set(take.path)
    key = pd.concat(picked).sample(frac=1, random_state=SEED).reset_index(drop=True)
    key.insert(0, "id", [f"{i:03d}" for i in range(1, len(key) + 1)])

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for row in key.itertuples():
        im = Image.open(IMAGES / row.path).convert("RGB")
        im.resize((im.width * 2, im.height * 2), Image.LANCZOS).save(OUT / "images" / f"{row.id}.jpg", quality=92)
    key.to_csv(OUT / "key.csv", index=False)
    (OUT / "count.html").write_text(PAGE.replace("__IDS__", json.dumps(key.id.tolist())), encoding="utf-8")
    print(key.stratum.value_counts().to_dict(), "| cameras:", key.camera_id.nunique())
    print("Open", OUT / "count.html")


def score() -> None:
    hand = pd.read_csv(OUT / "hand_counts.csv", dtype={"id": str})
    df = pd.read_csv(OUT / "key.csv", dtype={"id": str}).merge(hand, on="id")
    print(f"{len(df)} images counted by hand\n")
    rows = []
    for c in CLASSES:
        y, h = df[f"n_{c}"], df[c]
        rows.append({"class": c, "hand_total": int(h.sum()), "yolo_total": int(y.sum()),
                     "yolo_per_hand": round(y.sum() / h.sum(), 2) if h.sum() else None,
                     "mean_abs_error": round((y - h).abs().mean(), 2)})
    by_class = pd.DataFrame(rows)
    by_class["correlation"] = [round(df[c].corr(df[f"n_{c}"]), 2) for c in CLASSES]
    print(by_class.to_string(index=False))

    # YOLO side uses the pipeline's formula (its "truck" counts as a car); the hand side weights real trucks 2.5.
    df["pcu_yolo"] = df.n_car + df.n_truck + 0.33 * df.n_motorcycle + 2.5 * df.n_bus
    df["pcu_hand"] = df.car + 0.33 * df.motorcycle + 2.5 * (df.bus + df.truck)
    by_stratum = df.groupby("stratum").agg(images=("id", "size"), pcu_hand=("pcu_hand", "mean"),
                                           pcu_yolo=("pcu_yolo", "mean")).round(2)
    by_stratum["yolo_per_hand"] = (by_stratum.pcu_yolo / by_stratum.pcu_hand).round(2)
    print("\nVehicles in view (PCU) by image type\n" + by_stratum.to_string())
    print("\nPCU correlation, hand vs YOLO: %.2f" % df.pcu_hand.corr(df.pcu_yolo))
    df[["id", "path", "stratum"] + [f"n_{c}" for c in CLASSES] + CLASSES].to_csv(EXPORT / "hand_count_images.csv", index=False)
    by_class.to_csv(EXPORT / "hand_count_by_class.csv", index=False)
    by_stratum.reset_index().to_csv(EXPORT / "hand_count_by_stratum.csv", index=False)


PAGE = """<!doctype html><meta charset="utf-8"><title>Hand count</title>
<style>
body{font:15px system-ui;margin:0 auto;max-width:1100px;padding:16px;background:#f5f6f8;color:#16212b}
.top{position:sticky;top:0;background:#f5f6f8;padding:10px 0;border-bottom:1px solid #ccd;z-index:1}
.row{display:flex;gap:20px;align-items:flex-start;padding:14px 0;border-bottom:1px solid #dde}
.row.done{opacity:.55}
img{width:704px;height:576px;border-radius:6px}
label{display:flex;justify-content:space-between;gap:10px;margin:6px 0;width:210px}
input{width:70px;font:inherit;padding:4px}
button{font:inherit;padding:8px 14px}
</style>
<div class="top"><b>Hand count</b> · <span id="prog"></span> · <button id="dl">Download hand_counts.csv</button>
<div>Count everything you can recognise anywhere in the frame, moving or parked. <b>Car</b> includes pickups, vans, taxis and
tuk-tuks. <b>Truck</b> means six wheels or more. <b>Person</b> means on foot, not riders. Fill in <b>car</b> for every image
(0 if none); other boxes left empty count as 0. Your entries are saved in this browser as you type.</div></div>
<div id="rows"></div>
<script>
const ids = __IDS__, classes = ["car", "motorcycle", "bus", "truck", "person"], KEY = "handcount403";
let data = {};
try { data = JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) {}
const rows = document.getElementById("rows");
function refresh() {
  const done = ids.filter(id => data[id] && data[id].car !== undefined && data[id].car !== "");
  document.getElementById("prog").textContent = done.length + " of " + ids.length + " counted";
  ids.forEach(id => document.getElementById("r" + id).classList.toggle("done", done.includes(id)));
  return done;
}
ids.forEach(id => {
  const row = document.createElement("div");
  row.className = "row"; row.id = "r" + id;
  row.innerHTML = '<img loading="lazy" src="images/' + id + '.jpg"><div><b>Image ' + id + "</b>" +
    classes.map(c => "<label>" + c + ' <input type="number" min="0" data-id="' + id + '" data-c="' + c + '"></label>').join("") + "</div>";
  rows.appendChild(row);
});
document.querySelectorAll("input").forEach(el => {
  const saved = data[el.dataset.id];
  if (saved && saved[el.dataset.c] !== undefined) el.value = saved[el.dataset.c];
  el.addEventListener("input", () => {
    (data[el.dataset.id] = data[el.dataset.id] || {})[el.dataset.c] = el.value;
    try { localStorage.setItem(KEY, JSON.stringify(data)); } catch (e) {}
    refresh();
  });
});
document.getElementById("dl").addEventListener("click", () => {
  const lines = ["id," + classes.join(",")].concat(
    refresh().map(id => id + "," + classes.map(c => Number(data[id][c] || 0)).join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {type: "text/csv"}));
  a.download = "hand_counts.csv";
  a.click();
});
refresh();
</script>"""

if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
