"""Extra source 2: road and neighbourhood context per camera from OpenStreetMap
(Overpass API): the class/lanes of the nearest road and counts of nearby places."""
import time

import pandas as pd
import requests

from pipeline.config import BRONZE, CAMERA_CSV

URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "dsde-student-project/0.1"}
ROAD_RANK = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
QUERY = """[out:json][timeout:25];
(
  way(around:50,{lat},{lon})[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"];
  node(around:500,{lat},{lon})[railway=station];
  nwr(around:500,{lat},{lon})[amenity~"^(school|university|college|hospital)$"];
  nwr(around:500,{lat},{lon})[shop~"^(mall|supermarket|department_store)$"];
  node(around:150,{lat},{lon})[highway=traffic_signals];
);
out tags;"""


def summarise(elements: list[dict]) -> dict:
    tags = [e.get("tags", {}) for e in elements]
    roads = [t for t in tags if t.get("highway") in ROAD_RANK]
    best = min(roads, key=lambda t: ROAD_RANK.index(t["highway"]), default={})
    lanes = pd.to_numeric(best.get("lanes"), errors="coerce")
    return {
        "road_class": best.get("highway"),
        "road_name": best.get("name:en") or best.get("name"),
        "lanes": lanes,
        "oneway": best.get("oneway") == "yes",
        "n_stations_500m": sum("railway" in t for t in tags),
        "n_schools_500m": sum(t.get("amenity") in ("school", "university", "college") for t in tags),
        "n_hospitals_500m": sum(t.get("amenity") == "hospital" for t in tags),
        "n_malls_500m": sum("shop" in t for t in tags),
        "n_signals_150m": sum(t.get("highway") == "traffic_signals" for t in tags),
    }


def run() -> pd.DataFrame:
    rows = []
    for cam in pd.read_csv(CAMERA_CSV).itertuples():
        q = QUERY.format(lat=cam.latitude, lon=cam.longitude)
        for attempt in range(5):
            r = requests.post(URL, data={"data": q}, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))  # 429/504: the public server is busy
        r.raise_for_status()
        rows.append({"camera_id": cam.camera_id, **summarise(r.json()["elements"])})
        time.sleep(1)
    df = pd.DataFrame(rows)
    df.to_parquet(BRONZE / "osm_context.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.shape)
    print(df.road_class.value_counts(dropna=False))
    print(df.describe().T[["mean", "max"]])
