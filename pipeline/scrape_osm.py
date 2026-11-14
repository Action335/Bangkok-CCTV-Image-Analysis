"""Extra source 2: road and neighbourhood context per camera from OpenStreetMap
(Overpass API): the road the camera stands on and counts of nearby places.

The camera's road is the NEAREST street-level road. Expressways and bridges pass
over many of these streets, so elevated or tolled ways are skipped: taking the
highest road class nearby labelled surface-street cameras as motorways.
The raw responses are kept in bronze so the choice can be re-made without re-scraping.
"""
import json
import math
import time

import pandas as pd
import requests

from pipeline.config import BRONZE, CAMERA_CSV

URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "dsde-student-project/0.1"}
RAW = BRONZE / "osm_raw.json"
STREET_CLASSES = ("trunk", "primary", "secondary", "tertiary", "residential", "unclassified")
QUERY = """[out:json][timeout:25];
way(around:60,{lat},{lon})[highway];
out tags geom;
(
  node(around:500,{lat},{lon})[railway=station];
  nwr(around:500,{lat},{lon})[amenity~"^(school|university|college|hospital)$"];
  nwr(around:500,{lat},{lon})[shop~"^(mall|supermarket|department_store)$"];
  node(around:150,{lat},{lon})[highway=traffic_signals];
);
out tags;"""


def fetch_raw() -> dict:
    raw = {}
    for cam in pd.read_csv(CAMERA_CSV).itertuples():
        q = QUERY.format(lat=cam.latitude, lon=cam.longitude)
        for attempt in range(5):
            r = requests.post(URL, data={"data": q}, headers=HEADERS, timeout=90)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))  # 429/504: the public server is busy
        r.raise_for_status()
        raw[str(cam.camera_id)] = r.json()["elements"]
        time.sleep(1)
        if len(raw) % 10 == 0:
            print(f"{len(raw)}/100 cameras", flush=True)
    RAW.write_text(json.dumps(raw), encoding="utf-8")
    return raw


def metres_to_way(lat: float, lon: float, geometry: list[dict]) -> float:
    """Shortest distance from a point to a polyline, on a local flat projection."""
    kx, ky = 111_320 * math.cos(math.radians(lat)), 111_320
    pts = [((p["lon"] - lon) * kx, (p["lat"] - lat) * ky) for p in geometry]
    best = min(math.hypot(x, y) for x, y in pts)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        if length2:
            t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / length2))
            best = min(best, math.hypot(x1 + t * dx, y1 + t * dy))
    return best


def is_street_level(tags: dict) -> bool:
    elevated = tags.get("bridge") not in (None, "no") or tags.get("layer", "0") not in ("0", "-1")
    return tags.get("highway") in STREET_CLASSES and not elevated and tags.get("toll") != "yes"


def summarise(lat: float, lon: float, elements: list[dict]) -> dict:
    ways = [e for e in elements if e["type"] == "way" and "geometry" in e and is_street_level(e.get("tags", {}))]
    nearest = min(ways, key=lambda e: metres_to_way(lat, lon, e["geometry"]), default=None)
    road = nearest["tags"] if nearest else {}
    places = [e.get("tags", {}) for e in elements if not (e["type"] == "way" and "geometry" in e)]
    return {
        "road_class": road.get("highway"),
        "road_name": road.get("name:en") or road.get("name"),
        "road_m": round(metres_to_way(lat, lon, nearest["geometry"]), 1) if nearest else None,
        "lanes": pd.to_numeric(road.get("lanes"), errors="coerce"),
        "oneway": road.get("oneway") == "yes",
        "n_stations_500m": sum("railway" in t for t in places),
        "n_schools_500m": sum(t.get("amenity") in ("school", "university", "college") for t in places),
        "n_hospitals_500m": sum(t.get("amenity") == "hospital" for t in places),
        "n_malls_500m": sum("shop" in t for t in places),
        "n_signals_150m": sum(t.get("highway") == "traffic_signals" for t in places),
    }


def run(refetch: bool = True) -> pd.DataFrame:
    raw = fetch_raw() if refetch or not RAW.exists() else json.loads(RAW.read_text(encoding="utf-8"))
    cams = pd.read_csv(CAMERA_CSV)
    df = pd.DataFrame([{"camera_id": c.camera_id, **summarise(c.latitude, c.longitude, raw[str(c.camera_id)])}
                       for c in cams.itertuples()])
    df.to_parquet(BRONZE / "osm_context.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.shape)
    print(df.road_class.value_counts(dropna=False))
    print(df[["road_m", "lanes"]].describe().T[["mean", "max"]])
