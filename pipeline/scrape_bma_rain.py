"""Extra source 4: 5-minute rainfall from the BMA Drainage Department rain gauges
(weather.bangkok.go.th/rain). ~129 gauges with coordinates, far finer than the
ERA5 weather grid. The history form returns the readings as a JSON array inside
the page and accepts at most a 2-day range, so we ask gauge by gauge, 2 days at a time."""
import json
import re
import time

import pandas as pd
import requests

from pipeline.config import BRONZE, END_DATE, START_DATE

URL = "https://weather.bangkok.go.th/rain/RainHistory"
HEADERS = {"User-Agent": "dsde-student-project/0.1"}
STATIONS_RE = re.compile(r"var datawater = (\[.*?\]);", re.S)
READINGS_RE = re.compile(r"rainhistoryList\s*=\s*(\[.*?\]);\s*if \(!rainhistoryList\)", re.S)
KEEP = ["rain_id", "site_timestamp", "rf5min", "rf15min", "rf1hr"]


def stations(session: requests.Session) -> pd.DataFrame:
    html = session.get(URL, headers=HEADERS, timeout=60).text
    df = pd.DataFrame(json.loads(STATIONS_RE.search(html).group(1)))
    return df[["rain_id", "rain_code", "rain_name_en", "district_name_en", "latitude", "longitude"]]


def readings(session: requests.Session, rain_id: int, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    form = {"rain_station": rain_id, "rain_field_selected": 0, "rain_data": 0, "txtFilter": "0.0",
            "datePick_start": start.strftime("%d/%m/%Y"), "StationTime_start": "00:00",
            "datePick_end": end.strftime("%d/%m/%Y"), "StationTime_end": "23:59"}
    for attempt in range(4):
        try:
            r = session.post(URL, data=form, headers=HEADERS, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"gauge {rain_id} {start:%d/%m} failed 4 times")
    m = READINGS_RE.search(r.text)  # no match = the gauge has no data in this window
    return json.loads(m.group(1)) if m else []


def run() -> pd.DataFrame:
    session = requests.Session()
    st = stations(session)
    st.to_parquet(BRONZE / "bma_rain_stations.parquet", index=False)
    windows = [(d, min(d + pd.Timedelta(days=1), pd.Timestamp(END_DATE)))
               for d in pd.date_range(START_DATE, END_DATE, freq="2D")]
    rows = []
    for n, rain_id in enumerate(st.rain_id):
        for start, end in windows:
            rows += [{k: x[k] for k in KEEP} for x in readings(session, rain_id, start, end)]
            time.sleep(1)
        if n % 10 == 0:
            print(f"{n}/{len(st)} gauges, {len(rows)} readings", flush=True)
    if not rows:
        raise RuntimeError("No rain readings parsed for any gauge: the BMA page layout may have changed")
    df = pd.DataFrame(rows).rename(columns={"site_timestamp": "ts"})
    silent = sorted(set(st.rain_id) - set(df.rain_id))
    print(f"{df.rain_id.nunique()} gauges returned data; none from gauge ids {silent}")
    df["ts"] = pd.to_datetime(df.ts)
    df.to_parquet(BRONZE / "bma_rain_5min.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(df.shape, df.rain_id.nunique(), "gauges", df.ts.min(), df.ts.max())
