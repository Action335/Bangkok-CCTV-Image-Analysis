"""Extra source 1 + 3: hourly weather (ERA5) and air quality (CAMS) per camera
from the Open-Meteo APIs. Both are model grids, so nearby cameras share values."""
import pandas as pd
import requests

from pipeline.config import BRONZE, CAMERA_CSV, END_DATE, START_DATE, TZ

APIS = {
    "weather": ("https://archive-api.open-meteo.com/v1/archive",
                "precipitation,rain,temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m"),
    "air_quality": ("https://air-quality-api.open-meteo.com/v1/air-quality",
                    "pm2_5,pm10,nitrogen_dioxide,carbon_monoxide"),
}
CHUNK = 50  # locations per request


def fetch(name: str, cams: pd.DataFrame) -> pd.DataFrame:
    url, hourly = APIS[name]
    frames = []
    for i in range(0, len(cams), CHUNK):
        part = cams.iloc[i:i + CHUNK]
        r = requests.get(url, timeout=60, params={
            "latitude": ",".join(part.latitude.astype(str)),
            "longitude": ",".join(part.longitude.astype(str)),
            "start_date": START_DATE, "end_date": END_DATE,
            "hourly": hourly, "timezone": TZ,
        })
        r.raise_for_status()
        for cam_id, loc in zip(part.camera_id, r.json()):
            df = pd.DataFrame(loc["hourly"]).rename(columns={"time": "hour"})
            df["hour"] = pd.to_datetime(df["hour"])
            df.insert(0, "camera_id", cam_id)
            df["grid_lat"], df["grid_lon"] = loc["latitude"], loc["longitude"]
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run() -> None:
    cams = pd.read_csv(CAMERA_CSV)
    for name in APIS:
        df = fetch(name, cams)
        df.to_parquet(BRONZE / f"{name}.parquet", index=False)
        print(name, df.shape, "grid cells:", df[["grid_lat", "grid_lon"]].drop_duplicates().shape[0])


if __name__ == "__main__":
    run()
