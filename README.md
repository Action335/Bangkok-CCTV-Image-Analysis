# Bangkok CCTV Image Analysis

What does rain do to Bangkok traffic? This project turns one week of raw traffic-camera images into hourly traffic data, joins it with rainfall from the city's own rain gauges, and shows the result in a dashboard.

**Live dashboard:** https://bangkok-cctv-image-analysis-3ljqcfiugtcowj7c877sgm.streamlit.app/

Course project for 2110403 Data Science and Data Engineering, Chulalongkorn University (2026/1).

## Findings

Data: 61,319 images from 100 Bangkok Metropolitan Administration (BMA) traffic cameras, one image every 10 minutes, 06:00–21:00, 25–31 Aug 2026.

1. **Rain takes motorcycles off the road; cars stay.** In rainy daylight hours YOLO counts 28% fewer motorcycles than in dry hours on the same day, while cars and buses do not change. YOLO misses most motorcycles, so the size is overstated, but our own count of 100 images shows the same direction: 4.7 motorcycles per image when dry against 2.7 in rain.
2. **Weekends are quieter.** About 15% fewer vehicles in view than on weekdays in daylight, with Sunday the quietest day. There is only one weekend in the data, so this is suggestive.
3. **Road type does not change how full the road looks.** Primary, secondary and tertiary roads all show about 7.5 vehicles in view.

## How it works

```
cctv_images.zip ─ extract ─► bronze.image_index ─┐
                  detect (YOLO11m) ─► yolo_counts ├─► silver.observations ─► gold.camera_hour ─► analysis ─► dashboard
Open-Meteo weather + air quality, OpenStreetMap, BMA rain gauges ─┘                          gold.cameras
```

| Step | File | Output |
|---|---|---|
| Unzip, timestamps, hashes, pixel stats | `pipeline/extract.py` | `data/bronze/image_index.parquet` |
| Vehicle counts (YOLO11m, COCO classes) | `pipeline/detect.py` | `data/bronze/yolo_counts.parquet` |
| 5-minute rainfall from 125 BMA rain gauges, nearest gauge per camera | `pipeline/scrape_bma_rain.py` | `bma_rain_stations.parquet`, `bma_rain_5min.parquet` |
| Road class and lanes of the nearest street-level road, nearby stations, schools and malls (OpenStreetMap Overpass) | `pipeline/scrape_osm.py` | `osm_context.parquet` |
| Hourly weather (ERA5) and air quality (CAMS) from Open-Meteo | `pipeline/scrape_weather.py` | `weather.parquet`, `air_quality.parquet` |
| DuckLake tables, bronze → silver → gold, with cleaning rules | `pipeline/transform.py` | `data/lake/`, `export/*.parquet` |
| K-means camera groups and rain regression | `pipeline/analyze.py` | `export/camera_clusters.parquet`, `export/rain_effect.json` |
| Prefect flow that runs every step | `pipeline/flow.py` | |
| Hand count used to check YOLO | `analysis/hand_count.py` | `export/hand_count_*.csv` |
| Analysis notebook | `analysis/analysis.ipynb` | |
| Streamlit dashboard | `dashboard/app.py` | |

## Run it

The dashboard and the notebook only need the small tables in `export/`, which are committed:

```
uv sync
uv run streamlit run dashboard/app.py
```

To rebuild everything from the raw images, put `cctv_images.zip` and `cctv_info.csv` in `data/`. The image set was provided by the course and is not redistributed here.

```
uv run python -m pipeline.flow            # add --force to redo unzip + YOLO (about 1 h on an RTX 5060)
```

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). `pyproject.toml` pins the CUDA 12.8 build of PyTorch.

## Definitions

- **Vehicles in view (PCU)** = 0.33 × motorcycles + cars + 2.5 × buses. YOLO's "truck" class is counted as a car: in the hand count it reported 116 trucks where 16 were real, the rest being pickups and vans.
- **Congestion** = PCU as a z-score within each camera, because camera angles differ. After 18:30 it is scored against the camera's own night level, because YOLO sees fewer vehicles in the dark.
- **Rain level** = rainfall at the nearest BMA gauge (median 0.85 km away): dry < 0.5 mm/h, light < 2.5 mm/h, heavy ≥ 2.5 mm/h.

## Cleaning rules

Rows are flagged, never deleted. `export/quality_report.parquet` and the dashboard's Data quality tab show how many rows each rule touches.

- **Blank frame:** pixel contrast below 10 (grey or no-signal). Excluded from counts.
- **Frozen frame:** identical to the same camera's previous frame. Excluded from counts.
- **Thin hour:** a camera-hour with fewer than 3 usable images. Its averages are left empty.
- **Dead rain gauge:** a gauge that read zero all week. Not matched to any camera.

## How far to trust the counts

We counted 100 images by hand (dry daylight, rain, night, and images where YOLO reported many trucks), blind to YOLO's numbers.

- YOLO11m found about 60% of the cars and 11% of the motorcycles. Distant vehicles are only a few pixels in these 352×288 images.
- The vehicles-in-view score is therefore too low in level (about 60% of the hand count by day, 45% at night), but it ranks images well (correlation 0.86). It is used to compare, not to measure.
- How many motorcycles YOLO finds depends on the road (5% on primary roads, 19% on secondary roads), so motorcycle counts are compared within a camera, never between cameras.

## Limits

- One week of images with one weekend; almost all rain fell on 29–31 Aug.
- A photo shows vehicles in view, not speed, so a jam and free-flowing traffic can look alike.
- The source images have gaps: Thursday 27 Aug has almost no usable images before 09:00, and 21:00 has one frame per camera.
- Air quality and non-rain weather come from coarse model grids and are not used for conclusions.

## Data sources and credits

- Traffic camera images and rain gauge readings: Bangkok Metropolitan Administration.
- Road and place data: © OpenStreetMap contributors, ODbL.
- Weather and air quality: [Open-Meteo](https://open-meteo.com/), CC BY 4.0.
- Object detection: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).
