# Bangkok CCTV Image Analysis (2110403 DSDE project)

**Question:** where and when does Bangkok traffic get congested, and how much does rain add?

**Live dashboard:** https://bangkok-cctv-image-analysis-3ljqcfiugtcowj7c877sgm.streamlit.app/

Data: 61,319 images from 100 BMA traffic cameras, every 10 min, 06:00–21:00, 25–31 Aug 2026.

## Pipeline
```
cctv_images.zip ─ extract ─► bronze.image_index ─┐
                  detect (YOLO11m) ─► yolo_counts ├─► silver.observations ─► gold.camera_hour ─► analysis ─► dashboard
Open-Meteo weather + air quality, OpenStreetMap, BMA rain gauges ─┘                          gold.cameras
```
| Step | File | Output |
|---|---|---|
| Unzip, timestamps, hashes, pixel stats | `pipeline/extract.py` | `data/bronze/image_index.parquet` |
| Vehicle counts (YOLO11m, COCO) | `pipeline/detect.py` | `data/bronze/yolo_counts.parquet` |
| Weather (ERA5) + air quality (CAMS) via Open-Meteo | `pipeline/scrape_weather.py` | `weather.parquet`, `air_quality.parquet` |
| Road class and lanes of the nearest street-level road (elevated and toll roads skipped), nearby stations/schools/malls (OpenStreetMap Overpass) | `pipeline/scrape_osm.py` | `osm_context.parquet` |
| 5-minute rainfall from 125 BMA rain gauges (weather.bangkok.go.th/rain), nearest gauge per camera | `pipeline/scrape_bma_rain.py` | `bma_rain_stations.parquet`, `bma_rain_5min.parquet` |
| DuckLake bronze → silver → gold | `pipeline/transform.py` | `data/lake/`, `export/*.parquet` |
| K-means camera groups, rain regression | `pipeline/analyze.py` | `export/camera_clusters.parquet`, `export/rain_effect.json` |
| Prefect flow that runs all of it | `pipeline/flow.py` | |
| Streamlit dashboard | `dashboard/app.py` | |

## Run
```
uv sync
# put cctv_images.zip and cctv_info.csv (course Drive folder) in data/
uv run python -m pipeline.flow            # add --force to redo unzip + YOLO (~1 h on an RTX 5060)
uv run streamlit run dashboard/app.py     # only needs export/, no images
```

## Definitions
- **PCU** (vehicles in view) = 0.33 × motorcycles + cars + 2.5 × buses. YOLO's "truck" class is counted as a car: the hand count (`analysis/hand_count.py`) found 16 real trucks in 98 images where YOLO reported 116, mostly pickups and vans.
- **Congestion** = PCU as a z-score within each camera, because camera angles differ.
- **Congestion at night** (after 18:30) is scored against the camera's own night level, because YOLO sees fewer vehicles in the dark.

## Cleaning rules
Rows are flagged, never deleted; `export/quality_report.parquet` (and the dashboard's Data quality tab) shows how many rows each rule touches.
- **Blank frame** = pixel contrast < 10 (grey / no-signal). Excluded from counts.
- **Frozen frame** = identical to the same camera's previous frame (mean pixel change < 1 on a 32×32 thumbnail). Excluded from counts.
- **Thin hour** = a camera-hour with fewer than 3 usable images. Its averages are left empty.
- **Dead rain gauge** = a gauge that read zero all week. Not matched to any camera. Gauge-hours need at least 10 of 12 readings.
- **Rain level** (nearest BMA gauge, median 0.85 km away) = dry < 0.5 mm/h, light < 2.5 mm/h, heavy ≥ 2.5 mm/h.

See `TEAM_PLAN.md` for roles, milestones and the limits to state in the slides.
