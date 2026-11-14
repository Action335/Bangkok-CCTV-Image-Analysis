# Bangkok CCTV Image Analysis (2110403 DSDE project)

**Question:** where and when does Bangkok traffic get congested, and how much does rain add?

Data: 61,319 images from 100 BMA traffic cameras, every 10 min, 06:00–21:00, 25–31 Aug 2026.

## Pipeline
```
cctv_images.zip ─ extract ─► bronze.image_index ─┐
                  detect (YOLO11m) ─► yolo_counts ├─► silver.observations ─► gold.camera_hour ─► analysis ─► dashboard
Open-Meteo weather + air quality, OpenStreetMap ──┘                          gold.cameras
```
| Step | File | Output |
|---|---|---|
| Unzip, timestamps, hashes, pixel stats | `pipeline/extract.py` | `data/bronze/image_index.parquet` |
| Vehicle counts (YOLO11m, COCO) | `pipeline/detect.py` | `data/bronze/yolo_counts.parquet` |
| Weather (ERA5) + air quality (CAMS) via Open-Meteo | `pipeline/scrape_weather.py` | `weather.parquet`, `air_quality.parquet` |
| Road class, lanes, nearby stations/schools/malls (OpenStreetMap Overpass) | `pipeline/scrape_osm.py` | `osm_context.parquet` |
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
- **PCU** (vehicles in view) = 0.33 × motorcycles + cars + 2.5 × (buses + trucks).
- **Congestion** = PCU as a z-score within each camera, because camera angles differ.
- **Blank frame** = pixel contrast < 10 (grey / no-signal). Flagged, not deleted.
- **Rain level** = dry < 0.5 mm/h, light < 2.5 mm/h, heavy ≥ 2.5 mm/h.

See `TEAM_PLAN.md` for roles, milestones and the limits to state in the slides.
