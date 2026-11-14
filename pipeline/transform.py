"""Step 4: load everything into a DuckLake lakehouse (bronze -> silver -> gold)
and export the small gold tables for the dashboard.

bronze  raw tables exactly as collected
silver  one row per image: cleaning flags, weighted vehicle count, per-camera congestion score
gold    camera x hour table joined with rain, weather and air quality; one row per camera;
        a quality report with one row per cleaning rule
"""
import duckdb

from pipeline.config import BRONZE, CAMERA_CSV, DATA, EXPORT

LAKE = DATA / "lake"

MIN_VALID = 3  # usable images needed before a camera-hour's averages are trusted

# Cleaning rules, applied as flags (nothing is deleted, so every rule can be audited):
#   is_blank   grey / "no signal" frame                    -> not usable
#   is_frozen  identical to the camera's previous frame     -> not usable (feed was stuck)
#   is_night   after 18:30, YOLO misses many vehicles       -> scored against the camera's night level
SILVER = """
CREATE OR REPLACE TABLE lake.silver.observations AS
WITH obs AS (
    SELECT i.camera_id, i.ts, date_trunc('hour', i.ts) AS hour, i.path,
           i.brightness, i.contrast, i.sharpness, i.glare,
           i.contrast < 10 AS is_blank,
           coalesce(i.change_from_prev < 1, false) AND i.contrast >= 10 AS is_frozen,
           i.ts::TIME >= TIME '18:30:00' AS is_night,
           y.n_person, y.n_bicycle, y.n_car, y.n_motorcycle, y.n_bus, y.n_truck, y.mean_conf,
           -- passenger-car units: motorcycle 0.33, car 1, bus 2.5. YOLO's "truck" is weighted as a car:
           -- the hand count found 16 real trucks where YOLO reported 116 (mostly pickups and vans).
           0.33 * y.n_motorcycle + y.n_car + y.n_truck + 2.5 * y.n_bus AS pcu
    FROM lake.bronze.image_index i
    JOIN lake.bronze.yolo_counts y USING (path)
), flagged AS (
    SELECT *, NOT is_blank AND NOT is_frozen AS usable FROM obs
)
SELECT *,
       -- congestion relative to this camera's own normal level, separately for day and
       -- night, so the drop in detections after dark is not read as empty roads
       CASE WHEN usable THEN
           (pcu - avg(pcu) FILTER (usable) OVER cam)
           / nullif(stddev(pcu) FILTER (usable) OVER cam, 0) END AS congestion_z
FROM flagged
WINDOW cam AS (PARTITION BY camera_id, is_night)
"""

GAUGES = """
-- hourly rain per BMA gauge; a gauge that read zero all week is treated as dead
CREATE OR REPLACE TABLE lake.silver.gauge_hour AS
SELECT rain_id, date_trunc('hour', ts) AS hour, sum(rf5min) AS rain_mm, count(rf5min) AS n_readings
FROM lake.bronze.bma_rain_5min
WHERE rf5min >= 0
  AND rain_id IN (SELECT rain_id FROM lake.bronze.bma_rain_5min GROUP BY rain_id HAVING sum(rf5min) > 0)
GROUP BY ALL
HAVING count(rf5min) >= 10;   -- at least 10 of the 12 five-minute readings in the hour

-- each camera's nearest working gauge
CREATE OR REPLACE TABLE lake.silver.camera_gauge AS
SELECT c.camera_id, s.rain_id, s.rain_name_en AS gauge_name,
       -- flat-earth distance in km, fine at city scale
       111.32 * sqrt(pow(c.latitude - s.latitude, 2)
                     + pow((c.longitude - s.longitude) * cos(radians(c.latitude)), 2)) AS gauge_km
FROM lake.bronze.cameras c
CROSS JOIN lake.bronze.bma_rain_stations s
WHERE s.rain_id IN (SELECT rain_id FROM lake.silver.gauge_hour)
QUALIFY row_number() OVER (PARTITION BY c.camera_id ORDER BY gauge_km) = 1;
"""

METRICS = ("pcu", "congestion_z", "n_car", "n_motorcycle", "n_bus", "n_person")

GOLD_HOUR = f"""
CREATE OR REPLACE TABLE lake.gold.camera_hour AS
WITH agg AS (
    SELECT camera_id, hour,
           count(*) AS n_images,
           count(*) FILTER (usable) AS n_valid,
           avg(pcu) FILTER (usable) AS pcu,
           avg(congestion_z) AS congestion_z,
           avg(n_car + n_truck) FILTER (usable) AS n_car,  -- cars include YOLO's "truck" (pickups, vans)
           avg(n_motorcycle) FILTER (usable) AS n_motorcycle,
           avg(n_bus) FILTER (usable) AS n_bus,
           avg(n_person) FILTER (usable) AS n_person,
           avg(brightness) AS brightness, avg(sharpness) AS sharpness, avg(glare) AS glare
    FROM lake.silver.observations
    GROUP BY ALL
)
SELECT a.camera_id, a.hour, a.n_images, a.n_valid,
       -- hours with too few usable images keep their row but lose their averages
       {", ".join(f"CASE WHEN a.n_valid >= {MIN_VALID} THEN a.{m} END AS {m}" for m in METRICS)},
       a.brightness, a.sharpness, a.glare,
       hour(a.hour) AS hour_of_day,
       dayname(a.hour) AS weekday,
       isodow(a.hour) >= 6 AS is_weekend,
       g.rain_mm AS precipitation,               -- nearest BMA gauge, mm in that hour
       w.precipitation AS era5_precipitation,    -- model grid, kept for comparison
       w.temperature_2m, w.relative_humidity_2m, w.cloud_cover, w.wind_speed_10m,
       CASE WHEN g.rain_mm < 0.5 THEN 'dry'
            WHEN g.rain_mm < 2.5 THEN 'light rain'
            WHEN g.rain_mm >= 2.5 THEN 'heavy rain' END AS rain_level,
       q.pm2_5, q.pm10, q.nitrogen_dioxide, q.carbon_monoxide
FROM agg a
LEFT JOIN lake.bronze.weather w USING (camera_id, hour)
LEFT JOIN lake.silver.camera_gauge cg USING (camera_id)
LEFT JOIN lake.silver.gauge_hour g ON g.rain_id = cg.rain_id AND g.hour = a.hour
LEFT JOIN lake.bronze.air_quality q USING (camera_id, hour)
"""

GOLD_CAMERAS = """
CREATE OR REPLACE TABLE lake.gold.cameras AS
SELECT c.camera_id,
       coalesce(nullif(c.location_en, 'connect BMAcctv'), c.location) AS name,
       c.location AS name_th, c.latitude, c.longitude,
       o.* EXCLUDE (camera_id),
       cg.gauge_name, cg.gauge_km,
       s.n_images, s.blank_share, s.frozen_share, s.mean_pcu, s.mean_motorcycle_share
FROM lake.bronze.cameras c
LEFT JOIN lake.bronze.osm_context o USING (camera_id)
LEFT JOIN lake.silver.camera_gauge cg USING (camera_id)
LEFT JOIN (
    SELECT camera_id, count(*) AS n_images,
           avg(is_blank::INT) AS blank_share, avg(is_frozen::INT) AS frozen_share,
           avg(pcu) FILTER (usable AND NOT is_night) AS mean_pcu,
           sum(n_motorcycle) FILTER (usable)
               / nullif(sum(n_motorcycle + n_car + n_bus + n_truck) FILTER (usable), 0) AS mean_motorcycle_share
    FROM lake.silver.observations GROUP BY camera_id
) s USING (camera_id)
"""

# One row per cleaning rule: how many rows it touches and what happens to them.
QUALITY = f"""
CREATE OR REPLACE TABLE lake.gold.quality_report AS
WITH o AS (SELECT * FROM lake.silver.observations), h AS (SELECT * FROM lake.gold.camera_hour)
SELECT * FROM (VALUES
    (1, 'Images collected', (SELECT count(*) FROM o), 'images', 'starting point'),
    (2, 'Corrupt files', (SELECT count(*) FROM lake.bronze.image_index WHERE corrupt), 'images', 'would be dropped'),
    (3, 'Blank / no-signal frames', (SELECT count(*) FROM o WHERE is_blank), 'images', 'flagged, excluded from counts'),
    (4, 'Frozen frames (same as previous frame)', (SELECT count(*) FROM o WHERE is_frozen), 'images', 'flagged, excluded from counts'),
    (5, 'Usable images', (SELECT count(*) FROM o WHERE usable), 'images', 'used in all averages'),
    (6, 'Night frames (after 18:30)', (SELECT count(*) FROM o WHERE usable AND is_night), 'images', 'kept, scored against the night level of the same camera'),
    (7, 'Camera-hours with fewer than {MIN_VALID} usable images', (SELECT count(*) FROM h WHERE n_valid < {MIN_VALID}), 'camera-hours', 'averages set to empty'),
    (8, 'Camera-hours with trusted averages', (SELECT count(*) FROM h WHERE n_valid >= {MIN_VALID}), 'camera-hours', 'used in analysis'),
    (9, 'Rain gauges reading zero all week', (SELECT count(*) FROM (SELECT rain_id FROM lake.bronze.bma_rain_5min GROUP BY rain_id HAVING sum(rf5min) = 0)), 'gauges', 'treated as dead, not matched to cameras'),
    (10, 'Rain readings that are negative or missing', (SELECT count(*) FROM lake.bronze.bma_rain_5min WHERE rf5min < 0 OR rf5min IS NULL), 'readings', 'dropped'),
    (11, 'Cameras whose nearest gauge is over 2 km away', (SELECT count(*) FROM lake.silver.camera_gauge WHERE gauge_km > 2), 'cameras', 'kept, distance recorded'),
    (12, 'Cameras with no lane count in OpenStreetMap', (SELECT count(*) FROM lake.bronze.osm_context WHERE lanes IS NULL), 'cameras', 'left empty')
) t(step, rule, n, unit, action)
ORDER BY step
"""


def connect() -> duckdb.DuckDBPyConnection:
    LAKE.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL ducklake; LOAD ducklake;")
    con.execute(f"ATTACH 'ducklake:{(LAKE / 'catalog.ducklake').as_posix()}' AS lake "
                f"(DATA_PATH '{(LAKE / 'files').as_posix()}')")
    return con


def run() -> None:
    con = connect()
    for schema in ("bronze", "silver", "gold"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS lake.{schema}")
    for name in ("image_index", "yolo_counts", "weather", "air_quality", "osm_context",
                 "bma_rain_stations", "bma_rain_5min"):
        con.execute(f"CREATE OR REPLACE TABLE lake.bronze.{name} AS "
                    f"SELECT * FROM read_parquet('{(BRONZE / name).as_posix()}.parquet')")
    con.execute(f"CREATE OR REPLACE TABLE lake.bronze.cameras AS "
                f"SELECT * FROM read_csv('{CAMERA_CSV.as_posix()}')")
    for sql in (SILVER, GAUGES, GOLD_HOUR, GOLD_CAMERAS, QUALITY):
        con.execute(sql)
    for name in ("camera_hour", "cameras", "quality_report"):
        con.execute(f"COPY lake.gold.{name} TO '{(EXPORT / name).as_posix()}.parquet' (FORMAT parquet)")
        print(name, con.execute(f"SELECT count(*) FROM lake.gold.{name}").fetchone()[0])


if __name__ == "__main__":
    run()
