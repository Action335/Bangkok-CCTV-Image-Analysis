"""Step 4: load everything into a DuckLake lakehouse (bronze -> silver -> gold)
and export the small gold tables for the dashboard.

bronze  raw tables exactly as collected
silver  one row per image: quality flags, weighted vehicle count, per-camera congestion score
gold    camera x hour table joined with weather and air quality; one row per camera
"""
import duckdb

from pipeline.config import BRONZE, CAMERA_CSV, DATA, EXPORT

LAKE = DATA / "lake"

SILVER = """
CREATE OR REPLACE TABLE lake.silver.observations AS
WITH obs AS (
    SELECT i.camera_id, i.ts, date_trunc('hour', i.ts) AS hour, i.path,
           i.brightness, i.contrast, i.sharpness, i.glare,
           i.contrast < 10 AS is_blank,            -- grey / "no signal" frames
           hour(i.ts) >= 19 AS is_night,           -- YOLO is less reliable after dark
           y.n_person, y.n_bicycle, y.n_car, y.n_motorcycle, y.n_bus, y.n_truck, y.mean_conf,
           -- passenger-car units: motorcycle 0.33, car 1, bus/truck 2.5
           0.33 * y.n_motorcycle + y.n_car + 2.5 * (y.n_bus + y.n_truck) AS pcu
    FROM lake.bronze.image_index i
    JOIN lake.bronze.yolo_counts y USING (path)
)
SELECT *,
       -- congestion relative to this camera's own normal level (camera views differ)
       CASE WHEN NOT is_blank THEN
           (pcu - avg(pcu) FILTER (NOT is_blank) OVER cam)
           / nullif(stddev(pcu) FILTER (NOT is_blank) OVER cam, 0) END AS congestion_z
FROM obs
WINDOW cam AS (PARTITION BY camera_id)
"""

GAUGES = """
-- hourly rain per BMA gauge, and each camera's nearest gauge that reported that week
CREATE OR REPLACE TABLE lake.silver.gauge_hour AS
SELECT rain_id, date_trunc('hour', ts) AS hour, sum(rf5min) AS rain_mm, count(rf5min) AS n_readings
FROM lake.bronze.bma_rain_5min GROUP BY ALL;

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

GOLD_HOUR = """
CREATE OR REPLACE TABLE lake.gold.camera_hour AS
WITH agg AS (
    SELECT camera_id, hour,
           count(*) AS n_images,
           count(*) FILTER (NOT is_blank) AS n_valid,
           avg(pcu) FILTER (NOT is_blank) AS pcu,
           avg(congestion_z) AS congestion_z,
           avg(n_car) FILTER (NOT is_blank) AS n_car,
           avg(n_motorcycle) FILTER (NOT is_blank) AS n_motorcycle,
           avg(n_bus + n_truck) FILTER (NOT is_blank) AS n_heavy,
           avg(n_person) FILTER (NOT is_blank) AS n_person,
           avg(brightness) AS brightness, avg(sharpness) AS sharpness, avg(glare) AS glare
    FROM lake.silver.observations
    GROUP BY ALL
)
SELECT a.*,
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
       s.n_images, s.blank_share, s.mean_pcu, s.mean_motorcycle_share
FROM lake.bronze.cameras c
LEFT JOIN lake.bronze.osm_context o USING (camera_id)
LEFT JOIN lake.silver.camera_gauge cg USING (camera_id)
LEFT JOIN (
    SELECT camera_id, count(*) AS n_images, avg(is_blank::INT) AS blank_share,
           avg(pcu) FILTER (NOT is_blank) AS mean_pcu,
           sum(n_motorcycle) / nullif(sum(n_motorcycle + n_car + n_bus + n_truck), 0) AS mean_motorcycle_share
    FROM lake.silver.observations GROUP BY camera_id
) s USING (camera_id)
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
    for sql in (SILVER, GAUGES, GOLD_HOUR, GOLD_CAMERAS):
        con.execute(sql)
    for name in ("camera_hour", "cameras"):
        con.execute(f"COPY lake.gold.{name} TO '{(EXPORT / name).as_posix()}.parquet' (FORMAT parquet)")
        print(name, con.execute(f"SELECT count(*) FROM lake.gold.{name}").fetchone()[0])


if __name__ == "__main__":
    run()
