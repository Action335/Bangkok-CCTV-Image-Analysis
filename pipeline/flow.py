"""Runs the whole pipeline end to end with Prefect:  uv run python -m pipeline.flow

The two slow steps (unzip + YOLO, ~1 h on a laptop GPU) are skipped when their
output already exists; pass --force to redo them.
"""
import sys

import pyarrow.parquet as pq
from prefect import flow, task

from pipeline import analyze, detect, extract, scrape_bma_rain, scrape_osm, scrape_weather, transform
from pipeline.config import BRONZE


@task
def extract_images(force: bool):
    index = BRONZE / "image_index.parquet"
    if force or not index.exists() or "change_from_prev" not in pq.read_schema(index).names:
        extract.run()


@task
def detect_vehicles(force: bool):
    if force or not (BRONZE / "yolo_counts.parquet").exists():
        detect.run()


@task(retries=2, retry_delay_seconds=30)
def scrape_weather_and_air():
    scrape_weather.run()


@task(retries=2, retry_delay_seconds=60)
def scrape_osm_context(force: bool):
    if force or not (BRONZE / "osm_context.parquet").exists():
        scrape_osm.run()


@task(retries=1, retry_delay_seconds=60)
def scrape_rain_gauges(force: bool):
    if force or not (BRONZE / "bma_rain_5min.parquet").exists():
        scrape_bma_rain.run()


@task
def build_lakehouse():
    transform.run()


@task
def run_analysis():
    analyze.run()


@flow(name="bkk-cctv-pipeline")
def pipeline(force: bool = False):
    extract_images(force)
    detect_vehicles(force)
    scrape_weather_and_air()
    scrape_osm_context(force)
    scrape_rain_gauges(force)
    build_lakehouse()
    run_analysis()


if __name__ == "__main__":
    pipeline(force="--force" in sys.argv)
