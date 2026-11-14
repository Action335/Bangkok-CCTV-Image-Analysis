"""Bangkok CCTV traffic dashboard:  uv run streamlit run dashboard/app.py"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

EXPORT = Path(__file__).resolve().parents[1] / "export"
RAIN_ORDER = ["dry", "light rain", "heavy rain"]
st.set_page_config("Bangkok CCTV Traffic", layout="wide")


@st.cache_data
def load(version: float):  # version = newest file time in export/, so new data replaces the cached copy
    hourly = pd.read_parquet(EXPORT / "camera_hour.parquet")
    cams = pd.read_parquet(EXPORT / "cameras.parquet").merge(
        pd.read_parquet(EXPORT / "camera_clusters.parquet"), on="camera_id", how="left")
    hourly["date"] = hourly.hour.dt.date
    return (hourly, cams, json.loads((EXPORT / "rain_effect.json").read_text()),
            pd.read_parquet(EXPORT / "quality_report.parquet"))


hourly, cams, effect, quality = load(max(f.stat().st_mtime for f in EXPORT.iterdir()))

st.title("Bangkok CCTV Traffic: where, when, and what rain does")
st.caption("100 BMA cameras, 25–31 Aug 2026, one image every 10 min (06:00–21:00). Vehicles counted with "
           "YOLO11m. Congestion = weighted vehicle count scaled to each camera's own normal (z-score).")

with st.sidebar:
    st.header("Filters")
    dates = st.multiselect("Days", sorted(hourly.date.unique()), default=sorted(hourly.date.unique()),
                           format_func=lambda d: d.strftime("%a %d %b"))
    h0, h1 = st.slider("Hour of day", 6, 21, (6, 21))
    rain = st.multiselect("Rain level", RAIN_ORDER, default=RAIN_ORDER)

sel = hourly[hourly.date.isin(dates) & hourly.hour_of_day.between(h0, h1) & hourly.rain_level.isin(rain)]
if sel.empty:
    st.warning("No data for these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Camera-hours", f"{len(sel):,}")
c2.metric("Avg vehicles in view (PCU)", f"{sel.pcu.mean():.1f}")
c3.metric("Rainy share of hours", f"{(sel.rain_level != 'dry').mean():.0%}")
c4.metric("Avg PM2.5 (µg/m³)", f"{sel.pm2_5.mean():.1f}")

tab_insight, tab_map, tab_time, tab_rain, tab_cam, tab_quality = st.tabs(
    ["Insights", "Map", "Time patterns", "Rain & camera groups", "One camera", "Data quality"])

with tab_insight:
    st.caption("Findings for the whole week. They do not change with the filters.")
    light = hourly[hourly.hour_of_day.between(7, 17)]

    st.subheader("1. Rain takes motorcycles and pedestrians off the road; cars stay")
    strict = pd.DataFrame(effect["strict"]).T.reset_index(names="road user")
    strict["change in rain"] = strict.pct.map("{:+.0f}%".format)
    strict["clear effect?"] = strict.p.map(lambda p: "yes" if p < 0.05 else "no")
    left, right = st.columns([3, 2])
    left.plotly_chart(px.bar(strict, x="road user", y="pct", error_y=100 * (strict.ci_high - strict.effect) / strict.dry_mean,
                             error_y_minus=100 * (strict.effect - strict.ci_low) / strict.dry_mean,
                             labels={"pct": "% change in count per image when raining"}), width="stretch")
    right.dataframe(strict[["road user", "dry_mean", "effect", "change in rain", "p", "clear effect?"]], hide_index=True)
    right.caption(f"Rainy vs dry hours on the same day, daylight only (07–17), {effect['strict_rainy_hours']:,} rainy "
                  f"camera-hours out of {effect['strict_n_obs']:,}. Rain = nearest BMA gauge ≥ 0.5 mm in the hour.")
    checks = effect["checks"]
    st.markdown("**Is it behaviour, or YOLO seeing less in the rain?** Two checks point to behaviour: the drop is "
                "already there in light rain, which barely hurts visibility, and it lasts into the dry hour after "
                "rain stops, when the camera sees clearly again.")
    chk = pd.DataFrame(checks["by_road_user"]).T.reset_index(names="road user")
    for col in ("light_rain", "heavy_rain", "hour_after"):
        chk[col] = chk[f"{col}_pct"].map("{:+.0f}%".format) + chk[f"{col}_p"].map(lambda p: "" if p < 0.05 else " (not clear)")
    st.dataframe(chk[["road user", "light_rain", "heavy_rain", "hour_after"]].rename(columns={
        "light_rain": f"light rain, 0.5–2.5 mm/h ({checks['n_light']} h)",
        "heavy_rain": f"heavy rain, 2.5+ mm/h ({checks['n_heavy']} h)",
        "hour_after": f"dry hour after rain ({checks['n_hour_after']} h)"}), hide_index=True)

    st.subheader("2. Weekends are quieter, Sunday most of all")
    by_day = light.groupby("weekday").pcu.mean().reindex(
        ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"])
    wk, we = light[~light.is_weekend].pcu.mean(), light[light.is_weekend].pcu.mean()
    left, right = st.columns([3, 2])
    left.plotly_chart(px.bar(by_day.reset_index(), x="weekday", y="pcu",
                             labels={"pcu": "vehicles in view (PCU), daylight"}), width="stretch")
    right.metric("Weekday vs weekend (daylight)", f"{wk:.1f} vs {we:.1f}", f"{100 * (we - wk) / wk:.0f}% on weekends")
    right.caption("One weekend only, so treat this as suggestive.")

    st.subheader("3. Road type does not change how full the road looks, but secondary roads have far more motorcycles")
    roads = cams.groupby("road_class", as_index=False).agg(
        cameras=("camera_id", "size"), vehicles_in_view=("mean_pcu", "mean"),
        motorcycle_share=("mean_motorcycle_share", "mean")).sort_values("vehicles_in_view", ascending=False)
    left, right = st.columns(2)
    left.dataframe(roads.round(2), hide_index=True)
    right.dataframe(cams.nlargest(5, "mean_pcu")[["name", "road_class", "mean_pcu"]].round(1)
                    .rename(columns={"name": "busiest cameras", "mean_pcu": "vehicles in view"}), hide_index=True)

    st.caption("Primary, secondary and tertiary roads all show about 9 vehicles in view. Motorcycle share is the "
               "difference: about 4% on primary roads against 14% on secondary roads.")

    st.subheader("Limits")
    st.markdown("""
- One week of images (25–31 Aug 2026) with one weekend; almost all rain fell on Sat 29 – Mon 31 Aug.
- A photo counts vehicles in view, not speed, so a jam and free-flowing traffic can look alike.
- YOLO11m misses distant vehicles, more so at night and in heavy rain, so part of the heavy-rain drop may be missed detections.
- Truck counts are the least reliable: pickups and some fixed objects are counted as trucks, and trucks weigh 2.5 in the vehicles-in-view score.
- Road type is the nearest street-level road in OpenStreetMap; expressways passing overhead are ignored.
- Air quality and non-rain weather come from coarse model grids and are not used for conclusions.
""")

with tab_map:
    per_cam = sel.groupby("camera_id", as_index=False).agg(congestion=("congestion_z", "mean"), pcu=("pcu", "mean"))
    m = cams.merge(per_cam, on="camera_id")
    m["red"] = (255 * ((m.congestion.clip(-1, 1) + 1) / 2)).astype(int)
    m["green"] = 255 - m.red
    m["label"] = m.name + "<br/>congestion " + m.congestion.round(2).astype(str) + " · PCU " + m.pcu.round(1).astype(str)
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=m.latitude.mean(), longitude=m.longitude.mean(), zoom=10.5),
        layers=[pdk.Layer("ScatterplotLayer", m, get_position="[longitude, latitude]", get_radius=220,
                          get_fill_color="[red, green, 60, 200]", pickable=True)],
        tooltip={"html": "{label}"}))
    st.caption("Red = busier than that camera's normal for the selected filters, green = quieter.")

with tab_time:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = sel.pivot_table(index="weekday", columns="hour_of_day", values="congestion_z").reindex(order).dropna(how="all")
    st.plotly_chart(px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto", zmin=-1, zmax=1,
                              title="Congestion by weekday and hour"), width="stretch")
    mix = sel.groupby("hour_of_day")[["n_car", "n_motorcycle", "n_heavy", "n_person"]].mean().reset_index()
    st.plotly_chart(px.line(mix, x="hour_of_day", y=["n_car", "n_motorcycle", "n_heavy", "n_person"],
                            title="Average count per image by hour"), width="stretch")

with tab_rain:
    left, right = st.columns(2)
    by_rain = sel.groupby(["hour_of_day", "rain_level"], as_index=False).congestion_z.mean()
    left.plotly_chart(px.line(by_rain, x="hour_of_day", y="congestion_z", color="rain_level",
                              category_orders={"rain_level": RAIN_ORDER}, title="Congestion by hour: dry vs rain"),
                      width="stretch")
    eff = pd.DataFrame(effect["effects"]).T.reset_index(names="rain level")
    left.markdown("**Rain effect on congestion (regression, whole week, all hours)** — change in z-score vs dry hours, "
                  "controlling for hour of day × weekend; 95% CI clustered by camera.")
    left.dataframe(eff, hide_index=True)
    prof = hourly.merge(cams[["camera_id", "cluster"]], on="camera_id").dropna(subset=["cluster"])
    prof["cluster"] = "group " + prof.cluster.astype(int).astype(str)
    right.plotly_chart(px.line(prof.groupby(["cluster", "hour_of_day"], as_index=False).congestion_z.mean(),
                               x="hour_of_day", y="congestion_z", color="cluster",
                               title="Camera groups (K-means on daily shape)"), width="stretch")
    right.dataframe(prof.drop_duplicates("camera_id").cluster.value_counts().rename("cameras"))

with tab_cam:
    name = st.selectbox("Camera", cams.sort_values("name").name)
    cam = cams[cams.name == name].iloc[0]
    one = sel[sel.camera_id == cam.camera_id].sort_values("hour")
    st.write(f"Road: **{cam.road_name or '—'}** ({cam.road_class or 'unknown'}) · "
             f"stations within 500 m: {cam.n_stations_500m} · schools: {cam.n_schools_500m} · malls: {cam.n_malls_500m}")
    st.plotly_chart(px.line(one, x="hour", y="pcu", title="Vehicles in view (PCU)"), width="stretch")
    st.plotly_chart(px.bar(one, x="hour", y="precipitation", title="Rain (mm/h)"), width="stretch")

with tab_quality:
    st.markdown("**Cleaning rules and how many rows each one touches**")
    st.dataframe(quality.drop(columns="step"), hide_index=True)
    bad = cams.assign(unusable=cams.blank_share + cams.frozen_share).nlargest(20, "unusable")
    st.plotly_chart(px.bar(bad, x="name", y=["blank_share", "frozen_share"],
                           title="Share of blank and frozen frames (worst 20 cameras)"), width="stretch")
    cover = sel.groupby("hour", as_index=False).agg(images=("n_images", "sum"), valid=("n_valid", "sum"))
    st.plotly_chart(px.line(cover, x="hour", y=["images", "valid"], title="Images collected vs usable per hour"),
                    width="stretch")
