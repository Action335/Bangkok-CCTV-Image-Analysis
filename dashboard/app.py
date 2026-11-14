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
c4.metric("Avg PM2.5 (µg/m³, model grid)", f"{sel.pm2_5.mean():.1f}")

tab_insight, tab_map, tab_time, tab_rain, tab_cam, tab_quality = st.tabs(
    ["Insights", "Map", "Time patterns", "Rain & camera groups", "One camera", "Data quality"])

with tab_insight:
    st.caption("Findings for the whole week. They do not change with the filters.")
    light = hourly[hourly.hour_of_day.between(7, 17)]

    st.subheader("1. Rain takes motorcycles off the road; cars stay")
    strict = pd.DataFrame(effect["strict"]).T.reset_index(names="road user")
    strict["change in rain"] = strict.pct.map("{:+.0f}%".format)
    strict["clear in YOLO counts?"] = strict.p.map(lambda p: "yes" if p < 0.05 else "no")
    hand = pd.read_csv(EXPORT / "hand_count_images.csv").groupby("stratum")[["motorcycle", "person", "bus", "car"]].mean()
    hand.columns = ["Motorcycles", "People", "Buses", "Cars"]
    strict["hand count, dry → rain"] = strict["road user"].map(
        lambda u: f"{hand.loc['dry day', u]:.1f} → {hand.loc['rain day', u]:.1f}")
    left, right = st.columns([3, 2])
    left.plotly_chart(px.bar(strict, x="road user", y="pct", error_y=100 * (strict.ci_high - strict.effect) / strict.dry_mean,
                             error_y_minus=100 * (strict.effect - strict.ci_low) / strict.dry_mean,
                             labels={"pct": "% change in count per image when raining"}), width="stretch")
    right.dataframe(strict[["road user", "change in rain", "p", "clear in YOLO counts?", "hand count, dry → rain"]],
                    hide_index=True)
    right.caption(f"Rainy vs dry hours on the same day, daylight only (07–17), {effect['strict_rainy_hours']:,} rainy "
                  f"camera-hours out of {effect['strict_n_obs']:,}. Rain = nearest BMA gauge ≥ 0.5 mm in the hour. "
                  "Last column: our own count per image, 40 dry and 24 rainy daylight images.")
    checks = effect["checks"]
    st.markdown("**Is it behaviour, or YOLO seeing less in the rain?** Two checks point to behaviour: the drop is "
                "already there in light rain, which barely hurts visibility, and it lasts into the dry hour after "
                "rain stops, when the camera sees clearly again. These checks alone are not enough: people pass them "
                "too, and the hand count below does not confirm a drop in people.")
    chk = pd.DataFrame(checks["by_road_user"]).T.reset_index(names="road user")
    for col in ("light_rain", "heavy_rain", "hour_after"):
        chk[col] = chk[f"{col}_pct"].map("{:+.0f}%".format) + chk[f"{col}_p"].map(lambda p: "" if p < 0.05 else " (not clear)")
    st.dataframe(chk[["road user", "light_rain", "heavy_rain", "hour_after"]].rename(columns={
        "light_rain": f"light rain, 0.5–2.5 mm/h ({checks['n_light']} h)",
        "heavy_rain": f"heavy rain, 2.5+ mm/h ({checks['n_heavy']} h)",
        "hour_after": f"dry hour after rain ({checks['n_hour_after']} h)"}), hide_index=True)

    st.markdown("**What the hand count adds.** YOLO finds only about 1 motorcycle in 9, and fewer still in rain, so "
                "the size of the motorcycle drop is overstated. The direction holds in our own count: 4.7 motorcycles "
                "per image in dry daylight (40 images) against 2.8 in rain (24 images). The same count shows no drop "
                "in people (0.53 against 0.50), so the pedestrian result is not confirmed. Only the motorcycle "
                "drop is supported by both YOLO and the hand count.")

    st.subheader("2. Weekends are quieter, Sunday most of all")
    by_day = light.groupby("weekday").pcu.mean().reindex(
        ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"])
    wk, we = light[~light.is_weekend].pcu.mean(), light[light.is_weekend].pcu.mean()
    left, right = st.columns([3, 2])
    left.plotly_chart(px.bar(by_day.reset_index(), x="weekday", y="pcu",
                             labels={"pcu": "vehicles in view (PCU), daylight"}), width="stretch")
    right.metric("Weekday vs weekend (daylight)", f"{wk:.1f} vs {we:.1f}", f"{100 * (we - wk) / wk:.0f}% on weekends")
    right.caption("One weekend only, so treat this as suggestive.")

    st.subheader("3. Road type does not change how full the road looks")
    roads = cams.groupby("road_class", as_index=False).agg(
        cameras=("camera_id", "size"), vehicles_in_view=("mean_pcu", "mean"),
        motorcycle_share_yolo=("mean_motorcycle_share", "mean")).sort_values("vehicles_in_view", ascending=False)
    left, right = st.columns(2)
    left.dataframe(roads.round(2), hide_index=True)
    left.caption("Primary, secondary and tertiary roads all show about 7.5 vehicles in view.")
    right.dataframe(pd.read_csv(EXPORT / "hand_count_by_road.csv"), hide_index=True)
    right.caption("YOLO reports a motorcycle share of 4% on primary roads against 14% on secondary roads. That gap is "
                  "not real: in our hand count the share is 26% against 28%. YOLO finds 1 motorcycle in 20 on the wide "
                  "primary roads and 1 in 5 on secondary roads, so we do not claim a difference.")

    st.subheader("Limits")
    st.markdown("""
- One week of images (25–31 Aug 2026) with one weekend; almost all rain fell on Sat 29 – Mon 31 Aug.
- The source images have gaps: Thursday 27 Aug has almost no usable images before 09:00 (219 images in three hours against about 600 per hour normally, most of them blank or stuck frames), and 21:00 has one frame per camera. Those hours are left empty.
- A photo counts vehicles in view, not speed, so a jam and free-flowing traffic can look alike.
- YOLO11m undercounts. Against our hand count of 98 images it found about 60% of the cars and 11% of the motorcycles: distant, blurry vehicles are missed, more so at night and in rain.
- The vehicles-in-view score is therefore too low in level (about 60% of the hand count by day, 45% at night), but it ranks images well (correlation 0.85), so it is used for comparisons only.
- YOLO labels pickups and vans as trucks (116 reported, 16 real), so its "truck" class is counted as a car.
- Road type is the nearest street-level road in OpenStreetMap; expressways passing overhead are ignored.
- How many motorcycles YOLO finds depends on the road (5% on primary, 19% on secondary), so YOLO motorcycle counts are compared within a camera, never between cameras.
- Air quality and non-rain weather come from coarse model grids and are not used for conclusions.
""")
    st.markdown("**Hand count against YOLO** (98 images: dry day, rain, night, and images where YOLO saw many trucks)")
    left, right = st.columns(2)
    left.dataframe(pd.read_csv(EXPORT / "hand_count_by_class.csv"), hide_index=True)
    right.dataframe(pd.read_csv(EXPORT / "hand_count_by_stratum.csv"), hide_index=True)

with tab_map:
    per_cam = sel.groupby("camera_id", as_index=False).agg(congestion=("congestion_z", "mean"), pcu=("pcu", "mean"))
    m = cams.merge(per_cam, on="camera_id").dropna(subset=["congestion"])  # cameras with no usable hour here
    m["radius"] = 90 + 30 * m.pcu
    m["red"] = (255 * ((m.congestion.clip(-1, 1) + 1) / 2)).astype(int)
    m["green"] = 255 - m.red
    m["label"] = m.name + "<br/>congestion " + m.congestion.round(2).astype(str) + " · PCU " + m.pcu.round(1).astype(str)
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=m.latitude.mean(), longitude=m.longitude.mean(), zoom=10.5),
        layers=[pdk.Layer("ScatterplotLayer", m, get_position="[longitude, latitude]", get_radius="radius",
                          get_fill_color="[red, green, 60, 200]", pickable=True)],
        tooltip={"html": "{label}"}))
    st.caption("Dot size = vehicles in view (it also depends on how much road the camera sees). Colour = red when "
               "busier than that camera's own weekly normal, green when quieter. With every filter left on, the "
               "colour is neutral by definition: narrow the days, hours or rain level to see where it changes.")

with tab_time:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = sel.pivot_table(index="weekday", columns="hour_of_day", values="congestion_z").reindex(order).dropna(how="all")
    st.plotly_chart(px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto", zmin=-1, zmax=1,
                              title="Congestion by weekday and hour"), width="stretch")
    st.caption("Empty cells are gaps in the source images, not quiet roads. Thursday 27 Aug has almost no images "
               "before 09:00 (the camera feed was down), and 21:00 has a single frame per camera. Hours with fewer "
               "than 3 usable images per camera are left empty.")
    mix = sel.groupby("hour_of_day")[["n_car", "n_motorcycle", "n_bus", "n_person"]].mean().reset_index()
    st.plotly_chart(px.line(mix, x="hour_of_day", y=["n_car", "n_motorcycle", "n_bus", "n_person"],
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
    right.caption("Groups use the whole week and do not change with the filters.")

with tab_cam:
    names = cams.set_index("camera_id").name
    cam_id = st.selectbox("Camera", cams.sort_values(["name", "camera_id"]).camera_id,
                          format_func=lambda i: f"{names[i]} (camera {i})")
    cam = cams[cams.camera_id == cam_id].iloc[0]
    one = sel[sel.camera_id == cam.camera_id].sort_values("hour")
    st.write(f"Road: **{cam.road_name or '—'}** ({cam.road_class or 'unknown'}) · "
             f"stations within 500 m: {cam.n_stations_500m} · schools: {cam.n_schools_500m} · malls: {cam.n_malls_500m}")
    st.plotly_chart(px.line(one, x="hour", y="pcu", title="Vehicles in view (PCU)"), width="stretch")
    st.plotly_chart(px.bar(one, x="hour", y="precipitation", title="Rain (mm/h)"), width="stretch")

with tab_quality:
    st.markdown("**Cleaning rules and how many rows each one touches**")
    st.dataframe(quality.drop(columns="step"), hide_index=True)
    bad = cams.assign(unusable=cams.blank_share + cams.frozen_share,
                      camera=cams.name + " (" + cams.camera_id.astype(str) + ")").nlargest(20, "unusable")
    st.plotly_chart(px.bar(bad, x="camera", y=["blank_share", "frozen_share"],
                           title="Share of blank and frozen frames (worst 20 cameras)"), width="stretch")
    cover = sel.groupby("hour", as_index=False).agg(images=("n_images", "sum"), valid=("n_valid", "sum"))
    st.plotly_chart(px.line(cover, x="hour", y=["images", "valid"], title="Images collected vs usable per hour"),
                    width="stretch")
