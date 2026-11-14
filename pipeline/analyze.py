"""Step 5: data science on the gold tables.

1. K-means groups cameras by the shape of their average day (hourly congestion profile).
2. OLS regressions measure how much rain shifts what the cameras see, with
   standard errors clustered by camera:
   - overall: congestion score by rain level, all hours;
   - strict:  each road-user type, comparing rainy and dry hours on the same day,
              daylight only (07-17), where YOLO is most reliable.
Results are written to export/ for the dashboard and slides.
"""
import json

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pipeline.config import EXPORT

ROAD_USERS = {"n_motorcycle": "Motorcycles", "n_person": "People",
              "n_heavy": "Buses and trucks", "n_car": "Cars"}


def cluster_cameras(hourly: pd.DataFrame) -> pd.DataFrame:
    day = hourly[hourly.hour_of_day.between(6, 20)]
    profile = day.pivot_table(index="camera_id", columns="hour_of_day", values="congestion_z").dropna()
    best = max(
        (KMeans(n_clusters=k, n_init=20, random_state=0).fit(profile) for k in range(2, 7)),
        key=lambda m: silhouette_score(profile, m.labels_),
    )
    print("k =", best.n_clusters, "silhouette =", round(silhouette_score(profile, best.labels_), 3))
    return pd.DataFrame({"camera_id": profile.index, "cluster": best.labels_})


def _fit(formula: str, df: pd.DataFrame):
    return smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": df.camera_id})


def _term(fit, term: str) -> dict:
    lo, hi = fit.conf_int().loc[term]
    return {"effect": round(fit.params[term], 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "p": float(f"{fit.pvalues[term]:.3g}")}


def rain_effect(hourly: pd.DataFrame) -> dict:
    df = hourly.dropna(subset=["congestion_z", "precipitation"]).copy()
    df["rain_level"] = pd.Categorical(df.rain_level, ["dry", "light rain", "heavy rain"])
    overall = _fit("congestion_z ~ rain_level + C(hour_of_day) * is_weekend", df)
    out = {"n_obs": int(overall.nobs), "effects": {}, "strict": {}}
    for level in ("light rain", "heavy rain"):
        out["effects"][level] = {**_term(overall, f"rain_level[T.{level}]"),
                                 "n_hours": int((df.rain_level == level).sum())}

    # Strict version. Day effects: Sunday and the first day share the baseline label,
    # because the weekend terms already separate Sat+Sun from weekdays.
    light = df[df.hour_of_day.between(7, 17)].copy()
    light["rain"] = (light.precipitation >= 0.5).astype(int)
    day = light.hour.dt.date
    light["day"] = day.astype(str).where((light.weekday != "Sunday") & (day != day.min()), "baseline")
    for col, label in ROAD_USERS.items():
        fit = _fit(f"{col} ~ rain + C(hour_of_day) * is_weekend + C(camera_id) + C(day)", light)
        dry_mean = light.loc[light.rain == 0, col].mean()
        out["strict"][label] = {**_term(fit, "rain"), "dry_mean": round(dry_mean, 2),
                                "pct": round(100 * fit.params["rain"] / dry_mean, 1)}
    out["strict_n_obs"] = len(light)
    out["strict_rainy_hours"] = int(light.rain.sum())

    # Two checks that the drop is behaviour and not YOLO seeing less in rain:
    # (a) light rain alone, which barely hurts visibility;
    # (b) the dry hour right after rain stops, when visibility is back.
    light["level"] = pd.cut(light.precipitation, [-1, 0.5, 2.5, 1e6], right=False, labels=["dry", "light", "heavy"])
    light = light.sort_values(["camera_id", "hour"])
    by_cam = light.groupby("camera_id")
    light["after_rain"] = ((light.rain == 0) & (by_cam.precipitation.shift(1) >= 0.5)
                           & (light.hour - by_cam.hour.shift(1) == pd.Timedelta(hours=1))).astype(int)
    controls = "C(hour_of_day) * is_weekend + C(camera_id) + C(day)"
    out["checks"] = {"n_light": int((light.level == "light").sum()), "n_heavy": int((light.level == "heavy").sum()),
                     "n_hour_after": int(light.after_rain.sum()), "by_road_user": {}}
    for col, label in ROAD_USERS.items():
        by_level = _fit(f"{col} ~ level + {controls}", light)
        lagged = _fit(f"{col} ~ rain + after_rain + {controls}", light)
        dry = light.loc[light.level == "dry", col].mean()
        fully_dry = light.loc[(light.rain == 0) & (light.after_rain == 0), col].mean()
        out["checks"]["by_road_user"][label] = {
            "light_rain_pct": round(100 * by_level.params["level[T.light]"] / dry, 1),
            "light_rain_p": float(f"{by_level.pvalues['level[T.light]']:.3g}"),
            "heavy_rain_pct": round(100 * by_level.params["level[T.heavy]"] / dry, 1),
            "heavy_rain_p": float(f"{by_level.pvalues['level[T.heavy]']:.3g}"),
            "hour_after_pct": round(100 * lagged.params["after_rain"] / fully_dry, 1),
            "hour_after_p": float(f"{lagged.pvalues['after_rain']:.3g}"),
        }
    return out


def run() -> None:
    hourly = pd.read_parquet(EXPORT / "camera_hour.parquet")
    cluster_cameras(hourly).to_parquet(EXPORT / "camera_clusters.parquet", index=False)
    effect = rain_effect(hourly)
    (EXPORT / "rain_effect.json").write_text(json.dumps(effect, indent=2))
    print(json.dumps(effect, indent=2))


if __name__ == "__main__":
    run()
