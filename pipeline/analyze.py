"""Step 5: data science on the gold tables.

1. K-means groups cameras by the shape of their average day (hourly congestion profile).
2. OLS regression measures how much rain shifts congestion, controlling for
   hour of day x weekend, with standard errors clustered by camera.
Results are written to export/ for the dashboard and slides.
"""
import json

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pipeline.config import EXPORT


def cluster_cameras(hourly: pd.DataFrame) -> pd.DataFrame:
    day = hourly[hourly.hour_of_day.between(6, 20)]
    profile = day.pivot_table(index="camera_id", columns="hour_of_day", values="congestion_z").dropna()
    best = max(
        (KMeans(n_clusters=k, n_init=20, random_state=0).fit(profile) for k in range(2, 7)),
        key=lambda m: silhouette_score(profile, m.labels_),
    )
    print("k =", best.n_clusters, "silhouette =", round(silhouette_score(profile, best.labels_), 3))
    return pd.DataFrame({"camera_id": profile.index, "cluster": best.labels_})


def rain_effect(hourly: pd.DataFrame) -> dict:
    df = hourly.dropna(subset=["congestion_z", "precipitation"]).copy()
    df["rain_level"] = pd.Categorical(df.rain_level, ["dry", "light rain", "heavy rain"])
    fit = smf.ols("congestion_z ~ rain_level + C(hour_of_day) * is_weekend", df).fit(
        cov_type="cluster", cov_kwds={"groups": df.camera_id})
    out = {"n_obs": int(fit.nobs), "r2": round(fit.rsquared, 3), "effects": {}}
    for level in ("light rain", "heavy rain"):
        term = f"rain_level[T.{level}]"
        lo, hi = fit.conf_int().loc[term]
        out["effects"][level] = {"coef": round(fit.params[term], 3), "ci_low": round(lo, 3),
                                 "ci_high": round(hi, 3), "p": float(f"{fit.pvalues[term]:.3g}"),
                                 "n_hours": int((df.rain_level == level).sum())}
    return out


def run() -> None:
    hourly = pd.read_parquet(EXPORT / "camera_hour.parquet")
    cluster_cameras(hourly).to_parquet(EXPORT / "camera_clusters.parquet", index=False)
    effect = rain_effect(hourly)
    (EXPORT / "rain_effect.json").write_text(json.dumps(effect, indent=2))
    print(json.dumps(effect, indent=2))


if __name__ == "__main__":
    run()
