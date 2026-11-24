import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src import config


def detect_zone_hour_anomalies(train_df: pd.DataFrame):
    """
    Detects anomalous (LocationID, hour_ts) observations within the Train period (Jan 1-21)
    using two complementary methods:
      1. Robust Z-score (Median/MAD) of net_flow relative to each zone's typical (weekday/weekend, hour) baseline.
      2. Multivariate Isolation Forest on (net_flow, pickups, dropoffs, avg_speed, avg_duration).
    """
    df = train_df.copy()

    # 1. Context-aware Robust Z-Score (per zone, weekday/weekend, hour)
    grp_cols = ["LocationID", "is_weekend", "hour"]
    stats = df.groupby(grp_cols)["net_flow"].agg(
        ctx_median="median",
        ctx_mad=lambda s: np.median(np.abs(s - np.median(s)))
    ).reset_index()

    df = df.merge(stats, on=grp_cols, how="left")
    # 1.4826 scales MAD to match std dev of a normal distribution
    df["robust_z_netflow"] = (df["net_flow"] - df["ctx_median"]) / (1.4826 * df["ctx_mad"] + 1.0)
    df["abs_robust_z"] = df["robust_z_netflow"].abs()

    # 2. Multivariate Isolation Forest on high-activity zone-hours
    iso_features = ["net_flow", "pickups", "dropoffs", "pu_avg_speed_lag_1h", "pu_avg_duration_lag_1h"]
    X_iso = StandardScaler().fit_transform(df[iso_features].fillna(0).values)
    iso = IsolationForest(n_estimators=150, contamination=0.005, random_state=config.RANDOM_SEED, n_jobs=-1)
    df["iso_flag"] = (iso.fit_predict(X_iso) == -1).astype(np.int8)
    df["iso_score"] = -iso.decision_function(X_iso)  # Higher = more anomalous

    # Combine: high-confidence anomaly candidates flagged by BOTH high robust Z and Isolation Forest
    candidates = (
        df[(df["abs_robust_z"] >= 5.0) & (df["iso_flag"] == 1)]
        .sort_values("abs_robust_z", ascending=False)
        .copy()
    )

    # Heuristic categorization to aid manual qualitative inspection
    def classify_candidate(row):
        date_str = row["hour_ts"].strftime("%Y-%m-%d")
        if date_str == "2025-01-01":
            return "Real Event Candidate (New Year's Day)"
        if date_str == "2025-01-20":
            return "Real Event Candidate (MLK Jr. Day Holiday)"
        if "Airport" in str(row["Zone"]):
            return "Real Event Candidate (Airport Flight Wave / Weather)"
        if row["total_activity"] == 0 and row["ctx_median"] != 0:
            return "Potential Telemetry / Data Drop"
        return "Local Event / Congestion Shock (Inspect)"

    candidates["preliminary_category"] = candidates.apply(classify_candidate, axis=1)

    top_export = candidates[
        ["hour_ts", "LocationID", "Zone", "Borough", "pickups", "dropoffs", "net_flow", "ctx_median", "robust_z_netflow", "iso_score", "preliminary_category"]
    ].head(50)
    top_export.to_csv(config.REPORTS_DIR / "anomaly_candidates.csv", index=False)

    # Plot citywide hourly activity & highlight top anomalous days/hours
    city_ts = df.groupby("hour_ts", as_index=False).agg(
        total_pickups=("pickups", "sum"),
        num_zone_anomalies=("iso_flag", "sum")
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    axes[0].plot(city_ts["hour_ts"], city_ts["total_pickups"], color="#2b8cbe", linewidth=1.1, label="Hourly NYC Pickups")
    # Annotate key real-world dates
    axes[0].axvline(pd.Timestamp("2025-01-01 01:00:00"), color="red", linestyle="--", alpha=0.7, label="Jan 1 (New Year's Day)")
    axes[0].axvline(pd.Timestamp("2025-01-05 00:00:00"), color="purple", linestyle="--", alpha=0.7, label="Jan 5 (Congestion Pricing Start)")
    axes[0].axvline(pd.Timestamp("2025-01-20 12:00:00"), color="orange", linestyle="--", alpha=0.7, label="Jan 20 (MLK Jr. Day)")
    axes[0].set_title("Citywide Hourly Pickups Across Train Period (Jan 1-21, 2025)")
    axes[0].set_ylabel("Total Pickups / Hour")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].bar(city_ts["hour_ts"], city_ts["num_zone_anomalies"], width=0.04, color="#d95f02", alpha=0.8)
    axes[1].set_title("Number of Anomalous Zones Flagged per Hour (Isolation Forest)")
    axes[1].set_ylabel("Flagged Zones Count")
    axes[1].set_xlabel("Date (Train Period)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig5_anomaly_candidates_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()

    return top_export