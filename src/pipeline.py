import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src import config


def download_data_if_needed():
    config.ensure_directories()
    if not config.TRIP_FILE.exists():
        print(f"Downloading {config.TRIP_URL} ...")
        urllib.request.urlretrieve(config.TRIP_URL, config.TRIP_FILE)
    if not config.ZONE_FILE.exists():
        print(f"Downloading {config.ZONE_URL} ...")
        urllib.request.urlretrieve(config.ZONE_URL, config.ZONE_FILE)


def load_and_clean_trips():
    """
    Reproducible trip-level cleaning pipeline matching Phase 1 audit decisions.
    Returns cleaned trip DataFrame, zone lookup DataFrame, and an audit log dict.
    """
    download_data_if_needed()

    cols_needed = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "congestion_surcharge",
        "Airport_fee",
        "cbd_congestion_fee"
    ]

    df = pd.read_parquet(config.TRIP_FILE, columns=cols_needed)
    zones = pd.read_csv(config.ZONE_FILE)

    audit = {"raw_rows": len(df)}

    # 1. Missing metadata block flag (E-Hail / Flex-Fare rides)
    df["is_ehail"] = df["passenger_count"].isna().astype(np.int8)
    audit["ehail_rows_flagged_kept"] = int(df["is_ehail"].sum())

    # 2. Parse timestamps and derive duration & speed
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df["speed_mph"] = np.where(
        df["duration_min"] > 0, df["trip_distance"] / (df["duration_min"] / 60.0), np.nan
    )

    # 3. Deterministic physical & scope filters
    m_start = pd.Timestamp(config.MONTH_START)
    m_end = pd.Timestamp(config.MONTH_END) + pd.Timedelta(minutes=59, seconds=59)

    valid_mask = (
        df["tpep_pickup_datetime"].between(m_start, m_end)
        & df["tpep_dropoff_datetime"].between(m_start, m_end)
        & (~df["PULocationID"].isin(config.UNKNOWN_ZONES))
        & (~df["DOLocationID"].isin(config.UNKNOWN_ZONES))
        & df["PULocationID"].between(1, 263)
        & df["DOLocationID"].between(1, 263)
        & df["duration_min"].between(config.MIN_DURATION_MIN, config.MAX_DURATION_MIN)
        & df["trip_distance"].between(config.MIN_DISTANCE_MILES, config.MAX_DISTANCE_MILES)
        & df["fare_amount"].between(config.MIN_FARE_USD, config.MAX_FARE_USD)
        & df["speed_mph"].between(config.MIN_SPEED_MPH, config.MAX_SPEED_MPH)
    )

    df_clean = df.loc[valid_mask].copy()
    audit["clean_rows"] = len(df_clean)
    audit["rows_removed"] = audit["raw_rows"] - audit["clean_rows"]
    audit["pct_retained"] = round(100.0 * audit["clean_rows"] / audit["raw_rows"], 2)

    # 4. Leak-free median imputation for passenger_count (0 or NaN) using TRAIN trips ONLY
    train_trip_mask = df_clean["tpep_pickup_datetime"] <= pd.Timestamp(config.TRAIN_END) + pd.Timedelta(minutes=59, seconds=59)
    valid_pass_train = df_clean.loc[train_trip_mask & (df_clean["passenger_count"] > 0), "passenger_count"]
    train_passenger_median = float(valid_pass_train.median())
    audit["train_passenger_median"] = train_passenger_median

    df_clean["passenger_count_imputed"] = df_clean["passenger_count"].replace(0, np.nan).fillna(train_passenger_median)

    # Downcast numeric columns to save RAM
    for c in ["trip_distance", "duration_min", "speed_mph", "fare_amount", "total_amount", "passenger_count_imputed"]:
        df_clean[c] = df_clean[c].astype(np.float32)

    return df_clean, zones, audit


def build_hourly_net_flow_panel(df_trips: pd.DataFrame, zones: pd.DataFrame):
    """
    Aggregates clean trips into a complete (LocationID, hour_ts) panel.
    Missing zone-hours structurally represent 0 pickups / 0 dropoffs.
    """
    df_trips = df_trips.copy()
    df_trips["pu_hour"] = df_trips["tpep_pickup_datetime"].dt.floor("h")
    df_trips["do_hour"] = df_trips["tpep_dropoff_datetime"].dt.floor("h")

    # Outflow (Pickups) aggregated by pickup zone & pickup hour
    pu_agg = (
        df_trips.groupby(["PULocationID", "pu_hour"], as_index=False)
        .agg(
            pickups=("fare_amount", "size"),
            pu_avg_distance=("trip_distance", "mean"),
            pu_avg_duration=("duration_min", "mean"),
            pu_avg_speed=("speed_mph", "mean"),
            pu_avg_fare=("fare_amount", "mean"),
            pu_ehail_share=("is_ehail", "mean"),
        )
        .rename(columns={"PULocationID": "LocationID", "pu_hour": "hour_ts"})
    )

    # Inflow (Dropoffs) aggregated by dropoff zone & dropoff hour
    do_agg = (
        df_trips.groupby(["DOLocationID", "do_hour"], as_index=False)
        .agg(
            dropoffs=("fare_amount", "size"),
            do_avg_distance=("trip_distance", "mean"),
            do_avg_duration=("duration_min", "mean"),
            do_avg_speed=("speed_mph", "mean"),
        )
        .rename(columns={"DOLocationID": "LocationID", "do_hour": "hour_ts"})
    )

    # Determine valid active zones strictly from TRAIN period to prevent selection leakage
    train_end_ts = pd.Timestamp(config.TRAIN_END)
    pu_train_counts = pu_agg.loc[pu_agg["hour_ts"] <= train_end_ts].groupby("LocationID")["pickups"].sum()
    do_train_counts = do_agg.loc[do_agg["hour_ts"] <= train_end_ts].groupby("LocationID")["dropoffs"].sum()
    total_train_events = pu_train_counts.add(do_train_counts, fill_value=0)
    active_zones = sorted(total_train_events[total_train_events >= config.MIN_TRAIN_EVENTS_FOR_ACTIVE_ZONE].index.tolist())

    # Build complete Cartesian grid: active_zones x 744 hours in Jan 2025
    all_hours = pd.date_range(config.MONTH_START, config.MONTH_END, freq="h")
    grid = pd.MultiIndex.from_product([active_zones, all_hours], names=["LocationID", "hour_ts"]).to_frame(index=False)

    panel = (
        grid.merge(pu_agg, on=["LocationID", "hour_ts"], how="left")
        .merge(do_agg, on=["LocationID", "hour_ts"], how="left")
        .merge(zones[["LocationID", "Borough", "Zone", "service_zone"]], on="LocationID", how="left")
    )

    # Structural zero-filling for counts
    panel["pickups"] = panel["pickups"].fillna(0).astype(np.int32)
    panel["dropoffs"] = panel["dropoffs"].fillna(0).astype(np.int32)
    panel["total_activity"] = panel["pickups"] + panel["dropoffs"]
    panel["net_flow"] = (panel["dropoffs"] - panel["pickups"]).astype(np.int32)

    # Sort chronologically within each zone
    panel = panel.sort_values(["LocationID", "hour_ts"]).reset_index(drop=True)
    return panel, active_zones


def add_temporal_and_causal_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      1. Numeric & cyclical calendar/temporal features (known in advance).
      2. Strictly causal autoregressive lag & rolling features (shift >= 1h).
      3. Strictly causal spatial spillover features (borough & citywide lag 1h).
    """
    df = panel.copy()

    # --- 1. Calendar & Temporal Features ---
    df["hour"] = df["hour_ts"].dt.hour
    df["dayofweek"] = df["hour_ts"].dt.dayofweek          # 0=Mon, 6=Sun
    df["day_of_month"] = df["hour_ts"].dt.day
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["is_holiday"] = df["hour_ts"].dt.strftime("%Y-%m-%d").isin(config.JAN_2025_HOLIDAYS).astype(np.int8)
    df["is_rush_hour"] = (
        ((df["is_weekend"] == 0) & (df["hour"].between(7, 9) | df["hour"].between(17, 19)))
    ).astype(np.int8)
    df["is_nightlife"] = (
        ((df["dayofweek"].isin([4, 5, 6])) & (df["hour"].between(0, 4)))
    ).astype(np.int8)
    df["is_post_congestion_pricing"] = (
        df["hour_ts"] >= pd.Timestamp(config.CONGESTION_PRICING_START)
    ).astype(np.int8)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # --- 2. Strictly Causal Lag & Rolling Features (NEVER use current hour t!) ---
    g = df.groupby("LocationID", sort=False)

    # Autoregressive net_flow lags
    for lag in [1, 2, 3, 24, 48, 168]:
        df[f"net_flow_lag_{lag}h"] = g["net_flow"].shift(lag)

    # Lagged inflow & outflow components
    for lag in [1, 24, 168]:
        df[f"pickups_lag_{lag}h"] = g["pickups"].shift(lag)
        df[f"dropoffs_lag_{lag}h"] = g["dropoffs"].shift(lag)

    # Momentum / short-term trend (past only)
    df["net_flow_diff_1h"] = df["net_flow_lag_1h"] - df["net_flow_lag_2h"]
    df["net_flow_diff_24h"] = df["net_flow_lag_1h"] - df["net_flow_lag_24h"]

    # Rolling stats shifted by 1 hour so current hour t is strictly excluded
    df["net_flow_roll_mean_3h"] = g["net_flow"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["net_flow_roll_std_3h"] = g["net_flow"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std()).fillna(0.0)
    df["net_flow_roll_mean_24h"] = g["net_flow"].transform(lambda s: s.shift(1).rolling(24, min_periods=6).mean())
    df["activity_roll_mean_3h"] = g["total_activity"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # Lagged trip-level traffic conditions (available strictly before hour t)
    df["pu_avg_speed_lag_1h"] = g["pu_avg_speed"].shift(1)
    df["pu_avg_duration_lag_1h"] = g["pu_avg_duration"].shift(1)

    # --- 3. Strictly Causal Spatial Spillover Features (City & Borough at t - 1h) ---
    city_hourly = df.groupby("hour_ts", as_index=False)["total_activity"].sum().rename(
        columns={"total_activity": "city_total_activity"}
    )
    city_hourly["city_activity_lag_1h"] = city_hourly["city_total_activity"].shift(1)
    df = df.merge(city_hourly[["hour_ts", "city_activity_lag_1h"]], on="hour_ts", how="left")

    boro_hourly = df.groupby(["Borough", "hour_ts"], as_index=False)["net_flow"].sum().rename(
        columns={"net_flow": "boro_net_flow"}
    )
    boro_hourly = boro_hourly.sort_values(["Borough", "hour_ts"])
    boro_hourly["boro_net_flow_lag_1h"] = boro_hourly.groupby("Borough")["boro_net_flow"].shift(1)
    df = df.merge(boro_hourly[["Borough", "hour_ts", "boro_net_flow_lag_1h"]], on=["Borough", "hour_ts"], how="left")

    # Zone's share of its borough's net flow in the previous hour
    df["other_boro_net_flow_lag_1h"] = df["boro_net_flow_lag_1h"] - df["net_flow_lag_1h"]

    return df


class TrainOnlySpatialFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that learns static zone-level statistics
    and lagged-speed medians STRICTLY on the Train split, then transforms
    Train, Validation, and Test without any data leakage.
    """
    def __init__(self):
        self.zone_stats_ = None
        self.speed_medians_ = None
        self.dur_medians_ = None
        self.global_speed_median_ = None
        self.global_dur_median_ = None
        self.borough_categories_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # Compute static zone historical profiles strictly on Train
        self.zone_stats_ = (
            X.groupby("LocationID", as_index=False)
            .agg(
                zone_train_mean_netflow=("net_flow", "mean"),
                zone_train_std_netflow=("net_flow", "std"),
                zone_train_mean_pickups=("pickups", "mean"),
                zone_train_mean_dropoffs=("dropoffs", "mean"),
                zone_train_avg_dist=("pu_avg_distance", "mean"),
                zone_train_avg_speed=("pu_avg_speed", "mean"),
                zone_train_ehail_share=("pu_ehail_share", "mean"),
            )
        )
        self.zone_stats_["zone_train_inflow_ratio"] = (
            self.zone_stats_["zone_train_mean_dropoffs"]
            / (self.zone_stats_["zone_train_mean_pickups"] + self.zone_stats_["zone_train_mean_dropoffs"] + 1e-6)
        )
        self.zone_stats_["zone_train_std_netflow"] = self.zone_stats_["zone_train_std_netflow"].fillna(0.0)

        # Zone-hour medians for imputing zero-trip hours' lagged speed/duration
        self.speed_medians_ = X.groupby("LocationID")["pu_avg_speed_lag_1h"].median().to_dict()
        self.dur_medians_ = X.groupby("LocationID")["pu_avg_duration_lag_1h"].median().to_dict()
        self.global_speed_median_ = float(X["pu_avg_speed_lag_1h"].median())
        self.global_dur_median_ = float(X["pu_avg_duration_lag_1h"].median())
        self.borough_categories_ = sorted(X["Borough"].dropna().unique().tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = df.merge(self.zone_stats_, on="LocationID", how="left")

        # Airport indicator (static domain geography)
        df["is_airport_zone"] = df["Zone"].str.contains("Airport", case=False, na=False).astype(np.int8)

        # Impute lagged speed/duration using Train-only zone medians (when previous hour had 0 pickups)
        df["pu_avg_speed_lag_1h"] = df["pu_avg_speed_lag_1h"].fillna(
            df["LocationID"].map(self.speed_medians_).fillna(self.global_speed_median_)
        )
        df["pu_avg_duration_lag_1h"] = df["pu_avg_duration_lag_1h"].fillna(
            df["LocationID"].map(self.dur_medians_).fillna(self.global_dur_median_)
        )
        df["zone_train_avg_dist"] = df["zone_train_avg_dist"].fillna(df["zone_train_avg_dist"].median())
        df["zone_train_avg_speed"] = df["zone_train_avg_speed"].fillna(self.global_speed_median_)
        df["zone_train_ehail_share"] = df["zone_train_ehail_share"].fillna(0.0)

        # One-hot encode Borough with fixed Train categories
        for boro in self.borough_categories_:
            col_name = f"boro_{boro.lower().replace(' ', '_')}"
            df[col_name] = (df["Borough"] == boro).astype(np.int8)

        return df


def compare_cyclical_vs_numeric_encoding(train_df: pd.DataFrame):
    """
    Path A Requirement: Compare cyclical (sin/cos) vs. simple numeric encoding
    strictly within the Train set (using TimeSeriesSplit CV with a linear model).
    """
    # 1. Geometric continuity check (23:00 vs 00:00)
    dist_numeric = abs(23 - 0) / 23.0  # normalized to [0, 1] -> distance = 1.0 (worst possible!)
    h23 = np.array([np.sin(2 * np.pi * 23 / 24), np.cos(2 * np.pi * 23 / 24)])
    h0 = np.array([np.sin(2 * np.pi * 0 / 24), np.cos(2 * np.pi * 0 / 24)])
    dist_cyclical = float(np.linalg.norm(h23 - h0)) / 2.0  # normalized by circle diameter 2

    # 2. Predictive diagnostic on Train set only (predicting Midtown Center hourly net_flow from time only)
    # Using a linear model highlights why numeric hour fails to capture wrap-around waves
    sub = train_df[train_df["Zone"] == "Midtown Center"].sort_values("hour_ts").dropna(subset=["net_flow"])
    y = sub["net_flow"].values
    X_num = sub[["hour", "dayofweek"]].values
    X_cyc = sub[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].values

    tscv = TimeSeriesSplit(n_splits=4)
    results = {}
    for name, X_mat in [("Numeric (hour, dow)", X_num), ("Cyclical (sin/cos)", X_cyc)]:
        maes, r2s = [], []
        for tr_idx, va_idx in tscv.split(X_mat):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_mat[tr_idx])
            X_va = scaler.transform(X_mat[va_idx])
            model = Ridge(alpha=1.0).fit(X_tr, y[tr_idx])
            pred = model.predict(X_va)
            maes.append(mean_absolute_error(y[va_idx], pred))
            r2s.append(r2_score(y[va_idx], pred))
        results[name] = {"MAE": np.mean(maes), "R2": np.mean(r2s)}

    # Plot cyclical representation + comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    hours = np.arange(24)
    axes[0].scatter(np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24), c=hours, cmap="twilight", s=80)
    for h in [0, 3, 6, 9, 12, 15, 18, 21, 23]:
        axes[0].annotate(
            f"{h:02d}:00",
            (np.sin(2 * np.pi * h / 24) * 1.15, np.cos(2 * np.pi * h / 24) * 1.15),
            ha="center", va="center", fontsize=9
        )
    axes[0].set_title("Cyclical 24-Hour Clock Encoding (23:00 Adjacent to 00:00)")
    axes[0].set_xlabel("hour_sin")
    axes[0].set_ylabel("hour_cos")
    axes[0].set_xlim(-1.35, 1.35)
    axes[0].set_ylim(-1.35, 1.35)
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    names = list(results.keys())
    maes_plot = [results[n]["MAE"] for n in names]
    bars = axes[1].bar(names, maes_plot, color=["#d95f02", "#1b9e77"], width=0.5)
    axes[1].set_title("Midtown Center Net-Flow: Time-Only Linear CV MAE (Train Only)")
    axes[1].set_ylabel("Mean Absolute Error (taxis/hour)")
    for bar, n in zip(bars, names):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"MAE={results[n]['MAE']:.1f}\nR2={results[n]['R2']:.2f}",
            ha="center", va="bottom", fontsize=9
        )
    axes[1].set_ylim(0, max(maes_plot) * 1.25)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig1_cyclical_vs_numeric.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "normalized_23_to_00_dist_numeric": round(dist_numeric, 3),
        "normalized_23_to_00_dist_cyclical": round(dist_cyclical, 3),
        "cv_results": results,
    }


def export_feature_catalog():
    """Creates a clean, documented catalog of all engineered features."""
    catalog = [
        ("hour_sin, hour_cos", "Cyclical Temporal", "sin/cos(2*pi*hour/24)", "Preserves 23:00 -> 00:00 continuity for linear/distance models; known in advance."),
        ("dow_sin, dow_cos", "Cyclical Temporal", "sin/cos(2*pi*dayofweek/7)", "Preserves Sunday -> Monday continuity; known in advance."),
        ("is_weekend, is_holiday", "Calendar", "Sat/Sun flag; Jan 1 & Jan 20 flags", "Captures non-commute schedule drops; known strictly from calendar."),
        ("is_rush_hour, is_nightlife", "Domain Temporal", "Weekday 7-9/17-19; Fri-Sun 0-4", "Encodes commuter peaks and weekend nightlife waves identified in Phase 1."),
        ("is_post_congestion_pricing", "Policy Regime", "hour_ts >= 2025-01-05", "Captures NYC CBD congestion pricing launch regime shift (publicly pre-announced)."),
        ("net_flow_lag_{1,2,3,24,48,168}h", "Autoregressive Lag", "Zone net_flow at t-k (k >= 1)", "Captures immediate momentum, daily seasonality (24h), and weekly seasonality (168h)."),
        ("pickups/dropoffs_lag_{1,24,168}h", "Component Lag", "Inflow & outflow separately at t-k", "Distinguishes high-volume balanced hubs (+200/-200) from quiet zones (+1/-1)."),
        ("net_flow_diff_1h, _diff_24h", "Trend / Momentum", "lag_1h - lag_2h; lag_1h - lag_24h", "Measures whether surplus/shortage is accelerating or reversing."),
        ("net_flow_roll_mean/std_3h, _24h", "Rolling Temporal", "Rolling stats on shift(1) values", "Smooths noisy 1-hour spikes without leaking current hour t."),
        ("pu_avg_speed/duration_lag_1h", "Traffic Congestion", "Prior hour zone speed & duration", "Congestion slows taxi turnover and delays dropoffs in subsequent hours."),
        ("city_activity_lag_1h", "Spatial Macro", "Citywide total trips at t-1h", "Captures macro NYC demand shocks (e.g., rain/snowstorms or major events)."),
        ("other_boro_net_flow_lag_1h", "Spatial Spillover", "Same-borough net_flow minus own zone at t-1h", "Empty cabs accumulating in neighboring zones spill over into nearby shortage zones."),
        ("zone_train_mean/std_netflow", "Spatial Static (Train)", "Train-only zone net_flow mean & std", "Encodes baseline structural tendency of each zone (fit on Jan 1-21 only)."),
        ("zone_train_inflow_ratio", "Spatial Static (Train)", "Train dropoffs / (pickups + dropoffs)", "Captures chronic sinks (>0.5, e.g., Newark/outer zones) vs sources (<0.5)."),
        ("zone_train_avg_dist, _speed, _ehail_share", "Spatial Static (Train)", "Train-only zone trip traits", "Distinguishes airport/long-haul hubs from short-hop Manhattan corridors."),
        ("is_airport_zone, boro_*", "Geographic Lookup", "From taxi_zone_lookup.csv", "Encodes static geographic boundaries and airport flat-rate/regulatory dynamics."),
        ("zone_cluster_id", "Unsupervised Archetype", "K-Means cluster fit on Train zone profiles", "Groups zones sharing similar 24h weekday/weekend flow waves (fit on Jan 1-21 only)."),
    ]
    df_cat = pd.DataFrame(catalog, columns=["Feature(s)", "Category", "Formula / Source", "Domain Logic & Leakage Prevention"])
    df_cat.to_csv(config.REPORTS_DIR / "feature_catalog.csv", index=False)
    return df_cat


def run_preprocessing_and_fe():
    """Executes the full preparation pipeline."""
    df_trips, zones, audit = load_and_clean_trips()
    panel, active_zones = build_hourly_net_flow_panel(df_trips, zones)
    panel = add_temporal_and_causal_lag_features(panel)

    # Strict chronological split BEFORE fitting any stateful encoders/scalers
    train_raw = panel[panel["hour_ts"] <= pd.Timestamp(config.TRAIN_END)].copy()
    val_raw = panel[panel["hour_ts"].between(pd.Timestamp(config.VAL_START), pd.Timestamp(config.VAL_END))].copy()
    test_raw = panel[panel["hour_ts"] >= pd.Timestamp(config.TEST_START)].copy()

    # Fit spatial encoder strictly on Train
    spatial_enc = TrainOnlySpatialFeatureEncoder()
    train_df = spatial_enc.fit_transform(train_raw)
    val_df = spatial_enc.transform(val_raw)
    test_df = spatial_enc.transform(test_raw)

    # Run cyclical vs numeric diagnostic strictly on Train
    cyc_comparison = compare_cyclical_vs_numeric_encoding(train_df)

    # Note: Keep full train_df for EDA/clustering; drop the 168-hour warm-up NaNs ONLY when saving ML-ready train set
    train_ml = train_df.dropna(subset=["net_flow_lag_168h"]).reset_index(drop=True)
    val_ml = val_df.dropna(subset=["net_flow_lag_168h"]).reset_index(drop=True)
    test_ml = test_df.dropna(subset=["net_flow_lag_168h"]).reset_index(drop=True)

    # Save feature catalog
    feature_catalog = export_feature_catalog()

    return {
        "df_trips": df_trips,
        "zones": zones,
        "audit": audit,
        "panel_full": pd.concat([train_df, val_df, test_df], ignore_index=True),
        "train_full": train_df,
        "train_ml": train_ml,
        "val_ml": val_ml,
        "test_ml": test_ml,
        "active_zones": active_zones,
        "cyc_comparison": cyc_comparison,
        "feature_catalog": feature_catalog,
    }