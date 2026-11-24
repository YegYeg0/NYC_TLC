import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

from src import config

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False


def build_zone_profiles(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a rich behavioral profile for each active NYC zone using TRAIN data ONLY.
    Combines:
      - Normalized 24-hour weekday net-flow ratio wave: (dropoffs - pickups) / (dropoffs + pickups + 5)
      - Normalized 24-hour weekend net-flow ratio wave
      - Structural summary features (log volume, avg distance, avg duration, avg speed, ehail share)
    Why normalized net-flow ratio? So clustering discovers functional commuting archetypes
    (Office vs Residential vs Airport vs Nightlife) rather than merely separating huge vs small zones.
    """
    df = train_df.copy()
    # Smoothed normalized net-flow ratio in [-1, +1]
    df["flow_ratio"] = df["net_flow"] / (df["total_activity"] + 5.0)

    # 24h weekday wave
    wkday = (
        df[df["is_weekend"] == 0]
        .pivot_table(index="LocationID", columns="hour", values="flow_ratio", aggfunc="mean")
        .fillna(0.0)
    )
    wkday.columns = [f"wkday_flow_h{h:02d}" for h in wkday.columns]

    # 24h weekend wave
    wkend = (
        df[df["is_weekend"] == 1]
        .pivot_table(index="LocationID", columns="hour", values="flow_ratio", aggfunc="mean")
        .fillna(0.0)
    )
    wkend.columns = [f"wkend_flow_h{h:02d}" for h in wkend.columns]

    # Structural summary features per zone (Train only)
    summary = df.groupby("LocationID").agg(
        Zone=("Zone", "first"),
        Borough=("Borough", "first"),
        log_mean_activity=("total_activity", lambda s: np.log1p(s.mean())),
        mean_abs_netflow=("net_flow", lambda s: np.abs(s).mean()),
        overall_inflow_ratio=("zone_train_inflow_ratio", "first"),
        avg_dist=("zone_train_avg_dist", "first"),
        avg_speed=("zone_train_avg_speed", "first"),
        ehail_share=("zone_train_ehail_share", "first"),
    )

    profiles = summary.join(wkday).join(wkend).reset_index()
    return profiles


def run_pattern_discovery(train_df: pd.DataFrame):
    """
    Executes PCA, UMAP/t-SNE, K-Means vs Hierarchical parameter selection,
    3-way stability checks (seed, subsample, scaler), and generates all plots.
    """
    profiles = build_zone_profiles(train_df)
    meta_cols = ["LocationID", "Zone", "Borough"]
    feature_cols = [c for c in profiles.columns if c not in meta_cols]

    X_raw = profiles[feature_cols].values
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X_raw)

    # ---------------------------------------------------------
    # 1. Dimensionality Reduction: PCA
    # ---------------------------------------------------------
    pca_full = PCA(random_state=config.RANDOM_SEED)
    X_pca_full = pca_full.fit_transform(X_std)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp_80 = int(np.argmax(cum_var >= 0.80) + 1)

    # Keep top PCs explaining ~80% variance as denoised clustering representation if desired,
    # or cluster directly on X_std. Here we compare and use X_pca (10 components) for noise reduction.
    n_pca_cluster = min(10, X_std.shape[1])
    X_cluster = X_pca_full[:, :n_pca_cluster]

    # ---------------------------------------------------------
    # 2. Parameter Selection (k = 2 .. 10) for K-Means & Hierarchical
    # ---------------------------------------------------------
    k_range = list(range(2, 11))
    metrics_rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=30, random_state=config.RANDOM_SEED).fit(X_cluster)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_cluster)
        metrics_rows.append({
            "k": k,
            "km_inertia": km.inertia_,
            "km_silhouette": silhouette_score(X_cluster, km.labels_),
            "km_davies_bouldin": davies_bouldin_score(X_cluster, km.labels_),
            "hc_silhouette": silhouette_score(X_cluster, hc.labels_),
            "km_vs_hc_ari": adjusted_rand_score(km.labels_, hc.labels_),
        })
    metrics_df = pd.DataFrame(metrics_rows)

    # Choose optimal k balancing Silhouette (high) & Davies-Bouldin (low) for k >= 4 (interpretable archetypes)
    # You can inspect fig2_cluster_selection.png and adjust k if needed!
    cand_df = metrics_df[metrics_df["k"].between(4, 6)]
    best_k = int(cand_df.loc[cand_df["km_silhouette"].idxmax(), "k"])

    # Fit final K-Means & Hierarchical with best_k
    final_km = KMeans(n_clusters=best_k, n_init=50, random_state=config.RANDOM_SEED).fit(X_cluster)
    final_hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X_cluster)
    profiles["cluster_id"] = final_km.labels_
    profiles["hc_cluster_id"] = final_hc.labels_

    # ---------------------------------------------------------
    # 3. Pattern Stability Tests (Seed, Subsample, Scaling)
    # ---------------------------------------------------------
    # (a) Seed Stability (20 random seeds)
    seed_labels = [
        KMeans(n_clusters=best_k, n_init=10, random_state=s).fit_predict(X_cluster)
        for s in range(20)
    ]
    seed_aris = [adjusted_rand_score(final_km.labels_, lbl) for lbl in seed_labels]
    seed_nmis = [normalized_mutual_info_score(final_km.labels_, lbl) for lbl in seed_labels]

    # (b) Subsampling Stability (20 random 80% zone subsamples)
    rng = np.random.default_rng(config.RANDOM_SEED)
    sub_aris, sub_nmis = [], []
    n_zones = len(X_cluster)
    for _ in range(20):
        idx = rng.choice(n_zones, size=int(0.8 * n_zones), replace=False)
        sub_lbl = KMeans(n_clusters=best_k, n_init=20, random_state=config.RANDOM_SEED).fit_predict(X_cluster[idx])
        sub_aris.append(adjusted_rand_score(final_km.labels_[idx], sub_lbl))
        sub_nmis.append(normalized_mutual_info_score(final_km.labels_[idx], sub_lbl))

    # (c) Scaling Stability (StandardScaler vs RobustScaler)
    X_rob = RobustScaler().fit_transform(X_raw)
    X_rob_pca = PCA(n_components=n_pca_cluster, random_state=config.RANDOM_SEED).fit_transform(X_rob)
    rob_labels = KMeans(n_clusters=best_k, n_init=50, random_state=config.RANDOM_SEED).fit_predict(X_rob_pca)
    scaler_ari = adjusted_rand_score(final_km.labels_, rob_labels)
    scaler_nmi = normalized_mutual_info_score(final_km.labels_, rob_labels)

    # (d) Temporal Stability (Week 1-2 vs Week 2-3 within Train)
    w12_prof = build_zone_profiles(train_df[train_df["day_of_month"] <= 14])
    w23_prof = build_zone_profiles(train_df[train_df["day_of_month"] >= 8])
    common_ids = sorted(set(w12_prof["LocationID"]).intersection(w23_prof["LocationID"]))
    X_w12 = PCA(n_components=n_pca_cluster, random_state=config.RANDOM_SEED).fit_transform(
        StandardScaler().fit_transform(w12_prof.set_index("LocationID").loc[common_ids, feature_cols].values)
    )
    X_w23 = PCA(n_components=n_pca_cluster, random_state=config.RANDOM_SEED).fit_transform(
        StandardScaler().fit_transform(w23_prof.set_index("LocationID").loc[common_ids, feature_cols].values)
    )
    lbl_w12 = KMeans(n_clusters=best_k, n_init=30, random_state=config.RANDOM_SEED).fit_predict(X_w12)
    lbl_w23 = KMeans(n_clusters=best_k, n_init=30, random_state=config.RANDOM_SEED).fit_predict(X_w23)
    temporal_ari = adjusted_rand_score(lbl_w12, lbl_w23)
    temporal_nmi = normalized_mutual_info_score(lbl_w12, lbl_w23)

    stability_summary = pd.DataFrame([
        {"Stability Test": "Random Seed (20 seeds)", "Mean ARI": np.mean(seed_aris), "Std ARI": np.std(seed_aris), "Mean NMI": np.mean(seed_nmis)},
        {"Stability Test": "80% Zone Subsampling (20 runs)", "Mean ARI": np.mean(sub_aris), "Std ARI": np.std(sub_aris), "Mean NMI": np.mean(sub_nmis)},
        {"Stability Test": "Scaler (Standard vs Robust)", "Mean ARI": scaler_ari, "Std ARI": 0.0, "Mean NMI": scaler_nmi},
        {"Stability Test": "Temporal Window (Wk 1-2 vs Wk 2-3)", "Mean ARI": temporal_ari, "Std ARI": 0.0, "Mean NMI": temporal_nmi},
        {"Stability Test": "Algorithm (K-Means vs Hierarchical)", "Mean ARI": adjusted_rand_score(final_km.labels_, final_hc.labels_), "Std ARI": 0.0, "Mean NMI": normalized_mutual_info_score(final_km.labels_, final_hc.labels_)},
    ])
    stability_summary.to_csv(config.REPORTS_DIR / "clustering_stability.csv", index=False)

    # ---------------------------------------------------------
    # 4. Nonlinear 2D Embedding (UMAP or t-SNE) for Visualization
    # ---------------------------------------------------------
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=12, min_dist=0.15, random_state=config.RANDOM_SEED)
        X_2d_nl = reducer.fit_transform(X_cluster)
        nl_name = "UMAP"
    else:
        reducer = TSNE(n_components=2, perplexity=min(20, len(X_cluster) - 1), random_state=config.RANDOM_SEED)
        X_2d_nl = reducer.fit_transform(X_cluster)
        nl_name = "t-SNE"

    profiles["PC1"] = X_pca_full[:, 0]
    profiles["PC2"] = X_pca_full[:, 1]
    profiles["NL1"] = X_2d_nl[:, 0]
    profiles["NL2"] = X_2d_nl[:, 1]

    # ---------------------------------------------------------
    # 5. Generate Publication-Quality Figures
    # ---------------------------------------------------------
    # Figure 2: Parameter Selection (Scree + Elbow + Silhouette)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(range(1, 21), cum_var[:20] * 100, marker="o", color="#2b8cbe")
    axes[0].axhline(80, color="red", linestyle="--", alpha=0.7, label=f"80% Var ({n_comp_80} PCs)")
    axes[0].set_title("PCA Cumulative Explained Variance")
    axes[0].set_xlabel("Number of Principal Components")
    axes[0].set_ylabel("Cumulative Variance (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(metrics_df["k"], metrics_df["km_inertia"], marker="o", color="#1b9e77")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Selected k={best_k}")
    axes[1].set_title("K-Means Elbow Curve (Inertia)")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Within-Cluster SSE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(metrics_df["k"], metrics_df["km_silhouette"], marker="o", label="K-Means Silhouette")
    axes[2].plot(metrics_df["k"], metrics_df["hc_silhouette"], marker="s", linestyle="--", label="Hierarchical (Ward)")
    axes[2].axvline(best_k, color="red", linestyle="--", alpha=0.7)
    axes[2].set_title("Silhouette Score vs. Number of Clusters")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Silhouette Score")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig2_pca_and_cluster_selection.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3: PCA vs UMAP/t-SNE Projection Colored by Cluster
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = sns.color_palette("Set2", n_colors=best_k)

    sns.scatterplot(
        data=profiles, x="PC1", y="PC2", hue="cluster_id", size="log_mean_activity",
        sizes=(30, 220), palette=palette, alpha=0.85, ax=axes[0]
    )
    axes[0].set_title(f"PCA 2D Projection (PC1={pca_full.explained_variance_ratio_[0]*100:.1f}%, PC2={pca_full.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].grid(alpha=0.3)

    sns.scatterplot(
        data=profiles, x="NL1", y="NL2", hue="cluster_id", size="log_mean_activity",
        sizes=(30, 220), palette=palette, alpha=0.85, ax=axes[1]
    )
    axes[1].set_title(f"{nl_name} 2D Projection of Active NYC Taxi Zones (k={best_k})")
    axes[1].set_xlabel(f"{nl_name} Dimension 1")
    axes[1].set_ylabel(f"{nl_name} Dimension 2")
    axes[1].grid(alpha=0.3)

    # Annotate key landmark zones on PCA plot
    landmarks = ["Midtown Center", "Upper East Side North", "JFK Airport", "LaGuardia Airport", "East Village", "Financial District North"]
    for _, row in profiles[profiles["Zone"].isin(landmarks)].iterrows():
        axes[0].annotate(row["Zone"], (row["PC1"], row["PC2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig3_zone_embeddings_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4: Average 24-Hour Weekday vs Weekend Net-Flow Wave per Cluster (Interpretation!)
    wkday_cols = [f"wkday_flow_h{h:02d}" for h in range(24)]
    wkend_cols = [f"wkend_flow_h{h:02d}" for h in range(24)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    hours = np.arange(24)
    for cid in range(best_k):
        sub = profiles[profiles["cluster_id"] == cid]
        n_c = len(sub)
        axes[0].plot(hours, sub[wkday_cols].mean().values, marker="o", markersize=4, label=f"Cluster {cid} (n={n_c})", color=palette[cid])
        axes[1].plot(hours, sub[wkend_cols].mean().values, marker="o", markersize=4, label=f"Cluster {cid} (n={n_c})", color=palette[cid])

    for ax, title in zip(axes, ["Weekday 24h Normalized Net-Flow Wave", "Weekend 24h Normalized Net-Flow Wave"]):
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Hour of Day (0-23)")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Normalized Net Flow: (Dropoffs - Pickups) / (Total + 5)")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig4_cluster_daily_waves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Summarize cluster profiles for report table
    cluster_summary = []
    for cid in range(best_k):
        sub = profiles[profiles["cluster_id"] == cid]
        top_zones = sub.sort_values("log_mean_activity", ascending=False)["Zone"].head(4).tolist()
        cluster_summary.append({
            "cluster_id": cid,
            "num_zones": len(sub),
            "avg_hourly_activity": round(float(np.expm1(sub["log_mean_activity"]).mean()), 1),
            "avg_inflow_ratio": round(float(sub["overall_inflow_ratio"].mean()), 3),
            "avg_trip_dist_mi": round(float(sub["avg_dist"].mean()), 2),
            "wkday_morning_8am_flow": round(float(sub["wkday_flow_h08"].mean()), 3),
            "wkday_evening_6pm_flow": round(float(sub["wkday_flow_h18"].mean()), 3),
            "top_representative_zones": ", ".join(top_zones),
        })
    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df.to_csv(config.REPORTS_DIR / "cluster_profiles.csv", index=False)

    zone_cluster_map = profiles.set_index("LocationID")["cluster_id"].to_dict()
    return {
        "profiles": profiles,
        "metrics_df": metrics_df,
        "best_k": best_k,
        "stability_summary": stability_summary,
        "cluster_summary": cluster_summary_df,
        "zone_cluster_map": zone_cluster_map,
    }