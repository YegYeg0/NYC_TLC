import json
from src import config
from src.pipeline import run_preprocessing_and_fe
from src.pattern_discovery import run_pattern_discovery
from src.anomaly_detection import detect_zone_hour_anomalies


def main():
    config.ensure_directories()
    print("=== [1/4] Running Reproducible Preprocessing & Feature Engineering ===")
    prep = run_preprocessing_and_fe()
    print("Cleaning Audit:", json.dumps(prep["audit"], indent=2))
    print("Cyclical vs Numeric Check:", json.dumps(prep["cyc_comparison"], indent=2))
    print(f"Active Zones (selected on Train): {len(prep['active_zones'])}")

    print("\n=== [2/4] Running Dimensionality Reduction, Clustering & Stability Tests ===")
    pd_res = run_pattern_discovery(prep["train_full"])
    print(f"Selected Optimal Clusters: k = {pd_res['best_k']}")
    print("\nCluster Profiles:\n", pd_res["cluster_summary"].to_string(index=False))
    print("\nStability Summary:\n", pd_res["stability_summary"].to_string(index=False))

    print("\n=== [3/4] Adding Train-Only Cluster Feature & Saving Phase 3 Datasets ===")
    cmap = pd_res["zone_cluster_map"]
    for split_name in ["train_ml", "val_ml", "test_ml"]:
        df_s = prep[split_name].copy()
        df_s["zone_cluster_id"] = df_s["LocationID"].map(cmap).fillna(-1).astype("int8")
        out_path = config.PROCESSED_DATA_DIR / f"{split_name}.parquet"
        df_s.to_parquet(out_path, index=False)
        print(f"Saved {out_path} | shape = {df_s.shape}")

    print("\n=== [4/4] Detecting Zone-Hour Anomaly Candidates ===")
    top_anoms = detect_zone_hour_anomalies(prep["train_full"])
    print("Top 10 Zone-Hour Anomaly Candidates:")
    print(top_anoms.head(10)[["hour_ts", "Zone", "net_flow", "ctx_median", "robust_z_netflow", "preliminary_category"]].to_string(index=False))

    print("\nALL PHASE 2 ARTIFACTS GENERATED SUCCESSFULLY!")
    print(f"Figures saved in: {config.FIGURES_DIR}")
    print(f"Tables saved in:  {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()