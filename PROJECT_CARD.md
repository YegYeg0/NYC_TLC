# PROJECT CARD

## 1. Selected Path and Dataset Subset
**Selected Path:** Path A (`NYC TLC Trip Record Data`).
**Selected Subset:** NYC Yellow Taxi trip records for **January 2025** (`yellow_tripdata_2025-01.parquet`, 3,475,226 raw records; 3,202,048 clean records retained after Phase 2 deterministic filtering) paired with the official spatial lookup table (`taxi_zone_lookup.csv`, 265 zones).

## 2. Temporary Title
**Predicting Hourly Taxi Surplus and Shortage Across NYC Zones Using Net-Flow ($\text{Dropoffs} - \text{Pickups}$)**

## 3. Real-World Problem & Decision Supported by the Analysis
Every finished taxi ride takes a cab away from the pickup zone (`PULocationID`) and leaves an empty, available cab in the dropoff zone (`DOLocationID`).
- When **Dropoffs are much higher than Pickups** in a zone during an hour, too many taxis are empty (**Taxi Surplus**). Drivers waste time and fuel driving around without passengers.
- When **Pickups are much higher than Dropoffs**, the zone runs out of cabs (**Taxi Shortage**), and passengers face long wait times.
- **Decision Supported:** By predicting the hourly **Net Flow ($\text{Dropoffs} - \text{Pickups}$)** for each zone, operators can identify surplus zones in advance and reposition empty taxis to nearby zones facing an upcoming shortage.

## 4. Unit of Analysis and Unit of Prediction
- **Unit of Observation (Raw Data):** A single completed taxi trip record.
- **Unit of Analysis (EDA & Pattern Discovery):** (1) Trip-level records (for physical/quality auditing), (2) Hourly zone panel `(LocationID, hour_ts)`, and (3) Zone-level 24-hour behavioral profiles (for clustering).
- **Unit of Prediction:** A `(LocationID, hour_ts)` pair—one active NYC Taxi Zone during a 1-hour time window.

## 5. Target Domain / Population
NYC Yellow Taxi trips across **220 active New York City taxi zones** (selected strictly on Train data with $\ge 200$ total Train events, primarily covering Manhattan, Queens, Brooklyn, and JFK/LaGuardia airports) during January 2025.

## 6. Main Analytical / Predictive Question
*"Can we use past trip patterns, cyclical time features, traffic conditions, and spatial zone archetypes to predict the hourly **Net Flow ($\text{Dropoffs} - \text{Pickups}$)** in each active taxi zone, and how does prediction error vary across zone archetypes, holidays, and major local demand shocks?"*

## 7. Target Variable / Output
- **Main Target Variable (`net_flow`):** For each active zone $z$ and hour $t$:
  $$\text{net\_flow}_{z,t} = \text{Dropoffs}_{z,t} - \text{Pickups}_{z,t}$$

## 8. Initial Baseline
- **24-Hour Naive Baseline:** Predict that a zone's net flow at hour $t$ equals its net flow at the same hour yesterday ($\hat{y}_{z,t} = y_{z,t-24}$).
- **168-Hour (Weekly) Naive Baseline:** Predict that a zone's net flow at hour $t$ equals its net flow at the same hour and day of the week 7 days earlier ($\hat{y}_{z,t} = y_{z,t-168}$).

## 9. Proposed Evaluation Metrics
- **MAE (Mean Absolute Error):** Primary operational metric showing average error in taxis per hour.
- **RMSE (Root Mean Squared Error):** Penalizes large errors during peak rush hours and major events.
- **$R^2$ Score:** Measures the share of net-flow variance explained relative to a constant mean baseline.
- **Subgroup Evaluation:** Breakdown of MAE/RMSE across the 6 discovered zone archetypes (`zone_cluster_id`) and peak vs. off-peak hours.

## 10. Train, Validation, and Test Split Design
Because this is time-series data, **random splitting is prohibited**. We enforce a strict chronological split before fitting any stateful transformers or clusters:
- **Train Set:** Jan 1 – Jan 21, 2025 (~68% of the month; full period used for EDA, zone selection, clustering, and stateful fitting; Jan 8 – Jan 21 saved as `train_ml.parquet` with 73,920 rows after the 168-hour lag warm-up window).
- **Validation Set:** Jan 22 – Jan 26, 2025 (~16%; `val_ml.parquet`, 26,400 rows).
- **Test Set:** Jan 27 – Jan 31, 2025 (~16%; `test_ml.parquet`, 26,400 rows — strictly locked until Phase 3 final evaluation).

## 11. Major Data Leakage Risks
1. **Future Time Leakage (Look-Ahead Bias):** Using future hours to predict past hours, or fitting scalers, medians, active-zone filters, or clusters on the full month instead of strictly on Train (Jan 1–21).
2. **Same-Hour Dropoff Leakage:** Since $\text{net\_flow} = \text{Dropoffs} - \text{Pickups}$ in hour $t$, no features may use trips or speeds from hour $t$; all lag, rolling, and spatial spillover features are strictly shifted by $\ge 1$ hour.
3. **Target-Encoding / Spatial-Profile Leakage:** All static zone statistics (`TrainOnlySpatialFeatureEncoder`) and cluster assignments (`zone_cluster_id`) are fit exclusively on Train and mapped to Validation/Test.

## 12. Unsupervised / Pattern Discovery Component (Completed in Phase 2)
- **Representation & Dimensionality Reduction:** Each of the 220 active zones is represented by a 54-dimensional Train-only behavioral vector (24h weekday normalized net-flow wave, 24h weekend wave, and 6 structural trip traits). PCA denoises the space (5 PCs explain $>80\%$ variance; top 10 PCs retained for clustering), and t-SNE provides 2D visualization.
- **Clustering & Validation:** K-Means ($k=6$, selected via Silhouette peak $\approx 0.414$ and Elbow curve, compared against Ward Hierarchical clustering) uncovers six interpretable urban archetypes: *Peripheral Dropoff Sinks*, *Quiet Outer Periphery*, *Dense Uptown Residential*, *Office/Commuter Hubs*, *Transit/Nightlife/Mixed*, and *Major Airports*.
- **Stability:** Validated across 5 tests (20 random seeds, 80% subsampling, Standard vs. Robust scalers, temporal windows, and K-Means vs. Ward) with ARI ranging from $0.768$ to $0.961$. The Train-only `zone_cluster_id` is added as a feature for Phase 3 modeling and subgroup error analysis.

## 13. Path-Specific Advanced Component (Path A: Time-Series, Event & Anomaly Analysis)
- **Forecasting vs. Seasonal Baselines:** Chronological multi-horizon forecasting benchmarked against 24h and 168h naive baselines, evaluating linear (Ridge) vs. nonlinear ensemble models.
- **Zone-Hour Anomaly Detection (Data Error vs. Real Event):** Dual detection using context-aware Robust Z-score ($|\text{robust\_z}| \ge 5.0$) and multivariate Isolation Forest to separate trip-level telemetry errors from real-world demand shocks (e.g., New Year's Day East Village exodus, Jan 13–14 Javits Center/NRF trade show outflow in Hudson Yards, and MLK Jr. Day airport waves).
- **Spatiotemporal & Subgroup Shift Analysis:** Evaluating model generalization and error patterns across the 6 spatial zone archetypes and across calendar/event regimes without making unsupported causal claims.

## 14. Ethical, Privacy, and Potential Bias Risks
- **Geographic & Socioeconomic Bias:** Yellow taxis concentrate heavily in Manhattan and at JFK/LaGuardia airports, whereas outer-borough residents rely more on Green Taxis and FHVs (Uber/Lyft). Repositioning or policy decisions based solely on Yellow Taxi data could underserve outer-borough neighborhoods.
- **Unrecorded Unmet Demand:** Taximeters only log completed rides; passengers who gave up waiting during severe shortages are unobserved.
- **Privacy & Governance:** The dataset is publicly released by NYC TLC at an anonymized trip/zone level; no attempt is made to re-identify drivers or passengers, and all findings are framed as predictive associations rather than causal claims.