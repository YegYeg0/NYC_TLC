# PROJECT CARD

## 1. Selected Path and Dataset Subset
**Selected Path:** Path A (`NYC TLC Trip Record Data`).
**Selected Subset:** NYC Yellow Taxi trip records for **January 2025** (`yellow_tripdata_2025-01.parquet`, ~3.48 million records) paired with the official spatial lookup table (`taxi_zone_lookup.csv`, 265 zones).

## 2. Temporary Title
**Predicting Hourly Taxi Surplus and Shortage Across NYC Zones Using Net-Flow ($\text{Dropoffs} - \text{Pickups}$)**

## 3. Real-World Problem & Decision Supported by the Analysis
Every finished taxi ride takes a cab away from the pickup zone (`PULocationID`) and leaves an empty, available cab in the dropoff zone (`DOLocationID`).
* When **Dropoffs are much higher than Pickups** in a zone during an hour, too many taxis are empty (**Taxi Surplus**). Drivers waste time and fuel driving around without passengers.
* When **Pickups are much higher than Dropoffs**, the zone runs out of cabs (**Taxi Shortage**), and passengers have to wait a long time.
* **Decision Supported:** By predicting the **Net Flow ($\text{Dropoffs} - \text{Pickups}$)** for each zone every hour, operators can see which zones will have extra empty taxis and move them to zones that are about to run out of taxis.

## 4. Unit of Analysis and Unit of Prediction
* **Unit of Observation (Raw Data):** A single taxi trip record.
* **Unit of Analysis (EDA):** We analyze the data at two levels: (1) Single trips (to check distance, duration, speed, and fare errors), and (2) Grouped by **Zone and Hour**.
* **Unit of Prediction:** A `(LocationID, Hour)` pair—meaning one specific NYC Taxi Zone during a 1-hour time window.

## 5. Target Domain / Population
NYC Yellow Taxi trips across active New York City taxi zones (mainly Manhattan, Queens, Brooklyn, and JFK/LaGuardia airports) during January 2025.

## 6. Main Analytical / Predictive Question
*"Can we use past trip patterns, time of day, day of week, and zone characteristics to predict the hourly **Net Flow ($\text{Dropoffs} - \text{Pickups}$)** in each taxi zone, and how much does prediction error change during holidays/bad weather or across different types of neighborhoods?"*

## 7. Target Variable / Output
* **Main Target Variable (`net_flow`):** For each zone $z$ and hour $t$:
  $$\text{net\_flow}_{z,t} = \text{Dropoffs}_{z,t} - \text{Pickups}_{z,t}$$

## 8. Initial Baseline
* **24-Hour Baseline:** Predict that a zone's net flow at hour $t$ will be the exact same as its net flow at the same hour yesterday ($t - 24$).
* **7-Day (Weekly) Naive Baseline:** Predict that a zone's net flow at hour $t$ will be the same as the exact same hour and day of the week last week ($t - 168$).

* ## 9. Proposed Evaluation Metrics
* **MAE (Mean Absolute Error):** Easy to understand because it shows the average error in number of taxis per hour.
* **RMSE (Root Mean Squared Error):** Penalizes large mistakes during busy rush hours.
* **$R^2$ Score:** Shows how much of the variation in net flow our model explains compared to a simple average.

* ## 10. Initial Train, Validation, and Test Split Design
Because this is time-series data, **we cannot use a random split** (that would leak future data into the past). Instead, we split the data strictly by date:
* **Train Set:** Jan 1 to Jan 21, 2024 (First 3 weeks — ~68% of the month)
* **Validation Set:** Jan 22 to Jan 26, 2024 (Next 5 days — ~16% of the month)
* **Test Set:** Jan 27 to Jan 31, 2024 (Last 5 days — ~16% of the month, kept locked until final testing in Phase 3)

## 11. Major Data Leakage Risks
1. **Time Leakage (Look-Ahead Bias):** Using future hours to predict past hours, or calculating averages/scalers on the whole month before splitting into train and test sets.
2. **Current-Hour Dropoff Leakage:** Since $\text{net\_flow} = \text{Dropoffs} - \text{Pickups}$ in hour $t$, we cannot use trips that finish inside hour $t$ as input features to predict hour $t$. We can only use trips that finished *before* hour $t$ started.

## 12. Proposed Unsupervised / Pattern Discovery Component

## 13. Path-Specific Advanced Component

## 14. Ethical, Privacy, and Potential Bias Risks
* **Geographic Bias:** Yellow taxis mostly operate in Manhattan and at the airports. People living in outer boroughs (like the Bronx, Staten Island, and parts of Queens/Brooklyn) mostly use Green Taxis or Uber/Lyft. If city planners only look at Yellow Taxi data, they might ignore outer-borough neighborhoods.
