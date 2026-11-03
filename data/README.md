# Data Documentation (2025 Yellow Taxi Trip Data)

This document explains the official data sources, access date, and automated reconstruction steps.

## 1. Official Data Source
- **Source:** New York City Taxi and Limousine Commission (NYC TLC) / TLC Trip Record Data - January 2025.
- **Official Portal:** [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Date Retrieved:** October 2026

## 2. Required Files and Links
* Link of the used subset page: https://data.cityofnewyork.us/Transportation/2025-Yellow-Taxi-Trip-Data/irx2-vgm4/about_data
* Download Link of the subset(2025 Yellow Taxi Trip Data): https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
* Download Link of the zone map lookup table of NYC: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

## 3. Reconstructing the Analytical Dataset
You do not need to download the files manually. Running the second cell of the notebook will automatically Create the `data/` directory if it does not exist and download files directly.
