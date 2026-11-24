from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "figures" / "phase2"
REPORTS_DIR = ROOT_DIR / "reports" / "phase2"

# Raw data files & official download URLs
TRIP_FILE = RAW_DATA_DIR / "yellow_tripdata_2025-01.parquet"
ZONE_FILE = RAW_DATA_DIR / "taxi_zone_lookup.csv"

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# Strict chronological splits (January 2025)
MONTH_START = "2025-01-01 00:00:00"
MONTH_END = "2025-01-31 23:00:00"

TRAIN_END = "2025-01-21 23:00:00"   # Jan 1 - Jan 21 (~68%)
VAL_START = "2025-01-22 00:00:00"
VAL_END = "2025-01-26 23:00:00"     # Jan 22 - Jan 26 (~16%)
TEST_START = "2025-01-27 00:00:00"
TEST_END = "2025-01-31 23:00:00"    # Jan 27 - Jan 31 (~16%, LOCKED until Phase 3 final test)

# Deterministic physical cleaning bounds (established in Phase 1 BEFORE seeing test results)
MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 180.0
MIN_DISTANCE_MILES = 0.1
MAX_DISTANCE_MILES = 50.0
MIN_FARE_USD = 1.0
MAX_FARE_USD = 250.0
MIN_SPEED_MPH = 1.0
MAX_SPEED_MPH = 70.0
UNKNOWN_ZONES = {264, 265}

# Calendar holidays in Jan 2025 (known in advance - no leakage)
JAN_2025_HOLIDAYS = {"2025-01-01", "2025-01-20"}  # New Year's Day, MLK Jr. Day
# NYC Congestion Pricing started Jan 5, 2025 (publicly scheduled policy date)
CONGESTION_PRICING_START = "2025-01-05 00:00:00"

# Minimum total events (pickups + dropoffs) in TRAIN set to consider a zone "active" for clustering
MIN_TRAIN_EVENTS_FOR_ACTIVE_ZONE = 200


def ensure_directories():
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)