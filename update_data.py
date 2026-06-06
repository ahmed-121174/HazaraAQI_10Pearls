"""
Hourly ingestion and cleaning script for Hazara AQI dashboard.
Fetches raw data from Open-Meteo and cleans it immediately to keep local CSV files fresh.
"""

import os
import sys

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from run_pipeline import collect_all_data, clean_data

if __name__ == "__main__":
    print("=== Hourly Data Ingestion & Cleaning ===")
    try:
        raw_df = collect_all_data()
        if raw_df is not None and not raw_df.empty:
            clean_df = clean_data(raw_df)
            print("=== Ingestion & Cleaning Complete ===")
        else:
            print("Error: No data collected.")
            sys.exit(1)
    except Exception as e:
        print(f"Pipeline error: {e}")
        sys.exit(1)
