import pyulog
import pandas as pd
import sys
import os

def build_ml_dataset(ulog_path, output_filename):
    print(f"[*] Initializing ML Extraction Pipeline...")
    print(f"[*] Target Log: {ulog_path}")
    
    if not os.path.exists(ulog_path):
        print("[!] ERROR: Log file not found. Check the path.")
        return

    # 1. Crack open the ULog file
    ulog = pyulog.ULog(ulog_path)
    
    try:
        # 2. Extract the core vibration/movement telemetry
        sensor_data = ulog.get_dataset('sensor_combined')
    except Exception as e:
        print(f"[!] ERROR: Could not find 'sensor_combined' data in this log. {e}")
        return

    # 3. Convert raw data into a clean Pandas DataFrame
    df = pd.DataFrame(sensor_data.data)

    # 4. Feature Engineering: Normalize timestamps to seconds
    df['timestamp_sec'] = df['timestamp'] / 1e6
    
    # Reorder columns to put our new timestamp first
    cols = ['timestamp_sec'] + [c for c in df.columns if c != 'timestamp_sec']
    df = df[cols]

    # 5. Export to CSV
    df.to_csv(output_filename, index=False)
    
    print(f"\n[+] Pipeline Success! Extracted {len(df)} rows of telemetry.")
    print(f"[+] Clean ML dataset saved to: {output_filename}")
    print("\n--- Data Preview ---")
    print(df.head())

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ml_exporter.py <path_to_ulog_file>")
    else:
        target_log = sys.argv[1]
        build_ml_dataset(target_log, "ml_ready_flight_data.csv")
