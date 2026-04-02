import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

def run_anomaly_detection(csv_path):
    print(f"[*] Loading telemetry from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print("[!] ERROR: CSV not found. Did you run the exporter?")
        return

    # 1. Load data
    df = pd.read_csv(csv_path)
    features = ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]']
    ml_data = df[features].dropna()
    
    # 2. Train AI Model
    print("[*] Training the Isolation Forest Model...")
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(ml_data)
    df['anomaly'] = model.predict(ml_data)
    
    anomalies = df[df['anomaly'] == -1]
    print(f"[+] AI Analysis Complete! Found {len(anomalies)} anomalous data points out of {len(df)}.")

    # 3. Visualize
    print("[*] Generating visual report...")
    plt.figure(figsize=(12, 6))
    
    # CONVERT TO NUMPY ARRAYS TO PREVENT CRASHES
    time_normal = df['timestamp_sec'].to_numpy()
    gyro_normal = df['gyro_rad[0]'].to_numpy()
    time_anomaly = anomalies['timestamp_sec'].to_numpy()
    gyro_anomaly = anomalies['gyro_rad[0]'].to_numpy()

    # Plot
    plt.plot(time_normal, gyro_normal, label='Normal Gyro X', color='blue', alpha=0.6)
    plt.scatter(time_anomaly, gyro_anomaly, color='red', label='ANOMALY DETECTED', zorder=5)
    
    plt.title('Flight Telemetry AI Anomaly Detection')
    plt.xlabel('Flight Time (Seconds)')
    plt.ylabel('Gyroscope Rotation (Rad/s)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('ai_flight_analysis.png')
    print("[+] Graph saved as 'ai_flight_analysis.png'")

if __name__ == '__main__':
    run_anomaly_detection('ml_ready_flight_data.csv')
