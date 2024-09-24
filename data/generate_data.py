import pandas as pd
import numpy as np
from datetime import timedelta

# Number of records
num_records = 100
timestamp_start = pd.Timestamp('2024-08-01 00:00:00')

# Create synthetic device operation data with the required dimensions
device_data = pd.DataFrame({
    "timestamp": [timestamp_start + timedelta(hours=i) for i in range(num_records)],
    "时间步数": np.arange(num_records),  # 增加时间步数特征
    "车速": np.random.randint(0, 120, size=num_records),
    "发动机转速": np.random.randint(500, 3000, size=num_records),
    "油耗": np.random.uniform(5, 20, size=num_records),
    "故障码数量": np.random.randint(0, 5, size=num_records),
    "发动机扭矩": np.random.uniform(100, 400, size=num_records),
    "摩擦扭矩": np.random.uniform(10, 100, size=num_records),
    "燃料流量": np.random.uniform(10, 50, size=num_records),
    "空气进气量": np.random.uniform(50, 200, size=num_records),
    "SCR上游NOX传感器输出值": np.random.uniform(10, 100, size=num_records),
    "SCR下游NOX传感器输出值": np.random.uniform(5, 50, size=num_records),
    "SCR入口温度": np.random.uniform(100, 300, size=num_records),
    "SCR出口温度": np.random.uniform(80, 250, size=num_records),
    "水温": np.random.uniform(60, 90, size=num_records)
})

# Introduce additional time window feature
# Add explicit time window length (in weeks) for each record
time_windows = [1, 2, 3, 4, 5, 6]  # Example time windows: 1 week to 6 weeks
device_data["时间窗口长度"] = np.random.choice(time_windows, size=num_records)  # 随机分配时间窗口长度

# Create synthetic repair data (maintenance logs)
repair_data = pd.DataFrame({
    "timestamp": [timestamp_start + timedelta(hours=i * 10) for i in range(10)],  # Repairs every 10 hours
    "repair_type": np.random.choice(
        ["发动机维修", "排气系统维修", "故障码异常", "压力传感器异常", "水温异常", "变速箱异常"],
        size=10
    )
})

# Save both datasets to CSV files
device_data_file = 'device_operation_data.csv'
repair_data_file = 'maintenance_logs.csv'

device_data.to_csv(device_data_file, index=False)
repair_data.to_csv(repair_data_file, index=False)

print(f"Device operation data saved to {device_data_file}")
print(f"Maintenance logs saved to {repair_data_file}")
